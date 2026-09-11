"""Bind a focused native trace to its original autoregressive sampling event.

No generated text is re-tokenized. The exact context IDs and selected logit
bytes must match the original generation. Target support, random selection,
intermediate readouts, linear accounting, and removal effects stay separate.
"""

import argparse
import bisect
import json
import math
from pathlib import Path

import numpy as np

from .checkpoint import GPT2Checkpoint
from .late_mlp_paths import file_record, write_exclusive
from .phrase_reference import bf16
from .phrase_trace_analysis import (load_native_array, predictions,
                                    residual_stage_names, stage_widths,
                                    validate_intervention_arms)
from .token_path_math import target_ledger, neuron_operations
from .verify import gpt2_token_bytes


def production_distribution(logits, temperature, uniform):
    """Reproduce production sampling, including FP32 subtraction before exp.

    Python 3.12's sum(float) may compensate errors. Explicit loops below match
    C++ std::accumulate/partial_sum instead. This is a numerical reconstruction;
    original recorded native tokens remain the authority and are checked.
    """
    row = np.asarray(logits, dtype=np.float32)
    if row.ndim != 1 or len(row) < 2 or not np.isfinite(row).all():
        raise ValueError('sampling needs a finite vocabulary logit row')
    if not math.isfinite(temperature) or temperature <= 0 or not 0 <= uniform < 1:
        raise ValueError('invalid temperature or uniform draw')
    shifted = np.subtract(row, np.max(row), dtype=np.float32)
    weights = [math.exp(float(value)/temperature) for value in shifted]
    total = 0.0
    for value in weights:
        total += value
    probabilities = [value/total for value in weights]
    cdf, running = [], 0.0
    for value in probabilities:
        running += value
        cdf.append(running)
    cdf[-1] = 1.0
    selected = bisect.bisect_left(cdf, uniform)
    return np.asarray(probabilities), cdf, selected


def readout(logits, target, token_bytes, temperature, uniform):
    info = predictions(np.asarray(logits)[None, :], token_bytes, [target])[0]
    info['target'] = info.pop('final_winner')
    probabilities, cdf, selected = production_distribution(logits, temperature, uniform)
    info['target_probability_at_generation_temperature'] = float(probabilities[target])
    info['same_uniform_selected_id'] = selected
    info['same_uniform_selected_piece'] = repr(token_bytes[selected].decode('utf-8', errors='backslashreplace'))
    info['target_cdf_lower'] = cdf[target-1] if target else 0.0
    info['target_cdf_upper'] = cdf[target]
    return info


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: jsonable(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(entry) for entry in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run(native_directory, generation_directory, generation_step, checkpoint_directory,
        tokenizer_directory, output_directory):
    from tokenizers import Tokenizer

    native, generation, output = map(Path, (native_directory, generation_directory, output_directory))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    inputs = {}

    def local_file(directory, filename):
        relative = Path(filename)
        if (relative.is_absolute() or len(relative.parts) != 1
                or relative.name in ('', '.', '..')):
            raise ValueError('generation evidence must use a local filename')
        path = directory/relative
        if path.is_symlink() or not path.is_file():
            raise ValueError('generation evidence must be a regular nonsymlink file')
        return path

    def recorded_json(path, label):
        if path.is_symlink() or not path.is_file():
            raise ValueError('metadata must be a regular nonsymlink file')
        inputs[label] = file_record(path)
        return json.loads(path.read_text())

    metadata = recorded_json(native/'metadata.json', 'native_metadata')
    gen = recorded_json(generation/'metadata.json', 'generation_metadata')
    events_path = local_file(generation, gen['events_file'])
    inputs['events'] = file_record(events_path)
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    if (gen.get('complete') is not True or metadata.get('complete') is not True
            or metadata.get('probe_kind') != 'token_trace'
            or type(generation_step) is not int or not 0 <= generation_step < len(events)):
        raise ValueError('need completed native generation and exact-token focused trace')
    event = events[generation_step]
    checkpoint = GPT2Checkpoint(checkpoint_directory, check_finite=True)
    config = checkpoint.config
    ids = metadata['token_ids']
    for values in (ids, gen.get('initial_token_ids'), gen.get('generated_token_ids'), gen.get('all_token_ids')):
        if (not isinstance(values, list) or any(type(token) is not int or
                not 0 <= token < config.vocab_size for token in values)):
            raise ValueError('native token IDs must be valid integers')
    if (not 0 < len(ids) <= config.context_length or not gen['initial_token_ids']
            or type(gen.get('steps')) is not int or gen['steps'] != len(events)
            or len(gen['generated_token_ids']) != len(events)
            or metadata.get('no_bos') is not True or metadata.get('retokenized') is not False
            or metadata.get('autoregressive') is not False):
        raise ValueError('invalid generation count or exact-token trace contract')
    for name in ('index', 'absolute_token_index', 'context_start', 'context_length', 'output_row', 'token_id', 'logits_byte_offset'):
        if type(event.get(name)) is not int:
            raise ValueError('native event coordinates must be integers')
    for name in ('prompt_rows', 'selected_row', 'target_id', 'context_length', 'vocab_size', 'padded_vocab_size'):
        if type(metadata.get(name)) is not int:
            raise ValueError('native trace coordinates must be integers')
    row = len(ids)-1
    absolute = len(gen['initial_token_ids']) + generation_step
    expected_start = max(0, absolute-config.context_length)
    if (gen['all_token_ids'] != gen['initial_token_ids'] + gen['generated_token_ids']
            or event['index'] != generation_step or event['absolute_token_index'] != absolute
            or event['context_start'] != expected_start
            or event['context_length'] != absolute-expected_start
            or event['output_row'] != row
            or ids != gen['all_token_ids'][expected_start:absolute]
            or event['token_id'] != gen['generated_token_ids'][generation_step]
            or metadata['target_id'] != event['token_id'] or metadata['selected_row'] != row
            or metadata['prompt_rows'] != len(ids)):
        raise ValueError('trace does not use the original sampled token and exact native context')
    for item in (metadata, gen):
        if (Path(item['checkpoint_directory']).resolve() != Path(checkpoint_directory).resolve()
                or item['context_length'] != config.context_length
                or item['vocab_size'] != config.vocab_size):
            raise ValueError('trace/generation checkpoint or geometry mismatch')
    if metadata['padded_vocab_size'] != config.padded_vocab_size:
        raise ValueError('physical vocabulary mismatch')
    if Path(gen['tokenizer_directory']).resolve() != Path(tokenizer_directory).resolve():
        raise ValueError('generation tokenizer differs from supplied tokenizer')
    required = ('final_lens_logits_byte_equal', 'alternate_padding_logits_byte_equal',
                'clean_replay_logits_byte_equal', 'attention_residual_replay_byte_equal',
                'mlp_residual_replay_byte_equal')
    if any(metadata.get('checks', {}).get(key) is not True for key in required):
        raise ValueError('native parity/replay failed')
    tokenizer_path = Path(tokenizer_directory)/'tokenizer.json'
    inputs['tokenizer'] = file_record(tokenizer_path)
    token_bytes = gpt2_token_bytes(Tokenizer.from_file(str(tokenizer_path)))
    if len(token_bytes) != config.vocab_size or bytes.fromhex(event['piece_hex']) != token_bytes[event['token_id']]:
        raise ValueError('tokenizer/event byte mismatch')
    for spec in checkpoint.manifest:
        inputs['weight:'+spec.name] = file_record(checkpoint.directory/spec.filename)
    for name in ('token_trace_analysis.py', 'token_path_math.py', 'phrase_reference.py',
                 'phrase_trace_analysis.py', 'checkpoint.py', 'verify.py', 'late_mlp_paths.py'):
        inputs['source:'+name] = file_record(Path(__file__).with_name(name))

    evidence_paths = set()

    def load(name, record, dtype, shape):
        if record['dtype'] != dtype or record['shape'] != shape:
            raise ValueError('unexpected native array dtype/shape: '+name)
        values, identity = load_native_array(native, record)
        if identity['path'] in evidence_paths:
            raise ValueError('native evidence files must be independently named')
        evidence_paths.add(identity['path'])
        inputs[name] = identity
        return values

    widths = stage_widths(config)
    if set(metadata['files']) != set(widths) | {'tokens', 'logits'}:
        raise ValueError('focused activation schema is incomplete')
    stages = {name: load('stage:'+name, metadata['files'][name], 'bf16', [len(ids), width])
              for name, width in widths.items()}
    tokens = load('tokens', metadata['files']['tokens'], 'int32', [len(ids), 1])
    if tokens[:, 0].tolist() != ids:
        raise ValueError('native token dump differs from metadata')
    native_logits = load('logits', metadata['files']['logits'], 'float32', [1, config.padded_vocab_size])
    if set(metadata['parity_files']) != {'alternate_padding', 'clean_replay'}:
        raise ValueError('missing parity evidence')
    for name, file in metadata['parity_files'].items():
        values = load('parity:'+name, dict(file=file, dtype='float32', shape=[1, config.padded_vocab_size]),
                      'float32', [1, config.padded_vocab_size])
        if values.tobytes() != native_logits.tobytes():
            raise ValueError('native selected-row replay bytes disagree')
    generation_logits_path = local_file(generation, gen['logits_file'])
    inputs['generation_logits'] = file_record(generation_logits_path)
    if (gen['logits_shape'] != [gen['steps'], config.vocab_size]
            or generation_logits_path.stat().st_size != gen['steps']*config.vocab_size*4
            or event['logits_byte_offset'] != generation_step*config.vocab_size*4):
        raise ValueError('generation logit shape/offset mismatch')
    original = np.memmap(generation_logits_path, dtype='<f4', mode='r', shape=tuple(gen['logits_shape']))[generation_step]
    logits = native_logits[0, :config.vocab_size]
    if original.tobytes() != logits.tobytes():
        raise ValueError('focused trace is not byte-identical to original autoregressive logits')
    target = event['token_id']
    temperature, uniform = float(gen['temperature']), float(event['uniform'])
    baseline = readout(logits, target, token_bytes, temperature, uniform)
    if (baseline['same_uniform_selected_id'] != target
            or not math.isclose(baseline['target_probability_at_generation_temperature'],
                                event['sampled_probability'], rel_tol=2e-12, abs_tol=1e-17)):
        raise ValueError('reconstructed sampling does not select original token')
    if (type(event.get('raw_rank')) is not int or type(event.get('raw_argmax_token_id')) is not int
            or event['raw_rank'] != baseline['target']['rank']
            or event['raw_argmax_token_id'] != baseline['winner']['id']
            or event['raw_logit'] != float(logits[target])):
        raise ValueError('event raw logit/rank/argmax disagrees with native logits')
    for name in ('lower', 'upper'):
        if not math.isclose(float(event['cdf_'+name]), baseline['target_cdf_'+name], rel_tol=0, abs_tol=8e-15):
            raise ValueError('recorded sampling interval disagrees with reconstructed CDF')
    order = np.argsort(-logits, kind='stable')
    competitor = int(order[0] if order[0] != target else order[1])
    lens = {}
    names = residual_stage_names(config.n_layers)
    if set(metadata['lens']) != set(names):
        raise ValueError('incomplete intermediate readouts')
    for name in names:
        values = load('lens:'+name, metadata['lens'][name], 'float32', [1, config.padded_vocab_size])
        lens[name] = readout(values[0, :config.vocab_size], target, token_bytes, temperature, uniform)
        if name == names[-1] and values.tobytes() != native_logits.tobytes():
            raise ValueError('final lens does not equal original output')
    ledger = target_ledger(checkpoint, stages, row, target, competitor, logits)
    validate_intervention_arms(metadata['interventions'], 1, config)
    interventions = []
    base_probability = baseline['target_probability_at_generation_temperature']
    for arm in metadata['interventions']:
        values = load('intervention:'+arm['name'],
                      dict(file=arm['logits_file'], dtype='float32', shape=arm['shape']),
                      'float32', [1, config.padded_vocab_size])[0, :config.vocab_size]
        info = readout(values, target, token_bytes, temperature, uniform)
        probability = info['target_probability_at_generation_temperature']
        margin = float(values[target])-float(values[competitor])
        interventions.append(dict(**arm, readout=info, target_margin=margin,
                                  margin_change=margin-ledger['native_margin'],
                                  delta_log_probability=math.log(probability/base_probability) if probability else None))
    features = []
    for block in range(config.n_layers):
        values = ledger['neurons'][f'blocks.{block}']
        chosen = np.unique(np.concatenate((np.argsort(values)[:3], np.argsort(-values)[:3])))
        for neuron in chosen:
            features.append(neuron_operations(checkpoint, stages, row, block, int(neuron), ledger['direction']))
    output.mkdir()
    result = dict(schema_version=1, complete=True, generation_step=generation_step,
                  target_id=target, target_piece=repr(token_bytes[target].decode('utf-8', errors='backslashreplace')),
                  context_token_ids=ids, context_bytes_hex=b''.join(token_bytes[t] for t in ids).hex(),
                  baseline_logits_equal_original_generation=True, original_event=event,
                  temperature=temperature, baseline=baseline,
                  competitor_id=competitor, competitor_piece=repr(token_bytes[competitor].decode('utf-8', errors='backslashreplace')),
                  lens=lens, ledger=ledger, selected_neuron_operations=features,
                  interventions=interventions, inputs=inputs,
                  limits=['Target support is distinct from actual RNG selection.',
                          'Intermediate final-head readouts are diagnostics, not discrete hidden tokens.',
                          'Fixed-final-normalizer terms are accounting, not removal effects.',
                          'Reusing a uniform draw is a sampler counterfactual dependent on vocabulary CDF ordering.',
                          'Only this exact context, checkpoint, and selected token are analyzed.'])
    for record in inputs.values():
        if file_record(record['path']) != record:
            raise ValueError('input changed while analyzing')
    write_exclusive(output/'analysis.json', jsonable(result))
    print('Analyzed generation step', generation_step, 'target', result['target_piece'],
          'rank', baseline['target']['rank'], 'and', len(interventions), 'causal arms.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native-directory', 'generation-directory', 'checkpoint-directory', 'tokenizer-directory', 'output-directory'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--generation-step', type=int, required=True)
    args = parser.parse_args()
    run(args.native_directory, args.generation_directory, args.generation_step,
        args.checkpoint_directory, args.tokenizer_directory, args.output_directory)


if __name__ == '__main__':
    main()
