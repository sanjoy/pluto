"""Prepare fixed historical first-piece head-position assays, without a GPU.

The new artifact is an authenticated input descriptor, NOT a measurement. It
fixes three previously studied heads at three previously selected historical
events. Clean and all-query weight-zero logits are calibration requirements for
a future context-zero replay; query-only/other-query results do not exist here.

All paths supplied by historical JSON are read-only inputs. The only command
this module can execute is a validated, read-only Git source recovery. Providing
an explicit producer snapshot avoids even that command; its bytes must match
the producer record in the authenticated historical plan.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from . import checkpoint
from . import historical_word_ablations as historical
from . import paired_word_cases

FORMAT = 'pluto-head-position-cases-v1'
# Explicit coordinates prevent a future run from selecting events by effects.
EVENTS = (('grandam', 365, 4490, '206772616e64'),
          ('corse', 700, 1162, '20636f72'),
          ('Exeunt', 1340, 1475, '204578'))
HEADS = ((0, 5), (2, 2), (3, 6))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(evidence_manifest, run_directory, repository, output, *,
            producer_source=None, config=checkpoint.GPT2Config(),
            events=EVENTS, heads=HEADS):
    """Write NEW output/cases.json only after authenticating every used input.

    Config/event/head overrides exist only for small CPU fixtures and are
    recorded as such. The CLI always uses the fixed historical production assay.
    Checkpoint files, prefix files, and native tensors are referenced, not copied
    or modified. Descriptor consumers must reverify the complete provenance.
    """
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    require(output.parent.resolve(strict=True) == output.parent,
            'output parent must be a real existing directory')
    root = Path(run_directory).absolute()
    require(root.is_dir() and root.resolve() == root, 'historical root is not a real directory')
    events, heads = tuple(tuple(e) for e in events), tuple(tuple(h) for h in heads)
    require(events and heads and all(len(e) == 4 for e in events)
            and len(set(events)) == len(events)
            and len({e[0] for e in events}) == len(events)
            and len({e[1] for e in events}) == len(events)
            and len(set(heads)) == len(heads), 'empty or duplicate event/head selection')
    for event in events:
        require(len(event) == 4 and isinstance(event[0], str) and event[0].isalpha()
                and type(event[1]) is int and event[1] >= 0
                and type(event[2]) is int and 0 <= event[2] < config.vocab_size
                and isinstance(event[3], str) and bytes.fromhex(event[3]),
                'invalid fixed event')
    for block, head in heads:
        require(type(block) is int and 0 <= block < config.n_layers
                and type(head) is int and 0 <= head < config.n_heads,
                'head outside configured model')
    sources = [historical.record(p) for p in
               (__file__, historical.__file__, checkpoint.__file__, paired_word_cases.__file__)]
    evidence = historical.Evidence(evidence_manifest)
    plan_path = evidence.checked(root / 'plan.json')
    plan = historical._json(plan_path)
    require(plan['output_directory'] == str(root)
            and all(plan[k] == getattr(config, k) for k in
                    ('vocab_size', 'padded_vocab_size', 'n_layers', 'n_heads')),
            'historical plan geometry/root differs')
    inputs = historical._index(plan['inputs'], 'path')
    repository = Path(repository).resolve(strict=True)
    expected_source = inputs[str(repository / historical.PRODUCER_SOURCE)]
    if producer_source is None:
        producer = historical.recover_producer_source(
            repository, evidence.manifest['source_commit_sha'], expected_source)
    else:
        source = historical.record(producer_source)
        require((source['bytes'], source['sha256']) ==
                (expected_source['bytes'], expected_source['sha256']),
                'explicit producer source differs from historical plan')
        sources.append(source)
        producer = dict(snapshot=source, matched_historical_plan=True,
                        historical_record=expected_source, command_executed=False)

    directory = Path(plan['checkpoint_directory'])
    specs = checkpoint.tensor_manifest(config)
    require(directory.is_dir() and directory.resolve() == directory
            and {p.name for p in directory.iterdir()} == {s.filename for s in specs},
            'checkpoint must contain exactly the canonical weight files')
    weight_records = []
    for spec in specs:
        path = evidence.checked(directory / spec.filename, inputs[str(directory / spec.filename)])
        require(path.stat().st_size == spec.nbytes, 'checkpoint weight byte size differs')
        require(np.isfinite(np.memmap(path, dtype='<f4', mode='r')).all(),
                'nonfinite checkpoint weight')
        weight_records.append(evidence.used[str(path)])
    generation_dir = Path(plan['generation_directory'])
    for protected in (root, directory, generation_dir):
        require(output != protected and protected not in output.parents,
                'output overlaps preserved input directory')
    path = generation_dir / 'metadata.json'
    generation = historical._json(evidence.checked(path, inputs[str(path)]))
    initial, generated = generation['initial_token_ids'], generation['generated_token_ids']
    all_ids = generation['all_token_ids']
    require(generation.get('complete') is True and initial
            and generation['checkpoint_directory'] == str(directory)
            and generation['context_length'] == config.context_length
            and generation['vocab_size'] == config.vocab_size
            and generation['padded_vocab_size'] == config.padded_vocab_size
            and generation['steps'] == len(generated) and all_ids == initial + generated
            and all(type(i) is int and 0 <= i < config.vocab_size for i in all_ids),
            'invalid historical generation identity or token stream')
    path = historical._local(generation_dir, generation['events_file'])
    generation_events = [json.loads(line) for line in
                         evidence.checked(path, inputs[str(path)]).read_text().splitlines()]
    require(len(generation_events) == len(generated), 'generation event count differs')
    path = historical._local(generation_dir, generation['logits_file'])
    require(str(path) == plan['generation_logits'], 'generation logits path differs')
    evidence.checked(path, inputs[str(path)])
    require(path.stat().st_size == len(generated) * config.vocab_size * 4,
            'generation logits byte size differs')
    generation_logits = np.memmap(path, dtype='<f4', mode='r').reshape(-1, config.vocab_size)
    runs = {r['step']: r for r in plan['runs']}
    selections = {s['word']: s for s in plan['selections']}
    require(len(runs) == len(plan['runs']) and len(selections) == len(plan['selections']),
            'duplicate historical run/word selection')
    cases = []

    def capture(native, metadata, name, shape, dtype):
        item = metadata['files'][name]
        require(item['shape'] == shape and item['dtype'] == dtype,
                'native capture shape/dtype differs: ' + name)
        expected_role = 'selected_row' if dtype == 'float32' else 'full_prefix'
        # Old logits metadata omits role; all BF16 captures explicitly name it.
        require(item.get('role', expected_role if dtype == 'float32' else None) == expected_role,
                'native capture scope differs: ' + name)
        path = evidence.checked(historical._local(native, item['file']))
        size = 4 if dtype == 'float32' else 2
        require(path.stat().st_size == int(np.prod(shape)) * size,
                'native capture byte size differs: ' + name)
        if dtype == 'float32':
            values = np.fromfile(path, dtype='<f4')
        else:
            values = (np.fromfile(path, dtype='<u2').astype(np.uint32) << 16).view(np.float32)
        require(np.isfinite(values).all(), 'nonfinite native capture: ' + name)
        return dict(record=evidence.used[str(path)], shape=shape, dtype=dtype,
                    role=expected_role)

    for word, step, target, piece_hex in events:
        selection = selections[word]
        require(selection['first'] == step and type(selection['end']) is int
                and step < selection['end'] <= len(generated),
                'event is not the selected historical word first piece')
        word_bytes = b''.join(bytes.fromhex(e['piece_hex'])
                              for e in generation_events[step:selection['end']])
        require(word_bytes.lstrip(b' ') == word.encode(),
                'recorded native word pieces do not spell the selected word')
        run = runs[step]
        require(all(run['event'].get(k) == v for k, v in generation_events[step].items())
                and run['event']['piece_hex'] == piece_hex
                and run['event']['token_id'] == target, 'historical event/target differs')
        native = root / f'trace_step_{step}'
        metadata_path = evidence.checked(native / 'metadata.json')
        metadata = historical._json(metadata_path)
        ids, actual_target, _ = historical.validate_context(
            metadata, run, all_ids, len(initial), config)
        require(actual_target == target, 'historical target ID differs')
        rows = len(ids)
        prefix = evidence.checked(run['prefix']['path'], run['prefix'])
        require(prefix.stat().st_size == rows * 4
                and np.fromfile(prefix, dtype='<i4').tolist() == ids
                and metadata['tokens_file'] == str(prefix), 'native prefix bytes/path differ')
        require(metadata.get('complete') is True and metadata.get('probe_kind') == 'token_trace'
                and metadata['checkpoint_directory'] == str(directory)
                and metadata['vocab_size'] == config.vocab_size
                and metadata['padded_vocab_size'] == config.padded_vocab_size
                and metadata['context_length'] == config.context_length
                and metadata['checkpoint_unique_weight_count'] == len(specs)
                and metadata['byte_order'] == 'little' and metadata['retokenized'] is False
                and metadata['pad_token_id'] == ids[-1]
                and all(metadata[k] == 0 for k in
                        ('optimizer_steps', 'backward_calls', 'checkpoint_writes'))
                and all(metadata['checks'].get(k) is True for k in
                        ('clean_replay_logits_byte_equal', 'alternate_padding_logits_byte_equal',
                         'final_lens_logits_byte_equal', 'attention_residual_replay_byte_equal',
                         'mlp_residual_replay_byte_equal')),
                'native geometry, padding, parity, or read-only checks differ')
        clean = capture(native, metadata, 'logits', [1, config.padded_vocab_size], 'float32')
        clean_bytes = Path(clean['record']['path']).read_bytes()
        require(clean_bytes[:config.vocab_size * 4] == generation_logits[step].tobytes(),
                'native clean logits differ from original generation')
        for key in ('clean_replay', 'alternate_padding'):
            parity = evidence.checked(historical._local(native, metadata['parity_files'][key]))
            require(parity.read_bytes() == clean_bytes, 'clean/replay/padding bytes differ')
        arms = historical.validate_arms(metadata['interventions'], config)
        for block, head in heads:
            arm = arms[f'ablation.block{block}.head{head}']
            path = evidence.checked(historical._local(native, arm['logits_file']))
            require(path.stat().st_size == config.padded_vocab_size * 4
                    and np.isfinite(np.fromfile(path, dtype='<f4')).all(),
                    'invalid all-query weight-zero calibration logits')
            before_name = 'positioned' if block == 0 else f'blocks.{block-1}.after_mlp'
            captures = {name: capture(native, metadata, key, [rows, config.d_model], 'bf16')
                        for name, key in (('before', before_name),
                                          ('context', f'blocks.{block}.attention'),
                                          ('projected', f'blocks.{block}.attention_projected'),
                                          ('after_attention', f'blocks.{block}.after_attention'))}
            first, last = head * config.head_dim, (head + 1) * config.head_dim
            cases.append(dict(case_index=len(cases), word=word, generation_step=step,
                absolute_target_index=run['event']['absolute_token_index'],
                context_start=run['event']['context_start'], query=rows - 1,
                target_id=target, target_piece_hex=piece_hex, block=block, head=head,
                prefix=evidence.used[str(prefix)], visible_rows=rows,
                model_context_length=config.context_length,
                padding=dict(strategy='repeat_last_prefix_token', token_id=ids[-1],
                             row_interval=[rows, config.context_length]),
                metadata=evidence.used[str(metadata_path)], captures=captures,
                head_channel_interval=[first, last],
                # Native execution always has the full context, including
                # future padding absent from the archived visible captures.
                # The other/all scopes MUST include those future rows. Their
                # intervention is causal-null at the selected earlier query,
                # which is a calibration requirement, not an excuse to silently
                # replace the requested full-context scope with a shorter one.
                interventions=dict(query_only=[[rows-1, rows]],
                    other_queries=[interval for interval in
                                   ([0, rows-1], [rows, config.context_length])
                                   if interval[0] < interval[1]],
                    all_queries=[[0, config.context_length]],
                    interval_convention='half_open',
                    visible_query_interval=[0, rows], captured_query_interval=[0, rows],
                    future_padding_interval=[rows, config.context_length],
                    future_padding_causal_null_at_selected_query=True),
                projection=dict(weight=weight_records[6 + 12 * block],
                                bias=weight_records[7 + 12 * block],
                                input_feature_rows=[first, last], bias_unchanged=True),
                calibration=dict(clean_logits=clean['record'],
                    all_query_weight_zero_logits=evidence.used[str(path)],
                    full_padded_logits_shape=[1, config.padded_vocab_size],
                    require_clean_replay_byte_equal=True,
                    require_all_query_context_zero_matches_weight_zero_bytes=True,
                    calibration_executed=False)))
    evidence.recheck()
    require(all(historical.record(r['path']) == r for r in sources),
            'implementation/source snapshot changed while preparing')
    result = dict(format=FORMAT, complete=True,
        completion_meaning='Authenticated fixed assay inputs prepared; no native replay or new effect measured.',
        historical_only=True, new_paired_model_result=False, gpu_work_performed=False,
        goal_completion_claimed=False, calibration_executed=False,
        production_fixed_assay=(config == checkpoint.GPT2Config() and events == EVENTS and heads == HEADS),
        config=asdict(config), checkpoint_directory=str(directory), weights=weight_records,
        producer_source=producer, case_count=len(cases), cases=cases,
        provenance=list(evidence.used.values()), implementation=sources,
        limitations=['Post-hoc heads and unmatched historical contexts, not lexical selectivity or unique storage.',
                     'First-piece predictions only, not full-word spelling or word-boundary probabilities.',
                     'Query-only and other-query effects need not add; preserve interaction effects.',
                     'Future replay must preserve original weights/bias, BF16 arithmetic, padding and query coordinates.',
                     'A current executable is not authenticated by source recovery; native parity is still required.'])
    output.mkdir()
    paired_word_cases._write_json(output / 'cases.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evidence-manifest', 'run-directory', 'repository', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--producer-source', type=Path)
    args = parser.parse_args(argv)
    prepare(args.evidence_manifest, args.run_directory, args.repository, args.output,
            producer_source=args.producer_source)


if __name__ == '__main__':
    main()
