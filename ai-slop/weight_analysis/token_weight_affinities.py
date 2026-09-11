"""Inspect selected MLP weights without naming neurons as word detectors.

Selection is frozen from existing context-specific margin ledgers before any
all-vocabulary weight affinity is computed. Output affinities are independent
of context activation/variance, but use the checkpoint's final LN gain. They
are readout numerators, not logits, probabilities, or causal removal effects.
"""

import argparse
import datetime
import math
from pathlib import Path

import numpy as np

from .autoregressive_audit import read_json, token_bytes
from .checkpoint import GPT2Checkpoint
from .late_mlp_paths import file_record, write_exclusive


STEPS = (51, 52, 53, 54)
FRAGMENTS = (427, 363, 368, 525)


def require(condition, description):
    if not condition:
        raise ValueError(description)


def select_features(documents):
    """Exactly the predeclared signed ranking; ties use block then neuron ID."""
    selected = []
    for step in STEPS:
        document = documents[step]
        values = document['ledger']['neurons']
        require(set(values) == {f'blocks.{block}' for block in range(8)},
                'incomplete ledger block set')
        candidates = []
        for block in range(8):
            row = np.asarray(values[f'blocks.{block}'], dtype=np.float64)
            require(row.shape == (2048,) and np.isfinite(row).all(),
                    'invalid neuron ledger vector')
            candidates.extend((float(value), block, neuron)
                              for neuron, value in enumerate(row))
        positive = sorted((entry for entry in candidates if entry[0] > 0),
                          key=lambda entry: (-entry[0], entry[1], entry[2]))
        negative = sorted((entry for entry in candidates if entry[0] < 0))
        require(len(positive) >= 2 and negative, 'missing signed selection candidates')
        requests = [(entry, f'global_positive_rank_{rank + 1}')
                    for rank, entry in enumerate(positive[:2])]
        requests.append((negative[0], 'global_negative_rank_1'))
        if step in (53, 54):
            block4 = [entry for entry in positive if entry[1] == 4]
            require(len(block4) >= 2, 'not enough positive B4 features')
            requests.extend((entry, f'block4_positive_rank_{rank + 1}')
                            for rank, entry in enumerate(block4[:2]))
        for (value, block, neuron), reason in requests:
            selected.append({'generation_step': step, 'block': block, 'neuron': neuron,
                             'ledger_margin_contribution': value, 'selection_reason': reason})
    return selected


def bf16_values(values):
    raw = np.asarray(values, dtype=np.float32)
    require(np.isfinite(raw).all(), 'nonfinite master weight')
    bits = raw.view(np.uint32)
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16) << 16
    result = rounded.view(np.float32).astype(np.float64)
    require(np.isfinite(result).all(), 'BF16 conversion overflowed')
    return result


def output_affinities(embedding_bf16, final_gamma, output_row_bf16):
    """E[t] dot gamma*(W2row - mean(W2row)), using FP64 analysis arithmetic."""
    embedding = np.asarray(embedding_bf16, dtype=np.float64)
    gamma = np.asarray(final_gamma, dtype=np.float64)
    row = np.asarray(output_row_bf16, dtype=np.float64)
    require(embedding.ndim == 2 and min(embedding.shape) > 0 and
            gamma.shape == row.shape == (embedding.shape[1],) and
            np.isfinite(embedding).all() and np.isfinite(gamma).all() and
            np.isfinite(row).all(), 'invalid affinity geometry or values')
    return embedding @ (gamma * (row - row.mean()))


def actual_contribution(scores, target, competitor, activation, final_std):
    scores = np.asarray(scores, dtype=np.float64)
    require(scores.ndim == 1 and np.isfinite(scores).all() and
            type(target) is int and type(competitor) is int and target != competitor
            and 0 <= target < len(scores) and 0 <= competitor < len(scores)
            and math.isfinite(activation) and math.isfinite(final_std) and final_std > 0,
            'invalid contribution arguments')
    return float(activation / final_std * (scores[target] - scores[competitor]))


def sources():
    directory = Path(__file__).parent
    return [file_record(directory / filename) for filename in
            ('token_weight_affinities.py', 'checkpoint.py', 'autoregressive_audit.py',
             'late_mlp_paths.py')]


def freeze(root, checkpoint, tokenizer, plan_path):
    root, checkpoint, tokenizer, plan_path = map(Path, (root, checkpoint, tokenizer, plan_path))
    require(not plan_path.exists() and not plan_path.is_symlink(), 'selection plan already exists')
    documents, inputs = {}, {}

    def register(record):
        previous = inputs.get(record['path'])
        require(previous is None or previous == record, 'conflicting input identities')
        inputs[record['path']] = record

    analysis_files = []
    for step, target in zip(STEPS, FRAGMENTS):
        path = root / f'token_analysis_step_{step}' / 'analysis.json'
        record = file_record(path)
        register(record)
        document = read_json(path)
        require(document['complete'] is True and document['generation_step'] == step
                and document['target_id'] == target and
                document['baseline_logits_equal_original_generation'] is True,
                'incomplete or mismatched focused token analysis')
        for identity in document['inputs'].values():
            register(identity)
        metadata = read_json(document['inputs']['native_metadata']['path'])
        require(Path(metadata['checkpoint_directory']).resolve() == checkpoint.resolve(),
                'analysis checkpoint differs from supplied checkpoint')
        documents[step] = document
        analysis_files.append({'generation_step': step, 'identity': record})
    for record in sources() + [file_record(tokenizer / 'tokenizer.json')]:
        register(record)
    for record in inputs.values():
        require(file_record(record['path']) == record, 'changed/unverified selection input')
    selected = select_features(documents)
    plan = {'schema_version': 1, 'stage': 'frozen_selection_before_weight_affinities',
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'checkpoint_directory': str(checkpoint.resolve()),
            'tokenizer_directory': str(tokenizer.resolve()),
            'analysis_files': analysis_files, 'inputs': list(inputs.values()),
            'selection_rule': 'Per step: top2 strictly positive and top1 strictly negative across all8x2048 neuron ledger terms; additionally B4 top2positive for53/54; ties by block then neuron ID.',
            'scope': 'Score every unique selected neuron against all50257 token rows; inspect that feature in all4 recorded contexts, whether or not selected in that context. No corpus input or post-affinity selection.',
            'selected': selected,
            'unique_features': sorted({(entry['block'], entry['neuron']) for entry in selected}),
            'formula': 'E_BF16[t] dot (gamma_final_FP32 * (W2_BF16[row] - mean(W2_BF16[row])))',
            'tracked_fragment_ids': list(FRAGMENTS)}
    write_exclusive(plan_path, plan)
    print({'plan': file_record(plan_path), 'selected': selected})
    return plan


def analyze(plan_path, output):
    plan_path, output = Path(plan_path), Path(output)
    require(not output.exists() and not output.is_symlink(), 'affinity output already exists')
    plan_identity = file_record(plan_path)
    plan = read_json(plan_path)
    require(plan['stage'] == 'frozen_selection_before_weight_affinities', 'selection not frozen')
    for record in plan['inputs']:
        require(file_record(record['path']) == record, 'frozen selection input changed')
    documents = {record['generation_step']: read_json(record['identity']['path'])
                 for record in plan['analysis_files']}
    require(select_features(documents) == plan['selected'], 'frozen choices differ from declared rule')
    checkpoint = GPT2Checkpoint(plan['checkpoint_directory'], check_finite=True)
    config = checkpoint.config
    require((config.vocab_size, config.d_model, config.d_ff, config.n_layers) == (50257, 512, 2048, 8),
            'checkpoint architecture differs')
    tensors = checkpoint.tensors
    dictionary = token_bytes(read_json(Path(plan['tokenizer_directory']) / 'tokenizer.json'))
    embedding = bf16_values(tensors['token_embedding.weight'][:50257])
    gamma = np.asarray(tensors['final_norm.scale'], dtype=np.float64)
    scores_saved, vectors_saved, results = {}, {'final_gamma_fp32': gamma}, []
    checks = []
    tracked = sorted(set(FRAGMENTS) | {document['competitor_id'] for document in documents.values()})
    native_cache = {}

    def native(step, stage, width):
        key = step, stage
        if key not in native_cache:
            document = documents[step]
            record = document['inputs']['stage:' + stage]
            require(file_record(record['path']) == record, 'native stage changed')
            rows = len(document['context_token_ids'])
            bits = np.fromfile(record['path'], dtype='<u2').reshape(rows, width)
            native_cache[key] = (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            require(np.isfinite(native_cache[key]).all(), 'nonfinite native activation')
        return native_cache[key]

    for block, neuron in plan['unique_features']:
        prefix = f'blocks.{block}'
        key = f'block{block}.neuron{neuron}'
        input_master = np.asarray(tensors[prefix + '.mlp.input.weight'][:, neuron], dtype=np.float64)
        input_bf16 = bf16_values(input_master)
        bias = float(tensors[prefix + '.mlp.input.bias'][neuron])
        output_master = np.asarray(tensors[prefix + '.mlp.output.weight'][neuron], dtype=np.float64)
        output_bf16 = bf16_values(output_master)
        scores = output_affinities(embedding, gamma, output_bf16)
        scores_saved[key] = scores
        for name, values in [('input_column_master', input_master), ('input_column_bf16', input_bf16),
                             ('output_row_master', output_master), ('output_row_bf16', output_bf16)]:
            vectors_saved[key + '.' + name] = values
        vectors_saved[key + '.input_bias_master'] = np.asarray([bias])
        order = np.lexsort((np.arange(50257), -scores))
        ranks = np.empty(50257, dtype=np.int32)
        ranks[order] = np.arange(1, 50258)

        def token_entry(token):
            return {'id': int(token), 'piece_hex': dictionary[int(token)].hex(),
                    'piece': repr(dictionary[int(token)].decode('utf-8', errors='backslashreplace')),
                    'score': float(scores[token]), 'descending_rank': int(ranks[token])}

        positive = [int(token) for token in order if scores[token] > 0][:12]
        negative = sorted((int(token) for token in order if scores[token] < 0),
                          key=lambda token: (scores[token], token))[:12]
        contexts = []
        for step in STEPS:
            document = documents[step]
            row = document['ledger']['row']
            target, competitor = document['target_id'], document['competitor_id']
            normalized = native(step, prefix + '.ln2', 512)[row]
            pre = float(native(step, prefix + '.fc1', 2048)[row, neuron])
            post = float(native(step, prefix + '.gelu', 2048)[row, neuron])
            final = native(step, 'blocks.7.after_mlp', 512)[row]
            std = math.sqrt(float(np.mean((final - final.mean()) ** 2)) + 1e-5)
            require(std == document['ledger']['final_std'], 'native final residual variance differs')
            analytical_pre = math.fsum(float(x) * float(y) for x, y in zip(normalized, input_bf16)) + bias
            contribution = actual_contribution(scores, target, competitor, post, std)
            expected = document['ledger']['neurons'][prefix][neuron]
            error = contribution - expected
            require(math.isclose(contribution, expected, rel_tol=2e-12, abs_tol=2e-12),
                    'weight affinity/gate identity differs from recorded ledger')
            for operation in document['selected_neuron_operations']:
                if operation['block'] == block and operation['neuron'] == neuron:
                    require(operation['native_pre_gelu'] == pre and operation['native_post_gelu'] == post
                            and operation['input_bias'] == bias and
                            math.isclose(operation['analytical_pre_gelu'], analytical_pre, abs_tol=1e-14),
                            'native gate or W1-column calculation differs')
            checks.append(abs(error))
            contexts.append({'generation_step': step, 'row': row,
                             'target': token_entry(target), 'competitor': token_entry(competitor),
                             'input_bias': bias, 'analytical_input_dot_plus_bias': analytical_pre,
                             'native_pre_gelu': pre, 'native_post_gelu': post,
                             'native_pre_minus_analytical': pre - analytical_pre,
                             'final_std': std, 'activation_over_final_std': post / std,
                             'target_minus_competitor_affinity': float(scores[target] - scores[competitor]),
                             'affinity_times_gate_contribution': contribution,
                             'recorded_ledger_contribution': expected, 'identity_error': error})
        root = checkpoint.directory
        results.append({'block': block, 'neuron': neuron, 'vector_key': key,
                        'selection_reasons': [entry for entry in plan['selected'] if
                                              entry['block'] == block and entry['neuron'] == neuron],
                        'input_column': {'file': str(root / f'weight_{10 + 12 * block}.bin'),
                                         'first_byte': 4 * neuron, 'byte_stride': 8192, 'elements': 512,
                                         'last_element_first_byte': 4 * neuron + 511 * 8192,
                                         'master_l2': float(np.linalg.norm(input_master)),
                                         'bf16_l2': float(np.linalg.norm(input_bf16))},
                        'input_bias': {'file': str(root / f'weight_{11 + 12 * block}.bin'),
                                       'byte_offset': 4 * neuron, 'value': bias},
                        'output_row': {'file': str(root / f'weight_{12 + 12 * block}.bin'),
                                       'byte_offset': 4 * neuron * 512, 'bytes': 2048,
                                       'master_l2': float(np.linalg.norm(output_master)),
                                       'bf16_l2': float(np.linalg.norm(output_bf16)),
                                       'bf16_mean': float(output_bf16.mean()),
                                       'bf16_centered_l2': float(np.linalg.norm(output_bf16 - output_bf16.mean()))},
                        'positive_top12': [token_entry(token) for token in positive],
                        'negative_top12': [token_entry(token) for token in negative],
                        'tracked_fragments_and_competitors': [token_entry(token) for token in tracked],
                        'contexts': contexts})
    score_path = output.with_name(output.stem + '_scores.npz')
    vector_path = output.with_name(output.stem + '_weight_vectors.npz')
    with score_path.open('xb') as handle:
        np.savez(handle, **scores_saved)
    with vector_path.open('xb') as handle:
        np.savez(handle, **vectors_saved)
    for record in plan['inputs'] + [plan_identity]:
        require(file_record(record['path']) == record, 'input changed during affinity analysis')
    report = {'schema_version': 1, 'complete': True,
              'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'selection_plan': plan_identity, 'inputs': plan['inputs'],
              'scores_file': file_record(score_path), 'selected_weight_vectors_file': file_record(vector_path),
              'formula': plan['formula'], 'physical_master_dtype': 'little-endian FP32',
              'compute_operand_dtype': 'BF16 round-to-nearest-even; analysis uses FP64 arithmetic',
              'vocabulary_size': 50257, 'unique_feature_count': len(results),
              'selected_context_feature_count': len(plan['selected']),
              'all_context_identity_count': len(checks), 'max_identity_abs_error': max(checks),
              'all_inputs_unchanged': True, 'features': results,
              'limits': ['A weight affinity is a context-independent frozen-final-LN readout numerator, not a logit or probability.',
                         'Actual ledger contribution multiplies the target-minus-competitor affinity by native post-GELU activation / actual final residual standard deviation.',
                         'LayerNorm gamma is fixed; standard deviation is context-dependent. Earlier/later transformer responses to removing a neuron are not captured by this identity.',
                         'W1 columns respond to contextualized normalized vectors, not isolated words. Top-affinity vocabulary pieces do not name a unique semantic detector.',
                         'Selection used only pre-existing ledgers under a frozen rule. No corpus text or vocabulary-affinity outcome influenced feature selection.']}
    write_exclusive(output, report)
    print({'output': file_record(output), 'unique_features': len(results),
           'identity_checks': len(checks), 'max_identity_abs_error': max(checks)})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    freezing = sub.add_parser('freeze')
    for name in ('root', 'checkpoint', 'tokenizer', 'plan'):
        freezing.add_argument('--' + name, type=Path, required=True)
    analysis = sub.add_parser('analyze')
    analysis.add_argument('--plan', type=Path, required=True)
    analysis.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.root, args.checkpoint, args.tokenizer, args.plan)
    else:
        analyze(args.plan, args.output)


if __name__ == '__main__':
    main()
