"""Explain a real native forward trace without replacing it by a surrogate.

Native per-position logits and intermediate readouts are authoritative. The
CPU reconstruction is a separate numerical cross-check. Fixed-final-normalizer
contributions are an additive accounting identity, NOT causal effects; actual
component-removal forward passes are reported separately for that reason.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from .checkpoint import GPT2Checkpoint, GPT2Config
from .late_mlp_paths import file_record, write_exclusive
from .verify import gpt2_token_bytes


def checked_matrix(value):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not min(result.shape) or not np.isfinite(result).all():
        raise ValueError('expected a finite nonempty matrix')
    return result


def predictions(logits, token_bytes, fixed_winners=None, next_inputs=None):
    """Read native FP32 logits in FP64; rank ties by ascending token ID.

Probabilities are a stable temperature-one softmax of those saved logits,
not samples. Each position predicts the token AFTER its input token.
    """
    logits = checked_matrix(logits)
    rows, vocabulary = logits.shape
    if vocabulary < 2 or set(token_bytes) != set(range(vocabulary)):
        raise ValueError('need at least two logits and exact logical token-byte coverage')
    if any(not isinstance(value, bytes) for value in token_bytes.values()):
        raise ValueError('token spellings must be exact bytes')
    for label, ids in (('fixed_winners', fixed_winners), ('next_inputs', next_inputs)):
        if ids is not None:
            ids = np.asarray(ids)
            if (ids.shape != (rows,) or ids.dtype.kind not in 'iu'
                    or np.any(ids < 0) or np.any(ids >= vocabulary)):
                raise ValueError(label + ' must contain one valid integer ID per row')
    result = []
    for index, row in enumerate(logits):
        order = np.lexsort((np.arange(len(row)), -row))
        winner, runner = map(int, order[:2])
        shifted = row - row[winner]
        normalizer = float(np.exp(shifted).sum())
        probabilities = np.exp(shifted) / normalizer

        def entry(token):
            token = int(token)
            piece = token_bytes[token]
            rank = 1 + int(np.count_nonzero(row > row[token])) + int(np.count_nonzero(row[:token] == row[token]))
            return {'id': token, 'bytes_hex': piece.hex(),
                    'piece_escaped': repr(piece.decode('utf-8', errors='backslashreplace')),
                    'rank': rank, 'logit': float(row[token]), 'probability': float(probabilities[token])}

        item = {'position': index, 'winner': entry(winner), 'runner_up': entry(runner),
                'top5': [entry(token) for token in order[:5]],
                'winner_margin': float(row[winner] - row[runner]),
                'entropy_nats': float(np.log(normalizer) - np.dot(probabilities, shifted))}
        if fixed_winners is not None:
            item['final_winner'] = entry(fixed_winners[index])
        if next_inputs is not None and index + 1 < len(next_inputs):
            item['observed_next_input'] = entry(next_inputs[index + 1])
        result.append(item)
    return result


def compare_arrays(native, reference):
    native, reference = checked_matrix(native), checked_matrix(reference)
    if native.shape != reference.shape:
        raise ValueError('comparison shapes differ')
    error = reference - native
    norm = float(np.linalg.norm(native))
    return {'max_abs_error': float(np.max(np.abs(error))),
            'rms_error': float(np.sqrt(np.mean(error * error))),
            'relative_l2_error': float(np.linalg.norm(error) / norm) if norm else None,
            'values_equal': bool(np.array_equal(native, reference))}


def residual_stage_names(layers):
    return ['positioned', *[f'blocks.{b}.{kind}' for b in range(layers)
                            for kind in ('after_attention', 'after_mlp')]]


def stage_widths(config):
    """The exact named activation schema shared by validation and accounting."""
    expected = {'embedding': config.d_model, 'positioned': config.d_model,
                'final_norm': config.d_model}
    for block in range(config.n_layers):
        for name in ('ln1', 'attention', 'attention_projected', 'after_attention',
                     'ln2', 'mlp_projected', 'after_mlp'):
            expected[f'blocks.{block}.{name}'] = config.d_model
        expected[f'blocks.{block}.qkv'] = 3 * config.d_model
        expected[f'blocks.{block}.fc1'] = config.d_ff
        expected[f'blocks.{block}.gelu'] = config.d_ff
    return expected


def validate_stages(stages, rows, config):
    for name, width in stage_widths(config).items():
        if name not in stages or checked_matrix(stages[name]).shape != (rows, width):
            raise ValueError('missing or incorrectly shaped native stage: ' + name)


def emergence(lens_predictions, winners):
    """First and first-persistent readout agreement; no decision-time claim."""
    names = list(lens_predictions)
    results = []
    for row, target in enumerate(winners):
        match = [lens_predictions[name][row]['winner']['id'] == target for name in names]
        first = next((i for i, yes in enumerate(match) if yes), None)
        persistent = next((i for i in range(len(match)) if all(match[i:])), None)
        results.append({'position': row, 'final_winner_id': int(target),
                        'first_readout_match': names[first] if first is not None else None,
                        'first_persistent_readout_match': names[persistent] if persistent is not None else None,
                        'matching_stage_count': sum(match),
                        'warning': 'Intermediate final-head readout is a heuristic, not when the network irreversibly decides.'})
    return results


def reconstructed_attention(qkv, heads):
    """Diagnostic probabilities from captured Q/K; they were not stored by FlashAttention.

    FP64 evaluation makes this an independent mathematical reconstruction, not
    an assertion of native online-softmax arithmetic identity. Its predicted
    context is compared against the captured native context before use.
    """
    qkv = checked_matrix(qkv)
    rows, triple_width = qkv.shape
    if type(heads) is not int or heads <= 0 or triple_width % (3 * heads):
        raise ValueError('invalid packed QKV width or head count')
    width = triple_width // 3
    dimension = width // heads
    q, k, v = [x.reshape(rows, heads, dimension).transpose(1, 0, 2) for x in np.split(qkv, 3, axis=1)]
    scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(dimension)
    scores = np.where(np.tril(np.ones((rows, rows), dtype=bool))[None], scores, -np.inf)
    scores -= scores.max(axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    context = (probabilities @ v).transpose(1, 0, 2).reshape(rows, width)
    return probabilities, v, context


def margin_accounting(checkpoint, stages, logits):
    """Decompose the fixed native winner-vs-runner margin, retaining all rounding.

    Freeze only the final residual's per-token normalization scale. This makes
    the final logit difference a linear readout of its centered residual, plus
    final-LN bias and explicit native numerical remainders. Earlier residual
    additions telescope exactly; each MLP neuron and attention head is then
    assigned its linear contribution in that fixed readout direction.

    This is NOT a statement of how the model would behave if a contribution
    were removed: later layers and their normalizers would also change. Native
    intervention runs measure that different, genuinely causal question.
    """
    from .phrase_reference import bf16

    config = checkpoint.config
    tensors = checkpoint.tensors
    logits = checked_matrix(logits)
    n = len(logits)
    if logits.shape[1] != config.vocab_size or config.vocab_size < 2:
        raise ValueError('accounting needs the complete logical vocabulary logits')
    validate_stages(stages, n, config)
    order = np.argsort(-logits, axis=1, kind='stable')
    winners, runners = order[:, 0], order[:, 1]
    embedding = bf16(tensors['token_embedding.weight']).astype(np.float64)
    readout = embedding[winners] - embedding[runners]
    final = checked_matrix(stages[f'blocks.{config.n_layers - 1}.after_mlp'])
    gamma = np.asarray(tensors['final_norm.scale'], dtype=np.float64)
    beta = np.asarray(tensors['final_norm.bias'], dtype=np.float64)
    mean = final.mean(axis=1, keepdims=True)
    deviation = np.sqrt(np.mean((final - mean) ** 2, axis=1, keepdims=True) + 1e-5)
    direction = readout * gamma / deviation
    direction -= direction.mean(axis=1, keepdims=True)
    ideal_normalized = (final - mean) / deviation * gamma + beta
    terms = {}

    def project(value):
        return np.sum(np.asarray(value, dtype=np.float64) * direction, axis=1)

    def add(name, value):
        terms[name] = project(value)

    base = checked_matrix(stages['embedding'])
    positions = np.asarray(tensors['position_embedding.weight'][:n], dtype=np.float64)
    add('token_embedding', base)
    add('position_embedding', positions)
    add('position_add_rounding', checked_matrix(stages['positioned']) - base - positions)
    head_terms, neuron_terms, attention_diagnostics = {}, {}, {}
    residual = checked_matrix(stages['positioned'])
    for block in range(config.n_layers):
        key = f'blocks.{block}'
        context = checked_matrix(stages[key + '.attention'])
        attention_weight = bf16(tensors[key + '.attn.output.weight']).astype(np.float64)
        attention_bias = np.asarray(tensors[key + '.attn.output.bias'], dtype=np.float64)
        by_head = np.empty((n, config.n_heads), dtype=np.float64)
        probabilities, values, predicted_context = reconstructed_attention(stages[key + '.qkv'], config.n_heads)
        source_terms = np.zeros((config.n_heads, n, n), dtype=np.float64)
        for head in range(config.n_heads):
            first, last = head * config.head_dim, (head + 1) * config.head_dim
            weight = attention_weight[first:last]
            contribution = context[:, first:last] @ weight
            by_head[:, head] = project(contribution)
            # Position/source decomposition is diagnostic; compare its sum
            # against native-context head accounting to expose online-rounding.
            source_values = values[head] @ weight
            source_terms[head] = probabilities[head] * (direction @ source_values.T)
        head_terms[key] = by_head
        terms[key + '.attention_heads'] = by_head.sum(axis=1)
        add(key + '.attention_bias', attention_bias)
        projected = checked_matrix(stages[key + '.attention_projected'])
        add(key + '.attention_projection_rounding', projected - (context @ attention_weight + attention_bias))
        after_attention = checked_matrix(stages[key + '.after_attention'])
        add(key + '.attention_residual_rounding', after_attention - residual - projected)
        attention_diagnostics[key] = {'probabilities': probabilities, 'source_margin_terms': source_terms,
                                      'context_math_vs_native': compare_arrays(context, predicted_context),
                                      'source_sum_vs_native_head_max_abs': float(np.max(np.abs(source_terms.sum(axis=2).T - by_head)))}
        active = checked_matrix(stages[key + '.gelu'])
        mlp_weight = bf16(tensors[key + '.mlp.output.weight']).astype(np.float64)
        mlp_bias = np.asarray(tensors[key + '.mlp.output.bias'], dtype=np.float64)
        by_neuron = active * (direction @ mlp_weight.T)
        neuron_terms[key] = by_neuron
        terms[key + '.mlp_neurons'] = by_neuron.sum(axis=1)
        add(key + '.mlp_bias', mlp_bias)
        projected = checked_matrix(stages[key + '.mlp_projected'])
        add(key + '.mlp_projection_rounding', projected - (active @ mlp_weight + mlp_bias))
        after_mlp = checked_matrix(stages[key + '.after_mlp'])
        add(key + '.mlp_residual_rounding', after_mlp - after_attention - projected)
        residual = after_mlp
    terms['final_norm_bias'] = np.sum(beta * readout, axis=1)
    native_normalized = checked_matrix(stages['final_norm'])
    terms['final_norm_rounding_and_reduction'] = np.sum((native_normalized - ideal_normalized) * readout, axis=1)
    native_margin = logits[np.arange(n), winners] - logits[np.arange(n), runners]
    terms['head_fp32_accumulation_remainder'] = native_margin - np.sum(native_normalized * readout, axis=1)
    total = np.sum(list(terms.values()), axis=0)
    if not np.allclose(total, native_margin, atol=2e-9, rtol=2e-11):
        raise ValueError('fixed-final-normalizer margin accounting did not close')
    return {'winners': winners, 'runners': runners, 'native_margin': native_margin,
            'terms': terms, 'head_terms': head_terms, 'neuron_terms': neuron_terms,
            'attention': attention_diagnostics, 'max_closure_error': float(np.max(np.abs(total - native_margin))),
            'final_std': deviation[:, 0], 'direction': direction}


def load_native_array(directory, record):
    """Read a small prefix dump, never a guessed shape or an escaping path."""
    directory = Path(directory).resolve()
    relative = Path(record['file'])
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name in ('', '.', '..'):
        raise ValueError('native array must be a filename within its run directory')
    path = directory / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError('native array is not a regular nonsymlink file')
    shape = record['shape']
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(size) is not int or size <= 0 for size in shape)):
        raise ValueError('invalid native matrix shape')
    dtypes = {'bf16': np.dtype('<u2'), 'float32': np.dtype('<f4'), 'int32': np.dtype('<i4')}
    if record['dtype'] not in dtypes:
        raise ValueError('unsupported native dump dtype')
    dtype = dtypes[record['dtype']]
    if path.stat().st_size != shape[0] * shape[1] * dtype.itemsize:
        raise ValueError('native dump byte size disagrees with shape')
    identity = file_record(path)
    values = np.fromfile(path, dtype=dtype).reshape(shape)
    if record['dtype'] == 'bf16':
        values = (values.astype(np.uint32) << 16).view(np.float32)
    if not np.isfinite(values).all() or file_record(path) != identity:
        raise ValueError('native array is nonfinite or changed while reading')
    return values, identity


def stage_summary(values):
    values = checked_matrix(values)
    return {'shape': list(values.shape), 'row_l2': np.linalg.norm(values, axis=1).tolist(),
            'row_mean': values.mean(axis=1).tolist(), 'row_std': values.std(axis=1).tolist(),
            'minimum': values.min(axis=1).tolist(), 'maximum': values.max(axis=1).tolist()}


def validate_native_dimensions(metadata, config):
    """Reject guessed dimensions and token identities before loading any dump."""
    for name in ('context_length', 'vocab_size', 'padded_vocab_size'):
        if type(metadata.get(name)) is not int or metadata[name] != getattr(config, name):
            raise ValueError('native ' + name + ' differs from checkpoint architecture')
    ids = metadata.get('token_ids')
    if (not isinstance(ids, list) or not 0 < len(ids) <= config.context_length
            or type(metadata.get('prompt_rows')) is not int
            or len(ids) != metadata['prompt_rows']
            or any(type(token) is not int or not 0 <= token < config.vocab_size for token in ids)):
        raise ValueError('invalid native prompt token IDs')
    return ids


def validate_intervention_arms(arms, rows, config):
    """Accept either the complete standard sweep or none, plus chosen neurons.

    The production sweep has 16 branches + 64 heads = 80 arms. Its smaller
    counterpart for a synthetic GPT2Config allows tests without huge arrays.
    Names and semantic identities must both be unique. This validates the
    recorded intervention contract, not the truth of a self-reported restore;
    independently checked final clean replay provides an additional gate.
    """
    if not isinstance(arms, list):
        raise ValueError('interventions must be a list')
    names, identities, standard = set(), set(), set()
    for arm in arms:
        if not isinstance(arm, dict):
            raise ValueError('invalid intervention record')
        kind, block = arm.get('kind'), arm.get('block')
        if (kind not in ('attention_branch', 'mlp_branch', 'attention_head', 'mlp_neuron')
                or type(block) is not int or not 0 <= block < config.n_layers):
            raise ValueError('invalid intervention kind or block')
        scale = arm.get('scale')
        if type(scale) not in (int, float) or not np.isfinite(scale) or scale != 0:
            raise ValueError('intervention must use exact zero scale')
        if arm.get('restoration_verified_bytes') is not True:
            raise ValueError('intervention restoration was not verified')
        element = arm.get('head_or_neuron')
        if kind in ('attention_branch', 'mlp_branch'):
            if 'head_or_neuron' in arm:
                raise ValueError('a whole-branch intervention cannot name an element')
            suffix = kind
        else:
            bound = config.n_heads if kind == 'attention_head' else config.d_ff
            if type(element) is not int or not 0 <= element < bound:
                raise ValueError('invalid intervention head or neuron')
            suffix = ('head' if kind == 'attention_head' else 'neuron') + str(element)
        expected_name = f'ablation.block{block}.{suffix}'
        identity = (kind, block, element)
        if (arm.get('name') != expected_name or expected_name in names
                or identity in identities):
            raise ValueError('duplicate or inconsistent intervention name/identity')
        shape = arm.get('shape')
        if (not isinstance(shape, list) or len(shape) != 2
                or any(type(size) is not int for size in shape)
                or shape != [rows, config.padded_vocab_size]):
            raise ValueError('intervention must contain all physical logit columns')
        if arm.get('logits_file') != expected_name + '.logits.f32':
            raise ValueError('intervention filename disagrees with its identity')
        names.add(expected_name)
        identities.add(identity)
        if kind != 'mlp_neuron':
            standard.add(identity)
    expected_standard = {(kind, block, None) for block in range(config.n_layers)
                         for kind in ('attention_branch', 'mlp_branch')}
    expected_standard.update(('attention_head', block, head)
                             for block in range(config.n_layers)
                             for head in range(config.n_heads))
    if standard and standard != expected_standard:
        raise ValueError('standard interventions must be absent or a complete branch/head sweep')


def load_native_evidence(directory, metadata, config):
    """Load and hash the native evidence independently of metadata check flags.

    Metadata assertions are not sufficient evidence that a replay agrees. Both
    exported replay files and the final lens are loaded independently and their
    exact FP32 bytes compared with baseline, preserving signed-zero differences.
    Every file identity is retained for the caller's final unchanged-input scan.
    This gate is also usable with small, fully synthetic traces in unit tests.
    """
    ids = validate_native_dimensions(metadata, config)
    rows = len(ids)
    widths = stage_widths(config)
    files = metadata.get('files')
    if not isinstance(files, dict) or set(files) != set(widths) | {'logits', 'tokens'}:
        raise ValueError('native files do not match the exact activation/logit/token schema')
    records, seen_paths = {}, set()

    def load(label, record, dtype, shape):
        if (not isinstance(record, dict) or record.get('dtype') != dtype
                or record.get('shape') != shape):
            raise ValueError('wrong dtype or shape for native ' + label)
        values, identity = load_native_array(directory, record)
        if identity['path'] in seen_paths:
            raise ValueError('native evidence files must be independently named: ' + label)
        seen_paths.add(identity['path'])
        records[label] = identity
        return values

    arrays = {name: load('stage:' + name, files[name], 'bf16', [rows, width])
              for name, width in widths.items()}
    arrays['logits'] = load('stage:logits', files['logits'], 'float32',
                           [rows, config.padded_vocab_size])
    arrays['tokens'] = load('stage:tokens', files['tokens'], 'int32', [rows, 1])
    if not np.array_equal(arrays['tokens'][:, 0], np.asarray(ids, dtype=np.int32)):
        raise ValueError('native token dump disagrees with metadata token IDs')
    baseline_bytes = arrays['logits'].tobytes(order='C')
    parity = metadata.get('parity_files')
    if not isinstance(parity, dict) or set(parity) != {'alternate_padding', 'clean_replay'}:
        raise ValueError('missing independently saved native parity files')
    parity_checks = {}
    for name, filename in parity.items():
        values = load('parity:' + name,
                      {'file': filename, 'dtype': 'float32',
                       'shape': [rows, config.padded_vocab_size]},
                      'float32', [rows, config.padded_vocab_size])
        if values.tobytes(order='C') != baseline_bytes:
            raise ValueError('native parity file is not byte-identical: ' + name)
        parity_checks[name + '_logits_byte_equal'] = True
    names = residual_stage_names(config.n_layers)
    lens_metadata = metadata.get('lens')
    if not isinstance(lens_metadata, dict) or set(lens_metadata) != set(names):
        raise ValueError('incomplete residual-stage lens')
    lens = {name: load('lens:' + name, lens_metadata[name], 'float32',
                       [rows, config.padded_vocab_size]) for name in names}
    if lens[names[-1]].tobytes(order='C') != baseline_bytes:
        raise ValueError('final native lens is not byte-identical to baseline logits')
    parity_checks['final_lens_logits_byte_equal'] = True
    arms = metadata.get('interventions')
    validate_intervention_arms(arms, rows, config)
    interventions = {arm['name']: load(
        'intervention:' + arm['name'],
        {'file': arm['logits_file'], 'dtype': 'float32', 'shape': arm['shape']},
        'float32', [rows, config.padded_vocab_size]) for arm in arms}
    return dict(arrays=arrays, lens=lens, interventions=interventions,
                records=records, parity_checks=parity_checks)


def run(native_directory, checkpoint_directory, tokenizer_directory, output_directory):
    from tokenizers import Tokenizer
    from .phrase_reference import forward

    native_directory = Path(native_directory).resolve()
    output = Path(output_directory).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    metadata_path = native_directory / 'metadata.json'
    metadata_identity = file_record(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('complete') is not True or metadata.get('autoregressive') is not False or metadata.get('no_bos') is not True:
        raise ValueError('need a completed non-autoregressive native trace without BOS')
    required_checks = ('final_lens_logits_byte_equal', 'alternate_padding_logits_byte_equal',
                       'clean_replay_logits_byte_equal', 'attention_residual_replay_byte_equal',
                       'mlp_residual_replay_byte_equal')
    if any(metadata.get('checks', {}).get(key) is not True for key in required_checks):
        raise ValueError('native replay or parity gate did not pass')
    if Path(metadata['checkpoint_directory']).resolve() != Path(checkpoint_directory).resolve():
        raise ValueError('analysis checkpoint differs from native run')
    if Path(metadata['tokenizer_directory']).resolve() != Path(tokenizer_directory).resolve():
        raise ValueError('analysis tokenizer differs from native run')
    checkpoint = GPT2Checkpoint(checkpoint_directory, check_finite=True)
    config = checkpoint.config
    ids = validate_native_dimensions(metadata, config)
    n = len(ids)
    tokenizer_path = Path(tokenizer_directory) / 'tokenizer.json'
    input_records = {'metadata': metadata_identity, 'tokenizer': file_record(tokenizer_path),
                     **{spec.name: file_record(checkpoint.directory / spec.filename) for spec in checkpoint.manifest}}
    token_bytes = gpt2_token_bytes(Tokenizer.from_file(str(tokenizer_path)))
    if b''.join(token_bytes[token] for token in ids) != metadata['prompt'].encode('utf-8'):
        raise ValueError('native prompt token bytes do not reproduce the literal phrase')
    output.mkdir()
    write_exclusive(output / 'plan.json', {
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'native_metadata': metadata_identity, 'inputs': input_records,
        'analysis_sources': {name: file_record(Path(__file__).with_name(name)) for name in
                             ('phrase_trace_analysis.py', 'phrase_reference.py', 'checkpoint.py', 'verify.py',
                              'late_mlp_paths.py')},
        'scope': 'Native short-phrase forward trace, readout diagnostics and causal ablations; not training-data extraction.'})
    plan_identity = file_record(output / 'plan.json')
    evidence = load_native_evidence(native_directory, metadata, config)
    arrays, native_records = evidence['arrays'], evidence['records']
    native_logits = checked_matrix(arrays['logits'])[:, :config.vocab_size]
    final_predictions = predictions(native_logits, token_bytes, next_inputs=ids)
    winners = [row['winner']['id'] for row in final_predictions]
    runners = [row['runner_up']['id'] for row in final_predictions]
    lens = {}
    stage_names = residual_stage_names(config.n_layers)
    for name in stage_names:
        values = evidence['lens'][name]
        lens[name] = predictions(values[:, :config.vocab_size], token_bytes, winners, ids)
    accounting = margin_accounting(checkpoint, arrays, native_logits)
    reference_comparisons = {}
    for mode in ('bf16', 'fp32'):
        reference = forward(checkpoint, ids, mode=mode)
        stages = reference['stages']
        reference_comparisons[mode] = {
            'stages': {name: compare_arrays(arrays[name], value) for name, value in stages.items() if name in arrays},
            'logits': compare_arrays(native_logits, reference['logits'][:, :config.vocab_size]),
            'predictions': predictions(reference['logits'][:, :config.vocab_size], token_bytes, winners, ids),
            'winner_agreement_count': int(np.count_nonzero(np.argmax(reference['logits'][:, :config.vocab_size], axis=1) == winners))}
    interventions = []
    base_margin = np.array([row['winner_margin'] for row in final_predictions])
    for arm in metadata['interventions']:
        # Compare saved FP32 logits in FP64, exactly as predictions() does;
        # do not introduce an extra FP32 rounding in the intervention margin.
        values = checked_matrix(evidence['interventions'][arm['name']])[:, :config.vocab_size]
        fixed_margin = values[np.arange(n), winners] - values[np.arange(n), runners]
        interventions.append({**arm, 'predictions': predictions(values, token_bytes, winners, ids),
                              'fixed_winner_runner_margin': fixed_margin.tolist(),
                              'margin_change_from_clean': (fixed_margin - base_margin).tolist(),
                              'changed_top1_positions': np.flatnonzero(np.argmax(values, axis=1) != winners).tolist()})
    neuron_details = {}
    for block in range(config.n_layers):
        key = f'blocks.{block}'
        values = accounting['neuron_terms'][key]
        neuron_details[key] = []
        for row in range(n):
            increasing = np.lexsort((np.arange(config.d_ff), values[row]))
            decreasing = np.lexsort((np.arange(config.d_ff), -values[row]))
            def describe(neuron):
                return {'neuron': int(neuron), 'margin_contribution': float(values[row, neuron]),
                        'pre_gelu': float(arrays[key + '.fc1'][row, neuron]),
                        'post_gelu': float(arrays[key + '.gelu'][row, neuron]),
                        'output_weight_file': f'weight_{12 + 12 * block}.bin',
                        'output_weight_row_byte_offset': int(neuron) * config.d_model * 4}
            neuron_details[key].append({'position': row, 'top_positive': [describe(j) for j in decreasing[:5]],
                                        'top_negative': [describe(j) for j in increasing[:5]]})
    numerical_arrays = {**{key + '.heads': value for key, value in accounting['head_terms'].items()},
                        **{key + '.neurons': value for key, value in accounting['neuron_terms'].items()},
                        **{key + '.attention_probabilities': value['probabilities'] for key, value in accounting['attention'].items()},
                        **{key + '.source_margin_terms': value['source_margin_terms'] for key, value in accounting['attention'].items()},
                        'final_readout_direction': accounting['direction']}
    with (output / 'accounting_arrays.npz').open('xb') as handle:
        np.savez(handle, **numerical_arrays)
    arrays_identity = file_record(output / 'accounting_arrays.npz')
    for record in [*input_records.values(), *native_records.values(), plan_identity]:
        if file_record(record['path']) != record:
            raise ValueError('input or native trace changed during analysis')
    plan = json.loads((output / 'plan.json').read_text())
    for record in plan['analysis_sources'].values():
        if file_record(record['path']) != record:
            raise ValueError('analysis source changed during execution')
    result = {'schema_version': 1, 'complete': True,
              'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'plan': plan_identity, 'inputs': input_records, 'native_files': native_records,
              'accounting_arrays': arrays_identity, 'prompt': metadata['prompt'], 'token_ids': ids,
              'input_pieces': [repr(token_bytes[t].decode('utf-8', errors='backslashreplace')) for t in ids],
              'native_checks': metadata['checks'],
              'independently_verified_parity': evidence['parity_checks'],
              'final_predictions': final_predictions,
              'lens': lens, 'readout_emergence': emergence(lens, winners),
              'stage_statistics': {name: stage_summary(values) for name, values in arrays.items() if name != 'logits'},
              'reference_comparisons': reference_comparisons,
              'margin_accounting': {'native_margin': accounting['native_margin'].tolist(),
                                    'terms': {key: value.tolist() for key, value in accounting['terms'].items()},
                                    'final_std': accounting['final_std'].tolist(),
                                    'max_closure_error': accounting['max_closure_error'],
                                    'head_terms': {key: value.tolist() for key, value in accounting['head_terms'].items()},
                                    'neuron_details': neuron_details,
                                    'attention_diagnostics': {key: {name: value for name, value in entry.items()
                                                                    if name not in ('probabilities', 'source_margin_terms')}
                                                              for key, entry in accounting['attention'].items()}},
              'interventions': interventions, 'all_checked_inputs_unchanged': True,
              'limitations': ['The intermediate head readout is not a causal decision-time measurement.',
                              'Frozen-final-normalizer attribution is accounting, not removal effect.',
                              'QK probabilities/source contributions are reconstructed diagnostics, not captured native softmax tensors.',
                              'Only this literal prompt and checkpoint are characterized; neuron names do not imply general semantic functions.']}
    write_exclusive(output / 'analysis.json', result)
    print('Analyzed', n, 'positions,', len(lens), 'native residual readouts,', len(interventions), 'causal arms.')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native-directory', 'checkpoint-directory', 'tokenizer-directory', 'output-directory'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.native_directory, args.checkpoint_directory, args.tokenizer_directory, args.output_directory)


if __name__ == '__main__':
    main()
