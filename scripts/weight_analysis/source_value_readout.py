"""Independently audit one-source-V native attention intervention evidence.

This module never executes a model. It authenticates the recorded invocation,
all its inputs and outputs, checks the exact BF16 interventions, and recomputes
selected-token scores from the complete logical vocabulary. It does not infer
the native attention output numerically: that output is captured by the probe.
In particular a plausible reconstructed softmax is not a native replay gate.

The public entry point is intentionally restricted to the frozen full-size
GPT-2 assay. Pure numerical helpers accept small arrays for focused CPU tests.
Single-token changes are not complete-word or unique-storage evidence.
"""

from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np

from . import paired_causal_followup as causal
from .checkpoint import GPT2Config, tensor_manifest


FORMAT = 'pluto-source-value-readout-v1'
GEOMETRY = {'context_length': 1024, 'width': 512, 'head_dim': 64,
            'vocabulary': 50257, 'padded_vocabulary': 50272}
DOSES = {'one': 1.0, 'half': 0.5, 'zero': 0.0, 'one_after': 1.0}
SCORE_ATOL = 2e-10  # Native sequential vs independent pairwise FP64 summation.
SCOPE = ('one source V; only one query/head context retained; Q/K unchanged; '
         'no masking or renormalization')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'duplicate JSON field: ' + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError('nonfinite JSON literal: ' + value)

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=invalid)


def _finite_bf16(bits):
    _require(np.asarray(bits).dtype == np.dtype('<u2'), 'expected raw BF16 bits')
    _require(not np.any((bits & 0x7f80) == 0x7f80), 'nonfinite BF16 tensor')


def scale_bf16(bits, scale):
    """Nearest representable-value oracle, independent of the native bit shift.

    BF16 values, including the smallest subnormal, are exact in FP64. Halve
    those values, search the complete positive finite BF16 number line, then
    choose the nearest neighbor (even significand on a tie). This deliberately
    differs from the native implementation's exponent/significand arithmetic.
    Zero dose erases signs; identity and half preserve negative zero.
    """
    bits = np.asarray(bits)
    _finite_bf16(bits)
    _require(type(scale) in (int, float) and scale in (0.0, 0.5, 1.0), 'invalid value dose')
    if scale == 0:
        return np.zeros_like(bits)
    if scale == 1:
        return bits.copy()
    positive_bits = np.arange(0x7f80, dtype=np.uint32)
    number_line = (positive_bits << 16).view(np.float32).astype(np.float64)
    magnitudes = ((bits.astype(np.uint32) & 0x7fff) << 16).view(np.float32)
    desired = magnitudes.astype(np.float64) * 0.5
    upper = np.searchsorted(number_line, desired)
    lower = np.maximum(upper - 1, 0)
    below = desired - number_line[lower]
    above = number_line[upper] - desired
    choose_upper = (above < below) | ((above == below) & ((upper & 1) == 0))
    rounded = np.where(choose_upper, upper, lower).astype(np.uint16)
    return (rounded | (bits & 0x8000)).astype('<u2')


def score_logits(logits, target, rival=None):
    """FP64 full-vocabulary score, with lowest-ID tie breaks like native code."""
    values = np.asarray(logits, dtype=np.float64)
    _require(values.ndim == 1 and values.size > 1 and np.isfinite(values).all(),
             'expected finite vocabulary row')
    _require(type(target) is int and 0 <= target < values.size, 'invalid target')
    ids = np.arange(values.size)
    if rival is None:
        candidates = np.where(ids != target, values, -np.inf)
        rival = int(np.argmax(candidates))
    _require(type(rival) is int and 0 <= rival < values.size and rival != target,
             'invalid fixed rival')
    maximum = float(values.max())
    nll = float(np.log(np.exp(values - maximum).sum(dtype=np.float64))
                + (maximum - values[target]))
    return {'nll': nll, 'probability': math.exp(-nll),
            'argmax': int(np.argmax(values)),
            'target_rank': int(1 + np.count_nonzero(
                (values > values[target]) | ((values == values[target]) & (ids < target)))),
            'target_logit': float(values[target]), 'rival_logit': float(values[rival]),
            'fixed_rival_id': rival,
            'fixed_rival_margin': float(values[target] - values[rival])}


def audit_tensors(original_qkv, original_context, qkv, replayed, spliced,
                  *, head, head_dim, query, source, scale):
    """Check byte-level isolation, not an approximate attention reconstruction."""
    arrays = [original_qkv, original_context, qkv, replayed, spliced]
    for bits in arrays:
        _finite_bf16(bits)
    rows, width = original_context.shape
    _require(original_qkv.shape == qkv.shape == (rows, 3 * width)
             and replayed.shape == spliced.shape == (rows, width), 'tensor shape mismatch')
    _require(all(type(value) is int for value in (head_dim, head, query, source))
             and head_dim > 0 and width % head_dim == 0 and 0 <= head < width // head_dim
             and 0 <= query < rows and 0 <= source < rows, 'invalid tensor selection')
    head_slice = slice(head * head_dim, (head + 1) * head_dim)
    value_slice = slice(2 * width + head * head_dim, 2 * width + (head + 1) * head_dim)
    expected_qkv = original_qkv.copy()
    expected_qkv[source, value_slice] = scale_bf16(original_qkv[source, value_slice], scale)
    _require(np.array_equal(qkv, expected_qkv), 'QKV differs from exact source-V dose')
    expected_context = original_context.copy()
    expected_context[query, head_slice] = replayed[query, head_slice]
    _require(np.array_equal(spliced, expected_context), 'context splice changed outside selected query/head')
    if scale == 1:
        _require(np.array_equal(replayed, original_context), 'identity attention differs from original')
        _require(np.array_equal(spliced, original_context), 'identity spliced context differs from original')
    if source > query:
        _require(np.array_equal(spliced, original_context), 'future source changed causal query context')
    return {'qkv_changed_elements': int(np.count_nonzero(qkv != original_qkv)),
            'spliced_context_changed_elements': int(np.count_nonzero(spliced != original_context)),
            'replayed_attention_changed_elements': int(np.count_nonzero(replayed != original_context))}


def _records(items, name):
    _require(isinstance(items, list) and items, 'missing ' + name + ' records')
    paths = []
    for item in items:
        _require(isinstance(item, dict) and set(item) == {'path', 'bytes', 'sha256'},
                 'invalid ' + name + ' record')
        path = Path(item['path'])
        _require(path.is_absolute() and path == path.resolve() and path.is_file()
                 and not path.is_symlink(), 'noncanonical/nonregular recorded file')
        paths.append(str(path))
    _require(len(paths) == len(set(paths)), 'duplicate ' + name + ' records')
    causal.verify_records(items)
    return dict(zip(paths, items))


def _read_tensor(path, dtype, shape):
    path = Path(path)
    _require(path.stat().st_size == math.prod(shape) * np.dtype(dtype).itemsize,
             'tensor byte count differs: ' + path.name)
    value = np.fromfile(path, dtype=dtype).reshape(shape)
    if dtype == '<u2':
        _finite_bf16(value)
    else:
        _require(np.isfinite(value).all(), 'nonfinite native tensor: ' + path.name)
    return value


def _scores_match(declared, computed, fields):
    for declared_name, computed_name in fields.items():
        actual, expected = declared.get(declared_name), computed[computed_name]
        if computed_name in ('argmax', 'target_rank', 'fixed_rival_id'):
            _require(type(actual) is int and actual == expected, 'declared integer score differs: ' + declared_name)
        else:
            absolute_tolerance = 0.0 if computed_name in ('probability', 'target_logit', 'rival_logit') else SCORE_ATOL
            _require(type(actual) in (int, float) and math.isfinite(actual)
                     and math.isclose(actual, expected, rel_tol=2e-12, abs_tol=absolute_tolerance),
                     'declared score differs: ' + declared_name)


def summarize(directory, execution_path, assay):
    """Validate a five-source, four-dose native assay and return audited scores.

    ``assay`` supplies block/head/query/target_id/sources, fixed GPT-2 geometry,
    historical_checkpoint, and authenticated prefix/expected_logits/
    expected_qkv/expected_context records. The caller must separately bind
    these records to the frozen historical manifest and predeclared selection.
    No experiment is launched and no evidence file is written by this function.
    """
    directory, execution_path = Path(directory), Path(execution_path)
    _require(directory.is_absolute() and directory == directory.resolve()
             and directory.is_dir() and not directory.is_symlink(), 'invalid evidence directory')
    _require(all(type(assay.get(key)) is int and assay[key] == value
                 for key, value in GEOMETRY.items()), 'assay requires fixed native GPT-2 geometry')
    for key, limit in (('block', 8), ('head', 8), ('query', 1024), ('target_id', 50257)):
        _require(type(assay.get(key)) is int and 0 <= assay[key] < limit, 'invalid assay ' + key)
    sources = assay.get('sources')
    _require(isinstance(sources, list) and len(sources) == 5
             and all(type(source) is int and 0 <= source < 1024 for source in sources)
             and len(set(sources)) == 5, 'assay requires five distinct sources')
    frozen = [assay[name] for name in ('prefix', 'expected_logits', 'expected_qkv', 'expected_context')]
    _records(frozen, 'historical')
    checkpoint = Path(assay['historical_checkpoint'])
    _require(checkpoint.is_absolute() and checkpoint == checkpoint.resolve()
             and checkpoint.is_dir() and not checkpoint.is_symlink(), 'invalid historical checkpoint')
    metadata = _json(directory / 'metadata.json')
    execution_record = causal.record(execution_path)
    execution = _json(execution_path)
    _require(execution.get('format') == 'pluto-source-value-execution-v1'
             and type(execution.get('returncode')) is int and execution['returncode'] == 0,
             'missing successful source-value execution')
    _require(type(execution.get('pid')) is int and execution['pid'] > 0, 'missing child PID')
    times = [datetime.fromisoformat(execution[key]) for key in ('started_utc', 'finished_utc')]
    _require(all(t.tzinfo is not None for t in times) and times[1] >= times[0], 'invalid execution times')
    _require(execution.get('inputs_before') == execution.get('inputs_after'), 'probe inputs changed')
    inputs = _records(execution.get('inputs_before'), 'input')
    output_records = _records(execution.get('outputs'), 'output')
    _records([execution['log']], 'log')
    command = execution.get('command')
    _require(isinstance(command, list) and len(command) == 10
             and all(isinstance(arg, str) for arg in command), 'invalid execution command')
    binary = Path(command[0])
    _require(binary.is_absolute() and str(binary) in inputs, 'binary not bound to execution')
    flags = {}
    for argument in command[1:]:
        _require(argument.startswith('--') and '=' in argument, 'expected explicit --flag=value')
        key, value = argument[2:].split('=', 1)
        _require(key not in flags, 'duplicate execution flag')
        flags[key] = value
    expected_flags = {'checkpoint': str(checkpoint), 'tokens_file': assay['prefix']['path'],
                      'output_dir': str(directory), 'block': str(assay['block']),
                      'head': str(assay['head']), 'query': str(assay['query']),
                      'sources': ','.join(map(str, sources)), 'target_id': str(assay['target_id']),
                      'expected_logits': assay['expected_logits']['path']}
    # There are nine flags, plus the executable. Keep this exact, not a subset
    # test that would admit a different actual intervention via duplicate flags.
    _require(flags == expected_flags, 'execution flags differ from frozen assay')
    required_inputs = {item['path'] for item in frozen} | {str(binary)}
    weight_names = {f'weight_{i}.bin' for i in range(100)}
    _require({path.name for path in checkpoint.iterdir()} <= weight_names | {'patch.json'}
             and all((checkpoint / name).is_file() for name in weight_names), 'invalid checkpoint inventory')
    for spec in tensor_manifest(GPT2Config()):
        _require((checkpoint / spec.filename).stat().st_size == spec.nbytes,
                 'invalid checkpoint weight size: ' + spec.filename)
    required_inputs |= {str(path) for path in checkpoint.iterdir()}
    _require(required_inputs <= inputs.keys(), 'execution missing historical/checkpoint inputs')
    for item in frozen:
        _require(inputs[item['path']] == item, 'execution uses different historical input')
    _require(metadata.get('format') == 'pluto-source-value-probe-v1'
             and metadata.get('complete') is True and metadata.get('scope') == SCOPE,
             'incomplete or different native intervention')
    for flag in ('native_full_attention_identity', 'native_all_row_padded_logit_identity',
                 'weight_disk_and_device_bytes_unchanged', 'original_full_forward_replay_equal'):
        _require(metadata.get(flag) is True, 'missing native gate: ' + flag)
    for key in ('checkpoint', 'tokens_file', 'expected_logits'):
        _require(metadata.get(key) == expected_flags[key], 'metadata path differs: ' + key)
    _require(metadata.get('binary') == str(binary), 'native binary differs from command')
    for key in ('block', 'head', 'query', 'target_id', 'vocabulary', 'padded_vocabulary'):
        _require(type(metadata.get(key)) is int and metadata[key] == assay[key], 'metadata differs: ' + key)
    _require(type(metadata.get('execution_rows')) is int and metadata['execution_rows'] == 1024,
             'incorrect native execution rows')
    prefix_path = Path(assay['prefix']['path'])
    _require(prefix_path.stat().st_size % 4 == 0 and 0 < prefix_path.stat().st_size <= 4096,
             'invalid historical prefix size')
    prefix = np.fromfile(prefix_path, dtype='<i4')
    _require(np.all((prefix >= 0) & (prefix < 50257)) and assay['query'] < len(prefix)
             and max(sources) < len(prefix), 'invalid prefix IDs/positions')
    _require(type(metadata.get('prefix_rows')) is int and metadata['prefix_rows'] == len(prefix)
             and type(metadata.get('query_token_id')) is int
             and metadata['query_token_id'] == int(prefix[assay['query']]), 'metadata prefix differs')
    _require((directory / 'tokens.i32').read_bytes() == prefix_path.read_bytes(), 'prefix bytes differ')
    padded = np.full(1024, prefix[-1], dtype='<i4')
    padded[:len(prefix)] = prefix
    _require((directory / 'padded_tokens.i32').read_bytes() == padded.tobytes(), 'prefix padding differs')
    baseline = _read_tensor(directory / 'baseline.f32', '<f4', (50272,))
    original_qkv = _read_tensor(directory / 'original_qkv.bf16', '<u2', (1024, 1536))
    original_context = _read_tensor(directory / 'original_context.bf16', '<u2', (1024, 512))
    for name, file in (('expected_logits', 'baseline.f32'), ('expected_qkv', 'original_qkv.bf16'),
                       ('expected_context', 'original_context.bf16')):
        _require((directory / file).read_bytes() == Path(assay[name]['path']).read_bytes(),
                 'native baseline differs from historical ' + name)
    baseline_score = score_logits(baseline[:50257], assay['target_id'])
    _scores_match(metadata, baseline_score, {'baseline_nll': 'nll', 'baseline_probability': 'probability',
        'baseline_rank': 'target_rank', 'baseline_argmax': 'argmax', 'fixed_rival_id': 'fixed_rival_id',
        'baseline_fixed_rival_margin': 'fixed_rival_margin'})
    expected_arms = [(f'source_{source}_{label}', source, scale)
                     for source in sources for label, scale in DOSES.items()]
    arms = metadata.get('arms')
    _require(isinstance(arms, list) and len(arms) == 20, 'expected exactly twenty arms')
    expected_files = {'metadata.json', 'baseline.f32', 'tokens.i32', 'padded_tokens.i32',
                      'original_qkv.bf16', 'original_context.bf16'}
    expected_directories = {name for name, _, _ in expected_arms}
    for name in expected_directories:
        expected_files |= {f'{name}/{file}' for file in ('logits.f32', 'qkv.bf16',
                            'replayed_attention.bf16', 'spliced_context.bf16')}
    actual_files, actual_directories = set(), set()
    for path in directory.rglob('*'):
        _require(not path.is_symlink(), 'symlink in native output')
        relative = str(path.relative_to(directory))
        if path.is_dir():
            actual_directories.add(relative)
        else:
            _require(path.is_file(), 'nonregular native output')
            actual_files.add(relative)
    _require(actual_files == expected_files and actual_directories == expected_directories,
             'extra or missing native output files/directories')
    _require(set(output_records) == {str(directory / name) for name in expected_files},
             'execution output inventory differs from complete nested outputs')
    results = []
    for arm, (name, source, scale) in zip(arms, expected_arms):
        _require(arm.get('directory') == name and type(arm.get('source')) is int
                 and arm['source'] == source and arm.get('scale') == scale
                 and type(arm.get('scale')) in (float, int)
                 and type(arm.get('source_token_id')) is int
                 and arm.get('source_token_id') == int(prefix[source]), 'arm order/site/dose differs')
        path = directory / name
        qkv = _read_tensor(path / 'qkv.bf16', '<u2', (1024, 1536))
        replayed = _read_tensor(path / 'replayed_attention.bf16', '<u2', (1024, 512))
        spliced = _read_tensor(path / 'spliced_context.bf16', '<u2', (1024, 512))
        validation = audit_tensors(original_qkv, original_context, qkv, replayed, spliced,
            head=assay['head'], head_dim=64, query=assay['query'], source=source, scale=scale)
        logits = _read_tensor(path / 'logits.f32', '<f4', (50272,))
        _require(logits[50257:].tobytes() == baseline[50257:].tobytes(), 'padded logit bytes changed')
        if scale == 1:
            _require(logits.tobytes() == baseline.tobytes(), 'identity full selected logits differ')
        if source > assay['query']:
            _require(logits.tobytes() == baseline.tobytes(), 'future source changed causal query logits')
        score = score_logits(logits[:50257], assay['target_id'], baseline_score['fixed_rival_id'])
        _scores_match(arm, score, {key: key for key in ('nll', 'probability', 'argmax',
            'target_rank', 'target_logit', 'rival_logit', 'fixed_rival_margin')})
        results.append({'directory': name, 'source': source, 'source_token_id': int(prefix[source]),
            'scale': scale, **score, 'nll_change': score['nll'] - baseline_score['nll'],
            'probability_change': score['probability'] - baseline_score['probability'],
            'fixed_rival_margin_change': score['fixed_rival_margin'] - baseline_score['fixed_rival_margin'],
            'validation': validation})
    # Catch mutation during analysis rather than endorsing stale initial hashes.
    causal.verify_records(list(inputs.values()) + list(output_records.values())
                          + [execution['log'], execution_record])
    return {'format': FORMAT, 'complete': True, 'scope': SCOPE, 'assay': assay,
            'baseline': baseline_score, 'arms': results,
            'execution': execution_record, 'metadata': causal.record(directory / 'metadata.json'),
            'inputs': list(inputs.values()), 'outputs': list(output_records.values()),
            'validation': {'exact_source_v_doses': True, 'exact_query_head_splices': True,
                'identity_full_selected_logits': True, 'historical_baselines_equal': True,
                'full_logical_vocabulary_scores': True, 'score_absolute_tolerance': SCORE_ATOL},
            'limitations': ['This is a single target-token intervention, not a complete-word measurement.',
                'No unique storage location or semantic exclusivity is established.',
                'Only selected-query logits are exported; all-row replay parity is a native producer gate.',
                'Counterfactual attention values are native captures, not independently CPU-reconstructed.'],
            'new_paired_model_result': False, 'goal_completion_claimed': False}
