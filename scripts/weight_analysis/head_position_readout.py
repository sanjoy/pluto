"""Audit calibrated historical head-position evidence without executing a model.

The invocation record MUST have format ``pluto-head-position-execution-v1``:
command (absolute executable plus nine explicit --flag=value arguments), pid,
timezone-aware started_utc/finished_utc, returncode=0, a hashed log, identical
inputs_before/inputs_after lists, and the COMPLETE recursive output inventory.
Inputs must bind the descriptor, every descriptor provenance/implementation/
weight record, and the actual executable. The caller records these around its
owned native child; metadata emitted by the child alone is insufficient.

Full FP32 tensors are memory-mapped and checked in bounded row chunks. Only
small selected logical-vocabulary rows survive across arms. Calibration is
against archived selected rows, not an invented archive of padded/future rows.
Before/context visible prefixes are independently compared to old captures;
all-row native model/tail reconstruction remains a producer-side gate.
"""

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
import math
from pathlib import Path
import re

import numpy as np

from . import checkpoint, head_position_cases, head_position_math as oracle
from . import historical_word_ablations as historical
from . import source_value_readout
from .source_value_readout import _json

FORMAT = 'pluto-head-position-readout-v1'
EXECUTION_FORMAT = 'pluto-head-position-execution-v1'
DOSES = (('one', 1), ('half', 0.5), ('zero', 0), ('one_after', 1))
CHECKS = (
    'native_full_model_all_row_padded_identity', 'native_clean_attention_identity',
    'native_clean_tail_all_row_padded_identity', 'identity_all_row_padded_logits',
    'causally_unaffected_rows_equal', 'physical_padding_logits_equal',
    'archived_clean_selected_logits_equal', 'archived_all_query_zero_selected_logits_equal',
    'calibration_precedes_factorial', 'repeated_all_query_zero_full_logits_equal',
    'checkpoint_disk_and_device_bytes_unchanged', 'original_input_bytes_unchanged',
    'original_full_forward_replay_equal')


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Ledger:
    """Bind actual regular bytes before consumption and check again at return."""
    def __init__(self):
        self.records = {}

    def add(self, item):
        require(isinstance(item, dict) and set(item) == {'path', 'bytes', 'sha256'}
                and isinstance(item['path'], str) and type(item['bytes']) is int
                and item['bytes'] >= 0 and isinstance(item['sha256'], str)
                and re.fullmatch('[0-9a-f]{64}', item['sha256']), 'invalid file record')
        require(historical.record(item['path']) == item, 'recorded bytes changed: ' + item['path'])
        require(item['path'] not in self.records or self.records[item['path']] == item,
                'conflicting records for one path')
        self.records[item['path']] = item
        return Path(item['path'])

    def file(self, path):
        item = historical.record(path)
        self.add(item)
        return item

    def many(self, items):
        require(isinstance(items, list) and items, 'missing recorded file list')
        require(len({r['path'] for r in items}) == len(items), 'duplicate file record')
        for item in items:
            self.add(item)
        return {r['path']: r for r in items}

    def recheck(self):
        for item in self.records.values():
            require(historical.record(item['path']) == item, 'input/output changed during readout')


@contextmanager
def mapped(path, dtype, shape):
    path = Path(path)
    require(path.stat().st_size == math.prod(shape) * np.dtype(dtype).itemsize,
            'tensor size differs: ' + path.name)
    array = np.memmap(path, dtype=dtype, mode='r', shape=shape)
    try:
        yield array
    finally:
        array._mmap.close()


def equal_bits(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype
            and np.array_equal(np.asarray(a).view(np.uint8), np.asarray(b).view(np.uint8)))


def validate_descriptor(path, ledger, config, allow_synthetic):
    descriptor_record = ledger.file(path)
    document = _json(path)
    production = config == checkpoint.GPT2Config()
    require(production or allow_synthetic is True, 'nonproduction geometry requires explicit synthetic permission')
    require(document.get('format') == head_position_cases.FORMAT
            and document.get('complete') is True and document.get('historical_only') is True
            and document.get('config') == asdict(config)
            and document.get('production_fixed_assay') is production,
            'wrong/incomplete head-position descriptor')
    for key in ('new_paired_model_result', 'gpu_work_performed', 'goal_completion_claimed', 'calibration_executed'):
        require(document.get(key) is False, 'descriptor incorrectly claims measured/completed research')
    provenance = ledger.many(document['provenance'])
    ledger.many(document['implementation'])
    producer = document['producer_source']
    require(producer.get('matched_historical_plan') is True, 'historical producer source not authenticated')
    if 'snapshot' in producer:
        ledger.add(producer['snapshot'])
    directory = Path(document['checkpoint_directory'])
    require(directory.resolve() == directory and directory.is_dir() and not directory.is_symlink(),
            'invalid checkpoint directory')
    specs = checkpoint.tensor_manifest(config)
    require(len(document['weights']) == len(specs)
            and {p.name for p in directory.iterdir()} == {s.filename for s in specs},
            'checkpoint weight inventory differs')
    for spec, item in zip(specs, document['weights']):
        require(item['path'] == str(directory / spec.filename) and item['bytes'] == spec.nbytes
                and provenance.get(item['path']) == item, 'checkpoint record/shape differs')
        ledger.add(item)
        with mapped(item['path'], '<f4', (spec.nbytes // 4,)) as weight:
            for start in range(0, len(weight), 65536):
                require(np.isfinite(weight[start:start + 65536]).all(), 'nonfinite checkpoint weights')
    cases = document['cases']
    require(type(document.get('case_count')) is int and document['case_count'] == len(cases)
            and cases and all(type(c.get('case_index')) is int for c in cases)
            and [c['case_index'] for c in cases] == list(range(len(cases))),
            'invalid descriptor case count/order')
    if production:
        expected = [(word, step, target, piece, block, head)
                    for word, step, target, piece in head_position_cases.EVENTS
                    for block, head in head_position_cases.HEADS]
        require([(c['word'], c['generation_step'], c['target_id'], c['target_piece_hex'], c['block'], c['head'])
                 for c in cases] == expected, 'production descriptor changed fixed event/head selection')
    for case in cases:
        for key, limit in (('block', config.n_layers), ('head', config.n_heads),
                           ('target_id', config.vocab_size), ('query', config.context_length)):
            require(type(case[key]) is int and 0 <= case[key] < limit, 'invalid case ' + key)
        rows = case['visible_rows']; query = case['query']
        require(type(rows) is int and 1 <= rows <= config.context_length and query == rows - 1
                and case['model_context_length'] == config.context_length,
                'historical visible prefix/query differs')
        for item in (case['prefix'], case['metadata']):
            require(provenance.get(item['path']) == item, 'case record absent from provenance')
            ledger.add(item)
        with mapped(case['prefix']['path'], '<i4', (rows,)) as tokens:
            require(np.all((tokens >= 0) & (tokens < config.vocab_size)), 'invalid historical prefix IDs')
            ids = tokens.tolist()
        require(case['padding'] == dict(strategy='repeat_last_prefix_token', token_id=ids[-1],
                    row_interval=[rows, config.context_length]), 'case padding differs')
        expected_interventions = dict(query_only=[[query, rows]],
            other_queries=[p for p in ([0, query], [rows, config.context_length]) if p[0] < p[1]],
            all_queries=[[0, config.context_length]], interval_convention='half_open',
            visible_query_interval=[0, rows], captured_query_interval=[0, rows],
            future_padding_interval=[rows, config.context_length], future_padding_causal_null_at_selected_query=True)
        require(case['interventions'] == expected_interventions, 'case position scopes differ')
        interval = [case['head'] * config.head_dim, (case['head'] + 1) * config.head_dim]
        require(case['head_channel_interval'] == interval and case['projection'] == dict(
            weight=document['weights'][6 + 12 * case['block']], bias=document['weights'][7 + 12 * case['block']],
            input_feature_rows=interval, bias_unchanged=True), 'case physical head mapping differs')
        metadata = _json(case['metadata']['path'])
        require(metadata.get('complete') is True and metadata.get('probe_kind') == 'token_trace'
                and metadata.get('checkpoint_directory') == str(directory)
                and metadata.get('tokens_file') == case['prefix']['path']
                and metadata.get('token_ids') == ids and metadata.get('selected_row') == query
                and metadata.get('target_id') == case['target_id'], 'archived metadata/event binding differs')
        capture_names = dict(before='positioned' if case['block'] == 0 else f'blocks.{case["block"]-1}.after_mlp',
            context=f'blocks.{case["block"]}.attention', projected=f'blocks.{case["block"]}.attention_projected',
            after_attention=f'blocks.{case["block"]}.after_attention')
        require(set(case['captures']) == set(capture_names), 'archived capture set differs')
        for name, key in capture_names.items():
            capture = case['captures'][name]; item = capture['record']; archived = metadata['files'][key]
            require(capture['shape'] == [rows, config.d_model] and capture['dtype'] == 'bf16'
                    and capture['role'] == 'full_prefix' and provenance.get(item['path']) == item
                    and archived['shape'] == capture['shape'] and archived['dtype'] == 'bf16'
                    and archived['role'] == 'full_prefix'
                    and Path(item['path']) == Path(case['metadata']['path']).parent / archived['file'],
                    'archived capture identity/geometry differs')
            ledger.add(item)
            with mapped(item['path'], '<u2', (rows, config.d_model)) as array:
                require(not np.any((array & 0x7f80) == 0x7f80), 'nonfinite archived BF16 capture')
        calibration = case['calibration']
        require(calibration['full_padded_logits_shape'] == [1, config.padded_vocab_size]
                and calibration['require_clean_replay_byte_equal'] is True
                and calibration['require_all_query_context_zero_matches_weight_zero_bytes'] is True
                and calibration['calibration_executed'] is False, 'descriptor calibration contract differs')
        arms = historical.validate_arms(metadata['interventions'], config)
        expected_paths = dict(clean_logits=metadata['files']['logits']['file'],
            all_query_weight_zero_logits=arms[f'ablation.block{case["block"]}.head{case["head"]}']['logits_file'])
        for name, filename in expected_paths.items():
            item = calibration[name]
            require(provenance.get(item['path']) == item
                    and Path(item['path']) == Path(case['metadata']['path']).parent / filename,
                    'archived calibration record differs')
            ledger.add(item)
    return document, descriptor_record


def native_inventory(directory):
    require(directory.is_absolute() and directory.resolve() == directory and directory.is_dir()
            and not directory.is_symlink(), 'invalid native evidence directory')
    names = [f'{scope}_{label}' for scope in oracle.SCOPES for label, _ in DOSES]
    directories = {'calibration_all_queries_zero', *names}
    files = {'metadata.json', 'tokens.i32', 'padded_tokens.i32', 'original_before.bf16',
             'original_qkv.bf16', 'original_context.bf16', 'clean_logits.f32'}
    files |= {f'{name}/{file}' for name in directories for file in ('context.bf16', 'logits.f32')}
    actual_files, actual_directories = set(), set()
    for path in directory.rglob('*'):
        require(not path.is_symlink(), 'symlink in native evidence')
        if path.is_dir(): actual_directories.add(str(path.relative_to(directory)))
        else:
            require(path.is_file(), 'special file in native evidence')
            actual_files.add(str(path.relative_to(directory)))
    require(actual_files == files and actual_directories == directories, 'extra/missing native output inventory')
    return files


def validate_execution(path, document, descriptor_record, case, directory, ledger):
    execution_record = ledger.file(path); execution = _json(path)
    require(execution.get('format') == EXECUTION_FORMAT and type(execution.get('returncode')) is int
            and execution['returncode'] == 0 and type(execution.get('pid')) is int and execution['pid'] > 0,
            'missing authenticated successful native invocation')
    times = [datetime.fromisoformat(execution[key]) for key in ('started_utc', 'finished_utc')]
    require(all(t.tzinfo is not None for t in times) and times[1] >= times[0], 'invalid invocation times')
    require(execution.get('inputs_before') == execution.get('inputs_after'), 'native inputs changed during invocation')
    inputs = ledger.many(execution['inputs_before']); outputs = ledger.many(execution['outputs'])
    ledger.add(execution['log'])
    command = execution.get('command')
    require(isinstance(command, list) and len(command) == 10
            and all(isinstance(s, str) for s in command), 'native command must contain executable and nine flags')
    require(command[0] in inputs and Path(command[0]).resolve() == Path(command[0]), 'native binary is not hash-bound')
    flags = {}
    for item in command[1:]:
        require(item.startswith('--') and '=' in item, 'native command needs explicit --flag=value')
        name, value = item[2:].split('=', 1)
        require(name not in flags, 'duplicate native command flag')
        flags[name] = value
    expected = dict(checkpoint=document['checkpoint_directory'], tokens_file=case['prefix']['path'],
        output_dir=str(directory), block=str(case['block']), head=str(case['head']), query=str(case['query']),
        target_id=str(case['target_id']), expected_clean_logits=case['calibration']['clean_logits']['path'],
        expected_all_query_zero_logits=case['calibration']['all_query_weight_zero_logits']['path'])
    require(flags == expected, 'native command differs from fixed descriptor case')
    required = [descriptor_record, *document['provenance'], *document['implementation'], *document['weights']]
    if 'snapshot' in document['producer_source']: required.append(document['producer_source']['snapshot'])
    require(all(inputs.get(r['path']) == r for r in required), 'invocation missing descriptor/provenance/source inputs')
    require(set(outputs) == {str(directory / name) for name in native_inventory(directory)},
            'invocation output records do not match exact recursive inventory')
    require(not (set(inputs) & set(outputs)) and execution_record['path'] not in outputs
            and execution['log']['path'] not in outputs, 'native outputs overlap authenticated inputs/log')
    return execution, execution_record, outputs


def validate_metadata(metadata, execution, document, case, directory, config, ids):
    expected_arms = [dict(directory=f'{scope}_{label}', scope=scope, dose_label=label,
                         scale=scale, context_file='context.bf16', logits_file='logits.f32')
                     for scope in oracle.SCOPES for label, scale in DOSES]
    expected = dict(format='pluto-head-context-probe-v1', complete=True,
        binary=execution['command'][0], checkpoint=document['checkpoint_directory'], tokens_file=case['prefix']['path'],
        output_dir=str(directory), expected_clean_logits=case['calibration']['clean_logits']['path'],
        expected_all_query_zero_logits=case['calibration']['all_query_weight_zero_logits']['path'],
        block=case['block'], head=case['head'], sequence=0, query=case['query'], query_token_id=int(ids[case['query']]),
        target_id=case['target_id'], prefix_rows=len(ids), execution_rows=config.context_length,
        context_length=config.context_length, width=config.d_model, heads=config.n_heads,
        head_dimension=config.head_dim, blocks=config.n_layers, vocabulary=config.vocab_size,
        padded_vocabulary=config.padded_vocab_size, byte_order='little', padding='repeat_last_token',
        context_dtype='BF16/<u2', logits_dtype='<f4', context_shape=[config.context_length, config.d_model],
        qkv_shape=[config.context_length, config.d_model * 3], logits_shape=[config.context_length, config.padded_vocab_size],
        scopes=list(oracle.SCOPES), doses=[1, 0.5, 0, 1], calibration_directory='calibration_all_queries_zero',
        arm_count=12, arms=expected_arms, **{k: True for k in CHECKS},
        new_paired_model_result=False, goal_completion_claimed=False, command=execution['command'])
    require(metadata == expected, 'native metadata/geometry/checks differ from actual invocation')
    for key, value in expected.items():
        if type(value) in (int, bool): require(type(metadata[key]) is type(value), 'metadata scalar type differs')
    for key in ('context_shape', 'qkv_shape', 'logits_shape'):
        require(all(type(n) is int for n in metadata[key]), 'metadata shape type differs')
    require(all(type(n) in (int, float) for n in metadata['doses'])
            and all(type(a['scale']) in (int, float) for a in metadata['arms']), 'invalid metadata dose type')


def audit_logits(clean, changed, config, query, scope, scale, chunk_rows):
    """Streaming equivalent of the oracle's causal mask; no full-array copies."""
    positions = oracle.selected_rows(config.context_length, config.context_length, 0, query, scope)
    first_edit = int(positions[0]) if scale != 1 and len(positions) else config.context_length
    for start in range(0, config.context_length, chunk_rows):
        end = min(config.context_length, start + chunk_rows)
        a, b = clean[start:end], changed[start:end]
        require(np.isfinite(b).all(), 'nonfinite native full logits')
        unchanged = max(0, min(end, first_edit) - start)
        require(equal_bits(a[:unchanged], b[:unchanged]), 'logits changed outside causal descendants')
        require(equal_bits(a[:, config.vocab_size:], b[:, config.vocab_size:]), 'physical padding logits changed')
    return dict(unchanged_row_count=first_edit, potentially_affected_row_count=config.context_length - first_edit,
                all_physical_columns_checked=config.padded_vocab_size)


def equal_logit_files(a, b, config, chunk_rows):
    with mapped(a, '<f4', (config.context_length, config.padded_vocab_size)) as left, \
         mapped(b, '<f4', (config.context_length, config.padded_vocab_size)) as right:
        for start in range(0, config.context_length, chunk_rows):
            require(equal_bits(left[start:start + chunk_rows], right[start:start + chunk_rows]),
                    'repeated calibration all-query zero full logits differ')


def analyze_case(descriptor_path, case_index, directory, execution_path, *,
                 config=checkpoint.GPT2Config(), allow_synthetic=False, chunk_rows=16):
    """Return audited historical first-piece effects; never write or launch jobs."""
    require(type(chunk_rows) is int and 1 <= chunk_rows <= 64, 'invalid bounded row chunk size')
    ledger = Ledger()
    implementations = [ledger.file(p) for p in (__file__, oracle.__file__, checkpoint.__file__,
        historical.__file__, source_value_readout.__file__, head_position_cases.__file__)]
    document, descriptor_record = validate_descriptor(descriptor_path, ledger, config, allow_synthetic)
    require(type(case_index) is int and 0 <= case_index < len(document['cases']), 'invalid case index')
    case = document['cases'][case_index]; directory = Path(directory)
    execution, execution_record, outputs = validate_execution(execution_path, document, descriptor_record, case, directory, ledger)
    metadata = _json(directory / 'metadata.json')
    with mapped(case['prefix']['path'], '<i4', (case['visible_rows'],)) as original_ids:
        ids = np.array(original_ids)
    validate_metadata(metadata, execution, document, case, directory, config, ids)
    require((directory / 'tokens.i32').read_bytes() == ids.tobytes(), 'native prefix bytes differ')
    expected_padding = np.full(config.context_length, ids[-1], dtype='<i4'); expected_padding[:len(ids)] = ids
    require((directory / 'padded_tokens.i32').read_bytes() == expected_padding.tobytes(), 'native repeat-last padding differs')
    for name, width in (('before', config.d_model), ('qkv', config.d_model * 3), ('context', config.d_model)):
        with mapped(directory / f'original_{name}.bf16', '<u2', (config.context_length, width)) as native:
            require(not np.any((native & 0x7f80) == 0x7f80), 'nonfinite original native BF16 tensor')
            if name != 'qkv':
                with mapped(case['captures'][name]['record']['path'], '<u2', (case['visible_rows'], width)) as archived:
                    require(equal_bits(native[:case['visible_rows']], archived), 'native visible capture differs from archive')
    selected = {str(dose): {} for dose in (0, 0.5, 1)}
    validations = []
    shape = (config.context_length, config.padded_vocab_size)
    context_options = dict(context_length=config.context_length, sequence=0, query=case['query'],
                           head=case['head'], head_dim=config.head_dim)
    with mapped(directory / 'clean_logits.f32', '<f4', shape) as clean, \
         mapped(directory / 'original_context.bf16', '<u2', (config.context_length, config.d_model)) as original:
        for start in range(0, len(clean), chunk_rows):
            part = clean[start:start + chunk_rows]
            require(np.isfinite(part).all(), 'nonfinite clean full logits')
            require(np.all(part[:, config.vocab_size:] == -np.finfo(np.float32).max), 'clean physical padding is not -FLT_MAX')
        with mapped(case['calibration']['clean_logits']['path'], '<f4', (config.padded_vocab_size,)) as archived:
            require(equal_bits(clean[case['query']], archived), 'clean selected logits differ from archive')
        baseline_row = np.array(clean[case['query'], :config.vocab_size])
        calibration = directory / 'calibration_all_queries_zero'
        with mapped(calibration / 'context.bf16', '<u2', original.shape) as changed:
            oracle.audit_context(original, changed, scope='all_queries', scale=0, **context_options)
        with mapped(calibration / 'logits.f32', '<f4', shape) as zero:
            audit_logits(clean, zero, config, case['query'], 'all_queries', 0, chunk_rows)
            with mapped(case['calibration']['all_query_weight_zero_logits']['path'], '<f4', (config.padded_vocab_size,)) as archived:
                require(equal_bits(zero[case['query']], archived), 'zero selected logits differ from archived weight ablation')
        for scope in oracle.SCOPES:
            for label, dose in DOSES:
                arm = directory / f'{scope}_{label}'
                with mapped(arm / 'context.bf16', '<u2', original.shape) as changed:
                    context_validation = oracle.audit_context(original, changed, scope=scope, scale=dose, **context_options)
                with mapped(arm / 'logits.f32', '<f4', shape) as logits:
                    causal_validation = audit_logits(clean, logits, config, case['query'], scope, dose, chunk_rows)
                    if label != 'one_after': selected[str(dose)][scope] = np.array(logits[case['query'], :config.vocab_size])
                validations.append(dict(directory=arm.name, context=context_validation, causal_logits=causal_validation))
    equal_logit_files(directory / 'all_queries_zero/logits.f32',
                      directory / 'calibration_all_queries_zero/logits.f32', config, chunk_rows)
    effects = {dose: oracle.partition_effects(dict(clean=baseline_row, **rows), case['target_id'])
               for dose, rows in selected.items()}
    ledger.recheck()
    return dict(format=FORMAT, complete=True, historical_only=True, new_paired_model_result=False,
        goal_completion_claimed=False, full_word_probability_claimed=False, synthetic_fixture=config != checkpoint.GPT2Config(),
        descriptor=descriptor_record, case_index=case_index, case=case, execution=execution_record,
        metadata=outputs[str(directory / 'metadata.json')], clean_logits=outputs[str(directory / 'clean_logits.f32')],
        baseline=oracle.score_logits(baseline_row, case['target_id']), effects_by_dose=effects, arms=validations,
        records=list(ledger.records.values()), implementation=implementations,
        limitations=['Historical first-piece events only; no complete-word, lexical-selectivity or unique-storage claim.',
            'Native all-row model/attention/tail identity is a producer gate; independent checks cover exported bytes and causal isolation.',
            'Archived before/context captures cover visible prefix rows only; QKV/future-row provenance is not an archived numerical oracle.',
            'Head-context half scaling is not half-weight scaling; nonadditive interactions are not mediation fractions.'])


def analyze_all(descriptor_path, runs, *, config=checkpoint.GPT2Config(), allow_synthetic=False, chunk_rows=16):
    """Require the entire fixed screen and identical clean FULL tensors per event."""
    ledger = Ledger()
    document, descriptor = validate_descriptor(descriptor_path, ledger, config, allow_synthetic)
    require(isinstance(runs, list) and len(runs) == document['case_count']
            and all(type(r.get('case_index')) is int for r in runs)
            and {r['case_index'] for r in runs} == set(range(document['case_count'])), 'aggregate needs every case exactly once')
    results, baseline_groups = [], {}
    for run in sorted(runs, key=lambda r: r['case_index']):
        result = analyze_case(descriptor_path, run['case_index'], Path(run['directory']), run['execution'],
            config=config, allow_synthetic=allow_synthetic, chunk_rows=chunk_rows)
        case = result['case']
        key = (case['word'], case['generation_step'])
        signature = (case['prefix']['sha256'], case['query'], case['target_id'], result['clean_logits']['sha256'])
        require(key not in baseline_groups or baseline_groups[key] == signature,
                'clean full-logit baselines differ across heads for one historical event')
        baseline_groups[key] = signature
        results.append(result)
    # Per-case checks cannot authorize mutations after earlier cases completed.
    for result in results:
        for item in result['records']: ledger.add(item)
    ledger.recheck()
    return dict(format='pluto-head-position-aggregate-v1', complete=True, historical_only=True,
        new_paired_model_result=False, goal_completion_claimed=False, full_word_probability_claimed=False,
        synthetic_fixture=config != checkpoint.GPT2Config(), descriptor=descriptor,
        clean_full_logits_equal_across_heads=True, cases=results)
