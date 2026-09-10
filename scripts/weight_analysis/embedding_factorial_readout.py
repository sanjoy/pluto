"""Validate native embedding-factorial evidence and report causal contrasts.

All stored logical-vocabulary logits are inspected; JSON scores are assertions
to check, never the authority for likelihoods or rankings. The four cells are
AA (recipient), AJ (output-only), JA (input-only), JJ (joint embedding patch).
Effects use log probability: a positive effect supports the scored event.

Only this script's new JSON output is written. It never runs a model, edits a
checkpoint, chooses a patch from observed effects, or treats a changed weight
as proof of a unique word-storage location. The CLI uses production geometry;
an explicit small GPT2Config is available only to CPU fixture callers.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .checkpoint import GPT2Checkpoint, GPT2Config, sha256_file, tensor_manifest
from .paired_input_exposure import exposures
from . import paired_case_contract as case_contract


CELLS = ('AA', 'AJ', 'JA', 'JJ')
WORDS = ('Exeunt', 'Nuveth')
CHECKS = ('native_diagonals_byte_equal', 'nonembedding_weights_identical',
          'padding_weights_identical', 'device_weight_bytes_unchanged',
          'disk_weight_bytes_unchanged', 'input_bytes_unchanged')
KINDS = ('word', 'control', 'word_next_native', 'shared_piece')


def _require(value, message):
    if not value:
        raise ValueError(message)


def _int(value, minimum=0):
    return type(value) is int and value >= minimum


def _close(actual, expected, description):
    _require(type(actual) in (int, float) and math.isfinite(actual)
             and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9),
             f'inconsistent {description}')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def _json(path):
    path = Path(path)
    _require(not path.is_symlink() and path.is_file(), 'JSON must be a regular nonsymlink file')
    _require(path.stat().st_size <= 64 * 1024 * 1024, 'JSON exceeds 64 MiB')
    def nonfinite(value):
        raise ValueError('nonfinite JSON number: ' + value)
    return json.loads(path.read_text(), object_pairs_hook=_pairs, parse_constant=nonfinite)


def _record(path):
    path = Path(path)
    _require(not path.is_symlink() and path.is_file(), 'input must be a regular nonsymlink file')
    path = path.resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}


def _register(records, path, expected=None):
    actual = _record(path)
    if expected is not None:
        _require(actual == expected, 'input hash/size/path mismatch: ' + str(path))
    previous = records.setdefault(actual['path'], actual)
    _require(previous == actual, 'input changed during validation: ' + str(path))
    return actual


def _bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(
        a.view(np.uint8).reshape(-1), b.view(np.uint8).reshape(-1))


def score_logits(logits, target):
    """Independent FP64 full-vocabulary score; stable lower-ID tie handling."""
    row = np.asarray(logits)
    _require(row.ndim == 1 and len(row) > 0 and row.dtype == np.dtype('<f4')
             and _int(target) and target < len(row) and np.isfinite(row).all(),
             'invalid finite FP32 logit row or target')
    winner = int(np.argmax(row))
    maximum = float(row[winner])
    # Unlike the C++ sequential sum, NumPy may use pairwise summation. The
    # strict numerical check allows only small FP64 summation differences.
    denominator = float(np.exp(row.astype(np.float64) - maximum).sum(dtype=np.float64))
    nll = math.log(denominator) + (maximum - float(row[target]))
    rank = 1 + int(np.count_nonzero(row > row[target])) + int(
        np.count_nonzero(row[:target] == row[target]))
    return {'nll': nll, 'rank': rank, 'argmax': winner,
            'log_probability': -nll, 'probability': math.exp(-nll)}


def factorial_effects(values):
    """Contrasts of log probabilities (or log odds), not nonlinear fractions."""
    _require(set(values) == set(CELLS) and all(math.isfinite(v) for v in values.values()),
             'invalid factorial cells')
    return {'input': values['JA'] - values['AA'],
            'output': values['AJ'] - values['AA'],
            'interaction': math.fsum((values['JJ'], -values['JA'],
                                      -values['AJ'], values['AA'])),
            'joint': values['JJ'] - values['AA']}


def _validate_patch(patch_path, expected_rows, metadata, records, config):
    _register(records, patch_path)
    patch = _json(patch_path)
    specs = tensor_manifest(config)
    _require(patch.get('format') == 'pluto-paired-weight-patch-v1'
             and patch.get('complete') is True and patch.get('config') == asdict(config),
             'unsupported/incomplete patch provenance or geometry')
    _require(GPT2Config(**patch['config']) == config, 'invalid patch geometry field types')
    _require(isinstance(expected_rows, (list, tuple))
             and all(_int(i) and i < config.vocab_size for i in expected_rows)
             and len(set(expected_rows)) == len(expected_rows), 'invalid expected patch rows')
    expected_rows = sorted(expected_rows)
    selection = patch['selection']
    row_bytes = config.d_model * 4
    _require(selection.get('tensors') == []
             and selection.get('embedding_rows') == expected_rows
             and all(type(row) is int for row in selection['embedding_rows'])
             and selection.get('embedding_is_tied_to_lm_head') is True
             and selection.get('embedding_row_byte_ranges') == [
                 [i * row_bytes, (i + 1) * row_bytes] for i in expected_rows],
             'patch does not match the predeclared embedding-only selection')
    for check in ('all_weights_finite', 'sources_unchanged',
                  'selected_bytes_equal_replacement', 'unselected_bytes_equal_original',
                  'no_hardlinks'):
        _require(patch['validation'].get(check) is True, 'missing patch validation: ' + check)
    paths = {'A': Path(patch['sources']['original']['path']),
             'D': Path(patch['sources']['replacement']['path']),
             'J': Path(patch['output']['path'])}
    _require(paths['A'].resolve() == Path(metadata['recipient']).resolve()
             and paths['J'].resolve() == Path(metadata['patched']).resolve()
             and Path(patch_path).resolve() == paths['J'].resolve() / 'patch.json',
             'native recipient/patched paths disagree with patch provenance')
    checkpoints = {}
    weight_records = {}
    for side, path in paths.items():
        _require(path.is_dir() and not path.is_symlink(), 'checkpoint must be a real directory')
        allowed = {spec.filename for spec in specs}
        _require({p.name for p in path.iterdir()} <= allowed | {'patch.json'}
                 and all((path / name).is_file() for name in allowed),
                 'unexpected checkpoint layout')
        checkpoints[side] = GPT2Checkpoint(path, config, check_finite=True)
        declared = (patch['output']['weights_sha256'] if side == 'J' else
                    patch['sources']['original' if side == 'A' else 'replacement']['weights_sha256'])
        _require(set(declared) == allowed, 'incomplete checkpoint hash manifest')
        weight_records[side] = []
        for spec in specs:
            item = _register(records, path / spec.filename)
            _require(item['bytes'] == spec.nbytes and item['sha256'] == declared[spec.filename],
                     'checkpoint weight hash mismatch: ' + item['path'])
            weight_records[side].append(item)
    a, d, j = (checkpoints[side] for side in ('A', 'D', 'J'))
    for spec in specs:
        if spec.index:
            _require(_bits_equal(a[spec.name], j[spec.name]), 'nonembedding patch bytes differ')
        destination = (paths['J'] / spec.filename).stat()
        for source in ('A', 'D'):
            identity = (paths[source] / spec.filename).stat()
            _require((identity.st_dev, identity.st_ino) !=
                     (destination.st_dev, destination.st_ino), 'patch shares a source inode')
    a_embedding, d_embedding, j_embedding = (
        checkpoint['token_embedding.weight'] for checkpoint in (a, d, j))
    previous = 0
    for row in expected_rows:
        _require(_bits_equal(a_embedding[previous:row], j_embedding[previous:row]),
                 'unselected embedding bytes changed')
        _require(_bits_equal(d_embedding[row], j_embedding[row]),
                 'selected embedding row differs from donor')
        previous = row + 1
    _require(_bits_equal(a_embedding[previous:], j_embedding[previous:]),
             'unselected embedding/padding bytes changed')
    changed = np.flatnonzero(np.any(
        a_embedding.view('<u4') != j_embedding.view('<u4'), axis=1)).tolist()
    _require(metadata.get('changed_embedding_rows') == changed
             and all(type(row) is int for row in metadata['changed_embedding_rows']),
             'native changed-row list disagrees with actual checkpoint bytes')
    return {'selected_rows': expected_rows, 'actual_changed_rows': changed,
            'paths': {key: str(path.resolve()) for key, path in paths.items()},
            'weight_records': weight_records, 'patch': _record(patch_path)}


def _validate_cases(cases_path, metadata, case_kind, records, config):
    case_record = _register(records, cases_path)
    plan = _json(cases_path)
    amended = case_contract.validate_plan(plan,
        lambda path, expected: _register(records,path,expected),_json)
    count, length = plan['case_count'], plan['context_length']
    _require(_int(count, 1) and count == len(plan['cases'])
             and length == config.context_length and plan['vocab_size'] == config.vocab_size
             and plan.get('eos_token_id') == config.vocab_size - 1,
             'frozen case geometry or candidate identity mismatch')
    batch_record = _register(records, plan['packed_batch']['path'], plan['packed_batch'])
    _require(batch_record['bytes'] == 8 * count * length, 'frozen packed size mismatch')
    packed = np.memmap(batch_record['path'], dtype='<i4', mode='r', shape=(2, count, length))
    _require(not np.any(packed < 0) and not np.any(packed >= config.vocab_size)
             and np.array_equal(packed[0, :, 1:], packed[1, :, :-1]),
             'invalid frozen token IDs/next-token alignment')
    selected = []
    for index, case in enumerate(plan['cases']):
        kind = case.get('kind')
        _require(kind in KINDS and type(case['case_index']) is int and case['case_index'] == index
                 and case['split'] in ('training', 'test')
                 and case['prefix_domain'] in ('original', 'replacement', 'shared'),
                 'invalid frozen case identity')
        prefix_length = case['prefix']['length']
        ids = case['target_ids']
        expected_length = 4 if kind == 'word_next_native' else 3
        _require(_int(prefix_length, 1) and isinstance(ids, list)
                 and len(ids) == expected_length
                 and all(_int(i) and i < config.vocab_size for i in ids),
                 'invalid candidate length/IDs')
        rows = list(range(prefix_length - 1, prefix_length - 1 + len(ids)))
        _require(case['scored_rows'] == rows and rows[-1] < length
                 and packed[1, index, rows].tolist() == ids, 'frozen row/target mismatch')
        prefix = packed[0, index, :prefix_length].tobytes()
        _require(hashlib.sha256(prefix).hexdigest() == case['prefix']['token_ids_sha256'],
                 'frozen prefix hash mismatch')
        case_contract.metadata(case,amended=amended)
        if kind in ('word', 'word_next_native'):
            _require(case['prefix_domain'] != 'shared',
                     'invalid word identity/domain')
            source = case['target_source']
            pieces = [bytes.fromhex(s) for s in source['native_piece_bytes_hex']]
            _require(len(pieces) == len(ids) and b''.join(pieces).hex() == source['bytes_hex']
                     and b''.join(pieces[:3]) in (case['target'].encode(),
                                                b' ' + case['target'].encode()),
                     'word pieces do not spell the declared candidate')
            if kind == 'word_next_native':
                _require(case.get('word_token_count') == 3
                         and case['next_native_token']['id'] == ids[3]
                         and case['next_native_token']['bytes_hex'] == pieces[3].hex(),
                         'fourth target is not the declared exact next native token')
        else:
            _require(case['target'] == 'control_next_3' and case['prefix_domain'] == 'shared',
                     'invalid control identity')
            if kind == 'shared_piece':
                _require(case.get('piece_id') == ids[0], 'shared-piece target mismatch')
        if case_kind is None or case_kind == kind:
            selected.append(index)
    _require(case_kind is None or case_kind in KINDS, 'unknown case kind selection')
    _require(selected, 'empty selected suite')
    cases = [plan['cases'][index] for index in selected]
    rows_per_case = len(cases[0]['target_ids'])
    _require(all(len(case['target_ids']) == rows_per_case for case in cases),
             'mixed row counts: select the four-row word or three-row control subset explicitly')
    _require(metadata['case_count'] == len(cases) and type(metadata['case_count']) is int
             and metadata['rows_per_case'] == rows_per_case
             and type(metadata['rows_per_case']) is int, 'native case-count/row-count mismatch')
    native_batch = _register(records, metadata['batch'])
    _require(native_batch['bytes'] == 8 * len(cases) * length, 'native packed batch size mismatch')
    actual = np.memmap(native_batch['path'], dtype='<i4', mode='r',
                       shape=(2, len(cases), length))
    _require(_bits_equal(actual, np.asarray(packed[:, selected, :], dtype='<i4')),
             'native packed batch differs from the selected frozen cases')
    row_record = _register(records, metadata['rows'])
    expected_rows = np.asarray([case['scored_rows'] for case in cases], dtype='<i4')
    _require(row_record['bytes'] == expected_rows.nbytes, 'native row-file size mismatch')
    actual_rows = np.fromfile(row_record['path'], dtype='<i4').reshape(expected_rows.shape)
    _require(_bits_equal(actual_rows, expected_rows), 'native row file differs from frozen rows')
    return plan, cases, actual, case_record


def _execution_record(path, metadata, scores_directory, cases_record, patch,
                      records, output_records):
    if path is None:
        return {'verified': False, 'reason': 'No execution-time before/after hash record supplied.'}
    record = _register(records, path)
    execution = _json(path)
    _require(execution.get('format') == 'pluto-paired-probe-execution-v1'
             and _int(execution.get('pid'), 1)
             and type(execution.get('returncode')) is int and execution['returncode'] == 0,
             'incomplete/failed native execution record')
    times = []
    for key in ('started_utc', 'finished_utc'):
        value = datetime.fromisoformat(execution[key].replace('Z', '+00:00'))
        _require(value.tzinfo is not None and value.utcoffset() == timezone.utc.utcoffset(value),
                 'execution timestamps must be UTC')
        times.append(value)
    _require(times[0] <= times[1], 'execution finish precedes start')
    before, after = execution['inputs_before'], execution['inputs_after']
    _require(isinstance(before, list) and before == after, 'execution input hashes changed')
    expected_paths = {item['path'] for side in ('A', 'J')
                       for item in patch['weight_records'][side]}
    expected_paths.update((patch['patch']['path'], cases_record['path'],
                           str(Path(metadata['batch']).resolve()),
                           str(Path(metadata['rows']).resolve()),
                           str(Path(metadata['binary']).resolve())))
    supplied = set()
    for item in before:
        actual = _register(records, item['path'], item)
        _require(actual['path'] not in supplied, 'duplicate execution input record')
        supplied.add(actual['path'])
    _require(expected_paths <= supplied, 'execution record omits required frozen inputs')
    supplied_outputs = set()
    for item in execution['outputs']:
        actual = _register(records, item['path'], item)
        _require(actual['path'] not in supplied_outputs, 'duplicate execution output record')
        supplied_outputs.add(actual['path'])
    _require({item['path'] for item in output_records} <= supplied_outputs,
             'execution record omits native metadata or logits')
    command = execution['command']
    _require(isinstance(command, list) and command and all(type(s) is str for s in command)
             and Path(command[0]).resolve() == Path(metadata['binary']).resolve(),
             'execution command/binary mismatch')
    flags = {}
    index = 1
    while index < len(command):
        part = command[index]
        _require(part.startswith('--'), 'unexpected positional execution argument')
        if '=' in part:
            key, value = part[2:].split('=', 1)
        else:
            _require(index + 1 < len(command), 'missing execution argument value')
            key, value = part[2:], command[index + 1]
            index += 1
        _require(key not in flags, 'duplicate execution flag')
        flags[key] = value
        index += 1
    _require(set(flags) <= {'recipient', 'patched', 'batch', 'rows', 'output_dir',
                            'rows_per_case', 'batch_sequences'}, 'unknown execution flags')
    for key in ('recipient', 'patched', 'batch', 'rows'):
        _require(key in flags and Path(flags[key]).resolve() == Path(metadata[key]).resolve(),
                 'execution argument path mismatch: ' + key)
    _require('output_dir' in flags and Path(flags['output_dir']).resolve() == scores_directory.resolve(),
             'execution output directory mismatch')
    _require(int(flags.get('rows_per_case', '3')) == metadata['rows_per_case']
             and int(flags.get('batch_sequences', '1')) == metadata['batch_sequences'],
             'execution batching/scored-row arguments mismatch')
    return {'verified': True, 'record': record, 'pid': execution['pid'],
            'started_utc': execution['started_utc'], 'finished_utc': execution['finished_utc'],
            'required_input_count': len(expected_paths), 'required_output_count': len(output_records)}


def _summary(items, positions):
    """Mean scores on the already deduplicated event set, preserving both means."""
    cells = {}
    for cell in CELLS:
        nlls = [math.fsum(item['cells'][cell]['token_nll'][p] for p in positions) for item in items]
        mean = math.fsum(nlls) / len(nlls)
        cells[cell] = {'mean_nll': mean, 'mean_log_probability': -mean,
                       'geometric_mean_probability': math.exp(-mean),
                       'arithmetic_mean_probability': math.fsum(math.exp(-n) for n in nlls) / len(nlls),
                       'all_selected_argmax_fraction': sum(all(
                           item['cells'][cell]['argmax_ids'][p] == item['target_ids'][p]
                           for p in positions) for item in items) / len(items)}
    return {'unique_event_count': len(items), 'cells': cells,
            'log_probability_effects': factorial_effects({
                cell: value['mean_log_probability'] for cell, value in cells.items()})}


def _dedup(items, key):
    result = {}
    for item in items:
        result.setdefault(key(item), item)
    return list(result.values())


def _aggregates(per_case):
    groups = defaultdict(list)
    for item in per_case:
        for domain in ('deduplicated_all', item['prefix_domain']):
            groups[(item['kind'], item['split'], domain, item['target'],
                    item.get('piece_id', -1))].append(item)
    reports = []
    for (kind, split, domain, target, piece), items in sorted(groups.items()):
        base = lambda item: (item['predictions'][0]['causal_prefix_sha256'],
                              item['predictions'][0]['causal_prefix_length'])
        events = _dedup(items, lambda item: (base(item), tuple(item['target_ids'])))
        length = len(events[0]['target_ids'])
        report = {'kind': kind, 'split': split, 'prefix_domain': domain, 'target': target,
                  'spelling_variant':items[0].get('spelling_variant'),
                  'piece_id': None if piece < 0 else piece, 'frozen_case_count': len(items),
                  'selected_sequence': _summary(events, list(range(length))), 'tokens': []}
        aliases = defaultdict(list)
        for item in items:
            aliases[(base(item), tuple(item['target_ids']))].append(item['source_case_index'])
        report['sequence_aliases'] = [
            {'representative_source_case_index': values[0], 'source_case_indices': values}
            for values in aliases.values()]
        for position in range(length):
            unique = _dedup(items, lambda item: (
                item['predictions'][position]['causal_prefix_sha256'],
                item['predictions'][position]['causal_prefix_length'], item['target_ids'][position]))
            report['tokens'].append({'target_position': position,
                                      **_summary(unique, [position])})
        if kind in ('word', 'word_next_native'):
            words = _dedup(items, lambda item: (base(item), tuple(item['target_ids'][:3])))
            report['word_three'] = _summary(words, [0, 1, 2])
            if kind == 'word_next_native':
                report['exact_next_native_token'] = report['tokens'][3]
                report['exact_next_native_token']['interpretation'] = (
                    'Conditional probability of the one supplied next native token, not any delimiter.')
        reports.append(report)
    return reports


def _odds(per_case):
    groups = defaultdict(list)
    for item in per_case:
        if item['kind'] not in ('word', 'word_next_native'):
            continue
        for domain in ('deduplicated_all', item['prefix_domain']):
            groups[(item['kind'], item['split'], domain,item.get('spelling_variant','title'))].append(item)
    results = []
    for (kind, split, domain,variant), items in sorted(groups.items()):
        candidates=case_contract.PAIRS[variant]
        words_in_order=tuple(candidates[role] for role in ('original','replacement'))
        contexts = defaultdict(lambda: defaultdict(list))
        for item in items:
            first = item['predictions'][0]
            key = (first['causal_prefix_sha256'], first['causal_prefix_length'],
                   item['word_has_leading_space'])
            contexts[key][item['target']].append(item)
        pairs = []
        for key, words in sorted(contexts.items()):
            _require(set(words) == set(words_in_order), 'missing paired word candidate in one causal context')
            pair = {}
            for word in words_in_order:
                unique = _dedup(words[word], lambda item: tuple(item['target_ids'][:3]))
                _require(len(unique) == 1, 'ambiguous candidate tokenization within a matched context')
                pair[word] = unique[0]
            original,replacement=words_in_order
            cell_odds = {cell: math.fsum(pair[original]['cells'][cell]['token_nll'][:3])
                         - math.fsum(pair[replacement]['cells'][cell]['token_nll'][:3]) for cell in CELLS}
            token_odds = {cell: [pair[original]['cells'][cell]['token_nll'][p]
                                 - pair[replacement]['cells'][cell]['token_nll'][p]
                                 for p in range(3)] for cell in CELLS}
            pairs.append({'causal_prefix_sha256': key[0], 'causal_prefix_length': key[1],
                           'word_has_leading_space': key[2], 'cell_log_odds': cell_odds,
                           'token_log_odds': token_odds,
                           'log_odds_effects': factorial_effects(cell_odds),
                           'aliases': {word: [item['source_case_index'] for item in words[word]]
                                       for word in words_in_order}})
        means = {cell: math.fsum(p['cell_log_odds'][cell] for p in pairs) / len(pairs)
                 for cell in CELLS}
        results.append({'kind': kind, 'split': split, 'prefix_domain': domain,
                        'spelling_variant':variant,'candidate_pair':candidates,
                        'unique_paired_context_count': len(pairs), 'contexts': pairs,
                        'mean_log_odds': means, 'log_odds_effects': factorial_effects(means),
                        'mean_token_log_odds': {cell: [math.fsum(
                            p['token_log_odds'][cell][i] for p in pairs) / len(pairs)
                            for i in range(3)] for cell in CELLS}})
    return results


def _exposure_groups(per_case):
    groups = defaultdict(dict)
    for item in per_case:
        for position, prediction in enumerate(item['predictions']):
            group = (item['kind'], item['split'], position,item.get('spelling_variant') or '')
            key = (prediction['causal_prefix_length'], prediction['causal_prefix_sha256'])
            groups[group][key] = prediction['input_path_must_be_unchanged']
    return [{'kind': kind, 'split': split, 'target_position': position,
             'spelling_variant':variant or None,
             'unique_causal_prefix_count': len(values),
             'unexposed_causal_prefix_count': sum(values.values()),
             'unexposed_full_logit_invariance_verified': True}
            for (kind, split, position,variant), values in sorted(groups.items())]


def analyze(cases_path, scores_directory, patch_path, output, *, expected_rows,
            case_kind=None, execution_record=None, config=GPT2Config()):
    """Validate frozen inputs, all four logit dumps, contrasts, and provenance."""
    output, scores_directory = Path(output), Path(scores_directory)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    _require(not scores_directory.is_symlink() and scores_directory.is_dir(),
             'native scores must be a real directory')
    records = {}
    metadata_path = scores_directory / 'metadata.json'
    metadata_record = _register(records, metadata_path)
    metadata = _json(metadata_path)
    _require(metadata.get('format') == 'pluto-embedding-factorial-v1'
             and metadata.get('complete') is True
             and type(metadata.get('temperature')) in (int, float) and metadata['temperature'] == 1
             and metadata.get('context_length') == config.context_length
             and metadata.get('vocabulary') == config.vocab_size
             and metadata.get('logits_dtype') == '<f4'
             and _int(metadata.get('batch_sequences'), 1)
             and metadata.get('normalization') == 'recipient final LayerNorm fixed in all four cells'
             and metadata.get('score_definition') ==
                 'CPU FP64 log-sum-exp over all logical native FP32 logits'
             and set(metadata.get('cells', {})) == set(CELLS), 'incompatible native factorial metadata')
    for check in CHECKS:
        _require(metadata.get('checks', {}).get(check) is True, 'failed/missing native check: ' + check)
    _require(metadata['checks'].get('native_diagonal_verification_scope') ==
             'selected_rows_full_padded_vocabulary', 'incorrect native diagonal verification scope')
    _register(records, metadata['binary'])
    patch = _validate_patch(patch_path, expected_rows, metadata, records, config)
    for directory in patch['paths'].values():
        parent = Path(directory)
        _require(output.resolve() != parent and parent not in output.resolve().parents,
                 'readout output must not be inside a source checkpoint')
    plan, cases, packed, case_record = _validate_cases(
        cases_path, metadata, case_kind, records, config)
    count, length = len(cases), len(cases[0]['target_ids'])
    shape = (count, length, config.vocab_size)
    _require(metadata.get('logits_shape') == list(shape), 'native logit shape mismatch')
    outputs, arrays = [metadata_record], {}
    for cell in CELLS:
        filename = cell + '.logits.f32.bin'
        item = metadata['cells'][cell]
        _require(item.get('logits_file') == filename and len(item['cases']) == count,
                 'native cell filename/case count mismatch')
        record = _register(records, scores_directory / filename)
        _require(record['bytes'] == math.prod(shape) * 4, 'native logit file size mismatch')
        outputs.append(record)
        arrays[cell] = np.memmap(record['path'], dtype='<f4', mode='r', shape=shape)
    provenance = _execution_record(execution_record, metadata, scores_directory,
                                    case_record, patch, records, outputs)
    seen_prefixes, per_case = {}, []
    invariance_count = 0
    max_nll_error = 0.0
    unselected_columns = np.ones(config.vocab_size, dtype=bool)
    unselected_columns[patch['actual_changed_rows']] = False
    for index, case in enumerate(cases):
        rows = case['scored_rows']
        changed = patch['actual_changed_rows']
        if changed:
            prediction_exposure = exposures(packed[0, index], rows, changed)
        else:
            prediction_exposure = [
                {'prediction_row': row, 'causal_prefix_length': row + 1,
                 'causal_prefix_sha256': hashlib.sha256(packed[0, index, :row + 1].tobytes()).hexdigest(),
                 'patched_input_ids': [], 'input_path_must_be_unchanged': True} for row in rows]
        item = {key: case[key] for key in ('kind', 'split', 'context_id', 'prefix_domain',
                                          'target', 'target_ids')}
        item.update(native_case_index=index, source_case_index=case['case_index'],
                    predictions=prediction_exposure, cells={})
        item.update(case_contract.metadata(case,amended=plan['format'] in case_contract.AMENDED))
        if 'piece_id' in case:
            item['piece_id'] = case['piece_id']
        if case['kind'] in ('word', 'word_next_native'):
            item['word_has_leading_space'] = bytes.fromhex(
                case['target_source']['native_piece_bytes_hex'][0]).startswith(b' ')
        for cell in CELLS:
            declared = metadata['cells'][cell]['cases'][index]
            _require(type(declared['case_index']) is int and declared['case_index'] == index
                     and len(declared['tokens']) == length, 'native case index or token count mismatch')
            measured = []
            for position, target in enumerate(case['target_ids']):
                score = score_logits(arrays[cell][index, position], target)
                reported = declared['tokens'][position]
                _require(type(reported['row']) is int and reported['row'] == rows[position]
                         and type(reported['target']) is int and reported['target'] == target
                         and type(reported['rank']) is int and reported['rank'] == score['rank']
                         and type(reported['argmax']) is int and reported['argmax'] == score['argmax'],
                         'native target/rank/argmax disagrees with full logits')
                _close(reported['nll'], score['nll'], 'native token NLL from full logits')
                max_nll_error = max(max_nll_error, abs(reported['nll'] - score['nll']))
                measured.append(score)
            nlls = [value['nll'] for value in measured]
            _close(declared['selected_rows_nll_sum'], math.fsum(nlls), 'native selected-row NLL sum')
            item['cells'][cell] = {'token_nll': nlls, 'token_probability': [m['probability'] for m in measured],
                                    'token_log_probability': [-n for n in nlls],
                                    'argmax_ids': [m['argmax'] for m in measured],
                                    'target_ranks': [m['rank'] for m in measured],
                                    'sequence_nll': math.fsum(nlls),
                                    'sequence_log_probability': -math.fsum(nlls),
                                    'sequence_probability': math.exp(-math.fsum(nlls))}
            if case['kind'] in ('word', 'word_next_native'):
                word_nll = math.fsum(nlls[:3])
                item['cells'][cell].update(word_three_nll=word_nll,
                                           word_three_log_probability=-word_nll,
                                           word_three_probability=math.exp(-word_nll))
                if case['kind'] == 'word_next_native':
                    item['cells'][cell].update(
                        exact_next_native_token_nll=nlls[3],
                        exact_next_native_token_conditional_probability=math.exp(-nlls[3]),
                        word_and_exact_next_native_token_probability=math.exp(-math.fsum(nlls)))
        for position, exposure in enumerate(prediction_exposure):
            # Use actual prefix bytes for dedup validation, not declared hashes
            # or full padded inputs containing causally irrelevant future IDs.
            prefix = packed[0, index, :rows[position] + 1].tobytes()
            key = (rows[position] + 1, prefix)
            previous = seen_prefixes.setdefault(key, (index, position))
            for cell in CELLS:
                _require(_bits_equal(arrays[cell][index, position],
                                      arrays[cell][previous[0], previous[1]]),
                         'identical causal prefixes have different full logits')
            if exposure['input_path_must_be_unchanged']:
                for left, right in (('AA', 'JA'), ('AJ', 'JJ')):
                    _require(_bits_equal(arrays[left][index, position], arrays[right][index, position]),
                             'unexposed input path violated full-logit invariance')
                invariance_count += 1
            # Untying the output dictionary changes only its selected rows.
            # All other RAW logits are invariant even when normalization of
            # their probabilities changes through the full-vocabulary sum.
            for left, right in (('AA', 'AJ'), ('JA', 'JJ')):
                _require(_bits_equal(arrays[left][index, position, unselected_columns],
                                      arrays[right][index, position, unselected_columns]),
                         'output-only patch changed an unselected raw-logit column')
        item['log_probability_effects'] = factorial_effects({
            cell: item['cells'][cell]['sequence_log_probability'] for cell in CELLS})
        item['token_log_probability_effects'] = [factorial_effects({
            cell: item['cells'][cell]['token_log_probability'][p] for cell in CELLS}) for p in range(length)]
        if case['kind'] in ('word', 'word_next_native'):
            item['word_three_log_probability_effects'] = factorial_effects({
                cell: item['cells'][cell]['word_three_log_probability'] for cell in CELLS})
        per_case.append(item)
    result = {'format': 'pluto-embedding-factorial-readout-v1', 'cases': case_record,
              'case_kind_selection': case_kind, 'native_metadata': metadata_record,
              'patch': patch, 'execution_provenance': provenance,
              'definitions': {
                  'cells': 'AA=recipient; AJ=output-only; JA=input-only; JJ=joint. Final LN is recipient and fixed.',
                  'effects': 'Log probability: input=JA-AA, output=AJ-AA, interaction=JJ-JA-AJ+AA, joint=JJ-AA. Positive supports event.',
                  'odds': 'Within each spelling variant, log P(replacement triple|prefix)/P(original triple|prefix) = NLL(original)-NLL(replacement). Title and lowercase are never pooled.',
                  'dedup': 'Sequence events use causal base-prefix bytes/length plus target IDs; individual tokens use their actual causal prefix and target. Domain aliases count once in deduplicated_all; domain-specific strata overlap.',
                  'next_native': 'Fourth target is the one exact supplied following native token. It is not any delimiter or any way to end the word.',
                  'aggregation': 'Equal weight per distinct event within each stratum, not corpus-frequency weighted. Word_three deduplicates triples independently of following-token variants.'},
              'checks': {'all_logical_logits_recomputed': True, 'maximum_native_nll_discrepancy': max_nll_error,
                         'native_diagonal_verification_scope': 'selected_rows_full_padded_vocabulary',
                         'identical_causal_prefix_logits_byte_equal': True,
                         'unique_causal_prefix_count': len(seen_prefixes),
                         'unexposed_predictions_checked_including_aliases': invariance_count,
                         'unexposed_full_logits_byte_equal': True,
                         'unselected_output_columns_byte_equal': True,
                         'selected_and_unselected_checkpoint_bytes_verified': True},
              'groups': _aggregates(per_case), 'word_odds': _odds(per_case),
              'exposure_groups': _exposure_groups(per_case), 'per_case': per_case,
              'limitations': [
                  'Complete-token-sequence probabilities do not sum alternative tokenizations of the spelling.',
                  'High preference transfer can result from damage to both words; absolute probabilities remain necessary.',
                  'Exposure permits an input-side effect but does not establish one. No exposure guarantees unchanged input computation only for the verified embedding-only patch.',
                  'Native diagonal equality and no-mutation claims are checked native assertions; CPU readout does not rerun GPU kernels.',
                  'Hash records establish recorded input/output identities, not independent attestation that the named binary executed.',
                  'Without execution_provenance.verified, current hashes cannot establish execution-time binary/input identity.',
                  'Context dependence and targeted functional contribution do not identify a unique word-storage location.']}
    for name in ('embedding_factorial_readout.py', 'checkpoint.py', 'paired_input_exposure.py','paired_case_contract.py'):
        _register(records, Path(__file__).with_name(name))
    # Detect changes after computation and before exclusive result publication.
    for record in records.values():
        _require(_record(record['path']) == record, 'input changed during readout')
    result['files'] = [records[path] for path in sorted(records)]
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cases', 'scores', 'patch', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-row', type=int, action='append', default=[])
    parser.add_argument('--case-kind', choices=KINDS)
    parser.add_argument('--execution-record', type=Path)
    args = parser.parse_args(argv)
    analyze(args.cases, args.scores, args.patch, args.output,
            expected_rows=args.expected_row, case_kind=args.case_kind,
            execution_record=args.execution_record)


if __name__ == '__main__':
    main()
