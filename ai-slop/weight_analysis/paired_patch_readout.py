"""CPU-only readout of four frozen paired-word score summaries.

Run with ``python -m weight_analysis.paired_patch_readout --help``.
Only the four JSON inputs are read; no model, checkpoint, or score dump is
loaded. Patch labels describe the supplied experiments, not an independently
verified intervention. Identical prefix/candidate inputs count once per split;
all original case values and prefix-domain aliases remain in the output.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


MODELS = ('original', 'replacement', 'original_with_donor_rows',
          'replacement_with_donor_rows')
WORDS = ('Exeunt', 'Nuveth')
SPLITS = ('training', 'test')
DOMAINS = ('original', 'replacement')
SCORE_FIELDS = ('token_nll', 'sequence_nll', 'sequence_log_probability',
                'sequence_probability', 'token_probabilities',
                'teacher_forced_argmax_ids', 'argmax_matches', 'all_three_argmax_match')
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_CASES = 100000


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _int(value, minimum=0):
    return type(value) is int and value >= minimum


def _finite_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)
    elif isinstance(value, float):
        _require(math.isfinite(value), 'nonfinite JSON number')


def _number(value, *, nonnegative=False):
    _require(type(value) in (int, float) and math.isfinite(value)
             and (not nonnegative or value >= 0), 'invalid finite numeric value')
    return value


def _same_number(actual, expected):
    _number(actual)
    _require(math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-10),
             'inconsistent derived score or group aggregate')


def _hash(value):
    _require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value),
             'invalid SHA-256 hash')


def _record(record):
    _require(isinstance(record, dict) and isinstance(record['path'], str)
             and record['path'] and _int(record['bytes']), 'invalid input record')
    _hash(record['sha256'])


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _load(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    _require(len(raw) <= MAX_INPUT_BYTES, 'summary exceeds 64 MiB read limit')
    value = json.loads(raw, object_pairs_hook=_unique_keys)
    _finite_tree(value)
    return value, {'path': str(path), 'bytes': len(raw),
                   'sha256': hashlib.sha256(raw).hexdigest()}


def _identity(case):
    return {key: value for key, value in case.items() if key not in SCORE_FIELDS}


def _scores(case):
    return {key: case[key] for key in SCORE_FIELDS}


def _input_key(case):
    return (case['prefix']['token_ids_sha256'], case['prefix']['length'],
            tuple(case['target_ids']))


def _validate_case(case, index, length):
    _require(case['case_index'] == index and type(case['case_index']) is int,
             'case indices must be ordered and contiguous')
    _require(case['kind'] in ('word', 'control') and case['split'] in SPLITS
             and isinstance(case['context_id'], str) and case['context_id'],
             'invalid case kind, split, or context ID')
    if case['kind'] == 'word':
        _require(case['prefix_domain'] in DOMAINS and case['target'] in WORDS
                 and _int(case['occurrence_index'])
                 and case['target_source_domain'] == DOMAINS[WORDS.index(case['target'])],
                 'invalid word candidate or prefix domain')
    else:
        _require(case['prefix_domain'] == 'shared' and case['target'] == 'control_next_3'
                 and case['occurrence_index'] is None
                 and case['target_source_domain'] == 'original', 'invalid control case')
    prefix = case['prefix']
    _require(_int(prefix['length'], 1) and prefix['length'] <= length - 2
             and _int(prefix['token_start']) and _int(prefix['token_end'])
             and prefix['token_end'] - prefix['token_start'] == prefix['length']
             and _int(prefix['byte_start']) and _int(prefix['byte_end'])
             and prefix['byte_end'] > prefix['byte_start'], 'invalid prefix coordinates')
    _hash(prefix['token_ids_sha256'])
    rows = [prefix['length'] - 1, prefix['length'], prefix['length'] + 1]
    _require(case['scored_rows'] == rows and all(type(row) is int for row in case['scored_rows']),
             'misaligned scored rows')
    for field in ('target_ids', 'teacher_forced_argmax_ids'):
        ids = case[field]
        _require(isinstance(ids, list) and len(ids) == 3
                 and all(_int(token) and token < 50257 for token in ids),
                 f'invalid {field}')
    source = case['target_source']
    _require(source['token_start'] == prefix['token_end']
             and source['token_end'] == source['token_start'] + 3
             and _int(source['byte_start']) and _int(source['byte_end'])
             and source['byte_end'] > source['byte_start'], 'invalid target coordinates')
    raw = bytes.fromhex(source['bytes_hex'])
    pieces = source['native_piece_bytes_hex']
    _require(len(pieces) == 3 and raw == b''.join(bytes.fromhex(piece) for piece in pieces)
             and len(raw) == source['byte_end'] - source['byte_start']
             and source['decoded_text'] == raw.decode('utf-8', errors='replace'),
             'invalid target bytes')
    if case['kind'] == 'word':
        word = case['target'].encode()
        _require(raw in (word, b' ' + word), 'target bytes disagree with candidate word')
    token_nll = case['token_nll']
    _require(isinstance(token_nll, list) and len(token_nll) == 3, 'invalid token NLLs')
    for value in token_nll:
        _number(value, nonnegative=True)
    total = math.fsum(token_nll)
    _number(case['sequence_nll'], nonnegative=True)
    _same_number(case['sequence_nll'], total)
    _same_number(case['sequence_log_probability'], -total)
    _require(0 <= _number(case['sequence_probability']) <= 1, 'invalid sequence probability')
    _same_number(case['sequence_probability'], math.exp(-total))
    _require(isinstance(case['token_probabilities'], list)
             and len(case['token_probabilities']) == 3, 'invalid token probabilities')
    for nll, probability in zip(token_nll, case['token_probabilities']):
        _require(0 <= _number(probability) <= 1, 'invalid token probability')
        _same_number(probability, math.exp(-nll))
    matches = [a == b for a, b in zip(case['target_ids'], case['teacher_forced_argmax_ids'])]
    _require(case['argmax_matches'] == matches
             and all(type(item) is bool for item in case['argmax_matches'])
             and type(case['all_three_argmax_match']) is bool
             and case['all_three_argmax_match'] == all(matches), 'invalid argmax matches')


def _validate_summary(summary):
    _require(summary['format'] == 'pluto-paired-word-score-summary-v1',
             'unsupported word-score summary format')
    count, length = summary['case_count'], summary['context_length']
    _require(_int(count, 1) and count <= MAX_CASES and _int(length, 3) and length <= 1024
             and isinstance(summary['per_case'], list) and len(summary['per_case']) == count,
             'invalid summary case count or context length')
    _record(summary['cases'])
    for name in ('metadata', 'losses', 'argmax'):
        _record(summary['scores'][name])
    for name in ('losses', 'argmax'):
        _require(summary['scores'][name]['bytes'] == count * length * 4,
                 'score byte count disagrees with summary dimensions')
    meta = summary['probe_metadata']
    _require(meta['kind'] == 'paired_loss_probe' and meta['complete'] is True
             and type(meta['temperature']) in (int, float) and meta['temperature'] == 1
             and meta['byte_order'] == 'little' and meta['loss_dtype'] == '<f4'
             and meta['argmax_dtype'] == '<i4' and meta['loss_file'] == 'losses.f32.bin'
             and meta['argmax_file'] == 'argmax.i32.bin'
             and meta['case_count'] == count and meta['passage_count'] == count
             and meta['context_length'] == length and meta['output_shape'] == [count, length]
             and isinstance(meta['batch_file'], str) and meta['batch_file']
             and meta.get('batch_bytes', count * length * 8) == count * length * 8,
             'incompatible paired-loss probe metadata')
    groups = defaultdict(list)
    for index, case in enumerate(summary['per_case']):
        _validate_case(case, index, length)
        groups[(case['kind'], case['split'], case['prefix_domain'], case['target'])].append(case)
    seen = set()
    _require(isinstance(summary['groups'], list), 'invalid summary groups')
    for group in summary['groups']:
        key = tuple(group[field] for field in ('kind', 'split', 'prefix_domain', 'target'))
        _require(key in groups and key not in seen, 'missing, duplicate, or unknown summary group')
        seen.add(key)
        items = groups[key]
        _require(type(group['case_count']) is int and group['case_count'] == len(items),
                 'group case count mismatch')
        mean = math.fsum(item['sequence_nll'] for item in items) / len(items)
        expected = {'mean_sequence_nll': mean, 'mean_token_nll': mean / 3,
                    'geometric_mean_sequence_probability': math.exp(-mean),
                    'arithmetic_mean_sequence_probability': math.fsum(
                        item['sequence_probability'] for item in items) / len(items),
                    'teacher_forced_argmax_token_accuracy': sum(
                        sum(item['argmax_matches']) for item in items) / (3 * len(items)),
                    'all_three_argmax_match_fraction': sum(
                        item['all_three_argmax_match'] for item in items) / len(items)}
        for field, value in expected.items():
            _same_number(group[field], value)
    _require(seen == set(groups), 'missing summary group')


def _metrics(original_case, replacement_case):
    a, b = original_case, replacement_case
    return {'word_nll': dict(zip(WORDS, (a['sequence_nll'], b['sequence_nll']))),
            'word_token_nll': dict(zip(WORDS, (a['token_nll'], b['token_nll']))),
            'nuveth_minus_exeunt_log_odds': a['sequence_nll'] - b['sequence_nll'],
            'token_log_odds': [x - y for x, y in zip(a['token_nll'], b['token_nll'])]}


def _mean(values):
    return math.fsum(values) / len(values)


def _average_metrics(metrics):
    return {'word_nll': {word: _mean([m['word_nll'][word] for m in metrics]) for word in WORDS},
            'word_token_nll': {word: [_mean([m['word_token_nll'][word][i] for m in metrics])
                                      for i in range(3)] for word in WORDS},
            'nuveth_minus_exeunt_log_odds': _mean(
                [m['nuveth_minus_exeunt_log_odds'] for m in metrics]),
            'token_log_odds': [_mean([m['token_log_odds'][i] for m in metrics]) for i in range(3)]}


def _fraction(change, donor_shift, floor):
    return change / donor_shift if abs(donor_shift) > floor else None


def _effects(metrics, floor):
    result = {}
    for patched, recipient, donor in ((MODELS[2], MODELS[0], MODELS[1]),
                                      (MODELS[3], MODELS[1], MODELS[0])):
        p, r, d = (metrics[name] for name in (patched, recipient, donor))
        shift = d['nuveth_minus_exeunt_log_odds'] - r['nuveth_minus_exeunt_log_odds']
        change = p['nuveth_minus_exeunt_log_odds'] - r['nuveth_minus_exeunt_log_odds']
        result[patched] = {
            'recipient': recipient, 'donor': donor,
            'log_odds_patch_change': change, 'full_donor_log_odds_shift': shift,
            'fraction_of_full_donor_shift': _fraction(change, shift, floor),
            'word_nll_changes': {word: p['word_nll'][word] - r['word_nll'][word] for word in WORDS},
            'word_token_nll_changes': {
                word: [x - y for x, y in zip(p['word_token_nll'][word], r['word_token_nll'][word])]
                for word in WORDS},
            'token_log_odds_patch_changes': [x - y for x, y in zip(p['token_log_odds'], r['token_log_odds'])],
            'token_full_donor_log_odds_shifts': [x - y for x, y in zip(d['token_log_odds'], r['token_log_odds'])],
        }
    return result


def _case_values(indices, summaries):
    return [{'case': _identity(summaries['original']['per_case'][i]),
             'scores': {model: _scores(summary['per_case'][i])
                        for model, summary in summaries.items()}} for i in sorted(indices)]


def _key_hash(key):
    return hashlib.sha256(json.dumps(key, separators=(',', ':')).encode()).hexdigest()


def _split_readout(split, summaries, absolute_tolerance, relative_tolerance, floor):
    cases = [case for case in summaries['original']['per_case'] if case['split'] == split]
    words, controls, duplicate_inputs = defaultdict(dict), defaultdict(list), defaultdict(list)
    for case in cases:
        duplicate_inputs[_input_key(case)].append(case['case_index'])
        if case['kind'] == 'word':
            key = (case['context_id'], case['prefix_domain'])
            _require(case['target'] not in words[key], 'duplicate word case for context/domain')
            words[key][case['target']] = case
        else:
            controls[_input_key(case)].append(case['case_index'])
    _require(words, f'no word contexts in {split}')
    duplicate_groups = 0
    maximum_difference = 0.0
    for indices in duplicate_inputs.values():
        if len(indices) < 2:
            continue
        duplicate_groups += 1
        for summary in summaries.values():
            observed = [summary['per_case'][i] for i in indices]
            _require(all(case['teacher_forced_argmax_ids'] == observed[0]['teacher_forced_argmax_ids']
                         for case in observed[1:]), 'identical-input argmax IDs disagree')
            for token in range(3):
                low = min(case['token_nll'][token] for case in observed)
                high = max(case['token_nll'][token] for case in observed)
                maximum_difference = max(maximum_difference, high - low)
                _require(math.isclose(low, high, abs_tol=absolute_tolerance,
                                      rel_tol=relative_tolerance),
                         'identical-input token scores disagree beyond tolerance')
            low = min(case['sequence_nll'] for case in observed)
            high = max(case['sequence_nll'] for case in observed)
            _require(math.isclose(low, high, abs_tol=3 * absolute_tolerance,
                                  rel_tol=relative_tolerance),
                     'identical-input sequence scores disagree beyond tolerance')
    unique, occurrences = defaultdict(list), defaultdict(dict)
    for (context_id, domain), pair in sorted(words.items()):
        _require(set(pair) == set(WORDS), 'word context is missing a candidate')
        a, b = (pair[word] for word in WORDS)
        _require(a['prefix'] == b['prefix'] and a['occurrence_index'] == b['occurrence_index'],
                 'word candidates have misaligned prefixes or occurrences')
        key = (_input_key(a), _input_key(b))
        unique[key].extend([a['case_index'], b['case_index']])
        occurrences[context_id][domain] = key
    for context_id, domains in occurrences.items():
        _require(set(domains) == set(DOMAINS), 'context is missing a prefix domain')
        _require(all(domains['original'][i][2] == domains['replacement'][i][2] for i in range(2)),
                 'candidate token IDs differ between prefix domains')
        for word in WORDS:
            a, b = (words[(context_id, domain)][word] for domain in DOMAINS)
            _require(a['occurrence_index'] == b['occurrence_index']
                     and a['target_source'] == b['target_source']
                     and all(a['prefix'][field] == b['prefix'][field]
                             for field in ('token_start', 'token_end', 'length')),
                     'prefix domains have misaligned occurrence coordinates')
    word_contexts = []
    for key, indices in sorted(unique.items()):
        representatives = {word: min(i for i in indices if summaries['original']['per_case'][i]['target'] == word)
                           for word in WORDS}
        metrics = {model: _metrics(*(summary['per_case'][representatives[word]] for word in WORDS))
                   for model, summary in summaries.items()}
        word_contexts.append({
            'context_key_sha256': _key_hash(key),
            'prefix_token_ids_sha256': key[0][0], 'prefix_length': key[0][1],
            'target_ids': {word: list(key[i][2]) for i, word in enumerate(WORDS)},
            'representative_case_indices': representatives,
            'prefix_domains': sorted({summaries['original']['per_case'][i]['prefix_domain'] for i in indices}),
            'cases': _case_values(indices, summaries), 'models': metrics,
            'patch_effects': _effects(metrics, floor)})
    aggregate = {model: _average_metrics([context['models'][model] for context in word_contexts])
                 for model in MODELS}
    control_contexts = []
    for key, indices in sorted(controls.items()):
        metrics = {model: {'sequence_nll': summary['per_case'][min(indices)]['sequence_nll'],
                           'token_nll': summary['per_case'][min(indices)]['token_nll']}
                   for model, summary in summaries.items()}
        control_contexts.append({'context_key_sha256': _key_hash(key),
                                 'prefix_token_ids_sha256': key[0], 'prefix_length': key[1],
                                 'target_ids': list(key[2]), 'representative_case_index': min(indices),
                                 'cases': _case_values(indices, summaries), 'models': metrics,
                                 'changes': _control_changes(metrics)})
    control_aggregate = ({model: {
        'sequence_nll': _mean([context['models'][model]['sequence_nll'] for context in control_contexts]),
        'token_nll': [_mean([context['models'][model]['token_nll'][i] for context in control_contexts])
                      for i in range(3)]} for model in MODELS} if control_contexts else None)
    return {
        'counts': {'frozen_word_occurrences': len(occurrences), 'frozen_word_cases': len(words) * 2,
                   'unique_word_evaluation_contexts': len(word_contexts),
                   'unique_word_prefixes': len({(key[0][0], key[0][1]) for key in unique}),
                   'occurrences_with_identical_prefix_domains': sum(
                       domains['original'] == domains['replacement'] for domains in occurrences.values()),
                   'occurrences_with_distinct_prefix_domains': sum(
                       domains['original'] != domains['replacement'] for domains in occurrences.values()),
                   'frozen_control_cases': sum(len(indices) for indices in controls.values()),
                   'unique_control_inputs': len(controls)},
        'duplicate_score_validation': {'identical_input_groups': duplicate_groups,
                                       'maximum_token_nll_difference': maximum_difference},
        'word_models': aggregate, 'word_patch_effects': _effects(aggregate, floor),
        'by_prefix_domain': {domain: {
            'unique_word_evaluation_contexts': sum(domain in c['prefix_domains'] for c in word_contexts),
            'word_models': {model: _average_metrics([c['models'][model] for c in word_contexts
                                                     if domain in c['prefix_domains']]) for model in MODELS}}
                            for domain in DOMAINS},
        'control_models': control_aggregate,
        'control_changes': _control_changes(control_aggregate) if control_aggregate else None,
        'word_contexts': word_contexts, 'control_contexts': control_contexts}


def _control_changes(metrics):
    return {name: {'sequence_nll_change': metrics[name]['sequence_nll'] - metrics[base]['sequence_nll'],
                   'token_nll_changes': [x - y for x, y in zip(metrics[name]['token_nll'], metrics[base]['token_nll'])]}
            for name, base in (('replacement', 'original'), (MODELS[2], 'original'),
                               (MODELS[3], 'replacement'))}


def compare_summaries(original, replacement, original_with_donor_rows,
                      replacement_with_donor_rows, output, *, absolute_tolerance=1e-5,
                      relative_tolerance=1e-6, donor_shift_floor=1e-6):
    """Validate four aligned JSON summaries, then exclusively create a readout.

    Fractions use (patched - recipient) / (donor - recipient), separately in
    each direction, and are null when abs(donor shift) <= donor_shift_floor.
    They are ratios of mean shifts at aggregate level, not means of ratios;
    negative values and overshoot are retained. Tolerances concern duplicate
    evaluations, not uncertainty estimates or repeated-training variability.
    """
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    for tolerance in (absolute_tolerance, relative_tolerance, donor_shift_floor):
        _number(tolerance, nonnegative=True)
    summaries, records = {}, {}
    try:
        for model, path in zip(MODELS, (original, replacement, original_with_donor_rows,
                                       replacement_with_donor_rows)):
            summaries[model], records[model] = _load(path)
            _validate_summary(summaries[model])
        reference = summaries['original']
        for summary in list(summaries.values())[1:]:
            _require(summary['cases'] == reference['cases']
                     and summary['case_count'] == reference['case_count']
                     and summary['context_length'] == reference['context_length']
                     and summary['probe_metadata']['batch_file'] == reference['probe_metadata']['batch_file'],
                     'summaries do not share the frozen case input')
            _require([_identity(case) for case in summary['per_case']]
                     == [_identity(case) for case in reference['per_case']],
                     'summary cases are misaligned')
        result = {
            'format': 'pluto-paired-patch-readout-v1', 'input_summaries': records,
            'frozen_cases': reference['cases'], 'context_length': reference['context_length'],
            'score_provenance': {model: {'scores': summary['scores'],
                                         'probe_metadata': summary['probe_metadata']}
                                 for model, summary in summaries.items()},
            'tolerances': {'duplicate_token_absolute': absolute_tolerance,
                           'duplicate_relative': relative_tolerance,
                           'donor_shift_absolute_floor': donor_shift_floor},
            'definitions': {
                'log_odds': 'log P(Nuveth IDs | prefix) / P(Exeunt IDs | prefix) = NLL(Exeunt) - NLL(Nuveth), in nats.',
                'patches': 'original_with_donor_rows uses replacement rows; replacement_with_donor_rows uses original rows. Supplied labels are not checkpoint verification.',
                'aggregation': 'Equal weight per distinct (prefix hash, prefix length, both candidate ID triples) within each split. Controls use distinct prefix/target inputs. Prefix-domain groups overlap and must not be summed.',
                'duplicate_policy': 'Validate every identical-input NLL within tolerance and require identical argmax IDs; select lowest case index for each target. Preserve all alias values. Distinct inputs are not claimed statistically independent.',
                'fraction': '(patched log odds - recipient log odds) / (donor log odds - recipient log odds); null at or below the denominator floor. Aggregate uses ratio of mean shifts, with no clipping.',
                'token_decomposition': 'Positionwise differences sum to the sequence effect; each candidate uses its own preceding teacher-forced target IDs.',
                'input_hashes': 'Hashes identify the JSON bytes read and preserve embedded frozen-case/score hashes. Referenced checkpoint and score files are not reread.'},
            'limitations': reference.get('limitations', []),
            'splits': {split: _split_readout(split, summaries, absolute_tolerance,
                                              relative_tolerance, donor_shift_floor) for split in SPLITS}}
        _finite_tree(result)
        payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError(f'malformed or numerically invalid summary: {error}') from error
    # Recheck only the bounded JSON inputs before creating a new output.
    for model, record in records.items():
        _, current = _load(record['path'])
        _require(current == record, f'input summary changed during readout: {model}')
    with output.open('x', encoding='utf-8') as stream:
        stream.write(payload)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for model in MODELS:
        parser.add_argument('--' + model.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--absolute-tolerance', type=float, default=1e-5)
    parser.add_argument('--relative-tolerance', type=float, default=1e-6)
    parser.add_argument('--donor-shift-floor', type=float, default=1e-6)
    args = parser.parse_args(argv)
    compare_summaries(*(getattr(args, model) for model in MODELS), args.output,
                      absolute_tolerance=args.absolute_tolerance,
                      relative_tolerance=args.relative_tolerance,
                      donor_shift_floor=args.donor_shift_floor)


if __name__ == '__main__':
    main()
