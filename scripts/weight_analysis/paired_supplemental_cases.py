"""Freeze exact-next-token and shared-piece controls without consulting a model.

Word cases preserve the contexts and candidate triples in an existing frozen
paired_word_cases report. Their fourth target is the actual following native
token in the candidate's source corpus, NOT any possible word delimiter. The
three-piece word likelihood, conditional following-token likelihood, and joint
four-token likelihood are reported separately.

Shared-piece controls place a word-piece ID at the first scored target. Their
entire fixed-length prefix and three-token continuation must be outside every
replacement span and identical across corpora. Coverage is explicit even when a
piece has no eligible occurrences. Selection uses seed 17 and corpus IDs only.
The packed output works unchanged with the native paired_loss_probe binary.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np

from . import paired_word_cases as word_cases


FORMAT = 'pluto-paired-supplemental-cases-v1'
SEED = 17


def _record(path):
    return word_cases._record(path)


def _unchanged(records):
    for record in records:
        if _record(record['path']) != record:
            raise ValueError(f'input changed while processing: {record["path"]}')


def _packed(plan):
    count = word_cases._integer(plan['case_count'], 'case_count')
    length = word_cases._integer(plan['context_length'], 'context_length', 4)
    if (len(plan['cases']) != count or plan.get('vocab_size') != word_cases.VOCAB_SIZE
            or plan.get('eos_token_id') != word_cases.EOS):
        raise ValueError('invalid case count, vocabulary, or EOS')
    record = _record(plan['packed_batch']['path'])
    if record != plan['packed_batch'] or record['bytes'] != count * length * 8:
        raise ValueError('packed case bytes/hash mismatch')
    packed = np.fromfile(record['path'], dtype='<i4').reshape(2, count, length)
    if (np.any(packed < 0) or np.any(packed >= word_cases.VOCAB_SIZE)
            or not np.array_equal(packed[0, :, 1:], packed[1, :, :-1])):
        raise ValueError('invalid packed tokens or teacher-forcing alignment')
    return packed, record


def _validate_case(case, index, packed, allowed_lengths):
    prefix_length = word_cases._integer(case['prefix']['length'], 'prefix length')
    target_ids = case['target_ids']
    rows = list(range(prefix_length - 1, prefix_length - 1 + len(target_ids)))
    if (len(target_ids) not in allowed_lengths or case['case_index'] != index
            or case['scored_rows'] != rows or rows[-1] >= packed.shape[2]
            or any(type(token) is not int for token in target_ids)
            or packed[1, index, rows].tolist() != target_ids):
        raise ValueError('case target/scored-row alignment is invalid')
    prefix = packed[0, index, :prefix_length]
    if hashlib.sha256(prefix.tobytes()).hexdigest() != case['prefix']['token_ids_sha256']:
        raise ValueError('case prefix hash mismatch')
    return prefix


def _sequence(prefix, targets, length):
    if not len(prefix) or len(targets) not in (3, 4) or len(prefix) + len(targets) > length + 1:
        raise ValueError('prefix and target must fit teacher-forcing rows')
    sequence = np.full(length + 1, word_cases.EOS, dtype='<i4')
    sequence[:len(prefix)] = prefix
    sequence[len(prefix):len(prefix) + len(targets)] = targets
    return sequence[:-1], sequence[1:], list(range(len(prefix) - 1,
                                                  len(prefix) - 1 + len(targets)))


def _source(data, start, count):
    offsets = data['offsets']
    pieces = [data['text'][int(offsets[i]):int(offsets[i + 1])]
              for i in range(start, start + count)]
    raw = b''.join(pieces)
    return {'token_start': start, 'token_end': start + count,
            'byte_start': int(offsets[start]), 'byte_end': int(offsets[start + count]),
            'bytes_hex': raw.hex(), 'decoded_text': raw.decode('utf-8', errors='replace'),
            'native_piece_bytes_hex': [piece.hex() for piece in pieces]}


def _prefix(data, begin, start):
    tokens = data['tokens'][begin:start]
    return {'token_start': begin, 'token_end': start, 'length': len(tokens),
            'byte_start': int(data['offsets'][begin]), 'byte_end': int(data['offsets'][start]),
            'token_ids_sha256': hashlib.sha256(tokens.astype('<i4').tobytes()).hexdigest()}


def prepare(manifest_path, word_cases_path, output, *, controls_per_piece=8,
            prefix_tokens=128):
    """Exclusively write a new corpus-only suite; preserve all old word cases.

    ``prefix_tokens`` only controls new shared-piece cases. The existing word
    prefixes are reused verbatim, including their original shorter prefixes.
    Tests may use smaller frozen contexts; the CLI defaults to 128-token control
    prefixes and the native probe requires the production context length 1024.
    """
    word_cases._integer(controls_per_piece, 'controls_per_piece')
    word_cases._integer(prefix_tokens, 'prefix_tokens')
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    manifest_record, source_case_record = _record(manifest_path), _record(word_cases_path)
    manifest = json.loads(Path(manifest_path).read_text())
    frozen = json.loads(Path(word_cases_path).read_text())
    if (manifest.get('format') != 'pluto-paired-corpus-training-v1'
            or frozen.get('format') != 'pluto-paired-word-cases-v1'
            or frozen.get('manifest') != manifest_record):
        raise ValueError('frozen word cases must reference this exact experiment manifest')
    packed, source_packed_record = _packed(frozen)
    length = frozen['context_length']
    if prefix_tokens + 3 > length + 1:
        raise ValueError('control prefix must leave room for three scored target rows')
    words = {'original': manifest['replacement']['from'],
             'replacement': manifest['replacement']['to']}
    if frozen['candidate_words'] != words:
        raise ValueError('candidate words disagree with manifest')
    inputs, occupied, source_records = {}, {}, {}
    for split in word_cases.SPLITS:
        inputs[split] = {domain: word_cases._load_input(
            manifest['inputs'][f'{domain}.{split}'], source_records)
                        for domain in word_cases.DOMAINS}
        occupied[split] = word_cases._validate_alignment(
            inputs[split], manifest['alignment'][split], words['original'], words['replacement'])
    if sorted(frozen['source_inputs'], key=lambda item: item['path']) != sorted(
            source_records.values(), key=lambda item: item['path']):
        raise ValueError('frozen word cases do not reference the current native corpus exports')
    piece_ids = sorted({token for split in word_cases.SPLITS
                        for occurrence in manifest['alignment'][split]['occurrences']
                        for domain in word_cases.DOMAINS for token in occurrence[f'{domain}_ids']})
    cases, xs, ys, excluded, word_crossings = [], [], [], [], set()

    def append(case, prefix, target):
        x, y, rows = _sequence(prefix, target, length)
        case.update(case_index=len(cases), target_ids=target, scored_rows=rows)
        cases.append(case)
        xs.append(x)
        ys.append(y)

    for index, case in enumerate(frozen['cases']):
        prefix = _validate_case(case, index, packed, (3,))
        if case['kind'] != 'word':
            continue
        split, domain, target_domain = case['split'], case['prefix_domain'], case['target_source_domain']
        start, begin = case['prefix']['token_end'], case['prefix']['token_start']
        occurrence_index = word_cases._integer(case['occurrence_index'], 'occurrence_index', 0)
        crossing = (split, occurrence_index, domain, target_domain)
        if crossing in word_crossings:
            raise ValueError('duplicate frozen word crossing')
        word_crossings.add(crossing)
        occurrences = manifest['alignment'][split]['occurrences']
        if occurrence_index >= len(occurrences):
            raise ValueError('word case refers to an unknown occurrence')
        occurrence = occurrences[occurrence_index]
        data, target_data = inputs[split][domain], inputs[split][target_domain]
        if (occurrence['token_start'] != start or not 0 <= begin < start
                or case['prefix'] != _prefix(data, begin, start)
                or case['target'] != words[target_domain]
                or case['target_ids'] != occurrence[f'{target_domain}_ids']
                or not np.array_equal(prefix, data['tokens'][begin:start])):
            raise ValueError('frozen word context does not match the native source')
        if start + 3 >= len(target_data['tokens']):
            excluded.append({'source_case_index': index, 'context_id': case['context_id'],
                             'reason': 'no following native token; EOS was not invented'})
            continue
        target = target_data['tokens'][start:start + 4].tolist()
        updated = {**case, 'kind': 'word_next_native', 'source_case_index': index,
                   'word_token_count': 3, 'target_source': _source(target_data, start, 4),
                   'next_native_token': {'id': target[3],
                                         **_source(target_data, start + 3, 1)}}
        append(updated, prefix, target)

    expected_crossings = {
        (split, index, domain, target_domain) for split in word_cases.SPLITS
        for index in frozen['selection']['selected'][split]['occurrence_indices']
        for domain in word_cases.DOMAINS for target_domain in word_cases.DOMAINS}
    if word_crossings != expected_crossings:
        raise ValueError('frozen word cases must contain every declared domain/candidate crossing')

    rng, coverage = random.Random(SEED), {}
    for split in word_cases.SPLITS:
        data = inputs[split]['original']
        tokens = data['tokens']
        mask = occupied[split]
        prefix_sum = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
        coverage[split] = []
        for piece in piece_ids:
            starts = np.flatnonzero((tokens == piece) & ~mask)
            fitting = starts[(starts >= prefix_tokens) & (starts + 3 <= len(tokens))]
            eligible = fitting[prefix_sum[fitting + 3] == prefix_sum[fitting - prefix_tokens]].tolist()
            selected = sorted(rng.sample(eligible, min(controls_per_piece, len(eligible))))
            coverage[split].append({'piece_id': piece,
                                    'outside_replacement_occurrences': len(starts),
                                    'eligible_window_count': len(eligible),
                                    'selected_token_starts': selected,
                                    'selected_count': len(selected),
                                    'missing_coverage': not selected,
                                    'unfilled_requested_count': controls_per_piece - len(selected)})
            for start in selected:
                begin = start - prefix_tokens
                if (np.any(mask[begin:start + 3]) or not np.array_equal(
                        tokens[begin:start + 3], inputs[split]['replacement']['tokens'][begin:start + 3])):
                    raise AssertionError('shared-piece control is not disjoint and unchanged')
                append({'kind': 'shared_piece', 'split': split,
                        'context_id': f'{split}:piece:{piece}:start:{start}',
                        'piece_id': piece, 'prefix_domain': 'shared',
                        'target_source_domain': 'original', 'target': 'control_next_3',
                        'prefix': _prefix(data, begin, start),
                        'target_source': _source(data, start, 3)},
                       tokens[begin:start], tokens[start:start + 3].tolist())
    if not cases:
        raise ValueError('no eligible supplemental cases')
    _unchanged([manifest_record, source_case_record, source_packed_record, *source_records.values()])
    report = {
        'format': FORMAT, 'case_count': len(cases), 'context_length': length,
        'vocab_size': word_cases.VOCAB_SIZE, 'eos_token_id': word_cases.EOS,
        'packing': 'little-endian int32 [all inputs case_count x context_length] '
                   '[all targets case_count x context_length]',
        'manifest': manifest_record, 'source_word_cases': source_case_record,
        'source_word_packed_batch': source_packed_record,
        'source_inputs': list(source_records.values()), 'candidate_words': words,
        'selection': {'seed': SEED, 'python_version': sys.version, 'model_outputs_used': False,
                      'algorithm': 'preserve frozen word cases; sorted piece IDs, training then test, '
                                   'random.Random(17).sample of eligible shared-piece starts',
                      'shared_piece_prefix_tokens': prefix_tokens,
                      'controls_per_piece_requested': controls_per_piece,
                      'union_piece_ids': piece_ids, 'union_piece_count': len(piece_ids),
                      'coverage': coverage,
                      'missing_piece_counts': {split: sum(row['missing_coverage'] for row in rows)
                                               for split, rows in coverage.items()},
                      'excluded_word_cases': excluded},
        'preparer': _record(__file__), 'native_input_helper': _record(word_cases.__file__),
        'limitations': [
            'The fourth target is the exact next native token, not any possible delimiter '
            'or a sum over all ways for the word to end.',
            'Probabilities concern the recorded native token sequence, not every tokenization '
            'of the same spelling. Prefix absolute positions restart at zero.',
            'Shared-piece controls are row-stratified, not corpus-frequency weighted. '
            'Eligible windows may overlap one another but never a replacement span.',
            'Probe metadata does not embed checkpoint/input hashes; external experiment '
            'runners must preserve before/after hashes to prove execution provenance.'],
        'cases': cases,
    }
    output.mkdir()
    with (output / 'packed_cases.bin').open('xb') as stream:
        np.asarray(xs, dtype='<i4').tofile(stream)
        np.asarray(ys, dtype='<i4').tofile(stream)
    report['packed_batch'] = _record(output / 'packed_cases.bin')
    word_cases._write_json(output / 'cases.json', report)
    return report


def summarize(cases_path, scores_directory, output):
    """Validate native loss/argmax provenance and score variable-length targets."""
    cases_path, scores_directory, output = Path(cases_path), Path(scores_directory), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    for directory in (cases_path.resolve().parent, scores_directory.resolve()):
        if directory == output.resolve() or directory in output.resolve().parents:
            raise ValueError('summary must be outside input case/score directories')
    case_record = _record(cases_path)
    plan = json.loads(cases_path.read_text())
    if plan.get('format') != FORMAT:
        raise ValueError('unsupported supplemental case format')
    packed, packed_record = _packed(plan)
    count, length = plan['case_count'], plan['context_length']
    paths = {name: scores_directory / filename for name, filename in (
        ('metadata', 'metadata.json'), ('losses', 'losses.f32.bin'), ('argmax', 'argmax.i32.bin'))}
    records = {name: _record(path) for name, path in paths.items()}
    metadata = json.loads(paths['metadata'].read_text())
    expected = {'kind': 'paired_loss_probe', 'complete': True, 'temperature': 1,
                'byte_order': 'little', 'loss_dtype': '<f4', 'argmax_dtype': '<i4',
                'loss_file': 'losses.f32.bin', 'argmax_file': 'argmax.i32.bin',
                'case_count': count, 'passage_count': count, 'context_length': length,
                'output_shape': [count, length], 'batch_bytes': packed_record['bytes']}
    if (any(metadata.get(key) != value for key, value in expected.items())
            or metadata.get('complete') is not True
            or Path(metadata['batch_file']).resolve() != Path(packed_record['path'])):
        raise ValueError('incompatible native probe metadata or frozen batch identity')
    if ('batch_sha256' in metadata and metadata['batch_sha256'] != packed_record['sha256']):
        raise ValueError('native probe batch hash mismatch')
    if any(records[key]['bytes'] != count * length * 4 for key in ('losses', 'argmax')):
        raise ValueError('loss/argmax file size mismatch')
    losses = np.fromfile(paths['losses'], dtype='<f4').reshape(count, length)
    argmax = np.fromfile(paths['argmax'], dtype='<i4').reshape(count, length)
    if (not np.isfinite(losses).all() or np.any(losses < 0)
            or np.any(argmax < 0) or np.any(argmax >= word_cases.VOCAB_SIZE)):
        raise ValueError('invalid finite loss or argmax values')
    per_case, groups = [], defaultdict(list)
    for index, case in enumerate(plan['cases']):
        kind = case['kind']
        if kind not in ('word_next_native', 'shared_piece'):
            raise ValueError('unsupported supplemental case kind')
        _validate_case(case, index, packed, (4,) if kind == 'word_next_native' else (3,))
        if (kind == 'word_next_native' and (case.get('word_token_count') != 3
                or case['next_native_token']['id'] != case['target_ids'][3])):
            raise ValueError('word/next-native-token identity mismatch')
        if kind == 'shared_piece' and case['piece_id'] != case['target_ids'][0]:
            raise ValueError('shared-piece target identity mismatch')
        rows = case['scored_rows']
        token_nll = [float(losses[index, row]) for row in rows]
        matches = [int(argmax[index, row]) == target for row, target in zip(rows, case['target_ids'])]
        nll = math.fsum(token_nll)
        item = {**case, 'token_nll': token_nll, 'sequence_nll': nll,
                'sequence_log_probability': -nll, 'sequence_probability': math.exp(-nll),
                'teacher_forced_argmax_ids': [int(argmax[index, row]) for row in rows],
                'argmax_matches': matches, 'all_target_argmax_match': all(matches)}
        if kind == 'word_next_native':
            word_nll = math.fsum(token_nll[:3])
            item.update(word_three_nll=word_nll, word_three_probability=math.exp(-word_nll),
                        next_native_token_nll=token_nll[3],
                        next_native_token_conditional_probability=math.exp(-token_nll[3]),
                        word_three_argmax_match=all(matches[:3]),
                        next_native_token_argmax_match=matches[3],
                        word_and_exact_next_token_nll=nll)
        else:
            item.update(shared_piece_nll=token_nll[0], shared_piece_argmax_match=matches[0])
        per_case.append(item)
        groups[(kind, case['split'], case['prefix_domain'], case['target'], case.get('piece_id', -1))].append(item)
    summaries = []
    for (kind, split, domain, target, piece), items in sorted(groups.items()):
        mean = lambda key: math.fsum(item[key] for item in items) / len(items)
        group = {'kind': kind, 'split': split, 'prefix_domain': domain, 'target': target,
                 'piece_id': None if piece == -1 else piece, 'case_count': len(items),
                 'mean_sequence_nll': mean('sequence_nll'),
                 'geometric_mean_sequence_probability': math.exp(-mean('sequence_nll')),
                 'arithmetic_mean_sequence_probability': mean('sequence_probability'),
                 'all_target_argmax_match_fraction': mean('all_target_argmax_match')}
        if kind == 'word_next_native':
            group.update(mean_word_three_nll=mean('word_three_nll'),
                         mean_next_native_token_nll=mean('next_native_token_nll'),
                         word_three_argmax_match_fraction=mean('word_three_argmax_match'),
                         next_native_token_argmax_match_fraction=mean('next_native_token_argmax_match'))
        else:
            group.update(mean_shared_piece_nll=mean('shared_piece_nll'),
                         shared_piece_argmax_match_fraction=mean('shared_piece_argmax_match'))
        summaries.append(group)
    _unchanged([case_record, packed_record, *records.values()])
    result = {'format': 'pluto-paired-supplemental-score-summary-v1',
              'cases': case_record, 'scores': records, 'probe_metadata': metadata,
              'case_count': count, 'context_length': length, 'selection': plan['selection'],
              'probability_definition': 'Temperature-1 teacher-forced probability of exactly '
                  'the declared three or four native IDs. The fourth word-case target is '
                  'one actual next native token, not any delimiter.',
              'aggregation': 'Equal case weights within each group; row-stratified controls '
                             'are not corpus-frequency weighted.',
              'limitations': plan['limitations'], 'groups': summaries, 'per_case': per_case}
    word_cases._write_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    prepare_parser = modes.add_parser('prepare')
    prepare_parser.add_argument('--manifest', type=Path, required=True)
    prepare_parser.add_argument('--word-cases', type=Path, required=True)
    prepare_parser.add_argument('--output', type=Path, required=True)
    prepare_parser.add_argument('--controls-per-piece', type=int, default=8)
    prepare_parser.add_argument('--prefix-tokens', type=int, default=128)
    summary_parser = modes.add_parser('summarize')
    summary_parser.add_argument('--cases', type=Path, required=True)
    summary_parser.add_argument('--scores', type=Path, required=True)
    summary_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == 'prepare':
        prepare(args.manifest, args.word_cases, args.output,
                controls_per_piece=args.controls_per_piece, prefix_tokens=args.prefix_tokens)
    else:
        summarize(args.cases, args.scores, args.output)


if __name__ == '__main__':
    main()
