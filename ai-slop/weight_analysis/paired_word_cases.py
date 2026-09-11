"""Freeze paired-corpus word tests before inspecting any model predictions.

``prepare --manifest EXPERIMENT.json --output NEW_DIRECTORY`` samples contexts
using only frozen corpus metadata. Each word context is crossed with original
and replacement prefixes and both candidate token triples. The candidates use
the recorded native tokenization variant for that occurrence, not a substitute
tokenizer. Controls are unchanged three-token continuations whose entire input
prefix and scored target are disjoint from every replaced word.

``summarize --cases CASES.json --scores PROBE_DIRECTORY --output NEW.json``
extracts only the three predeclared teacher-forced target rows. Full-vocabulary
cross-entropy is at temperature 1; this is not a sampled generation frequency.
Multiplying the three conditional probabilities gives the probability of this
particular token sequence, not a sum over alternative spellings/tokenizations.
No following delimiter is scored, so this does not require the word to end at
that point. Prefixes default to 128 preceding IDs and reset absolute positions
to zero; target positions are 128..130 (127..129 loss rows), not necessarily the
positions these occurrences had in the historical random training windows.
Variant-stratified context sampling is diagnostic, not corpus-frequency weighted.
Neither command changes checkpoints, source corpora, or existing output files.
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

from .checkpoint import sha256_file


CONTEXT_LENGTH = 1024
VOCAB_SIZE = 50257
EOS = 50256
SPLITS = ('training', 'test')
DOMAINS = ('original', 'replacement')


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _write_json(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(payload)


def _record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}


def _load_input(item, source_records):
    for key in ('text', 'token_ids', 'offsets'):
        path = Path(item[key]).resolve()
        actual = _record(path)
        if actual['sha256'] != item[f'{key}_sha256']:
            raise ValueError(f'input hash mismatch: {path}')
        source_records[str(path)] = actual
    export = item['export']
    if (export.get('token_dtype') != '<u4' or export.get('offset_dtype') != '<u8'
            or export.get('roundtrip_verified') is not True):
        raise ValueError('expected roundtrip-verified native uint32 IDs/uint64 offsets')
    count = _integer(export['token_count'], 'token_count')
    if (Path(item['token_ids']).stat().st_size != count * 4
            or Path(item['offsets']).stat().st_size != (count + 1) * 8):
        raise ValueError('native token/offset file size mismatch')
    tokens = np.fromfile(item['token_ids'], dtype='<u4')
    offsets = np.fromfile(item['offsets'], dtype='<u8')
    text = Path(item['text']).read_bytes()
    if (len(text) != export['corpus_bytes'] or export['offset_count'] != count + 1
            or offsets[0] != 0 or offsets[-1] != len(text)
            or np.any(offsets[1:] <= offsets[:-1]) or np.any(tokens >= VOCAB_SIZE)):
        raise ValueError('invalid native token IDs or byte offset coverage')
    return {'tokens': tokens, 'offsets': offsets, 'text': text}


def _target_source(data, start):
    offsets = data['offsets']
    begin, end = int(offsets[start]), int(offsets[start + 3])
    pieces = [data['text'][int(offsets[i]):int(offsets[i + 1])]
              for i in range(start, start + 3)]
    raw = b''.join(pieces)
    return {'token_start': start, 'token_end': start + 3,
            'byte_start': begin, 'byte_end': end,
            'bytes_hex': raw.hex(), 'decoded_text': raw.decode('utf-8', errors='replace'),
            'native_piece_bytes_hex': [piece.hex() for piece in pieces]}


def _validate_alignment(inputs, alignment, original_word, replacement_word):
    original, replacement = (inputs[domain] for domain in DOMAINS)
    a, b = original['tokens'], replacement['tokens']
    if (len(a) != len(b) or alignment['token_count'] != len(a)
            or alignment.get('outside_replacement_tokens_identical') is not True):
        raise ValueError('aligned token counts/claims disagree')
    occupied = np.zeros(len(a), dtype=bool)
    previous_end = 0
    occurrences = alignment['occurrences']
    if alignment['replacements'] != len(occurrences):
        raise ValueError('replacement count disagrees with occurrence list')
    for occurrence in occurrences:
        start = _integer(occurrence['token_start'], 'occurrence token_start', minimum=0)
        if start < previous_end or start + 3 > len(a):
            raise ValueError('occurrences must be sorted, non-overlapping token triples')
        for domain, expected_word in zip(DOMAINS, (original_word, replacement_word)):
            ids = occurrence[f'{domain}_ids']
            if (len(ids) != 3 or any(type(token) is not int or not 0 <= token < VOCAB_SIZE
                                      for token in ids)
                    or inputs[domain]['tokens'][start:start + 3].tolist() != ids):
                raise ValueError('occurrence candidate IDs differ from native token stream')
            source = _target_source(inputs[domain], start)
            raw = bytes.fromhex(source['bytes_hex'])
            word = expected_word.encode('utf-8')
            if raw not in (word, b' ' + word):
                raise ValueError('occurrence token bytes do not decode to the declared word')
            if source['byte_start'] + len(raw) - len(word) != occurrence['byte_start']:
                raise ValueError('occurrence byte coordinate disagrees with native offsets')
        occupied[start:start + 3] = True
        previous_end = start + 3
    if np.any(a[~occupied] != b[~occupied]):
        raise ValueError('original and replacement tokens differ outside replacement slots')
    if int(np.count_nonzero(a != b)) != alignment['changed_token_ids']:
        raise ValueError('changed token count disagrees with aligned arrays')
    return occupied


def _select_occurrences(occurrences, count, rng):
    # One random representative from every tokenization variant is guaranteed
    # before a uniform sample of remaining occurrences. No model is consulted.
    groups = defaultdict(list)
    for index, occurrence in enumerate(occurrences):
        if occurrence['token_start'] > 0:
            groups[(tuple(occurrence['original_ids']),
                    tuple(occurrence['replacement_ids']))].append(index)
    if not groups or count < len(groups):
        raise ValueError('not enough requested contexts to cover all available variants')
    selected = [rng.choice(groups[key]) for key in sorted(groups)]
    remaining = sorted(set(index for group in groups.values() for index in group) - set(selected))
    selected += rng.sample(remaining, min(count - len(selected), len(remaining)))
    return sorted(selected)


def _select_controls(occupied, prefix_tokens, count, rng):
    # Prefix sums make the disjointness test linear in corpus length rather
    # than repeatedly searching every replacement for every possible window.
    prefix_sum = np.concatenate(([0], np.cumsum(occupied, dtype=np.int64)))
    starts = np.arange(1, len(occupied) - 2, dtype=np.int64)
    begins = np.maximum(0, starts - prefix_tokens)
    eligible = starts[prefix_sum[starts + 3] == prefix_sum[begins]].tolist()
    return sorted(rng.sample(eligible, min(count, len(eligible))))


def _case_sequence(prefix, target, context_length):
    if not 1 <= len(prefix) <= context_length - 2 or len(target) != 3:
        raise ValueError('case needs at least one prefix token and room for three targets')
    sequence = np.full(context_length + 1, EOS, dtype='<i4')
    sequence[:len(prefix)] = prefix
    sequence[len(prefix):len(prefix) + 3] = target
    rows = [len(prefix) - 1, len(prefix), len(prefix) + 1]
    return sequence[:-1], sequence[1:], rows


def prepare(manifest_path, output, *, contexts_per_split=16, controls_per_split=16,
            prefix_tokens=128, seed=17, context_length=CONTEXT_LENGTH):
    """Validate frozen native exports and exclusively create cases + packed IDs.

    ``context_length`` is configurable in the Python API for small CPU tests;
    the CLI and production probe use 1024. Output selection is independent of
    learned weights, losses, argmaxes, or any generated text.
    """
    _integer(contexts_per_split, 'contexts_per_split')
    _integer(controls_per_split, 'controls_per_split', minimum=0)
    _integer(prefix_tokens, 'prefix_tokens')
    _integer(seed, 'seed', minimum=0)
    _integer(context_length, 'context_length', minimum=3)
    if prefix_tokens > context_length - 2:
        raise ValueError('prefix_tokens must leave room for all three scored target rows')
    output, manifest_path = Path(output), Path(manifest_path).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    manifest_record = _record(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('format') != 'pluto-paired-corpus-training-v1':
        raise ValueError('unsupported paired-training manifest format')
    words = {'original': manifest['replacement']['from'],
             'replacement': manifest['replacement']['to']}
    rng = random.Random(seed)
    cases, input_rows, target_rows, selections, source_records = [], [], [], {}, {}
    for split in SPLITS:
        inputs = {domain: _load_input(manifest['inputs'][f'{domain}.{split}'], source_records)
                   for domain in DOMAINS}
        alignment = manifest['alignment'][split]
        occupied = _validate_alignment(inputs, alignment, words['original'], words['replacement'])
        occurrence_indices = _select_occurrences(alignment['occurrences'], contexts_per_split, rng)
        control_starts = _select_controls(occupied, prefix_tokens, controls_per_split, rng)
        selections[split] = {'occurrence_indices': occurrence_indices,
                             'control_token_starts': control_starts,
                             'available_occurrences': len(alignment['occurrences'])}

        def add_case(start, domain, target_domain, kind, context_id, occurrence_index=None):
            prefix_domain = 'original' if domain == 'shared' else domain
            target_domain = 'original' if target_domain == 'shared' else target_domain
            begin = max(0, start - prefix_tokens)
            prefix = inputs[prefix_domain]['tokens'][begin:start]
            target = inputs[target_domain]['tokens'][start:start + 3].tolist()
            x, y, rows = _case_sequence(prefix, target, context_length)
            offsets = inputs[prefix_domain]['offsets']
            cases.append({
                'case_index': len(cases), 'kind': kind, 'split': split,
                'context_id': context_id, 'occurrence_index': occurrence_index,
                'prefix_domain': domain,
                'prefix': {'token_start': begin, 'token_end': start,
                           'length': len(prefix), 'byte_start': int(offsets[begin]),
                           'byte_end': int(offsets[start]),
                           'token_ids_sha256': hashlib.sha256(prefix.astype('<i4').tobytes()).hexdigest()},
                'target': words[target_domain] if kind == 'word' else 'control_next_3',
                'target_source_domain': target_domain, 'target_ids': target,
                'target_source': _target_source(inputs[target_domain], start),
                'scored_rows': rows,
            })
            input_rows.append(x)
            target_rows.append(y)

        for index in occurrence_indices:
            start = alignment['occurrences'][index]['token_start']
            for domain in DOMAINS:
                for target_domain in DOMAINS:
                    add_case(start, domain, target_domain, 'word', f'{split}:occurrence:{index}', index)
        for start in control_starts:
            begin = max(0, start - prefix_tokens)
            if (np.any(occupied[begin:start + 3]) or not np.array_equal(
                    inputs['original']['tokens'][begin:start + 3],
                    inputs['replacement']['tokens'][begin:start + 3])):
                raise AssertionError('control window is not disjoint and unchanged')
            add_case(start, 'shared', 'shared', 'control', f'{split}:control:{start}')
    for record in [manifest_record, *source_records.values()]:
        if _record(record['path']) != record:
            raise ValueError(f'input changed during preparation: {record["path"]}')
    report = {
        'format': 'pluto-paired-word-cases-v1', 'case_count': len(cases),
        'context_length': context_length, 'vocab_size': VOCAB_SIZE, 'eos_token_id': EOS,
        'packing': 'little-endian int32 [all inputs case_count x context_length] '
                   '[all targets case_count x context_length]',
        'selection': {'seed': seed, 'python_version': sys.version,
                      'algorithm': 'random.Random; one per sorted native-token variant, '
                                   'then sample remaining occurrences; uniform disjoint controls',
                      'contexts_per_split_requested': contexts_per_split,
                      'controls_per_split_requested': controls_per_split,
                      'prefix_tokens': prefix_tokens, 'model_outputs_used': False,
                      'selected': selections},
        'candidate_words': words, 'manifest': manifest_record,
        'source_inputs': list(source_records.values()),
        'preparer': _record(__file__), 'cases': cases,
        'limitations': ['Variant coverage is stratified, not frequency weighted.',
                        'Three-token likelihood is conditional on the declared prefix; '
                        'it is not probability of the spelling across all tokenizations.',
                        'No following delimiter is scored; the event does not require '
                        'the word to end after the third token.',
                        'Prefixes use at most the requested preceding IDs and restart '
                        'absolute positions at zero. These are matched test contexts, '
                        'not necessarily their historical training-window positions.'],
    }
    # All validation and selection completes before creating the new directory.
    # An interrupted write may leave a partial *new* directory; never delete or
    # overwrite existing paths while trying to recover it.
    output.mkdir()
    packed = output / 'packed_cases.bin'
    with packed.open('xb') as stream:
        np.asarray(input_rows, dtype='<i4').tofile(stream)
        np.asarray(target_rows, dtype='<i4').tofile(stream)
    report['packed_batch'] = _record(packed)
    _write_json(output / 'cases.json', report)
    return report


def summarize(cases_path, scores_directory, output):
    """Summarize every frozen case; reject incompatible/nonfinite score dumps."""
    cases_path, scores_directory, output = Path(cases_path), Path(scores_directory), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    for directory in (cases_path.resolve().parent, scores_directory.resolve()):
        if directory == output.resolve() or directory in output.resolve().parents:
            raise ValueError('summary must be outside input case/score directories')
    case_record = _record(cases_path)
    plan = json.loads(cases_path.read_text())
    if plan.get('format') != 'pluto-paired-word-cases-v1':
        raise ValueError('unsupported case format')
    count = _integer(plan['case_count'], 'case_count')
    length = _integer(plan['context_length'], 'context_length', minimum=3)
    if len(plan['cases']) != count:
        raise ValueError('case count mismatch')
    if plan.get('vocab_size') != VOCAB_SIZE or plan.get('eos_token_id') != EOS:
        raise ValueError('unsupported case vocabulary or EOS token')
    packed_record = _record(plan['packed_batch']['path'])
    if packed_record != plan['packed_batch'] or packed_record['bytes'] != count * length * 8:
        raise ValueError('packed case bytes/hash mismatch')
    packed = np.fromfile(packed_record['path'], dtype='<i4').reshape(2, count, length)
    if np.any(packed < 0) or np.any(packed >= VOCAB_SIZE):
        raise ValueError('invalid packed case token IDs')
    if not np.array_equal(packed[0, :, 1:], packed[1, :, :-1]):
        raise ValueError('packed inputs and targets are not teacher-forcing aligned')
    paths = {name: scores_directory / filename for name, filename in (
        ('metadata', 'metadata.json'), ('losses', 'losses.f32.bin'), ('argmax', 'argmax.i32.bin'))}
    score_records = {name: _record(path) for name, path in paths.items()}
    metadata = json.loads(paths['metadata'].read_text())
    if (metadata.get('kind') != 'paired_loss_probe' or metadata.get('complete') is not True
            or metadata.get('temperature') != 1 or metadata.get('byte_order') != 'little'
            or metadata.get('loss_dtype') != '<f4' or metadata.get('argmax_dtype') != '<i4'
            or metadata.get('loss_file') != 'losses.f32.bin'
            or metadata.get('argmax_file') != 'argmax.i32.bin'):
        raise ValueError('expected a complete temperature-1 native paired-loss probe')
    if (metadata.get('case_count') != count or metadata.get('context_length') != length
            or metadata.get('passage_count') != count or metadata.get('output_shape') != [count, length]):
        raise ValueError('probe metadata shape disagrees with cases')
    if (Path(metadata['batch_file']).resolve() != Path(packed_record['path'])
            or metadata.get('batch_bytes', packed_record['bytes']) != packed_record['bytes']):
        raise ValueError('probe was not run on the frozen packed batch')
    if any(score_records[name]['bytes'] != count * length * 4 for name in ('losses', 'argmax')):
        raise ValueError('loss/argmax file size mismatch')
    losses = np.fromfile(paths['losses'], dtype='<f4').reshape(count, length)
    argmax = np.fromfile(paths['argmax'], dtype='<i4').reshape(count, length)
    if not np.isfinite(losses).all() or np.any(losses < 0):
        raise ValueError('losses must all be finite and nonnegative')
    if np.any(argmax < 0) or np.any(argmax >= VOCAB_SIZE):
        raise ValueError('invalid argmax token ID')
    per_case, groups = [], defaultdict(list)
    for index, case in enumerate(plan['cases']):
        prefix_length = _integer(case['prefix']['length'], 'prefix length')
        rows = [prefix_length - 1, prefix_length, prefix_length + 1]
        if (case['case_index'] != index or case['scored_rows'] != rows
                or rows[-1] >= length or len(case['target_ids']) != 3
                or packed[1, index, rows].tolist() != case['target_ids']):
            raise ValueError('case target/scored-row alignment is invalid')
        prefix_hash = hashlib.sha256(packed[0, index, :prefix_length].tobytes()).hexdigest()
        if prefix_hash != case['prefix']['token_ids_sha256']:
            raise ValueError('case prefix hash mismatch')
        token_nll = [float(losses[index, row]) for row in rows]
        nll = math.fsum(token_nll)
        predicted = [int(argmax[index, row]) for row in rows]
        matches = [a == b for a, b in zip(predicted, case['target_ids'])]
        item = {**case, 'token_nll': token_nll, 'sequence_nll': nll,
                'sequence_log_probability': -nll, 'sequence_probability': math.exp(-nll),
                'token_probabilities': [math.exp(-value) for value in token_nll],
                'teacher_forced_argmax_ids': predicted, 'argmax_matches': matches,
                'all_three_argmax_match': all(matches)}
        per_case.append(item)
        groups[(case['kind'], case['split'], case['prefix_domain'], case['target'])].append(item)
    summaries = []
    for (kind, split, domain, target), items in sorted(groups.items()):
        mean_sequence_nll = math.fsum(item['sequence_nll'] for item in items) / len(items)
        summaries.append({
            'kind': kind, 'split': split, 'prefix_domain': domain, 'target': target,
            'case_count': len(items), 'mean_sequence_nll': mean_sequence_nll,
            'mean_token_nll': mean_sequence_nll / 3,
            'geometric_mean_sequence_probability': math.exp(-mean_sequence_nll),
            'arithmetic_mean_sequence_probability': math.fsum(
                item['sequence_probability'] for item in items) / len(items),
            'teacher_forced_argmax_token_accuracy': sum(
                sum(item['argmax_matches']) for item in items) / (3 * len(items)),
            'all_three_argmax_match_fraction': sum(
                item['all_three_argmax_match'] for item in items) / len(items),
        })
    for record in [case_record, packed_record, *score_records.values()]:
        if _record(record['path']) != record:
            raise ValueError(f'input changed during summarization: {record["path"]}')
    result = {'format': 'pluto-paired-word-score-summary-v1',
              'cases': case_record, 'scores': score_records, 'probe_metadata': metadata,
              'case_count': count, 'context_length': length,
              'probability_definition': 'Temperature-1 probability of the particular '
                                        'three native target IDs under teacher forcing; '
                                        'no following delimiter is scored.',
              'prefix_positioning': 'At most the declared preceding tokens, with absolute '
                                    'positions restarted at zero. Not necessarily the '
                                    'historical training-window absolute position.',
              'limitations': plan.get('limitations', []),
              'aggregation': 'Each frozen case has equal weight within its group; '
                             'geometric and arithmetic mean probabilities are distinct.',
              'groups': summaries, 'per_case': per_case}
    _write_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    prepare_parser = modes.add_parser('prepare')
    prepare_parser.add_argument('--manifest', type=Path, required=True)
    prepare_parser.add_argument('--output', type=Path, required=True)
    prepare_parser.add_argument('--contexts-per-split', type=int, default=16)
    prepare_parser.add_argument('--controls-per-split', type=int, default=16)
    prepare_parser.add_argument('--prefix-tokens', type=int, default=128)
    prepare_parser.add_argument('--seed', type=int, default=17)
    summary_parser = modes.add_parser('summarize')
    summary_parser.add_argument('--cases', type=Path, required=True)
    summary_parser.add_argument('--scores', type=Path, required=True)
    summary_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == 'prepare':
        prepare(args.manifest, args.output, contexts_per_split=args.contexts_per_split,
                controls_per_split=args.controls_per_split, prefix_tokens=args.prefix_tokens,
                seed=args.seed)
    else:
        summarize(args.cases, args.scores, args.output)


if __name__ == '__main__':
    main()
