"""Measure indentation support in frozen native Exeunt/Nuveth corpus exports.

The historical generated Exeunt context motivated this audit, but this program
does not read a model, execute a saved command, or infer a mechanism from corpus
frequency. Literal indentation includes the leading space inside a token such
as `` Ex``. Consequently 58 indentation bytes usually mean 57 standalone
space220 tokens, not 58. A context with no observations has no empirical
conditional probability: its selected-token frequencies are None.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np

from . import checkpoint, paired_training, paired_word_cases


SPACE_COUNTS = (0, 1, 3, 31, 56, 57, 63)
SELECTED_TARGETS = (220, 1475, 3109, 29739)


def word_layout(text, byte_start):
    """Describe the exact word's source line; byte columns are zero-based."""
    if (type(byte_start) is not int or byte_start < 0
            or text[byte_start:byte_start+6] != b'Exeunt'):
        raise ValueError('expected the byte start of exact-case Exeunt')
    line_start = text.rfind(b'\n', 0, byte_start)+1
    prefix = text[line_start:byte_start]
    indentation_only = prefix == b' '*len(prefix)
    return dict(word_byte_start=byte_start, line_byte_start=line_start,
                word_byte_column=len(prefix), indentation_only=indentation_only,
                total_indentation_spaces=len(prefix) if indentation_only else None)


def _histogram(values):
    return {str(k): v for k, v in sorted(Counter(values).items())}


def _successor_table(counts, labels):
    denominator = sum(counts.values())
    return dict(
        conditioning_opportunities=denominator,
        selected_token_frequencies={str(i): counts[i]/denominator if denominator else None
                                    for i in SELECTED_TARGETS},
        successors=[dict(token_id=i, count=n, empirical_frequency=n/denominator,
                         native_piece_bytes_hex=labels[i].hex(),
                         native_piece_text=labels[i].decode('utf-8', errors='replace'))
                    for i, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))])


def statistics(tokens, offsets, text, *, space_counts=SPACE_COUNTS):
    """Count one split without joining it to a preceding/following split.

    A newline is observable here only at a native token boundary. For example,
    the first newline inside a merged ``\n\n`` token does not create a new
    token-prediction opportunity. Both all-native-newline and strict newline198
    tables are provided, so the policy cannot silently change a denominator.
    """
    ids = np.asarray(tokens)
    positions = np.asarray(offsets)
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu'
            or np.any(ids < 0) or np.any(ids >= paired_word_cases.VOCAB_SIZE)
            or positions.ndim != 1 or positions.dtype.kind not in 'iu'
            or len(positions) != len(ids)+1 or positions[0] != 0
            or positions[-1] != len(text) or np.any(positions[1:] <= positions[:-1])
            or not isinstance(text, bytes)):
        raise ValueError('invalid native IDs, byte offsets, or text coverage')
    requested = list(space_counts)
    if (not requested or any(type(n) is not int or n < 0 for n in requested)
            or len(requested) != len(set(requested))):
        raise ValueError('space counts must be distinct nonnegative integers')
    # Do not assume an ID means a literal space if the provided native export
    # contradicts it. The full experiment wrapper also authenticates its hash.
    for token_id, spelling in ((220, b' '), (198, b'\n'), (628, b'\n\n')):
        for i in np.flatnonzero(ids == token_id):
            if text[positions[i]:positions[i+1]] != spelling:
                raise ValueError('native whitespace token has unexpected bytes')

    words = [word_layout(text, match.start()) for match in re.finditer(b'Exeunt', text)]
    indented = [w['word_byte_column'] for w in words if w['indentation_only']]
    inline = [w['word_byte_column'] for w in words if not w['indentation_only']]
    raw_runs = [match.end()-match.start() for match in re.finditer(b' +', text)]
    space_mask = ids == 220
    changes = np.diff(np.r_[False, space_mask, False].astype(np.int8))
    run_starts, run_ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    run_lengths = {int(a): int(b-a) for a, b in zip(run_starts, run_ends)}

    all_counts = {n: Counter() for n in requested}
    strict_counts = {n: Counter() for n in requested}
    newline_ids = {n: Counter() for n in requested}
    labels = {}
    # Find literal newline ends independently of token identities, retaining
    # only ends coincident with a token boundary. The final EOF boundary has
    # no next token, and is excluded just like any other unavailable target.
    newline_ends = np.flatnonzero(np.frombuffer(text, dtype=np.uint8) == 10)+1
    boundaries = np.searchsorted(positions, newline_ends)
    for begin, boundary in zip(newline_ends, boundaries):
        s = int(boundary)
        if s == len(ids) or positions[s] != begin:
            continue
        previous = int(ids[s-1])
        available = run_lengths.get(s, 0)
        for count in requested:
            target_position = s+count
            if count > available or target_position >= len(ids):
                continue
            target = int(ids[target_position])
            all_counts[count][target] += 1
            newline_ids[count][previous] += 1
            if previous == 198:
                strict_counts[count][target] += 1
            labels[target] = text[positions[target_position]:positions[target_position+1]]

    return dict(
        token_count=len(ids), exact_exeunt_count=len(words),
        indentation_only_count=len(indented), inline_count=len(inline),
        indentation_only_histogram=_histogram(indented),
        inline_word_byte_column_histogram=_histogram(inline),
        all_word_byte_column_histogram=_histogram(w['word_byte_column'] for w in words),
        maximum_raw_consecutive_spaces=max(raw_runs, default=0),
        maximum_native_consecutive_space220_tokens=max(run_lengths.values(), default=0),
        raw_space_run_length_histogram=_histogram(raw_runs),
        native_space220_run_length_histogram=_histogram(run_lengths.values()),
        after_newline_spaces={str(n): dict(
            any_native_token_ending_newline=_successor_table(all_counts[n], labels),
            strict_newline198=_successor_table(strict_counts[n], labels),
            preceding_newline_token_counts={str(i): v for i, v in sorted(newline_ids[n].items())})
            for n in sorted(requested)})


def analyze(root):
    """Authenticate inputs before/after and report original train/test layout."""
    root = Path(root).resolve(strict=True)
    record = paired_word_cases._record
    sources = [record(m.__file__) for m in (checkpoint, paired_training, paired_word_cases)]
    sources.append(record(__file__))
    manifest_record = record(root/'manifest.json')
    manifest = json.loads((root/'manifest.json').read_text())
    cases_record = record(root/'word_cases/cases.json')
    cases = json.loads((root/'word_cases/cases.json').read_text())
    if (manifest.get('format') != 'pluto-paired-corpus-training-v1'
            or manifest.get('replacement') != {'from': 'Exeunt', 'to': 'Nuveth', 'case_sensitive': True}
            or cases.get('format') != 'pluto-paired-word-cases-v1'
            or cases.get('manifest') != manifest_record):
        raise ValueError('frozen word cases do not authenticate the expected paired experiment')
    used = {item['path']: item for item in (manifest_record, cases_record)}
    expected = {f'{domain}.{split}' for domain in ('original', 'replacement')
                for split in ('full', 'training', 'test')}
    if set(manifest['inputs']) != expected:
        raise ValueError('expected six frozen native corpus exports')
    inputs = {name: paired_word_cases._load_input(item, used)
              for name, item in sorted(manifest['inputs'].items())}
    for split in ('full', 'training', 'test'):
        a, b = (inputs[f'{domain}.{split}'] for domain in ('original', 'replacement'))
        if paired_training.verify_alignment(a['text'], b['text'], a['tokens'], b['tokens'],
                a['offsets'], b['offsets']) != manifest['alignment'][split]:
            raise ValueError('native alignment disagrees with frozen manifest')
    split_byte = manifest['split_byte']
    if type(split_byte) is not int or split_byte <= 0:
        raise ValueError('invalid split byte')
    for domain in ('original', 'replacement'):
        full, train, test = (inputs[f'{domain}.{split}'] for split in ('full', 'training', 'test'))
        if (len(train['text']) != split_byte or not test['text']
                or train['text']+test['text'] != full['text']
                or not np.array_equal(np.concatenate((train['tokens'], test['tokens'])), full['tokens'])
                or not np.array_equal(np.concatenate((train['offsets'], test['offsets'][1:]+split_byte)),
                                      full['offsets'])):
            raise ValueError('full exports disagree with the claimed split')
    reports = {}
    for split in ('training', 'test'):
        data = inputs[f'original.{split}']
        reports[split] = statistics(data['tokens'], data['offsets'], data['text'])
        if reports[split]['exact_exeunt_count'] != manifest['alignment'][split]['replacements']:
            raise ValueError('text occurrence count differs from replacement alignment')
    selected = {}
    for case in cases['cases']:
        if case['kind'] != 'word':
            continue
        split, index = case['split'], case['occurrence_index']
        if split not in reports or type(index) is not int or index < 0:
            raise ValueError('invalid selected word-case coordinate')
        occurrences = manifest['alignment'][split]['occurrences']
        if index >= len(occurrences):
            raise ValueError('selected word occurrence is outside alignment')
        occurrence = occurrences[index]
        if case['target_source']['token_start'] != occurrence['token_start']:
            raise ValueError('selected word token coordinate differs from alignment')
        layout = word_layout(inputs[f'original.{split}']['text'], occurrence['byte_start'])
        base = 0 if split == 'training' else split_byte
        selected[(split, index)] = dict(split=split, occurrence_index=index,
            token_start=occurrence['token_start'], full_word_byte_start=base+occurrence['byte_start'],
            **layout)
    for item in [*used.values(), *sources]:
        if record(item['path']) != item:
            raise ValueError('input or analysis source changed during indentation audit')
    return dict(
        format='pluto-paired-indentation-statistics-v1', complete=True,
        inference_performed=False, gpu_work_performed=False, goal_completion_claimed=False,
        provenance=list(used.values()), sources=sources, split_byte=split_byte,
        full_and_split_exports_identical=True, statistics=reports,
        selected_word_source_layouts=[selected[k] for k in sorted(selected)],
        definitions={
            'indentation_only': 'Every byte after the preceding newline and before exact-case Exeunt is a space.',
            'total_indentation_spaces': 'Literal preceding spaces, including the leading space inside the first word token.',
            'inline': 'Some preceding non-space content occurs on this line; includes brackets, not just dialogue.',
            'word_byte_column': 'Zero-based byte offset from the line start, not necessarily a count of spaces.',
            'conditioning_opportunities': 'Native newline boundary, followed by N standalone space220 tokens, with a next token available.',
            'newline_policies': 'Both any native token ending in newline and strict token198 are reported separately.',
            'zero_denominator': 'Selected-token frequencies are None; an empty successor table is not evidence of probability zero.',
            'selected_word_source_layouts': 'Deduplicated corpus-source occurrences, not proof of matching evaluation/training window positions.'},
        limitations=[
            'Corpus statistics only; no historical trace, learned weights, inference, or causal mechanism is measured.',
            'No native tokenizer or saved manifest command is executed.',
            'Frequencies concern distinct corpus positions, not random-window training exposures or model probabilities.',
            'Splits are counted separately; a context or target is never supplied across the split boundary.',
            'A generated context absent from this corpus may involve generalization, not an observed exact formatting pattern.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    paired_word_cases._write_json(args.output, analyze(args.root))


if __name__ == '__main__':
    main()
