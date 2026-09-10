"""Audit local spelling statistics in the frozen Exeunt/Nuveth corpora.

This reads native token IDs and byte offsets, not a substitute tokenizer or
model. Counts concern distinct positions in each split, not random-window
training exposures. They can motivate a short-context spelling hypothesis but
cannot establish that a model implements it, or that any weights store a word.

The exact-case replacement leaves lowercase ``exeunt`` untouched. Its native
`` ex/e/unt`` spelling and the remaining ``Bl/unt`` spellings are counted
explicitly so those shared-piece training examples cannot silently disappear
from the interpretation of the counterfactual experiment.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import re

import numpy as np

from . import checkpoint, paired_training, paired_word_cases


PIECES = (1475, 3109, 68, 2797)
KNOWN_BYTES = {1475: b' Ex', 3109: b'Ex', 68: b'e', 2797: b'unt',
               409: b' ex', 1086: b' Bl', 3629: b'Bl'}
PATTERNS = ((1475, 68), (3109, 68), (68, 2797),
            (1475, 68, 2797), (3109, 68, 2797), (409, 68, 2797),
            (1086, 2797), (3629, 2797))
DOMAINS = ('original', 'replacement')
SPLITS = ('training', 'test')


def _fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def sequence_statistics(token_ids, replacement_starts):
    """Count one split, without inventing a predecessor/successor at its ends.

    ``replacement_starts`` identifies three-token spans in this split. It is
    geometry only: the caller authenticates their native text/IDs separately.
    Ownership, rather than merely a boolean mask, distinguishes a pattern
    inside one replacement from a pattern crossing two adjacent replacements.
    Empty denominators have probability None, never an invented zero estimate.
    """
    values = list(token_ids)
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_))
           or not 0 <= int(v) < paired_word_cases.VOCAB_SIZE for v in values):
        raise ValueError('expected valid integer native token IDs')
    tokens = np.asarray(values, dtype=np.int64)
    starts = list(replacement_starts)
    if (any(type(s) is not int or not 0 <= s <= len(tokens)-3 for s in starts)
            or any(b < a+3 for a, b in zip(starts, starts[1:]))):
        raise ValueError('replacement spans must be sorted, disjoint token triples')
    owners = np.full(len(tokens), -1, dtype=np.int64)
    for index, start in enumerate(starts):
        owners[start:start+3] = index
    pieces = {}
    for token_id in PIECES:
        positions = np.flatnonzero(tokens == token_id)
        has_successor = positions[positions+1 < len(tokens)]
        successor_counts = Counter(tokens[has_successor+1].tolist())
        predecessor_counts = Counter(tokens[positions[positions > 0]-1].tolist())
        inside = int(np.count_nonzero(owners[positions] >= 0))
        targets = int(np.count_nonzero(positions > 0))
        pieces[str(token_id)] = dict(
            occurrences=len(positions), next_token_target_count=targets,
            next_token_target_frequency=_fraction(targets, max(0, len(tokens)-1)),
            inside_replacement_spans=inside,
            outside_replacement_spans=len(positions)-inside,
            fraction_inside_replacement_spans=_fraction(inside, len(positions)),
            successor_opportunities=len(has_successor),
            successors=[dict(token_id=i, count=n,
                             conditional_frequency=_fraction(n, len(has_successor)))
                        for i, n in sorted(successor_counts.items(), key=lambda x: (-x[1], x[0]))],
            predecessor_opportunities=int(np.count_nonzero(positions > 0)),
            predecessors=[dict(token_id=i, count=n) for i, n in sorted(
                predecessor_counts.items(), key=lambda x: (-x[1], x[0]))])
    patterns = []
    for pattern in PATTERNS:
        width = len(pattern)
        matches = np.ones(max(0, len(tokens)-width+1), dtype=bool)
        prefix = np.ones(len(matches), dtype=bool)
        for offset, token_id in enumerate(pattern):
            matches &= tokens[offset:offset+len(matches)] == token_id
            if offset < width-1:
                prefix &= tokens[offset:offset+len(matches)] == token_id
        positions = np.flatnonzero(matches)
        same_owner = sum(owners[p] >= 0 and np.all(
            owners[p:p+width] == owners[p]) for p in positions)
        outside = sum(np.all(owners[p:p+width] == -1) for p in positions)
        opportunities = int(np.count_nonzero(prefix))
        patterns.append(dict(
            token_ids=list(pattern), occurrences=len(positions),
            inside_one_replacement_span=int(same_owner),
            wholly_outside_replacement_spans=int(outside),
            other_span_overlap=int(len(positions)-same_owner-outside),
            prefix_with_successor_opportunities=opportunities,
            conditional_last_token_frequency=_fraction(len(positions), opportunities)))
    return dict(token_count=len(tokens), next_token_target_count=max(0, len(tokens)-1),
                replacement_span_count=len(starts), pieces=pieces, patterns=patterns)


def _native_details(data, starts, full_byte_base):
    """Attach native spellings and brief examples; byte positions are zero-based."""
    tokens, offsets, text = (data[k] for k in ('tokens', 'offsets', 'text'))
    result = sequence_statistics(tokens, starts)
    occupied = np.zeros(len(tokens), dtype=bool)
    for start in starts:
        occupied[start:start+3] = True
    for token_id, expected in KNOWN_BYTES.items():
        for position in np.flatnonzero(tokens == token_id):
            if text[offsets[position]:offsets[position+1]] != expected:
                raise ValueError('diagnostic token ID has unexpected native byte spelling')

    def example(position):
        begin, end = int(offsets[position]), int(offsets[position+1])
        snippet_begin, snippet_end = max(0, begin-15), min(len(text), end+22)
        return dict(token_index=int(position), token_id=int(tokens[position]),
                    split_byte_start=begin, full_byte_start=full_byte_base+begin,
                    piece_bytes_hex=text[begin:end].hex(),
                    snippet=text[snippet_begin:snippet_end].decode('utf-8', errors='replace'),
                    snippet_full_byte_start=full_byte_base+snippet_begin)

    for token_id in PIECES:
        item = result['pieces'][str(token_id)]
        item['native_piece_bytes_hex'] = KNOWN_BYTES[token_id].hex()
        outside = np.flatnonzero((tokens == token_id) & ~occupied)
        item['outside_examples'] = [example(p) for p in outside[:3]]
        for alternative in item['successors']:
            position = int(np.flatnonzero(tokens == alternative['token_id'])[0])
            raw = text[offsets[position]:offsets[position+1]]
            alternative['native_piece_bytes_hex'] = raw.hex()
            alternative['native_piece_text'] = raw.decode('utf-8', errors='replace')
    # List all surviving e/unt examples. Do not assume every such pair must be
    # lowercase exeunt: report its actual predecessor, which can falsify that.
    positions = np.flatnonzero((tokens[:-1] == 68) & (tokens[1:] == 2797))
    survivors = [int(p) for p in positions if not occupied[p:p+2].any()]
    result['e_unt_outside_replacement_examples'] = [dict(
        **example(p), predecessor_token_id=int(tokens[p-1]) if p else None)
        for p in survivors]
    result['case_insensitive_exeunt_text_counts'] = dict(sorted(Counter(
        match.group().decode('ascii') for match in re.finditer(b'exeunt', text, re.I)).items()))
    result['full_text_byte_base'] = full_byte_base
    return result


def analyze(root):
    """Authenticate the six frozen exports and return a CPU-only report.

    The word-case manifest reference anchors the experiment manifest. Every
    input and analysis dependency is hashed before and after the computation.
    Full exports validate the split only; all statistics use separate training
    and test streams, so no transition is counted across their boundary.
    """
    root = Path(root).resolve(strict=True)
    record = paired_word_cases._record
    sources = [record(module.__file__) for module in
               (checkpoint, paired_training, paired_word_cases)] + [record(__file__)]
    manifest_path = root/'manifest.json'
    manifest_record = record(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('format') != 'pluto-paired-corpus-training-v1'
            or manifest.get('replacement') != dict(
                **{'from': 'Exeunt', 'to': 'Nuveth'}, case_sensitive=True)):
        raise ValueError('expected exact-case Exeunt/Nuveth experiment')
    cases_path = root/'word_cases/cases.json'
    cases_record = record(cases_path)
    cases = json.loads(cases_path.read_text())
    if (cases.get('format') != 'pluto-paired-word-cases-v1'
            or cases.get('manifest') != manifest_record):
        raise ValueError('word-case manifest reference does not authenticate this experiment')
    used = {r['path']: r for r in (manifest_record, cases_record)}
    expected = {f'{domain}.{split}' for domain in DOMAINS
                for split in ('full', *SPLITS)}
    if set(manifest['inputs']) != expected:
        raise ValueError('expected exactly six original/replacement native exports')
    inputs = {name: paired_word_cases._load_input(item, used)
              for name, item in sorted(manifest['inputs'].items())}
    for split in ('full', *SPLITS):
        original, replacement = (inputs[f'{domain}.{split}'] for domain in DOMAINS)
        verified = paired_training.verify_alignment(
            original['text'], replacement['text'], original['tokens'], replacement['tokens'],
            original['offsets'], replacement['offsets'])
        if verified != manifest['alignment'][split]:
            raise ValueError('native replacement alignment differs from frozen manifest')
    boundary = manifest['split_byte']
    if type(boundary) is not int or boundary <= 0:
        raise ValueError('invalid split byte')
    for domain in DOMAINS:
        full, training, test = (inputs[f'{domain}.{split}']
                                for split in ('full', *SPLITS))
        if (len(training['text']) != boundary or not test['text']
                or training['text']+test['text'] != full['text']
                or not np.array_equal(np.concatenate((training['tokens'], test['tokens'])),
                                      full['tokens'])
                or not np.array_equal(np.concatenate((training['offsets'],
                    test['offsets'][1:]+boundary)), full['offsets'])):
            raise ValueError('native full exports do not equal the claimed split concatenation')
    reports = {}
    for split in SPLITS:
        starts = [occurrence['token_start']
                  for occurrence in manifest['alignment'][split]['occurrences']]
        reports[split] = {domain: _native_details(inputs[f'{domain}.{split}'], starts,
                              0 if split == 'training' else boundary) for domain in DOMAINS}
    for item in [*used.values(), *sources]:
        if record(item['path']) != item:
            raise ValueError('input or analysis source changed during spelling audit')
    return dict(
        format='pluto-paired-spelling-statistics-v1', complete=True,
        inference_performed=False, gpu_work_performed=False, goal_completion_claimed=False,
        provenance=list(used.values()), sources=sources,
        runtime=dict(python=platform.python_version(), numpy=np.__version__),
        split_byte=boundary, full_and_split_exports_identical=True,
        statistics=reports,
        denominator_definitions={
            'occurrences': 'Every occurrence in this split, including first/last token.',
            'next_token_target_count': 'tokens[1:]; the first token has no predecessor in this split.',
            'next_token_target_frequency': 'Selected ID target count / (split token count - 1).',
            'successor_opportunities': 'Occurrences before the final token, which have a next token.',
            'conditional_last_token_frequency': 'Pattern count / matching prefix count with room for its last token.',
            'inside_one_replacement_span': 'All pattern positions belong to the same exact-case three-token replacement.',
            'case_insensitive_exeunt_text_counts': 'Literal substring matches, classified by their actual letter case.'},
        limitations=[
            'Corpus statistics only: no model inference, weight effect, causal storage, or memorization is measured.',
            'Distinct corpus positions are counted, not random-window training exposures or completed training steps.',
            'No transition crosses the training/test split; first/last-token denominators remain separate.',
            'Exact-case replacement preserves lowercase exeunt and other words sharing the diagnostic token pieces.',
            'Conditional frequencies are unsmoothed empirical corpus estimates, not model probabilities.',
            'The unt suffix does not specify a following delimiter or prove word termination.',
            'Full exports validate split geometry; only training/test exports contribute to the statistics.'])


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
