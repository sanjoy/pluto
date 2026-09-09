"""Test frozen delta-ordering paths against exact native corpus token spans.

The extractor never sees this corpus. Verification retains all four arms and
all candidates, including bad predictions. Matching common token pairs is not
evidence of an ordered passage decoder or a unique text-storage address.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import delta_ordering as extraction
from .delta_followthrough_verify import read_finite_json
from .late_mlp_paths import file_record, write_exclusive
from .restart_delta_verify import check_record, load_native
from .verify import CorpusIndex, verify_candidates


def seed_matches(index, candidate):
    """Measure the original seed's two neighbors separately from remote edges."""
    ids, position = candidate['token_ids'], candidate['seed_position']
    if not 0 < position < len(ids) - 1 or ids[position] != candidate['seed_id']:
        raise ValueError('seed position does not identify the original seed')
    before = index.pair_positions(ids[position - 1], ids[position])
    after = index.pair_positions(ids[position], ids[position + 1])
    triple = before[before + 2 < len(index.tokens)]
    triple = triple[index.tokens[triple + 2] == ids[position + 1]]
    return {'predecessor_pair_occurrences': len(before),
            'successor_pair_occurrences': len(after),
            'seed_triple_occurrences': len(triple),
            'seed_triple_token_starts': triple.tolist()}


def run(frozen_path, corpus, tokens, offsets, tokenizer_dir, output):
    frozen_path, output = Path(frozen_path).resolve(), Path(output).resolve()
    if output.exists() or output.with_suffix('.start.json').exists():
        raise FileExistsError(output)
    frozen = read_finite_json(frozen_path)
    if (frozen.get('complete') is not True or frozen.get('stage') !=
            'frozen_weight_only_directional_ordering_heuristic' or
            frozen.get('candidate_count') != 512 or frozen.get('arms') != list(extraction.ARMS)):
        raise ValueError('incomplete or unexpected ordering freeze')
    records = [file_record(frozen_path), *frozen['sources'].values(), *frozen['files'].values()]
    for record in records:
        check_record(record)
    candidates = [extraction.strict_json(line) for line in
                  Path(frozen['files']['candidates']['path']).read_text().splitlines()]
    if Counter(c['method'] for c in candidates) != Counter({arm: 128 for arm in extraction.ARMS}):
        raise ValueError('candidate denominators differ from fixed protocol')
    if any(len(c['token_ids']) != 8 or c['seed_position'] != 3 for c in candidates):
        raise ValueError('candidate path geometry changed')
    sources = {name: file_record(Path(__file__).with_name(name)) for name in
               ('delta_ordering_verify.py', 'delta_ordering.py', 'restart_delta_verify.py',
                'verify.py', 'delta_followthrough_verify.py', 'late_mlp_paths.py')}
    write_exclusive(output.with_suffix('.start.json'), {
        'created_utc_before_corpus_access': datetime.now(timezone.utc).isoformat(),
        'ordering_freeze': records[0], 'sources': sources,
        'candidates': frozen['files']['candidates']})
    input_paths = dict(corpus=corpus, tokens=tokens, offsets=offsets,
                       tokenizer=Path(tokenizer_dir) / 'tokenizer.json')
    inputs = {key: file_record(path) for key, path in input_paths.items()}
    raw, ids, byte_offsets, vocabulary, boundary, prefix = load_native(*input_paths.values())
    report = verify_candidates(ids, candidates, 50257, seed=17)
    index = CorpusIndex(ids, 50257)
    seed_summaries = {}
    for record in report['results']:
        candidate = record['candidate']
        match = seed_matches(index, candidate)
        record['seed_neighbors'] = match
        token_ids = candidate['token_ids']
        candidate_bytes = b''.join(vocabulary[t] for t in token_ids)
        record['display'] = {'raw_hex': candidate_bytes.hex(),
                             'text': candidate_bytes.decode('utf-8', errors='replace')}
        for key in ('real', 'shuffled_labels'):
            longest = record[key]['longest_match']
            if longest['length']:
                start, end = int(byte_offsets[longest['corpus_token_start']]), int(byte_offsets[longest['corpus_token_end']])
                longest.update(byte_start=start, byte_end=end, raw_hex=raw[start:end].hex(),
                               text=raw[start:end].decode('utf-8', errors='replace'))
        counts = seed_summaries.setdefault(candidate['method'], {
            'candidates': 0, 'predecessor_pair_hits': 0, 'successor_pair_hits': 0, 'seed_triple_hits': 0})
        counts['candidates'] += 1
        for key, count in (('predecessor_pair_hits', 'predecessor_pair_occurrences'),
                           ('successor_pair_hits', 'successor_pair_occurrences'),
                           ('seed_triple_hits', 'seed_triple_occurrences')):
            counts[key] += int(match[count] > 0)
    for record in [*records, *sources.values(), *inputs.values()]:
        check_record(record)
    report.update(complete=True, stage='verification_of_frozen_directional_delta_paths',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  ordering_freeze=records[0], sources=sources, inputs=inputs,
                  seed_neighbors_by_method=seed_summaries,
                  native_export_every_byte_validated=True,
                  current_split=dict(byte_boundary=boundary, prefix_tokens=prefix),
                  all_input_source_and_candidate_identities_unchanged=True,
                  limitation='Exact corpus matches verify candidates, not historical membership or gradient inversion.')
    write_exclusive(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen', 'corpus', 'tokens', 'offsets', 'tokenizer-dir', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run(args.frozen, args.corpus, args.tokens, args.offsets, args.tokenizer_dir, args.output)


if __name__ == '__main__':
    main()
