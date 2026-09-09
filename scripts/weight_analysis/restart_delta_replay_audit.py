"""Independent scalar audit of the frozen checkpoint-delta replay verification.

This does not use the verifier's NumPy masks/counting or the C++ sampler.
It uses Python sets, Counter, explicit little-endian unpacking, and the scalar
MT19937-64/libstdc++ oracle from the sampler tests. It verifies the report,
not the historical training state and not a reconstructed text sequence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import struct

from .replay_sampler_test import libstdcxx13_starts


def identity(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def bag(tokens, starts, context):
    """Both input and shifted target positions; multiplicities discarded."""
    if type(context) is not int or not 0 < context < len(tokens):
        raise ValueError("context must be a positive integer smaller than corpus")
    result = set()
    for start in starts:
        if type(start) is not int or not 0 <= start < len(tokens) - context:
            raise ValueError("window outside corpus")
        result.update(tokens[start:start + context + 1])
    return result


def audit(report_path, output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    report_identity = identity(report_path)
    report = json.loads(Path(report_path).read_text())
    if report.get('complete') is not True or report.get('native_export_every_byte_validated') is not True:
        raise ValueError("incomplete verification")
    expected_names = {'shared_endpoint_norm'}
    for boundary in (580, 1780, 4280, 7230, 8160):
        expected_names.add(f'endpoint_norm_{boundary}')
        for kind in ('boundary', 'ordinary_before', 'ordinary_after'):
            for variant in ('raw', 'adjusted'):
                expected_names.add(f'{kind}_{variant}_{boundary}')
                expected_names.add(f'shared_{kind}_{variant}')
    if set(report.get('results', {})) != expected_names:
        raise ValueError("expected all 42 frozen comparison arms")
    checked_files = [report_identity, *report['inputs'].values()]
    for record in checked_files:
        if identity(record['path']) != record:
            raise ValueError("input identity differs")
    payload = Path(report['inputs']['native_tokens']['path']).read_bytes()
    tokens = [token for (token,) in struct.iter_unpack('<I', payload)]
    if len(tokens) != report['split']['corpus_tokens'] or any(t >= 50257 for t in tokens):
        raise ValueError("invalid native token export")
    prefix_count = report['split']['prefix_tokens']
    if (type(prefix_count) is not int or not 1024 < prefix_count < len(tokens)
            or report['split']['native_token_boundary'] != prefix_count
            or report['split']['suffix_tokens'] != len(tokens) - prefix_count):
        raise ValueError("inconsistent native prefix/suffix split")
    prefix = tokens[:prefix_count]
    counts = {'full': Counter(tokens), 'current_prefix': Counter(prefix),
              'current_suffix': Counter(tokens[prefix_count:])}
    replay = report['conditional_replay']
    if replay['context'] != 1024 or replay['window_counts'] != [10, 100]:
        raise ValueError("unexpected replay settings")
    for label, first, count in (('actual', 17, 1), ('alternatives', 10000, 1000)):
        if (replay[label]['seed_start'] != first or replay[label]['seed_count'] != count
                or len(replay[label]['starts_per_seed']) != count):
            raise ValueError("unexpected sampler seed labels/counts")
    saved_starts = replay['actual']['starts_per_seed'] + replay['alternatives']['starts_per_seed']
    seeds = [17, *range(10000, 11000)]
    # Independently regenerate every draw; do not trust the report's starts.
    starts = [libstdcxx13_starts(prefix_count, 1024, 100, seed) for seed in seeds]
    if starts != saved_starts:
        raise ValueError("independent sampler differs")
    overlap_checks = []
    for window_count in (10, 100):
        bags = [bag(prefix, row[:window_count], 1024) for row in starts]
        if list(map(len, bags)) != replay['distinct_token_ids_per_seed'][str(window_count)]:
            raise ValueError("replay vocabulary sizes differ")
        for name, variants in report['results'].items():
            if set(variants) != {'real', 'identity_shuffled'}:
                raise ValueError("missing or extra identity-control variants")
            for variant, values in variants.items():
                ids = values['token_ids_in_frozen_rank_order']
                if (len(ids) != 128 or any(type(t) is not int or not 0 <= t < 50257 for t in ids)
                        or len(set(ids)) != 128):
                    raise ValueError("candidate IDs are not 128 distinct tokens")
                overlap = [len(set(ids) & token_bag) for token_bag in bags]
                claimed = values['conditional_replay'][str(window_count)]
                if overlap != [claimed['seed17_overlap'], *claimed['alternative_overlap_counts_in_seed_order']]:
                    raise ValueError("bag intersections differ: " + name)
                alternatives = overlap[1:]
                summaries = {
                    'alternative_minimum': min(alternatives),
                    'alternative_maximum': max(alternatives),
                    'alternative_mean': statistics.mean(alternatives),
                    'alternative_median': statistics.median(alternatives),
                    'alternative_histogram': {str(k): v for k, v in Counter(alternatives).items()},
                    'alternatives_strictly_greater_than_seed17': sum(x > overlap[0] for x in alternatives),
                    'alternatives_equal_to_seed17': alternatives.count(overlap[0]),
                    'alternatives_greater_or_equal_to_seed17': sum(x >= overlap[0] for x in alternatives),
                    'seed17_descriptive_rank_among_all_seeds': 1 + sum(x > overlap[0] for x in alternatives),
                }
                if any(claimed[key] != value for key, value in summaries.items()):
                    raise ValueError("overlap summaries differ: " + name)
                for label, frequency in counts.items():
                    individual = [frequency[token] for token in ids]
                    saved = values['corpus_frequency'][label]
                    if (individual != saved['counts_in_rank_order']
                            or sum(x > 0 for x in individual) != saved['ranked_token_ids_present']
                            or sum(individual) != saved['selected_token_occurrences']):
                        raise ValueError("corpus frequencies differ: " + name)
                overlap_checks.append({'name': name, 'variant': variant,
                                       'windows': window_count, 'seed17_overlap': overlap[0],
                                       'alternatives_mean': summaries['alternative_mean'],
                                       'alternatives_maximum': summaries['alternative_maximum']})
    for record in checked_files:
        if identity(record['path']) != record:
            raise ValueError("input changed during audit")
    result = {'complete': True, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'report': report_identity, 'inputs': report['inputs'],
              'sources': {name: identity(Path(__file__).with_name(name)) for name in
                          ('restart_delta_replay_audit.py', 'replay_sampler_test.py')},
              'independent_sampler_rows_exact': len(starts), 'draws_per_row': 100,
              'independent_overlap_lists_exact': len(overlap_checks),
              'all_corpus_counts_in_rank_order_exact': True, 'checks': overlap_checks,
              'limitations': ['Audits conditional replay, not historical sampler state.',
                              'No token order reconstruction.',
                              'Native export byte validation and extraction ranking authentication are separate checks.']}
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({key: result[key] for key in
                      ('independent_sampler_rows_exact', 'independent_overlap_lists_exact',
                       'all_corpus_counts_in_rank_order_exact')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.report, args.output)


if __name__ == '__main__':
    main()
