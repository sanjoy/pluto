"""Verify frozen corpus-assisted retrieval against conditional sampler replay.

This stage never changes a retrieved window. The scalar sampler is separate
from the localization code; neither sampler starts nor this module are inputs
to retrieval. Token-position overlap is not independent historical provenance.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .late_mlp_paths import file_record, write_exclusive
from .delta_followthrough_verify import read_finite_json
from .replay_sampler_test import libstdcxx13_starts


SEEDS = (17, *range(10000, 11000))


def merge_intervals(intervals):
    """Return the sorted disjoint union of nonnegative half-open intervals."""
    output = []
    for start, end in sorted(intervals):
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError('invalid half-open interval')
        if output and start <= output[-1][1]:
            output[-1][1] = max(output[-1][1], end)
        else:
            output.append([start, end])
    return output


def replay_union_arrays(starts, count, width):
    """Padded endpoints permit exact vectorized union-intersection counts."""
    starts = np.asarray(starts)
    if (starts.ndim != 2 or starts.dtype.kind not in 'iu' or
            not 0 < count <= starts.shape[1] or width <= 0):
        raise ValueError('invalid replay geometry')
    left = np.zeros((len(starts), count), dtype=np.int64)
    right = np.zeros_like(left)
    for index, row in enumerate(starts):
        merged = merge_intervals([(int(s), int(s) + width) for s in row[:count]])
        if merged:
            left[index, :len(merged)], right[index, :len(merged)] = np.array(merged).T
    return left, right


def overlap_metrics(retrieved, target_left, target_right, total_tokens):
    """Count token POSITIONS once, regardless of repeated/overlapping windows.

    Prefix sums represent the retrieved union, not token identity frequency.
    A zero-length retrieval has undefined precision (None), and zero recall.
    The target arrays must already describe disjoint unions, including their
    harmless zero-length padding entries.
    """
    merged = merge_intervals(retrieved)
    if merged and merged[-1][1] > total_tokens:
        raise ValueError('retrieval outside corpus')
    left, right = np.asarray(target_left), np.asarray(target_right)
    if (left.ndim != 2 or left.shape != right.shape or
            left.dtype.kind not in 'iu' or right.dtype.kind not in 'iu' or
            np.any(left < 0) or np.any(right < left) or np.any(right > total_tokens)):
        raise ValueError('invalid target endpoint arrays')
    # Check the caller's disjointness precondition, excluding zero padding.
    for lrow, rrow in zip(left, right):
        active = rrow > lrow
        lrow, rrow = lrow[active], rrow[active]
        if len(lrow) > 1 and np.any(lrow[1:] < rrow[:-1]):
            raise ValueError('target intervals overlap or are unordered')
    coverage = np.zeros(total_tokens + 1, dtype=np.int32)
    for start, end in merged:
        coverage[start + 1:end + 1] = 1
    np.cumsum(coverage, out=coverage)
    intersection = (coverage[right] - coverage[left]).sum(axis=1)
    target_size = (right - left).sum(axis=1)
    if np.any(target_size == 0):
        raise ValueError('empty replay target')
    retrieved_size = sum(end - start for start, end in merged)
    return {
        'retrieved_union_tokens': retrieved_size,
        'target_union_tokens': target_size.tolist(),
        'intersection_tokens': intersection.tolist(),
        'precision': ((intersection / retrieved_size).tolist() if retrieved_size
                      else [None] * len(intersection)),
        'recall': (intersection / target_size).tolist(),
    }


def summarize(values):
    """Descriptive seed comparisons, not calibrated p-values."""
    actual, alternatives = values[0], values[1:]
    if actual is None:
        return {'actual': None, 'alternative_mean': None, 'alternative_max': None}
    return {'actual': actual, 'alternative_mean': float(np.mean(alternatives)),
            'alternative_max': max(alternatives),
            'alternatives_strictly_greater': sum(x > actual for x in alternatives),
            'alternatives_equal': sum(x == actual for x in alternatives)}


def check(record):
    if file_record(record['path']) != record:
        raise ValueError(f'changed artifact: {record["path"]}')


def run(localization_path, output):
    localization_path, output = Path(localization_path).resolve(), Path(output).resolve()
    if output.exists() or output.with_suffix('.starts.npz').exists():
        raise FileExistsError(output)
    identity = file_record(localization_path)
    result = read_finite_json(localization_path)
    if (result.get('complete') is not True or result.get('stage') !=
            'corpus_assisted_delta_localization_not_sequence_decoding' or
            result.get('window_length') != 1025):
        raise ValueError('not a complete expected localization')
    frozen_path = localization_path.parent / 'frozen.json'
    frozen = read_finite_json(frozen_path)
    if (frozen.get('complete') is not True or frozen.get('stage') !=
            'corpus_assisted_localization_freeze_before_replay' or
            frozen['files']['localization'] != identity or
            frozen['sources'] != result['sources'] or frozen['inputs'] != result['inputs'] or
            frozen['upstream_freezes'] != result['upstream_freezes'] or
            frozen.get('no_sampler_access') is not True):
        raise ValueError('missing or inconsistent final retrieval freeze')
    records = [identity, file_record(frozen_path), *frozen['files'].values(),
               *result['sources'].values(), *result['inputs'].values(),
               *result['upstream_freezes'].values()]
    for name, row in result['lists'].items():
        if frozen['files'].get('scores.' + name) != row['score_file']:
            raise ValueError('score file differs from final retrieval freeze')
    for record in records:
        check(record)
    sources = {name: file_record(Path(__file__).with_name(name)) for name in
               ('delta_localization_verify.py', 'replay_sampler_test.py', 'late_mlp_paths.py',
                'delta_followthrough_verify.py')}
    write_exclusive(output.with_suffix('.start.json'), {
        'created_utc_before_replay': datetime.now(timezone.utc).isoformat(),
        'localization': identity, 'sources': sources, 'seeds': list(SEEDS),
        'scope': 'conditional replay verification of fixed corpus-assisted retrieval'})
    corpus = result['corpus']
    starts = np.array([libstdcxx13_starts(corpus['prefix_tokens'], 1024, 100, seed)
                       for seed in SEEDS], dtype='<i8')
    starts_path = output.with_suffix('.starts.npz')
    with starts_path.open('xb') as stream:
        np.savez_compressed(stream, seeds=np.asarray(SEEDS, dtype='<i8'), starts=starts)
    unions = {count: replay_union_arrays(starts, count, 1025) for count in (10, 100)}
    lists = {}
    for index, (name, row) in enumerate(result['lists'].items()):
        with np.load(row['score_file']['path'], allow_pickle=False) as arrays:
            scores = arrays[row['score_array_key']]
        if (scores.shape != (corpus['total_tokens'] - 1024,) or scores.dtype != np.uint16
                or scores.max(initial=0) > len(row['token_ids'])):
            raise ValueError('invalid frozen score array')
        selected = row['selected_windows']
        intervals = [(window['start'], window['end']) for window in selected]
        if any(end - start != 1025 for start, end in intervals):
            raise ValueError('invalid selected width')
        if sum(e - s for s, e in merge_intervals(intervals)) != 1025 * len(intervals):
            raise ValueError('selected windows overlap')
        if any(int(scores[w['start']]) != w['score'] for w in selected):
            raise ValueError('selected score differs from frozen array')
        measurements = {}
        for budget in (10, 100):
            for count, endpoints in unions.items():
                value = overlap_metrics(intervals[:budget], *endpoints, corpus['total_tokens'])
                value['summary'] = {field: summarize(value[field]) for field in
                                    ('intersection_tokens', 'precision', 'recall')}
                measurements[f'top_{budget}_vs_first_{count}'] = value
        histogram = np.bincount(scores, minlength=len(row['token_ids']) + 1)
        below_or_equal = np.cumsum(histogram)
        at_actual = scores[starts[0]]
        per_window = []
        actual = starts[0]
        for w in selected:
            intersections = np.maximum(0, np.minimum(w['end'], actual + 1025) -
                                       np.maximum(w['start'], actual))
            per_window.append({
                'rank': w['rank'], 'start': w['start'],
                'max_intersection_first_10': int(intersections[:10].max()),
                'max_intersection_first_100': int(intersections.max()),
                'best_draw_first_10': int(np.argmax(intersections[:10])) + 1,
                'best_draw_first_100': int(np.argmax(intersections)) + 1})
        lists[name] = {
            'measurements': measurements, 'selected_window_overlap': per_window,
            'actual_replay_start_scores': at_actual.tolist(),
            'actual_replay_start_strictly_greater_score_counts':
                (len(scores) - below_or_equal[at_actual]).tolist(),
            'actual_replay_start_equal_score_counts': histogram[at_actual].tolist()}
        if index % 10 == 0:
            print(f'verified {index + 1}/{len(result["lists"])} lists', flush=True)
    for record in records + list(sources.values()):
        check(record)
    report = {'schema_version': 1, 'complete': True,
              'stage': 'conditional_replay_of_corpus_assisted_localization',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'localization': identity, 'sources': sources,
              'sampler_starts': file_record(starts_path), 'seeds': list(SEEDS),
              'lists': lists, 'input_identities_unchanged': True,
              'limitations': ['Retrieval used the corpus; not de novo text decoding.',
                              'Replay assumptions are not historical batch provenance.',
                              'Ten/hundred windows represent possible batch sizes one/ten.',
                              'Alternative-seed comparisons are descriptive, not p-values.']}
    write_exclusive(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--localization', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.localization, args.output)


if __name__ == '__main__':
    main()
