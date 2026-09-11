"""Independent numerical replay of frozen restart scores, without corpus use.

Does not import restart_delta or its numerical helpers. Uses scalar math.log,
Python sorting/grouping and heap selection for score/rank audits; independently
recomputes eight embedding endpoints and three interval activities. The strict
archive reader is shared; archive parsing is not an independent implementation.
"""

import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, '/home/ubuntu/code/pluto')
from scripts.weight_analysis.checkpoint_archive import read_embedding


RUN = Path('/tmp/pluto-restart-delta.fZs3LQ/run')
BOUNDARIES = [580, 1780, 4280, 7230, 8160]
VOCABULARY = 50257
K = 128
ERRORS = []
COMPARISONS = []


def file_record(path):
    path = Path(path).resolve()
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(8 * 1024 * 1024):
            checksum.update(data)
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': checksum.hexdigest()}


def check(condition, message):
    if not condition:
        ERRORS.append(message)


def compare(name, actual, expected, atol=1e-12, rtol=1e-12):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        ERRORS.append(name + ': shape mismatch')
        return
    difference = np.abs(actual - expected)
    maximum = float(difference.max()) if difference.size else 0.
    tolerance = atol + rtol * np.abs(expected)
    passed = bool(np.all(difference <= tolerance))
    COMPARISONS.append({'name': name, 'shape': list(actual.shape),
                        'maximum_absolute_error': maximum, 'absolute_tolerance': atol,
                        'relative_tolerance': rtol, 'passed': passed})
    check(passed, name + ': numerical mismatch')


def scalar_logs(values):
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    median = ordered[count // 2] if count % 2 else (ordered[count // 2 - 1] + ordered[count // 2]) / 2
    base = median if median else ordered[-1]
    if base == 0:
        return np.zeros(count, dtype=np.float64)
    log_base = math.log(base)
    return np.array([math.log(max(float(value), base * 1e-12)) - log_base
                     for value in values], dtype=np.float64)


def grouped_ranks(values):
    ordered_ids = sorted(range(len(values)), key=lambda index: float(values[index]))
    result = np.empty(len(values), dtype=np.float64)
    consumed = 0
    for _, group in itertools.groupby(ordered_ids, key=lambda index: float(values[index])):
        ids = list(group)
        percentile = (consumed + (len(ids) - 1) / 2 + 0.5) / len(values)
        for index in ids:
            result[index] = percentile
        consumed += len(ids)
    return result


def top_ids(score):
    return heapq.nsmallest(K, range(len(score)),
                          key=lambda index: (-float(score[index]), index))


def row_norm(values):
    # Different reduction implementation than the extractor's np.linalg.norm.
    return np.sqrt(np.einsum('ij,ij->i', values, values, optimize=False))


def centered_activities(before, after):
    delta = after - before
    # fsum per column reduces in a different, compensated order from ndarray.mean.
    mean_before = np.array([math.fsum(before[:, column]) / len(before)
                            for column in range(before.shape[1])])
    mean_delta = np.array([math.fsum(delta[:, column]) / len(delta)
                           for column in range(delta.shape[1])])
    center = before - mean_before
    centered_delta = delta - mean_delta
    denominator = float(np.einsum('ij,ij->', center, center, optimize=False))
    numerator = float(np.einsum('ij,ij->', center, centered_delta, optimize=False))
    radial = numerator / denominator if denominator else 0.
    residual = centered_delta - radial * center
    return {'raw': row_norm(delta), 'adjusted': row_norm(residual)}, {
        'shared_translation': mean_delta,
        'fitted_radial_coefficient_not_authenticated_decay': radial,
        'raw_squared_norm': float(np.einsum('ij,ij->', delta, delta, optimize=False)),
        'adjusted_squared_norm': float(np.einsum('ij,ij->', residual, residual, optimize=False)),
        'centered_before_squared_norm': denominator,
    }


def main():
    start = time.monotonic()
    frozen_record = file_record(RUN / 'frozen.json')
    check(frozen_record['sha256'] == 'a16d908bf7b496d6fbabf601d32d93e1e2bf6af5b83e2a428eec2825a59211ca',
          'unexpected frozen manifest identity')
    frozen = json.loads((RUN / 'frozen.json').read_text())
    candidates = json.loads((RUN / 'candidates.json').read_text())['candidate_lists']
    artifact_records = {name: file_record(item['path']) for name, item in frozen['files'].items()}
    for name, item in frozen['files'].items():
        check(artifact_records[name] == item, 'artifact hash mismatch: ' + name)
    source_records = {name: file_record(item['path']) for name, item in frozen['sources'].items()}
    for name, item in frozen['sources'].items():
        check(source_records[name] == item, 'source hash mismatch: ' + name)
    check(frozen['complete'] is True, 'freeze incomplete')
    check(len(candidates) == 42, 'wrong candidate list count')

    # Tiny analytically known fixtures check the independently authored helpers.
    compare('helper_scalar_logs', scalar_logs([1., 2., 4.]),
            [-math.log(2), 0., math.log(2)], atol=0, rtol=0)
    compare('helper_grouped_midrank', grouped_ranks([5., 1., 5., 3.]),
            [.75, .125, .75, .375], atol=0, rtol=0)
    check(top_ids(np.zeros(VOCABULARY)) == list(range(K)), 'helper tie order incorrect')

    with np.load(RUN / 'scores.npz', allow_pickle=False) as scores:
        stored = {key: scores[key] for key in scores.files}
    check(set(stored) == set(candidates) | {'identity_permutation'}, 'unexpected score keys')
    permutation = np.random.Generator(np.random.PCG64(20260909)).permutation(VOCABULARY)
    compare('identity_permutation', permutation, stored['identity_permutation'], atol=0, rtol=0)
    activity_vectors = {}
    recomputed = {}
    components = {}
    for boundary in BOUNDARIES:
        intervals = []
        for earlier in range(boundary - 30, boundary + 40, 10):
            key = f'interval_{earlier}_{earlier + 10}'
            with np.load(RUN / (key + '.npz'), allow_pickle=False) as activity:
                check(set(activity.files) == {'raw', 'adjusted'}, key + ': unexpected NPZ keys')
                activity_vectors[key] = {variant: activity[variant] for variant in ('raw', 'adjusted')}
            for variant, vector in activity_vectors[key].items():
                check(vector.shape == (VOCABULARY,) and vector.dtype == np.dtype('float64')
                      and np.isfinite(vector).all() and np.all(vector >= 0),
                      key + ':' + variant + ': invalid activity')
            intervals.append(key)
        for variant in ('raw', 'adjusted'):
            logs = [scalar_logs(activity_vectors[key][variant]) for key in intervals]
            for kind, center in [('ordinary_before', 1), ('boundary', 3), ('ordinary_after', 5)]:
                score = np.array([float(c) - 0.5 * (float(a) + float(b))
                                  for a, c, b in zip(logs[center - 1], logs[center], logs[center + 1])])
                key = f'{kind}_{variant}_{boundary}'
                recomputed[key] = score
                components.setdefault((kind, variant), []).append(score)
                compare(key, score, stored[key], atol=5e-14, rtol=0)

    # Eight rereads only: five E_N norm baselines and one complete boundary triplet.
    reread_steps = sorted(set(BOUNDARIES + [570, 590, 600]))
    embeddings = {}
    reread_records = {}
    for step in reread_steps:
        original = frozen['checkpoint_provenance'][str(step)]
        physical, provenance = read_embedding(original['path'], step)
        check(provenance['input_files'] == original['input_files'], f'step{step}: archive identity differs')
        check(provenance['embedding_sha256'] == original['embedding_sha256'],
              f'step{step}: embedding identity differs')
        reread_records[str(step)] = {'input_files': provenance['input_files'],
                                    'embedding_sha256': provenance['embedding_sha256'],
                                    'input_files_unchanged_during_read': provenance['input_files_unchanged_during_read']}
        logical = physical[:VOCABULARY].astype(np.float64)
        if step in BOUNDARIES:
            score = row_norm(logical)
            key = f'endpoint_norm_{step}'
            recomputed[key] = score
            compare(key, score, stored[key], atol=5e-14, rtol=5e-14)
        if step in (570, 580, 590, 600):
            embeddings[step] = logical
        print('Independent audit reread step', step, flush=True)

    for before, after in [(570, 580), (580, 590), (590, 600)]:
        key = f'interval_{before}_{after}'
        activity, stats = centered_activities(embeddings[before], embeddings[after])
        for variant in ('raw', 'adjusted'):
            compare(key + ':' + variant, activity[variant], activity_vectors[key][variant],
                    atol=5e-13, rtol=5e-12)
        recorded_stats = next(item for item in frozen['intervals'] if item['activity_file'] == key)
        for name, value in stats.items():
            compare(key + ':' + name, value, recorded_stats[name], atol=5e-12, rtol=5e-12)
        print('Independent activity reductions checked', key, flush=True)

    for (kind, variant), rows in components.items():
        ranks = [grouped_ranks(row) for row in rows]
        score = np.array([sum(float(value) for value in column) / 5 for column in zip(*ranks)])
        key = f'shared_{kind}_{variant}'
        recomputed[key] = score
        compare(key, score, stored[key], atol=2e-15, rtol=0)
    ranks = [grouped_ranks(recomputed[f'endpoint_norm_{step}']) for step in BOUNDARIES]
    recomputed['shared_endpoint_norm'] = np.array([sum(float(value) for value in column) / 5
                                                 for column in zip(*ranks)])
    compare('shared_endpoint_norm', recomputed['shared_endpoint_norm'],
            stored['shared_endpoint_norm'], atol=2e-15, rtol=0)

    checked_lists = []
    for key, candidate in candidates.items():
        ids = top_ids(recomputed[key])
        check(ids == candidate['token_ids'], key + ': ranked IDs mismatch')
        check([int(permutation[index]) for index in ids] == candidate['identity_shuffled_token_ids'],
              key + ': shuffled IDs mismatch')
        compare(key + ':selected_scores', recomputed[key][ids], candidate['scores'], atol=5e-14, rtol=5e-14)
        if key.startswith('shared_') and key != 'shared_endpoint_norm':
            suffix = key.removeprefix('shared_')
            kind, variant = suffix.rsplit('_', 1)
            rows = components[(kind, variant)]
            positive = [sum(row[index] > 0 for row in rows) for index in ids]
            check(positive == candidate['positive_component_count'], key + ': positive counts mismatch')
            compare(key + ':component_scores', [[float(row[index]) for index in ids] for row in rows],
                    candidate['component_scores'], atol=5e-14, rtol=0)
        checked_lists.append(key)

    for name, record in artifact_records.items():
        check(file_record(record['path']) == record, 'artifact changed during audit: ' + name)
    for name, record in source_records.items():
        check(file_record(record['path']) == record, 'source changed during audit: ' + name)
    check(file_record(RUN / 'frozen.json') == frozen_record, 'frozen manifest changed during audit')
    result = {'schema': 'restart_delta_independent_numerical_audit_v1',
              'passed': not ERRORS, 'errors': ERRORS, 'error_count': len(ERRORS),
              'no_corpus_or_tokenizer_access': True, 'no_model_execution': True,
              'frozen_manifest': frozen_record, 'artifact_records': artifact_records,
              'source_records': source_records, 'auditor_source': file_record(__file__),
              'checked_activity_file_count': len(activity_vectors),
              'activity_intervals_recomputed_from_weights': [[570, 580], [580, 590], [590, 600]],
              'other_activity_scope': 'All other 32 activity files hash/shape/finiteness checked, not recomputed from weights.',
              'endpoint_norms_recomputed_from_weights': BOUNDARIES,
              'reread_checkpoint_provenance': reread_records,
              'checked_candidate_lists': checked_lists, 'checked_candidate_list_count': len(checked_lists),
              'checked_top_ids': len(checked_lists) * K,
              'checked_shuffled_top_ids': len(checked_lists) * K,
              'complete_score_vectors_recomputed': len(recomputed),
              'comparisons': COMPARISONS,
              'algorithm_scope': 'Independent scalar log, sorted/grouped midrank, heap top-K, fsum means and einsum norms/contractions. Shared strict archive reader; not independent archive parsing. No calls to restart_delta numerical helpers.',
              'elapsed_seconds': time.monotonic() - start,
              'numpy_version': np.__version__}
    output = RUN / 'independent_check.json'
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': file_record(output), 'passed': result['passed'],
                      'error_count': len(ERRORS), 'errors': ERRORS,
                      'elapsed_seconds': result['elapsed_seconds']}, indent=2))
    return 0 if not ERRORS else 1


if __name__ == '__main__':
    raise SystemExit(main())
