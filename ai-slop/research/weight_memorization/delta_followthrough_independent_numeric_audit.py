"""Independent follow-through score/rank audit; no corpus or model access.

Reads only a completed frozen weight-extraction run, its named numerical
artifacts/sources, and complete original checkpoint bytes for final hashing.
Activities are authenticated, not recomputed from embedding payloads here.
Score calculations use scalar logarithms, Python sorting/grouping, and fsum,
not the production numerical helpers. Every discrepancy is retained.
"""

import argparse
import datetime
import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np


BOUNDARIES = (580, 1780, 4280, 7230, 8160)
VOCABULARY = 50257
K = 128


def identity(path):
    path = Path(path).resolve()
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            hasher.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': hasher.hexdigest()}


def scalar_logs(values):
    values = list(map(float, values))
    base = statistics.median(values)
    if base == 0:
        base = max(values)
    if base == 0:
        return [0.] * len(values)
    floor = base * 1e-12
    if floor == 0:
        raise ValueError('declared activity floor underflowed')
    baseline = math.log(base)
    return [math.log(max(value, floor)) - baseline for value in values]


def doubled_midranks(values):
    order = sorted(range(len(values)), key=lambda index: float(values[index]))
    ranks = np.empty(len(values), dtype=np.int64)
    first = 0
    for _, grouped in itertools.groupby(order, key=lambda index: float(values[index])):
        group = list(grouped)
        ranks[group] = 2 * first + len(group)
        first += len(group)
    return ranks


def top_ids(values):
    return heapq.nsmallest(K, range(len(values)), key=lambda index: (-float(values[index]), index))


def audit(run, output):
    run, output = Path(run).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    frozen_identity = identity(run / 'frozen.json')
    frozen = json.loads((run / 'frozen.json').read_text())
    if frozen.get('complete') is not True or frozen.get('stage') != 'delta_followthrough_candidate_freeze_not_sequence_recovery':
        raise ValueError('not a completed follow-through extraction')
    for record in [*frozen['sources'].values(), *frozen['files'].values()]:
        if identity(record['path']) != record:
            raise ValueError('frozen source or output hash mismatch: ' + record['path'])
    source_identity = identity(__file__)
    candidates = json.loads((run / 'candidates.json').read_text())['candidate_lists']
    names = ({f'followthrough_{variant}_{boundary}' for boundary in BOUNDARIES for variant in ('raw', 'adjusted')}
             | {'shared_followthrough_raw', 'shared_followthrough_adjusted'})
    if set(candidates) != names:
        raise ValueError('unexpected candidate list set')
    with np.load(run / 'scores.npz', allow_pickle=False) as arrays:
        stored = {key: arrays[key] for key in arrays.files}
    if set(stored) != names | {'identity_permutation'}:
        raise ValueError('unexpected complete score keys')
    permutation = np.random.Generator(np.random.PCG64(20260909)).permutation(VOCABULARY)
    if not np.array_equal(permutation, stored['identity_permutation']):
        raise ValueError('fixed identity permutation mismatch')
    if scalar_logs([1., 2., 4.]) != [-math.log(2), 0., math.log(2)]:
        raise ValueError('scalar logarithm fixture failed')
    if doubled_midranks([3., 1., 2., 2.]).tolist() != [7, 1, 4, 4]:
        raise ValueError('independent midrank fixture failed')
    if top_ids(np.zeros(VOCABULARY)) != list(range(K)):
        raise ValueError('independent ID tie fixture failed')
    activity_records = []
    independently_computed = {}
    doubled = {}
    for boundary in BOUNDARIES:
        intervals = []
        for before in (boundary + 90, boundary + 100, boundary + 110):
            key = f'interval_{before}_{before + 10}'
            record = frozen['files'][key]
            with np.load(record['path'], allow_pickle=False) as arrays:
                if set(arrays.files) != {'raw', 'adjusted'}:
                    raise ValueError('invalid activity file keys')
                vectors = {name: arrays[name] for name in arrays.files}
            for values in vectors.values():
                if values.shape != (VOCABULARY,) or values.dtype != np.dtype('float64') or not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError('activity shape, dtype, sign, or finiteness mismatch')
            intervals.append(vectors)
            activity_records.append({'name': key, 'file': record, 'shape_each_variant': [VOCABULARY],
                                     'dtype_each_variant': '<f8', 'both_variants_finite_nonnegative': True})
        for variant in ('raw', 'adjusted'):
            left, center, right = [scalar_logs(values[variant]) for values in intervals]
            score = np.array([c - .5 * (a + b) for a, c, b in zip(left, center, right)])
            independently_computed[f'followthrough_{variant}_{boundary}'] = score
    for variant in ('raw', 'adjusted'):
        rows = [independently_computed[f'followthrough_{variant}_{boundary}'] for boundary in BOUNDARIES]
        ranks = [doubled_midranks(row) for row in rows]
        key = f'shared_followthrough_{variant}'
        # Each term is rounded as a percentile before compensated averaging;
        # exact integer sums below separately diagnose mathematical score ties.
        independently_computed[key] = np.array([
            math.fsum(int(rank[index]) / (2 * VOCABULARY) for rank in ranks) / 5
            for index in range(VOCABULARY)])
        doubled[key] = np.sum(ranks, axis=0)
    checks, errors, warnings = [], [], []
    for name, ranking in candidates.items():
        values = stored[name]
        if values.shape != (VOCABULARY,) or values.dtype != np.dtype('float64') or not np.isfinite(values).all():
            raise ValueError('invalid complete stored score')
        oracle = independently_computed[name]
        ids = ranking['token_ids']
        stored_order = top_ids(values)
        independent_order = top_ids(oracle)
        absolute_tolerance = 2e-15 if name.startswith('shared_') else 5e-14
        maximum_error = float(np.max(np.abs(values - oracle)))
        exact_stored = stored_order == ids and values[ids].tolist() == ranking['scores']
        exact_shuffle = permutation[ids].tolist() == ranking['identity_shuffled_token_ids']
        same_set = set(independent_order) == set(ids)
        displaced = set(independent_order) ^ set(ids)
        boundary_tolerance = all(abs(float(oracle[token]) - float(oracle[independent_order[-1]])) <= 2 * absolute_tolerance
                                 for token in displaced)
        record = {'name': name, 'maximum_full_score_absolute_error': maximum_error,
                  'absolute_tolerance': absolute_tolerance, 'relative_tolerance': 0.,
                  'tolerance_scope': 'FP64 diagnostic comparison, not an interval-certified error bound',
                  'stored_scores_sort_matches_frozen_exactly': exact_stored,
                  'identity_shuffle_matches_frozen_exactly': exact_shuffle,
                  'independent_score_within_tolerance': maximum_error <= absolute_tolerance,
                  'independent_top128_set_matches_exactly': same_set,
                  'independent_top128_order_matches_exactly': independent_order == ids,
                  'independent_order_mismatch_count': sum(a != b for a, b in zip(independent_order, ids)),
                  'set_symmetric_difference': sorted(displaced),
                  'set_difference_compatible_with_declared_score_tolerance': boundary_tolerance,
                  'independent_order': independent_order, 'frozen_order': ids}
        if not exact_stored or not exact_shuffle or maximum_error > absolute_tolerance:
            errors.append(name + ': stored identity or numerical comparison failed')
        if not same_set:
            errors.append(name + ': independent candidate set differs (even if near-tie compatible)')
        if independent_order != ids:
            warnings.append(name + ': independent internal ordering differs; recorded explicitly')
        if name.startswith('shared_'):
            variant = name.rsplit('_', 1)[1]
            source_rows = np.array([stored[f'followthrough_{variant}_{boundary}'][ids] for boundary in BOUNDARIES])
            oracle_rows = np.array([independently_computed[f'followthrough_{variant}_{boundary}'][ids] for boundary in BOUNDARIES])
            stored_components_exact = source_rows.tolist() == ranking['component_scores']
            oracle_components_error = float(np.max(np.abs(oracle_rows - source_rows)))
            stored_positive = (source_rows > 0).sum(axis=0).tolist() == ranking['positive_component_count']
            oracle_positive = (oracle_rows > 0).sum(axis=0).tolist() == ranking['positive_component_count']
            numerator = doubled[name]
            rational_order = heapq.nsmallest(K, range(VOCABULARY), key=lambda token: (-int(numerator[token]), token))
            order_changes = [{'rank_1_based': rank, 'independent_id': actual, 'frozen_id': saved,
                              'doubled_rank_sums': [int(numerator[actual]), int(numerator[saved])],
                              'exact_mathematical_tie': bool(numerator[actual] == numerator[saved])}
                             for rank, (actual, saved) in enumerate(zip(independent_order, ids), 1) if actual != saved]
            record.update(stored_components_match_at_frozen_ids_exactly=stored_components_exact,
                          independent_components_maximum_absolute_error=oracle_components_error,
                          stored_positive_counts_match_exactly=stored_positive,
                          independent_positive_counts_match_exactly=oracle_positive,
                          exact_rational_midrank_top128_set_matches=set(rational_order) == set(ids),
                          exact_rational_midrank_top128_order_matches=rational_order == ids,
                          internal_order_differences=order_changes)
            if not stored_components_exact or not stored_positive or not oracle_positive or oracle_components_error > 5e-14:
                errors.append(name + ': per-frozen-ID component/positivity comparison failed')
            if set(rational_order) != set(ids):
                errors.append(name + ': exact-rational independent rank set differs')
            if any(not change['exact_mathematical_tie'] for change in order_changes):
                errors.append(name + ': internal ordering difference is not an exact midrank tie')
        checks.append(record)
    if len(activity_records) != 15 or len(frozen['checkpoint_provenance']) != 20:
        raise ValueError('wrong fixed activity or checkpoint count')
    archives = []
    for step, provenance in frozen['checkpoint_provenance'].items():
        if provenance.get('source_kind') != 'gzip_tar_archive' or len(provenance['input_files']) != 1:
            raise ValueError('expected exactly twenty original checkpoint archives')
        original = provenance['input_files'][0]
        current = identity(original['path'])
        if current != original:
            errors.append('final archive hash changed at step ' + step)
        archives.append({'step': int(step), 'original': original, 'current': current,
                         'unchanged': original == current})
    for record in [frozen_identity, source_identity, *frozen['sources'].values(), *frozen['files'].values()]:
        if identity(record['path']) != record:
            errors.append('source/frozen/numerical artifact changed during audit: ' + record['path'])
    result = {'schema_version': 1, 'complete': True, 'passed': not errors,
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'frozen': frozen_identity, 'auditor_source': source_identity,
              'all_activity_hashes_shapes_finiteness_checked': len(activity_records),
              'activity_records': activity_records, 'score_checks': checks,
              'all_stored_score_rankings_exact': all(row['stored_scores_sort_matches_frozen_exactly'] for row in checks),
              'all_independent_candidate_sets_exact': all(row['independent_top128_set_matches_exactly'] for row in checks),
              'all_independent_orders_exact': all(row['independent_top128_order_matches_exactly'] for row in checks),
              'final_archive_count': len(archives), 'all_final_archive_hashes_unchanged': all(row['unchanged'] for row in archives),
              'archives': archives, 'errors': errors, 'warnings': warnings,
              'no_corpus_or_tokenizer_access': True, 'no_model_execution': True,
              'limitations': ['Scalar recomputation starts from authenticated activity vectors, not fresh embedding reductions.',
                              'Only source/output/archive hashes are checked, not historical training provenance.',
                              'FP64 diagnostic tolerances are not interval-certified bounds.',
                              'A frozen candidate set is never modified to resolve an audit discrepancy.']}
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    with output.open('x') as stream:
        stream.write(payload)
    print(json.dumps({key: result[key] for key in ('passed', 'all_stored_score_rankings_exact',
                    'all_independent_candidate_sets_exact', 'all_independent_orders_exact',
                    'all_final_archive_hashes_unchanged', 'errors', 'warnings')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    audit(args.run, args.output)
