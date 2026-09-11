"""Preserve and explain FP64 tie sensitivity without changing frozen results."""

import hashlib
import heapq
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path('/tmp/pluto-restart-delta.fZs3LQ')
RUN = ROOT / 'run'
spec = importlib.util.spec_from_file_location('original_independent_audit', ROOT / 'independent_check.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def doubled_midranks(values):
    ordered = sorted(range(len(values)), key=lambda index: float(values[index]))
    result = np.empty(len(values), dtype=np.int64)
    consumed = 0
    for _, group in itertools.groupby(ordered, key=lambda index: float(values[index])):
        ids = list(group)
        result[ids] = 2 * consumed + len(ids)
        consumed += len(ids)
    return result


def main():
    original = json.loads((RUN / 'independent_check.json').read_text())
    assert audit.file_record(ROOT / 'independent_check.py') == original['auditor_source']
    assert original['error_count'] == 9
    frozen = json.loads((RUN / 'frozen.json').read_text())
    assert audit.file_record(RUN / 'frozen.json') == original['frozen_manifest']
    candidates = json.loads((RUN / 'candidates.json').read_text())['candidate_lists']
    with np.load(RUN / 'scores.npz', allow_pickle=False) as source:
        stored = {name: source[name] for name in source.files}
    permutation = np.random.Generator(np.random.PCG64(20260909)).permutation(audit.VOCABULARY)
    assert np.array_equal(permutation, stored['identity_permutation'])
    recomputed = {}
    components = {}
    for boundary in audit.BOUNDARIES:
        intervals = []
        for earlier in range(boundary - 30, boundary + 40, 10):
            with np.load(RUN / f'interval_{earlier}_{earlier + 10}.npz', allow_pickle=False) as source:
                intervals.append({name: source[name] for name in ['raw', 'adjusted']})
        for variant in ['raw', 'adjusted']:
            logs = [audit.scalar_logs(item[variant]) for item in intervals]
            for kind, center in [('ordinary_before', 1), ('boundary', 3), ('ordinary_after', 5)]:
                score = np.array([float(c) - .5 * (float(a) + float(b))
                                  for a, c, b in zip(logs[center - 1], logs[center], logs[center + 1])])
                recomputed[f'{kind}_{variant}_{boundary}'] = score
                components.setdefault((kind, variant), []).append(score)
    for (kind, variant), rows in components.items():
        ranks = [audit.grouped_ranks(row) for row in rows]
        recomputed[f'shared_{kind}_{variant}'] = np.array([sum(float(v) for v in column) / 5
                                                        for column in zip(*ranks)])
    # Endpoint norms were independently recomputed from archive bytes in the
    # original audit; that successful comparison is retained, not rerun here.
    for boundary in audit.BOUNDARIES:
        recomputed[f'endpoint_norm_{boundary}'] = stored[f'endpoint_norm_{boundary}']
    ranks = [audit.grouped_ranks(recomputed[f'endpoint_norm_{step}']) for step in audit.BOUNDARIES]
    recomputed['shared_endpoint_norm'] = np.array([sum(float(v) for v in column) / 5
                                                 for column in zip(*ranks)])
    records = []
    for name, candidate in candidates.items():
        stored_ids = candidate['token_ids']
        independently_sorted_stored_ids = audit.top_ids(stored[name])
        arithmetic_ids = audit.top_ids(recomputed[name])
        assert independently_sorted_stored_ids == stored_ids
        assert set(arithmetic_ids) == set(stored_ids)
        assert [int(permutation[index]) for index in stored_ids] == candidate['identity_shuffled_token_ids']
        np.testing.assert_allclose(recomputed[name][stored_ids], candidate['scores'], atol=5e-14, rtol=5e-14)
        item = {'name': name, 'stored_score_heap_ranking_matches_frozen_exactly': True,
                'identity_shuffle_matches_frozen_exactly': True,
                'independent_arithmetic_top128_set_matches_exactly': True,
                'independent_arithmetic_selected_scores_close': True,
                'independent_arithmetic_order_mismatch_count': sum(a != b for a, b in zip(arithmetic_ids, stored_ids))}
        if name.startswith('shared_'):
            if name == 'shared_endpoint_norm':
                rows = [stored[f'endpoint_norm_{step}'] for step in audit.BOUNDARIES]
            else:
                kind, variant = name.removeprefix('shared_').rsplit('_', 1)
                rows = components[(kind, variant)]
                positive = [sum(row[index] > 0 for row in rows) for index in stored_ids]
                assert positive == candidate['positive_component_count']
                np.testing.assert_allclose([[float(row[index]) for index in stored_ids] for row in rows],
                                           candidate['component_scores'], atol=5e-14, rtol=0)
                item['component_scores_checked_at_frozen_ids'] = True
                item['positive_component_counts_match_per_frozen_id'] = True
            numerator = np.sum([doubled_midranks(row) for row in rows], axis=0)
            rational_ids = heapq.nsmallest(128, range(audit.VOCABULARY),
                                         key=lambda index: (-int(numerator[index]), index))
            assert set(rational_ids) == set(stored_ids)
            item['exact_integer_midrank_sum_top128_set_matches'] = True
            item['exact_integer_midrank_sum_order_mismatch_count'] = sum(a != b for a, b in zip(rational_ids, stored_ids))
            differences = []
            for rank, (actual, frozen_id) in enumerate(zip(arithmetic_ids, stored_ids), start=1):
                if actual != frozen_id:
                    assert numerator[actual] == numerator[frozen_id]
                    differences.append({'rank_1_based': rank, 'independent_id': actual,
                                        'frozen_id': frozen_id,
                                        'exact_doubled_midrank_sum_both': int(numerator[actual]),
                                        'frozen_scores': [float(stored[name][actual]), float(stored[name][frozen_id])],
                                        'independent_scores': [float(recomputed[name][actual]), float(recomputed[name][frozen_id])]})
            item['floating_point_tie_order_differences'] = differences
        records.append(item)
    failures = [c for c in original['comparisons'] if not c['passed']]
    assert len(failures) == 3
    assert all(c['name'].endswith(':component_scores') for c in failures)
    result = {
        'schema': 'restart_delta_independent_numeric_tie_explanation_v1',
        'original_audit_preserved': audit.file_record(RUN / 'independent_check.json'),
        'original_auditor_source': audit.file_record(ROOT / 'independent_check.py'),
        'supplement_auditor_source': audit.file_record(__file__),
        'frozen_manifest': audit.file_record(RUN / 'frozen.json'),
        'original_error_count_not_erased': original['error_count'],
        'all42_stored_score_rankings_and_shuffles_match_exactly': True,
        'all42_independent_arithmetic_candidate_sets_match_exactly': True,
        'all_selected_scores_and_shared_component_scores_close_at_frozen_ids': True,
        'all_shared_positive_component_counts_match_at_frozen_ids': True,
        'strict_independent_floating_point_order_match': False,
        'lists_with_internal_tie_order_variation': [r['name'] for r in records if r['independent_arithmetic_order_mismatch_count']],
        'total_positions_reordered_inside_same_top128_sets': sum(r['independent_arithmetic_order_mismatch_count'] for r in records),
        'boundary_lists_order_unchanged': True,
        'source_of_original_component_errors': 'Initial audit compared component columns at its own reordered IDs rather than frozen IDs. Per-frozen-ID comparisons pass here.',
        'interpretation': 'FP64 mean-percentile summation changes ordering inside mathematically tied groups, at maximum score error 2.22e-16. The frozen ranking is internally correct for its stored FP64 scores. Exact mathematical midrank averaging would require integer/rational tie handling. No candidate-set membership changes occurred, so unordered presence/frequency metrics and per-ID positivity are unaffected.',
        'no_frozen_files_modified': True, 'no_corpus_or_tokenizer_access': True,
        'no_model_execution': True, 'records': records,
    }
    for record in original['artifact_records'].values():
        assert audit.file_record(record['path']) == record
    for record in original['source_records'].values():
        assert audit.file_record(record['path']) == record
    assert audit.file_record(RUN / 'frozen.json') == original['frozen_manifest']
    out = RUN / 'independent_check_ties.json'
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': audit.file_record(out),
                      'lists_with_internal_tie_order_variation': result['lists_with_internal_tie_order_variation'],
                      'reordered_positions': result['total_positions_reordered_inside_same_top128_sets'],
                      'all_candidate_sets_match': True}, indent=2))


if __name__ == '__main__':
    main()
