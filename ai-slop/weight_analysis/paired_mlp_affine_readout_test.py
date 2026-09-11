"""Independent CPU numeric oracles for the explicit E/N/F/M readout."""

import copy
import hashlib
import math
import unittest
from unittest import mock

from . import paired_mlp_affine_readout as readout


CELLS = ('E', 'N', 'F', 'M')
IDS = [1475, 68, 2797]


def case(prefix='one', *, probabilities=None, winners=None, ids=None, kind='word', **changes):
    ids = list(IDS if ids is None else ids)
    probabilities = {cell: [.25]*len(ids) for cell in CELLS} if probabilities is None else probabilities
    value = dict(suite='supplemental' if kind in ('word_next_native', 'shared_piece') else 'main',
        kind=kind, split='test', context_id=prefix, prefix_domain='original', target='Exeunt',
        spelling_variant='title', word_has_leading_space=True, target_ids=ids,
        prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(), prefix_length=128,
        cells={cell: dict(token_log_probability=[math.log(p) for p in probabilities[cell]],
                         argmax_ids=list(ids if winners is None else winners[cell])) for cell in CELLS})
    value.update(changes)
    return value


class NumericTest(unittest.TestCase):
    def metrics(self, *cases):
        groups = readout.aggregate(list(cases))
        self.assertEqual(len(groups), 1)
        return groups[0]['metrics']

    def test_exact_cells_and_only_five_explicit_contrasts(self):
        item = case()
        # Hand-computable suffix log scores: N-E=4, F-E=2, and interaction=1.
        scores = {'E': -10., 'N': -6., 'F': -8., 'M': -3.}
        for cell, score in scores.items():
            item['cells'][cell]['token_log_probability'] = [-4., score/2, score/2]
        with mock.patch.object(readout.core, 'score_cube', side_effect=AssertionError('not a cube')):
            metric = self.metrics(item)['suffix']
        self.assertEqual(metric['contrasts'], {'N-E': 4., 'F-E': 2., 'M-F': 5., 'M-N': 3., 'M-N-F+E': 1.})
        self.assertEqual(set(metric), {'unique_event_count', 'alias_case_count', 'cells', 'contrasts'})
        self.assertEqual(tuple(metric['cells']), CELLS)

    def test_negative_effects_and_nonadditive_union_keep_their_signs(self):
        item = case()
        for cell, score in {'E': -3., 'N': -5., 'F': -4., 'M': -2.}.items():
            item['cells'][cell]['token_log_probability'] = [score, -1., -1.]
        self.assertEqual(self.metrics(item)['token_0']['contrasts'],
            {'N-E': -2., 'F-E': -1., 'M-F': 2., 'M-N': 3., 'M-N-F+E': 4.})

    def test_identical_cells_have_zero_effects(self):
        for metric in self.metrics(case()).values():
            self.assertEqual(set(metric['contrasts'].values()), {0.})

    def test_suffix_and_word_are_products_not_length_normalized_means(self):
        probabilities = {cell: [.125, .25, .5] for cell in CELLS}
        metrics = self.metrics(case(probabilities=probabilities))
        self.assertAlmostEqual(metrics['token_0']['cells']['N']['geometric_mean_probability'], .125)
        self.assertAlmostEqual(metrics['token_1']['cells']['N']['geometric_mean_probability'], .25)
        self.assertAlmostEqual(metrics['token_2']['cells']['N']['geometric_mean_probability'], .5)
        self.assertAlmostEqual(metrics['suffix']['cells']['N']['geometric_mean_probability'], .125)
        self.assertAlmostEqual(metrics['word_three']['cells']['N']['geometric_mean_probability'], .015625)
        self.assertEqual(metrics['selected_sequence'], metrics['word_three'])
        self.assertEqual(metrics['first_piece'], metrics['token_0'])

    def test_geometric_mean_averages_log_scores_not_probabilities(self):
        one = {cell: [.01, .2, .5] for cell in CELLS}
        two = {cell: [.09, .8, .5] for cell in CELLS}
        metrics = self.metrics(case('one', probabilities=one), case('two', probabilities=two))
        self.assertAlmostEqual(metrics['token_0']['cells']['F']['geometric_mean_probability'], .03)
        self.assertAlmostEqual(metrics['suffix']['cells']['F']['geometric_mean_probability'], .2)
        self.assertAlmostEqual(metrics['word_three']['cells']['F']['geometric_mean_probability'], .006)
        self.assertEqual(metrics['word_three']['unique_event_count'], 2)

    def test_contrasts_use_paired_event_means(self):
        first, second = case('one'), case('two')
        for item, scores in ((first, {'E': -10., 'N': -6., 'F': -8., 'M': -3.}),
                             (second, {'E': -4., 'N': -8., 'F': -2., 'M': -5.})):
            for cell, score in scores.items():
                item['cells'][cell]['token_log_probability'][0] = score
        metric = self.metrics(first, second)['token_0']
        self.assertEqual(metric['contrasts'], {'N-E': 0., 'F-E': 2., 'M-F': 1., 'M-N': 3., 'M-N-F+E': 1.})

    def test_argmax_conjunction_uses_same_event_not_marginal_products(self):
        left = {cell: [1475, 999, 2797] for cell in CELLS}
        right = {cell: [1475, 68, 999] for cell in CELLS}
        metrics = self.metrics(case('left', winners=left), case('right', winners=right))
        for cell in CELLS:
            for name in ('token_1', 'token_2'):
                self.assertEqual(metrics[name]['cells'][cell]['joint_argmax_count'], 1)
                self.assertEqual(metrics[name]['cells'][cell]['joint_argmax_rate'], .5)
            for name in ('suffix', 'word_three'):
                self.assertEqual(metrics[name]['cells'][cell]['joint_argmax_count'], 0)
                self.assertEqual(metrics[name]['cells'][cell]['joint_argmax_rate'], 0.)

    def test_aliases_do_not_reweight_scores_or_argmax_denominators(self):
        first, other = case('same'), case('other')
        alias = copy.deepcopy(first); alias['context_id'] = 'different label'
        for cell in CELLS:
            other['cells'][cell]['argmax_ids'][2] = 999
        metrics = self.metrics(first, alias, other)
        for name in ('token_0', 'suffix', 'word_three'):
            self.assertEqual(metrics[name]['unique_event_count'], 2)
            self.assertEqual(metrics[name]['alias_case_count'], 3)
        self.assertEqual(metrics['word_three']['cells']['M']['joint_argmax_count'], 1)
        self.assertEqual(metrics['word_three']['cells']['M']['joint_argmax_rate'], .5)

    def test_fourth_target_changes_only_fourth_and_full_sequence_deduplication(self):
        first = case('same', kind='word_next_native', ids=IDS+[13])
        second = case('same', kind='word_next_native', ids=IDS+[198])
        for cell in CELLS:
            second['cells'][cell]['argmax_ids'][3] = 13
            second['cells'][cell]['token_log_probability'][3] = math.log(.125)
        metrics = self.metrics(first, second)
        self.assertEqual(metrics['word_three']['unique_event_count'], 1)
        self.assertEqual(metrics['suffix']['unique_event_count'], 1)
        for name in ('exact_next_native_token', 'token_3', 'selected_sequence'):
            self.assertEqual(metrics[name]['unique_event_count'], 2)
            self.assertEqual(metrics[name]['cells']['E']['joint_argmax_count'], 1)
            self.assertEqual(metrics[name]['cells']['E']['joint_argmax_rate'], .5)
        self.assertAlmostEqual(metrics['exact_next_native_token']['cells']['E']['geometric_mean_probability'],
                               math.sqrt(.25*.125))
        self.assertNotEqual(metrics['selected_sequence'], metrics['word_three'])

    def test_suffix_identity_includes_the_supplied_first_token(self):
        first = case('same')
        second = case('same', ids=[45, 68, 2797])
        for cell in CELLS:
            # Both candidate targets share the original prefix's one winner.
            second['cells'][cell]['argmax_ids'][0] = IDS[0]
            second['cells'][cell]['token_log_probability'][1] = math.log(.125)
        metric = self.metrics(first, second)['suffix']
        self.assertEqual(metric['unique_event_count'], 2)
        self.assertAlmostEqual(metric['cells']['N']['geometric_mean_probability'], math.sqrt(.0625*.03125))

    def test_all_strata_stay_separate_without_a_pooled_domain(self):
        changes = [{}, {'prefix_domain': 'replacement'}, {'split': 'training'},
                   {'spelling_variant': 'lowercase'}, {'word_has_leading_space': False},
                   {'target': 'Nuveth'},
                   {'kind': 'control', 'prefix_domain': 'shared', 'target': 'control_next_3'},
                   {'kind': 'shared_piece', 'prefix_domain': 'shared', 'piece_id': 1},
                   {'kind': 'shared_piece', 'prefix_domain': 'shared', 'piece_id': 2}]
        items = [case(str(index), **change) for index, change in enumerate(changes)]
        groups = readout.aggregate(items)
        self.assertEqual(len(groups), len(items))
        self.assertNotIn('deduplicated_all', {group['prefix_domain'] for group in groups})
        controls = [group for group in groups if group['kind'] in ('control', 'shared_piece')]
        self.assertTrue(all(set(group['metrics']) == {'selected_sequence', 'token_0', 'token_1', 'token_2'}
                            for group in controls))

    def test_main_and_supplemental_word_aliases_are_not_pooled(self):
        groups = readout.aggregate([case('same'), case('same', kind='word_next_native', ids=IDS+[13])])
        self.assertEqual(len(groups), 2)
        self.assertEqual({group['suite'] for group in groups}, {'main', 'supplemental'})
        for group in groups:
            self.assertEqual(group['metrics']['word_three']['unique_event_count'], 1)

    def test_probability_underflow_retains_finite_log_scores(self):
        item = case()
        for cell in CELLS:
            item['cells'][cell]['token_log_probability'] = [-900., -800., -700.]
        metric = self.metrics(item)['word_three']
        self.assertEqual(metric['cells']['M']['geometric_mean_probability'], 0.)
        self.assertEqual(metric['cells']['M']['mean_log_probability'], -2400.)
        self.assertEqual(set(metric['contrasts'].values()), {0.})

    def test_zero_log_probability_is_valid_and_remains_one(self):
        item = case()
        for cell in CELLS:
            item['cells'][cell]['token_log_probability'] = [0., -0., 0.]
        metric = self.metrics(item)['word_three']
        self.assertEqual(metric['cells']['E']['geometric_mean_probability'], 1.)
        self.assertEqual(set(metric['contrasts'].values()), {0.})

    def test_inputs_are_not_mutated_and_case_order_does_not_change_results(self):
        items = [case('one'), case('two')]
        saved = copy.deepcopy(items)
        expected = readout.aggregate(items)
        self.assertEqual(items, saved)
        self.assertEqual(readout.aggregate(list(reversed(items))), expected)


class ValidationTest(unittest.TestCase):
    def test_empty_missing_extra_and_relabelled_cells_are_rejected(self):
        with self.assertRaises(ValueError):
            readout.aggregate([])
        for cells in (('E', 'N', 'M'), ('E', 'N', 'F', 'M', 'R'), ('E', 'A', 'M', 'EC')):
            item = case()
            item['cells'] = {cell: copy.deepcopy(item['cells']['E']) for cell in cells}
            with self.subTest(cells=cells), self.assertRaises(ValueError):
                readout.aggregate([item])

    def test_nonfinite_positive_boolean_and_wrong_length_scores_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf, True, .1, '-1'):
            item = case(); item['cells']['N']['token_log_probability'][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                readout.aggregate([item])
        item = case(); item['cells']['F']['token_log_probability'].pop()
        with self.assertRaises(ValueError):
            readout.aggregate([item])

    def test_noninteger_negative_boolean_or_wrong_length_argmax_is_rejected(self):
        for values in ([True, 68, 2797], [-1, 68, 2797], [1., 68, 2797], [1475, 68]):
            item = case(); item['cells']['F']['argmax_ids'] = values
            with self.subTest(values=values), self.assertRaises(ValueError):
                readout.aggregate([item])

    def test_invalid_target_geometry_hash_and_strata_are_rejected(self):
        for changes in ({'target_ids': [True, 68, 2797]}, {'target_ids': IDS+[13]},
                        {'prefix_length': True}, {'prefix_sha256': 'not-a-hash'},
                        {'suite': 'supplemental'}, {'split': 'other'}, {'prefix_domain': 'pooled'},
                        {'word_has_leading_space': 1}, {'piece_id': True}, {'target': ''}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                readout.aggregate([case(**changes)])

    def test_alias_scores_must_match_per_token_even_if_sequence_sums_match(self):
        first, second = case('same'), case('same')
        first['cells']['N']['token_log_probability'] = [-1., -2., -3.]
        second['cells']['N']['token_log_probability'] = [-1., -2.5, -2.5]
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            readout.aggregate([first, second])

    def test_alias_argmax_mismatch_is_rejected_even_when_both_are_wrong(self):
        first, second = case('same'), case('same')
        first['cells']['N']['argmax_ids'][0] = 998
        second['cells']['N']['argmax_ids'][0] = 999
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            readout.aggregate([first, second])

    def test_conflicting_causal_aliases_are_rejected_across_reporting_strata(self):
        first = case('same')
        second = case('same', split='training', prefix_domain='replacement')
        second['cells']['M']['argmax_ids'][0] = 999
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            readout.aggregate([first, second])

    def test_different_targets_at_one_prefix_cannot_claim_different_winners(self):
        first = case('same')
        second = case('same', target='Nuveth', ids=[45, 68, 2797])
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            readout.aggregate([first, second])

    def test_unrepresentable_sequence_log_score_is_rejected(self):
        item = case()
        for cell in CELLS:
            item['cells'][cell]['token_log_probability'] = [-1e308]*3
        with self.assertRaisesRegex(ValueError, 'nonfinite aggregate'):
            readout.aggregate([item])


if __name__ == '__main__':
    unittest.main()
