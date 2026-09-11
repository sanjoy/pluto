"""Independent small-probability oracles for the early-branch partial reader.

These tests use hand-computable token probabilities, not a GPU or the cube's
aggregation implementation. In particular, a supplied-prefix suffix score is
a product of conditional probabilities, and a joint argmax requires all of
its predictions to succeed in the same event.
"""

import copy
import hashlib
import math
import unittest

from . import paired_early_branches_partial as partial


CELLS = ('E', 'EC', 'A', 'M', 'AM')
IDS = [1475, 68, 2797]


def example(prefix, probabilities=None, winners=None, *, kind='word', ids=None):
    target_ids = list(IDS if ids is None else ids)
    if probabilities is None:
        probabilities = {cell: [0.25] * len(target_ids) for cell in CELLS}
    return dict(
        suite='supplemental' if kind == 'word_next_native' else 'main',
        kind=kind, split='test', context_id=prefix, prefix_domain='original',
        target='Exeunt', spelling_variant='title', word_has_leading_space=True,
        target_ids=target_ids, prefix_length=128,
        prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
        cells={cell: dict(
            token_log_probability=[math.log(p) for p in probabilities[cell]],
            argmax_ids=list(target_ids if winners is None else winners[cell]))
            for cell in CELLS})


class IndependentNumericTest(unittest.TestCase):
    def metrics(self, *cases):
        groups = partial.aggregate(list(cases), CELLS)
        self.assertEqual(len(groups), 1)
        return groups[0]['metrics']

    def test_suffix_multiplies_two_conditionals_without_length_normalizing(self):
        probabilities = {cell: [.125, .25, .5] for cell in CELLS}
        metrics = self.metrics(example('one', probabilities))
        self.assertAlmostEqual(metrics['suffix']['cells']['A']['geometric_mean_probability'], .125)
        self.assertAlmostEqual(metrics['word_three']['cells']['A']['geometric_mean_probability'], .015625)
        self.assertAlmostEqual(metrics['token_1']['cells']['A']['geometric_mean_probability'], .25)

    def test_geometric_probability_is_not_arithmetic_average(self):
        one = {cell: [.01, .2, .5] for cell in CELLS}
        two = {cell: [.09, .8, .5] for cell in CELLS}
        metrics = self.metrics(example('one', one), example('two', two))
        self.assertAlmostEqual(metrics['token_0']['cells']['M']['geometric_mean_probability'], .03)
        self.assertAlmostEqual(metrics['suffix']['cells']['M']['geometric_mean_probability'], .2)
        self.assertAlmostEqual(metrics['word_three']['cells']['M']['geometric_mean_probability'], .006)

    def test_joint_argmax_does_not_multiply_marginal_success_rates(self):
        left = {cell: [1475, 999, 2797] for cell in CELLS}
        right = {cell: [1475, 68, 999] for cell in CELLS}
        metrics = self.metrics(example('left', winners=left), example('right', winners=right))
        for cell in CELLS:
            self.assertEqual(metrics['token_1']['cells'][cell]['joint_argmax_count'], 1)
            self.assertEqual(metrics['token_2']['cells'][cell]['joint_argmax_count'], 1)
            self.assertEqual(metrics['suffix']['cells'][cell]['joint_argmax_count'], 0)
            self.assertEqual(metrics['word_three']['cells'][cell]['joint_argmax_count'], 0)

    def test_aliases_do_not_change_probabilities_or_argmax_denominators(self):
        first = example('same')
        alias = copy.deepcopy(first)
        alias['context_id'] = 'different label, identical causal event'
        metrics = self.metrics(first, alias)
        for name in ('token_0', 'suffix', 'word_three'):
            self.assertEqual(metrics[name]['unique_event_count'], 1)
            self.assertEqual(metrics[name]['alias_case_count'], 2)
            self.assertEqual(metrics[name]['cells']['EC']['joint_argmax_count'], 1)

    def test_different_fourth_token_does_not_duplicate_word_metrics(self):
        first = example('same', kind='word_next_native', ids=IDS + [13])
        second = example('same', kind='word_next_native', ids=IDS + [198])
        # A prediction cannot have two winners at the same fourth-token prefix.
        for cell in CELLS:
            second['cells'][cell]['argmax_ids'][3] = 13
            second['cells'][cell]['token_log_probability'][3] = math.log(.125)
        metrics = self.metrics(first, second)
        self.assertEqual(metrics['word_three']['unique_event_count'], 1)
        self.assertEqual(metrics['word_three']['cells']['E']['joint_argmax_count'], 1)
        fourth = metrics['exact_next_native_token']
        self.assertEqual(fourth['unique_event_count'], 2)
        self.assertEqual(fourth['cells']['E']['joint_argmax_count'], 1)
        self.assertAlmostEqual(fourth['cells']['E']['geometric_mean_probability'], math.sqrt(.25 * .125))

    def test_alias_winner_mismatch_rejected_even_if_both_winners_are_wrong(self):
        first = example('same', winners={cell: [999, 68, 2797] for cell in CELLS})
        second = copy.deepcopy(first)
        second['cells']['A']['argmax_ids'][0] = 998
        with self.assertRaises(ValueError):
            self.metrics(first, second)

    def test_probability_underflow_does_not_destroy_log_score(self):
        item = example('tiny')
        for cell in CELLS:
            item['cells'][cell]['token_log_probability'] = [-900., -800., -700.]
        metrics = self.metrics(item)
        value = metrics['word_three']['cells']['AM']
        self.assertEqual(value['geometric_mean_probability'], 0.)
        self.assertEqual(value['mean_log_probability'], -2400.)

    def test_split_strata_remain_separate(self):
        first = example('same')
        second = copy.deepcopy(first)
        second['split'] = 'training'
        groups = partial.aggregate([first, second], CELLS)
        self.assertEqual({group['split'] for group in groups}, {'test', 'training'})
        self.assertEqual(len(groups), 2)

    def test_partial_interaction_uses_log_scores_and_omits_unmeasured_background(self):
        item = example('contrast')
        suffix_scores = {'E': -10., 'EC': -2., 'A': -6., 'M': -8., 'AM': -3.}
        for cell, score in suffix_scores.items():
            item['cells'][cell]['token_log_probability'] = [-4., score/2, score/2]
        metric = self.metrics(item)['suffix']
        self.assertEqual(metric['available_contrasts']['A-E'], 4.)
        self.assertEqual(metric['available_contrasts']['M-E'], 2.)
        self.assertEqual(metric['available_contrasts']['AM-A-M+E'], 1.)
        self.assertNotIn('EC-AR-MR+R', metric['available_contrasts'])
        self.assertNotIn('three_way', metric['available_contrasts'])
        self.assertNotIn('effects', metric)

    def test_two_background_interactions_have_independent_signs(self):
        item = example('full contrast')
        scores = {'E': -10., 'EC': -2., 'A': -6., 'M': -8., 'AM': -3.,
                  'R': -8., 'AR': -3., 'MR': -5.}
        for cell, score in scores.items():
            item['cells'][cell] = dict(token_log_probability=[-4., score/2, score/2],
                                       argmax_ids=IDS.copy())
        metric = partial.aggregate([item], tuple(scores))[0]['metrics']['suffix']
        self.assertEqual(metric['available_contrasts']['AM-A-M+E'], 1.)
        self.assertEqual(metric['available_contrasts']['EC-AR-MR+R'], -2.)
        self.assertEqual(metric['available_contrasts']['three_way'], -3.)
        self.assertEqual(metric['effects']['pair_interactions']['AM'],
                         {'absent': 1., 'present': -2.})


if __name__ == '__main__':
    unittest.main()
