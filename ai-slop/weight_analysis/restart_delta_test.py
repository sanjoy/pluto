import unittest

import numpy as np

from . import restart_delta as rd


class RestartDeltaTest(unittest.TestCase):
    def test_fixed_intervals(self):
        self.assertEqual(rd.checkpoint_steps(580), [550, 560, 570, 580, 590, 600, 610, 620])
        for invalid in (True, 31, 39, 581, -10):
            with self.assertRaises(ValueError): rd.checkpoint_steps(invalid)
        self.assertEqual(len({step for b in rd.BOUNDARIES for step in rd.checkpoint_steps(b)}), 40)

    def test_exact_orientation_and_projection(self):
        before = np.array([[1., 2], [3, 0], [-2, 4]])
        after = np.array([[2., 4], [2, 1], [-3, 7]])
        values, report = rd.activities(before, after)
        delta = after - before
        mean = sum(delta) / 3
        centered = before - sum(before) / 3
        coefficient = sum(float(x @ y) for x, y in zip(centered, delta - mean)) / sum(float(x @ x) for x in centered)
        np.testing.assert_allclose(report['shared_translation'], mean)
        self.assertAlmostEqual(report['fitted_radial_coefficient_not_authenticated_decay'], coefficient)
        np.testing.assert_allclose(values['raw'], [np.sqrt(sum(row**2)) for row in delta])
        np.testing.assert_allclose(values['adjusted'], [np.sqrt(sum(row**2)) for row in delta - mean - coefficient * centered])

    def test_translation_and_radial_control_and_inputs_unchanged(self):
        before = np.arange(40, dtype=np.float32).reshape(10, 4)
        original = before.tobytes()
        after = before * 1.25 + [3., -2, 4, 7]
        original_after = after.tobytes()
        values, report = rd.activities(before, after)
        np.testing.assert_allclose(values['adjusted'], 0, atol=1e-14)
        self.assertAlmostEqual(report['fitted_radial_coefficient_not_authenticated_decay'], .25)
        self.assertEqual(before.tobytes(), original)
        self.assertEqual(after.tobytes(), original_after)
        self.assertGreater(values['raw'].max(), 0)

    def test_zero_and_constant_embedding(self):
        zero = np.zeros((5, 3))
        values, report = rd.activities(zero, np.ones((5, 3)))
        np.testing.assert_array_equal(values['adjusted'], np.zeros(5))
        self.assertEqual(report['fitted_radial_coefficient_not_authenticated_decay'], 0)
        np.testing.assert_array_equal(rd.log_activity(np.zeros(5)), np.zeros(5))
        np.testing.assert_allclose(rd.log_activity([0., 0, 2]), [np.log(1e-12), np.log(1e-12), 0])

    def test_log_activity_scale_invariance_and_validation(self):
        values = np.array([0., 1, 3, 6, 10])
        np.testing.assert_allclose(rd.log_activity(values), rd.log_activity(values * 57), atol=1e-14)
        for invalid in ([], [-1], [np.nan], [np.inf], [[1]], [1e-320]):
            with self.assertRaises(ValueError): rd.log_activity(invalid)
        for first, second in (([[np.nan]], [[1]]), ([[1]], [[1, 2]]), ([], [])):
            with self.assertRaises(ValueError): rd.activities(first, second)

    def test_transient_center_positions(self):
        intervals = np.ones((7, 8))
        for center, name in ((1, 'ordinary_before'), (3, 'boundary'), (5, 'ordinary_after')):
            current = intervals.copy()
            current[center, 0] = np.e**2
            scores = rd.transient_scores(current)
            self.assertAlmostEqual(scores[name][0], 2)
            for other in set(rd.CENTERS) - {name}:
                self.assertEqual(scores[other][0], 0)
        with self.assertRaises(ValueError): rd.transient_scores(intervals[:6])

    def test_midrank_exact_ties(self):
        np.testing.assert_array_equal(rd.percentile_midrank([3., 1, 2, 2]), [.875, .125, .5, .5])
        np.testing.assert_array_equal(rd.percentile_midrank(np.zeros(5)), np.full(5, .5))
        for invalid in ([], [np.nan], [[1]]):
            with self.assertRaises(ValueError): rd.percentile_midrank(invalid)

    def test_candidate_ties_and_label_shuffle(self):
        ranking = rd.ranked_candidates([2., 3, 3, -1], [3, 2, 1, 0], k=3)
        self.assertEqual(ranking['token_ids'], [1, 2, 0])
        self.assertEqual(ranking['scores'], [3, 3, 2])
        self.assertEqual(ranking['identity_shuffled_token_ids'], [2, 1, 3])
        for scores, permutation, k in (([1, 2], [0, 0], 1), ([np.nan, 1], [0, 1], 1),
                                       ([1, 2], [0, 1], 3), ([1, 2], [0, 1], True)):
            with self.assertRaises(ValueError): rd.ranked_candidates(scores, permutation, k)


if __name__ == '__main__':
    unittest.main()
