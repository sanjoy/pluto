"""Toy exact-attention oracles for the weight-only bounds; no real model/data."""

import unittest

import numpy as np

from .attention_certificates import error_bound, top1_certificate


class AttentionCertificatesTest(unittest.TestCase):
    def test_error_bound_covers_exact_correction_with_signed_heads(self):
        rng = np.random.default_rng(17)
        for scale in (0., .01, .3, 3., 1000.):
            for _ in range(20):
                gaps, values = rng.normal(size=8) * scale, rng.normal(size=(8, 7))
                exact = (.5 * np.tanh(gaps / 2)) @ values
                linear = (gaps / 4) @ values
                bound = error_bound(gaps, np.linalg.norm(values, axis=1))
                self.assertLessEqual(np.linalg.norm(exact - linear), bound + 1e-12)

    def test_cancellation_does_not_turn_stacked_bound_into_false_lower_bound(self):
        gaps, values = [2., 2.], np.array([[1., 0.], [-1., 0.]])
        exact = (.5 * np.tanh(np.array(gaps) / 2)) @ values
        linear = (np.array(gaps) / 4) @ values
        np.testing.assert_array_equal(exact, linear)
        self.assertGreater(error_bound(gaps, np.linalg.norm(values, axis=1)), 0)

    def test_small_gap_and_saturated_bounds(self):
        self.assertAlmostEqual(error_bound([.1], [2.]), .1**3 * 2 / 48)
        self.assertAlmostEqual(error_bound([10.], [2.]), 5.)
        self.assertEqual(error_bound([0.], [100.]), 0.)
        self.assertEqual(error_bound([100.], [0.]), 0.)

    def test_tiny_gap_large_value_bound_does_not_underflow_early(self):
        result = error_bound([1e-200], [1e300])
        self.assertGreater(result, 0.)
        self.assertAlmostEqual(result / (1e-300 / 48), 1., places=14)

    def test_sufficient_condition_implies_exact_same_top1(self):
        rng = np.random.default_rng(29)
        certified = 0
        for _ in range(100):
            targets = rng.normal(size=(20, 7))
            targets -= targets.mean(axis=0)
            values, gaps = rng.normal(size=(3, 7)), rng.normal(size=3) * .05
            baseline = rng.normal(size=7)
            approximate = baseline + (gaps / 4) @ values
            exact = baseline + (.5 * np.tanh(gaps / 2)) @ values
            scores = targets @ approximate
            order = np.argsort(-scores)
            norms = np.linalg.norm(targets, axis=1)
            result = top1_certificate(scores[order[0]] - scores[order[1]],
                                      error_bound(gaps, np.linalg.norm(values, axis=1)),
                                      norms[order[0]], norms.max())
            if result['ideal_arithmetic_sufficient_condition_holds']:
                certified += 1
                self.assertEqual(order[0], np.argmax(targets @ exact))
        self.assertGreater(certified, 90)

    def test_strict_margin_and_inconclusive_bound(self):
        self.assertFalse(top1_certificate(1., .5, 1., 1.)['ideal_arithmetic_sufficient_condition_holds'])
        self.assertTrue(top1_certificate(1.1, .5, 1., 1.)['ideal_arithmetic_sufficient_condition_holds'])
        self.assertFalse(top1_certificate(0., 0., 1., 1.)['ideal_arithmetic_sufficient_condition_holds'])
        self.assertTrue(top1_certificate(.01, 0., 1., 1.)['ideal_arithmetic_sufficient_condition_holds'])

    def test_invalid_inputs(self):
        for gaps, norms in [([], []), ([1.], [1., 2.]), ([[1.]], [[1.]]),
                            ([np.inf], [0.]), ([1.], [-1.]), ([1.], [np.nan])]:
            with self.assertRaises(ValueError):
                error_bound(gaps, norms)
        with self.assertRaises(ValueError):
            error_bound([1e308], [1e308])
        for args in [(-1., 0., 1., 1.), (1., np.inf, 1., 1.),
                     (1., 0., 2., 1.), (1., 1e308, 1e308, 1e308)]:
            with self.assertRaises(ValueError):
                top1_certificate(*args)


if __name__ == '__main__':
    unittest.main()
