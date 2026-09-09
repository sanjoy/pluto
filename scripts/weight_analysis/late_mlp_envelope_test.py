"""Scalar-envelope tests use synthetic numbers, never real models or text."""

import unittest
from unittest import mock

import numpy as np

from . import late_mlp_envelope as envelope
from .late_mlp_polynomial import GELU_A, GELU_C, gelu_derivatives


def phi(value):
    return .5*value*(1+np.tanh(GELU_C*(value+GELU_A*value**3)))


class EnvelopeTest(unittest.TestCase):
    def test_analytic_gelu_envelope_including_negative_arguments(self):
        values = np.linspace(-20, 20, 1001)
        self.assertTrue(np.all(np.abs(phi(values)) <= np.abs(values)))

    def test_reported_lower_bound_below_actual_scalar_error_on_toys(self):
        anchor = np.array([-2., 0., 1.])
        g, h = gelu_derivatives(anchor)
        z = np.linspace(-20, 20, 100).reshape(2, 50, 1)*np.ones(3)
        result = envelope.envelope_bounds(anchor, phi(anchor), g, h, z)
        # Candidate-dependent phi is permitted ONLY in this independent toy
        # test oracle, never in the production envelope calculation.
        actual_error = np.abs(result["approximation"]-phi(anchor+z))
        self.assertTrue(np.all(result["nominal_lower_bound"] <= actual_error+1e-12))
        self.assertTrue(np.all(result["slack_adjusted_excess"] <= result["nominal_lower_bound"]))
        self.assertGreater(result["slack_adjusted_excess"].max(), 1)

    def test_zero_increment_cannot_violate_the_envelope(self):
        anchor = np.array([-2., 0., 3.])
        g, h = gelu_derivatives(anchor)
        result = envelope.summarize(anchor, phi(anchor), g, h, np.zeros((2, 4, 3)))
        self.assertEqual(result["violating_slots"], 0)
        self.assertEqual(result["prefixes"], 8)
        self.assertEqual(result["neuron_slots"], 24)

    def test_summary_counts_prefixes_slots_neurons_and_thresholds(self):
        # T=z^2; envelope |z|. Entries2 and3 have nominal excess2 and6.
        z = np.array([[[0., 2.], [1., 0.]], [[3., 0.], [0., 0.]]])
        result = envelope.summarize(np.zeros(2), np.zeros(2), np.zeros(2), np.full(2, 2.), z)
        self.assertEqual(result["violating_slots"], 2)
        self.assertEqual(result["violating_prefixes"], 2)
        self.assertEqual(result["restarts_with_any_violation"], 2)
        self.assertEqual(result["distinct_neurons_violating"], 2)
        self.assertEqual(result["violating_slot_fraction"], .25)
        self.assertEqual(result["slack_adjusted_exceeds_threshold"]["1.0"], {"slots": 2, "prefixes": 2})
        self.assertEqual(result["slack_adjusted_exceeds_threshold"]["10.0"], {"slots": 0, "prefixes": 0})
        self.assertLess(result["maximum_slack_adjusted_excess"], 6)
        self.assertEqual(result["maximum_nominal_lower_bound"], 6)

    def test_arithmetic_slack_excludes_ulp_sized_excess(self):
        result = envelope.envelope_bounds([0.], [1.+np.finfo(float).eps], [0.], [0.], [1.])
        self.assertGreater(result["nominal_lower_bound"][0], 0)
        self.assertEqual(result["slack_adjusted_excess"][0], 0)

    def test_nonviolation_is_not_an_accuracy_certificate(self):
        anchor = np.array([0.]); g, h = gelu_derivatives(anchor)
        z = np.array([.5])
        result = envelope.envelope_bounds(anchor, phi(anchor), g, h, z)
        self.assertEqual(result["nominal_lower_bound"][0], 0)
        self.assertGreater(abs(result["approximation"][0]-phi(z)[0]), 1e-3)

    def test_envelope_arithmetic_calls_no_nonlinear_candidate_function(self):
        with mock.patch.object(np, "tanh", side_effect=AssertionError("nonlinear candidate call")):
            result = envelope.envelope_bounds([0.], [0.], [.5], [1.], np.array([[[8.]]]))
            self.assertGreater(result["slack_adjusted_excess"][0, 0, 0], 0)

    def test_rejects_nonfinite_incompatible_or_overflowing_inputs(self):
        for args in (([0], [0], [0, 1], [1], [1]),
                     ([0], [0], [1], [1], [np.nan]),
                     ([0], [0], [1], [1], [1e308]),
                     ([], [], [], [], [])):
            with self.assertRaises(ValueError): envelope.envelope_bounds(*args)
        with self.assertRaises(ValueError):
            envelope.summarize([0], [0], [1], [1], np.ones((4, 1)))

    def test_signed_readout_cancellation_prevents_summing_scalar_bounds(self):
        result = envelope.envelope_bounds([0., 0.], [0., 0.], [.5, .5], [1., 1.], [[8., 8.]])
        self.assertTrue(np.all(result["slack_adjusted_excess"] > 0))
        self.assertEqual((result["approximation"] @ np.array([1., -1.])).item(), 0)


if __name__ == "__main__":
    unittest.main()
