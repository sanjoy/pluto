"""Independent scalar-attention oracles for the weight-only error bound."""

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from .routing_audit import audit_probe, head_components, main, summarize_bounds
from .trigram import TrigramProbe


class RoutingAuditTest(unittest.TestCase):
    def test_bound_is_below_actual_error_and_approaches_saturation(self):
        gaps = np.array([-100., -10., -2., -0.1, 0., 0.1, 2., 10., 100.]).reshape(1, 1, -1)
        norms = np.ones_like(gaps)
        actual = np.abs(gaps / 4 - np.tanh(gaps / 2) / 2)
        bound = np.maximum(np.abs(gaps) - 2, 0) / 4
        self.assertTrue(np.all(bound <= actual + 1e-15))
        self.assertAlmostEqual(float(bound[0, 0, -1]), float(actual[0, 0, -1]), places=10)
        report = summarize_bounds(gaps, norms)["stacked_heads"]
        self.assertAlmostEqual(report["error_lower_bound_over_linear_norm"],
                               np.linalg.norm(bound) / np.linalg.norm(gaps / 4))

    def test_randomized_vector_bound_against_exact_correction(self):
        rng = np.random.default_rng(42)
        gaps = rng.normal(size=(3, 4, 5)) * 8
        vectors = rng.normal(size=(3, 4, 5, 7))
        norms = np.linalg.norm(vectors, axis=-1)
        error = (gaps / 4 - np.tanh(gaps / 2) / 2)[..., None] * vectors
        report = summarize_bounds(gaps, norms)["stacked_heads"]
        self.assertLessEqual(report["error_lower_bound_frobenius_norm"], np.linalg.norm(error))
        self.assertGreaterEqual(report["error_upper_bound_frobenius_norm"], np.linalg.norm(error))

    def test_small_gap_has_informative_upper_bound_without_exact_model(self):
        gaps = np.full((1, 2, 3), 0.1)
        report = summarize_bounds(gaps, np.ones_like(gaps))["stacked_heads"]
        self.assertAlmostEqual(report["error_upper_bound_over_linear_norm"], 0.1**2 / 12)
        self.assertEqual(report["error_lower_bound_over_linear_norm"], 0)

    def test_zero_value_difference_and_zero_linear_are_not_bad_heads(self):
        report = summarize_bounds(np.full((1, 2, 2), 100.), np.zeros((1, 2, 2)))["stacked_heads"]
        self.assertEqual(report["error_lower_bound_frobenius_norm"], 0)
        self.assertIsNone(report["error_lower_bound_over_linear_norm"])
        self.assertIsNone(report["active_fraction_absolute_gap_above"]["2"])
        report = summarize_bounds(np.zeros((1, 2, 2)), np.ones((1, 2, 2)))["stacked_heads"]
        self.assertIsNone(report["error_lower_bound_over_linear_norm"])

    def test_large_finite_values_do_not_overflow_intermediate_squared_norm(self):
        report = summarize_bounds(np.full((1, 2, 3), 1e308), np.ones((1, 2, 3)))["stacked_heads"]
        self.assertTrue(np.isfinite(report["linear_correction_frobenius_norm"]))
        self.assertEqual(report["error_upper_bound_over_linear_norm"], 1.0)
        self.assertEqual(report["error_lower_bound_over_linear_norm"], 1.0)

    def test_projected_components_equal_direct_scalar_calculation(self):
        rng = np.random.default_rng(31)
        q, kc, vc = rng.normal(size=(3, 4, 2, 3))
        kp, vp = rng.normal(size=(2, 5, 2, 3))
        wo = rng.normal(size=(2, 3, 6))
        gaps, norms = head_components(q, kp, kc, vp, vc, wo)
        self.assertEqual(gaps.shape, (2, 4, 5))
        for h in range(2):
            for a in range(4):
                for b in range(5):
                    self.assertAlmostEqual(gaps[h, a, b],
                        q[a, h] @ (kp[b, h] - kc[a, h]) / np.sqrt(3))
                    self.assertAlmostEqual(norms[h, a, b],
                        np.linalg.norm((vp[b, h] - vc[a, h]) @ wo[h]))

    def test_probe_adapter_slots_and_broken_pairing(self):
        rng = np.random.default_rng(11)
        size, heads, channels, width = 3, 2, 2, 4
        inputs = rng.normal(size=(2 * size, width))
        q, k, v = rng.normal(size=(3, heads, width, channels))
        output = rng.normal(size=(heads, channels, width))
        for pairing in ((0, 1), (1, 0)):
            contraction = TrigramProbe(inputs, q, k, v, output, ov_pairing=pairing)
            probe = SimpleNamespace(embedding=inputs[:size], contraction=contraction)
            report = audit_probe(probe)
            self.assertEqual(report["stacked_heads"]["head_contexts"], heads * size * size)
            gaps, norms = head_components(contraction.queries[size:], contraction.keys[:size],
                contraction.keys[size:], contraction.values[:size], contraction.values[size:],
                contraction.output.reshape(heads, channels, width))
            self.assertEqual(report, summarize_bounds(gaps, norms))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            summarize_bounds(np.zeros((2, 3)), np.ones((2, 3)))
        with self.assertRaises(ValueError):
            summarize_bounds(np.zeros((1, 2, 3)), np.full((1, 2, 3), -1))
        with self.assertRaises(ValueError):
            summarize_bounds(np.full((1, 2, 3), np.inf), np.ones((1, 2, 3)))
        q = np.ones((2, 1, 2))
        with self.assertRaises(ValueError):
            head_components(q, q, q, q, q, np.ones((2, 1, 2)))

    def test_no_overwrite_before_checkpoint_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text("preserve")
            with self.assertRaises(FileExistsError):
                main(["--checkpoint", "/missing", "--vocabulary-checkpoint", "/missing",
                      "--output", str(path)])
            self.assertEqual(path.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
