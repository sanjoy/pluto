"""Dense geometry oracles on small synthetic matrices; no model or corpus."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import embedding_motion as motion
from .checkpoint import GPT2Checkpoint, GPT2Config, tensor_manifest


def dense_energies(before, delta, scale=1.):
    score = before@delta.T
    # Form the dense vocabulary-sized difference from D. Subtracting two
    # nearly equal endpoint Gram matrices is a less accurate tiny-D oracle.
    gram = scale*(score+score.T)+delta@delta.T
    return {"score_frobenius": float(np.sum(score**2)),
            "score_symmetric_frobenius": float(np.sum(((score+score.T)/2)**2)),
            "score_antisymmetric_frobenius": float(np.sum(((score-score.T)/2)**2)),
            "gram_change_frobenius": float(np.sum(gram**2))}


class GeometryTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(405)
        self.before = self.rng.normal(size=(17, 4))
        self.after = self.before+.2*self.rng.normal(size=(17, 4))

    def assert_dense(self, result, before, delta, scale=1.):
        for name, value in dense_energies(before, delta, scale).items():
            actual = result[name]
            error = abs(actual["computed_signed_squared_norm"]-value)
            self.assertLessEqual(error, actual["squared_roundoff_allowance"]+1e-25, name)
            self.assertAlmostEqual(actual["squared_norm"], value, delta=1e-10*max(1., value))

    def test_coordinate_contractions_equal_dense_vocabulary_matrices(self):
        before = self.before-self.before.mean(axis=0)
        delta = self.after-self.after.mean(axis=0)-before
        self.assert_dense(motion._energies(before, delta), before, delta)
        self.assert_dense(motion._energies(before, delta, 2.3), before, delta, 2.3)

    def test_all_three_motion_views_equal_independent_dense_oracle(self):
        result = motion.centered_motion(self.before, self.after)
        f0, f1 = self.before-self.before.mean(axis=0), self.after-self.after.mean(axis=0)
        u, _, vh = np.linalg.svd(f1.T@f0)
        aligned = f1@(u@vh)
        q = np.sum(aligned*f0)/np.sum(f0*f0)
        self.assert_dense(result["raw"], f0, f1-f0)
        self.assert_dense(result["procrustes"], f0, aligned-f0)
        self.assert_dense(result["rotation_and_fitted_scale"], f0, aligned-q*f0, q)
        self.assertEqual(result["fitted_scale"], q)

    def test_uncentered_mean_energy_has_exact_real_orthogonal_decomposition(self):
        after = self.after+np.array([4., -2., 1., .5])
        result = motion.centered_motion(self.before, after)
        delta = after-self.before
        mean = delta.mean(axis=0)
        full = np.sum(delta**2)
        shared = len(delta)*np.sum(mean**2)
        centered = np.sum((delta-mean)**2)
        self.assertAlmostEqual(result["full_raw_delta_frobenius_norm"], np.sqrt(full), places=13)
        self.assertAlmostEqual(result["shared_mean_energy_fraction"], shared/full, places=14)
        self.assertAlmostEqual(full, shared+centered, places=11)
        self.assertLess(abs(result["mean_energy_decomposition"]["signed_fp64_replay_error"]), 2e-12)
        unchanged = motion.centered_motion(self.before, self.before)
        self.assertIsNone(unchanged["shared_mean_energy_fraction"])

    def test_independent_endpoint_translation_leaves_centered_geometry_unchanged(self):
        original = motion.centered_motion(self.before, self.after)
        shifted = motion.centered_motion(self.before+[2., -1., 4., .5], self.after+[-3., .7, 1., 8.])
        for mode in ("raw", "procrustes", "rotation_and_fitted_scale"):
            for name in dense_energies(self.before, self.after):
                self.assertAlmostEqual(original[mode][name]["norm"], shifted[mode][name]["norm"], delta=2e-12)
        self.assertNotEqual(original["mean_motion_norm"], shifted["mean_motion_norm"])

    def test_common_orthogonal_coordinates_preserve_metrics(self):
        rotation, _ = np.linalg.qr(self.rng.normal(size=(4, 4)))
        a = motion.centered_motion(self.before, self.after)
        b = motion.centered_motion(self.before@rotation, self.after@rotation)
        for mode in ("raw", "procrustes", "rotation_and_fitted_scale"):
            for name in dense_energies(self.before, self.after):
                self.assertAlmostEqual(a[mode][name]["norm"], b[mode][name]["norm"], delta=2e-12)

    def test_pure_embedding_geometry_rotation_has_raw_directed_motion(self):
        rotation, _ = np.linalg.qr(self.rng.normal(size=(4, 4)))
        result = motion.centered_motion(self.before, self.before@rotation)
        self.assertGreater(result["raw"]["delta_frobenius_norm"], 1)
        self.assertGreater(result["raw"]["score_antisymmetric_frobenius"]["norm"], .1)
        gram = result["raw"]["gram_change_frobenius"]
        self.assertLessEqual(gram["squared_norm"], gram["squared_roundoff_allowance"])
        self.assertLess(result["procrustes"]["delta_frobenius_norm"], 2e-14)
        self.assertAlmostEqual(result["fitted_scale"], 1., places=14)

    def test_independent_rotation_and_scale_removed_only_by_declared_projection(self):
        rotation, _ = np.linalg.qr(self.rng.normal(size=(4, 4)))
        result = motion.centered_motion(self.before, 1.7*self.before@rotation+np.arange(4))
        self.assertAlmostEqual(result["fitted_scale"], 1.7, places=14)
        norm_before = np.linalg.norm(self.before-self.before.mean(axis=0))
        self.assertAlmostEqual(result["procrustes"]["delta_frobenius_norm"], .7*norm_before, places=13)
        self.assertLess(result["rotation_and_fitted_scale"]["delta_frobenius_norm"], 3e-14)
        self.assertEqual(result["rotation_and_fitted_scale"]["gram_reference_scale"], result["fitted_scale"])

    def test_procrustes_stationarity_does_not_make_token_score_symmetric(self):
        # F^TF=diag(1,4); F^T delta=[[0,.4],[.4,0]] is symmetric.
        # The endpoint covariance is SPD, so identity is already optimal R.
        f = np.array([[1/np.sqrt(2), 0], [-1/np.sqrt(2), 0],
                      [0, np.sqrt(2)], [0, -np.sqrt(2)]])
        a = np.array([[0., .4], [.1, 0.]])
        result = motion.centered_motion(f, f+f@a)
        self.assertLess(result["procrustes"]["coordinate_cross_skew_norm"], 1e-14)
        self.assertGreater(result["procrustes"]["score_antisymmetric_frobenius"]["norm"], .1)

    def test_zero_and_constant_endpoints_have_defined_zero_motion(self):
        for before, after in ((np.zeros((8, 3)), np.zeros((8, 3))),
                              (np.ones((8, 3)), np.full((8, 3), 7.))):
            result = motion.centered_motion(before, after)
            self.assertEqual(result["alignment_status"], "both_centered_zero")
            self.assertIsNone(result["fitted_scale"])
            self.assertEqual(result["raw"]["delta_frobenius_norm"], 0)

    def test_rank_deficient_alignment_is_declared_unavailable(self):
        before = np.arange(12).reshape(6, 2)
        result = motion.centered_motion(before, 2*before)
        self.assertIn("nonunique", result["alignment_status"])
        self.assertIsNone(result["procrustes"])
        self.assertIsNone(result["rotation_and_fitted_scale"])
        self.assertGreater(result["raw"]["delta_frobenius_norm"], 0)
        zero_before = motion.centered_motion(np.zeros((6, 2)), before)
        self.assertIsNone(zero_before["fitted_scale"])

    def test_roundoff_clamps_only_negative_values_inside_declared_allowance(self):
        tiny = motion._squared_norm(-1e-12, 2e-12, "toy")
        self.assertEqual(tiny["norm"], 0)
        self.assertTrue(tiny["negative_roundoff_clamped"])
        positive = motion._squared_norm(1e-12, 2e-12, "toy")
        self.assertEqual(positive["norm"], 1e-6)
        self.assertFalse(positive["negative_roundoff_clamped"])
        with self.assertRaises(ArithmeticError): motion._squared_norm(-3e-12, 2e-12, "toy")
        with self.assertRaises(ArithmeticError): motion._squared_norm(np.nan, 2e-12, "toy")

    def test_small_motion_cancellation_error_within_reported_allowance(self):
        before = self.before-self.before.mean(axis=0)
        for scale in (1e-7, 1e-3, 1.):
            delta = scale*self.rng.normal(size=before.shape)
            self.assert_dense(motion._energies(before, delta), before, delta)

    def test_invalid_shapes_types_and_nonfinite_inputs(self):
        for bad in ([1, 2], np.empty((0, 3)), [[np.nan]], [[np.inf]], [[True]], [["3"]]):
            with self.assertRaises(ValueError): motion.centered_motion(bad, bad)
        with self.assertRaises(ValueError): motion.centered_motion(np.ones((3, 2)), np.ones((4, 2)))
        with self.assertRaises(ArithmeticError): motion.centered_motion(np.full((3, 2), 1e308), np.zeros((3, 2)))


class FilesTest(unittest.TestCase):
    def test_exclusive_output_and_cli_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"out.json"
            motion.write_report(path, {"value": 1})
            with self.assertRaises(FileExistsError): motion.write_report(path, {})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})
            with self.assertRaises(FileExistsError): motion.audit([], path)
        with mock.patch.object(motion, "audit") as run:
            motion.main(["--checkpoint-root", "/tmp/toy", "--output", "/tmp/new.json"])
            self.assertEqual(run.call_args.args[0], [Path(f"/tmp/toy/step_{step}") for step in motion.STEPS])

    def test_fixed_interval_paths_and_output_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root/f"step_{step}" for step in motion.STEPS]
            with self.assertRaises(ValueError): motion.audit(paths[::-1], root/"new.json")
            with self.assertRaises(ValueError): motion.audit(paths[:-1], root/"new.json")
            with self.assertRaises(ValueError): motion.audit(paths, paths[0]/"new.json")

    def test_all_endpoint_provenance_and_mutation_guard_with_actual_tiny_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = GPT2Config(vocab_size=7, padded_vocab_size=8, context_length=3,
                                n_layers=8, d_model=4, n_heads=2, d_ff=6)
            rng = np.random.default_rng(64)
            initial = rng.normal(size=(8, 4)).astype(np.float32)
            checkpoints = {}
            for index, step in enumerate(motion.STEPS):
                directory = root/f"step_{step}"
                directory.mkdir()
                for spec in tensor_manifest(config):
                    values = np.zeros(spec.shape, dtype="<f4")
                    if spec.name == "token_embedding.weight":
                        values = initial+.01*index*rng.normal(size=initial.shape).astype(np.float32)
                    (directory/spec.filename).write_bytes(values.astype("<f4").tobytes())
                checkpoints[directory] = GPT2Checkpoint(directory, config, check_finite=True)
            paths = list(checkpoints)
            with mock.patch.object(motion, "GPT2Checkpoint", side_effect=lambda p, **kw: checkpoints[Path(p)]):
                report = motion.audit(paths, root/"report.json")
                self.assertEqual(len(report["endpoints"]), 5)
                self.assertTrue(all(len(p["weight_sha256"])==100 for p in report["endpoints"]))
                self.assertEqual(len(report["intervals"]), 4)
                self.assertTrue(report["all_500_endpoint_weight_hashes_unchanged"])
                original = motion.centered_motion
                mutated = False

                def change_after_snapshot(before, after):
                    nonlocal mutated
                    if not mutated:
                        (paths[-1]/"weight_99.bin").write_bytes(np.ones(4, dtype="<f4").tobytes())
                        mutated = True
                    return original(before, after)

                with mock.patch.object(motion, "centered_motion", side_effect=change_after_snapshot):
                    with self.assertRaisesRegex(ValueError, "changed during audit"):
                        motion.audit(paths, root/"invalid.json")
                self.assertFalse((root/"invalid.json").exists())


if __name__ == "__main__":
    unittest.main()
