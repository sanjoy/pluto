"""CPU regression tests; NumPy lives in the experiment's Python environment."""

import itertools
import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
    import analyze_dataset_weights as analysis
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    np = None


@unittest.skipIf(np is None, "experiment analysis requires NumPy")
class AnalysisTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = {"layers": 0, "model_width": 2, "feed_forward_width": 4,
                      "context_length": 3, "vocabulary_size": 5}
        self.layout = analysis.tensor_layout(self.model)

    def checkpoint(self, name, values):
        path = self.root / name
        path.mkdir()
        for i, tensor in enumerate(self.layout):
            np.asarray(values[tensor.start:tensor.stop], dtype="<f4").tofile(
                path / f"weight_{i}.bin")
        return name

    def fixture(self):
        initial = np.arange(self.layout[-1].stop) * .01
        self.checkpoint("initial", initial)
        trials = []
        permutations = ([0, 1, 2, 3, 4], [1, 0, 2, 3, 4],
                        [0, 2, 1, 3, 4], [2, 1, 0, 3, 4],
                        [0, 1, 2, 3, 4])
        names = ("baseline", "train_one", "train_two", "heldout", "baseline_repeat")
        for index, (name, permutation) in enumerate(zip(names, permutations)):
            permutation = np.array(permutation)
            (self.root / f"{name}.tokens").write_text(
                " ".join(map(str, permutation[:4])) + "\n")
            np.save(self.root / f"{name}.npy", permutation)
            delta = np.full(len(initial), 1.0 + (index if index < 4 else 0))
            self.checkpoint(name, initial + delta)
            trials.append({"id": name, "family": "token_rename",
                           "split": "test" if index == 3 else "control" if index == 4 else "train",
                           "status": "verified", "success": True, "step": 100,
                           "token_corpus": f"{name}.tokens", "permutation": f"{name}.npy",
                           "initial_checkpoint": "initial", "final_checkpoint": name,
                           "parameters": len(initial), "errors": 0})
        return {"model": self.model, "training_fingerprint": "fixed-seed-schedule", "trials": trials}

    def analyze(self, summary):
        path = self.root / "summary.json"
        path.write_text(json.dumps(summary))
        return analysis.analyze(path)

    def test_actual_model_layout(self):
        layout = analysis.tensor_layout({"layers": 4, "model_width": 10,
            "feed_forward_width": 20, "context_length": 27, "vocabulary_size": 4475})
        self.assertEqual(len(layout), 52)
        self.assertEqual(layout[-1].stop, 48680)
        self.assertEqual(layout[4].name, "block_0.attention.qkv.matrix")
        self.assertEqual(layout[4].shape, (10, 30))
        self.assertEqual(layout[10].shape, (10, 20))

    def test_checkpoint_sizes_and_nonfinite_rejected(self):
        size = self.layout[-1].stop
        path = self.checkpoint("checkpoint", np.arange(size))
        values, digest = analysis.read_checkpoint(self.root / path, self.layout)
        np.testing.assert_array_equal(values, np.arange(size))
        self.assertEqual(len(digest), 64)
        (self.root / path / "weight_0.bin").write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "size"):
            analysis.read_checkpoint(self.root / path, self.layout)
        values = np.arange(size, dtype=float)
        values[0] = np.nan
        path = self.checkpoint("nonfinite", values)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            analysis.read_checkpoint(self.root / path, self.layout)

    def test_hamming_is_invariant_to_numeric_label_values(self):
        tokens = np.array([[0, 1, 2], [2, 1, 0], [1, 0, 2]])
        renamed = np.array([10000, 1, 999999])[tokens]
        np.testing.assert_array_equal(analysis.hamming(tokens, tokens),
                                      analysis.hamming(renamed, renamed))

    def test_constant_function_is_predicted_without_test_targets(self):
        tokens = np.array([[0, 1], [1, 0], [2, 1]])
        outputs = np.tile([2.0, 3.0, -7.0], (3, 1))
        predictions, settings = analysis.fit_predict(tokens, outputs, np.array([[3, 1]]))
        for name in ("training_mean_delta", "nearest_dataset", "hamming_kernel_ridge"):
            np.testing.assert_allclose(predictions[name], [[2, 3, -7]], atol=1e-8)
        self.assertIn("training_leave_one_out_mse", settings)

    def test_kernel_leave_one_out_formula_matches_refitting(self):
        tokens = np.array([[0, 1], [1, 0], [2, 1], [0, 2]])
        outputs = np.array([[1, 7], [2, 3], [-1, 2], [5, 11]], dtype=float)
        distances = analysis.hamming(tokens, tokens)
        _, inverse = analysis.kernel_weights(distances, distances, .3, .1)
        implied = outputs - (inverse @ outputs) / np.diag(inverse)[:, None]
        for index in range(4):
            others = np.arange(4) != index
            weights, _ = analysis.kernel_weights(distances[others][:, others],
                                                 distances[index:index + 1, others], .3, .1)
            np.testing.assert_allclose(implied[index], (weights @ outputs[others])[0], atol=1e-12)

    def test_known_categorical_mapping_generalizes_to_unseen_datasets(self):
        tokens = np.array(list(itertools.product([0, 1], repeat=4)))
        outputs = (tokens[:, 0] * 2 - 1)[:, None] * np.array([[2.0, 3.0]])
        predictions, _ = analysis.fit_predict(tokens[1:-1], outputs[1:-1], tokens[[0, -1]])
        errors = analysis.metrics(outputs[[0, -1]].reshape(-1),
                                  predictions["hamming_kernel_ridge"].reshape(-1))
        self.assertLess(errors["normalized_rmse"], .01)
        # The targets for these two unseen datasets are never passed to fit.
        self.assertEqual(analysis.metrics(outputs[[0, -1]].reshape(-1),
                         predictions["training_mean_delta"].reshape(-1))["normalized_rmse"], 1)

    def test_alignment_undoes_embedding_permutation_only(self):
        values = np.arange(self.layout[-1].stop)
        permutation = np.array([2, 0, 1, 4, 3])
        renamed = analysis.align_embeddings(values, np.argsort(permutation), self.layout)
        aligned = analysis.align_embeddings(renamed, permutation, self.layout)
        np.testing.assert_array_equal(aligned, values)
        np.testing.assert_array_equal(renamed[self.layout[0].stop:], values[self.layout[0].stop:])

    def test_zero_target_norm_is_explicitly_undefined(self):
        metric = analysis.metrics(np.zeros(3), np.ones(3))
        self.assertIsNone(metric["normalized_rmse"])
        self.assertIsNone(metric["cosine"])
        self.assertEqual(metric["rmse"], 1)

    def test_end_to_end_heldout_and_control(self):
        report = self.analyze(self.fixture())
        group = report["groups"][0]
        self.assertTrue(group["controls"][0]["identical_weights"])
        all_runs = group["cohorts"][0]
        self.assertEqual(all_runs["heldout_ids"], ["heldout"])
        self.assertEqual(len(all_runs["training_ids"]), 3)
        predictions = all_runs["representations"]["raw"]["predictions"]
        self.assertIn("permuted_baseline_final", predictions)
        self.assertLess(predictions["training_mean_delta"]["overall"]["normalized_rmse"], 1)
        self.assertEqual(len(predictions["training_mean_delta"]["trials"][0]["tensors"]), 4)
        html = analysis.render_html(report)
        self.assertIn("held-out analysis", html)
        self.assertNotIn("<script src", html)
        json.dumps(report, allow_nan=False)

    def test_duplicate_training_heldout_dataset_is_rejected(self):
        summary = self.fixture()
        summary["trials"][3]["token_corpus"] = "train_one.tokens"
        summary["trials"][3]["permutation"] = "train_one.npy"
        with self.assertRaisesRegex(ValueError, "leakage"):
            self.analyze(summary)

    def test_mismatched_step_or_initialization_is_not_pooled(self):
        summary = self.fixture()
        for field, value in (("step", 101), ("initial_checkpoint", "train_one"),
                             ("training_fingerprint", "different-schedule")):
            with self.subTest(field=field):
                modified = json.loads(json.dumps(summary))
                modified["trials"][3][field] = value
                report = self.analyze(modified)
                self.assertEqual(len(report["groups"]), 2)
                self.assertTrue(all(cohort["status"] != "evaluated" for group in report["groups"]
                                    for cohort in group["cohorts"]))

    def test_incomplete_trials_are_never_scored(self):
        summary = self.fixture()
        summary["trials"][3]["status"] = "incomplete"
        report = self.analyze(summary)
        self.assertEqual(report["skipped"][0]["id"], "heldout")
        self.assertTrue(all(cohort["status"] != "evaluated"
                            for cohort in report["groups"][0]["cohorts"]))

    def test_success_and_failure_cohorts_are_distinct(self):
        summary = self.fixture()
        summary["trials"][3]["success"] = False
        summary["trials"][3]["errors"] = 3
        cohorts = self.analyze(summary)["groups"][0]["cohorts"]
        self.assertEqual(cohorts[0]["status"], "evaluated")
        self.assertEqual(cohorts[1]["heldout_ids"], [])
        self.assertEqual(cohorts[2]["heldout_ids"], ["heldout"])
        self.assertNotEqual(cohorts[2]["status"], "evaluated")

    def test_permutation_must_explain_dataset(self):
        summary = self.fixture()
        summary["trials"][3]["permutation"] = "baseline.npy"
        with self.assertRaisesRegex(ValueError, "declared token permutation"):
            self.analyze(summary)

    def test_report_escapes_user_controlled_text(self):
        report = self.analyze(self.fixture())
        report["source_summary"] = "<script>alert(1)</script>"
        self.assertNotIn("<script>", analysis.render_html(report))


if __name__ == "__main__":
    unittest.main()
