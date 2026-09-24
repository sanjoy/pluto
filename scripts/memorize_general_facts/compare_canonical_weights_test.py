"""Exact-comparison tests, including signed zero, NaN payloads and alignment."""

import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
    import compare_canonical_weights as comparison
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    np = None


@unittest.skipIf(np is None, "experiment analysis requires NumPy")
class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = dict(layers=0, model_width=2, feed_forward_width=4,
                          context_length=3, vocabulary_size=5)
        self.layout = comparison.tensor_layout(self.model)
        self.count = self.layout[-1].stop

    def bits(self, values):
        return np.asarray(values, dtype="<f4").view("<u4")

    def write_checkpoint(self, root, trial_id, step, bits):
        path = root / trial_id / "checkpoints/layers_0" / f"step_{step}"
        path.mkdir(parents=True)
        for index, tensor in enumerate(self.layout):
            (path / f"weight_{index}.bin").write_bytes(
                bits[tensor.start:tensor.stop].astype("<u4").tobytes())
        return path

    def fixture(self, root=None):
        root = root or self.root
        root.mkdir(exist_ok=True)
        initial = self.bits(np.arange(self.count) * .01)
        initial[:10] = np.tile(self.bits([.1, -.1]), 5)
        final = self.bits(np.arange(self.count) * .125 + 1)
        trials = []
        for name, values in (("baseline", [0, 1, 2, 3, 4]),
                             ("rename_two", [1, 0, 2, 3, 4]),
                             ("rename_three", [0, 2, 3, 1, 4])):
            permutation = np.asarray(values)
            trial_dir = root / name
            trial_dir.mkdir()
            (trial_dir / "permutation.tsv").write_text(" ".join(map(str, values)) + "\n")
            (trial_dir / "tokens.tsv").write_text(" ".join(map(str, permutation[:4])) + "\n")
            (trial_dir / "command.json").write_text(json.dumps([
                "binary", "--token_order_file=" + str(trial_dir / "permutation.tsv")]))
            self.write_checkpoint(root, name, 0, initial)
            renamed = comparison.align_bits(final, np.argsort(permutation), self.layout)
            self.write_checkpoint(root, name, 10, renamed)
            trials.append(dict(id=name, status="running", canonical_token_order=True,
                               training_fingerprint="same", token_corpus=f"{name}/tokens.tsv",
                               permutation=f"{name}/permutation.tsv",
                               initial_sha256=comparison.digest(initial)))
        summary = dict(status="running", model=self.model, canonical_token_order=True,
                       training_fingerprint="same", trials=trials)
        (root / "summary.json").write_text(json.dumps(summary))
        return summary

    def test_identical_finite_values_are_exact(self):
        values = self.bits([0, 1, -2, .1])
        result = comparison.differences(values, values.copy())
        self.assertTrue(result["exact_finite_match"])
        self.assertEqual(result["bitwise_mismatches"], 0)
        self.assertEqual(result["l2_finite"], 0)

    def test_signed_zero_is_a_bit_difference_not_a_numeric_difference(self):
        result = comparison.differences(self.bits([0, 1]), self.bits([-0., 1]))
        self.assertEqual(result["bitwise_mismatches"], 1)
        self.assertEqual(result["signed_zero_mismatches"], 1)
        self.assertEqual(result["numerical_mismatches"], 0)
        self.assertEqual(result["max_abs_finite"], 0)
        self.assertFalse(result["exact_finite_match"])

    def test_equal_nan_bits_do_not_imply_valid_weights(self):
        values = np.array([0x7fc00001], dtype="<u4")
        result = comparison.differences(values, values)
        self.assertTrue(result["bitwise_identical"])
        self.assertFalse(result["exact_finite_match"])
        self.assertEqual(result["left_nan_count"], 1)
        self.assertEqual(result["numerical_mismatches"], 1)
        self.assertIsNone(result["max_abs_finite"])
        json.dumps(result, allow_nan=False)

    def test_distinct_nan_payloads_and_infinities_are_preserved(self):
        left = np.array([0x7fc00001, 0x7f800000, 0xff800000], dtype="<u4")
        right = np.array([0x7fc00002, 0x7f800000, 0x7f800000], dtype="<u4")
        result = comparison.differences(left, right)
        self.assertEqual(result["bitwise_mismatches"], 2)
        self.assertEqual(result["left_infinity_count"], 2)
        self.assertFalse(result["finite_weights"])

    def test_finite_distances_use_float64_not_uint32_subtraction(self):
        result = comparison.differences(self.bits([1., -2.]), self.bits([-2., 2.]))
        self.assertEqual(result["max_abs_finite"], 4)
        self.assertEqual(result["l2_finite"], 5)
        self.assertEqual(result["numerical_mismatches"], 2)

    def test_read_checkpoint_preserves_all_bits_and_checks_shape(self):
        bits = np.arange(self.count, dtype="<u4")
        bits[0] = 0x7fc00005
        path = self.write_checkpoint(self.root, "one", 0, bits)
        np.testing.assert_array_equal(comparison.read_checkpoint_bits(path, self.layout), bits)
        (path / "weight_0.bin").write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "size"):
            comparison.read_checkpoint_bits(path, self.layout)

    def test_extra_checkpoint_weights_are_rejected(self):
        path = self.write_checkpoint(self.root, "one", 0, self.bits(np.zeros(self.count)))
        (path / "weight_999.bin").write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            comparison.read_checkpoint_bits(path, self.layout)

    def test_permutation_is_bijective_and_preserves_eos(self):
        path = self.root / "mapping.tsv"
        for values in ("0 0 2 3 4", "0 1 2 4 3", "0 1 2 3", "0 1 2 3 5"):
            path.write_text(values)
            with self.assertRaises(ValueError):
                comparison.read_permutation(path, 5)
        path.write_text("1 0 2 3 4")
        np.testing.assert_array_equal(comparison.read_permutation(path, 5), [1, 0, 2, 3, 4])

    def test_pairwise_alignment_including_noninvolutory_permutation(self):
        self.fixture()
        report = comparison.analyze(self.root, steps=(0, 10, 20))
        self.assertEqual(len(report["comparisons"]), 6)
        self.assertEqual(len(report["unavailable"]), 3)
        for pair in report["comparisons"]:
            self.assertTrue(pair["aligned"]["exact_finite_match"])
            self.assertEqual(pair["nonembedding"]["bitwise_mismatches"], 0)
            if pair["step"] == 10:
                self.assertGreater(pair["raw"]["bitwise_mismatches"], 0)
        primary = [pair for pair in report["comparisons"] if pair["kind"] == "permutation_pair"]
        self.assertEqual(len(primary), 2)
        self.assertTrue(all(control["initial_hash_matches_declaration"] for control in report["controls"]))
        self.assertTrue(all(control["identical_initial_embedding_rows"] for control in report["controls"]))
        self.assertTrue(all(control["canonical_order_command_verified"] for control in report["controls"]))
        json.dumps(report, allow_nan=False)

    def test_nonembedding_mismatch_has_precise_tensor_coordinate(self):
        self.fixture()
        path = self.root / "rename_two/checkpoints/layers_0/step_10/weight_1.bin"
        data = np.frombuffer(path.read_bytes(), dtype="<u4").copy()
        data[3] += 1
        path.write_bytes(data.tobytes())
        report = comparison.analyze(self.root, steps=(10,))
        pair = next(pair for pair in report["comparisons"] if pair["left"] == "baseline"
                    and pair["right"] == "rename_two")
        self.assertEqual(pair["aligned"]["bitwise_mismatches"], 1)
        self.assertEqual(pair["nonembedding"]["bitwise_mismatches"], 1)
        self.assertEqual(pair["first_aligned_mismatch_coordinates"][0]["tensor"], "position_embedding")
        self.assertEqual(pair["first_aligned_mismatch_coordinates"][0]["coordinate"], [1, 1])

    def test_partial_checkpoint_is_unavailable_not_equal(self):
        self.fixture()
        path = self.root / "rename_two/checkpoints/layers_0/step_10/weight_0.bin"
        path.write_bytes(b"incomplete")
        report = comparison.analyze(self.root, steps=(10,))
        self.assertEqual(len(report["comparisons"]), 1)
        self.assertEqual(len(report["unavailable"]), 1)
        self.assertEqual(report["unavailable"][0]["id"], "rename_two")

    def test_wrong_corpus_renaming_is_rejected(self):
        self.fixture()
        (self.root / "rename_two/tokens.tsv").write_text("0 1 2 3\n")
        with self.assertRaisesRegex(ValueError, "does not match"):
            comparison.analyze(self.root, steps=(0,))

    def test_mixed_training_configurations_are_rejected(self):
        summary = self.fixture()
        for field, value in (("training_fingerprint", "different"),
                             ("canonical_token_order", False)):
            with self.subTest(field=field):
                changed = json.loads(json.dumps(summary))
                changed["trials"][1][field] = value
                (self.root / "summary.json").write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    comparison.analyze(self.root, steps=(0,))

    def test_recorded_command_must_use_this_trials_mapping(self):
        self.fixture()
        (self.root / "rename_two/command.json").write_text(json.dumps([
            "binary", "--token_order_file=" + str(self.root / "baseline/permutation.tsv")]))
        with self.assertRaisesRegex(ValueError, "recorded command"):
            comparison.analyze(self.root, steps=(0,))

    def test_previous_baseline_compared_without_vocabulary_alignment(self):
        self.fixture()
        old = self.root / "old"
        self.fixture(old)
        report = comparison.analyze(self.root, steps=(0, 10), previous_run_dir=old)
        self.assertEqual(len(report["previous_baseline_comparisons"]), 2)
        self.assertTrue(all(pair["raw"]["exact_finite_match"]
                            for pair in report["previous_baseline_comparisons"]))

    def test_html_is_self_contained_and_escapes_metadata(self):
        self.fixture()
        report = comparison.analyze(self.root, steps=(0, 10))
        report["experiment_status"] = "<script>alert(1)</script>"
        output = comparison.render_html(report)
        self.assertNotIn("<script>", output)
        self.assertIn("&lt;script&gt;", output)
        self.assertIn("renamed runs compared with each other", output)
        self.assertIn("Unavailable does not mean equal", output)


if __name__ == "__main__":
    unittest.main()
