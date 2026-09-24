"""Synthetic, CPU-only checks of the permutation trace's numerical reasoning."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
    import analyze_permutation_trace as analysis
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    np = None


@unittest.skipIf(np is None, "experiment analysis requires NumPy")
class TraceAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_bf16_ties_round_to_even(self):
        source = np.array([0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000], dtype=np.uint32)
        result = analysis.round_bf16(source.view(np.float32)).astype(np.float32).view(np.uint32)
        np.testing.assert_array_equal(result, [0x3f800000, 0x3f820000, 0xbf800000, 0xbf820000])

    def test_bf16_special_values_preserve_class(self):
        source = np.array([0x00000000, 0x80000000, 0x7f800000, 0xff800000,
                           0x7f800001, 0x7fc00000], dtype=np.uint32).view(np.float32)
        result = analysis.round_bf16(source)
        self.assertEqual(result[0], 0)
        self.assertFalse(np.signbit(result[0]))
        self.assertTrue(np.signbit(result[1]))
        self.assertEqual(result[2], np.inf)
        self.assertEqual(result[3], -np.inf)
        self.assertTrue(np.isnan(result[4:]).all())

    def test_fp32_activation_mode_uses_half_mma_operands(self):
        values = np.array([1.001, -1.001])
        np.testing.assert_array_equal(analysis.round_operands(values, "FP32"),
                                      values.astype(np.float32).astype(np.float16).astype(np.float64))
        self.assertFalse(np.array_equal(analysis.round_operands(values, "FP32"),
                                       analysis.round_operands(values, "BF16")))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            analysis.round_operands([1e10], "FP32")

    def test_inverse_token_and_vocabulary_alignment_with_padding(self):
        permutation = np.array([2, 0, 1])
        ids = np.array([2, 0, 1, -1])
        np.testing.assert_array_equal(analysis.align(ids, "token_ids", permutation), [0, 1, 2, -1])
        columns = np.array([[20, 30, 10, 999]])
        np.testing.assert_array_equal(analysis.align(columns, "vocab_columns", permutation),
                                      [[10, 20, 30, 999]])
        np.testing.assert_array_equal(analysis.align(columns.T, "vocab_rows", permutation),
                                      [[10], [20], [30], [999]])

    def test_uniform_shared_embeddings_have_zero_ideal_backbone_gradient(self):
        probabilities = np.full((1, 4), .25)
        probabilities[0, 1] -= 1
        embeddings = np.tile([1.25, -2.0], (4, 1))
        result = analysis.head_reference(probabilities, embeddings, np.zeros((1, 2)), "BF16")
        np.testing.assert_array_equal(result["fp64"], [[0, 0]])
        self.assertEqual(result["gpu_vs_fp64"]["max_abs"], 0)
        # Learning nevertheless starts at the tied head: dE = dLogits.T @ h.
        direct_embedding_gradient = probabilities.T @ np.array([[.5, 2.0]])
        self.assertGreater(np.linalg.norm(direct_embedding_gradient), 0)

    def test_signed_zero_is_not_a_numerical_divergence(self):
        result = analysis.error_metrics(np.array([0.0, 1.0], dtype=np.float32),
                                        np.array([-0.0, 1.0], dtype=np.float32))
        self.assertEqual(result["changed"], 0)
        self.assertEqual(result["bitwise_changed"], 1)
        self.assertEqual(result["signed_zero_only"], 1)
        self.assertEqual(result["max_abs"], 0)

    def test_parameter_update_comparison_removes_pre_step_weights(self):
        permutation = np.array([1, 0])
        before = np.array([[1, 2], [3, 4]], dtype=np.float32)
        gradient = np.array([[.1, .2], [.3, .4]], dtype=np.float32)
        perturbed = gradient.copy()
        perturbed[0, 0] += .0001
        roles = {}
        for role in ("baseline", "renamed"):
            values = {"weights_before/token_embedding": before,
                      "parameter_gradients/token_embedding": gradient if role == "baseline" else perturbed,
                      "weights_after/token_embedding": before - .05 * (gradient if role == "baseline" else perturbed)}
            roles[role] = [analysis.Capture(i, stage, "fp32", data.shape, "vocab_rows", self.root,
                                           data if role == "baseline" else data[permutation])
                           for i, (stage, data) in enumerate(values.items())]
        result = analysis.compare_updates(roles, permutation)[0]
        self.assertEqual(result["before"]["changed"], 0)
        self.assertGreater(result["gradient"]["changed"], 0)
        self.assertGreater(result["update"]["changed"], 0)
        self.assertAlmostEqual(result["update_to_gradient_discrepancy_l2_ratio"], .05, places=3)

    def test_quantized_loss_gradients_can_break_exact_sum_cancellation(self):
        probabilities = np.full((1, 4475), 1 / 4475, dtype=np.float32)
        probabilities[0, 7] -= np.float32(1)
        embeddings = np.tile([.25, -.5], (4475, 1))
        result = analysis.head_reference(probabilities, embeddings, np.zeros((1, 2)), "BF16")
        self.assertGreater(np.linalg.norm(result["rounded_fp64"]),
                           100 * np.linalg.norm(result["fp64"]))
        self.assertGreater(result["operand_quantization"]["l2"], 0)

    def test_references_use_actual_nonuniform_embeddings(self):
        gradient = np.array([[.25, -.75, .5, 0.0]])
        embeddings = np.array([[1.01, 2.02], [3.03, 4.04], [-5.05, 6.06]])
        expected = gradient[:, :3] @ embeddings
        result = analysis.head_reference(gradient, embeddings, expected, "BF16")
        np.testing.assert_array_equal(result["fp64"], expected)
        self.assertEqual(result["gpu_vs_fp64"]["l2"], 0)
        self.assertGreater(result["operand_quantization"]["l2"], 0)

    def test_cross_entropy_reference_permutation_and_ignored_rows(self):
        logits = np.array([[.1, .2, 2.0], [1, -1, 3], [2, 3, 4]])
        targets = np.array([2, 0, -1])
        permutation = np.array([2, 0, 1])
        permuted = logits[:, np.argsort(permutation)]
        loss, gradient = analysis.cross_entropy_reference(logits, targets)
        renamed_targets = targets.copy()
        renamed_targets[:2] = permutation[targets[:2]]
        canonical_loss, canonical_gradient = analysis.cross_entropy_reference(
            analysis.align(permuted, "vocab_columns", permutation),
            analysis.align(renamed_targets, "token_ids", permutation))
        np.testing.assert_array_equal(loss, canonical_loss)
        np.testing.assert_array_equal(gradient, canonical_gradient)
        self.assertEqual(loss[2], 0)
        np.testing.assert_array_equal(gradient[2], 0)
        np.testing.assert_allclose(gradient.sum(axis=1), 0, atol=1e-16)

    def test_cross_entropy_entirely_ignored_is_zero(self):
        loss, gradient = analysis.cross_entropy_reference(np.ones((2, 3)), [-1, -1])
        np.testing.assert_array_equal(loss, 0)
        np.testing.assert_array_equal(gradient, 0)

    def write_role(self, name, records):
        path = self.root / name
        path.mkdir()
        with (path / "tensors.tsv").open("w") as output:
            writer = csv.writer(output, delimiter="\t", lineterminator="\n")
            writer.writerow(["sequence", "stage", "dtype", "shape", "alignment", "file", "bytes"])
            for index, (stage, data, alignment, dtype) in enumerate(records):
                data = np.asarray(data)
                stored = data.astype("<i4" if dtype == "int32" else "<f4")
                if dtype == "bf16":
                    stored = (analysis.round_bf16(data).astype(np.float32).view(np.uint32) >> 16).astype("<u2")
                filename = f"{index:05d}.bin"
                stored.tofile(path / filename)
                writer.writerow([index, stage, dtype, ",".join(map(str, data.shape)),
                                 alignment, filename, stored.nbytes])

    def fixture(self):
        (self.root / "metadata.txt").write_text("compute_type=BF16\ndivergent_step=17\n")
        permutation = np.array([2, 0, 1, 3, 4])
        (self.root / "permutation.tsv").write_text(" ".join(map(str, permutation)))
        embeddings = np.arange(10, dtype=np.float32).reshape(5, 2) * .02
        gradient = np.array([[.2, -.8, .2, .2, .2, 0, 0, 0],
                             [.2, .2, .2, -.8, .2, 0, 0, 0]], dtype=np.float32)
        padded = np.zeros((8, 2))
        padded[:5] = embeddings
        actual = (analysis.round_bf16(gradient) @ analysis.round_bf16(padded)).astype(np.float32)
        for name in ("baseline", "repeat", "renamed"):
            table, dlogits = embeddings.copy(), gradient.copy()
            targets = np.array([1, 3])
            gpu = actual.copy()
            if name == "renamed":
                table[permutation] = embeddings
                dlogits[:, permutation] = gradient[:, :5]
                targets = permutation[targets]
                gpu[0, 0] += .00001  # Synthetic reduction discrepancy, not a GPU claim.
            self.write_role(name, [
                ("weights_before/token_embedding", table, "vocab_rows", "fp32"),
                ("targets", targets, "token_ids", "int32"),
                ("fwd/embedding/0", np.ones((2, 2)), "none", "bf16"),
                ("loss/dlogits", dlogits, "vocab_columns", "fp32"),
                ("bwd/gpt2/LayerNormLayer/0", gpu, "none", "fp32"),
            ])

    def test_complete_synthetic_trace_distinguishes_inputs_and_gpu_residual(self):
        self.fixture()
        report = analysis.analyze(self.root)
        self.assertIsNone(report["first_repeat_difference"])
        self.assertEqual(report["first_renamed_difference"]["stage"], "bwd/gpt2/LayerNormLayer/0")
        compared = report["head_gradient"]["baseline_vs_renamed"]
        self.assertEqual(compared["pre_step_embeddings_aligned"]["changed"], 0)
        self.assertEqual(compared["dlogits_aligned"]["changed"], 0)
        self.assertEqual(compared["canonical_rounded_operand_fp64_reference"]["changed"], 0)
        self.assertGreater(compared["gpu_input_gradient"]["max_abs"], 0)
        self.assertGreater(compared["remaining_gpu_residual"]["max_abs"], 0)
        self.assertEqual(report["head_gradient"]["selected_rows"][0]["row"], 0)
        self.assertEqual(report["head_gradient"]["changed_channels"]["gpu_input_gradient_channels"], [0])
        self.assertIn("operand quantization", analysis.render_html(report))
        json.dumps(report, allow_nan=False)

    def test_batch_timeline_and_independent_head_discrepancy_reported(self):
        self.fixture()
        (self.root / "batches.tsv").write_text(
            "step\tchanged_input_ids\tchanged_target_ids\tunaligned_embedding_values_before\n"
            "1\t0\t0\t0\n2\t1\t0\t0\n3\t0\t0\t4\n")
        report = analysis.analyze(self.root)
        self.assertEqual(report["batches"][-1]["unaligned_embedding_values_before"], 4)
        self.assertIn("exactly matching aligned rounded operands", report["interpretations"][0])
        self.assertIn("Batch timeline", analysis.render_html(report))

    def test_incorrect_manifest_byte_count_rejected(self):
        self.fixture()
        (self.root / "baseline/00000.bin").write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "byte count"):
            analysis.analyze(self.root)

    def test_trace_first_numerical_difference_skips_signed_zero(self):
        self.fixture()
        for role in ("baseline", "repeat", "renamed"):
            path = self.root / role / "00002.bin"
            data = np.fromfile(path, dtype="<u2")
            data[0] = 0x8000 if role == "renamed" else 0
            data.tofile(path)
        report = analysis.analyze(self.root)
        self.assertEqual(report["first_renamed_bitwise_difference"]["stage"], "fwd/embedding/0")
        self.assertEqual(report["first_renamed_difference"]["stage"], "bwd/gpt2/LayerNormLayer/0")

    def test_unknown_alignment_rejected(self):
        with self.assertRaisesRegex(ValueError, "alignment"):
            analysis.align(np.ones((2, 3)), "guess", np.arange(3))

    def test_native_replay_table_is_optional_validated_and_rendered(self):
        self.fixture()
        self.assertEqual(analysis.read_native_replay(self.root / "absent.tsv"), [])
        path = self.root / "native_replay.tsv"
        path.write_text("comparison\tmismatches\tmax_abs\tl2\nhead_canonical_input_order\t0\t0\t0\n")
        report = analysis.analyze(self.root)
        self.assertEqual(report["native_replay"][0]["mismatches"], 0)
        self.assertFalse(report["native_canonical_replay_restores_baseline"])
        self.assertIn("Native GPU counterfactual replay", analysis.render_html(report))
        path.write_text("comparison\tmismatches\tmax_abs\tl2\nbad\t-1\t0\t0\n")
        with self.assertRaisesRegex(ValueError, "native replay"):
            analysis.read_native_replay(path)

    def test_report_escapes_metadata(self):
        self.fixture()
        report = analysis.analyze(self.root)
        report["metadata"]["source"] = "<script>bad</script>"
        rendered = analysis.render_html(report)
        self.assertNotIn("<script>", rendered)
        self.assertNotIn("<script src", rendered)


if __name__ == "__main__":
    unittest.main()
