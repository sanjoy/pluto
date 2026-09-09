"""Synthetic tests: corpus coverage cannot alter any frozen vocabulary ranking."""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import vocabulary_verify as verify


def artifact(scores, method="toy"):
    # Fixture construction is independent of the verifier. Its scores are in
    # canonical token-ID order, then serialized in the declared ranking order.
    ids = sorted(range(len(scores)), key=lambda token: (-scores[token], token))
    return {"schema_version": 1, "vocab_size": len(scores), "rankings": [
        {"method": method, "token_ids": ids, "scores": [scores[t] for t in ids]}]}


class CoverageTest(unittest.TestCase):
    def test_explicit_overlapping_windows_and_distinct_vs_weighted_metrics(self):
        tokens = [0, 1, 0, 2, 3, 4, 3, 4, 3]
        result = verify.coverage(tokens, [0, 1, 3, 4], 7)
        self.assertEqual(result["covered_token_positions"], 8)
        self.assertEqual(result["token_position_coverage"], 8 / 9)
        self.assertEqual(result["distinct_vocabulary_precision"], 1)
        self.assertEqual(result["distinct_vocabulary_recall"], 4 / 5)
        self.assertEqual(result["all_selected_windows"],
                         {"2": 6, "3": 4, "4": 2, "8": 0, "12": 0})
        self.assertEqual(result["longest_all_selected_run"], 5)
        partial = verify.coverage(tokens, [0, 1, 6], 7)
        self.assertEqual(partial["distinct_vocabulary_precision"], 2 / 3)
        self.assertEqual(partial["distinct_vocabulary_recall"], 2 / 5)
        self.assertEqual(partial["covered_token_positions"], 3)

    def test_randomized_against_obvious_brute_force_windows(self):
        rng = np.random.default_rng(8)
        for length in (0, 1, 2, 3, 8, 12, 80):
            tokens = rng.integers(7, size=length, dtype=np.int32)
            for size in (0, 1, 4, 7):
                selected = rng.choice(7, size=size, replace=False)
                selected_set = set(selected.tolist())
                result = verify.coverage(tokens, selected, 7)
                mask = [int(token) in selected_set for token in tokens]
                for window in verify.WINDOW_LENGTHS:
                    expected = sum(all(mask[start:start + window])
                                   for start in range(length - window + 1))
                    self.assertEqual(result["all_selected_windows"][str(window)], expected)
                longest, current = 0, 0
                for included in mask:
                    current = current + 1 if included else 0
                    longest = max(longest, current)
                self.assertEqual(result["longest_all_selected_run"], longest)
                self.assertEqual(result["covered_token_positions"], sum(mask))

    def test_empty_corpus_and_empty_selection_denominators(self):
        empty = np.asarray([], dtype=np.int32)
        result = verify.coverage(empty, [1], 3)
        self.assertEqual(result["distinct_vocabulary_precision"], 0)
        self.assertIsNone(result["distinct_vocabulary_recall"])
        self.assertIsNone(result["token_position_coverage"])
        result = verify.coverage([1], empty, 3)
        self.assertIsNone(result["distinct_vocabulary_precision"])
        self.assertEqual(result["distinct_vocabulary_recall"], 0)

    def test_invalid_corpus_and_selection(self):
        for tokens in ([-1], [3], [1.0], [True], [[1]],
                       np.asarray([2**32], dtype=np.uint64)):
            with self.assertRaises(ValueError):
                verify.coverage(tokens, [0], 3)
        for selected in ([-1], [3], [0, 0], [True], [0.0], [[1]]):
            with self.assertRaises(ValueError):
                verify.coverage([0, 1], selected, 3)


class CorrelationTest(unittest.TestCase):
    def test_ties_use_average_ranks(self):
        np.testing.assert_array_equal(verify.average_ranks([4, 2, 2, 8]),
                                      [3, 1.5, 1.5, 4])
        self.assertAlmostEqual(verify.spearman([1, 1, 2, 3], [4, 1, 1, 0]),
                               -5 / 6)

    def test_perfect_signs_and_undefined_constants(self):
        self.assertAlmostEqual(verify.spearman([1, 2, 3], [2, 4, 6]), 1)
        self.assertAlmostEqual(verify.spearman([1, 2, 3], [6, 4, 2]), -1)
        self.assertIsNone(verify.spearman([1, 1], [1, 2]))
        self.assertIsNone(verify.spearman([1, 2], [0, 0]))
        self.assertIsNone(verify.spearman([], []))
        self.assertIsNone(verify.spearman([1], [2]))

    def test_invalid_correlation_inputs(self):
        for bad in ([float("nan")], [float("inf")], [True], ["1"], [[1]]):
            with self.assertRaises(ValueError):
                verify.average_ranks(bad)
        with self.assertRaises(ValueError):
            verify.spearman([1, 2], [1])


class ArtifactTest(unittest.TestCase):
    def test_full_permutation_and_score_validation(self):
        valid = artifact([3, 3, 1])
        self.assertEqual(verify.validate_rankings(valid), 3)
        variants = []
        for ids in ([0, 0, 2], [0, 1], [0, 1, 3], [True, 1, 2],
                    [0.0, 1, 2], [1, 0, 2]):
            bad = copy.deepcopy(valid)
            bad["rankings"][0]["token_ids"] = ids
            variants.append(bad)
        for scores in ([3, 1], [1, 3, 0], [3, "3", 1], [3, True, 1],
                       [3, float("nan"), 1], [float("inf"), 3, 1], [10**400, 3, 1]):
            bad = copy.deepcopy(valid)
            bad["rankings"][0]["scores"] = scores
            variants.append(bad)
        for bad in variants:
            with self.subTest(bad=repr(bad)[:120]):
                with self.assertRaises(ValueError):
                    verify.validate_rankings(bad)

    def test_invalid_schema_vocab_and_methods(self):
        for bad in (None, [], {}, {"schema_version": True},
                    {"schema_version": 2}, {"schema_version": 1, "vocab_size": 0}):
            with self.assertRaises(ValueError):
                verify.validate_rankings(bad)
        for vocab in (True, 0, -1, 2**31 + 1, 3.0):
            bad = artifact([3, 2, 1])
            bad["vocab_size"] = vocab
            with self.assertRaises(ValueError):
                verify.validate_rankings(bad)
        for method in ("", 1, None):
            bad = artifact([1])
            bad["rankings"][0]["method"] = method
            with self.assertRaises(ValueError):
                verify.validate_rankings(bad)
        bad = artifact([1])
        bad["rankings"] *= 2
        with self.assertRaises(ValueError):
            verify.validate_rankings(bad)

    def test_fixed_cutoffs_control_and_no_mutation(self):
        frozen = artifact(list(range(260)))
        frozen["provenance"] = {"weight_hash": "frozen"}
        before = copy.deepcopy(frozen)
        tokens = np.asarray([0, 1, 259, 258, 257, 2], dtype=np.int32)
        original = tokens.copy()
        report = verify.evaluate(frozen, tokens)
        repeated = verify.evaluate(frozen, tokens)
        self.assertEqual(report, repeated)
        self.assertEqual(frozen, before)
        np.testing.assert_array_equal(tokens, original)
        self.assertEqual(report["top_k"], [128, 512, 2048, 8192])
        method = report["by_method"]["toy"]
        real = method["frozen_ranking"]
        self.assertEqual(real["top_k"]["128"]["effective_top_k"], 128)
        self.assertEqual(real["top_k"]["128"]["covered_token_positions"], 3)
        self.assertEqual(real["top_k"]["512"]["effective_top_k"], 260)
        self.assertEqual(real["top_k"]["512"]["covered_token_positions"], len(tokens))
        permutation = np.random.Generator(np.random.PCG64(17)).permutation(260)
        control_scores = np.arange(260)[permutation]
        expected_order = sorted(range(260), key=lambda token: (-control_scores[token], token))
        digest = hashlib.sha256(np.asarray(expected_order, dtype="<u4").tobytes()).hexdigest()
        self.assertEqual(method["permuted_score_control"]["ranking_ids_sha256"], digest)

    def test_control_preserves_ties_and_is_shared_across_methods(self):
        frozen = artifact([2, 2, 2])
        second = copy.deepcopy(frozen["rankings"][0])
        second["method"] = "second"
        frozen["rankings"].append(second)
        report = verify.evaluate(frozen, [0, 0, 1])
        first = report["by_method"]["toy"]
        self.assertEqual(first, report["by_method"]["second"])
        self.assertEqual(first["frozen_ranking"], first["permuted_score_control"])
        self.assertIsNone(first["frozen_ranking"]["score_frequency_spearman"])

    def test_empty_corpus_report_is_json_serializable(self):
        report = verify.evaluate(artifact([3, 1, 2]), np.asarray([], dtype=np.int32))
        json.dumps(report, allow_nan=False)
        self.assertIsNone(report["by_method"]["toy"]["frozen_ranking"]["score_frequency_spearman"])


class ReportTest(unittest.TestCase):
    def test_write_report_exclusive_and_no_partial_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            verify.write_report(output, {"result": 1})
            saved = output.read_bytes()
            with self.assertRaises(FileExistsError):
                verify.write_report(output, {"result": 2})
            self.assertEqual(output.read_bytes(), saved)
            invalid = Path(directory) / "invalid.json"
            with self.assertRaises(ValueError):
                verify.write_report(invalid, {"bad": float("nan")})
            self.assertFalse(invalid.exists())

    def test_cli_hashes_and_input_files_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ranking, tokens, output = (root / name for name in ("r.json", "t.bin", "o.json"))
            ranking.write_text(json.dumps(artifact([1, 3, 2])))
            tokens.write_bytes(np.asarray([0, 1, 1, 2], dtype="<u4").tobytes())
            before = (ranking.read_bytes(), tokens.read_bytes())
            args = ["--rankings", str(ranking), "--corpus-token-ids", str(tokens),
                    "--output", str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                verify.main(args)
            report = json.loads(output.read_text())
            self.assertEqual(report["sources"]["rankings"]["sha256"],
                             hashlib.sha256(before[0]).hexdigest())
            self.assertEqual(report["sources"]["corpus_token_ids"]["sha256"],
                             report["corpus_token_ids_sha256"])
            self.assertEqual(before, (ranking.read_bytes(), tokens.read_bytes()))
            with self.assertRaises(FileExistsError):
                verify.main(args)

    def test_cli_rejects_partial_uint32_without_creating_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ranking, tokens, output = (root / name for name in ("r.json", "t.bin", "o.json"))
            ranking.write_text(json.dumps(artifact([1])))
            tokens.write_bytes(b"\0")
            with self.assertRaisesRegex(ValueError, "incomplete uint32"):
                verify.main(["--rankings", str(ranking), "--corpus-token-ids",
                             str(tokens), "--output", str(output)])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
