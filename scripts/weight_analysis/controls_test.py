"""Audit exact pair preservation and avoid conflating substrings with passages."""

from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .controls import (bigram_digest, bigram_shuffle, evaluate_controls, main,
                       summarize_paths)


def candidate(tokens, identifier="x", method="paths"):
    return {"candidate_id": identifier, "method": method, "token_ids": tokens}


class ShuffleTest(unittest.TestCase):
    def test_preserves_multisets_and_endpoints_randomized(self):
        rng = np.random.default_rng(9)
        for length in (2, 3, 10, 100, 1000):
            for vocab in (1, 2, 13):
                tokens = rng.integers(vocab, size=length, dtype=np.int32)
                before = tokens.copy()
                shuffled = bigram_shuffle(tokens, vocab, 11)
                self.assertEqual(Counter(tokens.tolist()), Counter(shuffled.tolist()))
                self.assertEqual(Counter(zip(tokens[:-1], tokens[1:])),
                                 Counter(zip(shuffled[:-1], shuffled[1:])))
                self.assertEqual(int(tokens[0]), int(shuffled[0]))
                self.assertEqual(int(tokens[-1]), int(shuffled[-1]))
                np.testing.assert_array_equal(before, tokens)

    def test_not_just_an_unchanged_copy(self):
        tokens = np.asarray([0, 1, 2, 0, 3, 2, 0, 1, 4, 0], dtype=np.int32)
        trails = {tuple(bigram_shuffle(tokens, 5, seed)) for seed in range(12)}
        self.assertGreater(len(trails), 1)

    def test_determinism(self):
        tokens = np.random.default_rng(2).integers(10, size=200)
        np.testing.assert_array_equal(bigram_shuffle(tokens, 10, 19),
                                      bigram_shuffle(tokens, 10, 19))

    def test_empty_singleton_and_rigid(self):
        for tokens in ([], [2], [0, 1, 2], [1, 1, 1]):
            array = np.asarray(tokens, dtype=np.int32)
            np.testing.assert_array_equal(bigram_shuffle(array, 3, 0), array)

    def test_validates_ids_and_dimensions(self):
        for invalid in ([-1], [3], [1.0], [[1]], [True]):
            with self.assertRaises(ValueError):
                bigram_shuffle(np.asarray(invalid), 3, 1)
        with self.assertRaises(ValueError):
            bigram_shuffle([0], 2**32, 1)

    def test_digest_direction_and_duplicate_multiplicity(self):
        arrays = [np.asarray(tokens, dtype=np.int32)
                  for tokens in ([0, 1, 2], [2, 1, 0], [0, 1, 0, 1])]
        self.assertEqual(len({bigram_digest(tokens) for tokens in arrays}), 3)


class EvidenceTest(unittest.TestCase):
    def test_whole_candidate_distinct_from_substring(self):
        tokens = np.asarray([0, 1, 2, 3], dtype=np.int32)
        candidates = [candidate([0, 1, 2, 3, 4], "a"),
                      candidate([0, 1, 2, 3], "b"),
                      candidate([0, 1, 2, 3], "c")]
        summary = summarize_paths(tokens, candidates, 5)["paths"]
        self.assertEqual(summary["whole_candidate_matches"], 2)
        self.assertEqual(summary["candidates_with_match_at_least"]["4"], 3)
        self.assertEqual(summary["distinct_selected_longest_substrings_at_least"]["4"], 1)
        self.assertEqual(summary["distinct_candidate_sequences"], 2)

    def test_excludes_pairs_preserves_method_denominators(self):
        candidates = [candidate([0, 1], "pair", "pairs"),
                      candidate([0, 1, 2], "path", "paths")]
        report = evaluate_controls([0, 1, 2, 0, 1, 0], candidates, 3, (1, 2))
        self.assertEqual(report["omitted_candidates_shorter_than_three"], 1)
        self.assertEqual(report["eligible_candidates_by_method"], {"paths": 1})
        self.assertEqual(len(report["shuffled_corpora"]), 2)
        for control in report["shuffled_corpora"]:
            self.assertEqual(control["bigram_multiset_sha256"], report["bigram_multiset_sha256"])

    def test_empty_path_set_and_corpus(self):
        report = evaluate_controls(np.asarray([], dtype=np.int32), [], 3, (1,))
        self.assertEqual(report["real"], {})
        self.assertEqual(report["shuffled_corpora"][0]["positions_different_from_original"], 0)

    def test_bad_seed_and_candidate(self):
        for seeds in ((), (1, 1), (-1,), (True,)):
            with self.assertRaises(ValueError):
                evaluate_controls([0, 1], [], 2, seeds)
        with self.assertRaises(ValueError):
            evaluate_controls([0, 1], [candidate([2])], 2)

    def test_cli_provenance_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokens = root / "tokens.bin"
            records = root / "candidates.jsonl"
            output = root / "report.json"
            np.asarray([0, 1, 2, 0, 2, 1, 0], dtype="<u4").tofile(tokens)
            records.write_text(json.dumps(candidate([0, 1, 2])) + "\n")
            args = ["--corpus-token-ids", str(tokens), "--candidates", str(records),
                    "--output", str(output), "--vocab-size", "3", "--seeds", "1"]
            with contextlib.redirect_stdout(io.StringIO()):
                main(args)
            report = json.loads(output.read_text())
            self.assertEqual(report["sources"]["corpus_token_ids"]["sha256"],
                             report["corpus_token_sha256"])
            saved = output.read_bytes()
            with self.assertRaises(FileExistsError):
                main(args)
            self.assertEqual(saved, output.read_bytes())


if __name__ == "__main__":
    unittest.main()
