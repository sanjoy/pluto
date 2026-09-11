"""Synthetic, corpus-independent audits of higher-order verification controls."""

from collections import Counter
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import higher_order_controls as control


def candidate(tokens, identifier="x", method="paths"):
    return {"candidate_id": identifier, "method": method, "token_ids": tokens}


def ngrams(tokens, width):
    # Deliberately simple independent oracle; do not reuse the implementation's
    # sorted-row representation or Euler graph when checking multiplicities.
    return Counter(tuple(int(token) for token in tokens[start:start + width])
                   for start in range(len(tokens) - width + 1))


class ShuffleTest(unittest.TestCase):
    def test_randomized_multisets_endpoints_and_input_unchanged(self):
        rng = np.random.default_rng(9)
        for length in (0, 1, 2, 3, 4, 10, 100, 1000):
            for vocab in (1, 2, 13):
                tokens = rng.integers(vocab, size=length, dtype=np.int32)
                before = tokens.copy()
                for seed in (0, 17, 29):
                    with self.subTest(length=length, vocab=vocab, seed=seed):
                        shuffled = control.trigram_shuffle(tokens, vocab, seed)
                        self.assertEqual(len(shuffled), length)
                        self.assertEqual(shuffled.dtype, np.dtype(np.int32))
                        for width in (1, 2, 3):
                            self.assertEqual(ngrams(tokens, width),
                                             ngrams(shuffled, width))
                        np.testing.assert_array_equal(tokens[:2], shuffled[:2])
                        np.testing.assert_array_equal(tokens[-2:], shuffled[-2:])
                        np.testing.assert_array_equal(tokens, before)

    def test_branching_pair_graph_can_produce_different_trails(self):
        tokens = np.asarray([0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1],
                            dtype=np.int32)
        trails = {tuple(control.trigram_shuffle(tokens, 5, seed))
                  for seed in range(12)}
        self.assertGreater(len(trails), 1)
        # A real randomized result need not differ for EVERY seed.
        for trail in trails:
            self.assertEqual(ngrams(tokens, 3), ngrams(trail, 3))

    def test_seed_is_deterministic(self):
        tokens = np.random.default_rng(5).integers(7, size=500)
        np.testing.assert_array_equal(control.trigram_shuffle(tokens, 7, 29),
                                      control.trigram_shuffle(tokens, 7, 29))

    def test_empty_short_and_rigid_graphs(self):
        for sequence in ([], [2], [1, 2], [0, 1, 2], [1] * 30,
                         list(range(30))):
            tokens = np.asarray(sequence, dtype=np.int32)
            for seed in (0, 17, 29):
                np.testing.assert_array_equal(
                    control.trigram_shuffle(tokens, 30, seed), tokens)

    def test_sparse_nodes_do_not_allocate_vocabulary_squared(self):
        # A dense table of pair IDs (or even one count per vocabulary token)
        # would be infeasible here despite this graph having only a few nodes.
        largest = 2**31 - 1
        tokens = np.asarray([largest, 0, 1, largest, 0, 2, largest, 0],
                            dtype=np.uint32)
        shuffled = control.trigram_shuffle(tokens, 2**31, 17)
        self.assertEqual(ngrams(tokens, 3), ngrams(shuffled, 3))

    def test_invalid_tokens_vocab_and_seed_even_for_short_inputs(self):
        for invalid in ([-1], [3], [1.0], [True], [[1]], ["1"],
                        np.asarray([2**32], dtype=np.uint64)):
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaises(ValueError):
                    control.trigram_shuffle(invalid, 3, 1)
        for vocab in (0, -1, True, 3.0, 2**31 + 1):
            with self.assertRaises(ValueError):
                control.trigram_shuffle([0], vocab, 1)
        for seed in (-1, True, 1.0, "1", None, np.int64(1)):
            for tokens in (np.asarray([], dtype=np.int32), [0, 1, 0]):
                with self.assertRaises(ValueError):
                    control.trigram_shuffle(tokens, 2, seed)

    def test_digest_direction_multiplicity_and_shuffle_invariance(self):
        sequences = ([0, 1, 2], [2, 1, 0], [0, 1, 2, 0, 1, 2])
        self.assertEqual(len({control.trigram_digest(sequence)
                              for sequence in sequences}), 3)
        tokens = np.asarray([0, 1, 2, 0, 1, 3, 0, 1], dtype=np.int32)
        for seed in range(5):
            self.assertEqual(control.trigram_digest(tokens),
                             control.trigram_digest(
                                 control.trigram_shuffle(tokens, 4, seed)))
        with self.assertRaises(ValueError):
            control.trigram_digest([1.0, 2.0, 3.0])

    def test_digest_has_documented_little_endian_numeric_row_format(self):
        tokens = np.asarray([256, 1, 3, 0, 2], dtype=np.int32)
        rows = sorted(tuple(tokens[start:start + 3])
                      for start in range(len(tokens) - 2))
        expected = hashlib.sha256(np.asarray(rows, dtype="<u4").tobytes()).hexdigest()
        self.assertEqual(control.trigram_digest(tokens), expected)


class EvidenceTest(unittest.TestCase):
    def test_full_candidate_vs_substring_and_distinct_counts(self):
        candidates = [candidate([9, 0, 1, 2, 3, 8], "substring"),
                      candidate([0, 1, 2, 3], "full"),
                      candidate([0, 1, 2, 3], "duplicate")]
        summary = control.summarize_paths([0, 1, 2, 3], candidates, 10)["paths"]
        self.assertEqual(summary["candidates"], 3)
        self.assertEqual(summary["whole_candidate_matches"], 2)
        self.assertEqual(summary["distinct_whole_candidate_matches"], 1)
        self.assertEqual(summary["distinct_candidate_sequences"], 2)
        self.assertEqual(summary["candidate_length_histogram"], {"4": 2, "6": 1})
        self.assertEqual(summary["longest_match_length_histogram"], {"4": 3})
        self.assertEqual(summary["candidates_with_match_at_least"]["4"], 3)
        self.assertEqual(
            summary["distinct_selected_longest_substrings_at_least"]["4"], 1)

    def test_duplicate_sequences_are_analyzed_once_across_methods(self):
        candidates = [candidate([0, 1, 2, 3], "a", "real"),
                      candidate([0, 1, 2, 3], "b", "broken")]
        original = control.CorpusIndex.analyze
        calls = []

        def counted(index, sequence):
            calls.append(sequence)
            return original(index, sequence)

        with mock.patch.object(control.CorpusIndex, "analyze", counted):
            summary = control.summarize_paths([0, 1, 2, 3], candidates, 4)
        self.assertEqual(calls, [(0, 1, 2, 3)])
        self.assertEqual(summary["real"]["candidates"], 1)
        self.assertEqual(summary["broken"]["candidates"], 1)

    def test_excludes_triples_retains_denominators_and_preservation_checks(self):
        candidates = [candidate([0], "single", "singles"),
                      candidate([0, 1], "pair", "pairs"),
                      candidate([0, 1, 2], "triple", "triples"),
                      candidate([0, 1, 2, 0], "path", "paths"),
                      candidate([1, 2, 0, 1], "broken", "broken")]
        report = control.evaluate_controls(
            [0, 1, 2, 0, 1, 3, 0, 1], candidates, 4, (17, 29))
        self.assertEqual(report["omitted_candidates_shorter_than_four"], 3)
        self.assertEqual(report["eligible_candidates_by_method"],
                         {"broken": 1, "paths": 1})
        self.assertEqual(report["minimum_candidate_length"], 4)
        for shuffle in report["shuffled_corpora"]:
            self.assertTrue(all(shuffle["preservation_checks"].values()))
            for name in ("unigram", "bigram", "trigram"):
                key = name + "_multiset_sha256"
                self.assertEqual(report[key], shuffle[key])

    def test_candidates_and_corpus_are_not_modified_or_regenerated(self):
        candidates = [candidate([0, 1, 2, 0], "a"),
                      candidate([3, 1, 2, 3, 0], "b")]
        candidates[0]["weight_provenance"] = {"weight": 17, "score": 1.2}
        before = copy.deepcopy(candidates)
        tokens = np.asarray([0, 1, 2, 0, 1, 3, 0, 1], dtype=np.int32)
        original_tokens = tokens.copy()
        first = control.evaluate_controls(tokens, candidates, 4)
        second = control.evaluate_controls(tokens, candidates, 4)
        self.assertEqual(json.dumps(first), json.dumps(second))
        self.assertEqual(candidates, before)
        np.testing.assert_array_equal(tokens, original_tokens)
        self.assertEqual([item["seed"] for item in first["shuffled_corpora"]],
                         [17, 29, 43])

    def test_empty_and_unchanged_corpus_reported_without_retry(self):
        for tokens in (np.asarray([], dtype=np.int32), [0, 1, 2, 3, 4]):
            report = control.evaluate_controls(tokens, [], 5, (17,))
            self.assertEqual(report["real"], {})
            self.assertEqual(len(report["shuffled_corpora"]), 1)
            shuffle = report["shuffled_corpora"][0]
            self.assertTrue(shuffle["unchanged_sequence"])
            self.assertEqual(shuffle["positions_different_from_original"], 0)
            self.assertEqual(shuffle["fraction_positions_different"], 0.0)

    def test_bad_seeds_and_invalid_excluded_candidates(self):
        for seeds in ((), (1, 1), (-1,), (True,), (1.0,), ([1],), None, 4):
            with self.subTest(seeds=seeds):
                with self.assertRaises(ValueError):
                    control.evaluate_controls([0, 1], [], 2, seeds)
        # Even though it is too short to evaluate, this candidate must not slip
        # through the vocabulary/schema validation gate.
        with self.assertRaises(ValueError):
            control.evaluate_controls([0, 1], [candidate([2])], 2)

    def test_preservation_gate_rejects_corrupted_shuffle(self):
        with mock.patch.object(control, "trigram_shuffle",
                               return_value=np.asarray([0, 1, 3, 2, 0, 1])):
            with self.assertRaisesRegex(AssertionError, "preservation"):
                control.evaluate_controls([0, 1, 2, 3, 0, 1], [], 4, (17,))

    def test_cli_hashes_readonly_inputs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokens = root / "tokens.bin"
            records = root / "candidates.jsonl"
            output = root / "report.json"
            tokens.write_bytes(np.asarray([0, 1, 2, 0, 1, 3, 0, 1],
                                          dtype="<u4").tobytes())
            records.write_text(json.dumps(candidate([0, 1, 2, 0])) + "\n")
            token_bytes, candidate_bytes = tokens.read_bytes(), records.read_bytes()
            args = ["--corpus-token-ids", str(tokens), "--candidates", str(records),
                    "--output", str(output), "--vocab-size", "4", "--seeds", "17"]
            with contextlib.redirect_stdout(io.StringIO()):
                control.main(args)
            report = json.loads(output.read_text())
            self.assertEqual(report["sources"]["corpus_token_ids"]["sha256"],
                             report["corpus_token_sha256"])
            self.assertEqual(report["sources"]["candidates"]["sha256"],
                             hashlib.sha256(candidate_bytes).hexdigest())
            self.assertIn("higher_order_controls.py",
                          report["sources"]["code_sha256"])
            self.assertEqual(tokens.read_bytes(), token_bytes)
            self.assertEqual(records.read_bytes(), candidate_bytes)
            saved = output.read_bytes()
            with self.assertRaises(FileExistsError):
                control.main(args)
            self.assertEqual(saved, output.read_bytes())

    def test_cli_rejects_partial_uint32_without_creating_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokens, records, output = (root / name for name in ("t.bin", "c.jsonl", "r.json"))
            tokens.write_bytes(b"\x01\x00\x00")
            records.write_text("")
            with self.assertRaisesRegex(ValueError, "incomplete uint32"):
                control.main(["--corpus-token-ids", str(tokens),
                              "--candidates", str(records), "--output", str(output)])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
