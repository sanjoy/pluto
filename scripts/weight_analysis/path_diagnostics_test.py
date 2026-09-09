"""Synthetic structural checks; no training corpus or real candidate runs."""

import contextlib
import copy
import hashlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import path_diagnostics as diagnostics


def candidate(tokens, identifier="a", method="paths"):
    return dict(token_ids=tokens, candidate_id=identifier, method=method)


class StructureTest(unittest.TestCase):
    def test_explicit_period_definitions(self):
        for tokens, expected in (([1], (1, 1)), ([1] * 16, (1, 1)),
                                 ([1, 2] * 8, (2, 2)), ([1, 2, 1, 2, 1], (2, 5)),
                                 ([1, 2, 3], (3, 3)), ([1, 2, 3, 1], (3, 4))):
            self.assertEqual(diagnostics.periods(tokens), expected)
        with self.assertRaises(ValueError):
            diagnostics.periods([])

    def test_duplicates_runs_cycles_and_method_denominators(self):
        candidates = [candidate([1] * 16, "a"), candidate([1] * 16, "b"),
                      candidate([1, 2] * 8, "c"), candidate([1, 2, 1, 2, 1], "d"),
                      candidate([1, 1, 2, 2, 2, 3], "e", "other"),
                      candidate([2], "f", "other")]
        before = copy.deepcopy(candidates)
        report = diagnostics.analyze(candidates, 4)
        self.assertEqual(candidates, before)
        overall = report["overall"]
        self.assertEqual(overall["candidate_records"], 6)
        self.assertEqual(overall["distinct_paths"], 5)
        self.assertEqual(overall["duplicate_records"], 1)
        self.assertEqual(overall["constant_token_paths"], 3)
        self.assertEqual(overall["repeating_one_token_paths"], 2)
        self.assertEqual(overall["alternating_two_token_paths"], 2)
        self.assertEqual(overall["histograms"]["longest_same_token_run"], {"1": 3, "3": 1, "16": 2})
        self.assertEqual(overall["histograms"]["adjacent_equal_pairs"], {"0": 3, "3": 1, "15": 2})
        self.assertEqual(report["by_method"]["paths"]["candidate_records"], 4)
        self.assertEqual(report["by_method"]["other"]["candidate_records"], 2)
        self.assertIsNone(overall["ascii_whitespace_only_paths"])

    def test_exact_bytes_whitespace_without_lossy_utf8_decoding(self):
        decoder = {0: b" ", 1: b"\n\t\r", 2: b"x", 3: b"\xc2", 4: b"\xa0", 5: b""}
        report = diagnostics.analyze([candidate([0, 1], "a"), candidate([0, 2], "b"),
                                      candidate([3, 4], "c"), candidate([5], "d")], 6, decoder)
        # Nonbreaking space is Unicode whitespace, deliberately not ASCII.
        self.assertEqual(report["overall"]["ascii_whitespace_only_paths"], 1)
        self.assertEqual(report["overall"]["ascii_whitespace_only_fraction"], 0.25)

    def test_varied_prefix_does_not_hide_repetitive_generated_tails(self):
        candidates = [candidate([9, 8] + [1] * 14, "constant", "first"),
                      candidate([9, 8] + [1, 2] * 7, "alternating", "second"),
                      candidate([9, 8] + [1, 2] * 7, "duplicate", "second")]
        report = diagnostics.analyze(candidates, 10)
        overall = report["overall"]
        self.assertEqual(overall["histograms"]["truncated_period"], {"16": 3})
        self.assertEqual(overall["histograms"]["longest_constant_suffix"], {"1": 2, "14": 1})
        self.assertEqual(overall["histograms"]["longest_alternating_two_token_suffix"], {"0": 1, "14": 2})
        self.assertEqual(overall["constant_suffix_at_least_eight_paths"], 1)
        self.assertEqual(overall["alternating_suffix_at_least_eight_paths"], 2)
        self.assertEqual(report["by_method"]["second"]["distinct_paths"], 1)
        self.assertEqual(report["by_method"]["first"]["constant_suffix_at_least_eight_paths"], 1)

    def test_suffix_boundaries_odd_cycles_and_eight_token_threshold(self):
        for sequence, constant, alternating in (
                ([1], 1, 0), ([1] * 8, 8, 0), ([1, 2, 1], 1, 0),
                ([1, 2, 1, 2], 1, 4), ([9, 1, 2, 1, 2, 1], 1, 5),
                ([9] + [1] * 7, 7, 0), ([9] + [1, 2] * 4, 1, 8),
                ([1, 2, 1, 2, 2], 2, 0)):
            with self.subTest(sequence=sequence):
                result = diagnostics.analyze([candidate(sequence)], 10)["overall"]
                self.assertEqual(result["histograms"]["longest_constant_suffix"], {str(constant): 1})
                self.assertEqual(result["histograms"]["longest_alternating_two_token_suffix"], {str(alternating): 1})
                self.assertEqual(result["constant_suffix_at_least_eight_paths"], int(constant >= 8))
                self.assertEqual(result["alternating_suffix_at_least_eight_paths"], int(alternating >= 8))

    def test_suffixes_match_brute_force_enumeration(self):
        for length in range(1, 8):
            for tokens in itertools.product(range(2), repeat=length):
                suffixes = [tokens[start:] for start in range(length)]
                constant = max(len(s) for s in suffixes if len(set(s)) == 1)
                alternating = max((len(s) for s in suffixes
                                   if len(s) >= 4 and s[-1] != s[-2]
                                   and all(s[i] == s[i - 2] for i in range(2, len(s)))),
                                  default=0)
                result = diagnostics.analyze([candidate(list(tokens))], 2)["overall"]
                self.assertEqual(result["histograms"]["longest_constant_suffix"], {str(constant): 1})
                self.assertEqual(result["histograms"]["longest_alternating_two_token_suffix"], {str(alternating): 1})

    def test_cache_duplicate_sequences_even_across_methods(self):
        candidates = [candidate([0, 1, 0, 1], "a", "real"),
                      candidate([0, 1, 0, 1], "b", "control")]
        with mock.patch.object(diagnostics, "periods", wraps=diagnostics.periods) as call:
            diagnostics.analyze(candidates, 2)
        self.assertEqual(call.call_count, 1)

    def test_errors_and_empty_report(self):
        for tokens in ([], [True], [1.0], [-1], [2]):
            with self.assertRaises(ValueError):
                diagnostics.analyze([candidate(tokens)], 2)
        with self.assertRaises(ValueError):
            diagnostics.analyze([candidate([0]), candidate([1])], 2)
        for vocab in (0, True, 2**31 + 1):
            with self.assertRaises(ValueError):
                diagnostics.analyze([], vocab)
        with self.assertRaises(ValueError):
            diagnostics.analyze([candidate([0])], 2, {1: b" "})
        with self.assertRaises(ValueError):
            diagnostics.analyze([candidate([0])], 2, {0: " "})
        report = diagnostics.analyze([], 2, {})
        self.assertEqual(report["overall"]["distinct_paths"], 0)
        self.assertIsNone(report["overall"]["ascii_whitespace_only_fraction"])
        self.assertEqual(report["by_method"], {})

    def test_cli_hashes_and_refuses_overwrite_without_corpus_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "frozen.jsonl"
            output = Path(directory) / "diagnostics.json"
            raw = (json.dumps(candidate([0, 1, 0, 1])) + "\n").encode()
            source.write_bytes(raw)
            args = ["--candidates", str(source), "--vocab-size", "2",
                    "--output", str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                diagnostics.main(args)
            report = json.loads(output.read_text())
            self.assertEqual(report["sources"]["candidates"]["sha256"],
                             hashlib.sha256(raw).hexdigest())
            self.assertEqual(report["overall"]["alternating_two_token_paths"], 1)
            self.assertEqual(source.read_bytes(), raw)
            saved = output.read_bytes()
            with self.assertRaises(FileExistsError):
                diagnostics.main(args)
            self.assertEqual(output.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
