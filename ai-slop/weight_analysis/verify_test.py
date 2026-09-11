#!/usr/bin/env python3
"""Synthetic tests: no checkpoints, training text, model calls, or GPU needed."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

if __package__:
    from weight_analysis import verify
else:
    import verify


def candidate(identifier, tokens, method="static"):
    return {
        "candidate_id": identifier,
        "method": method,
        "token_ids": tokens,
        "weights": [{"file": "weight_3.bin", "row": 7}],
    }


class CorpusIndexTest(unittest.TestCase):
    def test_longest_match_can_start_inside_candidate(self):
        result = verify.CorpusIndex(np.array([1, 2, 3, 4, 1, 2, 3, 5]), 10).analyze((9, 1, 2, 3, 8))
        self.assertEqual(result["longest_match"], {
            "length": 3, "candidate_token_start": 1,
            "corpus_token_start": 0, "corpus_token_end": 3,
            "corpus_occurrences": 2, "unique_in_corpus": False,
        })
        self.assertEqual(result["ngrams"]["2"]["matched_candidate_windows"], 2)
        self.assertEqual(result["ngrams"]["2"]["corpus_occurrences_of_distinct_ngrams"], 4)

    def test_counts_overlapping_matches_and_deduplicates_repeated_ngrams(self):
        result = verify.CorpusIndex(np.array([2, 2, 2, 2, 2]), 3).analyze((2, 2, 2, 2))
        self.assertEqual(result["longest_match"]["corpus_occurrences"], 2)
        self.assertEqual(result["ngrams"]["2"], {
            "candidate_windows": 3, "matched_candidate_windows": 3,
            "matched_distinct_ngrams": 1, "corpus_occurrences_of_distinct_ngrams": 4,
        })
        self.assertEqual(result["ngrams"]["4"]["corpus_occurrences_of_distinct_ngrams"], 2)

    def test_ties_choose_earliest_candidate_then_earliest_corpus(self):
        result = verify.CorpusIndex(np.array([3, 4, 0, 1, 2, 0, 1, 2]), 10).analyze((9, 1, 2, 8, 3, 4))
        self.assertEqual(result["longest_match"]["candidate_token_start"], 1)
        self.assertEqual(result["longest_match"]["corpus_token_start"], 3)
        self.assertEqual(result["longest_match"]["corpus_occurrences"], 2)

    def test_empty_corpus_and_singletons(self):
        empty = verify.CorpusIndex(np.array([], dtype=np.int32), 10).analyze((1,))
        self.assertEqual(empty["longest_match"]["length"], 0)
        self.assertIsNone(empty["longest_match"]["corpus_token_start"])
        single = verify.CorpusIndex(np.array([3, 3]), 10).analyze((9, 3, 8))
        self.assertEqual(single["longest_match"]["length"], 1)
        self.assertEqual(single["longest_match"]["corpus_occurrences"], 2)

    def test_rare_second_token_anchor_respects_boundaries(self):
        result = verify.CorpusIndex(np.array([2, 1, 1, 1, 2, 1]), 4).analyze((1, 2, 1))
        self.assertEqual(result["longest_match"]["corpus_token_start"], 3)
        self.assertEqual(result["longest_match"]["length"], 3)

    def test_exact_eight_token_match_and_longer(self):
        corpus = np.array([0] + list(range(1, 11)) + [0])
        result = verify.CorpusIndex(corpus, 20).analyze(tuple(range(1, 11)))
        self.assertEqual(result["longest_match"]["length"], 10)
        self.assertTrue(result["longest_match"]["unique_in_corpus"])
        self.assertEqual(result["ngrams"]["8"]["matched_candidate_windows"], 3)

    def test_matches_brute_force_on_random_corpora(self):
        rng = np.random.default_rng(12)
        for _ in range(60):
            corpus = rng.integers(0, 5, size=80)
            tokens = tuple(int(x) for x in rng.integers(0, 7, size=12))
            actual = verify.CorpusIndex(corpus, 7).analyze(tokens)
            best = (0, None, None, 0)
            for length in range(1, len(tokens) + 1):
                for start in range(len(tokens) - length + 1):
                    matches = [position for position in range(len(corpus) - length + 1)
                               if tuple(corpus[position:position + length]) == tokens[start:start + length]]
                    if matches and length > best[0]:
                        best = (length, start, matches[0], len(matches))
            longest = actual["longest_match"]
            self.assertEqual((longest["length"], longest["candidate_token_start"],
                              longest["corpus_token_start"], longest["corpus_occurrences"]), best)

    def test_rejects_out_of_range_and_noninteger_data(self):
        for corpus in (np.array([-1]), np.array([5]), np.array([1.5]), np.array([[1]])):
            with self.assertRaises(ValueError):
                verify.CorpusIndex(corpus, 5)
        index = verify.CorpusIndex(np.array([1]), 5)
        for tokens in ((), (-1,), (5,), (1.0,), (True,)):
            with self.assertRaises(ValueError):
                index.analyze(tokens)


class VerificationTest(unittest.TestCase):
    def test_native_cli_does_not_retokenize_and_preserves_outputs(self):
        try:
            from tokenizers import Tokenizer, models
        except ImportError:
            self.skipTest("optional tokenizers package is not installed")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1, "ĊĊ": 2, "b": 3}, unk_token="[UNK]"))
            tokenizer.save(str(directory / "tokenizer.json"))
            corpus = directory / "corpus.txt"
            corpus.write_text("a\n\nb", encoding="utf-8")
            candidates = directory / "candidates.jsonl"
            candidates.write_text(json.dumps(candidate("test", [1, 2, 3])) + "\n", encoding="utf-8")
            ids = directory / "ids.bin"
            offsets = directory / "offsets.bin"
            np.array([1, 2, 3], dtype="<u4").tofile(ids)
            np.array([0, 1, 3, 4], dtype="<u8").tofile(offsets)
            output = directory / "report.json"
            command = [sys.executable, str(Path(verify.__file__)), "--candidates", str(candidates),
                       "--tokenizer-dir", str(directory), "--corpus", str(corpus),
                       "--corpus-token-ids", str(ids), "--corpus-byte-offsets", str(offsets),
                       "--output", str(output)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            before = output.read_bytes()
            report = json.loads(before)
            self.assertEqual(report["corpus_token_count"], 3)
            self.assertEqual(report["corpus_tokenization"]["backend"], "provided_native_token_ids")
            self.assertEqual(report["results"][0]["real"]["longest_match"]["length"], 3)
            self.assertEqual(report["results"][0]["real"]["longest_match"]["text"], "a\n\nb")
            self.assertEqual(len(report["sources"]["verifier"]["sha256"]), 64)
            again = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("already exists", again.stderr)
            self.assertEqual(output.read_bytes(), before)

    def test_native_tokens_verify_every_byte_and_boundary(self):
        tokens = np.array([1, 2, 3], dtype="<u4")
        offsets = np.array([0, 1, 2, 4], dtype="<u8")
        vocabulary = {1: b"a", 2: b"\xc3", 3: b"\xa9b"}
        verify.validate_native_tokens(tokens, offsets, "aéb".encode(), vocabulary)
        for broken in (np.array([1, 1, 2, 4]), np.array([0, 2, 1, 4]),
                       np.array([0, 1, 2]), np.array([0, 1, 2, 5])):
            with self.assertRaises(ValueError):
                verify.validate_native_tokens(tokens, broken, "aéb".encode(), vocabulary)
        with self.assertRaises(ValueError):
            verify.validate_native_tokens(np.array([1, 3, 2]), offsets, "aéb".encode(), vocabulary)

    def test_native_unicode_fragment_has_exact_byte_and_bounding_text_spans(self):
        class DisplayTokenizer:
            def decode(self, ids, skip_special_tokens):
                return "display only"
        report = verify.verify_candidates(np.array([1, 2, 3, 4]), [candidate("a", [3])], 5)
        verify.add_native_text_spans(report, "aéb".encode(), np.array([0, 1, 2, 3, 4]), DisplayTokenizer())
        longest = report["results"][0]["real"]["longest_match"]
        self.assertEqual(longest["text"], "é")
        self.assertEqual((longest["corpus_byte_start"], longest["corpus_byte_end"]), (2, 3))
        self.assertEqual((longest["corpus_character_start"], longest["corpus_character_end"]), (1, 2))
        self.assertTrue(longest["text_is_character_bounding_span"])

    def test_gpt2_byte_spelling_includes_partial_utf8_and_specials(self):
        class VocabularyTokenizer:
            def to_str(self):
                return json.dumps({"added_tokens": [{"id": 4, "special": True}]})
            def get_vocab(self, with_added_tokens):
                return {"Ġ": 0, "Ċ": 1, "Ã": 2, "©": 3, "<|endoftext|>": 4}
        self.assertEqual(verify.gpt2_token_bytes(VocabularyTokenizer()),
                         {0: b" ", 1: b"\n", 2: b"\xc3", 3: b"\xa9", 4: b"<|endoftext|>"})

    def test_candidates_and_provenance_are_unchanged(self):
        proposals = [candidate("a", [9, 1, 2, 3]), candidate("b", [2, 3], "broken_static")]
        before = copy.deepcopy(proposals)
        report = verify.verify_candidates(np.array([1, 2, 3, 4]), proposals, 10)
        self.assertEqual(proposals, before)
        self.assertEqual([row["candidate"] for row in report["results"]], before)
        self.assertNotIn("text", report["results"][0]["candidate"])
        self.assertEqual(set(report["by_method"]), {"static", "broken_static"})

    def test_permutation_is_seeded_global_and_corpus_independent(self):
        proposals = [candidate("a", [1, 2, 1]), candidate("b", [2, 3, 1])]
        one = verify.verify_candidates(np.array([1, 2, 1]), proposals, 100, seed=42)
        two = verify.verify_candidates(np.array([7, 8, 9]), proposals, 100, seed=42)
        first, second = [row["shuffled_labels"]["token_ids"] for row in one["results"]]
        self.assertEqual(first[0], first[2])
        self.assertEqual(first[1], second[0])
        self.assertEqual(first[0], second[2])
        self.assertEqual(one["negative_control"], two["negative_control"])
        self.assertEqual([row["shuffled_labels"]["token_ids"] for row in one["results"]],
                         [row["shuffled_labels"]["token_ids"] for row in two["results"]])
        self.assertEqual(json.dumps(one, sort_keys=True), json.dumps(
            verify.verify_candidates(np.array([1, 2, 1]), proposals, 100, seed=42), sort_keys=True))

    def test_repeated_candidates_share_matching_work(self):
        proposals = [candidate(str(i), [1, 2, 3]) for i in range(30)]
        original = verify.CorpusIndex.analyze
        calls = []
        def tracked(index, tokens):
            calls.append(tokens)
            return original(index, tokens)
        with mock.patch.object(verify.CorpusIndex, "analyze", tracked):
            report = verify.verify_candidates(np.array([1, 2, 3]), proposals, 100)
        self.assertEqual(len(calls), 2)
        self.assertEqual(report["summary"]["distinct_candidate_sequences"], 1)
        self.assertEqual(report["summary"]["real"]["candidates"], 30)

    def test_validation_rejects_ambiguous_ids_and_bad_candidates(self):
        invalid = [candidate("x", [True]), candidate("x", [1.0]), candidate("x", [-1]),
                   candidate("x", [5]), candidate("x", []), candidate("", [1])]
        for proposal in invalid:
            with self.assertRaises(ValueError):
                verify.verify_candidates(np.array([1]), [proposal], 5)
        with self.assertRaises(ValueError):
            verify.verify_candidates(np.array([1]), [candidate("x", [1]), candidate("x", [2])], 5)

    def test_unicode_byte_offsets_and_display_do_not_change_candidates(self):
        class DisplayTokenizer:
            def decode(self, ids, skip_special_tokens):
                return "display only"
        proposals = [candidate("a", [1, 2])]
        report = verify.verify_candidates(np.array([0, 1, 2, 3]), proposals, 4)
        verify.add_text_spans(report, "aé猫b", np.array([[0, 1], [1, 2], [2, 3], [3, 4]]), DisplayTokenizer())
        longest = report["results"][0]["real"]["longest_match"]
        self.assertEqual(longest["text"], "é猫")
        self.assertEqual((longest["corpus_byte_start"], longest["corpus_byte_end"]), (1, 6))
        self.assertEqual(report["results"][0]["candidate"], candidate("a", [1, 2]))

    def test_cli_with_local_synthetic_tokenizer(self):
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers
        except ImportError:
            self.skipTest("optional tokenizers package is not installed")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "one": 1, "two": 2, "three": 3}, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.save(str(directory / "tokenizer.json"))
            corpus = directory / "corpus.txt"
            corpus.write_text("one two three one two", encoding="utf-8")
            candidates = directory / "candidates.jsonl"
            candidates.write_text(json.dumps(candidate("test", [1, 2, 3])) + "\n", encoding="utf-8")
            original = candidates.read_bytes()
            output = directory / "report.json"
            subprocess.run([sys.executable, str(Path(verify.__file__)), "--candidates", str(candidates),
                            "--tokenizer-dir", str(directory), "--corpus", str(corpus), "--output", str(output)],
                           check=True, capture_output=True, text=True)
            report = json.loads(output.read_text())
            self.assertEqual(report["corpus_token_count"], 5)
            self.assertEqual(report["results"][0]["real"]["longest_match"]["text"], "one two three")
            self.assertEqual(len(report["sources"]["candidates"]["sha256"]), 64)
            self.assertEqual(candidates.read_bytes(), original)
            self.assertEqual(report["corpus_tokenization"]["backend"], "huggingface_tokenizers")


if __name__ == "__main__":
    unittest.main()
