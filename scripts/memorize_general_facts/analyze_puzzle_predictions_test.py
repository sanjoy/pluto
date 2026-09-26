"""Tests for descriptive metrics, text boundaries, and no-self-vote baselines."""

import unittest

from analyze_puzzle_predictions import (category, frequency_bin,
                                        leave_one_out_majority, summarize,
                                        unescape)


def row(target, prediction, position=4, fact=0):
    return {"fact_index": fact, "position": position, "target": target,
            "prediction": prediction, "correct": target == prediction,
            "source_prediction": target, "fresh_prediction": -1,
            "input_token": 10, "target_text": f" token{target}",
            "prediction_text": f" token{prediction}", "prefix": "some text",
            "category": "Other word/subword tokens", "top1_probability": .5,
            "target_probability": .5 if target == prediction else .1,
            "target_rank": 1 if target == prediction else 2,
            "margin": 1 if target == prediction else -1, "hidden_norm": 2}


class PredictionAnalysisTest(unittest.TestCase):
    def test_byte_escapes(self):
        self.assertEqual(unescape(r"hello\tworld\n\\\x22"), 'hello\tworld\n\\"')
        self.assertEqual(unescape(r"caf\xc3\xa9"), "café")
        self.assertEqual(unescape(r"\xc3"), "�")
        for value in ("\\", r"\q", r"\x", r"\xGG"):
            with self.assertRaises(ValueError):
                unescape(value)

    def test_categories_do_not_call_subwords_function_words(self):
        self.assertEqual(category("<|endoftext|>"), "EOS")
        self.assertEqual(category(" ."), "Punctuation")
        self.assertEqual(category("123"), "Number pieces")
        self.assertEqual(category(" is", "It", "It is blue"),
                         "Function-word tokens (fixed list)")
        self.assertEqual(category("is", "Par", "Paris shines"),
                         "Other word/subword tokens")
        self.assertEqual(category(" the", "Look", "Look there"),
                         "Other word/subword tokens")

    def test_baseline_cannot_vote_for_itself(self):
        rows = [row(1, 1), row(1, 1, fact=1), row(2, 2, fact=2)]
        self.assertEqual(leave_one_out_majority(rows, lambda r: 0), [1, 1, 1])
        self.assertEqual(leave_one_out_majority(rows, lambda r: r["fact_index"]),
                         [-1, -1, -1])
        # Removing a row may break an otherwise tied global frequency.
        self.assertEqual(leave_one_out_majority(rows[:1] + rows[2:], lambda r: 0),
                         [2, 1])

    def test_frequency_boundaries(self):
        self.assertEqual([frequency_bin(n) for n in (1, 2, 3, 5, 6, 100, 101)],
                         ["1", "2", "3-5", "3-5", "6-10", "51-100", "101+"])

    def test_summary_denominators_and_whole_facts(self):
        rows = [row(1, 1), row(2, 3, position=5), row(2, 2, fact=1)]
        result = summarize(rows)
        self.assertEqual(result["overall"], {"count": 3, "correct": 2, "accuracy": 2/3})
        self.assertEqual(result["complete_teacher_forced_facts"], 1)
        self.assertEqual(result["leading_correct_tokens_per_fact"], {1: 2})
        self.assertEqual(result["first_generated_token"]["correct"], 2)
        self.assertEqual(result["ever_correct_target_types"], 2)
        self.assertEqual(result["best_constant_prediction_correct"], 2)
        self.assertEqual(summarize([row(1, 2)])["correct_covered_by_top10_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
