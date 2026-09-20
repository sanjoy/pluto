#!/usr/bin/env python3
"""Small exact cases guarding the memorization feasibility calculation."""

import itertools
import math
import unittest

from audit_prefixes import audit_prefixes


class PrefixAuditTest(unittest.TestCase):
    def test_distinct_prefixes_can_predict_different_tokens(self):
        result = audit_prefixes([[1, 2, 3], [4, 2, 5]])
        self.assertEqual(result.targets, 4)
        self.assertEqual(result.contexts, 4)
        self.assertEqual(result.unavoidable_errors, 0)
        self.assertEqual(result.minimum_mean_cross_entropy, 0)

    def test_zero_blocks_cannot_distinguish_different_prefixes_with_same_end(self):
        rows = [[1, 2, 3, 4, 9, 10], [2, 3, 4, 5, 9, 11]]
        options = {"prompt_tokens": 5, "eos_id": 0}
        self.assertEqual(audit_prefixes(rows, **options).unavoidable_errors, 0)
        result = audit_prefixes(rows, context_mode="token_position", **options)
        self.assertEqual(result.targets, 4)
        self.assertEqual(result.contexts, 3)
        self.assertEqual(result.unavoidable_errors, 1)
        self.assertEqual(result.conflicts, {(4, 9): {10: 1, 11: 1}})
        self.assertAlmostEqual(result.minimum_mean_cross_entropy, math.log(2) / 2)
        self.assertEqual(result.summary()["maximum_top1_accuracy"], 0.75)

    def test_token_position_contexts_keep_different_positions_separate(self):
        # Token 2 predicts 3 at position 1 and 5 at position 2. Learned absolute
        # position embeddings can distinguish these; token-only keys could not.
        result = audit_prefixes(
            [[1, 2, 3], [4, 6, 2, 5]], context_mode="token_position"
        )
        self.assertEqual(result.targets, 5)
        self.assertEqual(result.contexts, 5)
        self.assertEqual(result.unavoidable_errors, 0)

    def test_token_position_mode_scores_bos_and_shifts_sentence_positions(self):
        first_tokens = audit_prefixes(
            [[1], [2]],
            prompt_tokens=0,
            bos_id=0,
            eos_id=0,
            context_mode="token_position",
        )
        self.assertEqual(first_tokens.targets, 4)
        self.assertEqual(first_tokens.conflicts, {(0, 0): {1: 1, 2: 1}})
        shifted = audit_prefixes(
            [[1, 2, 3], [4, 2, 5]],
            prompt_tokens=2,
            bos_id=0,
            eos_id=0,
            context_mode="token_position",
        )
        self.assertEqual(shifted.targets, 4)
        self.assertEqual(shifted.conflicts, {(2, 2): {3: 1, 5: 1}})

    def test_unknown_context_modes_fail(self):
        for mode in ("", "last_token", "PREFIX", None):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "context_mode"):
                    audit_prefixes([[1, 2]], context_mode=mode)

    def test_majority_label_gives_exact_accuracy_and_entropy_bounds(self):
        result = audit_prefixes([[1, 2], [1, 2], [1, 3]])
        self.assertEqual(result.targets, 3)
        self.assertEqual(result.unavoidable_errors, 1)
        self.assertEqual(result.conflicts, {(1,): {2: 2, 3: 1}})
        expected_entropy = -(2 / 3) * math.log(2 / 3) - math.log(1 / 3) / 3
        self.assertAlmostEqual(result.minimum_mean_cross_entropy, expected_entropy)
        self.assertAlmostEqual(result.summary()["maximum_top1_accuracy"], 2 / 3)

    def test_repeated_identical_samples_are_not_conflicts(self):
        result = audit_prefixes([[1, 2, 3]] * 4, eos_id=0)
        self.assertEqual(result.targets, 12)
        self.assertEqual(result.contexts, 3)
        self.assertEqual(result.unavoidable_errors, 0)

    def test_eos_reveals_ambiguity_when_one_sentence_is_a_prefix(self):
        rows = [[1, 2], [1, 2, 3]]
        self.assertEqual(audit_prefixes(rows).unavoidable_errors, 0)
        result = audit_prefixes(rows, eos_id=0)
        self.assertEqual(result.targets, 5)
        self.assertEqual(result.unavoidable_errors, 1)
        self.assertEqual(result.conflicts, {(1, 2): {0: 1, 3: 1}})

    def test_bos_can_score_first_token_without_changing_later_contexts(self):
        rows = [[1], [2]]
        self.assertEqual(audit_prefixes(rows, eos_id=0).unavoidable_errors, 0)
        result = audit_prefixes(rows, prompt_tokens=0, bos_id=0, eos_id=0)
        self.assertEqual(result.targets, 4)
        self.assertEqual(result.unavoidable_errors, 1)
        self.assertEqual(result.conflicts, {(0,): {1: 1, 2: 1}})

    def test_longer_prompt_excludes_ambiguous_targets(self):
        rows = [[1, 2, 3], [1, 4, 5]]
        self.assertEqual(audit_prefixes(rows, eos_id=0).unavoidable_errors, 1)
        result = audit_prefixes(rows, prompt_tokens=2, eos_id=0)
        self.assertEqual(result.targets, 4)
        self.assertEqual(result.unavoidable_errors, 0)

    def test_complete_sentence_prompt_still_scores_eos(self):
        result = audit_prefixes([[1, 2], [1, 2]], prompt_tokens=2, eos_id=0)
        self.assertEqual(result.targets, 2)
        self.assertEqual(result.contexts, 1)
        self.assertEqual(result.unavoidable_errors, 0)

    def test_order_does_not_change_the_bound(self):
        rows = [[1, 2, 3], [1, 2, 4], [1, 5, 6]]
        expected = audit_prefixes(rows, eos_id=0)
        for permutation in itertools.permutations(rows):
            self.assertEqual(audit_prefixes(permutation, eos_id=0), expected)

    def test_bound_matches_exhaustive_deterministic_predictors(self):
        rows = [[1, 2, 3], [1, 2, 4], [1, 5, 6], [1, 5, 6]]
        # Exhaustively select one predicted token for each observed prefix.
        # This independent finite oracle checks the histogram formula.
        targets = {}
        for row in rows:
            for position in range(1, len(row)):
                targets.setdefault(tuple(row[:position]), []).append(row[position])
        best = max(
            sum(
                labels.count(prediction)
                for labels, prediction in zip(targets.values(), p)
            )
            for p in itertools.product(*(set(labels) for labels in targets.values()))
        )
        result = audit_prefixes(rows)
        self.assertEqual(result.unavoidable_errors, result.targets - best)

    def test_invalid_or_empty_scoring_requests_fail(self):
        for rows, options in [
            ([], {}),
            ([[]], {}),
            ([[1]], {}),
            ([[1, 2]], {"prompt_tokens": -1}),
            ([[1, 2]], {"prompt_tokens": 0}),
            ([[1, 2]], {"prompt_tokens": 3, "eos_id": 0}),
        ]:
            with self.subTest(rows=rows, options=options):
                with self.assertRaises(ValueError):
                    audit_prefixes(rows, **options)


if __name__ == "__main__":
    unittest.main()
