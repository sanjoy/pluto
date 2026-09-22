#!/usr/bin/env python3
"""Small independent finite models exercise extraction and quotient semantics."""

import copy
import json
import random
import signal
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from discretize_core import (IntegerModel, ModelError, QuotientReducer, build_model,
                             evaluate_model, load_model, reduce_model, restore_membership,
                             save_model, _candidate_pairs, _eligible_pair_count)


def header(prompt_tokens=1):
    return {"schema": 1, "width": 1, "layers": 1, "vocab_size": 3,
            "eos_token": 2, "prompt_tokens": prompt_tokens,
            "vocabulary": [{"original_id": i, "hex": bytes([65 + i]).hex()} for i in range(3)]}


def sample(tokens, boundaries, predictions=None):
    return {"tokens": tokens, "predictions": predictions or tokens[1:] + [2],
            "boundaries": [[[word] for word in stage] for stage in boundaries]}


class DiscretizeCoreTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def build(self, samples, metadata=None):
        path = self.directory / "capture.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in [metadata or header()] + samples) + "\n")
        return build_model(path)

    def mergeable(self):
        return self.build([sample([0], [[100], [200], [300]]),
                           sample([1], [[101], [201], [301]])])

    def branching(self):
        return self.build([sample([0, 1], [[100, 102], [200, 202], [300, 302]]),
                           sample([1], [[101], [201], [301]])])

    @staticmethod
    def state(model, stage, word):
        return next(row["id"] for row in model["states"] if row["stage"] == stage and row["bits"] == [word])

    def test_exact_duplicates_and_bit_identity(self):
        row = sample([0], [[0x0000], [0x8000], [0x3f80]])
        model = self.build([row, row])
        self.assertEqual(len(model["states"]), 3)
        self.assertEqual(len(model["entry"]), 1)
        self.assertEqual(model["stats"]["verification"],
                         {"samples": 2, "targets": 2, "errors": 0, "explicit_eos": 2})
        self.assertEqual([state["bits"][0] for state in model["states"]], [0, 0x8000, 0x3f80])

    def test_duplicate_entry_conflict_is_rejected(self):
        with self.assertRaisesRegex(ModelError, "conflicting entry"):
            self.build([sample([0], [[100], [200], [300]]),
                        sample([0], [[101], [200], [300]])])

    def test_duplicate_attention_conflict_is_rejected(self):
        with self.assertRaisesRegex(ModelError, "conflicting attention"):
            self.build([sample([0], [[100], [200], [300]]),
                        sample([0], [[100], [201], [300]])])

    def test_duplicate_mlp_conflict_is_rejected(self):
        with self.assertRaisesRegex(ModelError, "conflicting MLP"):
            self.build([sample([0], [[100], [200], [300]]),
                        sample([1], [[101], [200], [301]])])

    def test_nonfinite_and_reference_error_rejected(self):
        with self.assertRaisesRegex(ModelError, "nonfinite"):
            self.build([sample([0], [[0x7f80], [200], [300]])])
        with self.assertRaisesRegex(ModelError, "suffix/EOS"):
            self.build([sample([0], [[100], [200], [300]], [1])])

    def test_autoregression_checks_suffix_and_eos(self):
        model = self.branching()
        self.assertEqual(evaluate_model(model)["targets"], 3)
        bad = copy.deepcopy(model)
        final = self.state(model, 2, 302)
        next(row for row in bad["snap"] if row[0] == final)[1] = 0
        with self.assertRaisesRegex(ModelError, "expected 2, got 0"):
            evaluate_model(bad)
        with self.assertRaisesRegex(ModelError, "undefined"):
            IntegerModel(model).predict([0, 0])

    def test_attention_uses_entire_ordered_causal_prefix(self):
        model = self.build([sample([0, 1], [[100, 102], [200, 202], [300, 302]]),
                            sample([1, 1], [[101, 102], [201, 203], [301, 303]])])
        table = model["attention"][0]
        two_token_rows = [row for row in table if len(row[0]) == 2]
        self.assertEqual(len(two_token_rows), 2)
        self.assertEqual(two_token_rows[0][0][-1], two_token_rows[1][0][-1])
        self.assertNotEqual(two_token_rows[0][1], two_token_rows[1][1])
        self.assertEqual(evaluate_model(model)["errors"], 0)

    def test_prompt_outputs_are_not_snap_constraints(self):
        model = self.build([sample([0, 1], [[100, 101], [200, 201], [300, 301]], [0, 2])],
                           header(prompt_tokens=2))
        self.assertEqual(len(model["snap"]), 1)
        reducer = QuotientReducer(model)
        self.assertTrue(reducer.try_merge(self.state(model, 2, 300), self.state(model, 2, 301)))
        self.assertEqual(evaluate_model(reducer.export())["explicit_eos"], 1)

    def test_accepted_merge_induces_attention_and_mlp_merges(self):
        model = self.mergeable()
        reducer = QuotientReducer(model)
        self.assertTrue(reducer.try_merge(self.state(model, 0, 100), self.state(model, 0, 101)))
        reduced = reducer.export()
        self.assertEqual(len(reduced["states"]), 3)
        self.assertEqual(reduced["stats"]["state_unions"], 3)
        self.assertEqual(reduced["stats"]["accepted_merges"][0]["induced_unions"], 2)
        self.assertEqual(len(reduced["attention"][0]), 1)
        self.assertEqual(len(reduced["mlp"][0]), 1)
        self.assertEqual(evaluate_model(reduced)["errors"], 0)

    def test_rejected_label_collision_rolls_back_every_index(self):
        model = self.branching()
        reducer = QuotientReducer(model)
        before = (copy.deepcopy(reducer.parent), copy.deepcopy(reducer.size),
                  copy.deepcopy(reducer.label), copy.deepcopy(reducer.uses),
                  copy.deepcopy(reducer.signatures), copy.deepcopy(reducer.term_signatures))
        first, second = self.state(model, 0, 100), self.state(model, 0, 101)
        self.assertFalse(reducer.try_merge(first, second))
        after = (reducer.parent, reducer.size, reducer.label, reducer.uses,
                 reducer.signatures, reducer.term_signatures)
        self.assertEqual(before, after)
        self.assertFalse(reducer.try_merge(first, second))
        self.assertEqual(reducer.cached_rejections, 1)
        self.assertEqual(reducer.attempted, 1)
        self.assertEqual(evaluate_model(reducer.export())["errors"], 0)
        self.assertTrue(reducer.try_merge(self.state(model, 2, 301), self.state(model, 2, 302)))
        self.assertEqual(evaluate_model(reducer.export())["errors"], 0)

    def test_cross_boundary_merge_rejected(self):
        model = self.mergeable()
        with self.assertRaisesRegex(ModelError, "different boundaries"):
            QuotientReducer(model).try_merge(self.state(model, 0, 100), self.state(model, 1, 200))

    def test_incremental_closure_matches_independent_full_rescan(self):
        randomizer = random.Random(2917)
        for trial in range(8):
            model = header()
            model.update(states=[{"id": 3 + i, "stage": i // 7, "bits": [100 + i]}
                                 for i in range(21)], samples=[], entry=[], attention=[[]],
                         mlp=[[[10 + i, 17 + randomizer.randrange(7)] for i in range(7)]],
                         snap=[[17 + i, randomizer.randrange(3)] for i in range(7)])
            prefixes = sorted({tuple(3 + randomizer.randrange(7) for _ in range(length))
                               for length in (1, 2, 3) for _ in range(20)})
            model["attention"][0] = [[list(prefix), 10 + randomizer.randrange(7)] for prefix in prefixes]
            reducer = QuotientReducer(model)
            equivalence = list(range(21))
            terms = [(1, tuple(state - 3 for state in prefix), output - 3)
                     for prefix, output in model["attention"][0]]
            terms += [(2, (source - 3,), output - 3) for source, output in model["mlp"][0]]
            def oracle(first, second):
                candidate = equivalence.copy()
                def unite(a, b):
                    a, b = candidate[a], candidate[b]
                    if a == b:
                        return False
                    candidate[:] = [min(a, b) if value in (a, b) else value for value in candidate]
                    return True
                unite(first, second)
                while True:
                    changed, signatures = False, {}
                    for stage, args, output in terms:
                        signature = (stage,) + tuple(candidate[arg] for arg in args)
                        if signature in signatures:
                            changed |= unite(output, signatures[signature])
                        else:
                            signatures[signature] = output
                    labels = {}
                    for state, token in model["snap"]:
                        root = candidate[state - 3]
                        if root in labels and labels[root] != token:
                            return None
                        labels[root] = token
                    if not changed:
                        return candidate
            for _ in range(70):
                stage = randomizer.randrange(3)
                first, second = (7 * stage + randomizer.randrange(7) for _ in range(2))
                expected = oracle(first, second)
                self.assertEqual(reducer.try_merge(first + 3, second + 3), expected is not None)
                if expected is not None:
                    equivalence = expected
                for a in range(21):
                    for b in range(21):
                        self.assertEqual(equivalence[a] == equivalence[b], reducer.find(a) == reducer.find(b))

    def test_atomic_persistence_and_resume(self):
        model = self.mergeable()
        checkpoint = self.directory / "nested" / "model.json"
        reducer = QuotientReducer(model)
        reducer.try_merge(self.state(model, 0, 100), self.state(model, 0, 101))
        reducer.save_checkpoint(checkpoint, search={"phase": "test"})
        restored = load_model(checkpoint)
        self.assertEqual(len(restored["states"]), 3)
        resumed = reduce_model(restored, checkpoint_path=checkpoint)
        self.assertTrue(resumed["stats"]["search"]["pairwise_irreducible"])
        self.assertEqual(resumed["stats"]["state_unions"], 3)
        self.assertEqual(load_model(checkpoint), resumed)
        with self.assertRaises(ValueError):
            save_model({"nan": float("nan")}, checkpoint)
        self.assertEqual(load_model(checkpoint), resumed)
        self.assertEqual(list(checkpoint.parent.glob(".model.json.*")), [])

    def test_exhaustive_certificate_and_honest_search_limits(self):
        model = self.branching()
        reduced = reduce_model(model, neighbors=1, max_passes=3)
        self.assertTrue(reduced["stats"]["search"]["pairwise_irreducible"])
        self.assertFalse(reduced["stats"]["search"]["global_minimum_proven"])
        limited = reduce_model(model, max_attempts=0)
        self.assertEqual(limited["stats"]["search"]["stopping_reason"], "attempt_limit")
        self.assertFalse(limited["stats"]["search"]["pairwise_irreducible"])
        shortlist = reduce_model(model, max_passes=1, exhaustive_pair_limit=0)
        self.assertFalse(shortlist["stats"]["search"]["pairwise_irreducible"])

    def test_already_merged_candidates_do_not_repeat_progress_reports(self):
        model = self.mergeable()
        state = model["states"][0]["id"]
        updates = []
        with mock.patch("discretize_core._candidate_pairs", return_value=[(0, state, state)] * 100), \
                mock.patch("discretize_core.time.monotonic", return_value=0):
            reduce_model(model, max_passes=1, exhaustive_pair_limit=0, progress=updates.append)
        # Initial report, one per stage, and the completed-pass report. A cached
        # or already-unified candidate must not repeatedly trigger at attempt 0.
        self.assertEqual(len(updates), 5)
        self.assertTrue(all(update["attempted"] == 0 for update in updates))

    def test_terminal_pruning_covers_every_unknown_label_pair(self):
        model = header()
        model.update(states=[{"id": 3 + i, "stage": i // 4, "bits": [100 + i]}
                             for i in range(12)], samples=[], entry=[],
                     attention=[[[[3 + i], 7 + i] for i in range(4)]],
                     mlp=[[[7 + i, 11 + i] for i in range(4)]], snap=[[12, 0], [13, 1]])
        reducer = QuotientReducer(model)
        for stage in (1, 2):
            start = 3 + 4 * stage
            pairs = {(first, second) for _, first, second in _candidate_pairs(reducer, stage, 1, exhaustive=True)}
            expected = {(start + first, start + second) for first in range(4)
                        for second in range(first + 1, 4)} - {(start + 1, start + 2)}
            self.assertEqual(pairs, expected)
        self.assertEqual(_eligible_pair_count(reducer), 16)

    def test_membership_survives_resume_and_upgrades_old_checkpoint(self):
        original = self.mergeable()
        reducer = QuotientReducer(original)
        reducer.try_merge(self.state(original, 2, 300), self.state(original, 2, 301))
        partial = reducer.export()
        self.assertEqual(sum(row["member_count"] for row in partial["states"]), 6)
        old = copy.deepcopy(partial)
        for row in old["states"]:
            del row["members"]
            del row["member_count"]
        self.assertFalse(QuotientReducer(old).export()["stats"]["membership_complete"])
        restored = restore_membership(old, original)
        self.assertEqual(restored["states"], partial["states"])
        resumed = QuotientReducer(restored)
        resumed.try_merge(self.state(restored, 0, 100), self.state(restored, 0, 101))
        result = resumed.export()
        self.assertEqual([row["member_count"] for row in result["states"]], [2, 2, 2])
        self.assertEqual(sorted(member for row in result["states"] for member in row["members"]),
                         sorted(row["id"] for row in original["states"]))
        bad = copy.deepcopy(old)
        bad["snap"][0][1] = 0
        with self.assertRaisesRegex(ModelError, "token label"):
            restore_membership(bad, original)

    def test_sigint_waits_for_complete_merge_then_saves_verified_checkpoint(self):
        model = self.mergeable()
        path = self.directory / "interrupted.json"
        previous = signal.getsignal(signal.SIGINT)
        original_try_merge = QuotientReducer.try_merge
        def interrupt_during_trial(reducer, first, second):
            signal.raise_signal(signal.SIGINT)
            return original_try_merge(reducer, first, second)
        with mock.patch.object(QuotientReducer, "try_merge", interrupt_during_trial):
            with self.assertRaises(KeyboardInterrupt):
                reduce_model(model, checkpoint_path=path)
        saved = load_model(path)
        self.assertEqual(len(saved["states"]), 3)
        self.assertEqual(saved["stats"]["verification"]["explicit_eos"], 2)
        self.assertTrue(saved["stats"]["search"]["interrupted"])
        self.assertIs(signal.getsignal(signal.SIGINT), previous)


if __name__ == "__main__":
    unittest.main()
