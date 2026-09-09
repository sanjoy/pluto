"""Synthetic causal-validation checks; no checkpoint, GPU, or real text runs."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from contextlib import ExitStack

import numpy as np

from . import causal_validation as cv


def fixture():
    # Byte/token addresses are deliberately trivial here. The production CLI
    # additionally checks the vocabulary's decoded bytes for EVERY token.
    corpus = b"a\n" * 150000
    tokens = np.arange(len(corpus), dtype=np.int32) % 17
    offsets = np.arange(len(corpus) + 1, dtype=np.uint64)
    plan, batch = cv.build_plan(corpus, tokens, offsets)
    plan["clean_control_tolerance"] = {"atol": 1e-6, "rtol": 1e-6, "argmax_exact": True}
    data = batch.tobytes()
    plan["batch_file"] = {"filename": "batch_tokens.bin", "dtype": "<i4",
                          "shape": [2, 32, 1024], "bytes": len(data), "sha256": cv.digest(data)}
    plan["checkpoint"] = {"checkpoint_directory": "/tmp/toy-checkpoint"}
    plan["inputs"] = {"binary": {"path": "/tmp/toy-probe"}}
    return plan, batch


def matrices():
    targets = np.zeros((32, 1024), dtype=np.int32)
    clean = np.broadcast_to((2 + np.arange(32, dtype=np.float32) / 32)[:, None], targets.shape).copy()
    losses = {name: clean.copy() for name in cv.CONTROLS + cv.ARMS}
    argmax = {name: targets.copy() for name in cv.CONTROLS + cv.ARMS}
    return losses, argmax, targets


def metadata(plan, batch_path):
    result = {"schema_version": 1, "complete": True, "context_length": 1024,
              "passage_count": 32, "vocab_size": 50257, "batch_sequences": 4,
              "padded_vocab_size": 50272, "checkpoint_raw_weight_count": 101,
              "checkpoint_weight_bytes": [spec.nbytes for spec in cv.tensor_manifest()],
              "batch_tokens_file": str(batch_path), "batch_tokens_bytes": 262144,
              "binary_file": plan["inputs"]["binary"]["path"],
              "checkpoint_directory": plan["checkpoint"]["checkpoint_directory"],
              "non_default_executor": True, "optimizer_steps": 0, "backward_calls": 0,
              "checkpoint_unique_weight_count": 100, "arms": []}
    for name in ("clean_before", "clean_repeat", *cv.ARMS, "clean_after"):
        indices, scale = [], 1.0
        if name in cv.ARMS:
            block, family, dose = name.split("_")
            first = int(block[5:]) * 12 + (6 if family == "attention" else 12)
            indices, scale = [first, first + 1], (0.5 if dose == "half" else 0.0)
        result["arms"].append({"name": name, "loss_file": name + ".losses.f32",
                               "argmax_file": name + ".argmax.i32",
                               "group_weight_indices": indices, "scale": scale,
                               "finite_losses": True, "argmax_valid": True,
                               "restoration_verified_bytes": True, "elapsed_seconds": 1.0})
    return result


class PlannerTest(unittest.TestCase):
    def test_split_follows_newline_and_keeps_it_in_prefix(self):
        self.assertEqual(cv.split_boundary(b"a" * 90 + b"bc\n" + b"d" * 7), 93)

    def test_split_utf8_fallback_and_empty_rejected(self):
        # The 90% boundary is the continuation byte of this final character.
        self.assertEqual(cv.split_boundary(b"a" * 8 + b"\xc3\xa9" + b"z"), 10)
        for data in (b"", b"a\n", "not bytes"):
            with self.assertRaises(ValueError):
                cv.split_boundary(data)

    def test_midpoints_exact_end_limit_and_nonoverlap(self):
        self.assertEqual(cv.midpoint_starts(100, 8, 4), [11, 34, 56, 79])
        for args in ((8, 8, 1), (18, 8, 4), (0, 8, 1), (100, True, 4), (100, 8, 0)):
            with self.assertRaises(ValueError):
                cv.midpoint_starts(*args)

    def test_inputs_targets_byte_intervals_and_half_scoring_off_by_one(self):
        plan, batch = fixture()
        self.assertEqual(batch.shape, (2, 32, 1024))
        self.assertEqual(batch.dtype, np.dtype("<i4"))
        for row, passage in enumerate(plan["passages"]):
            start, end = passage["token_interval"]
            self.assertEqual(end - start, 1025)
            np.testing.assert_array_equal(batch[0, row], np.arange(start, end - 1) % 17)
            np.testing.assert_array_equal(batch[1, row], np.arange(start + 1, end) % 17)
            self.assertEqual(passage["target_byte_offsets"], list(range(start + 1, end + 1)))
            self.assertEqual(passage["token_ids"][513], int(batch[1, row, 512]))
        self.assertEqual([p["split"] for p in plan["passages"]],
                         [cv.SPLITS[0]] * 16 + [cv.SPLITS[1]] * 16)

    def test_invalid_native_bounds_types_and_split_not_aligned(self):
        corpus = b"a\n" * 100
        tokens = np.zeros(200, dtype=np.int32)
        offsets = np.arange(201, dtype=np.uint64)
        for bad in (tokens.astype(float), tokens - 1, tokens + cv.VOCAB_SIZE):
            with self.assertRaises(ValueError):
                cv.build_plan(corpus, bad, offsets, context=2, per_split=2)
        for bad in (offsets[:-1], offsets.astype(float), np.zeros(201, dtype=np.int64)):
            with self.assertRaises(ValueError):
                cv.build_plan(corpus, tokens, bad, context=2, per_split=2)
        # Remove token boundary 182 while maintaining strictly increasing ends.
        with self.assertRaisesRegex(ValueError, "native token boundary"):
            cv.build_plan(corpus, tokens[:-1], np.delete(offsets, 182), context=2, per_split=2)

    def test_exclusive_directory_and_dangling_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.joinpath("dangling").symlink_to(root / "missing")
            for output in (root, root / "dangling"):
                with self.assertRaises(FileExistsError):
                    cv.plan_files(*([root / "absent"] * 7), output)

    def test_planner_files_byte_validation_and_freeze_without_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.txt"; corpus.write_bytes(b"a\n" * 150000)
            tokens = root / "tokens.bin"
            tokens.write_bytes(np.tile(np.array([0, 1], dtype="<u4"), 150000).tobytes())
            offsets = root / "offsets.bin"
            offsets.write_bytes(np.arange(300001, dtype="<u8").tobytes())
            (root / "tokenizer.json").write_text("{}")
            protocol = root / "protocol.md"; protocol.write_text("fixed synthetic protocol")
            binary = root / "probe"; binary.write_bytes(b"not executed")
            output = root / "frozen"
            # Only vocabulary and checkpoint loading are replaced. The real
            # native byte validator, full32 passage selection, hashes, output
            # serialization, and load_plan round trip all execute.
            with ExitStack() as stack:
                tokenizer = stack.enter_context(mock.patch("tokenizers.Tokenizer"))
                tokenizer.from_file.return_value.get_vocab_size.return_value = 50257
                stack.enter_context(mock.patch.object(cv, "gpt2_token_bytes", return_value={0: b"a", 1: b"\n"}))
                checkpoint = stack.enter_context(mock.patch.object(cv, "GPT2Checkpoint"))
                checkpoint.return_value.provenance.return_value = {"synthetic": True}
                plan = cv.plan_files(corpus, tokens, offsets, root, protocol, root / "weights", binary, output)
                self.assertEqual(plan["batch_file"]["bytes"], 262144)
                self.assertIn("dataset_split_source", plan["inputs"])
                loaded, batch = cv.load_plan(output / "manifest.json")
                self.assertEqual(loaded, plan)
                self.assertEqual(batch.shape, (2, 32, 1024))
                # A single corrupt native token must fail BEFORE output exists.
                bad = np.fromfile(tokens, dtype="<u4"); bad[10] = 1
                tokens.write_bytes(bad.tobytes())
                with self.assertRaisesRegex(ValueError, "does not match"):
                    cv.plan_files(corpus, tokens, offsets, root, protocol, root / "weights", binary, root / "invalid")
                self.assertFalse((root / "invalid").exists())


class MetricsTest(unittest.TestCase):
    def test_planted_paired_matrix_primary_secondary_dose_and_selectivity(self):
        plan, _ = fixture()
        losses, argmax, targets = matrices()
        half, zero = cv.ARMS[:2]
        changes = np.zeros(32, dtype=np.float32)
        changes[:4] = [1, 2, 3, 4]
        changes[16:] = 0.5
        losses[half][:, 512:] += changes[:, None]
        losses[zero][:, 512:] += 2 * changes[:, None]
        # Large early-position damage does not change the primary metric.
        losses[half][:, :512] += 10
        result = cv.summarize(plan, losses, argmax, targets)
        primary = result["metrics"][half]["primary_last512"]
        self.assertEqual(primary["per_passage_mean_delta"], changes.tolist())
        self.assertEqual(primary["splits"][cv.SPLITS[0]]["mean_delta"], 10 / 16)
        self.assertEqual(primary["prefix_minus_suffix_mean_delta"], .125)
        np.testing.assert_allclose(primary["per_passage_selective_delta"][:4], [-2, -2/3, 2/3, 2])
        self.assertEqual(result["selectivity_background_ids"]["training_prefix_00"],
                         ["training_prefix_01", "training_prefix_02", "training_prefix_03"])
        np.testing.assert_array_equal(result["metrics"][half]["secondary_all1024"]["per_passage_mean_delta"],
                                      (10 + changes) / 2)
        self.assertEqual(result["primary_prefix_delta_ranking"][0], zero)
        self.assertEqual(result["dose_comparisons"]["block0_attention"]["per_passage_zero_minus_half_delta"], changes.tolist())

    def test_greedy_prefix_stops_at_first_mismatch_not_total_matches(self):
        mask = np.array([[True] * 5, [False, True, True, True, True],
                         [True, True, False, True, True]], dtype=bool)
        np.testing.assert_array_equal(cv.greedy_prefix_lengths(mask), [5, 0, 2])
        for bad in (mask.astype(int), np.zeros((3, 0), dtype=bool), [True]):
            with self.assertRaises(ValueError):
                cv.greedy_prefix_lengths(bad)

    def test_teacher_forced_accuracy_not_greedy_continuation_and_prefix513(self):
        plan, _ = fixture()
        losses, argmax, targets = matrices()
        argmax[cv.ARMS[0]][0, 515] = 1
        result = cv.summarize(plan, losses, argmax, targets)
        first = result["metrics"][cv.ARMS[0]]["primary_last512"]
        self.assertEqual(first["conditioning_source_tokens"], 513)
        self.assertEqual(first["per_passage_exact_greedy_prefix_tokens"][0], 3)
        self.assertEqual(first["per_passage_teacher_forced_accuracy"][0], 511 / 512)
        self.assertEqual(first["splits"][cv.SPLITS[0]]["complete_exact_greedy_continuations"], 15)

    def test_clean_quartile_ties_use_ids_and_never_intervention_values(self):
        ids = [f"id{i:02}" for i in reversed(range(32))]
        backgrounds, groups = cv.clean_quartiles(np.ones(32), ids, [cv.SPLITS[0]] * 16 + [cv.SPLITS[1]] * 16)
        self.assertEqual(groups[0]["passage_ids"], ["id16", "id17", "id18", "id19"])
        self.assertEqual(backgrounds[15], [14, 13, 12])
        with self.assertRaises(ValueError):
            cv.clean_quartiles(np.ones(32), ["same"] * 32, [cv.SPLITS[0]] * 32)

    def test_missing_extra_nonfinite_negative_wrong_shape_and_padded_argmax(self):
        plan, _ = fixture()
        losses, argmax, targets = matrices()
        for value in (np.nan, np.inf, -1):
            bad = dict(losses); bad[cv.ARMS[0]] = losses[cv.ARMS[0]].copy()
            bad[cv.ARMS[0]][0, 0] = value
            with self.assertRaises(ValueError): cv.summarize(plan, bad, argmax, targets)
        for bad in (dict(list(losses.items())[:-1]), {**losses, "extra": targets},
                    {**losses, cv.ARMS[0]: losses[cv.ARMS[0]][:-1]},
                    {**losses, cv.ARMS[0]: losses[cv.ARMS[0]].astype(np.float64)}):
            with self.assertRaises(ValueError): cv.summarize(plan, bad, argmax, targets)
        bad = dict(argmax); bad[cv.ARMS[0]] = targets + cv.VOCAB_SIZE
        with self.assertRaises(ValueError): cv.summarize(plan, losses, bad, targets)

    def test_clean_controls_fail_closed_and_small_roundoff_recorded(self):
        plan, _ = fixture()
        losses, argmax, targets = matrices()
        losses["clean_repeat"][0, 0] += np.float32(1e-6)
        result = cv.summarize(plan, losses, argmax, targets)
        self.assertFalse(result["controls"]["clean_repeat"]["loss_bytes_exact"])
        losses["clean_after"][0, 0] += .01
        with self.assertRaisesRegex(ValueError, "gate failed"):
            cv.summarize(plan, losses, argmax, targets)

    def test_signed_zero_byte_difference_does_not_fail_numeric_control(self):
        plan, _ = fixture()
        losses, argmax, targets = matrices()
        for name in cv.CONTROLS:
            losses[name][0, 0] = np.float32(0)
        losses["clean_repeat"][0, 0] = np.float32(-0.0)
        result = cv.summarize(plan, losses, argmax, targets)
        self.assertTrue(result["controls"]["clean_repeat"]["loss_values_exact"])
        self.assertFalse(result["controls"]["clean_repeat"]["loss_bytes_exact"])
        losses["clean_after"] = losses["clean_before"].copy()
        argmax["clean_repeat"][0, 0] = 1
        with self.assertRaisesRegex(ValueError, "gate failed"):
            cv.summarize(plan, losses, argmax, targets)


class FilesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan, self.batch = fixture()
        self.plan_path = self.root / "manifest.json"
        self.batch_path = self.root / "batch_tokens.bin"
        self.batch_path.write_bytes(self.batch.tobytes())
        cv.write_report(self.plan_path, self.plan)

    def test_roundtrip_plan_and_target_token_addresses(self):
        plan, batch = cv.load_plan(self.plan_path)
        self.assertEqual(plan, self.plan)
        np.testing.assert_array_equal(batch, self.batch)

    def test_plan_rejects_duplicate_passages_changed_selection_and_wrong_batch(self):
        for field, value in (("passage_id", "current_suffix_00"), ("local_token_start", 0),
                             ("token_ids", [0] * 1025), ("target_byte_offsets", [0] * 1025)):
            changed = copy.deepcopy(self.plan)
            changed["passages"][0][field] = value
            self.plan_path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError): cv.load_plan(self.plan_path)
        self.plan_path.write_text(json.dumps(self.plan))
        self.batch_path.write_bytes(b"\0")
        with self.assertRaises(ValueError): cv.load_plan(self.plan_path)

    def test_metadata_requires_every_arm_restore_indices_paths_executor_and_no_updates(self):
        good = metadata(self.plan, self.batch_path)
        self.assertEqual(len(cv.validate_run_metadata(good, self.plan, self.batch_path)), 35)
        for key, value in (("complete", False), ("optimizer_steps", 1), ("backward_calls", 1),
                           ("non_default_executor", False), ("passage_count", 31),
                           ("checkpoint_directory", "/wrong"), ("binary_file", "/wrong"),
                           ("batch_tokens_file", "/wrong"), ("checkpoint_unique_weight_count", 99)):
            bad = copy.deepcopy(good); bad[key] = value
            with self.assertRaises(ValueError): cv.validate_run_metadata(bad, self.plan, self.batch_path)
        for key, value in (("padded_vocab_size", 50257), ("checkpoint_raw_weight_count", 100),
                           ("checkpoint_weight_bytes", [4] * 100)):
            bad = copy.deepcopy(good); bad[key] = value
            with self.assertRaises(ValueError): cv.validate_run_metadata(bad, self.plan, self.batch_path)
        for key, value in (("restoration_verified_bytes", False), ("group_weight_indices", [6]),
                           ("scale", .75), ("argmax_valid", False), ("finite_losses", False),
                           ("loss_file", "../outside"), ("name", "clean_before")):
            bad = copy.deepcopy(good); bad["arms"][2][key] = value
            with self.assertRaises(ValueError): cv.validate_run_metadata(bad, self.plan, self.batch_path)
        for arms in (good["arms"][:-1], good["arms"] + [good["arms"][0]], list(reversed(good["arms"]))):
            bad = {**good, "arms": arms}
            with self.assertRaises(ValueError): cv.validate_run_metadata(bad, self.plan, self.batch_path)
        for index in range(2, 34):
            bad = copy.deepcopy(good)
            bad["arms"][index]["restoration_verified_bytes"] = False
            with self.assertRaises(ValueError): cv.validate_run_metadata(bad, self.plan, self.batch_path)

    def test_exclusive_json_duplicate_keys_and_nonfinite(self):
        with self.assertRaises(FileExistsError): cv.write_report(self.plan_path, {})
        self.assertEqual(cv.read_json(self.plan_path), self.plan)
        bad = self.root / "bad.json"
        with self.assertRaises(ValueError): cv.write_report(bad, {"value": np.nan})
        self.assertFalse(bad.exists())
        bad.write_text('{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError, "duplicate"): cv.read_json(bad)
        bad.write_text('{"a":NaN}')
        with self.assertRaisesRegex(ValueError, "nonfinite"): cv.read_json(bad)

    def test_report_file_shapes_hashes_exclusivity_and_complete_outputs(self):
        # Only checkpoint authentication is mocked: all 70 binary files and
        # metadata are parsed and checked through the real reporter path.
        losses, argmax, _ = matrices()
        for name in cv.ARMS + cv.CONTROLS:
            (self.root / (name + ".losses.f32")).write_bytes(losses[name].astype("<f4").tobytes())
            (self.root / (name + ".argmax.i32")).write_bytes(argmax[name].astype("<i4").tobytes())
        run = self.root / "metadata.json"
        cv.write_report(run, metadata(self.plan, self.batch_path))
        output = self.root / "report.json"
        with mock.patch.object(cv, "authenticate_plan", return_value={"toy": "sha"}):
            report = cv.report_files(self.plan_path, run, output)
            self.assertEqual(len(report["raw_files"]), 70)
            self.assertTrue(report["checkpoint_and_frozen_inputs_unchanged"])
            with self.assertRaises(FileExistsError): cv.report_files(self.plan_path, run, output)
            (self.root / "clean_after.argmax.i32").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "size"):
                cv.report_files(self.plan_path, run, self.root / "bad_report.json")
            self.assertFalse((self.root / "bad_report.json").exists())

    def test_authentication_rejects_modified_inputs_and_checkpoint_snapshot(self):
        source = self.root / "source.bin"; source.write_bytes(b"source")
        snapshot = {"checkpoint_directory": "/toy", "unique_weight_files": 100,
                    "weight_sha256": {f"weight_{i}.bin": str(i) for i in range(100)}}
        plan = {"inputs": {"source": cv.file_record(source)}, "checkpoint": snapshot}
        with mock.patch.object(cv, "GPT2Checkpoint") as checkpoint:
            checkpoint.return_value.provenance.return_value = copy.deepcopy(snapshot)
            self.assertEqual(cv.authenticate_plan(plan), snapshot["weight_sha256"])
            checkpoint.return_value.provenance.return_value["weight_sha256"]["weight_0.bin"] = "changed"
            with self.assertRaises(ValueError): cv.authenticate_plan(plan)
        source.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "input changed"): cv.authenticate_plan(plan)

    def test_cli_dispatch_and_required_parameters(self):
        with mock.patch.object(cv, "report_files") as run:
            cv.main(["report", "--plan", "a", "--run", "b", "--output", "c"])
            run.assert_called_once_with(Path("a"), Path("b"), Path("c"))
        with mock.patch.object(cv, "plan_files") as run:
            args = ["plan"]
            for flag in ("corpus", "corpus-token-ids", "corpus-byte-offsets", "tokenizer-dir",
                         "protocol", "checkpoint", "binary", "output-dir"):
                args.extend(["--" + flag, flag])
            cv.main(args)
            self.assertEqual(len(run.call_args.args), 8)


if __name__ == "__main__":
    unittest.main()
