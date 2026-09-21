#!/usr/bin/env python3
"""CPU-only tests of bounded search traversal and fail-closed evidence handling."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from run_width_depth_search import model_dimensions, parse_args, run_search


class WidthDepthSearchTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.binary = self.root / "native"
        self.binary.write_text("fake executable never launched\n")
        self.binary.chmod(0o700)
        self.corpus = self.root / "corpus.txt"
        self.corpus.write_text("fact\n" * 1024)
        self.tokenizer = self.root / "tokenizer"
        self.tokenizer.mkdir()
        (self.tokenizer / "tokenizer.json").write_text("{}\n")
        self.arguments = [
            f"--binary={self.binary}", f"--corpus={self.corpus}",
            f"--tokenizer={self.tokenizer}",
            f"--checkpoint_dir={self.root / 'checkpoints'}",
            f"--output_dir={self.root / 'artifacts'}",
            "--depths=1,2,3", "--widths=128,64,32",
        ]
        self.args = parse_args(self.arguments)
        self.calls = []
        self.failures = {(1, 64), (2, 32)}
        self.statuses = {}
        self.mutate = lambda phase, point, path: None

    def fake_subprocess(self, command, *, check, stdout):
        # A strict signature rejects accidental shell=True, capture_output,
        # and timeout arguments that would alter streaming or liveness behavior.
        self.assertFalse(check)
        self.calls.append(command)
        flags = dict(argument[2:].split("=", 1) for argument in command
                     if argument.startswith("--"))
        if Path(command[0]) == self.binary:
            phase = "verify" if "verify_checkpoint" in flags else "train"
            layers, width = int(flags["layers"]), int(flags["model_width"])
            self.assertIsNone(stdout)
            dimensions = model_dimensions(layers, width, self.args.attention_heads)
            self.assertEqual(int(flags["attention_heads"]), dimensions["heads"])
            self.assertEqual(int(flags["feed_forward_width"]), 4 * width)
        else:
            phase = "audit"
            directory = Path(flags["predictions"]).parent
            layers = int(directory.parent.name.split("_")[1])
            width = int(directory.parent.parent.name.split("_")[1])
            self.assertIsNotNone(stdout)
        point = layers, width
        success = point not in self.failures
        errors, exact = (0, 1024) if success else (3, 1022)
        loss = 0.000123456789 if success else 0.12
        status = 0 if success else (1 if phase == "audit" else 2)
        if (phase, point) in self.statuses:
            return subprocess.CompletedProcess(command, self.statuses[phase, point])

        if phase == "train":
            self.assertEqual(flags["search"], "false")
            output = Path(flags["output_dir"]) / f"layers_{layers}"
            output.mkdir(parents=True)
            shutil.copyfile(flags["corpus"], output / "corpus.txt")
            shutil.copyfile(Path(flags["tokenizer"]) / "tokenizer.json", output / "tokenizer.json")
            step = 2 if success else int(flags["steps"])
            checkpoint = Path(flags["checkpoint_dir"]) / f"layers_{layers}" / f"step_{step}"
            checkpoint.mkdir(parents=True)
            result = {**dimensions, "success": int(success), "errors": errors,
                      "targets": 10002, "step": step, "checkpoint": checkpoint,
                      "reached_time_limit": 0}
            (output / "final_predictions.tsv").write_text(f"mock predictions {point}\n")
        elif phase == "verify":
            output = Path(flags["output_dir"])
            output.mkdir()
            result = {**dimensions, "checkpoint": flags["verify_checkpoint"],
                      "errors": errors, "targets": 10002, "sentences": 1024,
                      "exact_sentences": exact, "mean_loss": format(loss, ".6g")}
            (output / "final_predictions.tsv").write_text(f"mock predictions {point}\n")
        else:
            audit = {
                "success": success, "errors": errors, "targets": 10002,
                "sentences": 1024, "exact_sentences": exact, "mean_loss_nats": loss,
                "corpus_sha256": hashlib.sha256(Path(flags["corpus"]).read_bytes()).hexdigest(),
                "tokenizer_sha256": hashlib.sha256(Path(flags["tokenizer"]).read_bytes()).hexdigest(),
                "predictions_sha256": hashlib.sha256(Path(flags["predictions"]).read_bytes()).hexdigest(),
            }
            json.dump(audit, stdout)
            stdout.flush()
            self.mutate(phase, point, Path(stdout.name))
            return subprocess.CompletedProcess(command, status)
        result_path = output / "result.txt"
        result_path.write_text("".join(f"{key}={value}\n" for key, value in result.items()))
        self.mutate(phase, point, result_path)
        return subprocess.CompletedProcess(command, status)

    def run_driver(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return run_search(self.args, run_process=self.fake_subprocess)

    def summary(self):
        return json.loads((self.args.output_dir / "width_depth_search_summary.json").read_text())

    def replace_field(self, phase_to_change, key, value, point_to_change=None):
        def mutate(phase, point, path):
            if phase == phase_to_change and (point_to_change is None or point == point_to_change):
                result = dict(line.split("=", 1) for line in path.read_text().splitlines())
                result[key] = str(value)
                path.write_text("".join(f"{name}={item}\n" for name, item in result.items()))
        self.mutate = mutate

    def test_verified_failure_continues_at_next_depth_and_produces_frontier(self):
        self.assertEqual(self.run_driver(), 0)
        summary = self.summary()
        self.assertEqual(summary["status"], "completed")
        self.assertEqual([(trial["layers"], trial["width"]) for trial in summary["trials"]],
                         [(1, 128), (1, 64), (2, 64), (2, 32), (3, 32)])
        self.assertEqual([trial["status"] for trial in summary["trials"]],
                         ["verified_success", "verified_budget_failure", "verified_success",
                          "verified_budget_failure", "verified_success"])
        self.assertEqual(len(self.calls), 15)
        for trial in summary["trials"]:
            self.assertEqual(trial["phase"], "verified")
            self.assertEqual(len(trial["commands"]), 3)
            self.assertEqual(trial["parameters"], trial["expected_parameters"])
            self.assertTrue(Path(trial["prediction_artifact_verification"]).is_file())
        self.assertEqual([(point["layers"], point["width"])
                          for point in summary["verified_success_frontier"]],
                         [(1, 128), (2, 64), (3, 32)])
        self.assertEqual(summary["minimum_parameter_success"]["width"], 32)
        skipped = {(point["layers"], point["width"]): point["reason"]
                   for point in summary["skipped_configurations"]}
        self.assertEqual(skipped[1, 32], "untested_after_wider_budget_failure_not_a_failure_claim")
        self.assertEqual(skipped[2, 128], "dominated_by_verified_success")

    def test_all_successes_skip_all_deeper_dominated_configurations(self):
        self.failures.clear()
        self.assertEqual(self.run_driver(), 0)
        summary = self.summary()
        self.assertEqual(len(summary["trials"]), 3)
        self.assertEqual(len(summary["verified_success_frontier"]), 1)
        self.assertEqual(summary["minimum_parameter_success"]["layers"], 1)
        self.assertEqual(summary["minimum_parameter_success"]["width"], 32)
        self.assertEqual(len(summary["skipped_configurations"]), 6)

    def test_all_failures_are_a_completed_empirical_search_not_an_execution_error(self):
        self.failures = {(layers, width) for layers in (1, 2, 3) for width in (128, 64, 32)}
        self.assertEqual(self.run_driver(), 0)
        summary = self.summary()
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(len(summary["trials"]), 3)
        self.assertTrue(all(trial["width"] == 128 for trial in summary["trials"]))
        self.assertEqual(summary["verified_success_frontier"], [])
        self.assertIsNone(summary["minimum_parameter_success"])

    def test_training_execution_failure_stops_without_verification(self):
        self.statuses["train", (1, 128)] = 1
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertIn("Training execution failed", self.summary()["error"])

    def test_budget_exit_without_artifacts_is_not_accepted_as_a_verified_failure(self):
        self.statuses["train", (1, 128)] = 2
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.summary()["trials"][0]["status"], "error")

    def test_native_failure_verification_error_stops_and_retains_prior_success(self):
        self.statuses["verify", (1, 64)] = 1
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(self.summary()["minimum_parameter_success"]["width"], 128)
        self.assertEqual(self.summary()["trials"][0]["status"], "verified_success")

    def test_failure_audit_must_return_one_not_malformed_input_code_two(self):
        self.statuses["audit", (1, 64)] = 2
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.calls), 6)
        self.assertIn("Independent prediction audit", self.summary()["error"])

    def test_success_native_verifier_cannot_report_failure(self):
        self.statuses["verify", (1, 128)] = 2
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.calls), 2)

    def test_success_artifact_verifier_cannot_report_failure(self):
        self.statuses["audit", (1, 128)] = 1
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.calls), 3)

    def test_actual_parameter_count_must_match_architecture_in_both_native_phases(self):
        for phase in ("train", "verify"):
            with self.subTest(phase=phase):
                self.replace_field(phase, "parameters", 1)
                self.assertEqual(self.run_driver(), 1)
                self.assertIn("expected parameters=", self.summary()["error"])
                shutil.rmtree(self.args.output_dir)
                shutil.rmtree(self.args.checkpoint_dir)

    def test_each_architecture_field_must_match_requested_dimensions(self):
        for key in ("layers", "width", "heads", "feed_forward_width"):
            with self.subTest(key=key):
                self.replace_field("verify", key, 999)
                self.assertEqual(self.run_driver(), 1)
                self.assertIn(f"expected {key}=", self.summary()["error"])
                shutil.rmtree(self.args.output_dir)
                shutil.rmtree(self.args.checkpoint_dir)

    def test_success_code_does_not_override_error_count(self):
        self.replace_field("train", "errors", 1)
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertIn("error count disagrees", self.summary()["error"])

    def test_failure_code_does_not_override_success_result(self):
        self.replace_field("train", "success", 1, (1, 64))
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.calls), 4)

    def test_premature_non_memorized_trial_is_not_a_budget_failure(self):
        self.replace_field("train", "step", 2, (1, 64))
        self.assertEqual(self.run_driver(), 1)
        self.assertIn("before either requested training budget", self.summary()["error"])

    def test_enabled_time_budget_can_end_a_trial_before_the_step_cap(self):
        def mutate(phase, point, path):
            if phase == "train" and point == (1, 64):
                result = dict(line.split("=", 1) for line in path.read_text().splitlines())
                checkpoint = Path(result["checkpoint"])
                checkpoint.rename(checkpoint.parent / "step_2")
                result.update(step="2", reached_time_limit="1",
                              checkpoint=str(checkpoint.parent / "step_2"))
                path.write_text("".join(f"{key}={value}\n" for key, value in result.items()))
        self.mutate = mutate
        self.assertEqual(self.run_driver(), 0)
        trial = self.summary()["trials"][1]
        self.assertEqual(trial["status"], "verified_budget_failure")
        self.assertEqual(trial["step"], 2)

    def test_a_disabled_time_limit_cannot_explain_an_early_exit(self):
        self.args.training_seconds = 0
        self.replace_field("train", "reached_time_limit", 1, (1, 64))
        self.assertEqual(self.run_driver(), 1)
        self.assertIn("disabled time limit", self.summary()["error"])

    def test_invalid_time_limit_indicator_is_rejected(self):
        self.replace_field("train", "reached_time_limit", 2)
        self.assertEqual(self.run_driver(), 1)
        self.assertIn("invalid reached_time_limit", self.summary()["error"])

    def test_native_nonfinite_and_negative_losses_are_rejected(self):
        for loss in ("nan", "inf", "-inf", "-0.1"):
            with self.subTest(loss=loss):
                self.replace_field("verify", "mean_loss", loss)
                self.assertEqual(self.run_driver(), 1)
                self.assertIn("invalid mean_loss", self.summary()["error"])
                shutil.rmtree(self.args.output_dir)
                shutil.rmtree(self.args.checkpoint_dir)

    def test_audit_nonfinite_negative_and_disagreeing_losses_are_rejected(self):
        for loss in ("nan", "inf", "-0.1", "0.5"):
            with self.subTest(loss=loss):
                def mutate(phase, point, path):
                    if phase == "audit":
                        audit = json.loads(path.read_text())
                        audit["mean_loss_nats"] = loss
                        path.write_text(json.dumps(audit))
                self.mutate = mutate
                self.assertEqual(self.run_driver(), 1)
                self.assertEqual(self.summary()["status"], "error")
                shutil.rmtree(self.args.output_dir)
                shutil.rmtree(self.args.checkpoint_dir)

    def test_result_step_outside_budget_is_rejected(self):
        self.replace_field("train", "step", 5001)
        self.assertEqual(self.run_driver(), 1)
        self.assertIn("outside the requested budget", self.summary()["error"])

    def test_checkpoint_path_cannot_escape_explicit_configuration(self):
        self.replace_field("train", "checkpoint", self.root)
        self.assertEqual(self.run_driver(), 1)
        self.assertIn("Unexpected or missing final checkpoint", self.summary()["error"])

    def test_snapshot_change_is_rejected(self):
        def mutate(phase, point, path):
            if phase == "train":
                (path.parent / "corpus.txt").write_text("different corpus\n")
        self.mutate = mutate
        self.assertEqual(self.run_driver(), 1)
        self.assertIn("snapshots", self.summary()["error"])

    def test_fresh_native_prediction_difference_is_rejected(self):
        def mutate(phase, point, path):
            if phase == "verify":
                (path.parent / "final_predictions.tsv").write_text("different predictions\n")
        self.mutate = mutate
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.calls), 2)
        self.assertIn("predictions differ", self.summary()["error"])

    def test_audit_must_match_native_counts_and_all_artifact_hashes(self):
        for field in ("success", "errors", "targets", "sentences", "exact_sentences",
                      "corpus_sha256", "tokenizer_sha256", "predictions_sha256"):
            with self.subTest(field=field):
                def mutate(phase, point, path):
                    if phase == "audit":
                        audit = json.loads(path.read_text())
                        audit[field] = "invalid"
                        path.write_text(json.dumps(audit))
                self.mutate = mutate
                self.assertEqual(self.run_driver(), 1)
                self.assertIn(f"unexpected {field}", self.summary()["error"])
                shutil.rmtree(self.args.output_dir)
                shutil.rmtree(self.args.checkpoint_dir)

    def test_binary_mutation_is_rejected_before_next_trial(self):
        def mutate(phase, point, path):
            if phase == "audit":
                self.binary.write_text("different native executable\n")
        self.mutate = mutate
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.calls), 3)
        self.assertIn("Native binary changed", self.summary()["error"])
        self.assertEqual(self.summary()["trials"][0]["status"], "verified_success")

    def test_binary_mutation_is_rejected_before_fresh_verification(self):
        def mutate(phase, point, path):
            if phase == "train":
                self.binary.write_text("different native executable\n")
        self.mutate = mutate
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertIn("Native binary changed", self.summary()["error"])

    def test_original_input_mutation_is_rejected_before_next_trial(self):
        def mutate(phase, point, path):
            if phase == "audit":
                self.corpus.write_text("different original corpus\n")
        self.mutate = mutate
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.calls), 3)
        self.assertIn("Corpus changed", self.summary()["error"])

    def test_existing_roots_are_never_reused(self):
        for path in (self.args.output_dir, self.args.checkpoint_dir):
            with self.subTest(path=path):
                path.mkdir()
                with self.assertRaisesRegex(ValueError, "existing run directory"):
                    self.run_driver()
                self.assertEqual(self.calls, [])
                path.rmdir()

    def test_valid_lists_are_sorted_and_refinement_heads_are_derived(self):
        args = parse_args(self.arguments + ["--depths=3,1", "--widths=48,96,32"])
        self.assertEqual(args.depths, [1, 3])
        self.assertEqual(args.widths, [96, 48, 32])
        self.assertEqual(model_dimensions(1, 96)["heads"], 3)
        self.assertEqual(model_dimensions(1, 96)["head_dim"], 32)
        self.assertEqual(model_dimensions(1, 48)["heads"], 3)
        self.assertEqual(model_dimensions(1, 48)["head_dim"], 16)
        self.assertEqual(model_dimensions(1, 32)["heads"], 1)
        self.assertEqual(model_dimensions(1, 256)["heads"], 4)

    def test_fixed_head_count_is_validated_for_every_width_before_creating_directories(self):
        args = parse_args(self.arguments + ["--widths=32,24,8", "--attention_heads=2"])
        self.assertEqual(args.attention_heads, 2)
        for width in args.widths:
            shape = model_dimensions(1, width, args.attention_heads)
            self.assertEqual((shape["heads"], shape["head_dim"]), (2, width // 2))
            self.assertEqual(shape["parameters"], model_dimensions(1, width)["parameters"])
        for widths, heads in (("32,24", 3), ("24,8,3", 2), ("8", 16)):
            with self.subTest(widths=widths, heads=heads), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    parse_args(self.arguments + [f"--widths={widths}", f"--attention_heads={heads}"])
                self.assertEqual(raised.exception.code, 2)
        self.assertFalse(self.args.output_dir.exists())
        self.assertFalse(self.args.checkpoint_dir.exists())

    def test_invalid_head_override_is_rejected_by_runner_before_creating_directories(self):
        self.args.widths = [32, 24, 3]
        self.args.attention_heads = 2
        with self.assertRaisesRegex(ValueError, "must divide every requested width"):
            self.run_driver()
        self.assertEqual(self.calls, [])
        self.assertFalse(self.args.output_dir.exists())
        self.assertFalse(self.args.checkpoint_dir.exists())

    def test_single_head_override_is_recorded_and_used_for_training_and_verification(self):
        self.args = parse_args(self.arguments + ["--widths=32,24,3", "--attention_heads=1"])
        self.failures = {(1, 24), (2, 3)}
        self.assertEqual(self.run_driver(), 0)
        summary = self.summary()
        self.assertEqual(summary["configuration"]["attention_heads"], 1)
        self.assertEqual([(trial["layers"], trial["width"]) for trial in summary["trials"]],
                         [(1, 32), (1, 24), (2, 24), (2, 3), (3, 3)])
        for trial in summary["trials"]:
            self.assertEqual(trial["heads"], 1)
            self.assertEqual(trial["head_dim"], trial["width"])
            for command in trial["commands"][:2]:
                self.assertIn("--attention_heads=1", command)
                self.assertIn(f"--model_width={trial['width']}", command)
        self.assertEqual(len(self.calls), 15)

    def test_explicit_zero_override_preserves_legacy_shapes_in_both_native_phases(self):
        self.args = parse_args(self.arguments + ["--widths=24,3", "--attention_heads=0"])
        self.failures.clear()
        self.assertEqual(self.run_driver(), 0)
        summary = self.summary()
        self.assertEqual(summary["configuration"]["attention_heads"], 0)
        self.assertEqual([(trial["heads"], trial["head_dim"]) for trial in summary["trials"]],
                         [(3, 8), (3, 1)])
        for trial in summary["trials"]:
            for command in trial["commands"][:2]:
                self.assertIn("--attention_heads=3", command)

    def test_parameter_formula_matches_preceding_fixed_width_experiment(self):
        self.assertEqual(model_dimensions(1, 512)["parameters"], 29416960)
        self.assertEqual(model_dimensions(8, 512)["parameters"], 51483648)

    def test_compact_width_flags_preserve_logical_dimensions_and_parameter_counts(self):
        args = parse_args(self.arguments + ["--widths=3,24,8"])
        self.assertEqual(args.widths, [24, 8, 3])
        for width, heads, head_dim, parameters in (
            (3, 3, 1, 154041), (8, 1, 8, 411256), (24, 3, 8, 1238376)
        ):
            with self.subTest(width=width):
                shape = model_dimensions(1, width)
                self.assertEqual(shape["heads"], heads)
                self.assertEqual(shape["head_dim"], head_dim)
                self.assertEqual(shape["feed_forward_width"], 4 * width)
                self.assertEqual(shape["parameters"], parameters)

    def test_compact_width_search_verifies_success_and_failure_without_padding(self):
        self.args.widths = [24, 8, 3]
        self.failures = {(1, 8), (2, 3)}
        self.assertEqual(self.run_driver(), 0)
        summary = self.summary()
        self.assertEqual([(trial["layers"], trial["width"]) for trial in summary["trials"]],
                         [(1, 24), (1, 8), (2, 8), (2, 3), (3, 3)])
        self.assertEqual(len(self.calls), 15)  # Train, fresh reload, independent audit.
        self.assertEqual([(point["layers"], point["width"])
                          for point in summary["verified_success_frontier"]],
                         [(1, 24), (2, 8), (3, 3)])
        self.assertEqual(summary["minimum_parameter_success"]["parameters"], 154335)

    def test_explicit_deep_narrow_trials_preserve_all_verification_phases(self):
        for fail_width_eight in (False, True):
            with self.subTest(fail_width_eight=fail_width_eight):
                label = "failure" if fail_width_eight else "success"
                self.args = parse_args(self.arguments + [
                    "--depths=16", "--widths=12,8", "--attention_heads=1",
                    "--steps=40000",
                    f"--output_dir={self.root / ('artifacts_' + label)}",
                    f"--checkpoint_dir={self.root / ('checkpoints_' + label)}",
                ])
                self.failures = {(16, 8)} if fail_width_eight else set()
                self.calls.clear()
                self.assertEqual(self.run_driver(), 0)
                summary = self.summary()
                self.assertEqual(len(self.calls), 6)
                self.assertEqual([(trial["layers"], trial["width"], trial["parameters"])
                                  for trial in summary["trials"]],
                                 [(16, 12, 645720), (16, 8, 424336)])
                for trial in summary["trials"]:
                    self.assertEqual(trial["phase"], "verified")
                    self.assertEqual(len(trial["commands"]), 3)
                    for command in trial["commands"][:2]:
                        self.assertIn("--layers=16", command)
                        self.assertIn("--attention_heads=1", command)
                    self.assertIn(f"--verify_checkpoint={trial['checkpoint']}",
                                  trial["commands"][1])
                    self.assertTrue(Path(trial["prediction_artifact_verification"]).is_file())
                self.assertEqual(summary["trials"][1]["status"],
                                 "verified_budget_failure" if fail_width_eight
                                 else "verified_success")
                expected_width = 12 if fail_width_eight else 8
                self.assertEqual([(point["layers"], point["width"])
                                  for point in summary["verified_success_frontier"]],
                                 [(16, expected_width)])
                self.assertEqual(summary["minimum_parameter_success"]["width"], expected_width)

    def test_explicit_depths_accept_the_positive_native_int32_range(self):
        args = parse_args(self.arguments + ["--depths=2147483647,16,9,8,1"])
        self.assertEqual(args.depths, [1, 8, 9, 16, 2**31 - 1])
        self.assertEqual(model_dimensions(2**31 - 1, 8, 1)["layers"], 2**31 - 1)
        self.assertFalse(args.output_dir.exists())
        self.assertFalse(args.checkpoint_dir.exists())

    def test_invalid_depth_is_rejected_by_runner_before_creating_directories(self):
        for depth in (0, -1, True, 16.0, "16", None, 2**31):
            with self.subTest(depth=depth):
                self.args.depths = [depth]
                with self.assertRaisesRegex(ValueError, "positive int32"):
                    self.run_driver()
                self.assertEqual(self.calls, [])
                self.assertFalse(self.args.output_dir.exists())
                self.assertFalse(self.args.checkpoint_dir.exists())

    def test_invalid_flags_are_rejected_without_filesystem_mutation(self):
        for flag in ("--depths=0", "--depths=-1", "--depths=1.5", "--depths=2147483648",
                     "--depths=1,1", "--depths=",
                     "--widths=0", "--widths=-1", "--widths=16,16", "--widths=x",
                     "--widths=536870912", "--batch_size=0", "--batch_size=2097152",
                     "--steps=-1", "--steps=2147483647", "--eval_every=0",
                     "--checkpoint_every=0", "--warmup_steps=-1", "--learning_rate=nan",
                     "--learning_rate=0", "--training_seconds=-1", "--training_seconds=inf",
                     "--seed=2147483648", "--attention_heads=-1", "--attention_heads=2147483648"):
            with self.subTest(flag=flag), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    parse_args(self.arguments + [flag])
                self.assertEqual(raised.exception.code, 2)
        self.assertFalse(self.args.output_dir.exists())
        self.assertFalse(self.args.checkpoint_dir.exists())

    def test_artifact_and_checkpoint_roots_cannot_overlap(self):
        for path in (self.args.output_dir, self.args.output_dir / "child", self.args.output_dir.parent):
            with self.subTest(path=path), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(self.arguments + [f"--checkpoint_dir={path}"])

    def test_defaults_preserve_the_original_training_schedule(self):
        args = parse_args(self.arguments[:-2])
        self.assertEqual(args.depths, list(range(1, 9)))
        self.assertEqual(args.widths, [256, 128, 64, 32, 16])
        self.assertEqual(args.attention_heads, 0)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.steps, 5000)
        self.assertEqual(args.eval_every, 128)
        self.assertEqual(args.checkpoint_every, 512)
        self.assertEqual(args.learning_rate, 6e-4)
        self.assertEqual(args.warmup_steps, 100)
        self.assertEqual(args.seed, 1337)
        self.assertEqual(args.training_seconds, 10800)


if __name__ == "__main__":
    unittest.main()
