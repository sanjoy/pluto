#!/usr/bin/env python3
"""CPU-only checks for the from-scratch puzzle workflow's safety gates."""

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("reproduce_puzzle.sh").resolve()
BASH = shutil.which("bash")


class ReproducePuzzleTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="puzzle workflow ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model with spaces"
        self.trial_dir = self.model / "trial_000_L4_W10_FF20"
        self.checkpoint = self.trial_dir / "checkpoints" / "layers_4" / "step_67840"
        self.checkpoint.mkdir(parents=True)
        (self.checkpoint / "weight_0.bin").write_bytes(b"fixture weights")
        (self.checkpoint / "compact_vocabulary.tsv").write_text("fixture mapping\n")
        self.trial = {
            "layers": 4,
            "width": 10,
            "feed_forward_width": 20,
            "parameters": 48680,
            "steps": 120000,
            "learning_rate": 0.0012,
            "status": "verified",
            "directory": str(self.trial_dir),
            "step": 67840,
            "errors": 0,
            "checkpoint": str(self.checkpoint),
            "last_returncode": 0,
            "training_result": {
                "success": "1",
                "layers": "4",
                "width": "10",
                "feed_forward_width": "20",
                "heads": "1",
                "context_length": "27",
                "vocabulary": "4475",
                "parameters": "48680",
                "step": "67840",
                "errors": "0",
                "targets": "10002",
                "reached_time_limit": "0",
                "checkpoint": str(self.checkpoint),
            },
        }
        self.summary = {
            "status": "completed",
            "trials": [self.trial],
            "minimum_parameter_success": copy.deepcopy(self.trial),
        }
        self.prediction = {
            "success": True,
            "sentences": 1024,
            "exact_sentences": 1024,
            "targets": 10002,
            "errors": 0,
            "accuracy": 1.0,
            "mean_loss_nats": 0.004,
            "context_length": 27,
        }
        self.greedy = {
            "sentences": 1024,
            "exact_sentences": 1024,
            "errors": 0,
            "incorrect_lines_1based": [],
        }
        self.write_artifacts()

    def write_artifacts(self):
        for path, value in (
            (self.model / "summary.json", self.summary),
            (self.trial_dir / "prediction_verification.json", self.prediction),
            (self.trial_dir / "greedy_verification.json", self.greedy),
        ):
            path.write_text(json.dumps(value) + "\n")

    def select(self):
        self.write_artifacts()
        return self.run_selector()

    def run_selector(self):
        # Positional arguments deliberately preserve spaces without shell quoting.
        return subprocess.run(
            [BASH, "-c", 'source "$1"; select_verified_checkpoint "$2"',
             "selector-test", str(SCRIPT), str(self.model)],
            text=True, capture_output=True, timeout=10,
        )

    def assert_rejected(self, result):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), "", result.stdout)
        self.assertTrue(result.stderr.strip(), "Rejection needs an explanation")

    def test_source_is_safe_without_running_workflow(self):
        result = subprocess.run(
            [BASH, "-c", 'source "$1"; printf "source-only\\n"',
             "source-test", str(SCRIPT)],
            text=True, capture_output=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "source-only\n")

    def test_selects_actual_verified_step_with_spaces_in_paths(self):
        result = self.select()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, str(self.checkpoint) + "\n")

    def test_completed_driver_with_budget_failure_is_not_success(self):
        # The sweep intentionally returns zero for an exhausted training budget.
        self.trial.update(status="budget_fail", errors=3, last_returncode=0)
        self.assert_rejected(self.select())

    def test_driver_timeout_is_not_success_even_with_zero_return_code(self):
        self.summary["status"] = "deadline"
        self.trial.update(status="timeout", last_returncode=0)
        self.assert_rejected(self.select())

    def test_only_completed_summary_can_supply_checkpoint(self):
        for status in ("running", "deadline", "error", "interrupted", "unknown", None):
            with self.subTest(status=status):
                self.summary["status"] = status
                self.assert_rejected(self.select())

    def test_only_verified_trial_can_supply_checkpoint(self):
        for status in ("running", "budget_fail", "timeout", "error", "interrupted", None):
            with self.subTest(status=status):
                self.trial["status"] = status
                self.assert_rejected(self.select())

    def test_summary_requires_exactly_one_trial(self):
        for trials in ([], [self.trial, copy.deepcopy(self.trial)]):
            with self.subTest(count=len(trials)):
                self.summary["trials"] = trials
                self.assert_rejected(self.select())

    def test_trial_requires_zero_errors_and_expected_parameter_count(self):
        for key, value in (("errors", 1), ("parameters", 48679), ("parameters", 48681)):
            with self.subTest(key=key, value=value):
                original = self.trial[key]
                self.trial[key] = value
                self.assert_rejected(self.select())
                self.trial[key] = original

    def test_missing_checkpoint_directory_is_rejected(self):
        shutil.rmtree(self.checkpoint)
        self.assert_rejected(self.select())

    def test_file_cannot_masquerade_as_checkpoint_directory(self):
        shutil.rmtree(self.checkpoint)
        self.checkpoint.write_text("not a checkpoint directory\n")
        self.assert_rejected(self.select())

    def test_checkpoint_step_must_match_summary(self):
        self.trial["step"] += 256
        self.assert_rejected(self.select())

    def test_training_result_step_must_match_summary(self):
        self.trial["training_result"]["step"] = "67584"
        self.assert_rejected(self.select())

    def test_training_result_checkpoint_must_match_summary(self):
        self.trial["training_result"]["checkpoint"] = str(self.root / "different checkpoint")
        self.assert_rejected(self.select())

    def test_checkpoint_outside_recorded_trial_is_rejected(self):
        unrelated = self.root / "unrelated checkpoint" / "step_67840"
        unrelated.mkdir(parents=True)
        self.trial["checkpoint"] = str(unrelated)
        self.trial["training_result"]["checkpoint"] = str(unrelated)
        self.assert_rejected(self.select())

    def test_checkpoint_symlink_outside_trial_is_rejected(self):
        unrelated = self.root / "unrelated checkpoint"
        self.checkpoint.rename(unrelated)
        self.checkpoint.symlink_to(unrelated, target_is_directory=True)
        self.assert_rejected(self.select())

    def test_missing_independent_report_is_rejected(self):
        for name in ("prediction_verification.json", "greedy_verification.json"):
            with self.subTest(report=name):
                self.write_artifacts()
                (self.trial_dir / name).unlink()
                self.assert_rejected(self.run_selector())

    def test_failed_or_incomplete_prediction_audit_is_rejected(self):
        for key, value in (("success", False), ("errors", 1), ("targets", 10001),
                           ("sentences", 1023), ("exact_sentences", 1023)):
            with self.subTest(key=key, value=value):
                original = self.prediction[key]
                self.prediction[key] = value
                self.assert_rejected(self.select())
                self.prediction[key] = original

    def test_failed_or_incomplete_greedy_audit_is_rejected(self):
        for key, value in (("errors", 1), ("sentences", 1023),
                           ("exact_sentences", 1023), ("incorrect_lines_1based", [1])):
            with self.subTest(key=key, value=value):
                original = self.greedy[key]
                self.greedy[key] = value
                self.assert_rejected(self.select())
                self.greedy[key] = original

    def test_malformed_or_missing_summary_is_rejected(self):
        path = self.model / "summary.json"
        path.write_text("{unfinished JSON")
        self.assert_rejected(self.run_selector())
        path.unlink()
        self.assert_rejected(self.run_selector())

    def cli(self, *arguments):
        restricted = self.root / "restricted commands"
        restricted.mkdir(exist_ok=True)
        for name in ("bash", "cat", "dirname", "realpath", "readlink", "date", "pwd"):
            binary = shutil.which(name)
            link = restricted / name
            if binary and not link.exists():
                link.symlink_to(binary)
        python = restricted / "python3"
        if not python.exists():
            python.symlink_to(sys.executable)
        marker = self.root / "unexpected dependency call"
        for name in ("nvidia-smi", "nvcc", "tileiras", "bazel", "bazelisk",
                     "curl", "wget", "pip", "pip3", "git"):
            stub = restricted / name
            stub.write_text('#!/bin/bash\nprintf "%s\\n" "$0" >> "$WORKFLOW_TEST_MARKER"\nexit 98\n')
            stub.chmod(0o700)
        env = dict(os.environ, PATH=str(restricted), WORKFLOW_TEST_MARKER=str(marker))
        result = subprocess.run([BASH, str(SCRIPT), *arguments], env=env,
                                text=True, capture_output=True, timeout=10)
        self.assertFalse(marker.exists(),
                         f"Setup dependency ran before argument validation: {result.stderr}")
        self.assertNotIn("command not found", result.stderr)
        return result

    def test_help_does_not_require_gpu_network_or_build_tools(self):
        result = self.cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--run_dir", result.stdout)
        self.assertIn("--model_steps", result.stdout)
        self.assertIn("--puzzle_steps", result.stdout)

    def test_invalid_options_fail_before_setup(self):
        for arguments in ((), ("--unknown",), ("--run_dir",),
                          ("--run_dir", str(self.root / "new run"), "--model_steps", "0"),
                          ("--run_dir", str(self.root / "new run"), "--puzzle_steps", "-1"),
                          ("--run_dir", str(self.root / "new run"), "--mlp_width", "oops"),
                          ("--run_dir", str(self.root / "new run"), "--training_timeout", "0"),
                          ("--run_dir", str(self.root / "new run"), "--training_timeout=300"),
                          ("--run_dir", str(self.root / "new run"), "--model_steps=1000000000"),
                          ("--run_dir", str(self.root / "new run"), "--model_steps=1e5")):
            with self.subTest(arguments=arguments):
                result = self.cli(*arguments)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertTrue(result.stderr.strip())
                self.assertFalse((self.root / "new run").exists())

    def test_existing_run_directory_is_rejected_before_setup(self):
        run_dir = self.root / "already exists"
        run_dir.mkdir()
        sentinel = run_dir / "user data.txt"
        sentinel.write_text("preserve this\n")
        result = self.cli("--run_dir", str(run_dir))
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("exist", result.stderr.lower())
        self.assertEqual(sentinel.read_text(), "preserve this\n")

    def test_dangling_symlink_run_directory_is_rejected_before_setup(self):
        run_dir = self.root / "dangling run symlink"
        run_dir.symlink_to(self.root / "missing destination", target_is_directory=True)
        result = self.cli(f"--run_dir={run_dir}")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("exist", result.stderr.lower())
        self.assertTrue(run_dir.is_symlink())

    def test_run_directory_inside_checkout_is_rejected_before_setup(self):
        run_dir = SCRIPT.parent / "must-not-create-this-run"
        self.assertFalse(run_dir.exists())
        result = self.cli("--run_dir", str(run_dir))
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("outside", result.stderr.lower())
        self.assertFalse(run_dir.exists())


if __name__ == "__main__":
    unittest.main()
