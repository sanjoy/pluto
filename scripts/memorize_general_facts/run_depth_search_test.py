#!/usr/bin/env python3
"""No-GPU tests for sequential depth-search control and fail-closed artifacts."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from run_depth_search import parse_args, run_search


class DepthSearchTest(unittest.TestCase):
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
            f"--binary={self.binary}",
            f"--corpus={self.corpus}",
            f"--tokenizer={self.tokenizer}",
            f"--checkpoint_dir={self.root / 'checkpoints'}",
            f"--output_dir={self.root / 'artifacts'}",
            "--start_layers=2",
        ]
        self.args = parse_args(self.arguments)
        self.calls = []
        self.statuses = {}
        self.mutate = lambda phase, path: None

    def fake_subprocess(self, command, *, check, stdout):
        # This strict signature rejects accidental shell=True, capture_output,
        # and timeout additions that would break streaming/liveness semantics.
        self.assertFalse(check)
        self.calls.append(command)
        flags = dict(
            argument[2:].split("=", 1)
            for argument in command
            if argument.startswith("--")
        )
        if Path(command[0]) == self.binary:
            phase = "verify" if "verify_checkpoint" in flags else "train"
            layers = int(flags["layers"])
            self.assertIsNone(stdout)
        else:
            phase = "audit"
            verifier = Path(__file__).resolve().with_name("verify_predictions.py")
            self.assertEqual(Path(command[1]), verifier)
            self.assertTrue(verifier.is_file())
            layers = int(Path(flags["predictions"]).parent.parent.name.split("_")[1])
            self.assertIsNotNone(stdout)
        code = self.statuses.get((phase, layers), 0)
        if code:
            return subprocess.CompletedProcess(command, code)

        if phase == "train":
            self.assertEqual(flags["search"], "false")
            output = Path(flags["output_dir"]) / f"layers_{layers}"
            output.mkdir()
            shutil.copyfile(flags["corpus"], output / "corpus.txt")
            shutil.copyfile(
                Path(flags["tokenizer"]) / "tokenizer.json", output / "tokenizer.json"
            )
            checkpoint = Path(flags["checkpoint_dir"]) / f"layers_{layers}" / "step_2"
            checkpoint.mkdir(parents=True)
            result = {
                "success": 1,
                "layers": layers,
                "errors": 0,
                "targets": 10002,
                "step": 2,
                "checkpoint": checkpoint,
            }
        elif phase == "verify":
            output = Path(flags["output_dir"])
            output.mkdir()
            result = {
                "checkpoint": flags["verify_checkpoint"],
                "layers": layers,
                "errors": 0,
                "targets": 10002,
                "sentences": 1024,
                "exact_sentences": 1024,
            }
            (output / "final_predictions.tsv").write_text("mock prediction artifact\n")
        else:
            output = Path(flags["predictions"]).parent
            audit = {
                "success": True,
                "errors": 0,
                "targets": 10002,
                "sentences": 1024,
                "exact_sentences": 1024,
                "corpus_sha256": hashlib.sha256(self.corpus.read_bytes()).hexdigest(),
                "tokenizer_sha256": hashlib.sha256(
                    (self.tokenizer / "tokenizer.json").read_bytes()
                ).hexdigest(),
            }
            json.dump(audit, stdout)
            stdout.flush()
            self.mutate(phase, Path(stdout.name))
            return subprocess.CompletedProcess(command, 0)
        result_path = output / "result.txt"
        result_path.write_text(
            "".join(f"{key}={value}\n" for key, value in result.items())
        )
        self.mutate(phase, result_path)
        return subprocess.CompletedProcess(command, 0)

    def run_driver(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return run_search(self.args, run_process=self.fake_subprocess)

    def summary(self):
        return json.loads(
            (self.args.output_dir / "depth_search_summary.json").read_text()
        )

    def test_each_depth_is_verified_twice_before_starting_next_depth(self):
        self.assertEqual(self.run_driver(), 0)
        self.assertEqual(len(self.calls), 6)
        self.assertIn("--layers=2", self.calls[0])
        self.assertIn("--layers=2", self.calls[1])
        self.assertEqual(
            Path(self.calls[2][1]),
            Path(__file__).resolve().with_name("verify_predictions.py"),
        )
        self.assertIn("--layers=1", self.calls[3])
        self.assertIn("--layers=1", self.calls[4])
        self.assertFalse(any("--layers=0" in command for command in self.calls))
        summary = self.summary()
        self.assertEqual(summary["status"], "all_requested_depths_verified")
        self.assertEqual(summary["smallest_verified_layers"], 1)
        self.assertTrue(summary["skipped_higher_depths_require_external_verification"])
        self.assertEqual(
            [depth["status"] for depth in summary["depths"]], ["verified", "verified"]
        )

    def test_budget_failure_stops_immediately_without_verification_or_retry(self):
        self.statuses["train", 2] = 2
        self.assertEqual(self.run_driver(), 2)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.summary()["status"], "budget_exhausted")
        self.assertIsNone(self.summary()["smallest_verified_layers"])

    def test_historical_full_vocabulary_is_explicit_in_both_native_phases(self):
        self.assertEqual(self.run_driver(), 0)
        native = [command for command in self.calls if Path(command[0]) == self.binary]
        self.assertEqual(len(native), 4)
        for command in native:
            self.assertEqual(command.count("--compact_vocabulary=false"), 1)
        self.assertEqual(sum(any(flag.startswith("--verify_checkpoint=") for flag in command)
                             for command in native), 2)

    def test_later_failure_preserves_preceding_verified_minimum(self):
        self.statuses["train", 1] = 2
        self.assertEqual(self.run_driver(), 2)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(self.summary()["smallest_verified_layers"], 2)

    def test_native_commands_omit_removed_gradient_clipping_flag(self):
        self.assertEqual(self.run_driver(), 0)
        native = [command for command in self.calls if Path(command[0]) == self.binary]
        self.assertEqual(len(native), 4)
        for command in native:
            self.assertFalse(any(flag.startswith("--gradient_clip_norm")
                                 for flag in command))
        for depth in self.summary()["depths"]:
            for command in depth["commands"]:
                self.assertFalse(any(flag.startswith("--gradient_clip_norm")
                                     for flag in command))

    def test_training_execution_error_does_not_start_another_child(self):
        self.statuses["train", 2] = 1
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.summary()["status"], "training_failed")

    def test_native_verification_failure_prevents_artifact_audit_and_next_depth(self):
        self.statuses["verify", 2] = 2
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.summary()["status"], "checkpoint_verification_failed")

    def test_artifact_failure_prevents_next_depth(self):
        self.statuses["audit", 2] = 1
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(
            self.summary()["status"], "prediction_artifact_verification_failed"
        )

    def test_success_exit_cannot_override_nonzero_result_errors(self):
        def mutate(phase, path):
            if phase == "train":
                path.write_text(path.read_text().replace("errors=0", "errors=1"))

        self.mutate = mutate
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertIn("expected errors=0", self.summary()["error"])

    def test_checkpoint_must_be_exact_expected_step_under_explicit_parent(self):
        def mutate(phase, path):
            if phase == "train":
                text = path.read_text()
                start = text.index("checkpoint=")
                path.write_text(text[:start] + f"checkpoint={self.root}\n")

        self.mutate = mutate
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertIn("Unexpected or missing final checkpoint", self.summary()["error"])

    def test_mismatched_snapshot_is_not_verified_as_original_corpus(self):
        def mutate(phase, path):
            if phase == "train":
                (path.parent / "corpus.txt").write_text("different data\n")

        self.mutate = mutate
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertIn("snapshots", self.summary()["error"])

    def test_invalid_audit_success_claim_stops_search(self):
        def mutate(phase, path):
            if phase == "audit":
                audit = json.loads(path.read_text())
                audit["targets"] = 10001
                path.write_text(json.dumps(audit))

        self.mutate = mutate
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.calls), 3)
        self.assertIn("unexpected targets", self.summary()["error"])

    def test_missing_result_after_success_is_a_failure(self):
        def mutate(phase, path):
            if phase == "train":
                path.unlink()

        self.mutate = mutate
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.summary()["status"], "error")

    def test_binary_changes_between_depths_stop_the_controlled_comparison(self):
        def mutate(phase, path):
            if phase == "audit":
                self.binary.write_text("different executable\n")

        self.mutate = mutate
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.summary()["smallest_verified_layers"], 2)
        self.assertEqual(self.summary()["depths"][0]["status"], "verified")
        self.assertIn("binary changed", self.summary()["error"])

    def test_existing_output_or_checkpoint_roots_are_never_reused(self):
        for path in (self.args.output_dir, self.args.checkpoint_dir):
            with self.subTest(path=path):
                path.mkdir()
                with self.assertRaisesRegex(ValueError, "existing run directory"):
                    self.run_driver()
                self.assertEqual(self.calls, [])
                path.rmdir()

    def test_invalid_flags_are_rejected_before_any_work(self):
        for flag in (
            "--start_layers=0",
            "--start_layers=9",
            "--batch_size=0",
            "--batch_size=2097152",
            "--steps=-1",
            "--steps=2147483647",
            "--warmup_steps=-1",
            "--eval_every=0",
            "--checkpoint_every=0",
            "--learning_rate=nan",
            "--learning_rate=0",
            "--training_seconds=inf",
            "--training_seconds=-1",
            "--seed=2147483648",
        ):
            with self.subTest(flag=flag), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    parse_args(self.arguments + [flag])
                self.assertEqual(raised.exception.code, 2)
        self.assertFalse(self.args.output_dir.exists())
        self.assertFalse(self.args.checkpoint_dir.exists())

    def test_output_and_checkpoint_paths_cannot_overlap(self):
        for path in (
            self.args.output_dir,
            self.args.output_dir / "checkpoints",
            self.args.output_dir.parent,
        ):
            with self.subTest(path=path), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(self.arguments + [f"--checkpoint_dir={path}"])

    def test_default_training_configuration_is_unchanged(self):
        args = parse_args(self.arguments[:-1])
        self.assertEqual(args.start_layers, 8)
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
