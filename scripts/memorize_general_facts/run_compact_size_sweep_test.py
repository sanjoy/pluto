#!/usr/bin/env python3
"""CPU-only validation of compact-size sweep inputs and parameter accounting."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from run_compact_size_sweep import parameter_count, parse_args, parse_candidate, run_search
from verify_predictions import TSV_HEADER


class CompactSizeSweepTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.binary = self.root / "native"
        self.binary.write_text("fake executable never launched\n")
        self.binary.chmod(0o700)
        self.corpus = self.root / "corpus.txt"
        self.corpus.write_text("fact\n")
        self.tokenizer = self.root / "tokenizer"
        self.tokenizer.mkdir()
        (self.tokenizer / "tokenizer.json").write_text("{}\n")
        self.arguments = [
            f"--binary={self.binary}", f"--corpus={self.corpus}",
            f"--tokenizer={self.tokenizer}",
            f"--run_dir={self.root / 'run'}", "--deadline_unix=4000000000",
            "--candidates=8:12:48:1000:0.001,12:8:32:2000:0.002",
        ]

    def test_parameter_counts_for_compact_candidates(self):
        for dimensions, expected in [
            ((8, 16, 64), 98384),
            ((8, 12, 48), 69180),
            ((8, 10, 40), 55730),
            ((8, 14, 56), 83398),
            ((12, 8, 32), 46536),
        ]:
            with self.subTest(dimensions=dimensions):
                self.assertEqual(parameter_count(*dimensions), expected)

    def test_parameter_count_includes_each_unique_array_once(self):
        # Spell out the embedding, position embedding, two norms, attention
        # projections, feed-forward projections, and final norm independently.
        layers, width, ff, vocabulary, context = 3, 5, 11, 17, 7
        per_layer = [width, width, width * 3 * width, 3 * width,
                     width * width, width, width, width, width * ff, ff,
                     ff * width, width]
        expected = vocabulary * width + context * width
        expected += layers * sum(per_layer) + 2 * width
        self.assertEqual(parameter_count(layers, width, ff,
                                        vocabulary=vocabulary, context=context),
                         expected)

    def test_candidate_accepts_explicit_training_schedule(self):
        self.assertEqual(parse_candidate("8:12:48:16384:3e-3"), {
            "layers": 8, "width": 12, "feed_forward_width": 48,
            "steps": 16384, "learning_rate": 0.003,
        })

    def test_candidate_rejects_malformed_or_unsafe_values(self):
        for candidate in [
            "", "8:12:48:1000", "8:12:48:1000:0.001:extra",
            "bad:12:48:1000:0.001", "8:12.5:48:1000:0.001",
            "1:12:48:1000:0.001", "0:12:48:1000:0.001",
            "8:0:48:1000:0.001", "8:-1:48:1000:0.001",
            "8:12:0:1000:0.001", "8:12:48:0:0.001",
            "8:12:48:-1:0.001", "8:12:48:1000:0",
            "8:12:48:1000:-0.001", "8:12:48:1000:nan",
            "8:12:48:1000:inf", "8:12:48:1000:-inf",
        ]:
            with self.subTest(candidate=candidate):
                with self.assertRaises((ValueError, argparse.ArgumentTypeError)):
                    parse_candidate(candidate)

    def test_cli_defaults_and_candidate_order(self):
        args = parse_args(self.arguments)
        self.assertEqual(args.seed, 1337)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.context_length, 32)
        self.assertEqual(args.vocabulary, 4475)
        self.assertEqual(args.verification_reserve, 30)
        self.assertEqual(args.eval_every, 256)
        self.assertEqual(args.deadline_unix, 4000000000.0)
        self.assertEqual(args.candidates, [parse_candidate("8:12:48:1000:0.001"),
                                           parse_candidate("12:8:32:2000:0.002")])

    def test_cli_rejects_invalid_bounds(self):
        for option in [
            "--deadline_unix=nan", "--deadline_unix=inf",
            "--deadline_unix=-1", "--batch_size=0", "--context_length=0",
            "--vocabulary=0", "--verification_reserve=-1",
            "--verification_reserve=nan", "--eval_every=0",
            "--candidates=", "--candidates=1:12:48:1000:0.001",
        ]:
            with self.subTest(option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(self.arguments + [option])

    def test_cli_requires_explicit_paths_and_deadline(self):
        for option in ("--binary=", "--corpus=", "--tokenizer=",
                       "--run_dir=", "--deadline_unix="):
            arguments = [value for value in self.arguments
                         if not value.startswith(option)]
            with self.subTest(option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(arguments)

    def fake_inputs(self, *unused):
        # These lengths independently give exactly 10,002 scored targets.
        rows = [[1] * 14 for _ in range(786)] + [[1] * 13 for _ in range(238)]
        return {"lines": ["fact"] * 1024, "rows": rows,
                "prompts": ["prompt"] * 1024, "active": (1, 50256)}

    def fake_process(self, command, stdout_path, stderr_path, *, deadline,
                     input_path=None, on_start=None):
        self.calls.append(command)
        flags = dict(item[2:].split("=", 1) for item in command[1:])
        if on_start:
            on_start(12345)
        stdout_path.write_text("")
        stderr_path.write_text("")
        if "infer_checkpoint" in flags:
            self.assertIsNotNone(input_path)
            self.assertEqual(len(input_path.read_text().splitlines()), 1024)
            stdout_path.write_text(
                "Enter a prompt (Ctrl-D or Ctrl-C to quit). Each line starts a new completion.\n"
                + "> fact\n" * 1024 + "> ")
            return 0
        if "verify_checkpoint" in flags and self.timeout_verification:
            raise TimeoutError("mock verification deadline")
        layers, width, ff = (int(flags[name]) for name in
                             ("layers", "model_width", "feed_forward_width"))
        result = {"layers": layers, "width": width, "feed_forward_width": ff,
                  "parameters": parameter_count(layers, width, ff), "heads": 1,
                  "vocabulary": 4475, "targets": 10002, "errors": 0}
        output = Path(flags["output_dir"])
        if flags["mode"] == "train_model":
            self.assertEqual(flags["search"], "false")
            self.assertEqual(flags["context_length"], "32")
            self.assertEqual(flags["compact_vocabulary"], "true")
            output /= f"layers_{layers}"
            checkpoint = Path(flags["checkpoint_dir"]) / f"layers_{layers}" / "step_256"
            checkpoint.mkdir(parents=True)
            output.mkdir(parents=True)
            mapping = ("compact_vocabulary_v1\noriginal_vocab_size\t50257\n"
                       "original_eos_token\t50256\ncompact_vocab_size\t2\n"
                       "compact_id\toriginal_id\n0\t1\n1\t50256\n")
            for directory in (checkpoint, output):
                (directory / "compact_vocabulary.tsv").write_text(mapping)
            (output / "config.txt").write_text("context_length=32\nbatch_size=32\n")
            shutil.copyfile(flags["corpus"], output / "corpus.txt")
            shutil.copyfile(Path(flags["tokenizer"]) / "tokenizer.json",
                            output / "tokenizer.json")
            result.update(success=1, step=256, checkpoint=str(checkpoint), reached_time_limit=0)
        else:
            output.mkdir(parents=True)
            result.update(sentences=1024, exact_sentences=1024,
                          checkpoint=flags["verify_checkpoint"], mean_loss=0.01)
        (output / "result.txt").write_text(
            "".join(f"{key}={value}\n" for key, value in result.items()))
        records = ["\t".join(TSV_HEADER)]
        for line, row in enumerate(self.fake_inputs()["rows"], 1):
            for position in range(5, len(row) + 1):
                target = 50256 if position == len(row) else row[position]
                records.append(f"{line}\t{position}\t{target}\t{target}\t0.01")
        (output / "final_predictions.tsv").write_text("\n".join(records) + "\n")
        return 0

    def run_fake(self, **options):
        self.calls = []
        self.timeout_verification = options.pop("timeout_verification", False)
        args = parse_args(self.arguments)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            status = run_search(args, run_process=options.pop("run_process", self.fake_process),
                                loader=self.fake_inputs, clock=options.pop("clock", lambda: 100),
                                **options)
        return status, json.loads((args.run_dir / "summary.json").read_text())

    def test_expired_deadline_never_launches_or_creates_run(self):
        args = parse_args(self.arguments)
        runner = mock.Mock()
        with self.assertRaisesRegex(ValueError, "deadline"):
            run_search(args, run_process=runner, loader=self.fake_inputs,
                       clock=lambda: args.deadline_unix)
        runner.assert_not_called()
        self.assertFalse(args.run_dir.exists())

    def test_verification_reserve_prevents_a_training_launch(self):
        status, summary = self.run_fake(clock=lambda: 4000000000 - 20)
        self.assertEqual(status, 0)
        self.assertEqual(summary["status"], "deadline")
        self.assertEqual(summary["trials"], [])
        self.assertEqual(self.calls, [])

    def test_existing_run_directory_is_preserved(self):
        args = parse_args(self.arguments)
        args.run_dir.mkdir()
        sentinel = args.run_dir / "existing.txt"
        sentinel.write_text("preserve this")
        runner = mock.Mock()
        with self.assertRaisesRegex(ValueError, "existing"):
            run_search(args, run_process=runner, loader=self.fake_inputs,
                       clock=lambda: 100)
        runner.assert_not_called()
        self.assertEqual(sentinel.read_text(), "preserve this")
        self.assertEqual(list(args.run_dir.iterdir()), [sentinel])

    def test_missing_native_artifacts_cannot_claim_success(self):
        status, summary = self.run_fake(run_process=lambda *args, **kwargs: 0)
        self.assertEqual(status, 1)
        self.assertEqual(summary["status"], "error")
        self.assertEqual(summary["trials"][0]["status"], "error")
        self.assertIsNone(summary["minimum_parameter_success"])

    def test_timeout_during_fresh_verification_cannot_claim_success(self):
        status, summary = self.run_fake(timeout_verification=True)
        self.assertEqual(status, 0)
        self.assertEqual(summary["status"], "deadline")
        self.assertEqual(summary["trials"][0]["status"], "timeout")
        self.assertEqual(summary["trials"][0]["phase"], "verification")
        self.assertIsNone(summary["minimum_parameter_success"])
        self.assertEqual(len(self.calls), 2)

    def test_success_requires_fresh_reload_predictions_and_all_completions(self):
        status, summary = self.run_fake()
        self.assertEqual(status, 0)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual([trial["status"] for trial in summary["trials"]],
                         ["verified", "verified"])
        self.assertEqual(len(self.calls), 6)
        self.assertEqual(summary["minimum_parameter_success"]["parameters"], 46536)
        for trial in summary["trials"]:
            directory = Path(trial["directory"])
            prediction = json.loads((directory / "prediction_verification.json").read_text())
            greedy = json.loads((directory / "greedy_verification.json").read_text())
            self.assertEqual(prediction["targets"], 10002)
            self.assertEqual(prediction["errors"], 0)
            self.assertEqual(greedy["exact_sentences"], 1024)


if __name__ == "__main__":
    unittest.main()
