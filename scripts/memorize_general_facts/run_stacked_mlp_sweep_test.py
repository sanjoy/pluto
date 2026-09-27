#!/usr/bin/env python3
"""CPU-only protocol, lifecycle, and audit tests for the stacked MLP sweep."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest import mock

import run_stacked_mlp_sweep as sweep


class StackedMlpSweepTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.binary = self.root / "native"
        self.binary.write_text("unused fake executable\n")
        self.binary.chmod(0o700)
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        (self.checkpoint / "compact_vocabulary.tsv").write_text("mock mapping\n")
        elements = [44750, 270]
        elements += [10, 10, 300, 30, 100, 10, 10, 10, 200, 20, 200, 10] * 4
        self.write_weights(self.checkpoint, elements + [10, 10])
        self.tokenizer = self.root / "tokenizer"
        self.tokenizer.mkdir()
        (self.tokenizer / "tokenizer.json").write_text("{}\n")
        self.corpus = self.root / "corpus.txt"
        self.corpus.write_text("fact\n" * 1024)
        self.arguments = [f"--binary={self.binary}", f"--checkpoint={self.checkpoint}",
                          f"--tokenizer={self.tokenizer}", f"--corpus={self.corpus}",
                          f"--run_dir={self.root / 'run'}"]
        self.calls = []
        self.next_pid = 321000

    @staticmethod
    def write_weights(directory, elements):
        directory.mkdir(exist_ok=True)
        for index, count in enumerate(elements):
            (directory / f"weight_{index}.bin").write_bytes(b"\0" * (4 * count))

    def result_files(self, command, stdout, *, bad_field=None):
        flags = dict(item[2:].split("=", 1) for item in command[1:] if "=" in item)
        depth, width = int(flags["mlp_depth"]), int(flags["mlp_width"])
        self.assertEqual(flags["match_mlp_parameter_budget"], "false")
        self.assertEqual(flags["mode"], "puzzle")
        output = Path(flags["output_dir"])
        output.mkdir()
        run = {**sweep.SHAPE, "train_mlp": 1, "mlp_depth": depth,
               "requested_mlp_width": width, "mlp_width": width,
               "match_mlp_parameter_budget": 0, "source_tail_parameters": 1380,
               "mlp_parameters": depth * (21 * width + 10),
               "trainable_parameters": sweep.parameter_count(depth, width),
               "checkpoint": flags["puzzle_checkpoint"], "corpus": flags["corpus"]}
        run.update({key: flags[key] for key in
                    ("steps", "eval_every", "batch_size", "seed", "learning_rate")})
        if bad_field:
            run[bad_field] = "99999"
        (output / "run.txt").write_text("".join(f"{key}={value}\n" for key, value in run.items()))
        steps, interval = int(flags["steps"]), int(flags["eval_every"])
        evaluated = list(range(0, steps + 1, interval))
        if evaluated[-1] != steps:
            evaluated.append(steps)
        (output / "training.tsv").write_text(
            "step\tseconds\tmean_ce\tcorrect\tscored\tcomplete_facts\teos_correct\teos_scored\n" +
            "".join(f"{step}\t{index + 1}.0\t2.00000000\t{2000 + index}\t10002\t0\t900\t1024\n"
                    for index, step in enumerate(evaluated)))
        correct = 2000 + len(evaluated) - 1
        stdout.write(
            "Original model: correct=10002/10002 scored next tokens\n"
            "Verified A3 separation: no identical hidden vectors have different scored next-token "
            "targets; scored=10002 unique_vectors=10002 distinct_targets=4000\n"
            f"BEST step={steps} updates={steps} seconds={len(evaluated) + 1}.00 mean_ce=2.00000000 "
            f"correct={correct}/10002 token_accuracy={100 * correct / 10002:.4f}% "
            "greedy_complete=1/1024 frozen_head_unchanged=true\n"
            f"EOS breakdown: eos_correct=900 eos_scored=1024 non_eos_correct={correct - 900} "
            "non_eos_scored=8978\n"
            "Frozen source parameters unchanged.\n")
        stdout.flush()
        self.write_weights(output / "best_mlp", sweep.checkpoint_elements(depth, width))

    def fake_process(self, command, *, stdout, stderr, start_new_session):
        self.assertTrue(start_new_session)
        self.calls.append(command)
        self.result_files(command, stdout)
        self.next_pid += 1
        return mock.Mock(pid=self.next_pid, poll=mock.Mock(return_value=0),
                         returncode=0, wait=mock.Mock(return_value=0))

    def run_fake(self, *, process=None, extra=()):
        args = sweep.parse_args(self.arguments + ["--depths=1,2", "--widths=20,40",
                                                "--steps=2", "--eval_every=1"] + list(extra))
        with redirect_stdout(io.StringIO()), mock.patch.object(sweep.time, "sleep"):
            result = sweep.run_sweep(args, popen=process or self.fake_process)
        return result, json.loads((args.run_dir / "summary.json").read_text())

    def test_defaults_and_parameter_grid(self):
        args = sweep.parse_args(self.arguments)
        self.assertEqual(args.depths, [1, 2, 3, 4, 5])
        self.assertEqual(args.widths, [20, 40, 80, 150])
        self.assertEqual((args.steps, args.eval_every, args.batch_size, args.seed,
                          args.learning_rate, args.max_workers), (300000, 1000, 32, 3, .01, 1))
        self.assertEqual(sweep.parameter_count(1, 20), 470)
        self.assertEqual(sweep.parameter_count(1, 150), 3200)
        self.assertEqual(sweep.parameter_count(3, 20), 1370)
        self.assertEqual(sweep.parameter_count(5, 150), 15920)
        self.assertEqual([sweep.budget_relation(value) for value in (1370, 1380, 3200)],
                         ["below", "equal", "above"])

    def test_cli_requires_all_explicit_paths(self):
        for name in ("binary", "checkpoint", "tokenizer", "corpus", "run_dir"):
            with self.subTest(name=name), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    sweep.parse_args([arg for arg in self.arguments if not arg.startswith(f"--{name}=")])

    def test_cli_rejects_invalid_controls(self):
        for value in ("--depths=0", "--depths=6", "--depths=1,1", "--widths=", "--widths=0",
                      "--steps=0", "--eval_every=0", "--batch_size=0", "--seed=-1",
                      "--max_workers=0", "--max_workers=5", "--learning_rate=nan",
                      "--learning_rate=inf", "--learning_rate=-1", "--timeout_seconds=0",
                      "--timeout_seconds=inf", "--timeout_seconds=nan"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    sweep.parse_args(self.arguments + [value])

    def test_cli_refuses_repository_run_directory(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            sweep.parse_args(self.arguments + [f"--run_dir={Path(sweep.__file__).parent / 'generated'}"])

    def test_complete_grid_snapshots_inputs_and_audits_all_records(self):
        code, summary = self.run_fake()
        self.assertEqual(code, 0)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(len(self.calls), 4)
        self.assertTrue(all(row["validated"] and row["status"] == "completed" for row in summary["trials"]))
        self.assertEqual(summary["trials"][0]["best_non_eos_accuracy"], 1102 / 8978)
        self.assertEqual(summary["ranking"][0], "D1_H20")
        manifest = json.loads((self.root / "run/manifest.json").read_text())
        self.assertEqual(len(manifest["commands"]), 4)
        self.assertIn("scripts/run_stacked_mlp_sweep.py", manifest["files"])
        self.assertIn("checkpoint/weight_0.bin", manifest["files"])
        for command in self.calls:
            self.assertEqual(Path(command[0]), self.root / "run/inputs/memorize_general_facts")
            self.assertIn(f"--puzzle_checkpoint={self.root / 'run/inputs/checkpoint'}", command)
        self.binary.write_text("rebuilt original binary\n")
        sweep.verify_snapshot(self.root / "run/inputs", manifest["files"])
        self.assertIn("Non-EOS accuracy", (self.root / "run/summary.html").read_text())
        self.assertEqual(len((self.root / "run/summary.tsv").read_text().splitlines()), 5)

    def test_existing_directory_and_wrong_source_shape_never_launch(self):
        args = sweep.parse_args(self.arguments)
        (self.checkpoint / "weight_0.bin").write_bytes(b"bad")
        process = mock.Mock()
        with self.assertRaisesRegex(ValueError, "tensor shape"):
            sweep.run_sweep(args, popen=process)
        self.assertFalse(args.run_dir.exists())
        args.run_dir.mkdir()
        with self.assertRaisesRegex(ValueError, "existing"):
            sweep.run_sweep(args, popen=process)
        process.assert_not_called()

    def test_success_exit_with_wrong_native_shape_is_failure(self):
        def bad(command, **kwargs):
            process = self.fake_process(command, **kwargs)
            output = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--output_dir=")))
            path = output / "run.txt"
            path.write_text(path.read_text().replace("model_width=10\n", "model_width=11\n"))
            return process
        code, summary = self.run_fake(process=bad)
        self.assertEqual(code, 1)
        self.assertEqual(summary["status"], "completed_with_failures")
        self.assertEqual(summary["ranking"], [])
        self.assertTrue(all(row["status"] == "failed" and row["exact_greedy_facts"] is None
                            for row in summary["trials"]))

    def test_native_failure_is_not_ranked_but_partial_metrics_survive(self):
        def failed(command, **kwargs):
            process = self.fake_process(command, **kwargs)
            process.poll.return_value = 2
            process.returncode = 2
            return process
        code, summary = self.run_fake(process=failed)
        self.assertEqual(code, 1)
        self.assertEqual(summary["ranking"], [])
        self.assertTrue(all(row["best_step"] == 2 and row["exact_greedy_facts"] is None
                            for row in summary["trials"]))

    def test_best_parser_rejects_missing_audits_counts_history_and_weights(self):
        self.run_fake(extra=["--depths=1", "--widths=20"])
        args = sweep.parse_args(self.arguments + ["--depths=1", "--widths=20", "--steps=2", "--eval_every=1"])
        record = json.loads((args.run_dir / "summary.json").read_text())["trials"][0]
        log = Path(record["directory"]) / "stdout.log"
        original = log.read_text()
        replacements = [("frozen_head_unchanged=true", "frozen_head_unchanged=false"),
                        ("Frozen source parameters unchanged.\n", ""),
                        ("greedy_complete=1/1024", "greedy_complete=1025/1024"),
                        ("updates=2", "updates=1"),
                        ("correct=2002/10002", "correct=2002/9999"),
                        ("step=2", "step=1"),
                        ("unique_vectors=10002", "unique_vectors=9999"),
                        ("eos_correct=900", "eos_correct=899"),
                        ("non_eos_scored=8978", "non_eos_scored=9000")]
        for before, after in replacements:
            with self.subTest(before=before):
                log.write_text(original.replace(before, after))
                with self.assertRaises(ValueError):
                    sweep.validate_result(args, record)
        log.write_text(original + next(line for line in original.splitlines() if line.startswith("BEST")) + "\n")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            sweep.validate_result(args, record)
        log.write_text(original)
        weight = Path(record["directory"]) / "puzzle/best_mlp/weight_0.bin"
        weight.write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "tensor shape"):
            sweep.validate_result(args, record)

    def test_partial_history_ignores_only_unfinished_line(self):
        path = self.root / "partial.tsv"
        header = "step\tseconds\tmean_ce\tcorrect\tscored\tcomplete_facts\teos_correct\teos_scored\n"
        path.write_text(header + "0\t0.1\t3\t1000\t10002\t0\t500\t1024\n1\t")
        self.assertEqual(len(sweep.read_history(path)), 1)
        with self.assertRaisesRegex(ValueError, "truncated"):
            sweep.read_history(path, complete=True)
        path.write_text(header + "0\t0.1\tnan\t1000\t10002\t0\t500\t1024\n")
        with self.assertRaisesRegex(ValueError, "mean_ce"):
            sweep.read_history(path)

    def test_native_best_may_choose_later_step_when_printed_losses_tie(self):
        self.run_fake(extra=["--depths=1", "--widths=20"])
        args = sweep.parse_args(self.arguments + ["--depths=1", "--widths=20", "--steps=2", "--eval_every=1"])
        record = json.loads((args.run_dir / "summary.json").read_text())["trials"][0]
        history = Path(record["directory"]) / "puzzle/training.tsv"
        history.write_text(history.read_text().replace("\t2001\t", "\t2002\t"))
        self.assertEqual(sweep.history_metrics(sweep.read_history(history))["best_step"], 1)
        result = sweep.validate_result(args, record)
        self.assertEqual(result["best_step"], 2)
        self.assertEqual(result["last_step"], 2)

    def test_ranking_orders_by_token_count_loss_greedy_then_parameters(self):
        base = {"status": "completed", "validated": True, "best_correct": 2000,
                "best_mean_ce": 2., "exact_greedy_facts": 1, "trainable_parameters": 1000,
                "depth": 1, "width": 20}
        rows = [{**base, "id": "base"}, {**base, "id": "loss", "best_mean_ce": 1.9},
                {**base, "id": "tokens", "best_correct": 2001},
                {**base, "id": "small", "trainable_parameters": 900},
                {**base, "id": "greedy", "exact_greedy_facts": 2},
                {**base, "id": "failed", "status": "failed", "best_correct": 10002}]
        self.assertEqual([row["id"] for row in sweep.ranking(rows)],
                         ["tokens", "loss", "greedy", "small", "base"])

    def test_timeout_terminates_process_group_and_preserves_other_runs(self):
        def stalled(command, **kwargs):
            process = self.fake_process(command, **kwargs)
            process.poll.return_value = None
            process.returncode = -signal.SIGTERM
            return process
        with mock.patch.object(sweep.os, "killpg") as kill:
            code, summary = self.run_fake(process=stalled, extra=["--timeout_seconds=0.000000001"])
        self.assertEqual(code, 1)
        self.assertEqual(kill.call_count, 4)
        self.assertTrue(all(row["status"] == "timeout" for row in summary["trials"]))

    def test_sigterm_cleans_children_and_never_launches_pending_trials(self):
        original = signal.getsignal(signal.SIGTERM)
        def interrupted(command, **kwargs):
            process = self.fake_process(command, **kwargs)
            process.poll.return_value = None
            process.returncode = -signal.SIGTERM
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return process
        with mock.patch.object(sweep.os, "killpg") as kill:
            code, summary = self.run_fake(process=interrupted, extra=["--max_workers=1"])
        self.assertEqual(code, 128 + signal.SIGTERM)
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual([row["status"] for row in summary["trials"]],
                         ["interrupted", "not_started", "not_started", "not_started"])
        kill.assert_called_once_with(self.next_pid, signal.SIGTERM)
        self.assertEqual(signal.getsignal(signal.SIGTERM), original)

    def test_kill_escalation_reaps_child(self):
        process = mock.Mock(pid=98765, poll=mock.Mock(return_value=None))
        process.wait.side_effect = [subprocess.TimeoutExpired("mock", 5), -9]
        with mock.patch.object(sweep.os, "killpg") as kill:
            sweep.stop_processes({"mock": {"process": process}})
        self.assertEqual(kill.call_args_list,
                         [mock.call(98765, signal.SIGTERM), mock.call(98765, signal.SIGKILL)])
        self.assertEqual(process.wait.call_count, 2)


if __name__ == "__main__":
    unittest.main()
