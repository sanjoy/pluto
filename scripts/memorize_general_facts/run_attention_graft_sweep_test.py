#!/usr/bin/env python3
"""No-GPU tests for attention graft scheduling, metrics, and durable reports."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock

import run_attention_graft_sweep as sweep


EVALUATIONS = (
    "step=0 seconds=0.03 wrong=9992/10002 failed_facts=1024/1024 "
    "margin_loss=11.60000000 min_margin=-7.100000\n"
    "step=20000 seconds=20.03 wrong=6877/10002 failed_facts=1023/1024 "
    "margin_loss=7.20000000 min_margin=-15.100000\n"
)
FINAL = (
    "FINAL best_step=18000 updates=20000 seconds=20.41 wrong=6877/10002 "
    "autoregressive_complete=1/1024 frozen_weights_unchanged=true\n"
)


class AttentionGraftSweepTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.binary = self.root / "native"
        self.binary.write_text("test fixture: never executed\n")
        self.binary.chmod(0o700)
        self.corpus = self.root / "corpus.txt"
        self.corpus.write_text("a fact\n")
        self.tokenizer = self.root / "tokenizer"
        self.tokenizer.mkdir()
        (self.tokenizer / "tokenizer.json").write_text("{}\n")
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        (self.checkpoint / "weight_0.bin").write_bytes(b"checkpoint fixture")
        (self.checkpoint / "compact_vocabulary.tsv").write_text("compact vocabulary fixture\n")
        self.arguments = [f"--binary={self.binary}", f"--source-checkpoint={self.checkpoint}",
                          f"--corpus={self.corpus}", f"--tokenizer={self.tokenizer}",
                          f"--output={self.root / 'output'}"]

    def test_all_55_coordinate_variants_are_unique_and_one_based(self):
        variants = sweep.coordinate_variants()
        columns = [tuple(map(int, sweep.flag_map(v["flags"])["graft_columns"].split(",")))
                   for v in variants]
        self.assertEqual(len(columns), 55)
        self.assertEqual(len(set(columns)), 55)
        self.assertEqual(sum(len(c) == 1 for c in columns), 10)
        self.assertEqual(sum(len(c) == 2 for c in columns), 45)
        self.assertEqual(columns[0], (1,))
        self.assertEqual(columns[-1], (9, 10))

    def test_default_fresh_fits_have_exact_controls_and_compressed_schedule(self):
        args = sweep.parse_args(self.arguments)
        variants = sweep.load_variants(args)
        self.assertEqual(len(variants), 57)
        flags = dict(item[2:].split("=", 1) for item in
                     sweep.build_command(args, variants[1], self.root / "trial")[1:])
        self.assertEqual(flags["block"], "3")
        self.assertNotIn("graft_block", flags)
        self.assertEqual(flags["steps"], "20000")
        self.assertEqual(flags["random_init"], "3")
        self.assertEqual(flags["seed"], "3")
        self.assertEqual(flags["train_final_norm"], "true")
        self.assertEqual(flags["fresh_branch"], "true")
        self.assertEqual(flags["readout_width"], "20")
        self.assertEqual(flags["learning_rate"], "0.001")
        self.assertEqual(flags["final_rate_ratio"], "0.1")
        self.assertEqual(sweep.parse_args(self.arguments + ["--phase=full"]).steps, 120000)

    def test_named_plane_and_custom_cohort(self):
        plane = self.root / "test plane.txt"
        plane.write_text("test plane contents")
        variants = self.root / "variants.json"
        variants.write_text(json.dumps([{"id": "selected", "flags": ["--block=3", "--steps=3"]}]))
        args = sweep.parse_args(self.arguments + [f"--variants-file={variants}",
                                                 f"--plane=pca={plane}"])
        actual = sweep.load_variants(args)
        self.assertEqual([v["id"] for v in actual], ["selected", "pca"])
        command = sweep.build_command(args, actual[0], self.root / "selected")
        self.assertEqual(sum(x.startswith("--steps=") for x in command), 1)
        self.assertIn("--steps=3", command)
        self.assertIn(f"--graft_plane={plane}", actual[1]["flags"])

    def test_variant_identifiers_and_reserved_paths_are_validated(self):
        path = self.root / "variants.json"
        invalid = [
            [{"id": "../outside", "flags": []}],
            [{"id": "repeat", "flags": []}] * 2,
            [{"id": "override", "flags": ["--output=/tmp/other"]}],
            [{"id": "bad", "flags": ["--block", "3"]}],
            [],
        ]
        for variants in invalid:
            with self.subTest(variants=variants):
                path.write_text(json.dumps(variants))
                args = sweep.parse_args(self.arguments + [f"--variants-file={path}"])
                with self.assertRaises(ValueError):
                    sweep.load_variants(args)

    def test_final_accuracy_uses_wrong_fraction_not_completion_fraction(self):
        parsed = sweep.parse_log("informational banner\n" + EVALUATIONS + FINAL)
        self.assertEqual(len(parsed["evaluations"]), 2)
        final = parsed["final"]
        self.assertEqual(final["correct"], 3125)
        self.assertAlmostEqual(final["accuracy"], 3125 / 10002)
        self.assertEqual(final["complete_facts"], 1)
        self.assertEqual(final["fact_count"], 1024)
        self.assertEqual(final["best_step"], 18000)
        self.assertEqual(final["updates"], 20000)
        self.assertEqual(parsed["evaluations"][1]["step"], 20000)

    def test_last_evaluation_does_not_substitute_for_missing_final(self):
        self.assertIsNone(sweep.parse_log(EVALUATIONS)["final"])

    def test_impossible_and_nonfinite_metrics_are_rejected(self):
        for contents in (
            FINAL.replace("6877/10002", "10003/10002"),
            FINAL.replace("6877/10002", "0/0"),
            FINAL.replace("seconds=20.41", "seconds=nan"),
            FINAL.replace("best_step=18000", "best_step=20001"),
            EVALUATIONS.replace("min_margin=-7.100000", "min_margin=-inf"),
        ):
            with self.subTest(contents=contents), self.assertRaises(ValueError):
                sweep.parse_log(contents)

    def test_dry_run_persists_every_command_without_launching_children(self):
        args = sweep.parse_args(self.arguments + ["--dry-run"])
        with mock.patch.object(sweep.subprocess, "Popen") as child:
            manifest = sweep.run_sweep(args)
        child.assert_not_called()
        self.assertEqual(manifest["status"], "planned")
        self.assertEqual(len(manifest["runs"]), 57)
        self.assertEqual(len((args.output / "results.tsv").read_text().splitlines()), 58)
        self.assertIn("do not establish performance", (args.output / "report.html").read_text())
        self.assertIn(str(self.binary), manifest["input_sha256"])
        self.assertIn(str(self.checkpoint / "compact_vocabulary.tsv"), manifest["input_sha256"])
        self.assertTrue((args.output / "pair_09_10/command.json").is_file())
        with self.assertRaisesRegex(ValueError, "must be new"):
            sweep.run_sweep(args)

    def test_concurrent_children_are_bounded_and_failures_do_not_stop_cohort(self):
        variants = self.root / "variants.json"
        names = ["good1", "bad_exit", "missing_final", "bad_frozen", "good2"]
        variants.write_text(json.dumps([{"id": name, "flags": []} for name in names]))
        args = sweep.parse_args(self.arguments + [f"--variants-file={variants}", "--jobs=2"])
        lock = threading.Lock()
        counts = {"active": 0, "peak": 0}

        class FakeProcess:
            def __init__(self, command, *, stdout, stderr, stdin):
                self.name = Path(next(x.split("=", 1)[1] for x in command if x.startswith("--output="))).parent.name
                stdout.write(EVALUATIONS)
                if self.name != "missing_final":
                    stdout.write(FINAL.replace("=true", "=false") if self.name == "bad_frozen" else FINAL)
                self.returncode = 7 if self.name == "bad_exit" else 0
                with lock:
                    counts["active"] += 1
                    counts["peak"] = max(counts["peak"], counts["active"])

            def wait(self, timeout):
                time.sleep(0.015)
                with lock:
                    counts["active"] -= 1
                return self.returncode

        with mock.patch.object(sweep.subprocess, "Popen", FakeProcess), redirect_stdout(io.StringIO()):
            manifest = sweep.run_sweep(args)
        self.assertEqual(counts["peak"], 2)
        self.assertEqual(counts["active"], 0)
        self.assertEqual(manifest["status"], "complete_with_failures")
        records = {r["id"]: r for r in manifest["runs"]}
        self.assertEqual([records[name]["status"] for name in names],
                         ["complete", "failed", "failed", "failed", "complete"])
        self.assertEqual(records["bad_exit"]["returncode"], 7)
        self.assertIn("no FINAL", records["missing_final"]["error"])
        self.assertEqual(len(json.loads((args.output / "good2/evaluations.json").read_text())), 2)
        persisted = json.loads((args.output / "manifest.json").read_text())
        self.assertEqual(persisted["status"], "complete_with_failures")
        self.assertTrue(all((args.output / name / "result.json").exists() for name in names))

    def test_worker_records_launch_failure(self):
        args = sweep.parse_args(self.arguments)
        (args.output / "missing").mkdir(parents=True)
        with mock.patch.object(sweep.subprocess, "Popen", side_effect=OSError("fixture launch error")):
            result = sweep.run_variant({"id": "missing", "command": ["missing"]}, args, threading.Event())
        self.assertEqual(result["status"], "failed")
        self.assertIn("fixture launch error", result["error"])

    def test_cancel_terminates_child_and_retains_metrics(self):
        args = sweep.parse_args(self.arguments)
        (args.output / "cancel").mkdir(parents=True)
        process = mock.Mock(returncode=-15)
        process.wait.side_effect = [subprocess.TimeoutExpired("fake", 0.25), -15]
        cancel = threading.Event()
        cancel.set()
        with mock.patch.object(sweep.subprocess, "Popen", return_value=process):
            result = sweep.run_variant({"id": "cancel", "command": ["fake"]}, args, cancel)
        process.terminate.assert_called_once()
        self.assertEqual(result["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
