#!/usr/bin/env python3
"""Fixture-only checks for reporting recipe sweeps; no binaries are launched."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import analyze_readout_recipe_sweep as report


class ReadoutRecipeReportTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def fixture(self, relative="screen/trial", *, steps=20000, updates=None,
                phase="screen", flags=(), **changes):
        updates = steps if updates is None else updates
        directory = self.root / relative
        directory.mkdir(parents=True, exist_ok=True)
        record = {
            "status": "complete", "returncode": 0, "id": directory.name,
            "phase": phase, "frozen_weights_unchanged": True,
            "command": ["never-executed", f"--steps={steps}", "--model_width=10",
                        "--readout_width=150", "--train_final_norm=true", *flags],
            "scored": 10002, "wrong": 6500, "correct": 3502,
            "accuracy": 3502 / 10002, "complete_facts": 1,
            "fact_count": 1024, "best_step": max(0, updates - 1000),
            "updates": updates, "training_seconds": 120.5, "wall_seconds": 124.0,
        }
        record.update(changes)
        (directory / "result.json").write_text(json.dumps(record))
        evaluations = [{"step": 0, "wrong": 9999, "scored": 10002},
                       {"step": updates, "wrong": 6500, "scored": 10002}]
        (directory / "evaluations.json").write_text(json.dumps(evaluations))
        return directory

    def test_computes_exact_parameter_count_and_preserves_configuration(self):
        self.fixture(flags=("--random_init=19", "--seed=7", "--input_preprocess=sin",
                            "--preprocess_frequency=2.5", "--learning_rate=0.0003"))
        summary = report.collect(self.root)
        self.assertEqual(summary["warnings"], [])
        record = summary["records"][0]
        self.assertEqual(record["trainable_parameters"], 3200)
        self.assertEqual(record["readout_width"], 150)
        self.assertEqual(record["flags"]["random_init"], "19")
        self.assertEqual(record["flags"]["seed"], "7")
        self.assertEqual(record["flags"]["input_preprocess"], "sin")
        html = report.report_html(summary)
        self.assertIn("preprocess_frequency=2.5", html)
        self.assertIn("learning_rate=0.0003", html)
        self.assertIn("35.013%", html)
        self.assertIn("<svg", html)

    def test_screens_and_confirmations_and_step_budgets_are_separate(self):
        self.fixture("experiment/a", steps=20000)
        self.fixture("experiment/b", steps=60000)
        self.fixture("confirmation/c", steps=300000, phase="full")
        html = report.report_html(report.collect(self.root))
        self.assertEqual(html.count("<section>"), 3)
        self.assertIn("screen, 20,000 planned updates", html)
        self.assertIn("screen, 60,000 planned updates", html)
        self.assertIn("full, 300,000 planned updates", html)

    def test_excludes_replays_incomplete_runs_and_zero_step_evaluations(self):
        self.fixture()
        self.fixture("screen/trial/verify_saved")
        self.fixture("saved_replay/a")
        self.fixture("screen/failed", status="failed")
        self.fixture("screen/pending", status="running")
        self.fixture("screen/check", steps=0)
        summary = report.collect(self.root)
        self.assertEqual(len(summary["records"]), 1)
        self.assertEqual(summary["skipped"], {
            "not complete": 2, "saved-checkpoint replay": 2,
            "zero-update evaluation": 1,
        })

    def test_marks_shortened_budget_instead_of_implying_full_run(self):
        self.fixture(steps=300000, updates=7000)
        summary = report.collect(self.root)
        self.assertFalse(summary["records"][0]["completed_budget"])
        html = report.report_html(summary)
        self.assertIn("7,000/300,000", html)
        self.assertIn("<b>shortened</b>", html)

    def test_frozen_final_norm_and_other_widths_have_correct_parameter_count(self):
        directory = self.fixture()
        path = directory / "result.json"
        record = json.loads(path.read_text())
        record["command"] = ["not-run", "--steps=20000", "--readout_width=160",
                             "--train_final_norm=false"]
        path.write_text(json.dumps(record))
        normalized = report.collect(self.root)["records"][0]
        self.assertEqual(normalized["readout_width"], 160)
        self.assertEqual(normalized["trainable_parameters"], 3390)

    def test_malformed_counts_and_nonfinite_times_are_not_silently_reported(self):
        for index, changes in enumerate((
            {"wrong": 10003}, {"accuracy": .99}, {"correct": 5},
            {"best_step": 40000}, {"updates": 30000}, {"training_seconds": float("nan")},
            {"complete_facts": 2000}, {"frozen_weights_unchanged": False},
        )):
            self.fixture(f"screen/bad_{index}", **changes)
        summary = report.collect(self.root)
        self.assertEqual(summary["records"], [])
        self.assertEqual(len(summary["warnings"]), 8)
        self.assertIn("Excluded invalid records", report.report_html(summary))

    def test_missing_command_falls_back_to_saved_command_file(self):
        directory = self.fixture()
        path = directory / "result.json"
        record = json.loads(path.read_text())
        command = record.pop("command")
        path.write_text(json.dumps(record))
        (directory / "command.json").write_text(json.dumps({"command": command}))
        self.assertEqual(len(report.collect(self.root)["records"]), 1)

    def test_command_parser_accepts_spaced_and_boolean_flags_but_not_duplicates(self):
        flags = report.command_flags(["binary", "--steps", "20", "--train_final_norm",
                                      "--random_init=-1"])
        self.assertEqual(flags, {"steps": "20", "train_final_norm": "true", "random_init": "-1"})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            report.command_flags(["binary", "--seed=1", "--seed=2"])

    def test_html_escapes_labels_and_commands_and_is_self_contained(self):
        self.fixture(id="<script>alert('bad')</script>", flags=("--input_preprocess=<tag>",))
        html = report.report_html(report.collect(self.root))
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&lt;tag&gt;", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("<link ", html)

    def test_cli_writes_both_reports_without_running_a_training_command(self):
        self.fixture()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(report.main([str(self.root)]), 0)
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(summary["records"][0]["trainable_parameters"], 3200)
        self.assertTrue((self.root / "summary.html").is_file())
        self.assertFalse((self.root / "summary.html.tmp").exists())

    def test_empty_root_is_reportable_while_jobs_are_running(self):
        summary = report.collect(self.root)
        self.assertEqual(summary["records"], [])
        self.assertIn("No completed training runs yet.", report.report_html(summary))

    def test_bad_evaluation_curve_is_excluded_instead_of_drawing_invalid_data(self):
        directory = self.fixture()
        (directory / "evaluations.json").write_text(json.dumps([
            {"step": 5, "scored": 10002, "wrong": 10},
            {"step": 2, "scored": 10002, "wrong": 2},
        ]))
        summary = report.collect(self.root)
        self.assertEqual(summary["records"], [])
        self.assertIn("evaluation series", summary["warnings"][0])

    def test_selected_checkpoint_ce_comes_from_final_not_last_evaluation(self):
        directory = self.fixture()
        (directory / "training.log").write_text(
            "step=20000 cross_entropy=1.23\n"
            "FINAL best_step=19000 wrong=6500/10002 cross_entropy=2.345678\n")
        summary = report.collect(self.root)
        self.assertAlmostEqual(summary["records"][0]["cross_entropy"], 2.345678)
        self.assertIn("2.34568", report.report_html(summary))
        self.assertIn("Selected checkpoint CE", report.report_html(summary))

    def test_explicit_ce_is_preferred_but_missing_ce_is_not_invented(self):
        directory = self.fixture("screen/explicit", cross_entropy=2.5)
        (directory / "training.log").write_text("FINAL cross_entropy=4.0\n")
        self.fixture("screen/old")
        records = {r["id"]: r for r in report.collect(self.root)["records"]}
        self.assertEqual(records["explicit"]["cross_entropy"], 2.5)
        self.assertNotIn("cross_entropy", records["old"])

    def test_collision_audit_metadata_and_conflict_warning(self):
        directory = self.fixture()
        (directory / "training.log").write_text(
            "preprocessed_unique_vectors=10000 scored_vectors=10002 conflicting_vectors=1\n")
        summary = report.collect(self.root)
        collisions = summary["records"][0]["collisions"]
        self.assertEqual(collisions, {"preprocessed_unique_vectors": 10000,
                                     "scored_vectors": 10002, "conflicting_vectors": 1,
                                     "duplicate_vector_rows": 2})
        self.assertEqual(summary["warnings"], [])
        self.assertEqual(len(summary["diagnostic_warnings"]), 1)
        html = report.report_html(summary)
        self.assertIn("1 conflicts", html)
        self.assertIn("Preprocessing collision diagnostics", html)

    def test_unique_vectors_no_warning_and_new_flag_names_are_visible(self):
        directory = self.fixture(flags=("--input_init_std=0.02", "--output_init_std=0.03",
                                        "--scale_output_init=true", "--fresh_final_norm=true",
                                        "--preprocessing=dft", "--preprocessing_scale=2",
                                        "--preprocessing_seed=42", "--random_init=7", "--seed=19"))
        (directory / "training.log").write_text(
            "preprocessed_unique_vectors=10002 scored_vectors=10002 conflicting_vectors=0\n")
        summary = report.collect(self.root)
        self.assertEqual(summary["diagnostic_warnings"], [])
        html = report.report_html(summary)
        for text in ("input_init_std=0.02", "output_init_std=0.03", "scale_output_init=true",
                     "fresh_final_norm=true", "preprocessing=dft", "preprocessing_scale=2",
                     "preprocessing_seed=42", "random_init=7", "seed=19"):
            self.assertIn(text, html)

    def test_invalid_ce_mismatched_final_and_impossible_collisions_are_excluded(self):
        logs = ("FINAL cross_entropy=nan\n", "FINAL cross_entropy=-1\n",
                "FINAL best_step=2 cross_entropy=2.0\n",
                "FINAL wrong=1/10002 cross_entropy=2.0\n",
                "preprocessed_unique_vectors=10003 scored_vectors=10002 conflicting_vectors=0\n",
                "preprocessed_unique_vectors=10002 scored_vectors=10002 conflicting_vectors=10003\n")
        for index, log in enumerate(logs):
            directory = self.fixture(f"screen/invalid_{index}")
            (directory / "training.log").write_text(log)
        summary = report.collect(self.root)
        self.assertEqual(summary["records"], [])
        self.assertEqual(len(summary["warnings"]), len(logs))

    def test_ce_curves_read_explicit_ce_not_margin_loss(self):
        directory = self.fixture("screen/new")
        (directory / "training.log").write_text(
            "step=0 margin_loss=150 cross_entropy=8.2\n"
            "step=20000 margin_loss=40 cross_entropy=3.4\n"
            "FINAL best_step=19000 wrong=6500/10002 cross_entropy=3.3\n")
        old = self.fixture("screen/old")
        (old / "training.log").write_text(
            "step=0 margin_loss=350\nstep=20000 margin_loss=50\n")
        summary = report.collect(self.root)
        records = {r["id"]: r for r in summary["records"]}
        self.assertEqual(records["new"]["cross_entropy_evaluations"], [
            {"step": 0, "cross_entropy": 8.2},
            {"step": 20000, "cross_entropy": 3.4},
        ])
        self.assertEqual(records["new"]["cross_entropy"], 3.3)
        self.assertNotIn("cross_entropy_evaluations", records["old"])
        html = report.report_html(summary)
        self.assertIn("nats per scored token (lower is better)", html)
        self.assertEqual(html.count("<svg"), 2)
        self.assertNotIn("margin_loss=150", html)

    def test_fact_and_token_accuracy_are_separately_labeled_with_denominators(self):
        self.fixture(complete_facts=256)
        html = report.report_html(report.collect(self.root))
        self.assertIn("Teacher-forced token top-1", html)
        self.assertIn("Greedy exact facts", html)
        self.assertIn("25.00%<br>256/1024", html)
        self.assertIn("35.013%</strong><br>3,502/10,002", html)
        self.assertIn("3,200 trainable parameters", html)

    def test_invalid_ce_evaluation_curve_is_excluded(self):
        for index, log in enumerate((
            "step=40000 cross_entropy=3\n",
            "step=0 cross_entropy=nan\n",
            "step=5 cross_entropy=3\nstep=3 cross_entropy=4\n",
        )):
            directory = self.fixture(f"screen/bad_{index}")
            (directory / "training.log").write_text(log)
        summary = report.collect(self.root)
        self.assertEqual(summary["records"], [])
        self.assertEqual(len(summary["warnings"]), 3)


if __name__ == "__main__":
    unittest.main()
