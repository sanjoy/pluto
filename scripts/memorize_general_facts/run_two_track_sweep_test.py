#!/usr/bin/env python3
"""CPU-only controller tests; the native trial/audit pipeline is tested separately."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import run_two_track_sweep as sweep


def record(candidate, index=0, status="budget_fail", errors=50):
    steps, rate, cap = candidate.schedule()
    return {"shape": list(candidate.shape), "layers": candidate.layers,
            "width": candidate.width, "feed_forward_width": candidate.feed_forward_width,
            "parameters": candidate.parameters, "track": candidate.track,
            "stage": candidate.stage, "index": index, "status": status,
            "errors": errors, "steps": steps, "learning_rate": rate}


class FixedPlanner:
    def __init__(self, candidates):
        self.candidates = iter(candidates)
        self.preferences = []

    def next(self, trials, *, prefer_retries=False):
        self.preferences.append(prefer_retries)
        return next(self.candidates, None)


class TwoTrackSweepTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.binary = self.root / "native"
        self.binary.write_text("never executed\n")
        self.binary.chmod(0o700)
        self.corpus = self.root / "corpus.txt"
        self.corpus.write_text("pinned corpus\n")
        self.tokenizer = self.root / "tokenizer"
        self.tokenizer.mkdir()
        (self.tokenizer / "tokenizer.json").write_text("{}\n")
        self.args = sweep.parse_args([
            f"--binary={self.binary}", f"--corpus={self.corpus}",
            f"--tokenizer={self.tokenizer}", f"--run_dir={self.root / 'run'}"])
        self.time = 1000
        self.calls = []

    def fake_child(self, args, *, status="budget_fail", child_status="completed", code=0):
        self.calls.append(args)
        self.time += 1
        args.run_dir.mkdir()
        trial = {**args.candidates[0], "parameters": sweep.compact.parameter_count(
            args.candidates[0]["layers"], args.candidates[0]["width"],
            args.candidates[0]["feed_forward_width"]), "status": status,
            "step": 100, "errors": 0 if status == "verified" else 7,
            "checkpoint": str(args.run_dir / "checkpoint"), "commands": [["fake-native"]]}
        (args.run_dir / "summary.json").write_text(json.dumps({
            "status": child_status, "trials": [trial], "error": "fake failure"}))
        return code

    def run_controller(self, candidates=(), run_trial=None, **kwargs):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = sweep.run_sweep(self.args, run_trial=run_trial or self.fake_child,
                                   clock=lambda: self.time,
                                   planner=kwargs.pop("planner", FixedPlanner(candidates)), **kwargs)
        return code, json.loads((self.args.run_dir / "summary.json").read_text())

    def test_cli_defaults_and_rejects_invalid_limits(self):
        self.assertEqual(self.args.duration_seconds, 10800)
        self.assertEqual(self.args.seed, 1337)
        base = [f"--{key}={getattr(self.args, key)}"
                for key in ("binary", "corpus", "tokenizer", "run_dir")]
        for option in ("--duration_seconds=nan", "--duration_seconds=inf",
                       "--duration_seconds=31", "--duration_seconds=-1",
                       "--seed=2147483648", f"--run_dir={Path(sweep.__file__).parent}"):
            with self.subTest(option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    sweep.parse_args(base + [option])

    def test_candidate_accounting_and_depth_aware_schedules(self):
        baseline = sweep.Candidate(4, 13, 26)
        self.assertEqual(baseline.parameters, 64532)
        self.assertEqual(baseline.schedule(), (60000, 0.0006, 240))
        self.assertEqual(sweep.Candidate(8, 10, 40).schedule(), (80000, 0.0006, 480))
        self.assertEqual(sweep.Candidate(2, 12, 48).schedule(), (60000, 0.0006, 300))
        self.assertEqual(sweep.Candidate(8, 10, 40, 1).schedule(), (120000, 0.0006, 420))

    def test_initial_candidates_are_unique_and_include_ratios(self):
        for track in sweep.TRACKS:
            candidates = sweep.initial_candidates(track)
            self.assertEqual(len({item.shape for item in candidates}), len(candidates))
            self.assertTrue(all(item.track == track for item in candidates))
            self.assertTrue(any(item.feed_forward_width < item.width for item in candidates))
            self.assertTrue(any(item.feed_forward_width == 4 * item.width for item in candidates[:3]))

    def test_planner_alternates_tracks_without_inferring_failure_monotonicity(self):
        planner = sweep.Planner()
        trials = []
        choices = []
        for index in range(8):
            candidate = planner.next(trials)
            choices.append(candidate)
            trials.append(record(candidate, index, errors=500))
        self.assertEqual([item.track for item in choices], list(sweep.TRACKS) * 4)
        self.assertEqual(choices[0].shape, (1, 13, 52))
        self.assertEqual(choices[1].shape, (4, 12, 48))
        self.assertTrue(any(item.width < 12 for item in choices))

    def test_near_miss_retry_every_third_track_turn(self):
        planner = sweep.Planner()
        trials = []
        choices = []
        for index in range(6):
            candidate = planner.next(trials)
            choices.append(candidate)
            trials.append(record(candidate, index, errors=3 + index))
        self.assertEqual(choices[4], sweep.Candidate(1, 13, 52, 1))
        self.assertEqual(choices[5], sweep.Candidate(4, 12, 48, 1))

    def test_final_third_prefers_retries_even_without_tiny_error_count(self):
        planner = sweep.Planner()
        trial = record(sweep.Candidate(2, 10, 40), errors=1000)
        self.assertEqual(planner.next([trial], prefer_retries=True),
                         sweep.Candidate(2, 10, 40, 1))

    def test_exhausted_queue_promotes_remaining_failures_then_finishes(self):
        planner = sweep.Planner()
        planner.queues = {track: [] for track in sweep.TRACKS}
        trial = record(sweep.Candidate(4, 10, 40), errors=1000)
        self.assertEqual(planner.next([trial]), sweep.Candidate(4, 10, 40, 1))
        trial["stage"] = len(sweep.SCHEDULES) - 1
        self.assertIsNone(planner.next([trial]))

    def test_winners_require_verification_and_deep_success_qualifies_for_both(self):
        deep = record(sweep.Candidate(4, 10, 40), status="verified", errors=0)
        failed = record(sweep.Candidate(1, 2, 2), 1, status="timeout")
        best = sweep.winners([deep, failed])
        self.assertEqual(best, {"overall": deep, "four_plus": deep})
        shallow = record(sweep.Candidate(1, 8, 16), 2, status="verified", errors=0)
        best = sweep.winners([deep, shallow, failed])
        self.assertEqual(best, {"overall": shallow, "four_plus": deep})

    def test_snapshots_pin_inputs_commands_and_deadline(self):
        code, summary = self.run_controller([sweep.Candidate(1, 13, 52)])
        self.assertEqual(code, 0)
        self.assertEqual(summary["status"], "exhausted")
        self.assertEqual(len(summary["trials"]), 2)
        self.assertEqual(summary["deadline_unix"], 11800)
        self.assertEqual(len(summary["hashes"]), len(sweep.SOURCES) + 3)
        for args in self.calls:
            self.assertEqual(args.deadline_unix, 11800)
            self.assertEqual(args.context_length, 27)
            self.assertEqual(args.batch_size, 32)
            self.assertEqual(args.binary.parent, self.args.run_dir / "inputs")
            self.assertEqual(args.tokenizer, args.binary.parent)
            self.assertEqual(args.corpus.read_text(), "pinned corpus\n")
        self.assertEqual(summary["best"], {track: None for track in sweep.TRACKS})
        self.assertEqual(len((self.args.run_dir / "summary.tsv").read_text().splitlines()), 3)

    def test_larger_shapes_are_skipped_not_failed(self):
        def verified(args):
            return self.fake_child(args, status="verified")
        code, summary = self.run_controller(
            [sweep.Candidate(8, 13, 52), sweep.Candidate(1, 16, 64),
             sweep.Candidate(4, 12, 24)], run_trial=verified)
        self.assertEqual(code, 0)
        self.assertEqual([trial["status"] for trial in summary["trials"]],
                         ["verified", "skipped", "skipped", "verified"])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(summary["best"]["overall"]["width"], 12)
        self.assertEqual(summary["best"]["four_plus"]["width"], 12)

    def test_timeout_continues_other_candidates_before_shared_deadline(self):
        def timed_out(args):
            return self.fake_child(args, status="timeout", child_status="deadline")
        code, summary = self.run_controller([sweep.Candidate(1, 12, 48)], run_trial=timed_out)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(all(trial["status"] == "timeout" for trial in summary["trials"]))
        self.assertIsNone(summary["best"]["overall"])

    def test_deadline_stops_new_launches_and_final_third_is_visible(self):
        planner = FixedPlanner([sweep.Candidate(1, 12, 48)] * 2)
        self.args.duration_seconds = 100

        def expensive(args):
            result = self.fake_child(args)
            self.time += 68
            return result

        code, summary = self.run_controller(run_trial=expensive, planner=planner)
        self.assertEqual(code, 0)
        self.assertEqual(summary["status"], "deadline")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(planner.preferences, [True])

    def test_fatal_child_stops_and_is_not_a_success(self):
        def broken(args):
            return self.fake_child(args, status="error", child_status="error", code=1)
        code, summary = self.run_controller([sweep.Candidate(1, 12, 48)], run_trial=broken)
        self.assertEqual(code, 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(summary["status"], "error")
        self.assertIn("fake failure", summary["error"])
        self.assertIsNone(summary["best"]["overall"])

    def test_false_verified_claim_does_not_update_winner(self):
        def malformed(args):
            result = self.fake_child(args, status="verified")
            path = args.run_dir / "summary.json"
            child = json.loads(path.read_text())
            child["trials"][0]["errors"] = 1
            path.write_text(json.dumps(child))
            return result
        code, summary = self.run_controller(run_trial=malformed)
        self.assertEqual(code, 1)
        self.assertEqual(summary["trials"][0]["status"], "error")
        self.assertIsNone(summary["best"]["overall"])

    def test_pinned_input_modification_is_fatal(self):
        def changed(args):
            result = self.fake_child(args)
            args.corpus.write_text("changed\n")
            return result
        code, summary = self.run_controller([sweep.Candidate(1, 12, 48)], run_trial=changed)
        self.assertEqual(code, 1)
        self.assertEqual(len(self.calls), 1)
        self.assertIn("Pinned input changed: corpus.txt", summary["error"])

    def test_training_nonfinite_is_a_failed_candidate_not_a_fatal_sweep(self):
        def diverged(args):
            result = self.fake_child(args, status="error", child_status="error", code=1)
            path = args.run_dir / "summary.json"
            child = json.loads(path.read_text())
            child["error"] = "Native training failed with exit 1"
            child["trials"][0].update(phase="training", last_returncode=1,
                                      directory=str(args.run_dir))
            path.write_text(json.dumps(child))
            (args.run_dir / "training.stderr.log").write_text(
                "DATA_LOSS: nonfinite logits/loss in full-corpus evaluation\n"
                "=== Source Location Trace: ===\n")
            return result
        code, summary = self.run_controller([sweep.Candidate(1, 12, 48)], run_trial=diverged)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(all(trial["status"] == "diverged" for trial in summary["trials"]))
        self.assertIsNone(summary["best"]["overall"])

    def test_divergence_classification_rejects_other_errors_and_phases(self):
        path = self.root / "training.stderr.log"
        child = {"status": "error", "error": "Native training failed with exit 1",
                 "trials": [{"status": "error", "phase": "training",
                             "last_returncode": 1, "directory": str(self.root)}]}
        path.write_text("DATA_LOSS: different failure\n")
        self.assertFalse(sweep.is_training_divergence(1, child))
        path.write_text("DATA_LOSS: nonfinite logits/loss in full-corpus evaluation\n")
        self.assertTrue(sweep.is_training_divergence(1, child))
        child["trials"][0]["phase"] = "verification"
        self.assertFalse(sweep.is_training_divergence(1, child))
        child["trials"][0]["phase"] = "training"
        self.assertFalse(sweep.is_training_divergence(2, child))

    def test_refuses_existing_run_directory_without_overwriting(self):
        self.args.run_dir.mkdir()
        sentinel = self.args.run_dir / "preserve"
        sentinel.write_text("important")
        with self.assertRaisesRegex(ValueError, "Refusing existing"):
            sweep.run_sweep(self.args, run_trial=self.fake_child)
        self.assertEqual(sentinel.read_text(), "important")


if __name__ == "__main__":
    unittest.main()
