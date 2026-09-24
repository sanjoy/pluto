#!/usr/bin/env python3
"""Deadline-bounded search for the smallest verified model in two depth tracks.

Both tracks use the identical corpus, vocabulary, training seed, and verification
contract. A four-or-more-block result also competes in the unrestricted track.
Every attempt starts from scratch; changing a step budget changes its cosine
schedule, so retries are distinct experiments, not checkpoint continuations.
"""

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import run_compact_size_sweep as compact
from run_depth_search import _now, _sha256, _write_summary


TRACKS = ("overall", "four_plus")
SOURCES = ("run_two_track_sweep.py", "run_compact_size_sweep.py",
           "run_depth_search.py", "compact_checkpoint.py", "verify_predictions.py")
# (optimizer steps, initial learning rate, wall-clock training cap). The last
# stages are deliberately long, to distinguish slow learning from short screens.
SCHEDULES = ((60000, 0.0006, 240), (120000, 0.0006, 420),
             (120000, 0.0012, 420), (120000, 0.0003, 420),
             (240000, 0.0006, 700), (360000, 0.0003, 1000))


@dataclass(frozen=True)
class Candidate:
    layers: int
    width: int
    feed_forward_width: int
    stage: int = 0

    @property
    def shape(self):
        return (self.layers, self.width, self.feed_forward_width)

    @property
    def parameters(self):
        return compact.parameter_count(*self.shape)

    @property
    def track(self):
        return "four_plus" if self.layers >= 4 else "overall"

    def specification(self):
        steps, rate, _ = self.schedule()
        return f"{self.layers}:{self.width}:{self.feed_forward_width}:{steps}:{rate}"

    def schedule(self):
        if self.stage or self.shape == (4, 13, 26):
            return SCHEDULES[self.stage]
        return (80000, 0.0006, 480) if self.layers >= 6 else (60000, 0.0006, 300)


def initial_candidates(track):
    """Prioritize promising widths before interleaving deeper/narrower shapes."""
    if track == "overall":
        opening = [(1, 13, 52), (2, 12, 48), (3, 11, 44), (1, 12, 48),
                   (2, 11, 44), (3, 10, 40), (1, 13, 26), (1, 11, 44)]
        depths = (3, 2, 1)
    elif track == "four_plus":
        opening = [(4, 12, 48), (4, 11, 44), (8, 10, 40), (4, 12, 24),
                   (6, 10, 40), (4, 11, 22), (8, 9, 36), (4, 13, 13),
                   (4, 13, 6)]
        depths = (4, 5, 6, 7, 8, 10, 12)
    else:
        raise ValueError(f"Unknown track: {track}")
    # Diagonal waves avoid spending the whole budget on one width or FF ratio.
    shapes = list(opening)
    widths = (13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2)
    for wave in range(len(widths) + len(depths) - 1):
        for depth_index, layers in enumerate(depths):
            index = wave - depth_index
            if 0 <= index < len(widths):
                for expansion in (4, 2, 1):
                    shapes.append((layers, widths[index], widths[index] * expansion))
    shapes.extend((layers, width, max(1, width // 2))
                  for width in widths for layers in depths)
    return [Candidate(*shape) for shape in dict.fromkeys(shapes)]


def winners(trials):
    """Only fully audited successes establish an upper bound for either track."""
    verified = [trial for trial in trials if trial["status"] == "verified"]
    return {track: min((trial for trial in verified
                        if track == "overall" or trial["layers"] >= 4),
                       key=lambda item: (item["parameters"], item["index"]),
                       default=None) for track in TRACKS}


class Planner:
    """Alternate tracks, with every third turn available for a near-miss retry."""

    def __init__(self):
        self.queues = {track: initial_candidates(track) for track in TRACKS}
        self.turns = dict.fromkeys(TRACKS, 0)
        self.next_track = 0

    def next(self, trials, *, prefer_retries=False):
        best = winners(trials)
        for _ in TRACKS:
            track = TRACKS[self.next_track]
            self.next_track = (self.next_track + 1) % len(TRACKS)
            self.turns[track] += 1
            ceiling = best[track]["parameters"] if best[track] else math.inf
            latest = {}
            for trial in trials:
                if trial["track"] == track and trial["status"] != "skipped":
                    latest[tuple(trial["shape"])] = trial
            retries = [trial for trial in latest.values()
                       if trial["status"] in ("budget_fail", "timeout")
                       and trial["stage"] + 1 < len(SCHEDULES)
                       and trial["parameters"] < ceiling]
            retries.sort(key=lambda item: (item.get("errors", 10003),
                                           item["parameters"], item["index"]))
            # Near misses get recurring attention even while new shapes remain.
            if retries and (prefer_retries or (
                    self.turns[track] % 3 == 0 and retries[0].get("errors", 10003) <= 200)):
                trial = retries[0]
                return Candidate(*trial["shape"], trial["stage"] + 1)
            if self.queues[track]:
                return self.queues[track].pop(0)
            if retries:
                trial = retries[0]
                return Candidate(*trial["shape"], trial["stage"] + 1)
        return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "corpus", "tokenizer", "run_dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--duration_seconds", type=float, default=10800)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration_seconds) or args.duration_seconds <= 31:
        parser.error("duration_seconds must be finite and greater than 31")
    if not -(2**31) <= args.seed < 2**31:
        parser.error("seed must fit int32")
    for name in ("binary", "corpus", "tokenizer", "run_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    repository = Path(__file__).resolve().parents[2]
    if args.run_dir == repository or repository in args.run_dir.parents:
        parser.error("run_dir must be outside the repository")
    return args


def snapshot(args):
    """Pin every child to one immutable executable, input pair, and driver copy."""
    inputs = args.run_dir / "inputs"
    inputs.mkdir()
    shutil.copy2(args.binary, inputs / "memorize_general_facts")
    shutil.copy2(args.corpus, inputs / "corpus.txt")
    shutil.copy2(args.tokenizer / "tokenizer.json", inputs / "tokenizer.json")
    scripts = inputs / "scripts"
    scripts.mkdir()
    for name in SOURCES:
        shutil.copy2(Path(__file__).with_name(name), scripts / name)
    return {str(path.relative_to(inputs)): _sha256(path)
            for path in inputs.rglob("*") if path.is_file()}


def is_training_divergence(code, child):
    """Recognize one numerical failure, never configuration or audit failures."""
    if (code != 1 or child.get("status") != "error"
            or child.get("error") != "Native training failed with exit 1"
            or len(child.get("trials", [])) != 1):
        return False
    trial = child["trials"][0]
    if (trial.get("status") != "error" or trial.get("phase") != "training"
            or trial.get("last_returncode") != 1 or not trial.get("directory")):
        return False
    error_lines = (Path(trial["directory"]) / "training.stderr.log").read_text().splitlines()
    return bool(error_lines) and error_lines[0] == (
        "DATA_LOSS: nonfinite logits/loss in full-corpus evaluation")


def run_sweep(args, *, run_trial=None, clock=time.time, planner=None):
    if args.run_dir.exists() or args.run_dir.is_symlink():
        raise ValueError(f"Refusing existing run directory: {args.run_dir}")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        raise ValueError("binary must be an executable file")
    started = clock()
    deadline = started + args.duration_seconds
    args.run_dir.mkdir(parents=True, exist_ok=False)
    summary = {"status": "running", "driver_pid": os.getpid(), "started_utc": _now(),
               "deadline_unix": deadline, "configuration": {
                   **{key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
                   "context_length": 27, "vocabulary": 4475, "batch_size": 32,
                   "heads": 1, "eval_every": 256, "gradient_clipping": False},
               "scope": "Smallest verified among tested configurations, not a proof of global minimality.",
               "trials": [], "best": {}, "active_trial": None}

    def persist():
        summary["best"] = winners(summary["trials"])
        summary["elapsed_seconds"] = max(0, clock() - started)
        _write_summary(args.run_dir / "summary.json", summary)
        path = args.run_dir / ".summary.tsv.tmp"
        keys = ("index", "track", "layers", "width", "feed_forward_width", "parameters",
                "stage", "steps", "learning_rate", "status", "step", "errors", "checkpoint")
        with path.open("w") as output:
            output.write("\t".join(keys) + "\n")
            for trial in summary["trials"]:
                output.write("\t".join(str(trial.get(key, "")) for key in keys) + "\n")
        path.replace(args.run_dir / "summary.tsv")

    persist()
    try:
        summary["hashes"] = snapshot(args)
        summary["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "")
        inputs = args.run_dir / "inputs"
        # The child copies __file__ and its siblings for every trial. Import the
        # pinned module so later source edits cannot change those child snapshots.
        spec = importlib.util.spec_from_file_location(
            "pinned_compact_sweep", inputs / "scripts/run_compact_size_sweep.py")
        pinned = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pinned)
        planner = planner or Planner()
        candidate = Candidate(4, 13, 26)

        def next_candidate():
            return planner.next(summary["trials"], prefer_retries=(
                clock() - started >= args.duration_seconds * 2 / 3))

        while candidate is not None:
            if clock() >= deadline - 31:
                summary["status"] = "deadline"
                break
            steps, rate, cap = candidate.schedule()
            index = len(summary["trials"])
            record = {**asdict(candidate), "shape": list(candidate.shape), "index": index,
                      "track": candidate.track, "parameters": candidate.parameters,
                      "steps": steps, "learning_rate": rate, "max_training_seconds": cap,
                      "status": "running", "started_utc": _now()}
            summary["trials"].append(record)
            best = winners(summary["trials"])[candidate.track]
            if best and candidate.parameters >= best["parameters"]:
                record.update(status="skipped", reason="not smaller than verified track winner")
                persist()
                candidate = next_candidate()
                continue
            for name, digest in summary["hashes"].items():
                if _sha256(inputs / name) != digest:
                    raise ValueError(f"Pinned input changed: {name}")
            trial_dir = args.run_dir / f"attempt_{index:03d}_L{candidate.layers}_W{candidate.width}_FF{candidate.feed_forward_width}_stage{candidate.stage}"
            options = [f"--binary={inputs / 'memorize_general_facts'}",
                       f"--corpus={inputs / 'corpus.txt'}", f"--tokenizer={inputs}",
                       f"--run_dir={trial_dir}", f"--deadline_unix={deadline}",
                       f"--candidates={candidate.specification()}", f"--seed={args.seed}",
                       "--batch_size=32", "--context_length=27", "--verification_reserve=30",
                       "--eval_every=256", f"--max_training_seconds={cap}"]
            # The parser's repository guard is relative to its source path;
            # validate with the original module, execute with the pinned copy.
            child_args = compact.parse_args(options)
            record.update(directory=str(trial_dir), arguments=options,
                          child_summary=str(trial_dir / "summary.json"))
            summary["active_trial"] = index
            persist()
            print(f"[{_now()}] attempt={index} track={candidate.track} "
                  f"shape={candidate.shape} parameters={candidate.parameters} stage={candidate.stage}", flush=True)

            def bounded_execute(command, stdout_path, stderr_path, *, deadline, **kwargs):
                if "--mode=train_model" in command:
                    # Native training has its own cap; enforce an external grace
                    # bound as well in case it gets stuck between time checks.
                    deadline = min(deadline, clock() + cap + 15)
                return pinned.execute(command, stdout_path, stderr_path,
                                      deadline=deadline, **kwargs)

            code = (run_trial(child_args) if run_trial else
                    pinned.run_search(child_args, run_process=bounded_execute))
            child = json.loads((trial_dir / "summary.json").read_text())
            child_trials = child["trials"]
            if len(child_trials) > 1:
                raise ValueError("Single-candidate child emitted multiple trials")
            if child_trials:
                result = child_trials[0]
                if any(result.get(key) != record[key] for key in
                       ("layers", "width", "feed_forward_width", "parameters", "steps", "learning_rate")):
                    raise ValueError("Child trial configuration changed")
                record.update({key: result[key] for key in
                               ("status", "step", "errors", "checkpoint", "training_result", "commands")
                               if key in result})
            elif child["status"] != "deadline":
                raise ValueError("Child emitted no trial before deadline")
            else:
                record["status"] = "timeout"
            record["finished_utc"] = _now()
            if is_training_divergence(code, child):
                record.update(status="diverged", reason="nonfinite training logits/loss")
                summary["active_trial"] = None
                persist()
                candidate = next_candidate()
                continue
            if code != 0 or child["status"] not in ("completed", "deadline"):
                raise ValueError(f"Child failed: {child.get('error', child['status'])}; inspect {trial_dir}")
            if record["status"] not in ("verified", "budget_fail", "timeout"):
                raise ValueError(f"Unexpected child status: {record['status']}")
            if record["status"] == "verified" and (record.get("errors") != 0 or not record.get("checkpoint")):
                raise ValueError("Verified child omitted zero errors or checkpoint evidence")
            summary["active_trial"] = None
            persist()
            candidate = next_candidate()
        else:
            summary["status"] = "exhausted"
        return 0
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        return 130
    except (ImportError, OSError, ValueError, KeyError, TypeError, SystemExit) as error:
        summary.update(status="error", error=str(error))
        if summary["active_trial"] is not None:
            summary["trials"][summary["active_trial"]]["status"] = "error"
        print(f"Two-track sweep stopped: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        summary.update(driver_pid=None, finished_utc=_now())
        persist()


def main():
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return run_sweep(parse_args())
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
