#!/usr/bin/env python3
"""Measure width/depth trade-offs for exact five-token-prompt memorization.

Depths are visited in increasing order, widths in decreasing order. At a given
depth, a verified success lets us try the next narrower width; the first bounded
failure moves the search to the next depth. Widths already matched or beaten by
a shallower verified model are skipped because they cannot improve the measured
depth/width frontier. Skipped configurations are NOT declared failures: training
need not behave monotonically with width, and this is a bounded empirical search,
not a proof that all smaller networks are inadequate.
This historical search protocol explicitly keeps gradient clipping at norm 1;
the native binary's default for new standalone runs is unclipped training.

Both successful and budget-exhausted trials are reloaded in a fresh native
process and independently audited by retokenizing the corpus in Python. An
execution error, missing artifact, or inconsistent audit stops the whole search.
Training never resumes, retries, changes its schedule, or starts from another
trial's weights. Native output streams to the caller without a process timeout;
only the native training budget limits each trial. Exit 0 means the requested
search finished (even if some/all trials failed to memorize), and exit 1 means
an execution or evidence-validation error.
"""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from run_depth_search import _now, _read_result, _require_fields, _sha256, _write_summary


SCRIPT_DIRECTORY = Path(__file__).resolve().parent


def _integer_list(value):
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("list must be nonempty and contain no duplicates")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", type=Path,
        default=Path("bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts"),
    )
    parser.add_argument("--corpus", type=Path, default=Path("testdata/general_facts_dataset.txt"))
    parser.add_argument("--tokenizer", type=Path, required=True, help="GPT-2 tokenizer directory")
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--depths", type=_integer_list, default=list(range(1, 9)),
        help="Explicit positive int32 depths; the default search remains 1 through 8",
    )
    parser.add_argument("--widths", type=_integer_list, default=[256, 128, 64, 32, 16])
    parser.add_argument(
        "--attention_heads", type=int, default=0,
        help="Fixed head count dividing every width; 0 preserves the gcd(width, 64) head-dimension policy",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=128)
    parser.add_argument("--checkpoint_every", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--training_seconds", type=float, default=10800)
    args = parser.parse_args(argv)
    if any(not 0 < value < 2**31 for value in args.depths):
        parser.error("depths must be positive int32 values")
    if any(value <= 0 or value * 4 >= 2**31 for value in args.widths):
        parser.error("widths must be positive with 4 * width < INT_MAX")
    try:
        for width in args.widths:
            model_dimensions(1, width, args.attention_heads)
    except ValueError as error:
        parser.error(str(error))
    # Accept either ordering at the command line but record the actual traversal.
    args.depths.sort()
    args.widths.sort(reverse=True)
    if any(not 0 < getattr(args, name) < 2**31
           for name in ("batch_size", "eval_every", "checkpoint_every")):
        parser.error("batch_size, eval_every, and checkpoint_every must be positive int32 values")
    if not 0 <= args.steps < 2**31 - 1 or not 0 <= args.warmup_steps < 2**31:
        parser.error("steps and warmup_steps must be nonnegative, with steps < INT_MAX")
    if args.batch_size * 1024 >= 2**31 or not -(2**31) <= args.seed < 2**31:
        parser.error("batch token count and seed must fit native int32 flags")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning_rate must be finite and positive")
    if not math.isfinite(args.training_seconds) or args.training_seconds < 0:
        parser.error("training_seconds must be finite and nonnegative")
    for name in ("binary", "corpus", "tokenizer", "checkpoint_dir", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if (args.output_dir == args.checkpoint_dir
        or args.output_dir in args.checkpoint_dir.parents
        or args.checkpoint_dir in args.output_dir.parents):
        parser.error("artifact and checkpoint directories must be separate, non-nested paths")
    return args


def model_dimensions(layers, width, attention_heads=0):
    """Resolve a fixed head count or the original gcd head-dimension policy.

    The masked backend supports compact positive widths, including 8 and 24.
    With no override, width 24 uses three eight-wide heads; an odd width uses
    one-wide heads. A positive override holds head count fixed across widths.
    Head partitions are explicit choices, not inferred from checkpoint sizes.
    The parameter count includes the fixed GPT-2 vocabulary's padded rows and
    learned absolute positions, and counts tied embedding/LM-head storage once.
    """
    if type(layers) is not int or not 0 < layers < 2**31:
        raise ValueError("layers must be a positive int32 value")
    if type(attention_heads) is not int or not 0 <= attention_heads < 2**31:
        raise ValueError("attention_heads must be a nonnegative int32 value")
    if attention_heads and width % attention_heads:
        raise ValueError(f"attention_heads={attention_heads} must divide every requested width; got {width}")
    heads = attention_heads or width // math.gcd(width, 64)
    return {
        "layers": layers,
        "width": width,
        "heads": heads,
        "head_dim": width // heads,
        "feed_forward_width": 4 * width,
        "parameters": 51298 * width + layers * (12 * width * width + 13 * width),
    }


def measured_frontier(trials):
    """Return nondominated verified successes in (depth, width), not all models."""
    successes = [trial for trial in trials if trial.get("status") == "verified_success"]
    frontier = []
    for trial in successes:
        if any(other["layers"] <= trial["layers"] and other["width"] <= trial["width"]
               and (other["layers"] < trial["layers"] or other["width"] < trial["width"])
               for other in successes):
            continue
        frontier.append({key: trial[key] for key in
                         ("layers", "width", "heads", "head_dim", "feed_forward_width",
                          "parameters", "checkpoint", "step", "output_dir")})
    return sorted(frontier, key=lambda point: (point["layers"], point["width"]))


def _validate_counts(result, *, success, path, include_sentences):
    errors = int(result["errors"])
    if not 0 <= errors <= 10002 or (errors == 0) != success:
        raise ValueError(f"{path}: error count disagrees with success/return code")
    _require_fields(result, {"targets": 10002}, path)
    if include_sentences:
        _require_fields(result, {"sentences": 1024}, path)
        exact = int(result["exact_sentences"])
        if not 0 <= exact <= 1024 or (exact == 1024) != success:
            raise ValueError(f"{path}: exact sentence count disagrees with success")
        if 1024 - exact > errors:
            raise ValueError(f"{path}: fewer token errors than incorrect sentences")
    return errors


def run_search(args, *, run_process=subprocess.run):
    """Run and verify the bounded search; the injectable runner enables CPU tests."""
    # Validate every requested combination before creating directories, including
    # widths traversal may later skip. Reuse its resolved shape for both phases.
    resolved_dimensions = {
        (layers, width): model_dimensions(layers, width, args.attention_heads)
        for layers in args.depths for width in args.widths
    }
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        raise ValueError(f"Native binary is not executable: {args.binary}")
    if not args.corpus.is_file() or not (args.tokenizer / "tokenizer.json").is_file():
        raise ValueError("Corpus and tokenizer.json must exist before training")
    for directory in (args.output_dir, args.checkpoint_dir):
        if directory.exists() or directory.is_symlink():
            raise ValueError(f"Refusing existing run directory: {directory}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = args.output_dir / "width_depth_search_summary.json"
    summary = {
        "started_utc": _now(), "status": "running",
        "configuration": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
        "corpus_sha256": _sha256(args.corpus),
        "tokenizer_sha256": _sha256(args.tokenizer / "tokenizer.json"),
        "binary_sha256": _sha256(args.binary),
        "scope": "Measured fixed-vocabulary GPT-2 configurations under the stated training budget; untested configurations are not failures.",
        "trials": [], "skipped_configurations": [],
        "verified_success_frontier": [], "minimum_parameter_success": None,
    }
    _write_summary(summary_path, summary)
    active = None
    smallest_verified_width = None
    try:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=False)
        for layers in args.depths:
            budget_failed = False
            for width in args.widths:
                active = None
                if smallest_verified_width is not None and width >= smallest_verified_width:
                    summary["skipped_configurations"].append({
                        "layers": layers, "width": width,
                        "reason": "dominated_by_verified_success",
                    })
                    continue
                if budget_failed:
                    summary["skipped_configurations"].append({
                        "layers": layers, "width": width,
                        "reason": "untested_after_wider_budget_failure_not_a_failure_claim",
                    })
                    continue
                for path, expected, label in (
                    (args.binary, summary["binary_sha256"], "Native binary"),
                    (args.corpus, summary["corpus_sha256"], "Corpus"),
                    (args.tokenizer / "tokenizer.json", summary["tokenizer_sha256"], "Tokenizer"),
                ):
                    if _sha256(path) != expected:
                        raise ValueError(f"{label} changed during the width/depth search")
                dimensions = resolved_dimensions[layers, width]
                output_parent = args.output_dir / f"width_{width}"
                checkpoint_parent = args.checkpoint_dir / f"width_{width}"
                output = output_parent / f"layers_{layers}"
                verification = output / "independent_verification"
                active = {
                    **dimensions, "expected_parameters": dimensions["parameters"],
                    "phase": "training", "status": "running", "commands": [],
                    "output_dir": str(output), "started_utc": _now(),
                }
                summary["trials"].append(active)

                def execute(command, *, stdout=None):
                    if command[0] == str(args.binary) and _sha256(args.binary) != summary["binary_sha256"]:
                        raise ValueError("Native binary changed during the width/depth search")
                    active["commands"].append(command)
                    _write_summary(summary_path, summary)
                    print(f"[{_now()}] {shlex.join(command)}", flush=True)
                    completed = run_process(command, check=False, stdout=stdout)
                    active["last_returncode"] = completed.returncode
                    _write_summary(summary_path, summary)
                    return completed.returncode

                shape_flags = [f"--layers={layers}", f"--model_width={width}",
                               f"--attention_heads={dimensions['heads']}",
                               f"--feed_forward_width={dimensions['feed_forward_width']}",
                               "--compact_vocabulary=false"]
                training = ([str(args.binary)] + shape_flags + [
                    f"--{key}={getattr(args, key)}" for key in
                    ("corpus", "tokenizer", "batch_size", "steps", "eval_every",
                     "checkpoint_every", "seed", "learning_rate", "warmup_steps", "training_seconds")
                ] + [f"--output_dir={output_parent}", f"--checkpoint_dir={checkpoint_parent}",
                     "--search=false", "--gradient_clip_norm=1"])
                status = execute(training)
                if status not in (0, 2):
                    raise ValueError(f"Training execution failed with return code {status}")
                success = status == 0
                result_path = output / "result.txt"
                result = _read_result(result_path)
                expected_fields = {key: dimensions[key] for key in
                                   ("layers", "width", "heads", "feed_forward_width", "parameters")}
                _require_fields(result, {**expected_fields, "success": int(success)}, result_path)
                errors = _validate_counts(result, success=success, path=result_path, include_sentences=False)
                step = int(result["step"])
                if not 0 <= step <= args.steps:
                    raise ValueError("Completed training step is outside the requested budget")
                if result.get("reached_time_limit") not in ("0", "1"):
                    raise ValueError("Missing or invalid reached_time_limit result field")
                if result["reached_time_limit"] == "1" and args.training_seconds == 0:
                    raise ValueError("Trial claims a disabled time limit was reached")
                if not success and step < args.steps and result["reached_time_limit"] != "1":
                    raise ValueError("A failed trial ended before either requested training budget")
                checkpoint = Path(result["checkpoint"]).resolve()
                if checkpoint != checkpoint_parent / f"layers_{layers}" / f"step_{step}" or not checkpoint.is_dir():
                    raise ValueError(f"Unexpected or missing final checkpoint: {checkpoint}")
                if (_sha256(output / "corpus.txt") != summary["corpus_sha256"]
                    or _sha256(output / "tokenizer.json") != summary["tokenizer_sha256"]):
                    raise ValueError("Run snapshots do not match the original corpus/tokenizer")
                active.update(training_result=result, checkpoint=str(checkpoint), step=step,
                              phase="checkpoint_verification")
                independent = [str(args.binary)] + shape_flags + [
                    f"--verify_checkpoint={checkpoint}", f"--output_dir={verification}",
                    f"--corpus={output / 'corpus.txt'}", f"--tokenizer={output}",
                    f"--batch_size={args.batch_size}", f"--seed={args.seed}",
                ]
                if execute(independent) != (0 if success else 2):
                    raise ValueError("Fresh checkpoint verification disagrees with training status")
                native_path = verification / "result.txt"
                native = _read_result(native_path)
                _require_fields(native, {**expected_fields, "checkpoint": checkpoint, "errors": errors}, native_path)
                _validate_counts(native, success=success, path=native_path, include_sentences=True)
                native_loss = float(native["mean_loss"])
                if not math.isfinite(native_loss) or native_loss < 0:
                    raise ValueError("Fresh checkpoint verification has invalid mean_loss")
                prediction_hash = _sha256(verification / "final_predictions.tsv")
                if _sha256(output / "final_predictions.tsv") != prediction_hash:
                    raise ValueError("Fresh checkpoint predictions differ from the saved training audit")
                active.update(checkpoint_verification=native, phase="prediction_artifact_verification")
                audit_file = verification / "verified_predictions.json"
                audit_command = [
                    sys.executable, str(SCRIPT_DIRECTORY / "verify_predictions.py"),
                    f"--corpus={output / 'corpus.txt'}", f"--tokenizer={output / 'tokenizer.json'}",
                    f"--predictions={verification / 'final_predictions.tsv'}",
                ]
                with audit_file.open("x", encoding="utf-8") as audit_output:
                    audit_status = execute(audit_command, stdout=audit_output)
                    audit_output.flush()
                    os.fsync(audit_output.fileno())
                if audit_status != (0 if success else 1):
                    raise ValueError("Independent prediction audit disagrees with training status")
                audit = json.loads(audit_file.read_text())
                if not isinstance(audit, dict):
                    raise ValueError("Independent artifact audit must be a JSON object")
                expected_audit = {
                    "success": success, "errors": errors, "targets": 10002,
                    "sentences": 1024, "exact_sentences": int(native["exact_sentences"]),
                    "corpus_sha256": summary["corpus_sha256"],
                    "tokenizer_sha256": summary["tokenizer_sha256"],
                    "predictions_sha256": prediction_hash,
                }
                for key, expected in expected_audit.items():
                    if audit.get(key) != expected:
                        raise ValueError(f"Independent artifact audit has unexpected {key}")
                audited_loss = float(audit["mean_loss_nats"])
                if not math.isfinite(audited_loss) or audited_loss < 0:
                    raise ValueError("Independent artifact audit has invalid mean_loss_nats")
                # The native result prints six significant decimal digits;
                # account for that formatting without masking a different loss.
                if not math.isclose(native_loss, audited_loss, rel_tol=1e-5, abs_tol=1e-12):
                    raise ValueError("Native and independent mean losses disagree")
                active.update(prediction_artifact_verification=str(audit_file),
                              errors=errors, finished_utc=_now(), phase="verified",
                              status="verified_success" if success else "verified_budget_failure")
                if success:
                    smallest_verified_width = width
                else:
                    budget_failed = True
                summary["verified_success_frontier"] = measured_frontier(summary["trials"])
                summary["minimum_parameter_success"] = min(
                    summary["verified_success_frontier"],
                    key=lambda point: (point["parameters"], point["layers"], point["width"]),
                    default=None,
                )
                _write_summary(summary_path, summary)
        summary["status"] = "completed"
        _write_summary(summary_path, summary)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        summary.update(status="error", error=str(error))
        if active is not None:
            active["status"] = "error"
        _write_summary(summary_path, summary)
        print(f"Width/depth search stopped: {error}", file=sys.stderr, flush=True)
        return 1


def main():
    args = parse_args()
    if importlib.util.find_spec("tokenizers") is None:
        print("Run with a Python environment containing tokenizers", file=sys.stderr)
        return 1
    try:
        return run_search(args)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
