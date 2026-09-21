#!/usr/bin/env python3
"""Train successively shallower GPT-2 models, verifying each before continuing.

Run with a Python environment containing tokenizers and a prebuilt native
memorize_general_facts binary. Defaults retain the controlled eight-block trial's
settings except that training no longer clips gradients. New runs do not
reproduce the historical norm-one-clipped training protocol.
Every depth starts from scratch, and both output/checkpoint parents must be
fresh. --start_layers=7 is an explicit continuation after separately verifying
the eight-block run; this script neither reuses nor claims to verify skipped
depths. Zero blocks are excluded because the corpus's conflicting token/position
contexts establish a model-independent error floor for that architecture.

Native stdout/stderr stream directly to the caller. subprocess.run has no
timeout: observing the driver does not terminate an ongoing training process.
Only the native soft training budget limits a trial. Exit 2 means its budget
ended without memorization, and exit 1 means an execution/verification failure.
No trial is retried and no hyperparameters are changed automatically.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


SCRIPT_DIRECTORY = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(
            "bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts"
        ),
    )
    parser.add_argument(
        "--corpus", type=Path, default=Path("testdata/general_facts_dataset.txt")
    )
    parser.add_argument(
        "--tokenizer", type=Path, required=True, help="GPT-2 tokenizer directory"
    )
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--start_layers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=128)
    parser.add_argument("--checkpoint_every", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--training_seconds", type=float, default=10800)
    args = parser.parse_args(argv)
    if not 1 <= args.start_layers <= 8:
        parser.error("--start_layers must be between 1 and 8")
    if any(
        not 0 < getattr(args, flag) < 2**31
        for flag in ("batch_size", "eval_every", "checkpoint_every")
    ):
        parser.error(
            "batch_size, eval_every, and checkpoint_every must be positive int32 values"
        )
    if args.steps < 0 or args.steps >= 2**31 - 1 or args.warmup_steps < 0:
        parser.error("steps and warmup_steps must be nonnegative, with steps < INT_MAX")
    if args.batch_size * 1024 > 2**31 - 1:
        parser.error("batch token count exceeds the native int32 limit")
    if not -(2**31) <= args.seed < 2**31 or args.warmup_steps >= 2**31:
        parser.error("seed and warmup_steps must fit the native int32 flags")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning_rate must be finite and positive")
    if not math.isfinite(args.training_seconds) or args.training_seconds < 0:
        parser.error("training_seconds must be finite and nonnegative")
    for field in ("binary", "corpus", "tokenizer", "checkpoint_dir", "output_dir"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    if args.output_dir == args.checkpoint_dir or (
        args.output_dir in args.checkpoint_dir.parents
        or args.checkpoint_dir in args.output_dir.parents
    ):
        parser.error(
            "artifact and checkpoint directories must be separate, non-nested paths"
        )
    return args


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_summary(path, summary):
    # Replace only the driver's own summary, atomically. fsync both the file
    # and directory so a completed phase remains durable across a host crash.
    summary["updated_utc"] = _now()
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=".depth-search-summary-",
        delete=False,
    ) as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
        temporary = Path(output.name)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _read_result(path):
    result = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in result:
            raise ValueError(f"Malformed or duplicate result field in {path}: {line}")
        result[key] = value
    return result


def _require_fields(result, expected, path):
    for key, value in expected.items():
        if result.get(key) != str(value):
            raise ValueError(f"{path}: expected {key}={value}, got {result.get(key)!r}")


def run_search(args, *, run_process=subprocess.run):
    """Run bounded depths sequentially; injected subprocess runner enables tests."""
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        raise ValueError(f"Native binary is not executable: {args.binary}")
    if not args.corpus.is_file() or not (args.tokenizer / "tokenizer.json").is_file():
        raise ValueError("Corpus and tokenizer.json must exist before training")
    for directory in (args.output_dir, args.checkpoint_dir):
        if directory.exists() or directory.is_symlink():
            raise ValueError(f"Refusing existing run directory: {directory}")
    corpus_hash = _sha256(args.corpus)
    tokenizer_hash = _sha256(args.tokenizer / "tokenizer.json")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = args.output_dir / "depth_search_summary.json"
    summary = {
        "started_utc": _now(),
        "status": "running",
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "corpus_sha256": corpus_hash,
        "tokenizer_sha256": tokenizer_hash,
        "binary_sha256": _sha256(args.binary),
        "smallest_verified_layers": None,
        "depths": [],
        "skipped_higher_depths_require_external_verification": args.start_layers < 8,
        "zero_blocks_excluded": "conflicting current-token/position contexts give a nonzero error bound",
    }
    _write_summary(summary_path, summary)
    active = None
    try:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=False)
        for layers in range(args.start_layers, 0, -1):
            active = None
            if _sha256(args.binary) != summary["binary_sha256"]:
                raise ValueError("Native binary changed during the depth search")
            output = args.output_dir / f"layers_{layers}"
            verification = output / "independent_verification"
            active = {
                "layers": layers,
                "phase": "training",
                "commands": [],
                "output_dir": str(output),
            }
            summary["depths"].append(active)

            def execute(command, *, stdout=None):
                if (
                    command[0] == str(args.binary)
                    and _sha256(args.binary) != summary["binary_sha256"]
                ):
                    raise ValueError("Native binary changed during the depth search")
                active["commands"].append(command)
                _write_summary(summary_path, summary)
                print(f"[{_now()}] {shlex.join(command)}", flush=True)
                # No shell and no timeout. Native output is inherited unless
                # stdout is explicitly the short independent JSON audit file.
                completed = run_process(command, check=False, stdout=stdout)
                active["last_returncode"] = completed.returncode
                _write_summary(summary_path, summary)
                return completed.returncode

            training = (
                [str(args.binary)]
                + [
                    f"--{key}={getattr(args, key)}"
                    for key in (
                        "corpus",
                        "tokenizer",
                        "checkpoint_dir",
                        "output_dir",
                        "batch_size",
                        "steps",
                        "eval_every",
                        "checkpoint_every",
                        "seed",
                        "learning_rate",
                        "warmup_steps",
                        "training_seconds",
                    )
                ]
                + [f"--layers={layers}", "--search=false", "--compact_vocabulary=false"]
            )
            status = execute(training)
            if status != 0:
                summary["status"] = (
                    "budget_exhausted" if status == 2 else "training_failed"
                )
                active["status"] = summary["status"]
                _write_summary(summary_path, summary)
                return 2 if status == 2 else 1

            training_result = _read_result(output / "result.txt")
            _require_fields(
                training_result,
                {"success": 1, "layers": layers, "errors": 0, "targets": 10002},
                output / "result.txt",
            )
            step = int(training_result["step"])
            if step < 0 or step > args.steps:
                raise ValueError(
                    "Completed training step is outside the requested budget"
                )
            checkpoint = Path(training_result["checkpoint"]).resolve()
            expected_checkpoint = (
                args.checkpoint_dir / f"layers_{layers}" / f"step_{step}"
            )
            if checkpoint != expected_checkpoint or not checkpoint.is_dir():
                raise ValueError(
                    f"Unexpected or missing final checkpoint: {checkpoint}"
                )
            if (
                _sha256(output / "corpus.txt") != corpus_hash
                or _sha256(output / "tokenizer.json") != tokenizer_hash
            ):
                raise ValueError(
                    "Run snapshots do not match the original corpus/tokenizer"
                )
            active["training_result"] = training_result
            active["phase"] = "checkpoint_verification"
            independent = [
                str(args.binary),
                f"--layers={layers}",
                "--compact_vocabulary=false",
                f"--verify_checkpoint={checkpoint}",
                f"--output_dir={verification}",
                f"--corpus={output / 'corpus.txt'}",
                f"--tokenizer={output}",
                f"--batch_size={args.batch_size}",
                f"--seed={args.seed}",
            ]
            if execute(independent) != 0:
                summary["status"] = active["status"] = "checkpoint_verification_failed"
                _write_summary(summary_path, summary)
                return 1
            native_result = _read_result(verification / "result.txt")
            _require_fields(
                native_result,
                {
                    "checkpoint": checkpoint,
                    "layers": layers,
                    "errors": 0,
                    "targets": 10002,
                    "sentences": 1024,
                    "exact_sentences": 1024,
                },
                verification / "result.txt",
            )
            active["checkpoint_verification"] = native_result
            active["phase"] = "prediction_artifact_verification"
            audit_file = verification / "verified_predictions.json"
            command = [
                sys.executable,
                str(SCRIPT_DIRECTORY / "verify_predictions.py"),
                f"--corpus={output / 'corpus.txt'}",
                f"--tokenizer={output / 'tokenizer.json'}",
                f"--predictions={verification / 'final_predictions.tsv'}",
            ]
            with audit_file.open("x", encoding="utf-8") as audit_output:
                status = execute(command, stdout=audit_output)
                audit_output.flush()
                os.fsync(audit_output.fileno())
            if status != 0:
                summary["status"] = active["status"] = (
                    "prediction_artifact_verification_failed"
                )
                _write_summary(summary_path, summary)
                return 1
            audit = json.loads(audit_file.read_text())
            if not isinstance(audit, dict):
                raise ValueError("Independent artifact audit must be a JSON object")
            for key, expected in {
                "success": True,
                "errors": 0,
                "targets": 10002,
                "sentences": 1024,
                "exact_sentences": 1024,
                "corpus_sha256": corpus_hash,
                "tokenizer_sha256": tokenizer_hash,
            }.items():
                if audit.get(key) != expected:
                    raise ValueError(f"Independent artifact audit has unexpected {key}")
            active["prediction_artifact_verification"] = str(audit_file)
            active["phase"] = active["status"] = "verified"
            summary["smallest_verified_layers"] = layers
            _write_summary(summary_path, summary)
        summary["status"] = "all_requested_depths_verified"
        _write_summary(summary_path, summary)
        return 0
    except (OSError, ValueError, KeyError) as error:
        summary["status"] = "error"
        summary["error"] = str(error)
        if active is not None:
            active["status"] = "error"
        _write_summary(summary_path, summary)
        print(f"Depth search stopped: {error}", file=sys.stderr, flush=True)
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
