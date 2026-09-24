#!/usr/bin/env python3
"""Sequential, fresh compact-vocabulary trials with a hard shared deadline.

Candidates are explicit L:W:FF:steps:LR entries, in caller-selected order.
Only independently reloaded, prediction-audited, and greedily completed models
count as verified successes. A bounded failure is not a capacity conclusion.
Run with tokenizers installed; preserve LD_LIBRARY_PATH in the caller environment.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from compact_checkpoint import parse_mapping
from run_depth_search import _now, _read_result, _require_fields, _sha256, _write_summary
from verify_predictions import _corpus_lines, verify_predictions


def parameter_count(layers, width, ff, vocabulary=4475, context=27):
    return (vocabulary + context + 2) * width + layers * (
        4 * width * width + 2 * width * ff + 9 * width + ff)


def parse_candidate(value):
    try:
        layers, width, ff, steps, learning_rate = value.split(":")
        integers = list(map(int, (layers, width, ff, steps)))
        rate = float(learning_rate)
        if integers[0] < 2 or any(not 0 < item < 2**31 - 1 for item in integers):
            raise ValueError("layers must be >=2; dimensions and steps must fit positive int32")
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        if any(item >= 2**31 for item in (
                4475 * integers[1], 3 * integers[1]**2,
                integers[1] * integers[2], 27 * integers[2])):
            raise ValueError("candidate tensor exceeds int32 element limit")
        return dict(zip(("layers", "width", "feed_forward_width", "steps", "learning_rate"),
                        [*integers, rate]))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(f"Invalid candidate {value!r}: {error}") from error


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "corpus", "tokenizer", "run_dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--deadline_unix", type=float, required=True)
    parser.add_argument("--candidates", required=True,
                        help="Comma-separated L:W:FF:steps:LR entries, layers >=2")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, choices=(27,), default=27)
    parser.add_argument("--vocabulary", type=int, choices=(4475,), default=4475)
    parser.add_argument("--verification_reserve", type=float, default=30)
    parser.add_argument("--max_training_seconds", type=float, default=0,
                        help="Optional per-trial cap; zero uses all remaining training time")
    parser.add_argument("--eval_every", type=int, default=256)
    args = parser.parse_args(argv)
    try:
        args.candidates = [parse_candidate(item) for item in args.candidates.split(",")]
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if not math.isfinite(args.deadline_unix) or args.deadline_unix <= 0:
        parser.error("deadline_unix must be finite and positive")
    for name in ("verification_reserve", "max_training_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"{name} must be finite and nonnegative")
    if args.verification_reserve <= 0:
        parser.error("verification_reserve must be positive")
    if not 0 < args.batch_size * args.context_length < 2**31:
        parser.error("batch token count must fit positive int32")
    if not 0 < args.eval_every < 2**31 or not -(2**31) <= args.seed < 2**31:
        parser.error("eval_every must be positive int32; seed must fit int32")
    for name in ("binary", "corpus", "tokenizer", "run_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    repository = Path(__file__).resolve().parents[2]
    if args.run_dir == repository or repository in args.run_dir.parents:
        parser.error("run_dir must be outside the repository")
    return args


def load_inputs(corpus, tokenizer_path, context, vocabulary):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    lines = _corpus_lines(corpus.read_bytes())
    rows = [tokenizer.encode(line, add_special_tokens=False).ids for line in lines]
    if len(rows) != 1024 or any(not 5 <= len(row) <= context for row in rows):
        raise ValueError("Expected 1024 corpus lines fitting the requested context")
    if tokenizer.get_vocab_size(with_added_tokens=True) != 50257 or tokenizer.token_to_id("<|endoftext|>") != 50256:
        raise ValueError("Expected original GPT-2 tokenizer")
    active = tuple(sorted({50256}.union(*(set(row) for row in rows))))
    if len(active) != vocabulary or sum(len(row) - 4 for row in rows) != 10002:
        raise ValueError("Corpus compact vocabulary or scored-target count changed")
    prompts = [tokenizer.decode(row[:5], skip_special_tokens=False) for row in rows]
    if any(tokenizer.encode(prompt, add_special_tokens=False).ids != row[:5]
           for prompt, row in zip(prompts, rows)):
        raise ValueError("Decoded five-token prompts do not round-trip")
    return {"lines": lines, "rows": rows, "prompts": prompts, "active": active}


def execute(command, stdout_path, stderr_path, *, deadline, input_path=None, on_start=None):
    """File-backed output, visible progress, inherited environment, hard timeout."""
    if time.time() >= deadline:
        raise TimeoutError("deadline reached before subprocess launch")
    with stdout_path.open("x") as output, stderr_path.open("x") as errors:
        source = input_path.open() if input_path else subprocess.DEVNULL
        process = None
        try:
            process = subprocess.Popen(command, stdin=source, stdout=output, stderr=errors,
                                       start_new_session=True)
            if on_start:
                on_start(process.pid)
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("shared wall-clock deadline reached")
                try:
                    return process.wait(timeout=min(15, remaining))
                except subprocess.TimeoutExpired:
                    print(f"[{_now()}] pid={process.pid} remaining={max(0, deadline-time.time()):.0f}s log={stdout_path}", flush=True)
                    with stdout_path.open("rb") as progress:
                        progress.seek(max(0, stdout_path.stat().st_size - 2048))
                        tail = progress.read().decode("utf-8", errors="replace").splitlines()
                    if tail:
                        print(tail[-1], flush=True)
        finally:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            if input_path:
                source.close()


def greedy_audit(output, lines):
    records = output.read_text().splitlines()
    banner = "Enter a prompt (Ctrl-D or Ctrl-C to quit). Each line starts a new completion."
    if len(records) != len(lines) + 2 or records[0] != banner or records[-1] != "> ":
        raise ValueError("Malformed or incomplete interactive greedy audit")
    errors = [index for index, (actual, expected) in enumerate(zip(records[1:-1], lines), 1)
              if actual != "> " + expected]
    return {"sentences": len(lines), "exact_sentences": len(lines) - len(errors),
            "errors": len(errors), "incorrect_lines_1based": errors}


def run_search(args, *, run_process=execute, loader=load_inputs, clock=time.time):
    if args.run_dir.exists() or args.run_dir.is_symlink():
        raise ValueError(f"Refusing existing run directory: {args.run_dir}")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        raise ValueError("binary must be an executable file")
    if clock() >= args.deadline_unix:
        raise ValueError("deadline already passed")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    summary = {"status": "running", "started_utc": _now(), "driver_pid": os.getpid(),
               "configuration": {key: str(value) if isinstance(value, Path) else value
                                 for key, value in vars(args).items()},
               "scope": "Explicit bounded trials; failures do not prove a capacity limit.",
               "trials": [], "minimum_parameter_success": None}
    active = None

    def persist():
        successes = [trial for trial in summary["trials"] if trial["status"] == "verified"]
        summary["minimum_parameter_success"] = min(successes, key=lambda item: item["parameters"], default=None)
        _write_summary(args.run_dir / "summary.json", summary)
        temporary = args.run_dir / ".summary.tsv.tmp"
        keys = ("layers", "width", "feed_forward_width", "parameters", "steps", "learning_rate", "status", "step", "errors", "checkpoint")
        with temporary.open("w") as output:
            output.write("\t".join(keys) + "\n")
            for trial in summary["trials"]:
                output.write("\t".join(str(trial.get(key, "")) for key in keys) + "\n")
        temporary.replace(args.run_dir / "summary.tsv")

    persist()
    try:
        inputs = args.run_dir / "inputs"
        inputs.mkdir()
        binary = inputs / "memorize_general_facts"
        corpus = inputs / "corpus.txt"
        shutil.copy2(args.binary, binary)
        shutil.copy2(args.corpus, corpus)
        shutil.copy2(args.tokenizer / "tokenizer.json", inputs / "tokenizer.json")
        sources = inputs / "scripts"
        sources.mkdir()
        for name in (Path(__file__).name, "verify_predictions.py", "compact_checkpoint.py", "run_depth_search.py"):
            shutil.copy2(Path(__file__).with_name(name), sources / name)
        summary["hashes"] = {str(path.relative_to(inputs)): _sha256(path)
                             for path in inputs.rglob("*") if path.is_file()}
        summary["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "")
        data = loader(corpus, inputs, args.context_length, args.vocabulary)
        prompts = inputs / "prompts.txt"
        prompts.write_text("\n".join(data["prompts"]) + "\n")
        persist()
        for index, candidate in enumerate(args.candidates):
            remaining = args.deadline_unix - clock() - args.verification_reserve
            if remaining <= 1:
                summary["status"] = "deadline"
                break
            trial_dir = args.run_dir / f"trial_{index:03d}_L{candidate['layers']}_W{candidate['width']}_FF{candidate['feed_forward_width']}"
            trial_dir.mkdir()
            active = {**candidate, "parameters": parameter_count(candidate["layers"], candidate["width"], candidate["feed_forward_width"], args.vocabulary, args.context_length),
                      "status": "running", "started_utc": _now(), "commands": [], "directory": str(trial_dir)}
            summary["trials"].append(active)

            def launch(command, phase, deadline, input_path=None):
                active["phase"] = phase
                active["commands"].append(command)
                if _sha256(binary) != summary["hashes"]["memorize_general_facts"]:
                    raise ValueError("Pinned binary changed")
                persist()
                print(f"[{_now()}] {phase} L{candidate['layers']} W{candidate['width']} FF{candidate['feed_forward_width']}", flush=True)
                def started(pid):
                    active["child_pid"] = pid
                    persist()
                code = run_process(command, trial_dir / f"{phase}.stdout.log", trial_dir / f"{phase}.stderr.log",
                                   deadline=deadline, input_path=input_path, on_start=started)
                active["child_pid"] = None
                active["last_returncode"] = code
                persist()
                return code

            shape = [f"--layers={candidate['layers']}", f"--model_width={candidate['width']}",
                     f"--feed_forward_width={candidate['feed_forward_width']}", "--attention_heads=1",
                     f"--context_length={args.context_length}", "--compact_vocabulary=true", f"--seed={args.seed}"]
            output_parent = trial_dir / "training"
            output = output_parent / f"layers_{candidate['layers']}"
            checkpoints = trial_dir / "checkpoints"
            seconds = min(remaining - 1, args.max_training_seconds) if args.max_training_seconds else remaining - 1
            training = [str(binary), "--mode=train_model", *shape, f"--corpus={corpus}", f"--tokenizer={inputs}",
                        f"--output_dir={output_parent}", f"--checkpoint_dir={checkpoints}", "--search=false",
                        f"--batch_size={args.batch_size}", f"--steps={candidate['steps']}", f"--learning_rate={candidate['learning_rate']}",
                        "--warmup_steps=100", f"--eval_every={args.eval_every}", f"--checkpoint_every={candidate['steps']}",
                        f"--training_seconds={seconds}"]
            code = launch(training, "training", args.deadline_unix - args.verification_reserve)
            if code not in (0, 2):
                raise ValueError(f"Native training failed with exit {code}")
            result = _read_result(output / "result.txt")
            expected = {key: active[key] for key in ("layers", "width", "feed_forward_width", "parameters")}
            expected.update(heads=1, vocabulary=args.vocabulary, targets=10002,
                            context_length=args.context_length)
            _require_fields(result, {**expected, "success": int(code == 0)}, output / "result.txt")
            step, errors = int(result["step"]), int(result["errors"])
            if not 0 <= step <= candidate["steps"] or not 0 <= errors <= 10002 or (errors == 0) != (code == 0):
                raise ValueError("Invalid training step/error count")
            if result.get("reached_time_limit") not in ("0", "1"):
                raise ValueError("Missing or invalid native time-limit status")
            if code == 2 and step < candidate["steps"] and result["reached_time_limit"] != "1":
                raise ValueError("Budget failure stopped before either declared budget")
            checkpoint = checkpoints / f"layers_{candidate['layers']}" / f"step_{step}"
            if Path(result["checkpoint"]).resolve() != checkpoint or not checkpoint.is_dir():
                raise ValueError("Missing or unexpected final checkpoint")
            for path in (output, checkpoint):
                if parse_mapping((path / "compact_vocabulary.tsv").read_bytes()) != data["active"]:
                    raise ValueError("Compact mapping differs from independent tokenization")
            _require_fields(_read_result(output / "config.txt"), {"context_length": args.context_length, "batch_size": args.batch_size}, output / "config.txt")
            if _sha256(output / "corpus.txt") != summary["hashes"]["corpus.txt"] or _sha256(output / "tokenizer.json") != summary["hashes"]["tokenizer.json"]:
                raise ValueError("Native input snapshots differ from pinned inputs")
            active.update(training_result=result, step=step, errors=errors, checkpoint=str(checkpoint))
            if code == 2:
                active.update(status="budget_fail", finished_utc=_now())
                persist()
                continue
            verification = trial_dir / "verification"
            command = [str(binary), "--mode=infer_model", *shape, f"--verify_checkpoint={checkpoint}",
                       f"--output_dir={verification}", f"--corpus={corpus}", f"--tokenizer={inputs}", f"--batch_size={args.batch_size}"]
            if launch(command, "verification", args.deadline_unix) != 0:
                raise ValueError("Fresh checkpoint inference failed")
            native = _read_result(verification / "result.txt")
            _require_fields(native, {**expected, "checkpoint": checkpoint, "errors": 0, "sentences": 1024, "exact_sentences": 1024}, verification / "result.txt")
            predictions = verification / "final_predictions.tsv"
            with predictions.open() as source:
                audit = verify_predictions(data["rows"], source, eos_id=50256, vocabulary_size=50257, context_length=args.context_length)
            if audit.errors or _sha256(predictions) != _sha256(output / "final_predictions.tsv"):
                raise ValueError("Independent prediction audit differs or contains errors")
            native_loss = float(native["mean_loss"])
            if (audit.targets != 10002 or not math.isfinite(native_loss) or native_loss < 0
                    or not math.isclose(native_loss, audit.mean_loss, rel_tol=1e-5, abs_tol=1e-12)):
                raise ValueError("Independent target count or mean loss differs")
            report = {**audit.summary(), "context_length": args.context_length, "corpus_sha256": _sha256(corpus),
                      "tokenizer_sha256": _sha256(inputs / "tokenizer.json"), "predictions_sha256": _sha256(predictions)}
            (trial_dir / "prediction_verification.json").write_text(json.dumps(report, indent=2) + "\n")
            greedy = [str(binary), "--mode=infer_model", *shape, f"--infer_checkpoint={checkpoint}",
                      f"--tokenizer={inputs}", f"--generation_tokens={args.context_length}"]
            if launch(greedy, "greedy", args.deadline_unix, prompts) != 0:
                raise ValueError("Greedy inference process failed")
            report = greedy_audit(trial_dir / "greedy.stdout.log", data["lines"])
            (trial_dir / "greedy_verification.json").write_text(json.dumps(report, indent=2) + "\n")
            if report["errors"]:
                raise ValueError("Greedy completions are not all exact")
            active.update(status="verified", finished_utc=_now())
            persist()
        else:
            summary["status"] = "completed"
        return 0
    except TimeoutError as error:
        summary.update(status="deadline", error=str(error))
        if active:
            active.update(status="timeout", child_pid=None)
        return 0
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        if active:
            active.update(status="interrupted", child_pid=None)
        return 130
    except (ImportError, OSError, ValueError, KeyError, TypeError) as error:
        summary.update(status="error", error=str(error))
        if active:
            active.update(status="error", child_pid=None)
        print(f"Sweep stopped: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        summary.update(finished_utc=_now(), driver_pid=None)
        persist()


def main():
    args = parse_args()
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return run_search(args)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
