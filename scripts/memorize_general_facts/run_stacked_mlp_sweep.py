#!/usr/bin/env python3
"""Fit 1..5 residual MLP blocks to one frozen, already trained A3 checkpoint.

Standard-library orchestration only: no source training, tokenizer downloads, or
GPU dependency in this driver. Every native child uses the same pinned inputs.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time


SCORED = 10002
FACTS = 1024
SOURCE_BUDGET = 1380
SHAPE = {"source_layers": 4, "model_width": 10, "source_mlp_width": 20,
         "attention_heads": 1, "context_length": 27, "vocabulary_size": 4475}
TSV_FIELDS = ("depth", "width", "trainable_parameters", "budget_relation",
              "status", "best_step", "best_correct", "best_token_accuracy",
              "best_non_eos_accuracy", "exact_greedy_facts", "seconds",
              "wall_seconds", "error")
BEST = re.compile(
    r"^BEST step=(\d+) updates=(\d+) seconds=([\d.eE+-]+) "
    r"mean_ce=([\d.eE+-]+) correct=(\d+)/(\d+) "
    r"token_accuracy=([\d.eE+-]+)% greedy_complete=(\d+)/(\d+) "
    r"frozen_head_unchanged=true$", re.MULTILINE)
EOS = re.compile(r"^EOS breakdown: eos_correct=(\d+) eos_scored=(\d+) "
                 r"non_eos_correct=(\d+) non_eos_scored=(\d+)$", re.MULTILINE)


def parameter_count(depth, width):
    return depth * (21 * width + 30) + 20


def budget_relation(parameters):
    return "below" if parameters < SOURCE_BUDGET else (
        "equal" if parameters == SOURCE_BUDGET else "above")


def positive_list(value):
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item <= 0 or item > 2**31 - 1 for item in result):
        raise argparse.ArgumentTypeError("values must be positive int32 integers")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("duplicate values are not allowed")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "checkpoint", "tokenizer", "corpus", "run_dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--depths", type=positive_list, default=[1, 2, 3, 4, 5])
    parser.add_argument("--widths", type=positive_list, default=[20, 40, 80, 150])
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--timeout_seconds", type=float, default=3600)
    args = parser.parse_args(argv)
    if any(depth > 5 for depth in args.depths):
        parser.error("depths must be in 1..5")
    for name in ("steps", "eval_every", "batch_size"):
        if not 0 < getattr(args, name) < 2**31:
            parser.error(f"{name} must be a positive int32 integer")
    if not 0 <= args.seed < 2**31:
        parser.error("seed must be a nonnegative int32 integer")
    if not 1 <= args.max_workers <= 4:
        parser.error("max_workers must be in 1..4")
    for name in ("learning_rate", "timeout_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    if any(10 * width > 2**31 - 1 for width in args.widths):
        parser.error("width exceeds the native tensor-size bound")
    for name in ("binary", "checkpoint", "tokenizer", "corpus", "run_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    repository = Path(__file__).resolve().parents[2]
    if args.run_dir == repository or repository in args.run_dir.parents:
        parser.error("run_dir must be outside the repository")
    return args


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, content):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def checkpoint_elements(depth, width):
    return [value for _ in range(depth)
            for value in (10, 10, 10 * width, width, 10 * width, 10)] + [10, 10]


def validate_weights(directory, elements):
    expected = {f"weight_{index}.bin" for index in range(len(elements))}
    observed = {path.name for path in directory.glob("weight_*.bin")}
    if observed != expected:
        raise ValueError(f"unexpected checkpoint tensor set: {directory}")
    for index, count in enumerate(elements):
        if (directory / f"weight_{index}.bin").stat().st_size != 4 * count:
            raise ValueError(f"unexpected tensor shape: weight_{index}.bin")


def validate_inputs(args):
    if args.run_dir.exists() or args.run_dir.is_symlink():
        raise ValueError(f"refusing existing run directory: {args.run_dir}")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        raise ValueError("binary must be an executable file")
    if not (args.tokenizer / "tokenizer.json").is_file():
        raise ValueError("tokenizer must contain tokenizer.json")
    if len(args.corpus.read_text(encoding="utf-8").splitlines()) != FACTS:
        raise ValueError("corpus must contain exactly 1024 facts")
    if not (args.checkpoint / "compact_vocabulary.tsv").is_file():
        raise ValueError("checkpoint must include compact_vocabulary.tsv")
    source_elements = [4475 * 10, 27 * 10]
    source_elements += [10, 10, 300, 30, 100, 10, 10, 10, 200, 20, 200, 10] * 4
    validate_weights(args.checkpoint, source_elements + [10, 10])


def snapshot(args):
    inputs = args.run_dir / "inputs"
    inputs.mkdir()
    sources = {"memorize_general_facts": args.binary, "corpus.txt": args.corpus,
               "tokenizer.json": args.tokenizer / "tokenizer.json"}
    for path in sorted(args.checkpoint.rglob("*")):
        if path.is_file():
            sources[f"checkpoint/{path.relative_to(args.checkpoint)}"] = path
    for name in ("run_stacked_mlp_sweep.py", "run_stacked_mlp_sweep_test.py",
                 "STACKED_MLP_SWEEP.md"):
        sources[f"scripts/{name}"] = Path(__file__).with_name(name)
    repository = Path(__file__).resolve().parents[2]
    native = repository / "src/llm/experiments/memorize_general_facts"
    for name in ("puzzle.cc", "puzzle.h", "puzzle_readout.cc", "puzzle_readout.h",
                 "memorize_general_facts.cc"):
        path = native / name
        if path.is_file():
            sources[f"native_sources/{name}"] = path
    files = {}
    for name, source in sources.items():
        destination = inputs / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(source)
        shutil.copy2(source, destination)
        if sha256(destination) != before or sha256(source) != before:
            raise ValueError(f"input changed while snapshotting: {source}")
        destination.chmod(0o555 if name == "memorize_general_facts" else 0o444)
        files[name] = {"sha256": before, "bytes": destination.stat().st_size,
                       "source": str(source)}
    return files


def verify_snapshot(inputs, manifest):
    for name, metadata in manifest.items():
        if sha256(inputs / name) != metadata["sha256"]:
            raise ValueError(f"pinned input changed: {name}")


def command(args, record):
    inputs = args.run_dir / "inputs"
    return [str(inputs / "memorize_general_facts"), "--mode=puzzle", "--train_mlp",
            f"--puzzle_checkpoint={inputs / 'checkpoint'}",
            f"--tokenizer={inputs}", f"--corpus={inputs / 'corpus.txt'}",
            "--layers=4", "--model_width=10", "--attention_heads=1",
            "--feed_forward_width=20", "--context_length=27",
            "--compact_vocabulary=true", f"--mlp_depth={record['depth']}",
            f"--mlp_width={record['width']}", "--match_mlp_parameter_budget=false",
            f"--steps={args.steps}", f"--eval_every={args.eval_every}",
            f"--batch_size={args.batch_size}", f"--seed={args.seed}",
            f"--learning_rate={args.learning_rate}",
            f"--output_dir={Path(record['directory']) / 'puzzle'}"]


def finite(value, name):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"invalid {name}: {value}")
    return result


def read_history(path, *, complete=False):
    if not path.exists():
        if complete:
            raise ValueError("missing training.tsv")
        return []
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        if complete:
            raise ValueError("truncated training.tsv")
        text = text[:text.rfind("\n") + 1]
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    expected = ["step", "seconds", "mean_ce", "correct", "scored", "complete_facts",
                "eos_correct", "eos_scored"]
    if reader.fieldnames != expected:
        if not complete and not text:
            return []
        raise ValueError("unexpected training.tsv columns")
    result = []
    for row in reader:
        values = {key: finite(row[key], key) if key in ("seconds", "mean_ce")
                  else int(row[key]) for key in expected}
        if (values["step"] < 0 or values["scored"] != SCORED
                or not 0 <= values["correct"] <= SCORED
                or not 0 <= values["complete_facts"] <= FACTS
                or values["eos_scored"] != FACTS
                or not 0 <= values["eos_correct"] <= FACTS
                or not 0 <= values["correct"] - values["eos_correct"] <= SCORED - FACTS):
            raise ValueError("invalid training.tsv counts")
        if result and (values["step"] <= result[-1]["step"]
                       or values["seconds"] < result[-1]["seconds"]):
            raise ValueError("training.tsv must advance monotonically")
        result.append(values)
    return result


def history_metrics(history):
    if not history:
        return {}
    best = min(history, key=lambda row: (-row["correct"], row["mean_ce"], row["step"]))
    return {"best_step": best["step"], "best_correct": best["correct"],
            "best_mean_ce": best["mean_ce"],
            "best_token_accuracy": best["correct"] / SCORED,
            "best_non_eos_accuracy": (best["correct"] - best["eos_correct"]) / (SCORED - FACTS),
            "best_eos_accuracy": best["eos_correct"] / FACTS,
            "teacher_forced_complete_facts": best["complete_facts"],
            "last_step": history[-1]["step"], "seconds": history[-1]["seconds"]}


def validate_result(args, record):
    directory = Path(record["directory"])
    fields = {}
    for line in (directory / "puzzle/run.txt").read_text().splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise ValueError("malformed or duplicate run.txt field")
        fields[key] = value
    expected = {**SHAPE, "train_mlp": 1, "mlp_depth": record["depth"],
                "match_mlp_parameter_budget": 0,
                "requested_mlp_width": record["width"], "mlp_width": record["width"],
                "source_tail_parameters": SOURCE_BUDGET,
                "mlp_parameters": record["depth"] * (21 * record["width"] + 10),
                "trainable_parameters": record["trainable_parameters"],
                "steps": args.steps, "eval_every": args.eval_every,
                "batch_size": args.batch_size, "seed": args.seed}
    for key, value in expected.items():
        if fields.get(key) != str(value):
            raise ValueError(f"run.txt {key}: expected {value}, got {fields.get(key)!r}")
    for key, value in {"checkpoint": args.run_dir / "inputs/checkpoint",
                       "corpus": args.run_dir / "inputs/corpus.txt"}.items():
        if fields.get(key) != str(value):
            raise ValueError(f"run.txt {key} does not refer to the pinned input")
    if not math.isclose(float(fields["learning_rate"]), args.learning_rate, rel_tol=1e-5):
        raise ValueError("run.txt learning_rate mismatch")
    log = (directory / "stdout.log").read_text(encoding="utf-8")
    if f"Original model: correct={SCORED}/{SCORED} scored next tokens\n" not in log:
        raise ValueError("missing original-source correctness check")
    separation = re.findall(r"^Verified A3 separation: no identical hidden vectors have different "
                            r"scored next-token targets; scored=(\d+) unique_vectors=(\d+) "
                            r"distinct_targets=(\d+)$", log, re.MULTILINE)
    if (len(separation) != 1 or tuple(map(int, separation[0][:2])) != (SCORED, SCORED)
            or not 0 < int(separation[0][2]) <= SHAPE["vocabulary_size"]):
        raise ValueError("missing or invalid A3 separation check")
    if "Frozen source parameters unchanged.\n" not in log:
        raise ValueError("missing frozen-source check")
    matches = list(BEST.finditer(log))
    if len(matches) != 1:
        raise ValueError("expected exactly one valid BEST line and frozen-head check")
    step, updates, seconds, loss, correct, scored, accuracy, greedy, facts = matches[0].groups()
    step, updates, correct, scored, greedy, facts = map(int, (step, updates, correct, scored, greedy, facts))
    seconds, loss, accuracy = finite(seconds, "seconds"), finite(loss, "mean_ce"), finite(accuracy, "accuracy")
    if (updates != args.steps or not 0 <= step <= updates or scored != SCORED
            or facts != FACTS or not 0 <= correct <= SCORED or not 0 <= greedy <= FACTS
            or abs(accuracy - 100 * correct / scored) > 0.000051):
        raise ValueError("BEST counts, step budget, or accuracy mismatch")
    history = read_history(directory / "puzzle/training.tsv", complete=True)
    expected_steps = list(range(0, args.steps + 1, args.eval_every))
    if expected_steps[-1] != args.steps:
        expected_steps.append(args.steps)
    if [row["step"] for row in history] != expected_steps:
        raise ValueError("incomplete evaluation history")
    metrics = history_metrics(history)
    best_row = next((row for row in history if row["step"] == step), None)
    # Native selection uses unrounded loss. Distinct native losses may print
    # identically to eight decimals; accept its selected row within that bound.
    if (best_row is None or correct != metrics["best_correct"]
            or correct != best_row["correct"]
            or abs(loss - best_row["mean_ce"]) > 1.1e-8
            or best_row["mean_ce"] > metrics["best_mean_ce"] + 1.1e-8
            or seconds + 0.011 < history[-1]["seconds"]):
        raise ValueError("BEST does not match the best complete evaluation")
    metrics.update(history_metrics([best_row]))
    metrics["last_step"] = history[-1]["step"]
    eos_matches = EOS.findall(log)
    if len(eos_matches) != 1:
        raise ValueError("expected exactly one EOS breakdown")
    eos_correct, eos_scored, other_correct, other_scored = map(int, eos_matches[0])
    if (eos_scored != FACTS or other_scored != SCORED - FACTS
            or eos_correct != best_row["eos_correct"]
            or other_correct != correct - eos_correct):
        raise ValueError("EOS breakdown does not match the best checkpoint")
    validate_weights(directory / "puzzle/best_mlp",
                     checkpoint_elements(record["depth"], record["width"]))
    return {**metrics, "seconds": seconds, "exact_greedy_facts": greedy,
            "frozen_head_unchanged": True, "frozen_source_unchanged": True,
            "validated": True}


def ranking(records):
    return sorted((record for record in records if record["status"] == "completed"
                   and record.get("validated")),
                  key=lambda row: (-row["best_correct"], row["best_mean_ce"],
                                   -row["exact_greedy_facts"], row["trainable_parameters"],
                                   row["depth"], row["width"]))


def persist(args, summary, started):
    summary["updated_utc"] = now()
    summary["elapsed_seconds"] = max(0, time.monotonic() - started)
    summary["ranking"] = [row["id"] for row in ranking(summary["trials"])]
    atomic_write(args.run_dir / "summary.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=TSV_FIELDS, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(summary["trials"])
    atomic_write(args.run_dir / "summary.tsv", output.getvalue())
    columns = [("depth", "Blocks"), ("width", "Hidden width"),
               ("trainable_parameters", "Trainable parameters"),
               ("budget_relation", "vs. original 1,380"), ("status", "Status"),
               ("best_step", "Best step"), ("best_token_accuracy", "All-token accuracy"),
               ("best_non_eos_accuracy", "Non-EOS accuracy"),
               ("exact_greedy_facts", "Exact greedy / 1,024"), ("wall_seconds", "Wall seconds")]
    rows = []
    for record in summary["trials"]:
        cells = []
        for key, _ in columns:
            value = record.get(key)
            rendered = "—" if value is None else (
                f"{100 * value:.2f}%" if key.endswith("accuracy") else
                f"{value:.1f}" if key == "wall_seconds" else str(value))
            cells.append(f"<td>{html.escape(rendered)}</td>")
        error = html.escape(record.get("error", ""))
        rows.append(f'<tr title="{error}">' + "".join(cells) + "</tr>")
    header = "".join(f"<th>{label}</th>" for _, label in columns)
    top = ranking(summary["trials"])
    best = f"Highest validated token accuracy: {top[0]['id']}." if top else "No validated completion yet."
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="15"><title>Stacked MLP sweep</title>
<style>body{{font:15px system-ui;margin:32px;color:#18202b;background:#fafafa}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:right}}
th{{background:#edf1f5}}p{{max-width:1000px;line-height:1.5}}.scroll{{overflow:auto}}</style>
<h1>Frozen A3 · stacked MLP sweep</h1><p>Status: {html.escape(summary['status'])}. {best}
Updated {summary['updated_utc']}. This page refreshes every 15 seconds.</p>
<p>Each block is pre-LN → 10 → H → 10 with GELU and a residual connection;
the final LN is trainable and the original head is frozen. Parameters = D × (21H + 30) + 20.
One source checkpoint and one seed; depth also increases capacity at fixed width.</p>
<p>Token accuracy includes 1,024 EOS targets among 10,002 scored targets. Non-EOS accuracy excludes EOS.
Running or failed rows show provisional evaluations; an em dash means no measurement.
Exact greedy facts appear only after all completion checks pass. Budget labels compare all trainable
replacement parameters with the original 1,380-parameter suffix.</p>
<div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>Full records: summary.json · summary.tsv. Per-trial logs and native reports are in each trial directory.</p></html>"""
    atomic_write(args.run_dir / "summary.html", page)


def stop_processes(active):
    for child in active.values():
        if child["process"].poll() is None:
            try:
                os.killpg(child["process"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5
    for child in active.values():
        process = child["process"]
        try:
            process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_sweep(args, *, popen=None):
    validate_inputs(args)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summary = {"status": "preparing", "started_utc": now(), "driver_pid": os.getpid(),
               "configuration": {key: str(value) if isinstance(value, Path) else value
                                 for key, value in vars(args).items()},
               "source_shape": SHAPE, "source_tail_parameters": SOURCE_BUDGET,
               "scored_targets": SCORED, "facts": FACTS, "accuracy_units": "fraction",
               "scope": "One frozen source, one seed; no global optimality or depth-only claim.",
               "trials": []}
    for depth in args.depths:
        for width in args.widths:
            parameters = parameter_count(depth, width)
            identifier = f"D{depth}_H{width}"
            record = {"id": identifier, "depth": depth, "width": width,
                      "trainable_parameters": parameters,
                      "budget_relation": budget_relation(parameters), "status": "pending",
                      "directory": str(args.run_dir / identifier),
                      "exact_greedy_facts": None}
            record["command"] = command(args, record)
            summary["trials"].append(record)
    persist(args, summary, started)
    active = {}
    stop = {"signal": None}
    previous_handlers = {}

    def interrupted(number, frame):
        stop["signal"] = number

    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[number] = signal.signal(number, interrupted)
        files = snapshot(args)
        manifest = {"created_utc": now(), "files": files,
                    "commands": {row["id"]: row["command"] for row in summary["trials"]},
                    "environment": {key: os.environ[key] for key in
                                    ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")
                                    if key in os.environ}}
        atomic_write(args.run_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
        atomic_write(args.run_dir / "commands.sh", "#!/usr/bin/env bash\nset -euo pipefail\n" +
                     "\n".join(shlex.join(row["command"]) for row in summary["trials"]) + "\n")
        summary["manifest"] = "manifest.json"
        summary["status"] = "running"
        pending = iter(summary["trials"])
        exhausted = False
        while not exhausted or active:
            if stop["signal"] is not None:
                break
            while len(active) < args.max_workers and not exhausted and stop["signal"] is None:
                record = next(pending, None)
                if record is None:
                    exhausted = True
                    break
                verify_snapshot(args.run_dir / "inputs", files)
                directory = Path(record["directory"])
                directory.mkdir()
                stdout = (directory / "stdout.log").open("w")
                stderr = (directory / "stderr.log").open("w")
                try:
                    process = (popen or subprocess.Popen)(record["command"], stdout=stdout,
                                                          stderr=stderr, start_new_session=True)
                except BaseException:
                    stdout.close()
                    stderr.close()
                    raise
                record.update(status="running", started_utc=now(), pid=process.pid)
                active[record["id"]] = {"record": record, "process": process,
                                        "stdout": stdout, "stderr": stderr,
                                        "started": time.monotonic()}
                print(f"Started {record['id']}: {record['trainable_parameters']} parameters", flush=True)
            for identifier, child in list(active.items()):
                record, process = child["record"], child["process"]
                record["wall_seconds"] = max(0, time.monotonic() - child["started"])
                code = process.poll()
                timed_out = code is None and record["wall_seconds"] >= args.timeout_seconds
                if timed_out:
                    stop_processes({identifier: child})
                    code = process.returncode
                try:
                    record.update(history_metrics(read_history(Path(record["directory"]) / "puzzle/training.tsv")))
                except (OSError, ValueError, TypeError, KeyError) as error:
                    record["partial_history_error"] = str(error)
                if code is None:
                    continue
                child["stdout"].close()
                child["stderr"].close()
                record.update(returncode=code, finished_utc=now())
                if timed_out:
                    record.update(status="timeout", error="per-run wall-clock timeout")
                elif code != 0:
                    record.update(status="failed", error=f"native process exited {code}; see stderr.log")
                else:
                    try:
                        verify_snapshot(args.run_dir / "inputs", files)
                        record.update(validate_result(args, record))
                        record["status"] = "completed"
                    except (OSError, ValueError, TypeError, KeyError) as error:
                        record.update(status="failed", error=f"result validation: {error}")
                del active[identifier]
                print(f"{identifier}: {record['status']}", flush=True)
            persist(args, summary, started)
            if active and stop["signal"] is None:
                time.sleep(1)
        if stop["signal"] is not None:
            summary.update(status="interrupted", signal=stop["signal"])
        else:
            verify_snapshot(args.run_dir / "inputs", files)
            summary["status"] = ("completed" if all(row["status"] == "completed"
                                                   for row in summary["trials"]) else "completed_with_failures")
    except (Exception, KeyboardInterrupt) as error:
        summary.update(status="error", error=f"{type(error).__name__}: {error}")
    finally:
        stop_processes(active)
        for child in active.values():
            child["stdout"].close()
            child["stderr"].close()
            child["record"].update(status="interrupted" if stop["signal"] else "aborted",
                                   returncode=child["process"].returncode, finished_utc=now(),
                                   wall_seconds=max(0, time.monotonic() - child["started"]))
        for record in summary["trials"]:
            if record["status"] == "pending":
                record["status"] = "not_started"
        summary["finished_utc"] = now()
        persist(args, summary, started)
        for number, previous in previous_handlers.items():
            signal.signal(number, previous)
    print(f"Sweep {summary['status']}: {args.run_dir / 'summary.html'}", flush=True)
    return 0 if summary["status"] == "completed" else (
        128 + stop["signal"] if stop["signal"] else 1)


def main(argv=None):
    try:
        return run_sweep(parse_args(argv))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
