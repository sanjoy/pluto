#!/usr/bin/env python3
"""Run fresh attention-graft fits and publish an incremental local report.

The default screen contains all ten single columns, all 45 pairs, and fresh
A3/A4 controls. Coordinates are one-based; model block flags are zero-based.
Each run has its own complete cosine schedule, including short screens.
Only Python's standard library is required; --dry-run never launches the GPU.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
from datetime import datetime, timezone
import hashlib
import html
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time


WIDTH = 10
TSV_FIELDS = (
    "id", "phase", "kind", "columns", "status", "returncode", "accuracy",
    "correct", "scored", "wrong", "complete_facts", "fact_count", "best_step",
    "updates", "training_seconds", "wall_seconds", "error", "log", "checkpoint",
)
RESERVED_FLAGS = {"checkpoint", "tokenizer", "corpus", "output"}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def flag_map(flags):
    """Require unambiguous argv entries; no shell or whitespace interpretation."""
    if not isinstance(flags, list) or any(not isinstance(x, str) for x in flags):
        raise ValueError("variant flags must be a list of strings")
    result = {}
    for flag in flags:
        match = re.fullmatch(r"--([a-zA-Z][a-zA-Z0-9_]*)(?:=(.*))?", flag)
        if not match:
            raise ValueError(f"expected --name=value or --boolean, got {flag!r}")
        name, value = match.groups()
        if name in RESERVED_FLAGS:
            raise ValueError(f"variant cannot override --{name}")
        result[name] = "true" if value is None else value
    return result


def coordinate_variants():
    variants = []
    for count, kind in ((1, "single"), (2, "pair")):
        for columns in combinations(range(1, WIDTH + 1), count):
            variants.append({
                "id": kind + "_" + "_".join(f"{c:02d}" for c in columns),
                "flags": ["--graft_block=3", "--graft_columns=" +
                          ",".join(map(str, columns))],
            })
    return variants


def load_variants(args):
    if args.variants_file:
        variants = json.loads(args.variants_file.read_text())
        if not isinstance(variants, list):
            raise ValueError("variants-file must contain a JSON list")
    else:
        variants = [
            {"id": "a3_baseline", "flags": ["--block=2"]},
            {"id": "a4_baseline", "flags": ["--block=3"]},
            *coordinate_variants(),
        ]
    for spec in args.plane:
        name, separator, path = spec.partition("=")
        if not separator or not name or not path:
            raise ValueError("--plane requires name=path")
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"plane does not exist: {path}")
        variants.append({"id": name, "flags": ["--graft_block=3",
                                               f"--graft_plane={path}"]})
    seen = set()
    for variant in variants:
        if not isinstance(variant, dict) or set(variant) != {"id", "flags"}:
            raise ValueError("each variant must have exactly id and flags")
        name = variant["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError(f"invalid variant id: {name!r}")
        if name in seen:
            raise ValueError(f"duplicate variant id: {name}")
        seen.add(name)
        flag_map(variant["flags"])
    if not variants:
        raise ValueError("at least one variant is required")
    return variants


def build_command(args, variant, directory):
    flags = {
        "checkpoint": str(args.source_checkpoint), "tokenizer": str(args.tokenizer),
        "corpus": str(args.corpus), "output": str(directory / "fit"),
        "block": "2", "fresh_branch": "true", "random_init": str(args.initialization_seed),
        "train_final_norm": "true", "readout_width": "20", "batch_size": "32",
        "objective": "cross_entropy", "steps": str(args.steps),
        "learning_rate": "0.001", "final_rate_ratio": "0.1",
        "eval_every": str(args.eval_every), "seconds": str(args.seconds_per_run),
        "seed": str(args.shuffle_seed),
    }
    flags.update(flag_map(variant["flags"]))
    return [str(args.binary), *(f"--{key}={value}" for key, value in flags.items())]


def fraction(value, name):
    match = re.fullmatch(r"(\d+)/(\d+)", value)
    if not match:
        raise ValueError(f"malformed {name}: {value!r}")
    numerator, denominator = map(int, match.groups())
    if denominator <= 0 or numerator > denominator:
        raise ValueError(f"invalid {name}: {value!r}")
    return numerator, denominator


def parse_record(line, final=False):
    fields = dict(re.findall(r"([a-z_]+)=([^\s]+)", line))
    wrong, scored = fraction(fields["wrong"], "wrong")
    record = {"wrong": wrong, "scored": scored, "correct": scored - wrong,
              "accuracy": (scored - wrong) / scored,
              "training_seconds": float(fields["seconds"])}
    if not math.isfinite(record["training_seconds"]) or record["training_seconds"] < 0:
        raise ValueError("invalid training seconds")
    if final:
        complete, count = fraction(fields["autoregressive_complete"], "autoregressive_complete")
        record.update(best_step=int(fields["best_step"]), updates=int(fields["updates"]),
                      complete_facts=complete, fact_count=count,
                      frozen_weights_unchanged=fields["frozen_weights_unchanged"] == "true")
        if not 0 <= record["best_step"] <= record["updates"]:
            raise ValueError("invalid best_step/updates")
    else:
        failed, count = fraction(fields["failed_facts"], "failed_facts")
        record.update(step=int(fields["step"]), failed_facts=failed, fact_count=count,
                      margin_loss=float(fields["margin_loss"]),
                      min_margin=float(fields["min_margin"]))
        if record["step"] < 0 or not all(math.isfinite(record[key]) for key in ("margin_loss", "min_margin")):
            raise ValueError("invalid evaluation metrics")
    return record


def parse_log(contents):
    evaluations, final = [], None
    for line in contents.splitlines():
        if line.startswith("step="):
            evaluations.append(parse_record(line))
        elif line.startswith("FINAL "):
            final = parse_record(line, final=True)
    return {"evaluations": evaluations, "final": final}


def atomic_text(path, contents):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_variant(record, args, cancel):
    """Stream output to disk; a failed child cannot cancel unrelated trials."""
    directory = args.output / record["id"]
    log = directory / "training.log"
    started = time.monotonic()
    result = {"status": "failed", "finished_utc": None}
    try:
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(record["command"], stdout=stream,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            deadline = started + args.seconds_per_run + 120
            while True:
                try:
                    result["returncode"] = process.wait(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    if cancel.is_set() or time.monotonic() >= deadline:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        result.update(status="cancelled" if cancel.is_set() else "timeout",
                                      returncode=process.returncode,
                                      error="cancelled" if cancel.is_set() else "outer wall-clock cap")
                        break
        parsed = parse_log(log.read_text(encoding="utf-8", errors="replace"))
        write_json(directory / "evaluations.json", parsed["evaluations"])
        if parsed["final"]:
            result.update(parsed["final"])
        if result["status"] in ("timeout", "cancelled"):
            pass
        elif result["returncode"] != 0:
            result["error"] = f"process exited {result['returncode']}"
        elif parsed["final"] is None:
            result["error"] = "successful process emitted no FINAL metrics"
        elif not parsed["final"]["frozen_weights_unchanged"]:
            result["error"] = "frozen source/readout weights changed"
        else:
            result["status"] = "complete"
    except (OSError, ValueError, KeyError) as error:
        result["error"] = str(error)
    result.update(wall_seconds=round(time.monotonic() - started, 3), finished_utc=now())
    write_json(directory / "result.json", {**record, **result})
    return result


def record_kind(command):
    flags = dict(item[2:].split("=", 1) for item in command[1:])
    if flags.get("graft_plane"):
        return "plane", []
    if flags.get("graft_columns"):
        try:
            columns = [int(item) for item in flags["graft_columns"].split(",")]
        except ValueError:
            return "custom", []
        if len(set(columns)) == len(columns) and all(1 <= n <= WIDTH for n in columns):
            return {1: "single", 2: "pair"}.get(len(columns), "columns"), columns
    return "control", []


def report_html(manifest):
    records = manifest["runs"]
    completed = [r for r in records if r["status"] == "complete"]
    baseline = next((r for r in completed if r["id"] == "a3_baseline"), None)
    ordered = sorted(records, key=lambda r: (r["status"] != "complete",
                     -r.get("accuracy", -1), -r.get("complete_facts", -1), r["id"]))
    escape = lambda value: html.escape(str(value), quote=True)
    rows = []
    for index, record in enumerate(ordered, 1):
        valid = record["status"] == "complete"
        accuracy = f"{record['accuracy']:.4%}" if valid else "—"
        delta = f"{100 * (record['accuracy'] - baseline['accuracy']):+.3f}" if valid and baseline else "—"
        facts = f"{record['complete_facts']}/{record['fact_count']}" if valid else "—"
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in (
            str(index) if valid else "—", f'<a href="{escape(record["log"])}">{escape(record["id"])}</a>',
            escape(record["kind"]), escape(record["status"]), accuracy, delta, facts,
            escape(record.get("best_step", "—")), escape(record.get("updates", "—")),
            escape(record.get("error", "")))) + "</tr>")
    cells = {}
    for record in completed:
        columns = record["columns"]
        if record["kind"] == "single":
            cells[(columns[0], columns[0])] = record
        elif record["kind"] == "pair":
            a, b = columns
            cells[(a, b)] = cells[(b, a)] = record
    minimum = min((r["accuracy"] for r in cells.values()), default=0)
    maximum = max((r["accuracy"] for r in cells.values()), default=1)
    heat = ["<tr><th>A4 column</th>" + "".join(f"<th>{n}</th>" for n in range(1, WIDTH + 1)) + "</tr>"]
    for a in range(1, WIDTH + 1):
        row = [f"<tr><th>{a}</th>"]
        for b in range(1, WIDTH + 1):
            record = cells.get((a, b))
            if record:
                level = (record["accuracy"] - minimum) / max(maximum - minimum, 1e-12)
                color = f"hsl(170 52% {94 - 58 * level:.1f}%)"
                label = f"{record['accuracy']:.2%}"
                title = f"{record['id']}: {record['complete_facts']}/{record['fact_count']} exact facts"
                row.append(f'<td style="background:{color}" title="{escape(title)}"><a href="{escape(record["log"])}">{label}</a></td>')
            else:
                row.append('<td class="pending">—</td>')
        heat.append("".join(row) + "</tr>")
    config = manifest["configuration"]
    phase_note = ("Screening results rank short independent fits; they do not establish performance at the full 120,000-step budget."
                  if config["phase"] == "screen" else
                  "Full-budget cohort results apply to the tested configurations, initialization, and corpus.")
    return """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Attention graft sweep</title><style>
body{font:15px system-ui,sans-serif;background:#f5f7fa;color:#172b36;margin:32px auto;padding:0 24px;max-width:1400px}
h1{margin-bottom:8px}p{line-height:1.55;max-width:1050px}.note{padding:16px;background:#fff1cc;border-radius:8px}
table{border-collapse:collapse;background:white;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:10px 8px;border:1px solid #dce3e8}th{background:#e9eff4}td:nth-child(2){text-align:left}
a{color:inherit}.scroll{overflow:auto;margin:24px 0}.heat td{text-align:center;min-width:60px}.pending{color:#82909a}
small{color:#536774}code{background:#e9eff4;padding:2px 4px}h2{margin-top:32px}
</style><h1>Attention graft sweep</h1>""" + (
        f'<p>{escape(config["phase"].title())} · {len(completed)}/{len(records)} complete · '
        f'{escape(manifest["status"])} · updated {escape(manifest["updated_utc"])}</p>'
        f'<p class="note">{escape(phase_note)} Every default fit starts fresh at A3, with a width-20 readout and trainable final LayerNorm. '
        f'The defaults use {config["steps"]:,} updates and an independent cosine learning-rate schedule from 0.001 to 0.0001. '
        'Variant overrides are recorded in the manifest. Token accuracy uses the scored suffix-plus-EOS positions; '
        'exact facts use autoregressive completions.</p>'
        '<p><a href="manifest.json">Manifest and exact commands</a> · <a href="results.tsv">Results TSV</a>. '
        'Refresh this page as runs complete. Failed or incomplete runs are excluded from rankings and colors.</p>'
        '<h2>Single columns and pairs</h2><p>One-based A4 donor columns. Diagonal cells are single-column grafts; '
        'off-diagonal cells are pair grafts into A3. Colors span completed coordinate results in this cohort. '
        'Cell labels show token accuracy; hover for exact completions.</p><div class="scroll"><table class="heat">' +
        "".join(heat) + '</table></div><h2>Ranking</h2><div class="scroll"><table><tr>' +
        "".join(f"<th>{x}</th>" for x in ("Rank", "Variant / log", "Kind", "Status", "Token accuracy", "Δ A3 (pp)", "Exact facts", "Best step", "Updates", "Error")) +
        "</tr>" + "".join(rows) + '</table></div><small>Each run directory contains its command, streamed training log, evaluation history, result, and fitted checkpoint.</small></html>')


def persist(args, manifest):
    manifest["updated_utc"] = now()
    write_json(args.output / "manifest.json", manifest)
    temporary = args.output / "results.tsv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TSV_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for record in manifest["runs"]:
            writer.writerow({**record, "columns": ",".join(map(str, record["columns"]))})
    temporary.replace(args.output / "results.tsv")
    atomic_text(args.output / "report.html", report_html(manifest))


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def run_sweep(args):
    variants = load_variants(args)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError(f"output directory must be new: {args.output}")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        raise ValueError("binary must be an executable file")
    if not args.corpus.is_file() or not args.source_checkpoint.is_dir() or not args.tokenizer.is_dir():
        raise ValueError("corpus must be a file; source-checkpoint and tokenizer must be directories")
    args.output.mkdir(parents=True)
    manifest = {"status": "planned" if args.dry_run else "running", "started_utc": now(),
                "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "schedule_note": "Each fit independently decays over its specified steps; screens are not prefixes of full schedules.",
                "runs": []}
    inputs = {args.binary, args.corpus}
    inputs.update(path for path in args.source_checkpoint.iterdir() if path.is_file())
    inputs.update(path for path in args.tokenizer.iterdir() if path.is_file())
    for variant in variants:
        directory = args.output / variant["id"]
        directory.mkdir()
        command = build_command(args, variant, directory)
        kind, columns = record_kind(command)
        plane = flag_map(variant["flags"]).get("graft_plane")
        if plane:
            inputs.add(Path(plane).expanduser().resolve())
        record = {"id": variant["id"], "phase": args.phase, "kind": kind, "columns": columns,
                  "flags": variant["flags"], "command": command, "command_shell": shlex.join(command),
                  "status": "planned" if args.dry_run else "queued", "log": f"{variant['id']}/training.log",
                  "checkpoint": f"{variant['id']}/fit/best_mlp"}
        manifest["runs"].append(record)
        write_json(directory / "command.json", command)
    manifest["input_sha256"] = {str(path): digest(path) for path in sorted(inputs)}
    persist(args, manifest)
    if args.dry_run:
        return manifest
    cancel = threading.Event()
    executor = ThreadPoolExecutor(max_workers=args.jobs)
    pending = iter(manifest["runs"])
    futures = {}

    def submit():
        record = next(pending, None)
        if record is not None:
            record.update(status="running", started_utc=now())
            futures[executor.submit(run_variant, record, args, cancel)] = record
        return record is not None

    try:
        for _ in range(args.jobs):
            if not submit():
                break
        persist(args, manifest)
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                record = futures.pop(future)
                try:
                    record.update(future.result())
                except Exception as error:
                    record.update(status="failed", error=f"worker error: {error}")
                print(f"{record['id']}: {record['status']}" +
                      (f" accuracy={record['accuracy']:.4%}" if "accuracy" in record else ""), flush=True)
                submit()
            persist(args, manifest)
        manifest["status"] = "complete" if all(r["status"] == "complete" for r in manifest["runs"]) else "complete_with_failures"
    except KeyboardInterrupt:
        cancel.set()
        manifest["status"] = "cancelled"
        for future, record in futures.items():
            record.update(future.result())
        for record in pending:
            record.update(status="cancelled", error="not started before cancellation")
    finally:
        executor.shutdown(wait=True)
        manifest["finished_utc"] = now()
        persist(args, manifest)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "source-checkpoint", "tokenizer", "corpus", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--phase", choices=("screen", "full"), default="screen")
    parser.add_argument("--steps", type=int, default=None, help="default: 20000 for screen, 120000 for full")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--seconds-per-run", type=float, default=1800)
    parser.add_argument("--initialization-seed", type=int, default=3)
    parser.add_argument("--shuffle-seed", type=int, default=3)
    parser.add_argument("--plane", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--variants-file", type=Path, help="JSON list of {id, flags}; replaces the default cohort")
    parser.add_argument("--dry-run", action="store_true", help="write commands/reports without launching any fits")
    args = parser.parse_args(argv)
    if args.steps is None:
        args.steps = 20000 if args.phase == "screen" else 120000
    if args.steps < 0 or args.jobs < 1 or args.eval_every < 1:
        parser.error("steps must be nonnegative; jobs and eval-every must be positive")
    if not math.isfinite(args.seconds_per_run) or args.seconds_per_run <= 0:
        parser.error("seconds-per-run must be finite and positive")
    if not 0 <= args.initialization_seed < 2**31 or not 0 <= args.shuffle_seed < 2**31:
        parser.error("seeds must be nonnegative int32 values")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    return args


def main():
    args = parse_args()
    try:
        manifest = run_sweep(args)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"Report: {args.output / 'report.html'}")
    return 0 if manifest["status"] in ("planned", "complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
