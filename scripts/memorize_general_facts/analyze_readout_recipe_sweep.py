#!/usr/bin/env python3
"""Summarize completed frozen-attention readout fits without running the GPU.

Accepts a root containing any number of run_attention_graft_sweep.py cohorts.
Short screens and longer confirmations remain separate, including their actual
update counts. Replays are excluded: evaluating a saved checkpoint is not an
independent training experiment. The generated HTML has no external assets.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import re
import shlex


# Defaults match fit_attention_readout's CLI. Only parameters in the residual
# branch and optional final LayerNorm are trained; the vocabulary head is frozen.
DEFAULT_FLAGS = {
    "model_width": "10", "feed_forward_width": "20", "readout_width": "0",
    "train_final_norm": "false", "seed": "0", "random_init": "-1",
    "batch_size": "32", "objective": "squared_margin", "margin": "0.1",
    "learning_rate": "0.001", "final_rate_ratio": "0.1",
    "input_init_std": "0.2", "output_init_std": "0.1",
    "scale_output_init": "false", "fresh_final_norm": "false",
    "preprocessing": "identity", "preprocessing_scale": "1",
    "preprocessing_seed": "0",
}
PATH_FLAGS = {"checkpoint", "tokenizer", "corpus", "output",
              "readout_checkpoint", "branch_checkpoint"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def command_flags(command):
    """Parse saved argv, including --flag=value and --flag value spellings."""
    if not isinstance(command, list) or not command:
        raise ValueError("missing command argv")
    result = {}
    index = 1
    while index < len(command):
        item = command[index]
        if not isinstance(item, str) or not item.startswith("--"):
            raise ValueError(f"unexpected argv entry: {item!r}")
        name, separator, value = item[2:].partition("=")
        if not separator:
            if index + 1 < len(command) and not command[index + 1].startswith("--"):
                index += 1
                value = command[index]
            else:
                value = "true"
        if not name or name in result:
            raise ValueError(f"empty or duplicate flag: {name}")
        result[name] = value
        index += 1
    return result


def finite_number(value, name, minimum=0):
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"invalid {name}: {value!r}")
    return result


def integer(value, name, minimum=0):
    result = int(value)
    if str(result) != str(value) or result < minimum:
        raise ValueError(f"invalid {name}: {value!r}")
    return result


def log_metadata(directory, record):
    """Read newer native diagnostics that older sweep drivers do not retain.

    CE must come from the FINAL line (the selected checkpoint), never the last
    ordinary evaluation, which can describe a different checkpoint. Collisions
    count distinct transformed vectors associated with contradictory targets.
    """
    path = directory / "training.log"
    final, collision = {}, {}
    loss_evaluations = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("FINAL "):
                final = dict(re.findall(r"([a-z_]+)=([^\s]+)", line))
            elif line.startswith("preprocessed_unique_vectors="):
                collision = dict(re.findall(r"([a-z_]+)=([^\s]+)", line))
            elif line.startswith("step="):
                fields = dict(re.findall(r"([a-z_]+)=([^\s]+)", line))
                if "cross_entropy" in fields:
                    step = integer(fields["step"], "CE evaluation step")
                    if step > record["updates"] or (loss_evaluations and step < loss_evaluations[-1]["step"]):
                        raise ValueError("inconsistent CE evaluation series")
                    loss_evaluations.append({
                        "step": step,
                        "cross_entropy": finite_number(fields["cross_entropy"], "evaluation CE"),
                    })
    metadata = {}
    if loss_evaluations:
        metadata["cross_entropy_evaluations"] = loss_evaluations
    cross_entropy = record.get("cross_entropy", final.get("cross_entropy"))
    if cross_entropy is not None:
        metadata["cross_entropy"] = finite_number(cross_entropy, "selected checkpoint CE")
        if "best_step" in final and int(final["best_step"]) != record["best_step"]:
            raise ValueError("FINAL CE checkpoint differs from recorded best checkpoint")
        if "wrong" in final and final["wrong"] != f'{record["wrong"]}/{record["scored"]}':
            raise ValueError("FINAL CE checkpoint accuracy differs from recorded accuracy")
    if collision:
        keys = ("preprocessed_unique_vectors", "scored_vectors", "conflicting_vectors")
        counts = {key: integer(collision[key], key) for key in keys}
        unique, scored, conflicts = (counts[key] for key in keys)
        if scored != record["scored"] or unique > scored or conflicts > unique:
            raise ValueError("inconsistent preprocessed-vector collision counts")
        counts["duplicate_vector_rows"] = scored - unique
        metadata["collisions"] = counts
    return metadata


def normalize_record(root, path, record):
    """Check metrics and retain the complete command for reproducibility."""
    command = record.get("command")
    if command is None and (path.parent / "command.json").is_file():
        saved = read_json(path.parent / "command.json")
        command = saved.get("command", saved.get("argv")) if isinstance(saved, dict) else saved
    explicit = command_flags(command)
    flags = DEFAULT_FLAGS | explicit
    planned = integer(explicit.get("steps", 0), "planned steps")
    if planned == 0:
        return None
    if record.get("returncode", 0) != 0 or not record.get("frozen_weights_unchanged", False):
        raise ValueError("run did not finish successfully with frozen source/head")
    width = integer(flags["model_width"], "model width", 1)
    hidden = integer(flags["readout_width"], "readout width")
    hidden = hidden or integer(flags["feed_forward_width"], "feed-forward width", 1)
    train_final = flags["train_final_norm"].lower() in ("true", "1", "t")
    params = 2 * width * hidden + hidden + 3 * width + (2 * width if train_final else 0)
    scored = integer(record["scored"], "scored tokens", 1)
    wrong = integer(record["wrong"], "wrong tokens")
    facts = integer(record["complete_facts"], "complete facts")
    fact_count = integer(record["fact_count"], "fact count", 1)
    updates = integer(record["updates"], "updates")
    best_step = integer(record["best_step"], "best step")
    if wrong > scored or facts > fact_count or best_step > updates or updates > planned:
        raise ValueError("inconsistent counts or update budget")
    correct = scored - wrong
    accuracy = correct / scored
    if "accuracy" in record and not math.isclose(float(record["accuracy"]), accuracy, abs_tol=1e-12):
        raise ValueError("saved accuracy disagrees with wrong/scored counts")
    if "correct" in record and record["correct"] != correct:
        raise ValueError("saved correct count disagrees with wrong/scored counts")
    evaluations_path = path.parent / "evaluations.json"
    evaluations = []
    if evaluations_path.is_file():
        previous_step = -1
        for point in read_json(evaluations_path):
            step = integer(point["step"], "evaluation step")
            point_wrong = integer(point["wrong"], "evaluation wrong")
            point_scored = integer(point["scored"], "evaluation scored", 1)
            if point_wrong > point_scored or step < previous_step or step > updates:
                raise ValueError("inconsistent evaluation series")
            previous_step = step
            evaluations.append({"step": step, "accuracy": 1 - point_wrong / point_scored})
    relative = path.parent.relative_to(root)
    return {
        "id": record.get("id", path.parent.name), "directory": str(relative),
        "cohort": str(relative.parent), "phase": record.get("phase", "unspecified"),
        "planned_steps": planned, "updates": updates, "best_step": best_step,
        "completed_budget": updates == planned, "model_width": width,
        "readout_width": hidden, "trainable_parameters": params,
        "train_final_norm": train_final, "correct": correct, "scored": scored,
        "wrong": wrong, "accuracy": accuracy, "complete_facts": facts,
        "fact_count": fact_count, "training_seconds": finite_number(
            record["training_seconds"], "training seconds"),
        "wall_seconds": finite_number(record.get("wall_seconds", record["training_seconds"]),
                                      "wall seconds"),
        "flags": flags, "explicit_flags": explicit, "command": command,
        "evaluations": evaluations,
        **log_metadata(path.parent, record),
    }


def collect(root):
    root = root.resolve()
    records, warnings, skipped = [], [], Counter()
    diagnostic_warnings = []
    for path in sorted(root.rglob("result.json")):
        relative_parts = path.relative_to(root).parts[:-1]
        if any(part == "verify_saved" or "replay" in part.lower() for part in relative_parts):
            skipped["saved-checkpoint replay"] += 1
            continue
        try:
            record = read_json(path)
            if record.get("status") != "complete":
                skipped["not complete"] += 1
                continue
            normalized = normalize_record(root, path, record)
            if normalized is None:
                skipped["zero-update evaluation"] += 1
            else:
                records.append(normalized)
                collisions = normalized.get("collisions", {})
                if collisions.get("duplicate_vector_rows", 0):
                    diagnostic_warnings.append(
                        f'{normalized["directory"]}: preprocessing leaves '
                        f'{collisions["preprocessed_unique_vectors"]} distinct vectors '
                        f'for {collisions["scored_vectors"]} scored rows '
                        f'({collisions["duplicate_vector_rows"]} duplicate rows); '
                        f'{collisions["conflicting_vectors"]} vectors have contradictory targets.')
        except (OSError, ValueError, KeyError, TypeError) as error:
            warnings.append(f"{path.relative_to(root)}: {error}")
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root), "records": records, "warnings": warnings,
        "diagnostic_warnings": diagnostic_warnings,
        "skipped": dict(sorted(skipped.items())),
        "interpretation": {
            "token_accuracy": "Teacher-forced suffix-plus-EOS top-1 accuracy at the saved best checkpoint.",
            "complete_facts": "Greedy complete suffix-plus-EOS generations from five prompt tokens.",
            "cross_entropy": "Mean cross-entropy of the selected best checkpoint, when emitted by the native run; absent for older binaries. This need not be the lowest-CE checkpoint.",
            "collisions": "Exact BF16 preprocessing-vector audit: unique / scored counts and number of distinct vectors with contradictory target labels. Repeated vectors without target conflicts do not alone make perfect classification impossible.",
            "parameters": "Trainable residual MLP plus input LayerNorm and optionally final LayerNorm; frozen source and tied vocabulary head excluded.",
            "selection": "Best checkpoint within each run, not a held-out estimate; screen and confirmation results are not pooled.",
            "timing": "Concurrent jobs share the GPU; reported elapsed seconds are not isolated throughput benchmarks.",
        },
    }


def escape(value):
    return html.escape(str(value), quote=True)


def flag_description(record, names, default="—"):
    flags = record["flags"]
    return ", ".join(f"{key}={flags[key]}" for key in names if key in flags) or default


def curve_svg(records, metric="accuracy"):
    """Inline top-eight curves within a single cohort and step budget.

    The CE graph reads only explicitly named cross_entropy fields: the native
    margin_loss is a different measurement and is never relabeled as CE.
    """
    if metric not in ("accuracy", "cross_entropy"):
        raise ValueError("unknown curve metric")
    series_key = "evaluations" if metric == "accuracy" else "cross_entropy_evaluations"
    selected = [r for r in records if r.get(series_key)][:8]
    if not selected:
        return ""
    colors = ("#0066a8", "#c64225", "#2a8643", "#9653a0", "#ab6b00", "#15888b", "#5156bf", "#6b7075")
    width, height, left, top, plot_width, plot_height = 920, 410, 65, 25, 620, 330
    max_step = max(r["planned_steps"] for r in selected)
    max_value = 1 if metric == "accuracy" else max(
        1, math.ceil(max(p[metric] for r in selected for p in r[series_key])))
    title = "Teacher-forced token accuracy" if metric == "accuracy" else "Mean cross-entropy, nats per scored token"
    content = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title} by optimizer update">',
               '<rect width="100%" height="100%" fill="white"/>']
    for fraction in (0, .2, .4, .6, .8, 1):
        y = top + plot_height * (1 - fraction)
        label = f"{fraction:.0%}" if metric == "accuracy" else f"{max_value * fraction:.2f}"
        content.append(f'<path d="M{left},{y}h{plot_width}" stroke="#e1e6ec"/>')
        content.append(f'<text x="{left - 8}" y="{y + 4}" text-anchor="end">{label}</text>')
    for fraction in (0, .25, .5, .75, 1):
        x = left + plot_width * fraction
        content.append(f'<text x="{x}" y="{top + plot_height + 25}" text-anchor="middle">{max_step * fraction:,.0f}</text>')
    content.append(f'<text x="{left + plot_width / 2}" y="{height - 10}" text-anchor="middle">Optimizer updates</text>')
    for index, record in enumerate(selected):
        color = colors[index]
        points = " ".join(f'{left + p["step"] / max_step * plot_width:.2f},{top + (1 - p[metric] / max_value) * plot_height:.2f}'
                          for p in record[series_key])
        content.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"><title>{escape(record["id"])}</title></polyline>')
        y = top + index * 38
        label = record["id"] if len(record["id"]) <= 29 else record["id"][:26] + "…"
        content.append(f'<text x="{left + plot_width + 15}" y="{y + 12}" fill="{color}"><title>{escape(record["id"])}</title>{escape(label)}</text>')
    return "".join(content) + "</svg>"


def report_html(summary):
    groups = defaultdict(list)
    for record in summary["records"]:
        groups[(record["cohort"], record["phase"], record["planned_steps"])].append(record)
    sections = []
    for (cohort, phase, budget), records in sorted(groups.items()):
        records.sort(key=lambda r: (-r["accuracy"], -r["complete_facts"], r["id"]))
        rows = []
        for record in records:
            flags = record["flags"]
            preprocess = flag_description(record, sorted(k for k in flags if any(
                word in k for word in ("preprocess", "fourier", "frequency", "transform"))), "none")
            initialization = flag_description(record, sorted(k for k in flags if any(
                word in k for word in ("init", "stddev")) or k == "fresh_final_norm"))
            recipe = flag_description(record, ("objective", "margin", "batch_size", "learning_rate",
                                               "final_rate_ratio", "seed"))
            architecture = f'{record["model_width"]} → {record["readout_width"]} → {record["model_width"]}'
            cross_entropy = f'{record["cross_entropy"]:.5f}' if "cross_entropy" in record else "—"
            collisions = record.get("collisions")
            collision_text = (f'{collisions["preprocessed_unique_vectors"]:,}/{collisions["scored_vectors"]:,}; '
                              f'{collisions["conflicting_vectors"]} conflicts') if collisions else "not recorded"
            cells = [escape(record["id"]), escape(architecture), f'{record["trainable_parameters"]:,}',
                     escape(initialization), escape(recipe), escape(preprocess),
                     f'<strong>{record["accuracy"]:.3%}</strong><br>{record["correct"]:,}/{record["scored"]:,}',
                     cross_entropy,
                     f'{record["complete_facts"] / record["fact_count"]:.2%}<br>{record["complete_facts"]}/{record["fact_count"]}',
                     escape(collision_text),
                     f'{record["updates"]:,}/{budget:,}' + (" <b>shortened</b>" if not record["completed_budget"] else ""),
                     f'{record["best_step"]:,}', f'{record["training_seconds"]:.1f}s / {record["wall_seconds"]:.1f}s']
            details = escape(shlex.join(record["command"]))
            rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>" +
                        f'<tr class="command"><td colspan="13"><details><summary>Exact command and checkpoint location</summary><pre>{details}</pre><p>{escape(record["directory"])}/fit/best_mlp</p></details></td></tr>')
        headers = ("Run", "Architecture", "Trainable params", "Initialization", "Recipe", "Preprocessing",
                   "Teacher-forced token top-1", "Selected checkpoint CE", "Greedy exact facts", "Unique/scored vectors; conflicts",
                   "Actual / planned updates", "Best update", "Train / wall time")
        accuracy_curve = curve_svg(records)
        ce_curve = curve_svg(records, "cross_entropy")
        curves = ('<h3>Teacher-forced next-token accuracy (higher is better)</h3>' + accuracy_curve) if accuracy_curve else ""
        if ce_curve:
            curves += '<h3>Mean cross-entropy, nats per scored token (lower is better)</h3>' + ce_curve
        sections.append(f'<section><h2>{escape(cohort)} — {escape(phase)}, {budget:,} planned updates</h2>' +
                        curves + '<div class="scroll"><table><thead><tr>' +
                        "".join(f"<th>{header}</th>" for header in headers) +
                        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div></section>')
    warning_html = ""
    if summary["warnings"]:
        warning_html = '<aside><h2>Excluded invalid records</h2><ul>' + "".join(
            f'<li>{escape(warning)}</li>' for warning in summary["warnings"]) + '</ul></aside>'
    if summary["diagnostic_warnings"]:
        warning_html += '<aside><h2>Preprocessing collision diagnostics</h2><ul>' + "".join(
            f'<li>{escape(warning)}</li>' for warning in summary["diagnostic_warnings"]) + '</ul></aside>'
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Frozen A3 readout recipe and preprocessing experiments</title>
<style>body{font:15px system-ui,sans-serif;color:#202b39;background:#f6f8fb;margin:0;padding:26px}main{max-width:1800px;margin:auto}h1{font-size:28px}section,aside{background:white;border:1px solid #d9e1e8;border-radius:10px;margin:24px 0;padding:20px}p,li{line-height:1.6}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #dbe2e8}th{background:#eaf0f6}td:nth-child(5){min-width:210px}td:nth-child(4){min-width:160px}.command td{padding:2px 10px 12px;color:#59697a}pre{white-space:pre-wrap;overflow-wrap:anywhere}svg{width:100%;max-width:1000px;font:12px system-ui}aside{border-color:#d59119}code{overflow-wrap:anywhere}</style><main>
<h1>Frozen A3 readout: seeds, training recipes, and per-token preprocessing</h1>''' + \
        f'<p>{len(summary["records"])} completed training runs; generated {escape(summary["generated_utc"])}.</p>' + \
        '<p>The requested architecture is <strong>10 → 150 → 10</strong>: <strong>3,200 trainable parameters</strong> when both LayerNorms are trained. Actual dimensions and counts are shown for every run; controls with different settings are not relabeled.</p>' + \
        '<ul>' + "".join(f'<li>{escape(text)}</li>' for text in summary["interpretation"].values()) + '</ul>' + \
        '<p>Each table is a separate cohort and planned training budget. Longer schedules, different seeds, and continued fits are not independent confirmations of a short-screen ranking. A shortened run is marked explicitly. Curves show up to eight best runs in that table, not a pooled leaderboard.</p>' + \
        f'<p>Excluded records: {escape(summary["skipped"])}. Source root: <code>{escape(summary["root"])}</code>.</p>' + \
        warning_html + "".join(sections) + ('<p>No completed training runs yet.</p>' if not sections else '') + '</main></html>\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Root containing one or more sweep cohort directories")
    parser.add_argument("--output", type=Path, help="Output directory; defaults to root")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"input root does not exist: {root}")
    output = (args.output or root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = collect(root)
    for name, contents in (("summary.json", json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"),
                           ("summary.html", report_html(summary))):
        destination = output / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(contents, encoding="utf-8")
        temporary.replace(destination)
    print(f'{len(summary["records"])} completed runs; {len(summary["warnings"])} warnings: {output / "summary.html"}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
