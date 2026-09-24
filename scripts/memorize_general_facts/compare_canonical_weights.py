#!/usr/bin/env python3
"""Audit exact weight equivariance after canonical vocabulary reductions.

Compare checkpoints from all available pairs, including pairs of renamed runs.
Embedding rows are aligned as E_aligned[old] = E_renamed[permutation[old]];
every other parameter stays in its original position. Read uint32 bit patterns,
not rounded decimal values, so a zero difference really means exact FP32 bits.
Incomplete checkpoints are reported as unavailable, allowing progress reports
without interrupting training. NumPy is the only third-party dependency.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import html
import itertools
import json
from pathlib import Path

import numpy as np

from analyze_dataset_weights import tensor_layout


DEFAULT_STEPS = (0, 30000, 60000, 90000, 120000)


def read_checkpoint_bits(path, layout):
    """Require exactly the expected files and sizes, preserving every bit."""
    path = Path(path)
    expected = {f"weight_{index}.bin" for index in range(len(layout))}
    if {p.name for p in path.glob("weight_*.bin")} != expected:
        raise ValueError(f"{path}: incomplete or unexpected weight files")
    files = [path / f"weight_{index}.bin" for index in range(len(layout))]
    before = [(file.stat().st_size, file.stat().st_mtime_ns) for file in files]
    parts = []
    for index, tensor in enumerate(layout):
        data = (path / f"weight_{index}.bin").read_bytes()
        if len(data) != 4 * (tensor.stop - tensor.start):
            raise ValueError(f"{path}/weight_{index}.bin: wrong FP32 size")
        parts.append(np.frombuffer(data, dtype="<u4"))
    after = [(file.stat().st_size, file.stat().st_mtime_ns) for file in files]
    if before != after:
        raise ValueError(f"{path}: checkpoint changed while being read")
    return np.concatenate(parts)


def read_permutation(path, vocabulary_size):
    """Read a full old-ID to new-ID bijection; the final EOS ID stays fixed."""
    values = np.array([int(x) for x in Path(path).read_text().split()], dtype=np.int64)
    if (len(values) != vocabulary_size or not np.array_equal(
            np.sort(values), np.arange(vocabulary_size))):
        raise ValueError("token permutation must be a vocabulary bijection")
    if values[-1] != vocabulary_size - 1:
        raise ValueError("token permutation must preserve the final EOS ID")
    return values


def align_bits(bits, permutation, layout):
    """Undo token renaming without changing float precision or other tensors."""
    result = bits.copy()
    embedding = layout[0]
    result[:embedding.stop] = bits[:embedding.stop].reshape(
        embedding.shape)[permutation].reshape(-1)
    return result


def digest(bits):
    return hashlib.sha256(np.asarray(bits, dtype="<u4").tobytes()).hexdigest()


def differences(left, right):
    """Separate bit differences, numerical changes, signed zeros and nonfinite.

    Identical NaN payloads have identical bits but are not valid finite weights.
    NaNs count as numerical mismatches even when their payloads match. Distance
    metrics include only pairs of finite values; nonfinite counts are explicit.
    """
    left, right = np.asarray(left, dtype="<u4"), np.asarray(right, dtype="<u4")
    if left.shape != right.shape:
        raise ValueError("comparison requires matching parameter shapes")
    a, b = left.view("<f4"), right.view("<f4")
    different = left != right
    finite = np.isfinite(a) & np.isfinite(b)
    delta = b[finite].astype(np.float64) - a[finite].astype(np.float64)
    finite_weights = bool(finite.all())
    bit_count = int(np.count_nonzero(different))
    return {
        "parameters": int(left.size),
        "bitwise_mismatches": bit_count,
        "bitwise_identical": bit_count == 0,
        "numerical_mismatches": int(np.count_nonzero(a != b)),
        "signed_zero_mismatches": int(np.count_nonzero(different & (a == 0) & (b == 0))),
        "left_nan_count": int(np.count_nonzero(np.isnan(a))),
        "right_nan_count": int(np.count_nonzero(np.isnan(b))),
        "left_infinity_count": int(np.count_nonzero(np.isinf(a))),
        "right_infinity_count": int(np.count_nonzero(np.isinf(b))),
        "finite_weights": finite_weights,
        "exact_finite_match": finite_weights and bit_count == 0,
        "max_abs_finite": float(np.max(np.abs(delta))) if delta.size else None,
        "l2_finite": float(np.linalg.norm(delta)) if delta.size else None,
    }


def compare_pair(left, right, layout, *, step, kind):
    """Report raw and aligned differences plus bounded coordinate examples."""
    result = {"step": step, "kind": kind, "left": left["id"], "right": right["id"],
              "raw": differences(left["raw"], right["raw"]),
              "aligned": differences(left["aligned"], right["aligned"]),
              "tensors": []}
    embedding = layout[0]
    result["raw_embedding_rows_changed"] = int(np.count_nonzero(np.any(
        left["raw"][:embedding.stop].reshape(embedding.shape) !=
        right["raw"][:embedding.stop].reshape(embedding.shape), axis=1)))
    result["nonembedding"] = differences(left["raw"][embedding.stop:],
                                         right["raw"][embedding.stop:])
    examples = []
    for tensor in layout:
        section = slice(tensor.start, tensor.stop)
        raw = differences(left["raw"][section], right["raw"][section])
        aligned = differences(left["aligned"][section], right["aligned"][section])
        result["tensors"].append({"name": tensor.name, "shape": tensor.shape,
                                  "raw": raw, "aligned": aligned})
        indices = np.flatnonzero(left["aligned"][section] != right["aligned"][section])
        for index in indices[:max(0, 32 - len(examples))]:
            offset = tensor.start + index
            examples.append({"tensor": tensor.name,
                             "coordinate": [int(x) for x in np.unravel_index(index, tensor.shape)],
                             "left_bits": f"0x{int(left['aligned'][offset]):08x}",
                             "right_bits": f"0x{int(right['aligned'][offset]):08x}"})
    result["first_aligned_mismatch_coordinates"] = examples
    result["only_expected_embedding_relabeling"] = result["aligned"]["exact_finite_match"]
    return result


def _resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


def _checkpoint_path(root, trial, layers, step):
    declared = trial.get("checkpoints", {}).get(str(step))
    if step == 0:
        declared = trial.get("initial_checkpoint", declared)
    return (_resolve(root, declared) if declared else
            root / trial["id"] / "checkpoints" / f"layers_{layers}" / f"step_{step}")


def _load_available(root, summary, layout, steps):
    """Validate declared renamings and collect only complete snapshots."""
    vocabulary = int(summary["model"]["vocabulary_size"])
    baseline = next((trial for trial in summary["trials"] if trial["id"] == "baseline"), None)
    if baseline is None:
        raise ValueError("baseline trial is required")
    base_rows = [list(map(int, line.split())) for line in _resolve(
        root, baseline["token_corpus"]).read_text().splitlines() if line.strip()]
    if not base_rows or any(x < 0 or x >= vocabulary for row in base_rows for x in row):
        raise ValueError("baseline tokens must lie inside the vocabulary")
    loaded = {step: [] for step in steps}
    unavailable, controls, seen = [], [], set()
    for trial in summary["trials"]:
        trial_id = trial["id"]
        if trial_id in seen or Path(trial_id).name != trial_id or trial_id in (".", ".."):
            raise ValueError("trial IDs must be unique directory names")
        seen.add(trial_id)
        if (not summary.get("training_fingerprint") or trial.get("training_fingerprint") !=
                summary["training_fingerprint"]):
            raise ValueError(f"{trial_id}: mismatched training fingerprint")
        canonical_order = bool(summary.get("canonical_token_order", False))
        if bool(trial.get("canonical_token_order", False)) != canonical_order:
            raise ValueError(f"{trial_id}: mixed canonical-order configurations")
        permutation_path = _resolve(root, trial["permutation"])
        permutation = read_permutation(permutation_path, vocabulary)
        rows = [list(map(int, line.split())) for line in _resolve(
            root, trial["token_corpus"]).read_text().splitlines() if line.strip()]
        if rows != [[int(permutation[x]) for x in row] for row in base_rows]:
            raise ValueError(f"{trial_id}: token corpus does not match its permutation")
        if trial_id == "baseline" and not np.array_equal(permutation, np.arange(vocabulary)):
            raise ValueError("baseline permutation must be identity")
        command_path = root / trial_id / "command.json"
        command = json.loads(command_path.read_text()) if command_path.exists() else []
        order_flags = [arg.partition("=")[2] for arg in command
                       if arg.startswith("--token_order_file=")]
        command_verified = (len(order_flags) == 1 and
                            _resolve(root, order_flags[0]).resolve() == permutation_path.resolve())
        if command_path.exists() and canonical_order and not command_verified:
            raise ValueError(f"{trial_id}: recorded command does not use its canonical token order")
        if order_flags and not canonical_order:
            raise ValueError(f"{trial_id}: undeclared canonical token order in command")
        control = {"id": trial_id, "status": trial.get("status"),
                   "canonical_token_order_declared": trial.get("canonical_token_order", False),
                   "canonical_order_command_verified": command_verified,
                   "training_fingerprint_matches": trial.get("training_fingerprint") ==
                   summary.get("training_fingerprint"),
                   "corpus_permutation_verified": True,
                   "first_memorized_step": trial.get("first_memorized_step"),
                   "endpoint_audit": trial.get("audit"),
                   "memorization_audit": trial.get("memorization_audit")}
        controls.append(control)
        for step in steps:
            path = _checkpoint_path(root, trial, summary["model"]["layers"], step)
            try:
                bits = read_checkpoint_bits(path, layout)
            except (ValueError, OSError) as error:
                # Missing snapshots from an in-progress run are not evidence
                # of equality. Even malformed finalized files remain visible.
                unavailable.append({"id": trial_id, "step": step, "reason": str(error)})
                continue
            aligned = align_bits(bits, permutation, layout)
            loaded[step].append({"id": trial_id, "raw": bits, "aligned": aligned,
                                 "path": str(path), "raw_sha256": digest(bits),
                                 "aligned_sha256": digest(aligned)})
            if step == 0:
                embedding = bits[:layout[0].stop].reshape(layout[0].shape)
                control["initial_raw_sha256"] = digest(bits)
                control["initial_aligned_sha256"] = digest(aligned)
                control["identical_initial_embedding_rows"] = bool(np.all(embedding == embedding[0]))
                declared = trial.get("initial_sha256")
                control["initial_hash_matches_declaration"] = digest(bits) == declared if declared else None
    return loaded, unavailable, controls


def analyze(run_dir, *, steps=DEFAULT_STEPS, previous_run_dir=None):
    root = Path(run_dir).resolve()
    summary = json.loads((root / "summary.json").read_text())
    layout = tensor_layout(summary["model"])
    loaded, unavailable, controls = _load_available(root, summary, layout, steps)
    pairs = []
    for step in steps:
        for left, right in itertools.combinations(loaded[step], 2):
            kind = ("baseline_control" if "baseline" in (left["id"], right["id"])
                    else "permutation_pair")
            pairs.append(compare_pair(left, right, layout, step=step, kind=kind))
    previous_pairs = []
    if previous_run_dir:
        previous_root = Path(previous_run_dir).resolve()
        previous_summary = json.loads((previous_root / "summary.json").read_text())
        if summary["model"] != previous_summary["model"]:
            raise ValueError("previous model shape differs")
        previous_trial = next(t for t in previous_summary["trials"] if t["id"] == "baseline")
        for step in steps:
            current = next((t for t in loaded[step] if t["id"] == "baseline"), None)
            if current is None:
                continue
            path = _checkpoint_path(previous_root, previous_trial, summary["model"]["layers"], step)
            try:
                bits = read_checkpoint_bits(path, layout)
            except (ValueError, OSError) as error:
                unavailable.append({"id": "previous_baseline", "step": step, "reason": str(error)})
                continue
            previous_pairs.append(compare_pair(
                {"id": "previous_baseline", "raw": bits, "aligned": bits}, current,
                layout, step=step, kind="previous_baseline_control"))
    snapshots = [{"step": step, **{key: value for key, value in item.items()
                                  if key not in ("raw", "aligned")}}
                 for step in steps for item in loaded[step]]
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(root),
            "experiment_status": summary.get("status"), "model": summary["model"],
            "parameters": layout[-1].stop, "requested_steps": list(steps),
            "previous_run_dir": str(Path(previous_run_dir).resolve()) if previous_run_dir else None,
            "canonical_token_order_declared": summary.get("canonical_token_order", False),
            "controls": controls, "snapshots": snapshots, "comparisons": pairs,
            "previous_baseline_comparisons": previous_pairs, "unavailable": unavailable,
            "limitations": [
                "Only listed complete snapshots were compared; unrecorded steps are not verified.",
                "The three renamed runs preserve every fact and differ only in vocabulary labels.",
                "Alignment permutes token embedding rows only, including the tied language-modeling head.",
                "Bitwise equality includes signed zero and NaN payloads; finite weights are checked separately.",
                "Numerical distance excludes nonfinite pairs; those values are counted explicitly.",
                "Checkpoints contain FP32 master weights, not optimizer state.",
            ]}


def render_html(report):
    escape = lambda value: html.escape(str(value))
    parts = ["<!doctype html><meta charset='utf-8'><title>Canonical-order weight comparisons</title>",
             "<style>body{font:15px system-ui;max-width:1400px;margin:2em auto;padding:1em}"
             "table{border-collapse:collapse;width:100%;margin:1em 0}th,td{border:1px solid #ccc;"
             "padding:.45em;text-align:left}code{overflow-wrap:anywhere}details{margin:1em 0}</style>",
             "<h1>Canonical-order weight comparisons</h1>",
             f"<p>{report['parameters']:,} FP32 parameters. Experiment status: "
             f"{escape(report['experiment_status'])}. Captured {escape(report['created_utc'])}.</p>",
             f"<p>Source: <code>{escape(report['run_dir'])}</code></p>",
             "<p>For each run, align E[original token] = E[renamed token ID]. All other "
             "weights are compared without rearrangement. Zero aligned bit mismatches means "
             "identical stored FP32 parameters, not merely close numerical values.</p>",
             "<h2>Initialization and execution controls</h2><table><tr><th>Run</th>"
             "<th>Canonical order in recorded command</th><th>Identical initial embedding rows</th>"
             "<th>First perfect step</th><th>Final errors / sentences exact</th></tr>"]
    for item in report["controls"]:
        audit = item["endpoint_audit"] or {}
        parts.append(f"<tr><td>{escape(item['id'])}</td><td>"
                     f"{escape(item['canonical_order_command_verified'])}</td><td>"
                     f"{escape(item.get('identical_initial_embedding_rows', 'unavailable'))}</td>"
                     f"<td>{escape(item['first_memorized_step'])}</td><td>"
                     f"{escape(audit.get('errors', 'pending'))} / "
                     f"{escape(audit.get('exact_sentences', 'pending'))}</td></tr>")
    parts.append("</table>")
    for title, comparisons in (
            ("Primary: renamed runs compared with each other", [x for x in report["comparisons"]
                                                              if x["kind"] == "permutation_pair"]),
            ("Controls: renamed runs compared with the new baseline", [x for x in report["comparisons"]
                                                                     if x["kind"] == "baseline_control"]),
            ("Control: new baseline versus the previous experiment", report["previous_baseline_comparisons"])):
        parts.append(f"<h2>{title}</h2><table><tr><th>Step</th><th>Pair</th><th>Raw bit mismatches</th>"
                     "<th>Raw embedding rows changed</th><th>Non-embedding bit mismatches</th>"
                     "<th>Aligned bit mismatches</th><th>Aligned max |delta|</th><th>Aligned L2</th>"
                     "<th>Finite</th></tr>")
        for pair in comparisons:
            aligned = pair["aligned"]
            parts.append(f"<tr><td>{pair['step']:,}</td><td>{escape(pair['left'])} ↔ "
                         f"{escape(pair['right'])}</td><td>{pair['raw']['bitwise_mismatches']:,}</td>"
                         f"<td>{pair['raw_embedding_rows_changed']:,}</td>"
                         f"<td>{pair['nonembedding']['bitwise_mismatches']:,}</td>"
                         f"<td>{aligned['bitwise_mismatches']:,}</td>"
                         f"<td>{escape(aligned['max_abs_finite'])}</td><td>{escape(aligned['l2_finite'])}</td>"
                         f"<td>{aligned['finite_weights']}</td></tr>")
        parts.append("</table>")
        for pair in comparisons:
            parts.append(f"<details><summary>Step {pair['step']:,}: {escape(pair['left'])} ↔ "
                         f"{escape(pair['right'])}, per-tensor details</summary><table><tr><th>Tensor</th>"
                         "<th>Shape</th><th>Raw bit mismatches</th><th>Aligned bit mismatches</th></tr>")
            for tensor in pair["tensors"]:
                parts.append(f"<tr><td>{escape(tensor['name'])}</td><td>{escape(tensor['shape'])}</td>"
                             f"<td>{tensor['raw']['bitwise_mismatches']}</td>"
                             f"<td>{tensor['aligned']['bitwise_mismatches']}</td></tr>")
            parts.append("</table></details>")
    parts.append("<h2>Scope and caveats</h2><ul>" + "".join(
        f"<li>{escape(x)}</li>" for x in report["limitations"]) + "</ul>")
    parts.append(f"<p>{len(report['unavailable'])} requested snapshots unavailable. "
                 "Unavailable does not mean equal.</p><details><summary>Unavailable snapshots</summary><pre>" +
                 escape(json.dumps(report["unavailable"], indent=2)) + "</pre></details>")
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--previous_run_dir", type=Path)
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    parser.add_argument("--output_prefix", type=Path,
                        help="Output path without extension; defaults to RUN_DIR/exact_comparison")
    args = parser.parse_args()
    if any(step < 0 for step in args.steps) or len(set(args.steps)) != len(args.steps):
        parser.error("steps must be unique and nonnegative")
    report = analyze(args.run_dir, steps=args.steps, previous_run_dir=args.previous_run_dir)
    prefix = args.output_prefix or args.run_dir / "exact_comparison"
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    prefix.with_suffix(".html").write_text(render_html(report))
    for pair in report["comparisons"] + report["previous_baseline_comparisons"]:
        print(f"step={pair['step']} {pair['left']} vs {pair['right']}: "
              f"raw={pair['raw']['bitwise_mismatches']} "
              f"aligned={pair['aligned']['bitwise_mismatches']} "
              f"nonembedding={pair['nonembedding']['bitwise_mismatches']} "
              f"max_abs={pair['aligned']['max_abs_finite']} L2={pair['aligned']['l2_finite']} "
              f"finite={pair['aligned']['finite_weights']}")
    print(f"HTML: {prefix.with_suffix('.html')}")
    print(f"Unavailable requested snapshots: {len(report['unavailable'])}")


if __name__ == "__main__":
    main()
