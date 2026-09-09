"""Alternate-algorithm numerical check of a completed neuron reliance report.

This is a post-measurement audit, not a second forward experiment or text
extractor. It deliberately does not call the production reporter's numerical
helpers: scalar ``math.fsum`` checks raw averages, explicit centering matrices
check residuals, and symmetric eigendecomposition checks the discovery
projector. The production reporter remains responsible for authenticating its
frozen protocol and complete checkpoint. This auditor rehashes every raw file
and the two weight matrices needed for its independent norm comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def audit(report_path):
    report_identity = identity(report_path)
    report = json.loads(Path(report_path).read_text())
    if report.get("stage") != "fine_neuron_forward_reliance_not_extraction":
        raise ValueError("not a completed neuron-reliance report")
    if not report["all_checkpoints_and_inputs_unchanged"] or not report["clean_replays_byte_identical"]:
        raise ValueError("production integrity checks failed")
    if report["mapping"]["discovery"]["status"] != "unique_discovery_direction":
        raise ValueError("this alternate audit requires a unique discovery direction")
    records = report["raw_files"]
    for record in records.values():
        if identity(record["path"]) != record:
            raise ValueError("raw measurement changed")
    names = [arm["name"] for arm in report["run_metadata"]["arms"]]
    losses = {name: np.fromfile(records[name + ".losses.f32"]["path"],
                               dtype="<f4").reshape(32, 1024)
              for name in names}
    predictions = {name: np.fromfile(records[name + ".argmax.i32"]["path"],
                                    dtype="<i4").reshape(32, 1024)
                   for name in names}
    batch_identity = identity(report["run_metadata"]["batch_tokens_file"])
    batch = np.fromfile(batch_identity["path"],
                        dtype="<i4").reshape(2, 32, 1024)
    means, deltas = {}, {}
    max_raw_error = 0.0
    for label, start, end in (("discovery", 512, 768),
                              ("confirmation", 768, 1024), ("last512", 512, 1024)):
        means[label], deltas[label] = {}, {}
        for name in names:
            values, changes, accuracy = [], [], []
            for passage in range(32):
                current = [float(x) for x in losses[name][passage, start:end]]
                baseline = [float(x) for x in losses["clean_before"][passage, start:end]]
                values.append(math.fsum(current) / (end - start))
                changes.append(math.fsum(x - y for x, y in zip(current, baseline)) / (end - start))
                accuracy.append(sum(int(x == y) for x, y in zip(
                    predictions[name][passage, start:end], batch[1, passage, start:end])) / (end - start))
            means[label][name], deltas[label][name] = values, changes
            for key, actual in (("per_passage_mean_nll", values),
                                ("per_passage_mean_delta", changes),
                                ("per_passage_teacher_forced_accuracy", accuracy)):
                expected = report[key][label][name]
                max_raw_error = max(max_raw_error, max(abs(x - y) for x, y in zip(actual, expected)))
    if max_raw_error > 1e-12:
        raise ValueError("raw numerical report mismatch")

    groups = report["plan"]["groups"]
    matrices = {label: np.asarray([deltas[label][group["name"]][:16] for group in groups])
                for label in ("discovery", "confirmation")}
    # This dense operator is algebraically the reporter's double centering,
    # but follows a different floating-point evaluation order.
    center_groups = np.eye(64) - np.ones((64, 64)) / 64
    center_passages = np.eye(16) - np.ones((16, 16)) / 16
    centered = {label: center_groups @ matrix @ center_passages
                for label, matrix in matrices.items()}
    eigenvalues, eigenvectors = np.linalg.eigh(centered["discovery"] @ centered["discovery"].T)
    direction = eigenvectors[:, -1]
    projector = np.outer(direction, direction)
    original_direction = np.asarray(report["mapping"]["discovery"]["frozen_direction"])
    projector_error = float(np.max(np.abs(projector - np.outer(original_direction, original_direction))))
    residuals = {label: (np.eye(64) - projector) @ value for label, value in centered.items()}
    residual_error = max(float(np.max(np.abs(residuals["discovery"] -
                                           report["mapping"]["discovery"]["residual"]))),
                         float(np.max(np.abs(residuals["confirmation"] -
                                             report["mapping"]["confirmation_projected_residual"]))))
    if projector_error > 1e-11 or residual_error > 1e-11:
        raise ValueError("alternate projector does not agree")
    floor = 64 * np.finfo(np.float64).eps * 64 * math.sqrt(
        math.fsum(float(x) ** 2 for x in matrices["discovery"].flat))

    # Re-read W2 and sum squares scalarly; do not accept the plan's stored norms
    # as the source of its supposedly weight-only comparator assignments.
    checkpoint = report["passage_plan"]["checkpoint"]
    norms = []
    weight_records = []
    for block in (6, 7):
        name = f"weight_{12 + 12 * block}.bin"
        path = Path(checkpoint["checkpoint_directory"]) / name
        record = identity(path)
        if record["sha256"] != checkpoint["weight_sha256"][name]:
            raise ValueError("weight file changed")
        weight_records.append(record)
        weight = np.fromfile(path, dtype="<f4").reshape(2048, 512)
        for group in groups:
            if group["block"] == block:
                norms.append(math.sqrt(math.fsum(float(x) ** 2 for x in weight[group["rows"]].flat)))
    norm_error = max(abs(value - group["frobenius_norm_fp64"]) for value, group in zip(norms, groups))
    if norm_error > 1e-11:
        raise ValueError("independent weight norm differs")
    for index, group in enumerate(groups):
        base = index // 32 * 32
        comparators = sorted((i for i in range(base, base + 32) if i != index),
                             key=lambda i: (abs(norms[i] - norms[index]), i))[:4]
        if comparators != group["norm_comparators"]:
            raise ValueError("norm comparator mismatch")

    checks = []
    for passage, recorded in enumerate(report["mapping"]["selected"]):
        eligible = [i for i in range(64) if matrices["discovery"][i, passage] > 0
                    and residuals["discovery"][i, passage] > floor]
        selected = max(eligible, key=lambda i: (residuals["discovery"][i, passage], -i)) if eligible else None
        if selected != recorded["selected_group_index"]:
            raise ValueError("discovery selection mismatch")
        if selected is None:
            checks.append({"passage_index": passage, "selected_group_index": None})
            continue
        values = residuals["confirmation"][:, passage]
        base = selected // 32 * 32
        rank = 1 + sum(bool(float(value) > values[selected]) for value in values)
        block_rank = 1 + sum(bool(float(value) > values[selected]) for value in values[base:base + 32])
        difference = float(values[selected]) - math.fsum(float(values[i]) for i in
                                                        groups[selected]["norm_comparators"]) / 4
        if rank != recorded["confirmation_rank_all64"]["rank"] or block_rank != recorded["confirmation_rank_same_block"]["rank"]:
            raise ValueError("confirmation rank mismatch")
        if abs(difference - recorded["confirmation_minus_norm_comparator_mean"]) > 1e-11:
            raise ValueError("confirmation comparator mismatch")
        checks.append({"passage_index": passage, "selected_group_index": selected,
                       "confirmation_rank_all64": rank, "confirmation_rank_same_block": block_rank,
                       "confirmation_minus_norm_comparator_mean": difference})
    for record in [*records.values(), *weight_records, report_identity, batch_identity]:
        if identity(record["path"]) != record:
            raise ValueError("input changed during independent check")
    return {"schema_version": 1, "stage": "alternate_algorithm_neuron_report_check",
            "report": report_identity, "source": identity(__file__), "batch_tokens": batch_identity,
            "raw_files_rehashed": len(records), "weight_files": weight_records,
            "max_raw_mean_delta_accuracy_absolute_error": max_raw_error,
            "max_dense_eigh_projector_absolute_error": projector_error,
            "max_dense_residual_absolute_error": residual_error,
            "max_scalar_norm_absolute_error": norm_error,
            "checked_all_group_comparators": len(groups), "selected_checks": checks,
            "all_checks_passed": True,
            "limitations": ["Numerical report verification, not a new model experiment.",
                            "Does not independently authenticate the entire frozen plan.",
                            "No extracted text or private storage claim."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.report)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with args.output.open("x") as stream:
        stream.write(payload)
    print(json.dumps({key: value for key, value in result.items() if key.startswith("max_")}))


if __name__ == "__main__":
    main()
