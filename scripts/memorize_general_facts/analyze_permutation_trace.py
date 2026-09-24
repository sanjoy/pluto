#!/usr/bin/env python3
"""Explain a paired native token-permutation gradient trace using CPU FP64.

The native trace contains baseline, repeat and renamed tensor manifests. This
tool never trains or changes the model: it finds the first captured divergence
and separates the LM-head's operand quantization from the remaining GPU error.
"""

import argparse
import csv
from dataclasses import dataclass
import html
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Capture:
    """One concrete tensor recorded at a chronological stage of one step."""

    sequence: int
    stage: str
    dtype: str
    shape: tuple
    alignment: str
    path: Path
    data: np.ndarray


def round_bf16(values):
    """Round FP32 to BF16 using round-to-nearest, ties-to-even, then expand.

    NaNs stay NaNs, including FP32 NaNs whose payload exists only in the bits
    being discarded. Signed zero and infinities retain their representation.
    Returned FP64 values exactly represent each rounded BF16 operand.
    """
    source = np.asarray(values, dtype=np.float32)
    bits = source.view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    nan = ((bits & 0x7f800000) == 0x7f800000) & ((bits & 0x007fffff) != 0)
    rounded = np.where(nan, (bits & np.uint32(0xffff0000)) | np.uint32(0x00400000), rounded)
    return rounded.astype(np.uint32).view(np.float32).astype(np.float64)


def round_operands(values, compute_type):
    """Mirror the native LM-head MMA operand type, not its accumulator type.

    MmaType<float> in embedding.cc is FP16, so the model's FP32 activation mode
    is not a full-FP32 LM-head product. BF16 activation mode uses BF16 operands.
    Both kernels accumulate into FP32 and return FP32 input gradients.
    """
    if compute_type.upper() == "BF16":
        result = round_bf16(values)
    elif compute_type.upper() == "FP32":
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.asarray(values, dtype=np.float32).astype(np.float16).astype(np.float64)
    else:
        raise ValueError(f"unsupported diagnostic compute type: {compute_type}")
    if not np.isfinite(result).all():
        raise ValueError("nonfinite MMA operand after quantization; cannot use finite-error analysis")
    return result


def error_metrics(reference, observed):
    """Numerical changes and absolute/relative distances, with explicit zeros."""
    reference, observed = np.asarray(reference), np.asarray(observed)
    if reference.shape != observed.shape:
        raise ValueError("tensor shapes differ")
    reference64 = reference.astype(np.float64)
    observed64 = observed.astype(np.float64)
    difference = observed64 - reference64
    # FP64 losslessly represents each supported native value. This preserves
    # signed-zero differences without comparing unrelated storage widths.
    representations_differ = reference64.view(np.uint64) != observed64.view(np.uint64)
    norm = float(np.linalg.norm(reference.astype(np.float64)))
    distance = float(np.linalg.norm(difference))
    return {"elements": int(reference.size),
            "changed": int(np.count_nonzero(difference)),
            "bitwise_changed": int(np.count_nonzero(representations_differ)),
            "signed_zero_only": int(np.count_nonzero(representations_differ & (difference == 0))),
            "max_abs": float(np.max(np.abs(difference))) if difference.size else 0.0,
            "l2": distance, "reference_l2": norm,
            "relative_l2": distance / norm if norm else None}


def read_metadata(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or key in result:
            raise ValueError("metadata requires unique key=value lines")
        result[key.strip()] = value.strip()
    return result


def read_native_replay(path):
    """Load optional independently executed GPU counterfactual checks."""
    if not Path(path).exists():
        return []
    result, seen = [], set()
    with Path(path).open() as source:
        for row in csv.DictReader(source, delimiter="\t"):
            name = row["comparison"]
            values = {"comparison": name, "mismatches": int(row["mismatches"]),
                      "max_abs": float(row["max_abs"]), "l2": float(row["l2"])}
            if (name in seen or values["mismatches"] < 0 or
                    not all(np.isfinite(values[key]) and values[key] >= 0
                            for key in ("max_abs", "l2"))):
                raise ValueError("invalid native replay comparison")
            seen.add(name)
            result.append(values)
    return result


def read_captures(directory):
    """Read the native TSV manifest, checking tensor sizes and finite values."""
    directory = Path(directory)
    captures = []
    with (directory / "tensors.tsv").open() as source:
        for row in csv.DictReader(source, delimiter="\t"):
            shape = tuple(int(value) for value in row["shape"].split(","))
            if not shape or min(shape) <= 0:
                raise ValueError("capture dimensions must be concrete positive integers")
            path = directory / row["file"]
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("tensor path escapes role directory")
            dtype = row["dtype"].lower()
            stored_dtype = {"bf16": "<u2", "fp32": "<f4", "int32": "<i4"}.get(dtype)
            if stored_dtype is None:
                raise ValueError(f"unsupported capture dtype: {dtype}")
            contents = path.read_bytes()
            expected = int(np.prod(shape)) * np.dtype(stored_dtype).itemsize
            if len(contents) != expected or int(row["bytes"]) != expected:
                raise ValueError(f"wrong tensor byte count: {path}")
            data = np.frombuffer(contents, dtype=stored_dtype)
            if dtype == "bf16":
                data = (data.astype(np.uint32) << 16).view(np.float32)
            if not np.isfinite(data).all():
                raise ValueError(f"nonfinite captured tensor: {path}")
            captures.append(Capture(int(row["sequence"]), row["stage"], dtype,
                                    shape, row["alignment"], path, data.reshape(shape)))
    keys = [(capture.sequence, capture.stage) for capture in captures]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate capture sequence/stage")
    if [key[0] for key in keys] != sorted(key[0] for key in keys):
        raise ValueError("captures must be in chronological sequence order")
    return captures


def align(values, alignment, permutation):
    """Undo label renaming, leaving padded vocabulary lanes in their positions."""
    values = np.asarray(values)
    vocabulary = len(permutation)
    if alignment == "none":
        return values
    if alignment == "token_ids":
        if np.any(values >= vocabulary) or np.any(values < -1):
            raise ValueError("invalid token-ID capture")
        inverse = np.argsort(permutation)
        result = values.copy()
        valid = result >= 0
        result[valid] = inverse[result[valid]]
        return result
    axis = {"vocab_rows": 0, "vocab_columns": values.ndim - 1}.get(alignment)
    if axis is None or values.shape[axis] < vocabulary:
        raise ValueError(f"invalid alignment/shape: {alignment}, {values.shape}")
    indices = np.concatenate([permutation, np.arange(vocabulary, values.shape[axis])])
    return np.take(values, indices, axis=axis)


def head_reference(dlogits, embeddings, actual, compute_type):
    """Use actual pre-step operands; do not assume embeddings are still equal.

    There are two references: an FP64 product of the original FP32 operands,
    and an FP64 product after the native MMA's BF16/FP16 operand conversion.
    Their difference measures operand quantization. GPU minus the latter is
    the remaining implementation/reduction error, not proof of its cause.
    """
    gradients = np.asarray(dlogits, dtype=np.float64)
    table = np.asarray(embeddings, dtype=np.float64)
    if gradients.ndim != 2 or table.ndim != 2 or gradients.shape[1] < table.shape[0]:
        raise ValueError("incompatible head gradient/embedding shapes")
    padded = np.zeros((gradients.shape[1], table.shape[1]), dtype=np.float64)
    padded[:len(table)] = table
    if actual.shape != (len(gradients), table.shape[1]):
        raise ValueError("head input gradient shape mismatch")
    fp64 = gradients @ padded
    rounded = round_operands(gradients, compute_type) @ round_operands(padded, compute_type)
    return {"fp64": fp64, "rounded_fp64": rounded,
            "gpu_vs_fp64": error_metrics(fp64, actual),
            "operand_quantization": error_metrics(fp64, rounded),
            "gpu_vs_rounded_fp64": error_metrics(rounded, actual)}


def cross_entropy_reference(logits, targets):
    """Stable FP64 loss and valid-row-mean gradient, matching loss semantics."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets).reshape(-1)
    if logits.ndim != 2 or len(logits) != len(targets):
        raise ValueError("cross-entropy reference shapes disagree")
    valid = targets >= 0
    if np.any(targets < -1) or np.any(targets >= logits.shape[1]):
        raise ValueError("cross-entropy target out of range")
    loss, gradient = np.zeros(len(targets)), np.zeros_like(logits)
    if not np.any(valid):
        return loss, gradient
    selected = logits[valid]
    maximum = selected.max(axis=1)
    exponentials = np.exp(selected - maximum[:, None])
    denominator = exponentials.sum(axis=1)
    indices = np.arange(len(selected))
    loss[valid] = np.log(denominator) + maximum - selected[indices, targets[valid]]
    probability = exponentials / denominator[:, None]
    probability[indices, targets[valid]] -= 1
    gradient[valid] = probability / np.count_nonzero(valid)
    return loss, gradient


def compare_cross_entropy(roles, permutation):
    """Canonicalize vocab order before FP64 evaluation of identical inputs."""
    logits_stage = "fwd/gpt2/LanguageModelingHeadLayer/0"
    if not any(capture.stage == logits_stage for capture in roles["baseline"]):
        return None
    references, actuals, logits_by_role = {}, {}, {}
    for role, captures in roles.items():
        logits = _one_capture(captures, logits_stage)
        logits = logits.reshape(-1, logits.shape[-1])
        targets = _one_capture(captures, "targets").reshape(-1)
        dlogits = _one_capture(captures, "loss/dlogits").reshape(logits.shape)
        if role == "renamed":
            logits = align(logits, "vocab_columns", permutation)
            targets = align(targets, "token_ids", permutation)
            dlogits = align(dlogits, "vocab_columns", permutation)
        loss, gradient = cross_entropy_reference(logits, targets)
        actual = _one_capture(captures, "loss/fwd").reshape(-1)
        references[role] = (loss, gradient)
        actuals[role] = (actual, dlogits, targets)
        logits_by_role[role] = logits
    a, b = references["baseline"], references["renamed"]
    actual_a, actual_b = actuals["baseline"], actuals["renamed"]
    different_rows = np.flatnonzero(actual_a[0] != actual_b[0])
    return {"aligned_logits": error_metrics(logits_by_role["baseline"], logits_by_role["renamed"]),
            "canonical_fp64_losses": error_metrics(a[0], b[0]),
            "canonical_fp64_dlogits": error_metrics(a[1], b[1]),
            "gpu_losses": error_metrics(actual_a[0], actual_b[0]),
            "gpu_dlogits": error_metrics(actual_a[1], actual_b[1]),
            "different_loss_rows": [{"row": int(row), "target": int(actual_a[2][row]),
                "fp64_loss": float(a[0][row]), "gpu_baseline_loss": float(actual_a[0][row]),
                "gpu_renamed_loss": float(actual_b[0][row])} for row in different_rows],
            "roles": {role: {"gpu_loss_vs_fp64": error_metrics(ref[0], actuals[role][0]),
                             "gpu_dlogits_vs_fp64": error_metrics(ref[1], actuals[role][1])}
                      for role, ref in references.items()}}


def compare_stages(baseline, repeat, renamed, permutation):
    keys = lambda captures: [(c.sequence, c.stage, c.dtype, c.shape, c.alignment) for c in captures]
    if keys(baseline) != keys(repeat) or keys(baseline) != keys(renamed):
        raise ValueError("role manifests have mismatched stages/types/shapes")
    return [{"sequence": a.sequence, "stage": a.stage, "shape": a.shape,
             "alignment": a.alignment,
             "repeat": error_metrics(a.data, b.data),
             "renamed_aligned": error_metrics(a.data, align(c.data, c.alignment, permutation))}
            for a, b, c in zip(baseline, repeat, renamed)]


def _one_capture(captures, stage):
    matches = [capture for capture in captures if capture.stage == stage]
    if len(matches) != 1:
        raise ValueError(f"expected one capture for {stage}, got {len(matches)}")
    return matches[0].data


def compare_updates(roles, permutation):
    """Measure what Adam changes, separately from preexisting weight errors."""
    baseline = {capture.stage: capture for capture in roles["baseline"]}
    renamed = {capture.stage: capture for capture in roles["renamed"]}
    result = []
    for stage, before in baseline.items():
        if not stage.startswith("weights_before/"):
            continue
        name = stage.removeprefix("weights_before/")
        after_stage = "weights_after/" + name
        gradient_stage = "parameter_gradients/" + name
        if after_stage not in baseline or gradient_stage not in baseline:
            continue
        after, gradient = baseline[after_stage], baseline[gradient_stage]
        aligned_before = align(renamed[stage].data, before.alignment, permutation)
        aligned_after = align(renamed[after_stage].data, after.alignment, permutation)
        aligned_gradient = align(renamed[gradient_stage].data, gradient.alignment, permutation)
        update_a = after.data.astype(np.float64) - before.data.astype(np.float64)
        update_b = aligned_after.astype(np.float64) - aligned_before.astype(np.float64)
        gradient_error = error_metrics(gradient.data, aligned_gradient)
        update_error = error_metrics(update_a, update_b)
        result.append({"name": name, "shape": before.shape,
                       "before": error_metrics(before.data, aligned_before),
                       "gradient": gradient_error, "update": update_error,
                       "after": error_metrics(after.data, aligned_after),
                       "update_to_gradient_discrepancy_l2_ratio":
                       update_error["l2"] / gradient_error["l2"] if gradient_error["l2"] else None})
    return result


def analyze(directory):
    directory = Path(directory).resolve()
    metadata = read_metadata(directory / "metadata.txt")
    permutation = np.array([int(value) for value in (directory / "permutation.tsv").read_text().split()])
    if not np.array_equal(np.sort(permutation), np.arange(len(permutation))):
        raise ValueError("permutation must be an old-to-new bijection")
    compute_type = metadata["compute_type"].upper()
    roles = {name: read_captures(directory / name) for name in ("baseline", "repeat", "renamed")}
    stages = compare_stages(roles["baseline"], roles["repeat"], roles["renamed"], permutation)
    references, tensors = {}, {}
    for name, captures in roles.items():
        embedding = _one_capture(captures, "weights_before/token_embedding")
        if embedding.ndim != 2 or embedding.shape[0] != len(permutation):
            raise ValueError("token embedding must have shape [vocabulary, width]")
        dlogits = _one_capture(captures, "loss/dlogits")
        dlogits = dlogits.reshape(-1, dlogits.shape[-1])
        actual = _one_capture(captures, "bwd/gpt2/LayerNormLayer/0").reshape(-1, embedding.shape[1])
        references[name] = head_reference(dlogits, embedding, actual, compute_type)
        tensors[name] = {"embedding": embedding, "dlogits": dlogits, "actual": actual}

    a, b = references["baseline"], references["renamed"]
    actual_a, actual_b = tensors["baseline"]["actual"], tensors["renamed"]["actual"]
    residual_a = actual_a - a["rounded_fp64"]
    residual_b = actual_b - b["rounded_fp64"]
    head = {
        "mma_operand_type": "BF16" if compute_type == "BF16" else "FP16",
        "roles": {name: {key: value for key, value in reference.items()
                         if key not in ("fp64", "rounded_fp64")}
                  for name, reference in references.items()},
        "baseline_vs_renamed": {
            "pre_step_embeddings_aligned": error_metrics(tensors["baseline"]["embedding"],
                align(tensors["renamed"]["embedding"], "vocab_rows", permutation)),
            "dlogits_aligned": error_metrics(tensors["baseline"]["dlogits"],
                align(tensors["renamed"]["dlogits"], "vocab_columns", permutation)),
            "mma_rounded_dlogits_aligned": error_metrics(
                round_operands(tensors["baseline"]["dlogits"], compute_type),
                round_operands(align(tensors["renamed"]["dlogits"], "vocab_columns", permutation), compute_type)),
            "mma_rounded_embeddings_aligned": error_metrics(
                round_operands(tensors["baseline"]["embedding"], compute_type),
                round_operands(align(tensors["renamed"]["embedding"], "vocab_rows", permutation), compute_type)),
            "gpu_input_gradient": error_metrics(actual_a, actual_b),
            "fp64_reference": error_metrics(a["fp64"], b["fp64"]),
            "rounded_operand_fp64_reference": error_metrics(a["rounded_fp64"], b["rounded_fp64"]),
            "remaining_gpu_residual": error_metrics(residual_a, residual_b)},
    }
    # A canonical vocabulary order eliminates even FP64 summation reordering
    # from the paired reference, so equal aligned inputs produce exactly equal
    # CPU operands/products, independently of native vocabulary-ID grouping.
    canonical_b = head_reference(
        align(tensors["renamed"]["dlogits"], "vocab_columns", permutation),
        align(tensors["renamed"]["embedding"], "vocab_rows", permutation), actual_b, compute_type)
    head["baseline_vs_renamed"].update(
        canonical_fp64_reference=error_metrics(a["fp64"], canonical_b["fp64"]),
        canonical_rounded_operand_fp64_reference=error_metrics(a["rounded_fp64"], canonical_b["rounded_fp64"]))
    rounded_embedding_a = round_operands(tensors["baseline"]["embedding"], compute_type)
    rounded_embedding_b = round_operands(tensors["renamed"]["embedding"], compute_type)
    moved_ids = np.flatnonzero(permutation != np.arange(len(permutation)))
    head["changed_channels"] = {
        "raw_embedding_channels": np.flatnonzero(np.any(
            tensors["baseline"]["embedding"] != tensors["renamed"]["embedding"], axis=0)).tolist(),
        "mma_rounded_embedding_channels": np.flatnonzero(np.any(
            rounded_embedding_a != rounded_embedding_b, axis=0)).tolist(),
        "gpu_input_gradient_channels": np.flatnonzero(np.any(actual_a != actual_b, axis=0)).tolist(),
        "moved_vocabulary_ids": moved_ids.tolist(),
        "first_moved_rows": [{"token_id": int(token), "renamed_id": int(permutation[token]),
            "baseline_embedding": tensors["baseline"]["embedding"][token].tolist(),
            "baseline_embedding_at_renamed_id": tensors["baseline"]["embedding"][permutation[token]].tolist(),
            "rounded_baseline_embedding": rounded_embedding_a[token].tolist(),
            "rounded_baseline_embedding_at_renamed_id": rounded_embedding_a[permutation[token]].tolist()}
            for token in moved_ids[:16]]}
    largest_rows = np.argsort(np.max(np.abs(actual_b - actual_a), axis=1), kind="stable")[::-1][:8]
    targets_a = _one_capture(roles["baseline"], "targets").reshape(-1)
    targets_b = _one_capture(roles["renamed"], "targets").reshape(-1)
    head["selected_rows"] = [{"row": int(row), "baseline_target": int(targets_a[row]),
        "renamed_target": int(targets_b[row]),
        "gpu_baseline": actual_a[row].tolist(), "gpu_renamed": actual_b[row].tolist(),
        "fp64_baseline": a["fp64"][row].tolist(), "fp64_renamed": b["fp64"][row].tolist(),
        "rounded_fp64_baseline": a["rounded_fp64"][row].tolist(),
        "rounded_fp64_renamed": b["rounded_fp64"][row].tolist()}
        for row in largest_rows]
    first = next((stage for stage in stages if stage["renamed_aligned"]["changed"]), None)
    repeated = next((stage for stage in stages if stage["repeat"]["changed"]), None)
    batches = []
    if (directory / "batches.tsv").exists():
        with (directory / "batches.tsv").open() as source:
            batches = [{key: int(value) for key, value in row.items()}
                       for row in csv.DictReader(source, delimiter="\t")]
    paired = head["baseline_vs_renamed"]
    interpretations = []
    native_replay = read_native_replay(directory / "native_replay.tsv")
    native_by_name = {row["comparison"]: row for row in native_replay}
    canonical_controls = (
        "baseline_vs_saved/loss", "renamed_vs_saved/loss", "repeat/loss",
        "baseline_vs_saved/dlogits", "renamed_vs_saved/dlogits", "repeat/dlogits",
        "head_baseline_vs_saved", "head_renamed_vs_saved", "head_repeat",
        "canonical_input_order/loss", "canonical_input_order/dlogits", "head_canonical_input_order")
    replay_restores_baseline = all(name in native_by_name and native_by_name[name]["mismatches"] == 0
                                   for name in canonical_controls)
    if replay_restores_baseline:
        interpretations.append(
            "Native GPU replays exactly reproduce the saved baseline and renamed cross-entropy outputs "
            "and head input gradients. Returning vocabulary rows/columns to baseline order restores all "
            "three outputs bitwise, while repeat controls also match. This directly confirms ordering "
            "dependence in these kernels, not run-to-run randomness.")
    if (paired["mma_rounded_dlogits_aligned"]["changed"] == 0 and
            paired["mma_rounded_embeddings_aligned"]["changed"] == 0 and
            paired["gpu_input_gradient"]["changed"] > 0):
        interpretations.append(
            "The head receives exactly matching aligned rounded operands, yet its GPU input gradients differ. "
            "This isolates sensitivity to how this kernel processes the vocabulary ordering, rather than "
            "different mathematical operands. A native canonical-order replay is the direct counterfactual control.")
        if paired["dlogits_aligned"]["changed"]:
            interpretations.append(
                "The cross-entropy gradient differences disappear completely when converted to the head's "
                "MMA operand type. They are not the input perturbation causing this head-gradient difference: "
                "these are two independent numerical permutation-symmetry breaks, not demonstrated causal amplification.")
    return {"trace_directory": str(directory), "metadata": metadata, "stages": stages,
            "first_renamed_difference": first, "first_repeat_difference": repeated,
            "first_renamed_bitwise_difference": next(
                (stage for stage in stages if stage["renamed_aligned"]["bitwise_changed"]), None),
            "first_repeat_bitwise_difference": next(
                (stage for stage in stages if stage["repeat"]["bitwise_changed"]), None),
            "parameter_updates": compare_updates(roles, permutation),
            "cross_entropy": compare_cross_entropy(roles, permutation),
            "batches": batches, "interpretations": interpretations,
            "native_replay": native_replay,
            "native_canonical_replay_restores_baseline": replay_restores_baseline,
            "head_gradient": head, "limitations": [
                "This is a trace of one update, not proof of what caused the final trained weights to differ.",
                "Deterministic execution does not imply invariance to reordering floating-point reductions.",
                "FP64 references use the actual pre-step embeddings; these may already differ across vocabulary rows.",
                "FP64 summation is not exact arithmetic; canonical ordering controls its paired reduction order.",
                "All changed-value counts use exact numerical inequality with no tolerance, including FP64 references.",
                "GPU minus rounded-operand FP64 includes accumulation/implementation effects, not just proven rounding error.",
                "Relative errors are undefined when the reference norm is zero; inspect absolute errors too.",
                "The first numerical divergence excludes +0 versus −0; representation-only differences are reported separately.",
                "With identical embedding rows, the ideal initial backbone gradient is zero, but the tied output embedding gradient is generally nonzero.",
            ]}


def render_html(report):
    escape = lambda value: html.escape(str(value))
    number = lambda value: "undefined" if value is None else f"{value:.8g}"
    parts = ["<!doctype html><meta charset='utf-8'><title>Permutation gradient trace</title>",
             "<style>body{font:15px system-ui;max-width:1450px;margin:2em auto;padding:1em}"
             "table{border-collapse:collapse;margin:1em 0;width:100%}td,th{border:1px solid #bbb;"
             "text-align:left;padding:.4em}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
             "code{overflow-wrap:anywhere}</style><h1>First token-permutation gradient divergence</h1>",
             f"<p>Trace: <code>{escape(report['trace_directory'])}</code></p>",
             f"<p>Metadata: <code>{escape(report['metadata'])}</code></p>"]
    for label, key in (("First baseline-repeat difference", "first_repeat_difference"),
                       ("First aligned renamed difference", "first_renamed_difference")):
        first = report[key]
        parts.append(f"<p>{label}: <strong>{escape(first['stage']) if first else 'none'}</strong>"
                     f"{(' (sequence ' + str(first['sequence']) + ')') if first else ''}.</p>")
    for label, key in (("First baseline-repeat representation difference", "first_repeat_bitwise_difference"),
                       ("First renamed representation difference", "first_renamed_bitwise_difference")):
        first = report[key]
        parts.append(f"<p>{label} (including signed zero): "
                     f"{escape(first['stage']) if first else 'none'}.</p>")
    parts.append("<ul>" + "".join(f"<li>{escape(item)}</li>" for item in report["limitations"]) + "</ul>")
    parts.append("<h2>What this capture establishes</h2><ul>" + "".join(
        f"<li>{escape(item)}</li>" for item in report["interpretations"]) + "</ul>")
    if report["native_replay"]:
        parts.append("<h2>Native GPU counterfactual replay</h2><p>These results come from the native "
                     "replay tool, independently of the CPU FP64 calculations. Canonical input order "
                     "undoes the vocabulary permutation before launching the original GPU kernels.</p>"
                     "<table><tr><th>Comparison</th><th>Bitwise mismatches</th><th>Max absolute</th><th>L2</th></tr>")
        for row in report["native_replay"]:
            parts.append(f"<tr><td>{escape(row['comparison'])}</td><td>{row['mismatches']}</td>"
                         f"<td>{number(row['max_abs'])}</td><td>{number(row['l2'])}</td></tr>")
        parts.append("</table>")
    if report["batches"]:
        parts.append("<h2>Batch timeline before divergence</h2><p>Raw IDs count renamed occurrences "
                     "in each batch. Unaligned embedding differences compare the two weight files without "
                     "undoing token renaming; these are expected when the renamed rows separate and do not "
                     "by themselves indicate broken symmetry.</p><table><tr><th>Step</th><th>Changed raw "
                     "input IDs</th><th>Changed raw target IDs</th><th>Unaligned embedding values differing "
                     "before step</th></tr>")
        for batch in report["batches"]:
            parts.append(f"<tr><td>{batch['step']}</td><td>{batch['changed_input_ids']}</td>"
                         f"<td>{batch['changed_target_ids']}</td>"
                         f"<td>{batch['unaligned_embedding_values_before']}</td></tr>")
        parts.append("</table>")
    parts.append("<h2>LM-head reference decomposition</h2><p>For each position, dH = dLogits × E. "
                 "The original-operand reference evaluates this product in FP64. The rounded-operand "
                 "reference first applies the native MMA operand conversion, then multiplies in FP64. "
                 "Their difference measures operand quantization. Actual GPU minus rounded reference "
                 "isolates the remaining implementation/reduction discrepancy.</p>"
                 f"<p>Native MMA operand type: {report['head_gradient']['mma_operand_type']}; "
                 "native accumulation/output: FP32.</p>")

    def metric_table(rows):
        parts.append("<table><tr><th>Comparison</th><th>Changed values</th><th>Signed-zero only</th><th>Max absolute</th>"
                     "<th>L2</th><th>Relative L2</th></tr>")
        for label, metric in rows:
            parts.append(f"<tr><td>{escape(label)}</td><td>{metric['changed']}/{metric['elements']}</td>"
                         f"<td>{metric['signed_zero_only']}</td>"
                         f"<td>{number(metric['max_abs'])}</td><td>{number(metric['l2'])}</td>"
                         f"<td>{number(metric['relative_l2'])}</td></tr>")
        parts.append("</table>")

    metric_table(report["head_gradient"]["baseline_vs_renamed"].items())
    channels = report["head_gradient"]["changed_channels"]
    parts.append("<h3>Which channels carry the reordered products?</h3>"
                 f"<p>Raw embedding channels changed before alignment: {escape(channels['raw_embedding_channels'])}. "
                 f"After MMA rounding: {escape(channels['mma_rounded_embedding_channels'])}. "
                 f"Channels with differing GPU input gradients: {escape(channels['gpu_input_gradient_channels'])}.</p>"
                 "<details><summary>Moved embedding rows (first 16)</summary><pre>"
                 f"{escape(json.dumps(channels['first_moved_rows'], indent=2))}</pre></details>")
    for role, comparisons in report["head_gradient"]["roles"].items():
        parts.append(f"<h3>{escape(role)}</h3>")
        metric_table(comparisons.items())
    if report["cross_entropy"]:
        cross_entropy = report["cross_entropy"]
        parts.append("<h2>Cross-entropy at the first differing step</h2><p>These FP64 references "
                     "use a shared canonical vocabulary order. Identical aligned logits and targets "
                     "must give identical reference losses and gradients. Native softmax sums are "
                     "instead grouped by vocabulary ID in 16-column tiles.</p>")
        metric_table((key, value) for key, value in cross_entropy.items()
                     if key not in ("roles", "different_loss_rows"))
        parts.append("<details><summary>Rows whose GPU loss differs</summary><pre>"
                     f"{escape(json.dumps(cross_entropy['different_loss_rows'], indent=2))}</pre></details>")
    parts.append("<h2>Largest differing rows</h2><p>Rows are ranked by largest absolute paired "
                 "GPU input-gradient difference. Targets are compact vocabulary IDs; −1 means ignored.</p>")
    for row in report["head_gradient"]["selected_rows"]:
        parts.append(f"<details><summary>Row {row['row']}, target {row['baseline_target']} → "
                     f"{row['renamed_target']}</summary><pre>{escape(json.dumps(row, indent=2))}</pre></details>")
    parts.append("<h2>Optimizer update discrepancies</h2><p>Update = post-step weight − pre-step weight. "
                 "The ratio compares the L2 discrepancy between updates to the L2 discrepancy between "
                 "parameter gradients; it is not a local derivative or a long-run amplification estimate.</p>"
                 "<table><tr><th>Tensor</th><th>Pre-step weight Δ L2</th><th>Gradient Δ L2</th>"
                 "<th>Update Δ L2</th><th>Update/gradient discrepancy ratio</th><th>Post-step weight Δ L2</th></tr>")
    for tensor in report["parameter_updates"]:
        parts.append(f"<tr><td>{escape(tensor['name'])}</td><td>{number(tensor['before']['l2'])}</td>"
                     f"<td>{number(tensor['gradient']['l2'])}</td><td>{number(tensor['update']['l2'])}</td>"
                     f"<td>{number(tensor['update_to_gradient_discrepancy_l2_ratio'])}</td>"
                     f"<td>{number(tensor['after']['l2'])}</td></tr>")
    parts.append("</table>")
    parts.append("<h2>Chronological captured stages</h2><table><tr><th>Sequence</th><th>Stage</th>"
                 "<th>Repeat changes</th><th>Aligned renamed changes</th><th>Signed-zero only</th>"
                 "<th>Aligned max absolute</th></tr>")
    for stage in report["stages"]:
        parts.append(f"<tr><td>{stage['sequence']}</td><td>{escape(stage['stage'])}</td>"
                     f"<td>{stage['repeat']['changed']}</td><td>{stage['renamed_aligned']['changed']}</td>"
                     f"<td>{stage['renamed_aligned']['signed_zero_only']}</td>"
                     f"<td>{number(stage['renamed_aligned']['max_abs'])}</td></tr>")
    parts.append("</table>")
    return "\n".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="self-contained HTML output")
    args = parser.parse_args(argv)
    report = analyze(args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(report))
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    first = report["first_renamed_difference"]
    compact_metric = lambda metric: {key: metric[key] for key in ("changed", "max_abs", "l2")}
    print(json.dumps({"output": str(args.output),
                      "divergent_step": report["metadata"].get("divergent_step"),
                      "first_numerical_stage": first["stage"] if first else None,
                      "repeat_numerically_identical": report["first_repeat_difference"] is None,
                      "repeat_bitwise_identical": report["first_repeat_bitwise_difference"] is None,
                      "cross_entropy": {key: compact_metric(value)
                          for key, value in (report["cross_entropy"] or {}).items()
                          if key not in ("roles", "different_loss_rows")},
                      "interpretations": report["interpretations"],
                      "native_canonical_replay_restores_baseline": report["native_canonical_replay_restores_baseline"],
                      "head_comparisons": {key: compact_metric(value) for key, value in
                          report["head_gradient"]["baseline_vs_renamed"].items()},
                      "changed_channels": {key: value for key, value in
                          report["head_gradient"]["changed_channels"].items()
                          if key.endswith("channels")},
                      "parameters_changed_after_update": sum(
                          tensor["after"]["changed"] for tensor in report["parameter_updates"]),
                      "largest_update_discrepancies": [{"name": tensor["name"],
                          "gradient_delta_l2": tensor["gradient"]["l2"],
                          "update_delta_l2": tensor["update"]["l2"],
                          "ratio": tensor["update_to_gradient_discrepancy_l2_ratio"]}
                          for tensor in sorted(report["parameter_updates"],
                          key=lambda tensor: tensor["update"]["l2"], reverse=True)[:5]]},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
