"""Exploratory, weight-only scalar-envelope audit of a FROZEN polynomial.

For tanh-GELU phi, |phi(x)| <= |x| for every real x. Consequently a local
quadratic T(a,z)=phi(a)+g(a)z+h(a)z^2/2 obeys

    |T(a,z)-phi(a+z)| >= max(0, |T(a,z)|-|a+z|).

Only phi(a) at the FIXED weight-derived anchor is evaluated. No nonlinear
function is evaluated on candidate-dependent a+z. The bound is scalar: signed
readout weights and signed errors can cancel, so summing it does NOT bound
the error in logits or prove anything about actual late-layer activations.

The theorem is exact, but its numerical application here uses FP64 approximate
coefficients. A declared conservative arithmetic slack is reported separately;
it is not interval certification of transcendental coefficients or of the
matrix contractions that produced z. No candidate is changed or reranked.
"""

import argparse
from pathlib import Path

import numpy as np

from .checkpoint import GPT2Checkpoint
from .late_mlp_paths import array_record, file_record, json_value, write_exclusive
from .late_mlp_polynomial import (
    GELU_A, GELU_C, compile_polynomial, layer_norm_and_jacobian,
)


THRESHOLDS = (0., 1e-6, 1e-3, .1, 1., 10.)
SLACK_ULPS = 64


def envelope_bounds(anchor, phi_anchor, first, second, increments):
    """Pure polynomial arithmetic; no candidate-dependent nonlinear function.

    The four coefficient vectors are fixed before any candidate is inspected.
    Their last dimension indexes neurons; increments can have any prefix shape.
    Nominal bounds and slack-adjusted excesses are BOTH retained for honesty.
    """
    a, value, g, h = [np.asarray(v, dtype=np.float64)
                      for v in (anchor, phi_anchor, first, second)]
    z = np.asarray(increments, dtype=np.float64)
    if (a.ndim != 1 or not a.size or any(v.shape != a.shape for v in (value, g, h))
            or not z.ndim or z.shape[-1] != len(a) or not z.size
            or any(not np.isfinite(v).all() for v in (a, value, g, h, z))):
        raise ValueError("finite fixed coefficient vectors and compatible increments required")
    with np.errstate(over="ignore", invalid="ignore"):
        linear = g*z
        quadratic = .5*h*z*z
        approximation = value+linear+quadratic
        argument = a+z
        excess = np.abs(approximation)-np.abs(argument)
        scale = np.maximum(1., np.abs(value)+np.abs(linear)+np.abs(quadratic)+np.abs(a)+np.abs(z))
        slack = SLACK_ULPS*np.finfo(np.float64).eps*scale
        adjusted = np.maximum(excess-slack, 0.)
    if any(not np.isfinite(v).all() for v in (approximation, argument, excess, slack, adjusted)):
        raise ValueError("envelope arithmetic exceeded finite FP64 range")
    return {"approximation": approximation, "argument": argument,
            "nominal_lower_bound": np.maximum(excess, 0.),
            "arithmetic_slack": slack, "slack_adjusted_excess": adjusted}


def summarize(anchor, phi_anchor, first, second, increments):
    """Summarize all restart/target-prefix/neuron slots without filtering."""
    z = np.asarray(increments, dtype=np.float64)
    if z.ndim != 3:
        raise ValueError("expected [restart, prefix, neuron] increments")
    bounds = envelope_bounds(anchor, phi_anchor, first, second, z)
    lower = bounds["slack_adjusted_excess"]
    violated = lower > 0
    prefix_mask = violated.any(axis=2)
    a = np.asarray(anchor)
    return {
        "shape": list(z.shape), "neuron_slots": int(z.size),
        "prefixes": int(prefix_mask.size), "restarts": z.shape[0],
        "violating_slots": int(violated.sum()), "violating_slot_fraction": float(violated.mean()),
        "violating_prefixes": int(prefix_mask.sum()), "violating_prefix_fraction": float(prefix_mask.mean()),
        "restarts_with_any_violation": int(prefix_mask.any(axis=1).sum()),
        "distinct_neurons_violating": int(violated.any(axis=(0, 1)).sum()),
        "absolute_z_quantiles": {str(q): float(np.quantile(np.abs(z), q)) for q in (0., .5, .9, .99, .999, 1.)},
        "absolute_anchor_quantiles": {str(q): float(np.quantile(np.abs(a), q)) for q in (0., .5, .9, .99, 1.)},
        "slots_absolute_z_exceeds_absolute_anchor": int((np.abs(z)>np.abs(a)).sum()),
        "slots_absolute_z_exceeds": {str(t): int((np.abs(z)>t).sum()) for t in (1., 2., 4., 8.)},
        "maximum_nominal_lower_bound": float(bounds["nominal_lower_bound"].max()),
        "maximum_arithmetic_slack": float(bounds["arithmetic_slack"].max()),
        "maximum_slack_adjusted_excess": float(lower.max()),
        "slack_adjusted_exceeds_threshold": {
            str(t): {"slots": int((lower>t).sum()), "prefixes": int((lower>t).any(axis=2).sum())}
            for t in THRESHOLDS},
        "per_restart_prefix_maximum_excess": lower.max(axis=2).tolist(),
        "per_restart_violating_slots": violated.sum(axis=(1, 2)).tolist(),
        "z_array": array_record(z, "<f8"),
        "slack_adjusted_excess_array": array_record(lower, "<f8"),
    }


def run(plan_path, metadata_path, output_path):
    """Recompile frozen weights, then audit saved final and initial IDs only."""
    import json

    plan_path, metadata_path, output_path = map(Path, (plan_path, metadata_path, output_path))
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    plan = json.loads(plan_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("complete") is not True or metadata["plan"] != file_record(plan_path):
        raise ValueError("completed metadata must authenticate the frozen plan")
    candidate_path = Path(metadata["candidates"]["path"])
    if file_record(candidate_path) != metadata["candidates"]:
        raise ValueError("frozen candidates changed")
    records = [json.loads(line) for line in candidate_path.read_text().splitlines()]
    checkpoint = GPT2Checkpoint(plan["checkpoint"]["checkpoint_directory"], check_finite=True)
    if output_path.resolve().is_relative_to(checkpoint.directory):
        raise ValueError("audit outputs must not be inside a checkpoint")
    if json_value(checkpoint.provenance(hash_weights=True)) != plan["checkpoint"]:
        raise ValueError("frozen checkpoint/source identity changed")
    for name, record in plan["sources"].items():
        if file_record(record["path"]) != record:
            raise ValueError(f"frozen input changed: {name}")
    params = plan["parameters"]
    probe = compile_polynomial(checkpoint, plan["selection"]["sorted_ids"],
                               params["length"], mode=params["mode"],
                               blocks=params["blocks"], alpha=params["alpha"])
    compiled = {name: array_record(getattr(probe, name), "<i8" if name=="token_ids" else "<f8")
                for name in metadata["compiled_arrays"]}
    if compiled != metadata["compiled_arrays"] or compiled != metadata["compiled_arrays_after"]:
        raise ValueError("recompiled fixed coefficients differ from frozen hashes")
    second = params["blocks"][1]
    anchor_row = np.asarray(probe.metadata["anchor7"])
    normalized, _, _ = layer_norm_and_jacobian(
        anchor_row, checkpoint[f"blocks.{second}.ln2.scale"], checkpoint[f"blocks.{second}.ln2.bias"])
    a = normalized @ np.asarray(checkpoint[f"blocks.{second}.mlp.input.weight"], dtype=np.float64)
    a += checkpoint[f"blocks.{second}.mlp.input.bias"]
    # This is the ONE permitted nonlinear evaluation, at the fixed anchor.
    # Candidate-specific a+z is never passed to tanh/GELU/LayerNorm.
    phi_a = .5*a*(1+np.tanh(GELU_C*(a+GELU_A*a**3)))
    initial = np.asarray(plan["initial_token_ids"], dtype=np.int64)
    final = np.asarray([record["token_ids"] for record in records], dtype=np.int64)
    if final.shape != initial.shape or len(records) != params["restarts"]:
        raise ValueError("candidate count/shape changed")
    results = {}
    for name, rows in (("initial_random_strings", initial), ("frozen_final_candidates", final)):
        components = probe.components(rows)
        expected = [record["initial_score" if name=="initial_random_strings" else "score"] for record in records]
        if not np.array_equal(components["total"], expected):
            raise ValueError("saved candidate scores do not exactly replay")
        results[name] = summarize(a, phi_a, probe.g7, probe.h7, components["z"][:, 1:])
    report = {
        "schema_version": 1, "stage": "exploratory_weight_only_post_extraction_scalar_envelope_audit",
        "plan": file_record(plan_path), "metadata": file_record(metadata_path),
        "candidates": file_record(candidate_path), "helper": file_record(__file__),
        "numpy_version": np.__version__, "parameters": params,
        "fixed_anchor_arguments": array_record(a, "<f8"), "fixed_phi_anchor": array_record(phi_a, "<f8"),
        "fixed_anchor_argument_range": [float(a.min()), float(a.max())],
        "scalar_theorem": "|T-phi(a+z)| >= max(0, |T|-|a+z|), since tanh-GELU obeys |phi(x)|<=|x|",
        "numeric_slack": {"multiple": SLACK_ULPS, "epsilon": float(np.finfo(np.float64).eps),
                          "scale": "max(1,abs(phi_a)+abs(g*z)+abs(.5*h*z*z)+abs(a)+abs(z))",
                          "scope": "conservative FP64 arithmetic diagnostic, not interval-certified coefficients or contractions"},
        "thresholds": list(THRESHOLDS), "results": results,
        "limitations": [
            "This is exploratory diagnosis of the already-frozen surrogate; candidates and extraction settings are unchanged.",
            "No candidate-dependent nonlinear activation, model forward, or corpus lookup is performed.",
            "The scalar bound does not validate weight-derived anchors as actual late-layer residual geometry.",
            "Signed neuron errors and signed readout coefficients can cancel; these scalar lower bounds cannot be summed into a total-logit lower bound.",
            "A nonviolation does not establish accurate approximation; the envelope gives only a one-sided sufficient diagnostic.",
        ],
    }
    if (json_value(checkpoint.provenance(hash_weights=True)) != plan["checkpoint"]
            or file_record(candidate_path) != metadata["candidates"]
            or file_record(plan_path) != metadata["plan"]):
        raise ValueError("frozen inputs changed during audit")
    write_exclusive(output_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "metadata", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.metadata, args.output)


if __name__ == "__main__":
    main()
