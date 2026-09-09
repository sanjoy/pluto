"""Weight-only validity bounds for the uniform-routing Taylor approximation.

For a two-position head let g be its attention-score difference and v the
previous-minus-current output-projected value vector. Subtracting the uniform
routing output, the exact correction is 0.5*tanh(g/2)*v. Its norm is at most
0.5*||v||, whereas the first-order approximation has norm |g|*||v||/4.
Consequently its error is AT LEAST max(|g|-2,0)*||v||/4. It is also at most
min(|g|^3/48, |g|/4)*||v||: integrate |tanh'(x)-1|=tanh(x)^2 <= x^2,
and use tanh's sign and |tanh(x)| <= |x|. These bounds need only
weight contractions and the range of probabilities: no tanh, softmax, prompts,
corpus, or model forward pass is evaluated by this module.

Bounds apply to separate head corrections around uniform routing, not full
attention outputs, model logits, or cosine-ranking errors. Summing heads can
cancel errors; never read a bound on the stacked heads as a bound on their sum.
A zero lower bound proves nothing about accuracy. A large lower bound falsifies
the assumption that the linearization is numerically close on these inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .checkpoint import GPT2Checkpoint, sha256_file


def _stable_norm(values):
    """Avoid squaring overflow when the true Frobenius norm is still finite."""
    scale = float(np.max(np.abs(values)))
    if not scale:
        return 0.0
    with np.errstate(over="ignore"):
        result = scale * float(np.sqrt(np.sum(np.square(values / scale))))
    if not np.isfinite(result):
        raise ValueError("nonfinite Frobenius norm")
    return result


def head_components(queries, previous_keys, current_keys, previous_values,
                    current_values, output):
    """Return score gaps and value-difference norms, each [H,current,previous].

    Projected arrays have shape [tokens,H,C]; output has shape [H,C,D]. Query
    bias is already included. Content heads must already be in routing-head
    order when auditing a broken pairing. Only a [previous,D] difference is
    allocated at a time, rather than a [H,current,previous,D] tensor.
    """
    arrays = [np.asarray(a, dtype=np.float64) for a in
              (queries, previous_keys, current_keys, previous_values, current_values, output)]
    q, kp, kc, vp, vc, wo = arrays
    if (q.ndim != 3 or not all(q.shape) or kp.ndim != 3 or not all(kp.shape)
            or kc.shape != q.shape or vp.shape != kp.shape or vc.shape != q.shape
            or kp.shape[1:] != q.shape[1:] or wo.ndim != 3
            or wo.shape[:2] != q.shape[1:] or not wo.shape[2]):
        raise ValueError("expected projected [tokens,H,C] and output [H,C,D] arrays")
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("projected weights must be finite")
    current_count, heads, channels = q.shape
    gaps = np.empty((heads, current_count, len(kp)))
    value_norms = np.empty_like(gaps)
    for head in range(heads):
        previous_writes = vp[:, head] @ wo[head]
        current_writes = vc[:, head] @ wo[head]
        for current in range(current_count):
            gaps[head, current] = ((kp[:, head] - kc[current, head]) @
                                   q[current, head]) / np.sqrt(channels)
            value_norms[head, current] = np.linalg.norm(
                previous_writes - current_writes[current], axis=1)
    if not np.isfinite(gaps).all() or not np.isfinite(value_norms).all():
        raise ValueError("nonfinite routing components")
    return gaps, value_norms


def summarize_bounds(gaps, value_norms):
    """Summarize analytic bounds; do not confuse them with actual error.

    The ratio uses Frobenius norms over the independently stacked head/context
    corrections, with the LINEAR approximation's norm as denominator. It is
    not relative error divided by the exact correction, nor after head sums.
    """
    gaps, value_norms = np.asarray(gaps, dtype=np.float64), np.asarray(value_norms, dtype=np.float64)
    if (gaps.ndim != 3 or not all(gaps.shape) or value_norms.shape != gaps.shape
            or not np.isfinite(gaps).all() or not np.isfinite(value_norms).all()
            or np.any(value_norms < 0)):
        raise ValueError("expected finite gaps and nonnegative norms shaped [H,current,previous]")

    def summarize(gap, norms):
        absolute = np.abs(gap)
        active = norms > 0
        linear_norm = (absolute / 4) * norms
        lower = (np.maximum(absolute - 2, 0) / 4) * norms
        # Clipping BEFORE squaring avoids overflow for a huge finite score gap.
        factor = np.minimum(absolute, np.sqrt(12))**2 / 12
        factor[absolute >= np.sqrt(12)] = 1.0
        upper = linear_norm * factor
        if not all(np.isfinite(a).all() for a in (linear_norm, lower, upper)):
            raise ValueError("nonfinite correction norms")
        linear = _stable_norm(linear_norm)
        bound = _stable_norm(lower)
        upper_bound = _stable_norm(upper)
        active_count = int(np.count_nonzero(active))
        return {
            "head_contexts": gap.size,
            "nonzero_value_difference_contexts": active_count,
            "absolute_gap_quantiles": {str(p): float(np.quantile(absolute, p))
                                       for p in (0.0, 0.5, 0.9, 0.99, 1.0)},
            "active_fraction_absolute_gap_above": {
                str(threshold): (float(np.count_nonzero(active & (absolute > threshold))) /
                                 active_count if active_count else None)
                for threshold in (2, 10, 100)
            },
            "linear_correction_frobenius_norm": linear,
            "error_lower_bound_frobenius_norm": bound,
            "error_lower_bound_over_linear_norm": bound / linear if linear else None,
            "error_upper_bound_frobenius_norm": upper_bound,
            "error_upper_bound_over_linear_norm": upper_bound / linear if linear else None,
        }

    return {
        "stacked_heads": summarize(gaps, value_norms),
        "by_head": [dict(head=head, **summarize(gaps[head], value_norms[head]))
                    for head in range(gaps.shape[0])],
    }


def audit_probe(probe):
    """Audit exactly the dictionaries and head pairing used by a closed probe."""
    size = len(probe.embedding)
    contraction = probe.contraction
    _, heads, channels = contraction.queries.shape
    output = contraction.output.reshape(heads, channels, -1)
    gaps, norms = head_components(
        contraction.queries[size:], contraction.keys[:size], contraction.keys[size:],
        contraction.values[:size], contraction.values[size:], output)
    return summarize_bounds(gaps, norms)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--vocabulary-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-geometry", choices=("normalized", "raw"), default="normalized")
    parser.add_argument("--vocabulary-size", type=int, default=128)
    parser.add_argument("--broken-routing-control", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite routing audit")
    # Delayed import keeps the pure bound/oracle tests independent of extraction
    # code. The real audit deliberately uses the SAME dictionaries as that code.
    from .closed_trigram import ClosedTrigramProbe, select_vocabulary, validate_recipe_epsilon
    epsilon_check = validate_recipe_epsilon()
    checkpoint = GPT2Checkpoint(args.checkpoint, check_finite=True)
    selector = GPT2Checkpoint(args.vocabulary_checkpoint, check_finite=True)
    if checkpoint.config != selector.config:
        raise ValueError("selector and evaluated checkpoint layouts must match")
    selection = select_vocabulary(selector, args.vocabulary_size)
    pairing = np.arange(checkpoint.config.n_heads)
    if args.broken_routing_control:
        if len(pairing) < 2:
            raise ValueError("broken routing needs at least two heads")
        pairing = (pairing + 1) % len(pairing)
    probe = ClosedTrigramProbe(checkpoint, selection.ids,
                               geometry=args.input_geometry, pairing=pairing)
    report = {
        "schema_version": 1,
        "stage": "weight_only_approximation_audit_not_extraction_or_model_inference",
        "parameters": {name: str(value) if isinstance(value, Path) else value
                       for name, value in vars(args).items()},
        "selected_token_ids": selection.ids.tolist(),
        "ov_pairing": pairing.tolist(),
        "epsilon_source_check": epsilon_check,
        "checkpoint": checkpoint.provenance(hash_weights=True),
        "vocabulary_checkpoint": selector.provenance(hash_weights=True),
        "bounds": audit_probe(probe),
        "definition": "max(abs(gap)-2,0)*||delta_value||/4 <= error <= min(abs(gap)^3/48,abs(gap)/4)*||delta_value||",
        "limitations": [
            "No exact attention/softmax/model output is computed; these are bounds only.",
            "A zero lower bound does not show the linear approximation is accurate.",
            "Bounds apply to separate head corrections, not their sum or model logits.",
            "Large error in vectors need not change cosine-ranked candidate token IDs.",
            "Dictionaries and selected contexts are weight-derived, not measured corpus activations.",
        ],
        "source_sha256": {name: sha256_file(Path(__file__).with_name(name))
                          for name in ("routing_audit.py", "closed_trigram.py", "trigram.py", "checkpoint.py")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(report["bounds"]["stacked_heads"], indent=2))


if __name__ == "__main__":
    main()
