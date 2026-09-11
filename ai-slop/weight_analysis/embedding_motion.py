"""Embedding-geometry motion, not gradient inversion or ordered-text recovery.

Each endpoint is centered over ALL logical vocabulary rows. Raw directed score
S=F0 D^T is invariant to common orthogonal coordinates but can be antisymmetric
under a pure rigid rotation of embedding geometry. Such rotations need not be
functional GPT-2 reparameterizations: learned diagonal LayerNorm gains constrain
the architecture's coordinate changes. Procrustes and fitted scale are declared
geometric nuisance projections, not authenticated AdamW decay correction.

No vocabulary-by-vocabulary matrix is materialized. All score/Gram energies
use dimension-by-dimension contractions. Arithmetic cancellation allowances are
reported; materially negative squared norms fail rather than being clipped.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .checkpoint import GPT2Checkpoint, sha256_file


STEPS = (12990, 13000, 13010, 13020, 13030)
EPS = np.finfo(np.float64).eps
BOUND_SAFETY = 8.
RANK_TOL_MULTIPLIER = 64.


def _gamma(operations):
    value = operations*EPS
    if value >= .01:
        raise ValueError("array dimensions exceed the declared FP64 error model")
    return value/(1-value)


def _checked(values, name):
    original = np.asarray(values)
    if original.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be numerical, not boolean or object")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not all(array.shape) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite [vocabulary,width] matrix")
    return array


def _squared_norm(value, allowance, name):
    """Clamp ONLY negative cancellation within an explicitly reported allowance."""
    value, allowance = float(value), float(allowance)
    if not np.isfinite(value) or not np.isfinite(allowance) or allowance < 0:
        raise ArithmeticError(f"nonfinite/invalid norm arithmetic: {name}")
    if value < -allowance:
        raise ArithmeticError(f"materially negative squared norm: {name}: {value} < {-allowance}")
    return {"squared_norm": max(value, 0.), "norm": float(np.sqrt(max(value, 0.))),
            "computed_signed_squared_norm": value,
            "squared_roundoff_allowance": allowance,
            "negative_roundoff_clamped": value < 0,
            "squared_norm_above_roundoff": value > allowance}


def _trace_product(first, second, error_first, error_second):
    """tr(A B), allowing entrywise errors already accumulated in A and B.

    Dot-product gamma bounds are conditional on the stored FP64 F and D.
    They do not certify earlier centering, SVD, or checkpoint-generation error.
    An explicit factor8 cushions arithmetic in the allowance calculation itself.
    """
    products = first*second.T
    value = np.sum(products)
    inherited = np.sum(np.abs(first)*error_second.T
                       + error_first*np.abs(second.T) + error_first*error_second.T)
    summation = _gamma(2*first.size)*np.sum(np.abs(products))
    return float(value), float(BOUND_SAFETY*(inherited+summation))


def _energies(before, delta, gram_reference_scale=1.):
    """Directed S=F D^T; Gram change compares (qF+D) to qF, not F+D.

    G=F^T F, H=D^T D, T=F^T D. Then
      ||sym(S)||^2  = [tr(GH)+tr(T^2)]/2,
      ||skew(S)||^2 = [tr(GH)-tr(T^2)]/2,
      ||delta Gram||^2 = 2q^2[tr(GH)+tr(T^2)]+4q tr(TH)+tr(H^2).
    The last identity keeps scale-projected motion tied to the intended pair
    of geometric endpoints. q is NOT inferred optimizer weight decay.
    """
    rows, width = before.shape
    q = float(gram_reference_scale)
    if not np.isfinite(q) or q < 0:
        raise ValueError("Gram reference scale must be finite and nonnegative")
    with np.errstate(over="ignore", invalid="ignore"):
        g, h, t = before.T@before, delta.T@delta, before.T@delta
    if not all(np.isfinite(a).all() for a in (g, h, t)):
        raise ArithmeticError("nonfinite coordinate Gram contraction")
    # gamma(2N) bounds ordinary dot products, including a non-FMA path.
    # Cauchy-Schwarz bounds |F|^T|D| using column norms, avoiding extra
    # vocabulary-sized temporaries or vocabulary-by-vocabulary products.
    gamma = _gamma(2*rows)
    f_upper = np.sqrt(np.maximum(np.diag(g), 0)/(1-gamma))*(1+8*EPS)
    d_upper = np.sqrt(np.maximum(np.diag(h), 0)/(1-gamma))*(1+8*EPS)
    eg = gamma*np.outer(f_upper, f_upper)
    eh = gamma*np.outer(d_upper, d_upper)
    et = gamma*np.outer(f_upper, d_upper)
    gh, egh = _trace_product(g, h, eg, eh)
    tt, ett = _trace_product(t, t, et, et)
    th, eth = _trace_product(t, h, et, eh)
    hh, ehh = _trace_product(h, h, eh, eh)
    pair_allowance = .5*(egh+ett)+_gamma(4)*(.5*abs(gh)+.5*abs(tt))
    terms = np.array([2*q*q*gh, 2*q*q*tt, 4*q*th, hh])
    gram_value = float(terms.sum())
    gram_allowance = 2*q*q*(egh+ett)+4*q*eth+ehh+_gamma(8)*float(np.abs(terms).sum())
    return {
        "delta_frobenius_norm": float(np.linalg.norm(delta)),
        "score_frobenius": _squared_norm(gh, egh, "score"),
        "score_symmetric_frobenius": _squared_norm(.5*(gh+tt), pair_allowance, "symmetric score"),
        "score_antisymmetric_frobenius": _squared_norm(.5*(gh-tt), pair_allowance, "antisymmetric score"),
        "gram_change_frobenius": _squared_norm(gram_value, gram_allowance, "Gram change"),
        "gram_reference_scale": q,
        "directed_score_definition": "S=F_before D^T",
        "gram_change_definition": "(q*F_before+D)(q*F_before+D)^T-q^2*F_before*F_before^T",
        "coordinate_cross_skew_norm": float(np.linalg.norm(t-t.T)),
    }


def centered_motion(before, after):
    """Compare two full-logical-vocabulary endpoints without estimating gradients.

    Nonzero rank-deficient cross-covariance has a nonunique Procrustes rotation;
    we retain raw metrics but decline to report an arbitrary aligned directed
    score. Both-zero centered endpoints have unambiguous zero motion, although
    their fitted scale is undefined. No vocabulary subsampling is performed.
    """
    before, after = _checked(before, "before"), _checked(after, "after")
    if before.shape != after.shape:
        raise ValueError("embedding endpoint shapes must agree")
    with np.errstate(over="ignore", invalid="ignore"):
        before_mean, after_mean = before.mean(axis=0), after.mean(axis=0)
        f0, f1 = before-before_mean, after-after_mean
    if not np.isfinite(f0).all() or not np.isfinite(f1).all():
        raise ArithmeticError("centering exceeded finite FP64 range")
    delta = f1-f0
    raw = _energies(f0, delta)
    norms = (float(np.linalg.norm(f0)), float(np.linalg.norm(f1)))
    uncentered_delta = after-before
    full_delta_squared = float(np.sum(uncentered_delta**2))
    shared_mean_squared = float(len(before)*np.sum((after_mean-before_mean)**2))
    centered_delta_squared = float(np.sum(delta**2))
    # Orthogonality of a constant-row shift and centered motion gives this
    # exact-real decomposition. Report its FP64 replay residual explicitly.
    mean_replay_error = full_delta_squared-shared_mean_squared-centered_delta_squared
    result = {
        "shape": list(before.shape), "centering": "each endpoint's ALL logical vocabulary rows",
        "mean_before": before_mean.tolist(), "mean_after": after_mean.tolist(),
        "mean_motion_norm": float(np.linalg.norm(after_mean-before_mean)),
        "full_raw_delta_frobenius_norm": float(np.sqrt(full_delta_squared)),
        "shared_mean_energy_fraction": (shared_mean_squared/full_delta_squared
                                        if full_delta_squared else None),
        "mean_energy_decomposition": {
            "definition": "||E_after-E_before||^2 = V*||mean_after-mean_before||^2 + ||F_after-F_before||^2",
            "full_delta_squared_norm": full_delta_squared,
            "shared_mean_squared_norm": shared_mean_squared,
            "centered_delta_squared_norm": centered_delta_squared,
            "signed_fp64_replay_error": mean_replay_error,
        },
        "centered_before_frobenius_norm": norms[0], "centered_after_frobenius_norm": norms[1],
        "raw": raw, "procrustes": None, "rotation_and_fitted_scale": None,
        "fitted_scale": None,
        "numeric_policy": {"dtype": "FP64", "unit_roundoff_used": EPS,
                           "dot_gamma_operations": 2*len(f0), "trace_gamma_operations": 2*f0.shape[1]**2,
                           "allowance_safety_factor": BOUND_SAFETY,
                           "scope": "conditional on stored centered arrays; not interval certification of centering or SVD"},
    }
    if norms == (0., 0.):
        result.update({"alignment_status": "both_centered_zero", "procrustes": raw,
                       "rotation_and_fitted_scale": raw, "cross_covariance_rank": 0,
                       "fitted_scale_reason": "undefined with a zero baseline"})
        return result
    cross = f1.T@f0
    if not np.isfinite(cross).all():
        raise ArithmeticError("nonfinite Procrustes covariance")
    left, singular, right = np.linalg.svd(cross, full_matrices=False)
    tolerance = RANK_TOL_MULTIPLIER*EPS*max(f0.shape)*float(singular[0])
    rank = int(np.count_nonzero(singular>tolerance))
    result.update({"cross_covariance_singular_values": singular.tolist(),
                   "cross_covariance_rank": rank, "alignment_rank_tolerance": tolerance,
                   "alignment_rank_tolerance_rule": "64*eps*max(V,D)*largest_singular_value"})
    if rank != f0.shape[1]:
        result["alignment_status"] = "unavailable_nonunique_rank_deficient_cross_covariance"
        return result
    rotation = left@right
    aligned = f1@rotation
    q = float(np.sum(aligned*f0)/np.sum(f0*f0))
    result.update({
        "alignment_status": "full_rank_procrustes", "fitted_scale": q,
        "fitted_scale_definition": "argmin_q ||F_after R-q F_before||_F; a geometry fit, not AdamW decay",
        "rotation_frobenius_distance_from_identity": float(np.linalg.norm(rotation-np.eye(len(rotation)))),
        "rotation_orthogonality_residual": float(np.linalg.norm(rotation.T@rotation-np.eye(len(rotation)))),
        "rotation_sha256": hashlib.sha256(rotation.astype("<f8").tobytes()).hexdigest(),
        "procrustes": _energies(f0, aligned-f0),
        "rotation_and_fitted_scale": _energies(f0, aligned-q*f0, q),
    })
    return result


def _record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_report(path, report):
    payload = json.dumps(report, indent=2, allow_nan=False)+"\n"
    with Path(path).open("x", encoding="utf-8") as output:
        output.write(payload)


def audit(checkpoint_paths, output_path):
    """Four prespecified intervals; authenticate every one of 100 endpoint files."""
    started = time.monotonic()
    paths = [Path(path).resolve() for path in checkpoint_paths]
    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    if len(paths)!=5 or [p.name for p in paths] != [f"step_{step}" for step in STEPS]:
        raise ValueError("require exactly the ordered fixed steps12990..13030")
    if any(output_path.resolve().is_relative_to(path) for path in paths):
        raise ValueError("output must not be inside a checkpoint")
    sources = {name: _record(Path(__file__).with_name(name))
               for name in ("embedding_motion.py", "checkpoint.py")}
    checkpoints = [GPT2Checkpoint(path, check_finite=True) for path in paths]
    before = [checkpoint.provenance(hash_weights=True) for checkpoint in checkpoints]
    intervals = []
    for index in range(4):
        intervals.append({"before_step": STEPS[index], "after_step": STEPS[index+1],
                          "embedding_motion": centered_motion(checkpoints[index].token_embedding,
                                                               checkpoints[index+1].token_embedding)})
    if any(checkpoint.provenance(hash_weights=True)!=snapshot
           for checkpoint, snapshot in zip(checkpoints, before)):
        raise ValueError("checkpoint or source changed during audit")
    if any(_record(record["path"])!=record for record in sources.values()):
        raise ValueError("audit source changed during audit")
    report = {"schema_version": 1, "stage": "weight_only_embedding_geometry_not_gradient_or_text_recovery",
              "sources": sources, "endpoints": before, "intervals": intervals,
              "all_500_endpoint_weight_hashes_unchanged": True,
              "runtime_environment": {"numpy": np.__version__,
                                      "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                                      "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
              "elapsed_seconds": time.monotonic()-started,
              "limitations": [
                  "All logical embedding rows include tied input/head effects; motion is not a raw gradient or token-occurrence certificate.",
                  "Rigid embedding-geometry rotation is not asserted to be a function-preserving GPT-2 reparameterization.",
                  "Fitted scale is descriptive geometric projection, not authenticated weight decay; genuine learned motion may be removed.",
                  "Antisymmetric directed motion does not identify source-token order; exact pairwise Gram change is symmetric.",
                  "Numerical cancellation allowances do not certify SVD alignment or hidden training history.",
              ]}
    write_report(output_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoints", nargs=5, type=Path)
    source.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = args.checkpoints or [args.checkpoint_root/f"step_{step}" for step in STEPS]
    audit(paths, args.output)


if __name__ == "__main__":
    main()
