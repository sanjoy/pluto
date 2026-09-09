"""Sufficient ideal-arithmetic rank bounds for a two-position attention Taylor term.

For head gap g and output-projected value difference v, the exact correction
to uniform routing is .5*tanh(g/2)*v. The linear correction is g*v/4, whose
error is at most min(|g|**3/48, |g|/4)*||v||. Sum these head bounds by the
triangle inequality to bound the TOTAL vector error, even if heads cancel.
The unchanged uniform-routing baseline and output bias do not affect error.

No attention or model is evaluated here. Floating-point computations of these
mathematical bounds are diagnostics, not machine-verified interval proofs.
They concern a fixed two-position subproblem, not the full GPT-2 prediction,
beam/path ranking, or actual corpus accuracy.
"""

import numpy as np


def error_bound(gaps, head_delta_norms):
    """Upper bound on ||exact_two_position_write - (w0+w1)|| in real arithmetic.

    Inputs contain one score gap and one projected value-difference norm per
    head. This is a bound on the norm of the SUM, not a stacked-head lower
    bound. Clipping before squaring avoids unnecessary overflow for large g.
    """
    gaps, norms = (np.asarray(x, dtype=np.float64) for x in (gaps, head_delta_norms))
    if (gaps.ndim != 1 or not len(gaps) or norms.shape != gaps.shape or
            not np.isfinite(gaps).all() or not np.isfinite(norms).all() or
            np.any(norms < 0)):
        raise ValueError('expected finite head gaps and nonnegative projected value norms')
    absolute = np.abs(gaps)
    factor = np.minimum(absolute / np.sqrt(12.), 1.)
    with np.errstate(over='ignore', invalid='ignore'):
        # Multiply large value norms before applying the two small factors.
        # Squaring a tiny gap first can underflow even when the final bound
        # is representable (for example gap=1e-200, value norm=1e300).
        result = float(np.sum((((absolute / 4) * norms) * factor) * factor))
    if not np.isfinite(result):
        raise ValueError('nonfinite attention error bound')
    return result


def top1_certificate(margin_raw, vector_error_bound, best_target_norm, max_target_norm):
    """A conservative sufficient condition for preserving a UNIQUE top token.

    margin_raw is the approximate raw dot-score difference between the first
    and second destinations, BEFORE division by write norm. If vector error
    is <=e, every pairwise score contrast changes by at most
    e*||target_best-target_other|| <= e*(||target_best||+max||target||).
    A strictly larger approximate margin therefore preserves its top1 choice.

    Targets must share the same centering used in the scored dictionary.
    Inputs must describe the SAME write and candidate set. False means only
    that this bound is inconclusive; it does not establish a ranking change.
    Callers must not interpret this as an interval-arithmetic or full-model
    certificate: input computations and comparisons here use ordinary FP64.
    """
    values = [float(x) for x in (margin_raw, vector_error_bound,
                                 best_target_norm, max_target_norm)]
    if not all(np.isfinite(x) and x >= 0 for x in values):
        raise ValueError('margin, error bound and target norms must be finite and nonnegative')
    margin, error, best, maximum = values
    if best > maximum:
        raise ValueError('best target norm exceeds maximum target norm')
    with np.errstate(over='ignore', invalid='ignore'):
        bound = float(np.float64(error) * (np.float64(best) + maximum))
    if not np.isfinite(bound):
        raise ValueError('nonfinite pairwise score error bound')
    return {
        'raw_top1_margin': margin,
        'vector_error_upper_bound': error,
        'pairwise_score_error_upper_bound': bound,
        'ideal_arithmetic_sufficient_condition_holds': margin > bound,
        'scope': 'unique top1 of exact two-position attention over this target set; not full model or paths',
        'numerics': 'ordinary FP64 diagnostic, not an interval-arithmetic proof',
    }
