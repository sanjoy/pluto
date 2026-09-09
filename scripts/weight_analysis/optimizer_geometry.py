"""Small ideal-arithmetic tests of what AdamW updates do and do not identify.

This is a CPU algebra tool, not optimizer bitwise emulation, model inference,
gradient inversion of real checkpoints, or a text decoder. In particular,
low matrix rank and preservation of a particular input subspace are different
properties. Elementwise adaptive scaling can destroy both.
"""

import argparse
from pathlib import Path

import numpy as np

from .late_mlp_paths import file_record, runtime_environment, write_exclusive


def finite_matrix(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not all(value.shape) or not np.isfinite(value).all():
        raise ValueError("expected a nonempty finite matrix")
    return value


def span_metrics(reference, observed):
    """Describe rank and observed energy outside a known reference column span.

    SVD rank uses the usual floating-point scale threshold, not a claim of
    algebraic rank for noisy measurements. A full-rank observed matrix can
    contain the reference span while failing to identify it specifically.
    """
    reference, observed = map(finite_matrix, (reference, observed))
    if reference.shape[0] != observed.shape[0]:
        raise ValueError("column spans must inhabit the same ambient space")
    u, singular, _ = np.linalg.svd(reference, full_matrices=False)
    other = np.linalg.svd(observed, compute_uv=False)
    cutoff = np.finfo(np.float64).eps * max(reference.shape) * singular[0]
    other_cutoff = np.finfo(np.float64).eps * max(observed.shape) * other[0]
    rank = int(np.count_nonzero(singular > cutoff))
    basis = u[:, :rank]
    residual = observed - basis @ (basis.T @ observed)
    norm = float(np.linalg.norm(observed))
    return {"reference_rank": rank, "observed_rank": int(np.count_nonzero(other > other_cutoff)),
            "reference_singular_values": singular.tolist(), "observed_singular_values": other.tolist(),
            "reference_rank_cutoff": float(cutoff), "observed_rank_cutoff": float(other_cutoff),
            "relative_energy_outside_reference_span": float(np.linalg.norm(residual)**2 / norm**2) if norm else 0.,
            "observed_zero": norm == 0.}


def simulate_adamw(initial, gradients, learning_rate=.0003, beta1=.9,
                   beta2=.95, epsilon=1e-8, weight_decay=.1):
    """Known zero moments, known uninterrupted local step count, ideal FP64.

    Store all moments only to demonstrate conditional recovery. Real Pluto
    checkpoints do not supply these arrays. Multiplication/update ordering and
    powers here are not the cuTile kernel's FP32 arithmetic.
    """
    initial = finite_matrix(initial)
    gradients = np.asarray(gradients, dtype=np.float64)
    if (gradients.ndim != 3 or not len(gradients) or gradients.shape[1:] != initial.shape
            or not np.isfinite(gradients).all()):
        raise ValueError("expected finite [steps,rows,columns] gradients")
    if (not all(np.isfinite(x) for x in (learning_rate, beta1, beta2, epsilon, weight_decay))
            or learning_rate <= 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1
            or epsilon <= 0 or weight_decay < 0):
        raise ValueError("invalid AdamW configuration")
    weights = [initial.copy()]
    first = [np.zeros_like(initial)]
    second = [np.zeros_like(initial)]
    updates = []
    for step, gradient in enumerate(gradients, 1):
        m = beta1*first[-1] + (1-beta1)*gradient
        v = beta2*second[-1] + (1-beta2)*gradient**2
        update = (m/(1-beta1**step))/(np.sqrt(v/(1-beta2**step))+epsilon)
        weight = (1-learning_rate*weight_decay)*weights[-1] - learning_rate*update
        if not all(np.isfinite(x).all() for x in (m, v, update, weight)):
            raise ValueError("AdamW arithmetic exceeded finite FP64 range")
        first.append(m); second.append(v); updates.append(update); weights.append(weight)
    return {"weights": np.asarray(weights), "first": np.asarray(first),
            "second": np.asarray(second), "updates": np.asarray(updates)}


def recover_from_first_moments(previous, current, beta1):
    previous, current = map(finite_matrix, (previous, current))
    if previous.shape != current.shape or not np.isfinite(beta1) or not 0 <= beta1 < 1:
        raise ValueError("matching moment shapes and beta1 in [0,1) required")
    return (current-beta1*previous)/(1-beta1)


def invert_known_first_update(update, epsilon):
    """Conditional ideal inverse, NOT an inverse of arbitrary Adam snapshots.

    With known zero initial moments and bias correction on the very first
    step, u=g/(|g|+epsilon). For positive epsilon and exact |u|<1 this gives
    g=epsilon*u/(1-|u|). It becomes extremely ill-conditioned near |u|=1.
    Rounded weights, unknown moments, or ten-step sums invalidate this use.
    """
    update = finite_matrix(update)
    if not np.isfinite(epsilon) or epsilon <= 0 or np.any(np.abs(update) >= 1):
        raise ValueError("positive epsilon and finite |first update|<1 required")
    return epsilon*update/(1-np.abs(update))


def run_experiment():
    """Deterministic known-input-span controls, with no real checkpoint inputs."""
    grid = np.arange(6, dtype=np.float64)
    gradient = grid[:, None]-grid[None, :]+.25  # Rank two, but its sign pattern has full rank.
    first = simulate_adamw(np.zeros((6, 6)), gradient[None])
    # At epsilon=0, sign(x*y^T) remains rank one while changing its column
    # span from x to sign(x). This limiting geometry is NOT production epsilon.
    x = np.array([1., 2., 4., 8., 16., 32.])
    y = np.array([-3., -1., .5, 2., 5.])
    rank_one = x[:, None]*y
    rank_one_sign = np.sign(rank_one)
    rng = np.random.Generator(np.random.PCG64(20260909))
    inputs = np.column_stack((np.ones(6), grid))
    gradients = np.asarray([inputs @ rng.normal(size=(2, 6)) for _ in range(10)])
    initial = rng.normal(size=(6, 6))
    eta, decay, beta1 = .0003, .1, .9
    simulation = simulate_adamw(initial, gradients, learning_rate=eta, weight_decay=decay, beta1=beta1)
    q = 1-eta*decay
    endpoint = (q**10*initial-simulation['weights'][-1])/eta
    weighted_updates = sum(q**(9-j)*simulation['updates'][j] for j in range(10))
    recovered = np.asarray([recover_from_first_moments(simulation['first'][j], simulation['first'][j+1], beta1)
                            for j in range(10)])
    pure_decay = simulate_adamw(initial, np.zeros_like(gradients), learning_rate=eta, weight_decay=decay)
    decay_residual = (q**10*initial-pure_decay['weights'][-1])/eta
    conditional = invert_known_first_update(first['updates'][0], 1e-8)
    # Exact identical FINAL FP32 bits can hide different gradients. This is
    # explicitly a scalar CPU rounding illustration, not a production replay.
    gs = np.array([.001, .002])
    rounded = np.asarray(1-eta*(gs/(np.abs(gs)+1e-8)+decay), dtype='<f4')
    return {
        "schema_version": 1, "stage": "synthetic_optimizer_identifiability_controls",
        "seed": 20260909, "runtime": runtime_environment(),
        "rank_two_first_step": {"sgd": span_metrics(gradient, gradient),
                                "adam": span_metrics(gradient, first['updates'][0])},
        "rank_one_zero_epsilon_limit": span_metrics(rank_one, rank_one_sign),
        "ten_steps_fixed_input_span": {
            "sgd_sum": span_metrics(inputs, gradients.sum(axis=0)),
            "adam_endpoint": span_metrics(inputs, endpoint),
            "known_final_first_moment": span_metrics(inputs, simulation['first'][-1]),
            "endpoint_weighted_update_max_error": float(np.max(np.abs(endpoint-weighted_updates))),
            "recover_each_gradient_with_saved_moments_max_error": float(np.max(np.abs(recovered-gradients))),
            "pure_decay_corrected_residual_max_abs": float(np.max(np.abs(decay_residual)))},
        "known_first_step_ideal_inverse": {
            "max_gradient_error": float(np.max(np.abs(conditional-gradient))),
            "minimum_distance_of_update_to_unit_magnitude": float(np.min(1-np.abs(first['updates'][0]))),
            "maximum_inverse_absolute_derivative": float(np.max(1e-8/(1-np.abs(first['updates'][0]))**2)),
            "assumptions": "known zero moments, first local step, known epsilon and normalized update, ideal real arithmetic"},
        "fp32_scalar_collision": {"gradients": gs.tolist(), "rounded_weights": rounded.tolist(),
                                  "weight_bits_hex": [hex(int(x)) for x in rounded.view('<u4')],
                                  "same_bits": bool(rounded[0].view('<u4') == rounded[1].view('<u4'))},
        "limitations": ["Not a CUDA bitwise optimizer replay or real-gradient recovery.",
                        "Synthetic constant input span is a controlled positive case, not a real-batch assumption.",
                        "Rank or observed-span containment alone cannot identify tokens.",
                        "Conditional first-step invertibility does not imply ten-step checkpoint invertibility."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run_experiment()
    report['source'] = file_record(__file__)
    write_exclusive(args.output, report)


if __name__ == '__main__':
    main()
