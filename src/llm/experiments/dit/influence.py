"""Dynamic Influence Tracker (arXiv:2502.10793v1), using real loss Hessians.

The reverse pass computes the paper's linearized deletion influence; it does
not claim to reproduce finite leave-one-out retraining exactly. Equivalently,
it differentiates a continuous downweighting of one example at weight one.
The full original batch Hessian is essential, even when reporting only a few
examples. No inverse Hessian, stationarity assumption, Fisher approximation,
or gradient-similarity replacement is used here.
"""

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from src.llm.experiments.dit.trajectory import LossFunction, Trajectory, snapshot

QueryFunction = Callable[[torch.nn.Module], torch.Tensor]
Vector = tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class InfluenceResult:
    """Positive scores mean removal increases the specified scalar query.

    For a loss query, positive means the training example was helpful. For a
    target logit, positive means the example suppressed that logit. Contributions
    retain the signed kicks at each reverse step, and sum to scores by sample.
    A kick before t1 can affect a window [t1,t2]; it must not be discarded.
    """

    scores: dict[int, float]
    query_start: float
    query_end: float
    contributions: tuple[dict, ...]


@contextmanager
def _preserve_model(model):
    # Replay intentionally mutates parameters. Restore them even when a query,
    # callback, corrupt checkpoint, or unsupported higher derivative fails.
    state = snapshot(model)
    modes = [(module, module.training) for module in model.modules()]
    gradients = [(p, p.grad) for p in model.parameters()]
    try:
        yield
    finally:
        model.load_state_dict(state, strict=True)
        for module, training in modes:
            module.training = training
        for parameter, gradient in gradients:
            parameter.grad = gradient


def _parameters(model):
    # parameters() deduplicates tied tensors; in GPT-2, the lookup and LM head
    # share one embedding Parameter, whose two gradient paths must both count.
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return parameters


def _zeros(parameters):
    return tuple(torch.zeros_like(p) for p in parameters)


def _gradient(scalar, parameters, *, create_graph=False, retain_graph=False):
    if not isinstance(scalar, torch.Tensor) or scalar.ndim != 0:
        raise ValueError("query/loss must return a scalar Tensor")
    if not bool(torch.isfinite(scalar)):
        raise ValueError("non-finite query/loss")
    if not scalar.requires_grad:
        return _zeros(parameters)
    gradients = torch.autograd.grad(
        scalar,
        parameters,
        allow_unused=True,
        create_graph=create_graph,
        retain_graph=retain_graph or create_graph,
    )
    return tuple(
        torch.zeros_like(p) if g is None else g for p, g in zip(parameters, gradients)
    )


def hessian_vector_product(
    loss: torch.Tensor,
    parameters: Sequence[torch.Tensor],
    vector: Sequence[torch.Tensor],
) -> Vector:
    """Compute Hessian(loss) @ vector by differentiating its gradient.

    vector is a constant direction, not itself a function to differentiate.
    Unused parameters and linear losses have exactly zero Hessian entries.
    Keeping tensors separate avoids flattening/copying the whole model. This
    consumes the loss graph; callers needing first gradients must take them
    first with retain_graph=True. No dense Hessian is ever constructed.
    """
    parameters, vector = tuple(parameters), tuple(vector)
    if len(parameters) != len(vector) or not parameters:
        raise ValueError("HVP needs one vector tensor per parameter")
    for parameter, direction in zip(parameters, vector):
        if (
            parameter.shape != direction.shape
            or parameter.dtype != direction.dtype
            or parameter.device != direction.device
        ):
            raise ValueError("HVP vector shape/dtype/device differs from parameters")
    gradients = _gradient(loss, parameters, create_graph=True)
    terms = [
        (g * v.detach()).sum() for g, v in zip(gradients, vector) if g.requires_grad
    ]
    if not terms:
        return _zeros(parameters)
    return tuple(g.detach() for g in _gradient(sum(terms), parameters))


def _dot(left, right):
    # FP64 accumulation reduces cancellation in signed influence scores while
    # model gradients/HVPs retain the recorded training precision.
    return sum((a * b).sum(dtype=torch.float64) for a, b in zip(left, right))


def _validate_request(trajectory, t1, t2, sample_ids):
    if (
        type(t1) is not int
        or type(t2) is not int
        or not 0 <= t1 < t2 <= trajectory.steps
    ):
        raise ValueError("window must satisfy 0 <= t1 < t2 <= training steps")
    selected = tuple(
        range(trajectory.sample_count) if sample_ids is None else sample_ids
    )
    if (
        not selected
        or any(
            type(i) is not int or not 0 <= i < trajectory.sample_count for i in selected
        )
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("sample_ids must be nonempty, unique, valid example IDs")
    return selected


def compute_influence(
    model: torch.nn.Module,
    trajectory: Trajectory,
    loss_fn: LossFunction,
    query_fn: QueryFunction,
    *,
    t1: int,
    t2: int,
    sample_ids=None,
    progress=None
) -> InfluenceResult:
    """Algorithm 2's adjoint, implemented with a single combined query vector.

    Let delta_j[t] be the first-order parameter change from deleting example j:
      delta_j[t+1] = (I - eta_t H_t) delta_j[t] + eta_t/|S_t| g_j[t].
    The requested score is q[t2].delta_j[t2] - q[t1].delta_j[t1], with
    q[t] = grad query(theta_t). Starting at q[t2], propagate backwards through
    EVERY update to zero, subtracting q[t1] after processing update t1. This
    subtraction combines Algorithm 2's two adjoints and saves one HVP.

    q[t1] and q[t2] are evaluated separately, not frozen to the final model.
    The start=0 replay is intentional: Appendix D's truncated loop is not valid
    for arbitrary windows when earlier removals affect later parameter changes.
    progress(step, number_of_selected_examples_in_batch) runs in reverse order.
    All caller weights, gradient accumulators, and module modes are restored.
    """
    selected = _validate_request(trajectory, t1, t2, sample_ids)
    parameters = _parameters(model)
    scores = {i: 0.0 for i in selected}
    contributions = []
    with _preserve_model(model), torch.enable_grad():
        trajectory.load_state(model, t1, loss_fn)
        start = query_fn(model)
        start_gradient = _gradient(start, parameters) if t1 else None
        if (
            not isinstance(start, torch.Tensor)
            or start.ndim
            or not bool(torch.isfinite(start))
        ):
            raise ValueError("query must return a finite scalar Tensor")
        query_start = float(start.detach())
        del start
        trajectory.load_state(model, t2, loss_fn)
        end = query_fn(model)
        vector = tuple(g.detach() for g in _gradient(end, parameters))
        query_end = float(end.detach())
        del end

        for step, _ in trajectory.iter_reverse_segments(model, 0, t2, loss_fn):
            ids = trajectory.batch(step)
            rate = trajectory.learning_rate(step)
            batch = trajectory.examples[list(ids)].to(parameters[0].device)
            losses = loss_fn(model, batch)
            if (
                losses.ndim != 1
                or len(losses) != len(ids)
                or not bool(torch.isfinite(losses).all())
            ):
                raise ValueError("loss_fn must return finite per-example losses")
            count = 0
            for position, sample in enumerate(ids):
                if sample not in scores:
                    continue
                gradient = _gradient(losses[position], parameters, retain_graph=True)
                value = float((_dot(vector, gradient) * (rate / len(ids))).detach())
                scores[sample] += value
                contributions.append(
                    {"step": step, "example_id": sample, "value": value}
                )
                count += 1
                del gradient
            # No earlier injection exists after step zero, so its HVP would be
            # unused. All other propagation uses the ORIGINAL full batch.
            if step:
                hvp = hessian_vector_product(losses.mean(), parameters, vector)
                vector = tuple(v - rate * h for v, h in zip(vector, hvp))
                del hvp
                if step == t1:
                    vector = tuple(v - q for v, q in zip(vector, start_gradient))
                    start_gradient = None
                if not bool(torch.stack([v.isfinite().all() for v in vector]).all()):
                    raise ValueError(
                        "non-finite DIT adjoint; try a shorter/stabler trajectory"
                    )
            del losses
            if progress is not None:
                progress(step, count)
    if not all(
        torch.isfinite(torch.tensor(value, dtype=torch.float64))
        for value in scores.values()
    ):
        raise ValueError("non-finite DIT score")
    return InfluenceResult(scores, query_start, query_end, tuple(contributions))


def validate_leave_one_out(
    model: torch.nn.Module,
    trajectory: Trajectory,
    loss_fn: LossFunction,
    query_fn: QueryFunction,
    *,
    t1: int,
    t2: int,
    sample_ids=None
) -> dict[int, float]:
    """Measure actual scalar-query changes by retraining without each sample.

    Both endpoint counterfactuals start at theta_0, retain batch denominators,
    and use the same recorded schedule. This expensive validation is separate
    from DIT: a finite deletion need not equal an infinitesimal derivative.
    """
    selected = _validate_request(trajectory, t1, t2, sample_ids)
    result = {}
    with _preserve_model(model), torch.enable_grad():

        def value():
            # A scalar query may itself differentiate an input feature (the
            # paper's feature-importance toolkit), so keep autograd available.
            scalar = query_fn(model)
            if scalar.ndim or not bool(torch.isfinite(scalar)):
                raise ValueError("query must return a finite scalar Tensor")
            return float(scalar.detach())

        trajectory.load_state(model, t1, loss_fn)
        baseline_start = value()
        trajectory.load_state(model, t2, loss_fn)
        baseline_end = value()
        for sample in selected:
            trajectory.leave_one_out_state(model, sample, end=t1, loss_fn=loss_fn)
            removed_start = value()
            trajectory.leave_one_out_state(model, sample, end=t2, loss_fn=loss_fn)
            removed_end = value()
            result[sample] = (removed_end - baseline_end) - (
                removed_start - baseline_start
            )
    return result
