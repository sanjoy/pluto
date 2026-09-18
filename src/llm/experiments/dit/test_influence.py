"""Analytical DIT oracles, independent of the reverse-adjoint implementation.

These tests distinguish the paper's first-order removal sensitivity from a
finite leave-one-out retraining effect. Both are useful quantities, but only the
former is exactly described by the Hessian recurrence. All small mathematical
oracles use double precision and explicit derivatives, not the production HVP.
"""

from pathlib import Path
import tempfile
import unittest

import torch

from src.llm.experiments.dit.influence import (
    compute_influence,
    hessian_vector_product,
    validate_leave_one_out,
)
from src.llm.experiments.dit.trajectory import train_trajectory


class ScalarModel(torch.nn.Module):
    def __init__(self, initial=1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float64))


class VectorModel(torch.nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float64))


def scalar_quadratic_loss(model, batch):
    return 0.5 * batch[:, 0] * model.weight.square()


def vector_quadratic_loss(model, batch):
    """Five numbers encode a symmetric 2x2 Hessian and a linear term."""
    x, y = model.weight
    return (
        0.5 * batch[:, 0] * x.square()
        + batch[:, 1] * x * y
        + 0.5 * batch[:, 2] * y.square()
        + batch[:, 3] * x
        + batch[:, 4] * y
    )


class HessianVectorProductTest(unittest.TestCase):
    def test_indefinite_hessian_including_unused_and_constant_gradients(self):
        nonlinear = torch.nn.Parameter(torch.tensor([0.2, -0.3], dtype=torch.float64))
        linear = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        unused = torch.nn.Parameter(torch.tensor([9.0], dtype=torch.float64))
        parameters = (nonlinear, linear, unused)
        vector = (
            torch.tensor([0.7, -0.4], dtype=torch.float64),
            torch.tensor([1.2], dtype=torch.float64),
            torch.tensor([3.0], dtype=torch.float64),
        )
        x, y = nonlinear
        loss = x.square() + 3 * x * y - 2 * y.square() + 4 * linear[0]
        actual = hessian_vector_product(loss, parameters, vector)
        expected = (
            torch.tensor([0.2, 3.7], dtype=torch.float64),
            torch.zeros_like(linear),
            torch.zeros_like(unused),
        )
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=1e-14)
        # A gradient-outer-product / Fisher substitute is PSD and cannot have
        # this negative quadratic form. This checks *loss* curvature directly.
        self.assertLess(sum((v * h).sum().item() for v, h in zip(vector, actual)), 0)

    def test_fully_constant_and_linear_losses_have_zero_hessian(self):
        parameter = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
        for loss in (torch.tensor(7.0, dtype=torch.float64), parameter.sum() * 3):
            with self.subTest(loss=loss):
                actual = hessian_vector_product(
                    loss, (parameter,), (torch.ones_like(parameter),)
                )
                torch.testing.assert_close(
                    actual[0], torch.zeros_like(parameter), rtol=0, atol=0
                )

    def test_does_not_write_parameter_gradient_accumulators(self):
        parameter = torch.nn.Parameter(torch.tensor([0.5, -0.2], dtype=torch.float64))
        parameter.grad = torch.tensor([17.0, 23.0], dtype=torch.float64)
        original_gradient = parameter.grad
        before = parameter.grad.clone()
        hessian_vector_product(
            parameter.pow(4).sum(), (parameter,), (torch.ones_like(parameter),)
        )
        self.assertIs(parameter.grad, original_gradient)
        torch.testing.assert_close(parameter.grad, before, rtol=0, atol=0)

    def test_direction_is_held_constant_during_differentiation(self):
        parameter = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
        # Differentiating this direction as well would incorrectly double the
        # answer instead of computing Hessian(loss)*direction with direction fixed.
        direction = parameter * 3
        result = hessian_vector_product(
            0.5 * parameter.square().sum(), (parameter,), (direction,)
        )
        torch.testing.assert_close(
            result[0], torch.tensor([6.0], dtype=torch.float64), rtol=0, atol=0
        )

    def test_rejects_invalid_hessian_vector_shapes_types_and_lengths(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        for vector in (
            (),
            (torch.ones(2, dtype=torch.float64),),
            (torch.ones(1, dtype=torch.float32),),
        ):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                hessian_vector_product(parameter.square().sum(), (parameter,), vector)


class InfluenceTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.next_directory = 0

    def train(self, model, examples, schedule, rates, loss_fn, interval=2):
        directory = Path(self.temporary.name) / f"trajectory_{self.next_directory}"
        self.next_directory += 1
        return train_trajectory(
            model,
            examples,
            directory,
            steps=len(schedule),
            batch_size=len(schedule[0]),
            learning_rate=rates,
            seed=123,
            checkpoint_interval=interval,
            loss_fn=loss_fn,
            schedule=schedule,
        )

    def scalar_trajectory(self, *, interval=2):
        model = ScalarModel()
        examples = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
        trajectory = self.train(
            model,
            examples,
            [[0], [1]],
            [0.1, 0.2],
            scalar_quadratic_loss,
            interval=interval,
        )
        return model, trajectory

    def test_pre_window_history_matters_without_any_in_window_occurrence(self):
        model, trajectory = self.scalar_trajectory()
        result = compute_influence(
            model,
            trajectory,
            scalar_quadratic_loss,
            lambda m: m.weight,
            t1=1,
            t2=2,
            sample_ids=[0],
        )
        # theta=(1,.9,.54), theta_without_0=(1,1,.6). Sample 0 is
        # absent from step 1, but the earlier displacement contracts there.
        self.assertEqual(set(result.scores), {0})
        self.assertAlmostEqual(result.scores[0], -0.04, places=14)
        self.assertAlmostEqual(result.query_start, 0.9, places=14)
        self.assertAlmostEqual(result.query_end, 0.54, places=14)

    def test_endpoint_query_gradients_differ_and_finite_loo_is_not_dit(self):
        model, trajectory = self.scalar_trajectory()
        query = lambda m: 0.5 * m.weight.square()
        result = compute_influence(
            model, trajectory, scalar_quadratic_loss, query, t1=1, t2=2, sample_ids=[0]
        )
        actual_loo = validate_leave_one_out(
            model, trajectory, scalar_quadratic_loss, query, t1=1, t2=2, sample_ids=[0]
        )
        # DIT is q_end*.06 - q_start*.10. The exact loss comparison has
        # additional quadratic displacement terms, which must not be hidden.
        self.assertAlmostEqual(result.scores[0], -0.0576, places=14)
        self.assertAlmostEqual(actual_loo[0], -0.0608, places=14)
        self.assertGreater(abs(result.scores[0] - actual_loo[0]), 0.003)

    def test_original_batch_denominator_survives_leave_one_out(self):
        model = ScalarModel()
        examples = torch.tensor([[1.0], [3.0]], dtype=torch.float64)
        trajectory = self.train(model, examples, [[0, 1]], [0.1], scalar_quadratic_loss)
        # Base update uses gradient (1+3)/2 -> theta=.8. Removing sample 0
        # keeps denominator 2 -> theta=.85, not .7 from renormalizing by 1.
        query = lambda m: m.weight
        result = compute_influence(
            model, trajectory, scalar_quadratic_loss, query, t1=0, t2=1, sample_ids=[0]
        )
        actual_loo = validate_leave_one_out(
            model, trajectory, scalar_quadratic_loss, query, t1=0, t2=1, sample_ids=[0]
        )
        self.assertAlmostEqual(result.scores[0], 0.05, places=14)
        self.assertAlmostEqual(actual_loo[0], 0.05, places=14)

    def test_feature_gradient_query_supports_both_dit_and_finite_loo(self):
        model, trajectory = self.scalar_trajectory()

        def feature_query(m):
            # A feature-importance query in Appendix C-C's sense: first
            # differentiate a prediction with respect to its input, then DIT
            # differentiates that scalar with respect to model parameters.
            # Query input 1.5 is fixed across all endpoints/counterfactuals.
            feature = m.weight.new_tensor(1.5, requires_grad=True)
            prediction = m.weight.square() * feature.square()
            return torch.autograd.grad(prediction, feature, create_graph=True)[0]

        # Supply unrelated caller state, so successful inference must not leave
        # the model at either endpoint or on a counterfactual training path.
        with torch.no_grad():
            model.weight.fill_(2.7)
        model.eval()
        model.weight.grad = model.weight.new_tensor(0.25)
        original_gradient = model.weight.grad
        original_weight = model.weight.detach().clone()

        result = compute_influence(
            model,
            trajectory,
            scalar_quadratic_loss,
            feature_query,
            t1=1,
            t2=2,
            sample_ids=[0],
        )
        # Feature query = 3*theta^2; its parameter gradient = 6*theta.
        # Original theta endpoints (.9,.54), deletion displacements (.1,.06).
        self.assertAlmostEqual(
            result.scores[0], 6 * 0.54 * 0.06 - 6 * 0.9 * 0.1, places=14
        )
        self.assertAlmostEqual(result.query_start, 3 * 0.9**2, places=14)
        self.assertAlmostEqual(result.query_end, 3 * 0.54**2, places=14)
        torch.testing.assert_close(model.weight, original_weight, rtol=0, atol=0)
        self.assertIs(model.weight.grad, original_gradient)
        self.assertFalse(model.training)

        actual_loo = validate_leave_one_out(
            model,
            trajectory,
            scalar_quadratic_loss,
            feature_query,
            t1=1,
            t2=2,
            sample_ids=[0],
        )
        # Removing sample 0 gives theta endpoints (1,.6), so evaluate the
        # feature query itself, not its linearized parameter-gradient formula.
        expected_loo = 3 * (0.6**2 - 0.54**2) - 3 * (1.0**2 - 0.9**2)
        self.assertAlmostEqual(actual_loo[0], expected_loo, places=14)
        torch.testing.assert_close(model.weight, original_weight, rtol=0, atol=0)
        self.assertIs(model.weight.grad, original_gradient)
        self.assertEqual(model.weight.grad.item(), 0.25)
        self.assertFalse(model.training)

    def test_noncommuting_hessians_against_independent_forward_sensitivity(self):
        examples = torch.tensor(
            [
                [2.0, 0.3, 1.0, 0.2, -0.1],
                [0.4, -0.5, 1.6, -0.3, 0.5],
                [-0.8, 0.6, 0.9, 0.1, 0.4],
            ],
            dtype=torch.float64,
        )
        schedule = [[0, 1], [1, 2], [0, 2], [0, 1]]
        rates = [0.1, 0.07, 0.05, 0.11]
        model = VectorModel([0.8, -0.6])
        trajectory = self.train(model, examples, schedule, rates, vector_quadratic_loss)
        hessians = torch.stack(
            [torch.tensor([[row[0], row[1]], [row[1], row[2]]]) for row in examples]
        )
        linear = examples[:, 3:]
        self.assertGreater(
            (hessians[0] @ hessians[1] - hessians[1] @ hessians[0]).abs().max().item(),
            0.1,
        )
        # Explicit polynomial derivatives and forward-time sensitivities are
        # deliberately independent of production autograd/adjoint/replay code.
        states = [torch.tensor([0.8, -0.6], dtype=torch.float64)]
        sensitivities = [torch.zeros((3, 2), dtype=torch.float64)]
        identity = torch.eye(2, dtype=torch.float64)
        for batch, rate in zip(schedule, rates):
            gradients = (hessians @ states[-1]) + linear
            average_hessian = hessians[batch].mean(0)
            next_sensitivity = (
                sensitivities[-1] @ (identity - rate * average_hessian).t()
            )
            for sample_id in batch:
                next_sensitivity[sample_id] += rate / len(batch) * gradients[sample_id]
            sensitivities.append(next_sensitivity)
            states.append(states[-1] - rate * gradients[batch].mean(0))
        query_hessian = torch.tensor([[1.0, 0.2], [0.2, -0.7]], dtype=torch.float64)
        query_linear = torch.tensor([0.4, 0.1], dtype=torch.float64)

        def query(m):
            return 0.5 * m.weight @ query_hessian @ m.weight + query_linear @ m.weight

        observed = {}
        for start, end in ((0, 1), (0, 4), (1, 4), (2, 3)):
            with self.subTest(window=(start, end)):
                result = compute_influence(
                    model, trajectory, vector_quadratic_loss, query, t1=start, t2=end
                )
                expected = sensitivities[end] @ (
                    query_hessian @ states[end] + query_linear
                ) - sensitivities[start] @ (
                    query_hessian @ states[start] + query_linear
                )
                self.assertEqual(set(result.scores), {0, 1, 2})
                for sample_id in range(3):
                    self.assertAlmostEqual(
                        result.scores[sample_id], expected[sample_id].item(), places=13
                    )
                observed[start, end] = result.scores
        for sample_id in range(3):
            self.assertAlmostEqual(
                observed[0, 4][sample_id],
                observed[0, 1][sample_id] + observed[1, 4][sample_id],
                places=13,
            )

    def test_nonlinear_multiple_steps_match_infinitesimal_downweighting(self):
        examples = torch.tensor(
            [[0.6, 0.3, 0.2], [1.2, -0.1, -0.4], [0.1, 0.8, 0.5]], dtype=torch.float64
        )
        schedule = [[0, 1], [1, 2], [0, 2], [1, 0]]
        rates = [0.04, 0.03, 0.05, 0.02]
        model = ScalarModel(0.7)

        def loss(m, batch):
            return (
                0.25 * batch[:, 0] * m.weight.pow(4)
                + 0.5 * batch[:, 1] * m.weight.square()
                - batch[:, 2] * m.weight
            )

        trajectory = self.train(model, examples, schedule, rates, loss)
        query = lambda m: m.weight.square() + 0.3 * m.weight
        result = compute_influence(model, trajectory, loss, query, t1=1, t2=4)

        def independently_retrain(sample_id, amount):
            # amount=0 is original training; amount=1 is full deletion. This
            # independent scalar arithmetic avoids autograd and sgd_step.
            theta = 0.7
            values = [theta * theta + 0.3 * theta]
            rows = examples.tolist()
            for batch, rate in zip(schedule, rates):
                gradient = 0.0
                for i in batch:
                    a, b, c = rows[i]
                    weight = 1 - amount if i == sample_id else 1
                    gradient += weight * (a * theta**3 + b * theta - c)
                theta -= rate * gradient / len(batch)
                values.append(theta * theta + 0.3 * theta)
            return values[4] - values[1]

        for sample_id in range(3):
            epsilon = 1e-5
            derivative = (
                independently_retrain(sample_id, epsilon)
                - independently_retrain(sample_id, -epsilon)
            ) / (2 * epsilon)
            self.assertAlmostEqual(result.scores[sample_id], derivative, places=9)

    def test_checkpoint_replay_spacing_does_not_change_influence(self):
        dense_model, dense = self.scalar_trajectory(interval=1)
        sparse_model, sparse = self.scalar_trajectory(interval=7)
        query = lambda m: m.weight.square()
        dense_result = compute_influence(
            dense_model, dense, scalar_quadratic_loss, query, t1=1, t2=2
        )
        sparse_result = compute_influence(
            sparse_model, sparse, scalar_quadratic_loss, query, t1=1, t2=2
        )
        self.assertEqual(dense_result.scores, sparse_result.scores)

    def test_contributions_and_progress_keep_the_full_reverse_history(self):
        model, trajectory = self.scalar_trajectory()
        progress = []
        result = compute_influence(
            model,
            trajectory,
            scalar_quadratic_loss,
            lambda m: m.weight,
            t1=1,
            t2=2,
            progress=lambda step, count: progress.append((step, count)),
        )
        self.assertEqual(progress, [(1, 1), (0, 1)])
        self.assertEqual(
            [(c["step"], c["example_id"]) for c in result.contributions],
            [(1, 1), (0, 0)],
        )
        for sample_id, score in result.scores.items():
            self.assertEqual(
                score,
                sum(
                    c["value"]
                    for c in result.contributions
                    if c["example_id"] == sample_id
                ),
            )
        selected = compute_influence(
            model,
            trajectory,
            scalar_quadratic_loss,
            lambda m: m.weight,
            t1=1,
            t2=2,
            sample_ids=[0],
        )
        self.assertEqual(selected.scores[0], result.scores[0])
        self.assertTrue(all(c["example_id"] == 0 for c in selected.contributions))

    def test_unused_training_example_and_constant_query_have_zero_scores(self):
        model = ScalarModel()
        examples = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        trajectory = self.train(
            model, examples, [[0], [1]], [0.1, 0.2], scalar_quadratic_loss
        )
        result = compute_influence(
            model, trajectory, scalar_quadratic_loss, lambda m: m.weight, t1=0, t2=2
        )
        self.assertEqual(result.scores[2], 0)
        constant = compute_influence(
            model,
            trajectory,
            scalar_quadratic_loss,
            lambda m: m.weight.new_tensor(17.0),
            t1=1,
            t2=2,
        )
        self.assertEqual(constant.scores, {0: 0.0, 1: 0.0, 2: 0.0})

    def test_restores_parameters_buffers_modes_and_gradients(self):
        model = ScalarModel()
        model.child = torch.nn.Linear(1, 1, dtype=torch.float64)
        model.register_buffer("marker", torch.tensor([7.0], dtype=torch.float64))
        examples = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
        trajectory = self.train(
            model, examples, [[0], [1]], [0.1, 0.2], scalar_quadratic_loss
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(2.5)
            model.marker.add_(9.0)
        model.train()
        model.child.eval()
        for index, parameter in enumerate(model.parameters()):
            parameter.grad = (
                None if index == 1 else torch.full_like(parameter, index + 0.125)
            )
        parameters = tuple(model.parameters())
        before_state = {
            name: value.clone() for name, value in model.state_dict().items()
        }
        before_modes = [module.training for module in model.modules()]
        before_grads = [None if p.grad is None else p.grad.clone() for p in parameters]

        def assert_unchanged():
            for name, expected in before_state.items():
                torch.testing.assert_close(
                    model.state_dict()[name], expected, rtol=0, atol=0
                )
            self.assertEqual(
                [module.training for module in model.modules()], before_modes
            )
            for parameter, expected in zip(parameters, before_grads):
                if expected is None:
                    self.assertIsNone(parameter.grad)
                else:
                    torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)

        for operation in (compute_influence, validate_leave_one_out):
            with self.subTest(operation=operation.__name__, failure=False):
                operation(
                    model,
                    trajectory,
                    scalar_quadratic_loss,
                    lambda m: m.weight.square(),
                    t1=1,
                    t2=2,
                    sample_ids=[0],
                )
                assert_unchanged()

            def failing_query(m):
                raise RuntimeError("intentional query failure")

            with self.subTest(operation=operation.__name__, failure=True):
                with self.assertRaisesRegex(RuntimeError, "intentional query failure"):
                    operation(
                        model,
                        trajectory,
                        scalar_quadratic_loss,
                        failing_query,
                        t1=1,
                        t2=2,
                        sample_ids=[0],
                    )
                assert_unchanged()

        def failing_progress(step, count):
            raise RuntimeError("intentional progress failure")

        # A callback failure happens after the first score/HVP, not just while
        # loading an endpoint. Restoration must cover partial computations too.
        with self.assertRaisesRegex(RuntimeError, "intentional progress failure"):
            compute_influence(
                model,
                trajectory,
                scalar_quadratic_loss,
                lambda m: m.weight.square(),
                t1=1,
                t2=2,
                progress=failing_progress,
            )
        assert_unchanged()

    def test_rejects_invalid_windows_and_sample_ids(self):
        model, trajectory = self.scalar_trajectory()
        for start, end in ((-1, 1), (0, 3), (2, 1), (1, 1)):
            with self.subTest(window=(start, end)), self.assertRaises(ValueError):
                compute_influence(
                    model,
                    trajectory,
                    scalar_quadratic_loss,
                    lambda m: m.weight,
                    t1=start,
                    t2=end,
                )
        for ids in ([], [-1], [2], [0, 0], [True], [0.0]):
            with self.subTest(sample_ids=ids), self.assertRaises(ValueError):
                compute_influence(
                    model,
                    trajectory,
                    scalar_quadratic_loss,
                    lambda m: m.weight,
                    t1=0,
                    t2=2,
                    sample_ids=ids,
                )

    def test_rejects_vector_and_nonfinite_query(self):
        model, trajectory = self.scalar_trajectory()
        for query in (
            lambda m: torch.stack((m.weight, m.weight)),
            lambda m: m.weight * float("nan"),
        ):
            with self.subTest(query=query), self.assertRaises(ValueError):
                compute_influence(
                    model, trajectory, scalar_quadratic_loss, query, t1=0, t2=2
                )


if __name__ == "__main__":
    unittest.main()
