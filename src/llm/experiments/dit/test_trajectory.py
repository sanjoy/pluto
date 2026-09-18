"""Small, independently solvable SGD trajectories exercise the storage contract."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from src.llm.experiments.dit.trajectory import (
    Trajectory,
    sgd_step,
    snapshot,
    train_trajectory,
)


class Affine(torch.nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.4, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.tensor(-0.2, dtype=dtype))

    def forward(self, x):
        return self.weight * x + self.bias


def quadratic_loss(model, examples):
    return (model(examples[:, 0]) - examples[:, 1]).square() / 2


class Tied(torch.nn.Module):
    def __init__(self, tied=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.4, dtype=torch.float64))
        self.other = (
            self.weight if tied else torch.nn.Parameter(self.weight.detach().clone())
        )

    def forward(self, x):
        return (self.weight + self.other) * x


class TrajectoryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.examples = torch.tensor(
            [[1.0, 2.0], [-2.0, 1.0], [3.0, -1.0]], dtype=torch.float64
        )
        self.schedule = [[0, 2], [1, 2], [0, 1], [0, 2], [1, 2]]
        self.rates = [0.03, 0.02, 0.01, 0.04, 0.005]

    def train(self, name="run", model=None, **overrides):
        model = Affine() if model is None else model
        options = dict(
            steps=5,
            batch_size=2,
            learning_rate=self.rates,
            seed=17,
            checkpoint_interval=2,
            loss_fn=quadratic_loss,
            metadata={"purpose": "quadratic oracle", "unicode": "λ"},
            schedule=self.schedule,
        )
        options.update(overrides)
        return train_trajectory(model, self.examples, self.root / name, **options)

    def oracle(self, omitted=None):
        """Plain Python scalar calculus, independent of autograd/replay code."""
        w, b = 0.4, -0.2
        states = [(w, b)]
        for batch, rate in zip(self.schedule, self.rates):
            gw = gb = 0.0
            for sample in batch:
                if sample == omitted:
                    continue
                x, y = self.examples[sample].tolist()
                residual = w * x + b - y
                gw += residual * x / len(batch)
                gb += residual / len(batch)
            w -= rate * gw
            b -= rate * gb
            states.append((w, b))
        return states

    def assert_model(self, model, expected):
        self.assertAlmostEqual(float(model.weight.detach()), expected[0], places=14)
        self.assertAlmostEqual(float(model.bias.detach()), expected[1], places=14)

    def edit_manifest(self, change, name="run"):
        path = self.root / name / "manifest.json"
        manifest = json.loads(path.read_text())
        change(manifest)
        path.write_text(json.dumps(manifest))

    def rehash(self, manifest, descriptor, state):
        path = self.root / "run" / descriptor["file"]
        torch.save(state, path)
        descriptor["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def test_sgd_uses_original_mean_loss_and_no_optimizer_state(self):
        model = Affine()
        before = copy.deepcopy(model)
        expected_loss = float(
            quadratic_loss(before, self.examples[[0, 2]]).mean().detach()
        )
        actual_loss = sgd_step(
            model, self.examples, [0, 2], self.rates[0], quadratic_loss
        )
        self.assertEqual(actual_loss, expected_loss)
        self.assert_model(model, self.oracle()[1])
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_checkpoint_roundtrip_and_every_intermediate_match_scalar_oracle(self):
        model = Affine()
        trajectory = self.train(model=model)
        self.assertEqual(trajectory.steps, 5)
        self.assertEqual(trajectory.sample_count, 3)
        self.assertEqual(
            [c["step"] for c in trajectory.manifest["checkpoints"]], [0, 2, 4, 5]
        )
        self.assertEqual(
            [trajectory.batch(t) for t in range(5)], list(map(tuple, self.schedule))
        )
        self.assertEqual([trajectory.learning_rate(t) for t in range(5)], self.rates)
        self.assertTrue(torch.equal(trajectory.examples, self.examples))
        self.assertEqual(trajectory.metadata["unicode"], "λ")
        reopened = Trajectory.open(self.root / "run")
        for t, expected in enumerate(self.oracle()):
            with self.subTest(step=t):
                reopened.load_state(model, t, quadratic_loss)
                self.assert_model(model, expected)

    def test_checkpoints_load_for_inference_without_loss_function(self):
        trajectory = self.train()
        model = Affine()
        trajectory.load_state(model, 5)
        self.assert_model(model, self.oracle()[5])
        before = snapshot(model)
        with self.assertRaisesRegex(ValueError, "requires loss_fn"):
            trajectory.load_state(model, 3)
        self.assertTrue(torch.equal(before["weight"], model.weight))

    def test_reverse_replay_indexes_pre_update_states(self):
        trajectory = self.train()
        model = Affine()
        seen = []
        expected = self.oracle()
        for t, borrowed in trajectory.iter_reverse_segments(
            model, 0, 5, quadratic_loss
        ):
            self.assertIs(borrowed, model)
            self.assert_model(borrowed, expected[t])
            seen.append(t)
        self.assertEqual(seen, [4, 3, 2, 1, 0])

    def test_reverse_partial_interval_and_empty_interval(self):
        trajectory = self.train()
        model = Affine()
        seen = []
        for t, borrowed in trajectory.iter_reverse_segments(
            model, 1, 4, quadratic_loss
        ):
            self.assert_model(borrowed, self.oracle()[t])
            seen.append(t)
        self.assertEqual(seen, [3, 2, 1])
        self.assertEqual(
            list(trajectory.iter_reverse_segments(model, 3, 3, quadratic_loss)), []
        )

    def test_replay_starts_from_nearest_checkpoint(self):
        trajectory = self.train()
        callback = mock.Mock(side_effect=quadratic_loss)
        trajectory.load_state(Affine(), 3, callback)
        self.assertEqual(callback.call_count, 1)
        self.assertTrue(torch.equal(callback.call_args.args[1], self.examples[[0, 1]]))

    def test_reverse_replay_uses_bounded_checkpoint_segments(self):
        trajectory = self.train()
        callback = mock.Mock(side_effect=quadratic_loss)
        list(trajectory.iter_reverse_segments(Affine(), 0, 5, callback))
        # [0,2), [2,4), [4,5) need respectively one, one, zero forward updates.
        # Reconstructing every state independently would redo more work.
        self.assertEqual(callback.call_count, 2)

    def test_deletion_preserves_original_denominator(self):
        model = Affine()
        sgd_step(
            model,
            self.examples,
            [0, 2],
            self.rates[0],
            quadratic_loss,
            omitted_sample=0,
        )
        self.assert_model(model, self.oracle(omitted=0)[1])
        residual = 0.4 * 3 - 0.2 - (-1)
        self.assertAlmostEqual(
            float(model.weight.detach()), 0.4 - self.rates[0] * residual * 3 / 2
        )

    def test_all_deleted_batch_is_noop(self):
        model = Affine()
        model.weight.grad = torch.ones_like(model.weight)
        result = sgd_step(
            model, self.examples, [0], 0.1, quadratic_loss, omitted_sample=0
        )
        self.assertEqual(result, 0.0)
        self.assert_model(model, (0.4, -0.2))
        self.assertIsNone(model.weight.grad)

    def test_leave_one_out_replays_from_initial_for_nonzero_window(self):
        trajectory = self.train()
        model = Affine()
        trajectory.leave_one_out_state(model, 0, end=2, loss_fn=quadratic_loss)
        theta_deleted_start = float(model.weight.detach())
        self.assert_model(model, self.oracle(omitted=0)[2])
        trajectory.leave_one_out_state(model, 0, end=5, loss_fn=quadratic_loss)
        self.assert_model(model, self.oracle(omitted=0)[5])
        deleted_change = float(model.weight.detach()) - theta_deleted_start
        original_change = self.oracle()[5][0] - self.oracle()[2][0]
        self.assertNotAlmostEqual(deleted_change, original_change)
        # A late-window restart from original theta_2 is a different experiment.
        trajectory.load_state(model, 2)
        for t in range(2, 5):
            sgd_step(
                model,
                self.examples,
                trajectory.batch(t),
                trajectory.learning_rate(t),
                quadratic_loss,
                omitted_sample=0,
            )
        self.assertNotAlmostEqual(
            float(model.weight.detach()), self.oracle(omitted=0)[5][0]
        )

    def test_random_schedules_and_replay_are_bitwise_repeatable(self):
        first_model, second_model = Affine(), Affine()
        first = self.train("first", model=first_model, schedule=None)
        second = self.train("second", model=second_model, schedule=None)
        self.assertEqual(first.manifest["schedule"], second.manifest["schedule"])
        for t in range(6):
            first.load_state(first_model, t, quadratic_loss)
            second.load_state(second_model, t, quadratic_loss)
            for left, right in zip(first_model.parameters(), second_model.parameters()):
                self.assertTrue(torch.equal(left, right))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_training_and_replay_are_bitwise_identical(self):
        model = Affine().cuda()
        observed = [snapshot(model)]
        trajectory = self.train(
            model=model, progress=lambda step, loss: observed.append(snapshot(model))
        )
        for step, expected in enumerate(observed):
            with self.subTest(step=step):
                trajectory.load_state(model, step, quadratic_loss)
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value.cpu(), expected[name]))
        for step, replayed in trajectory.iter_reverse_segments(
            model, 0, 5, quadratic_loss
        ):
            for name, value in replayed.state_dict().items():
                self.assertTrue(torch.equal(value.cpu(), observed[step][name]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cpu_saved_checkpoint_can_load_on_cuda_but_cannot_replay(self):
        trajectory = self.train()
        model = Affine().cuda()
        trajectory.load_state(model, 5)
        self.assert_model(model, self.oracle()[5])
        with self.assertRaisesRegex(ValueError, "original device"):
            trajectory.load_state(model, 3, quadratic_loss)

    def test_tied_parameter_update_deduplicates_after_both_contributions(self):
        model = Tied()
        trajectory = self.train(
            model=model, steps=1, schedule=[[0, 2]], learning_rate=0.03
        )
        # f(x)=2*w*x; d(half residual^2)/dw = residual*2*x, counted once.
        expected_gradient = sum(
            (0.8 * x - y) * 2 * x / 2 for x, y in self.examples[[0, 2]].tolist()
        )
        self.assertAlmostEqual(
            float(model.weight.detach()), 0.4 - 0.03 * expected_gradient
        )
        trajectory.load_state(model, 1)
        self.assertIs(model.weight, model.other)
        self.assertEqual(
            trajectory.manifest["parameter_layout"][1]["alias_of"], "weight"
        )
        with self.assertRaisesRegex(ValueError, "tying"):
            trajectory.load_state(Tied(tied=False), 1)

    def test_parameter_dtype_mismatch_rejected_without_mutation(self):
        trajectory = self.train()
        model = Affine(dtype=torch.float32)
        before = snapshot(model)
        with self.assertRaisesRegex(ValueError, "dtype"):
            trajectory.load_state(model, 5)
        self.assertTrue(torch.equal(before["weight"], model.weight))

    def test_no_overwrite_even_for_empty_directory(self):
        (self.root / "run").mkdir()
        model = Affine()
        with self.assertRaises(FileExistsError):
            self.train(model=model)
        self.assert_model(model, (0.4, -0.2))
        self.assertEqual(list((self.root / "run").iterdir()), [])

    def test_interrupted_training_is_not_a_completed_trajectory(self):
        def fail_progress(step, loss):
            raise RuntimeError("interrupted")

        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.train(progress=fail_progress)
        self.assertTrue((self.root / "run" / "step_00000000.pt").exists())
        self.assertFalse((self.root / "run" / "manifest.json").exists())
        with self.assertRaisesRegex(ValueError, "completed"):
            Trajectory.open(self.root / "run")

    def test_zero_steps_records_only_initial_checkpoint(self):
        trajectory = self.train(steps=0, schedule=[], learning_rate=0.1)
        self.assertEqual(trajectory.steps, 0)
        self.assertEqual(len(trajectory.manifest["checkpoints"]), 1)
        model = Affine()
        trajectory.load_state(model, 0)
        self.assert_model(model, (0.4, -0.2))

    def test_corrupt_examples_and_checkpoint_detected(self):
        first = self.train("first")
        (first.directory / "examples.pt").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            Trajectory.open(first.directory)
        second = self.train("second")
        checkpoint = second.directory / second.manifest["checkpoints"][-1]["file"]
        checkpoint.write_bytes(b"corrupt")
        model = Affine()
        with self.assertRaisesRegex(ValueError, "checksum"):
            second.load_state(model, 5)
        self.assert_model(model, (0.4, -0.2))

    def test_symlinked_trajectory_file_rejected(self):
        trajectory = self.train()
        examples = trajectory.directory / "examples.pt"
        target = trajectory.directory / "actual_examples.pt"
        examples.rename(target)
        examples.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlinked"):
            Trajectory.open(trajectory.directory)

    def test_checksummed_wrong_checkpoint_shape_rejected(self):
        self.train()

        def corrupt(manifest):
            state = snapshot(Affine())
            state["weight"] = torch.zeros(2, dtype=torch.float64)
            self.rehash(manifest, manifest["checkpoints"][0], state)

        self.edit_manifest(corrupt)
        trajectory = Trajectory.open(self.root / "run")
        with self.assertRaisesRegex(ValueError, "layout"):
            trajectory.load_state(Affine(), 0)

    def test_checksummed_nonfinite_checkpoint_rejected(self):
        self.train()

        def corrupt(manifest):
            state = snapshot(Affine())
            state["weight"].fill_(float("nan"))
            self.rehash(manifest, manifest["checkpoints"][0], state)

        self.edit_manifest(corrupt)
        trajectory = Trajectory.open(self.root / "run")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            trajectory.load_state(Affine(), 0)

    def test_checksummed_inconsistent_tied_weights_rejected(self):
        self.train(model=Tied())

        def corrupt(manifest):
            state = snapshot(Tied())
            state["other"].add_(1)
            self.rehash(manifest, manifest["checkpoints"][0], state)

        self.edit_manifest(corrupt)
        trajectory = Trajectory.open(self.root / "run")
        with self.assertRaisesRegex(ValueError, "tied weights"):
            trajectory.load_state(Tied(), 0)

    def test_invalid_manifests_are_rejected(self):
        self.train()
        path = self.root / "run" / "manifest.json"
        valid = path.read_text()
        changes = [
            lambda m: m.update(version=True),
            lambda m: m.update(complete=False),
            lambda m: m.update(seed=-1),
            lambda m: m.update(device_type="guessed"),
            lambda m: m.update(metadata={"invalid": float("nan")}),
            lambda m: m["schedule"].pop(),
            lambda m: m["schedule"][0].update(batch=[0, 0]),
            lambda m: m["schedule"][0].update(batch=[0, 3]),
            lambda m: m["schedule"][0].update(batch=[True, 1]),
            lambda m: m["schedule"][0].update(learning_rate=float("nan")),
            lambda m: m["schedule"][0].update(mean_loss=float("inf")),
            lambda m: m["checkpoints"].insert(1, None),
            lambda m: m["checkpoints"][1].update(step=True),
            lambda m: m["checkpoints"][0].update(file="../escape.pt"),
            lambda m: m["checkpoints"][0].update(sha256="invalid"),
            lambda m: m["parameter_layout"][0].update(shape=[-1]),
            lambda m: m["parameter_layout"][0].update(alias_of="not_a_parameter"),
            lambda m: m["parameter_layout"][0].update(dtype="torch.bad_type"),
            lambda m: m["state_layout"].append(m["state_layout"][0]),
            lambda m: m["examples"].update(shape=[4, 2]),
        ]
        for change in changes:
            with self.subTest(change=change):
                manifest = json.loads(valid)
                change(manifest)
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    Trajectory.open(self.root / "run")

    def test_invalid_inputs_fail_before_creating_directory(self):
        cases = [
            dict(batch_size=4),
            dict(steps=-1),
            dict(checkpoint_interval=0),
            dict(seed=-2),
            dict(metadata=[]),
            dict(learning_rate=0.0),
            dict(learning_rate=[0.1]),
            dict(schedule=[[0, 0]] * 5),
        ]
        for options in cases:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.train(**options)
                self.assertFalse((self.root / "run").exists())

    def test_nonfinite_initial_state_rejected_before_directory_creation(self):
        model = Affine()
        with torch.no_grad():
            model.weight.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "initial"):
            self.train(model=model)
        self.assertFalse((self.root / "run").exists())

    def test_nonfinite_loss_gradient_and_update_do_not_mutate_weights(self):
        for kind in ("loss", "gradient", "update", "shape"):
            with self.subTest(kind=kind):
                model = Affine()
                if kind == "loss":
                    fn = lambda m, b: quadratic_loss(m, b) * float("nan")
                elif kind == "gradient":
                    # sqrt has a finite value but an infinite derivative at 0.
                    fn = lambda m, b: (m.weight - 0.4).sqrt().expand(len(b))
                elif kind == "shape":
                    fn = lambda m, b: quadratic_loss(m, b).mean()
                else:
                    fn = lambda m, b: (m.weight + m.bias).expand(len(b)) * 1e300
                with self.assertRaises(ValueError):
                    sgd_step(
                        model,
                        self.examples,
                        [0, 1],
                        1e100 if kind == "update" else 0.1,
                        fn,
                    )
                self.assert_model(model, (0.4, -0.2))

    def test_dropout_rejected(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Dropout(0.1))
        with self.assertRaisesRegex(ValueError, "dropout"):
            self.train(model=model)
        self.assertFalse((self.root / "run").exists())

    def test_batchnorm_rejected_even_in_eval_mode_or_without_running_stats(self):
        for running_stats in (False, True):
            for training in (False, True):
                with self.subTest(running_stats=running_stats, training=training):
                    model = torch.nn.Sequential(
                        torch.nn.Linear(2, 2),
                        torch.nn.BatchNorm1d(2, track_running_stats=running_stats),
                    )
                    model.train(training)
                    with self.assertRaisesRegex(ValueError, "sample-separable"):
                        self.train(model=model)
                    self.assertFalse((self.root / "run").exists())

    def test_out_of_range_steps_and_deletions_are_rejected(self):
        trajectory = self.train()
        model = Affine()
        for t in (-1, 6, True):
            with self.subTest(step=t), self.assertRaises(ValueError):
                trajectory.load_state(model, t, quadratic_loss)
        with self.assertRaises(ValueError):
            trajectory.batch(5)
        with self.assertRaises(ValueError):
            trajectory.learning_rate(5)
        with self.assertRaises(ValueError):
            list(trajectory.iter_reverse_segments(model, 4, 2, quadratic_loss))
        with self.assertRaises(ValueError):
            trajectory.leave_one_out_state(model, 3, loss_fn=quadratic_loss)


if __name__ == "__main__":
    unittest.main()
