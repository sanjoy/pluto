"""Selection math, deterministic masks, and fail-closed metadata tests."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from . import neuron_mapping as nm
from . import causal_validation as cv
from .checkpoint import tensor_manifest


class NeuronMappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups = nm.fixed_groups()
        zero = np.zeros((2048, 512), dtype=np.float32)
        cls.records = nm.group_records(cls.groups, [zero, zero])

    def test_fixed_draws_and_partition(self):
        rng = np.random.Generator(np.random.PCG64(20260909))
        for index in range(2):
            np.testing.assert_array_equal(self.groups[index].ravel(), rng.permutation(2048))
            np.testing.assert_array_equal(np.sort(self.groups[index].ravel()), np.arange(2048))
        self.assertEqual(self.groups.dtype.str, "<i4")
        self.assertEqual(self.groups.nbytes, 16384)
        self.assertFalse(np.array_equal(self.groups[0], self.groups[1]))

    def test_group_addresses_and_norm_ties(self):
        for index, group in enumerate(self.records):
            self.assertEqual(group["weight_index"], 84 if index < 32 else 96)
            self.assertEqual(group["changed_weight_count"], 32768)
            for row, interval in zip(group["rows"], group["byte_intervals"]):
                self.assertEqual(interval, [row * 2048, (row + 1) * 2048])
            base = index // 32 * 32
            self.assertEqual(group["norm_comparators"], [i for i in range(base, base + 32) if i != index][:4])

    def test_norms_use_physical_selected_rows(self):
        values = np.broadcast_to(np.arange(2048)[:, None], (2048, 512))
        records = nm.group_records(self.groups, [values, values])
        expected = np.sqrt(sum(int(row)**2 * 512 for row in self.groups[0, 0]))
        self.assertAlmostEqual(records[0]["frobenius_norm_fp64"], expected)
        norm = records[0]["frobenius_norm_fp64"]
        order = sorted(range(1, 32), key=lambda i: (abs(records[i]["frobenius_norm_fp64"] - norm), i))
        self.assertEqual(records[0]["norm_comparators"], order[:4])

    def test_group_rejects_duplicates_wrong_dtype_and_shape(self):
        values = np.zeros((2048, 512))
        broken = self.groups.copy()
        broken[0, 0, 0] = broken[0, 0, 1]
        for groups in (broken, self.groups.astype(float), self.groups[:1]):
            with self.assertRaises(ValueError):
                nm.group_records(groups, [values, values])
        with self.assertRaises(ValueError):
            nm.group_records(self.groups, [values.T, values])

    def test_arm_order_bias_distinction_and_complete_count(self):
        arms = nm.arm_specs(self.records)
        self.assertEqual(len(arms), 69)
        self.assertEqual([arm["name"] for arm in arms[:4]],
                         ["clean_before", "clean_repeat", "block6_mlp_half", "block6_group00_half"])
        self.assertEqual(arms[35]["name"], "block7_mlp_half")
        self.assertEqual(arms[-1]["name"], "clean_after")
        self.assertEqual(arms[2]["group_weight_indices"], [84, 85])
        self.assertEqual(arms[3]["group_weight_indices"], [84])
        self.assertEqual(arms[3]["selected_rows"], self.groups[0, 0].tolist())

    def test_center_and_contrast_explicit_loops(self):
        values = np.arange(35).reshape(7, 5)**2
        centered = nm.double_center(values)
        contrasts = nm.ordinary_contrasts(values)
        for group in range(7):
            for passage in range(5):
                expected = values[group, passage] - sum(values[group]) / 5 - sum(values[:, passage]) / 7 + values.sum() / 35
                self.assertAlmostEqual(centered[group, passage], expected)
                other = [values[group, j] for j in range(5) if j != passage]
                self.assertAlmostEqual(contrasts[group, passage], values[group, passage] - sum(other) / 4)

    def test_rank_one_plus_additive_abstains(self):
        a = np.arange(1, 65, dtype=float)
        b = np.linspace(0.5, 2, 16)
        delta = a[:, None] * b + 5 * a[:, None] + b**2
        self.assertGreater(nm.ordinary_contrasts(delta).max(), 0)
        result = nm.discover(delta)
        self.assertEqual(result["selected_group_indices"], [None] * 16)
        self.assertAlmostEqual(result["removed_discovery_energy_fraction"], 1)

    def test_large_additive_cancellation_abstains(self):
        rng = np.random.default_rng(17)
        values = rng.normal(size=(64, 1)) * 1e14 + rng.normal(size=(1, 16)) * 1e14
        result = nm.discover(values)
        self.assertEqual(result["status"], "zero_centered_effect")
        self.assertEqual(result["selected_group_indices"], [None] * 16)

    def test_zero_and_ambiguous_modes_abstain(self):
        result = nm.discover(np.zeros((64, 16)))
        self.assertEqual(result["selected_group_indices"], [None] * 16)
        self.assertIsNone(result["removed_discovery_energy_fraction"])
        # The centered identity has a repeated top singular value.
        tied = np.eye(4) + 10
        result = nm.discover(tied)
        self.assertEqual(result["status"], "ambiguous_leading_direction")
        self.assertEqual(result["selected_group_indices"], [None] * 4)

    def test_confirmation_does_not_reselect_or_refit(self):
        rng = np.random.default_rng(73)
        discovery = 1 + rng.normal(size=(64, 16)) * 0.05
        first = nm.select_and_confirm(discovery, discovery, self.records)
        second = nm.select_and_confirm(discovery, -discovery, self.records)
        self.assertEqual(first["discovery"], second["discovery"])
        np.testing.assert_allclose(first["confirmation_projected_residual"],
                                   -np.asarray(second["confirmation_projected_residual"]), atol=1e-14)
        self.assertEqual(second["summary"]["confirmation_positive_raw_and_residual"], 0)
        self.assertGreater(first["summary"]["confirmation_positive_raw_and_residual"], 0)

    def test_harm_eligibility_and_projection_orthogonality(self):
        rng = np.random.default_rng(91)
        values = -10 + rng.normal(size=(64, 16))
        result = nm.discover(values)
        self.assertEqual(result["selected_group_indices"], [None] * 16)
        direction = np.asarray(result["frozen_direction"])
        residual = nm.confirm(values + 11, result)
        np.testing.assert_allclose(direction @ residual, 0, atol=1e-13)
        np.testing.assert_allclose(residual.mean(axis=1), 0, atol=1e-14)
        np.testing.assert_allclose(residual.mean(axis=0), 0, atol=1e-14)

    def test_rank_exact_ties_and_invalids(self):
        self.assertEqual(nm.score_rank([3, 2, 2, 1], 1),
                         {"rank": 2, "exact_tie_count_including_self": 2, "candidate_count": 4})
        for values, index in (([1, np.nan], 0), ([], 0), ([1], -1), ([1], 1), ([1], True)):
            with self.assertRaises(ValueError):
                nm.score_rank(values, index)
        for values in ([], [[1]], [[1, np.inf], [2, 3]], np.ones((2, 2, 2))):
            with self.assertRaises(ValueError):
                nm.discover(values)

    def metadata_fixture(self):
        plan = {"arms": nm.arm_specs(self.records), "group_file": {"path": "/tmp/groups"}}
        base = {"inputs": {"binary": {"path": "/tmp/binary"}},
                "checkpoint": {"checkpoint_directory": "/tmp/checkpoint"}}
        run = {"schema_version": 1, "complete": True, "context_length": 1024,
               "passage_count": 32, "vocab_size": 50257, "padded_vocab_size": 50272,
               "batch_sequences": 4, "checkpoint_raw_weight_count": 101,
               "checkpoint_unique_weight_count": 100,
               "checkpoint_weight_bytes": [spec.nbytes for spec in tensor_manifest()],
               "batch_tokens_bytes": 2 * 32 * 1024 * 4, "neuron_groups_bytes": 16384,
               "neuron_groups_shape": [2, 32, 64], "neuron_group_blocks": [6, 7],
               "neuron_groups_permutations_verified": True,
               "non_default_executor": True, "optimizer_steps": 0, "backward_calls": 0,
               "binary_file": "/tmp/binary", "checkpoint_directory": "/tmp/checkpoint",
               "batch_tokens_file": "/tmp/batch", "neuron_groups_file": "/tmp/groups",
               "arms": copy.deepcopy(plan["arms"])}
        for arm in run["arms"]:
            arm.update({"finite_losses": True, "argmax_valid": True,
                        "restoration_verified_bytes": True,
                        "loss_file": arm["name"] + ".losses.f32",
                        "argmax_file": arm["name"] + ".argmax.i32"})
        return run, plan, base

    def test_metadata_success_and_top_level_failures(self):
        run, plan, base = self.metadata_fixture()
        self.assertEqual(len(nm.validate_run(run, plan, base, Path('/tmp/batch'))), 69)
        for key, wrong in (("complete", False), ("batch_sequences", 1), ("optimizer_steps", 1),
                           ("backward_calls", False), ("neuron_groups_file", "/tmp/other"),
                           ("neuron_groups_permutations_verified", False)):
            broken = copy.deepcopy(run)
            broken[key] = wrong
            with self.assertRaises(ValueError):
                nm.validate_run(broken, plan, base, Path('/tmp/batch'))

    def test_metadata_rejects_changed_rows_bias_and_restoration(self):
        run, plan, base = self.metadata_fixture()
        for key, wrong in (("selected_rows", [1]), ("group_weight_indices", [84, 85]),
                           ("scale", 0), ("kind", "clean"),
                           ("restoration_verified_bytes", False), ("loss_file", "../other")):
            broken = copy.deepcopy(run)
            broken["arms"][3][key] = wrong
            with self.assertRaises(ValueError):
                nm.validate_run(broken, plan, base, Path('/tmp/batch'))
        run["arms"].reverse()
        with self.assertRaises(ValueError):
            nm.validate_run(run, plan, base, Path('/tmp/batch'))

    def test_authenticate_rejects_missing_sources_before_checkpoint_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'manifest.json'
            plan = {"schema_version": 1, "stage": "fine_neuron_reliance_plan_not_extraction",
                    "seed": nm.SEED, "group_shape": [2, 32, 64], "group_dtype": "<i4",
                    "discovery_indices": list(nm.DISCOVERY), "confirmation_indices": list(nm.CONFIRMATION),
                    "execution_batch_sequences": 4, "additional_sources": {}}
            cv.write_report(path, plan)
            with self.assertRaisesRegex(ValueError, 'source identities'):
                nm.authenticate(path)

    def test_planner_rejects_checkpoint_output_and_dangling_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'checkpoint'
            checkpoint.mkdir()
            forbidden = checkpoint / 'analysis'
            with self.assertRaisesRegex(ValueError, 'inside checkpoint'):
                nm.plan_files(SimpleNamespace(output_dir=forbidden, checkpoint=checkpoint))
            self.assertFalse(forbidden.exists())
            link = Path(temporary) / 'link'
            target = Path(temporary) / 'absent'
            link.symlink_to(target)
            with self.assertRaises(FileExistsError):
                nm.plan_files(SimpleNamespace(output_dir=link, checkpoint=checkpoint))
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
