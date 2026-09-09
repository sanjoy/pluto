"""CLI provenance and fixed-selection tests on tiny real checkpoint files."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from . import late_mlp_paths as paths
from .checkpoint import GPT2Checkpoint, GPT2Config, tensor_manifest
from .verify import validate_candidates


class HelperTest(unittest.TestCase):
    def test_canonical_array_hash_contiguous_strided_and_byte_order(self):
        base = np.arange(48, dtype=np.float64).reshape(6, 8)
        strided = base[::-2, ::2]
        contiguous = np.ascontiguousarray(strided)
        expected = paths.array_record(contiguous, "<f8", chunk_bytes=16)
        self.assertEqual(expected["shape"], [3, 4])
        self.assertEqual(expected["dtype"], "<f8")
        self.assertEqual(expected["bytes"], 3 * 4 * 8)
        self.assertEqual(expected["sha256"], hashlib.sha256(contiguous.tobytes()).hexdigest())
        for array in (strided, np.asfortranarray(strided), strided.astype(">f8"),
                      strided.astype(np.float32)):
            self.assertEqual(paths.array_record(array, "<f8", chunk_bytes=24), expected)
        self.assertEqual(paths.array_record(np.array([], dtype="<i8"), "<i8")["sha256"],
                         hashlib.sha256(b"").hexdigest())
        with self.assertRaises(ValueError): paths.array_record(base, "<f8", chunk_bytes=0)
        with self.assertRaises(ValueError): paths.array_record(base, object)

    def test_compiled_array_records_detect_value_and_shape_changes(self):
        probe = SimpleNamespace(token_ids=np.array([1, 5]),
                                **{name: np.arange(4., dtype=np.float64).reshape(2, 2)
                                   for name in ("direct", "features", "readout",
                                                "neuron_readout", "g7", "h7")})
        before = paths.compiled_array_records(probe)
        self.assertEqual(set(before), {"token_ids", "direct", "features", "readout",
                                       "neuron_readout", "g7", "h7"})
        self.assertEqual(paths.verify_compiled_arrays(probe, before), before)
        probe.g7[0, 0] = -0.0
        with self.assertRaisesRegex(ValueError, "coefficients changed"):
            paths.verify_compiled_arrays(probe, before)
        probe.g7[0, 0] = 0.0
        probe.features = probe.features.reshape(4)
        self.assertEqual(paths.array_hash(probe.features, "<f8"), before["features"]["sha256"])
        with self.assertRaisesRegex(ValueError, "coefficients changed"):
            paths.verify_compiled_arrays(probe, before)

    def test_runtime_environment_records_requested_threads_and_python(self):
        with mock.patch.dict(os.environ, {"OPENBLAS_NUM_THREADS": "3", "OMP_NUM_THREADS": "7"}):
            info = paths.runtime_environment()
        self.assertEqual(info["environment"], {"OPENBLAS_NUM_THREADS": "3", "OMP_NUM_THREADS": "7"})
        self.assertTrue(info["python_version"])
        self.assertTrue(info["python_implementation"])
        self.assertTrue(Path(info["python_executable"]).is_absolute())
        self.assertTrue(info["platform"])
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(paths.runtime_environment()["environment"],
                             {"OPENBLAS_NUM_THREADS": None, "OMP_NUM_THREADS": None})

    def test_selection_translation_and_ties(self):
        e = np.array([[1., 0.], [-1., 0.], [0., 2.], [0., -2.]])
        ids, info = paths.select_vocabulary(e, 3)
        np.testing.assert_array_equal(ids, [0, 2, 3])
        np.testing.assert_array_equal(info["ranked_ids"], [2, 3, 0])
        shifted, _ = paths.select_vocabulary(e + [15., -8.], 3)
        np.testing.assert_array_equal(shifted, ids)
        for value in (0, 5, True, 1.5):
            with self.assertRaises(ValueError): paths.select_vocabulary(e, value)
        with self.assertRaises(ValueError): paths.select_vocabulary([[np.nan]], 1)

    def test_initial_strings_exact_rng_contract(self):
        ids = np.array([2, 7, 11])
        actual = paths.initial_strings(ids, 5, 7, 20260909)
        rng = np.random.Generator(np.random.PCG64(20260909))
        expected = ids[rng.integers(0, 3, size=(5, 7), dtype=np.int64)]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(paths.initial_strings([5], 2, 4, 0), np.full((2, 4), 5))
        for bad in ([2, 2], [3, 1], [-1, 4], [1.5, 2.5]):
            with self.assertRaises(ValueError): paths.initial_strings(bad, 2, 4, 0)

    def test_exclusive_finite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            with self.assertRaises(ValueError): paths.write_exclusive(output, {"x": np.inf})
            self.assertFalse(output.exists())
            paths.write_exclusive(output, {"x": np.array([1, 2]), "p": Path("a")})
            self.assertEqual(json.loads(output.read_text()), {"x": [1, 2], "p": "a"})
            with self.assertRaises(FileExistsError): paths.write_exclusive(output, {})

    def test_order_audit_independent_scalar_and_no_cross_null(self):
        rng = np.random.default_rng(9)
        f = rng.normal(size=(4, 3, 2))
        n = rng.normal(size=(3, 2))
        h = np.array([.5, -.2])
        probe = SimpleNamespace(token_ids=np.array([2, 4, 7]), direct=np.zeros((4, 3, 5)),
                                features=f, neuron_readout=n, h7=h, alpha=.125)
        audit = paths.mixed_order_audit(probe)
        self.assertEqual(len(audit["pairs"]), 2)
        for row in audit["pairs"]:
            p, q = row["source_positions"]; r = row["target_position"]
            expected = np.empty((3, 3, 3))
            for a in range(3):
                for b in range(3):
                    for u in range(3):
                        expected[a, b, u] = sum(f[p, a, j] * f[q, b, j] * h[j] * n[u, j]
                                                for j in range(2)) * (.125 / r) ** 2
            np.testing.assert_allclose(row["mixed_coefficients"], expected, rtol=2e-15, atol=1e-18)
        probe.features = np.repeat(f[:1], 4, axis=0)
        for row in paths.mixed_order_audit(probe)["pairs"]:
            self.assertEqual(row["maximum_absolute_content_swap_difference"], 0)
        probe.h7 = np.zeros(2)
        for row in paths.mixed_order_audit(probe)["pairs"]:
            self.assertEqual(row["maximum_absolute_coefficient"], 0)
            self.assertEqual(row["zero_denominator_cases"], 27)


class CliTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "step_7"
        self.checkpoint.mkdir()
        self.config = GPT2Config(vocab_size=12, padded_vocab_size=12, context_length=16,
                                 n_layers=8, d_model=4, n_heads=2, d_ff=8)
        rng = np.random.default_rng(25)
        for spec in tensor_manifest(self.config):
            value = (rng.normal(size=spec.shape) * .08).astype("<f4")
            if spec.name.endswith(".scale"): value.fill(1)
            (self.checkpoint / spec.filename).write_bytes(value.tobytes())
        self.tokenizer = self.root / "tokenizer"
        self.tokenizer.mkdir()
        (self.tokenizer / "tokenizer.json").write_text(json.dumps({
            "model": {"vocab": {f"token_{i}": i for i in range(12)}}}))
        self.output = self.root / "candidates.jsonl"

    def run_extract(self, **kwargs):
        # Actual checkpoint parsing/hashing, compiler, optimizer and files;
        # only the physical configuration is redirected to the tiny fixture.
        def load(path, check_finite=True):
            return GPT2Checkpoint(path, self.config, check_finite=check_finite)
        defaults = dict(size=8, length=4, restarts=3, sweeps=2, label="test")
        defaults.update(kwargs)
        with mock.patch.object(paths, "GPT2Checkpoint", side_effect=load), contextlib.redirect_stdout(io.StringIO()):
            return paths.extract(self.checkpoint, self.checkpoint, self.tokenizer, self.output, **defaults)

    def test_complete_extraction_and_plan_precedes_compilation(self):
        real_compile = paths.compile_polynomial
        def compile_checked(*args, **kwargs):
            plan = self.output.with_suffix(".jsonl.plan.json")
            self.assertTrue(plan.exists())
            self.assertFalse(self.output.exists())
            self.assertEqual(json.loads(plan.read_text())["stage"], "frozen_weight_only_polynomial_plan")
            return real_compile(*args, **kwargs)
        with mock.patch.object(paths, "compile_polynomial", side_effect=compile_checked):
            metadata_path = self.run_extract()
        metadata = json.loads(metadata_path.read_text())
        records = [json.loads(line) for line in self.output.read_text().splitlines()]
        validate_candidates(records, 12)
        self.assertEqual(len(records), 3)
        self.assertTrue(metadata["complete"])
        self.assertTrue(metadata["checkpoint_and_sources_unchanged"])
        self.assertTrue(metadata["compiled_arrays_unchanged"])
        self.assertEqual(metadata["compiled_arrays"], metadata["compiled_arrays_after"])
        plan = json.loads(self.output.with_suffix(".jsonl.plan.json").read_text())
        self.assertEqual(metadata["runtime_environment"], plan["runtime_environment"])
        self.assertEqual(metadata["compiled_arrays"]["token_ids"]["dtype"], "<i8")
        self.assertEqual(metadata["compiled_arrays"]["features"]["shape"], [4, 8, 8])
        self.assertEqual(plan["sources"]["production_gelu"]["sha256"],
                         paths.sha256_file(Path(paths.__file__).resolve().parents[2] /
                                           "src/llm/layers/gelu.cc"))
        self.assertEqual(metadata["candidate_structure"]["overall"]["candidate_records"], 3)
        self.assertEqual(metadata["candidates"], paths.file_record(self.output))
        self.assertEqual(len(metadata["optimization_trace"]), 8)
        for record in records:
            self.assertEqual(len(record["token_ids"]), 4)
            self.assertGreaterEqual(record["score"] + 1e-12, record["initial_score"])
        with self.assertRaises(FileExistsError): self.run_extract()

    def test_rejects_dangling_symlink_and_checkpoint_output(self):
        self.output.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError): self.run_extract()
        self.output = self.checkpoint / "must_not_create.jsonl"
        with self.assertRaisesRegex(ValueError, "inside a checkpoint"): self.run_extract()
        self.assertFalse(self.output.exists())

    def test_weight_change_fails_before_candidates(self):
        real_compile = paths.compile_polynomial
        def compile_changed(*args, **kwargs):
            probe = real_compile(*args, **kwargs)
            real_optimize = probe.optimize
            def changed(*a, **k):
                result = real_optimize(*a, **k)
                path = self.checkpoint / "weight_0.bin"
                value = np.frombuffer(path.read_bytes(), dtype="<f4").copy()
                value[0] += 1
                path.write_bytes(value.tobytes())
                return result
            return SimpleNamespace(**{name: getattr(probe, name) for name in
                                      ("token_ids", "direct", "features", "readout", "neuron_readout", "g7", "h7", "alpha", "metadata", "score_components")}, optimize=changed)
        with mock.patch.object(paths, "compile_polynomial", side_effect=compile_changed):
            with self.assertRaisesRegex(ValueError, "checkpoint or production source changed"):
                self.run_extract()
        self.assertTrue(self.output.with_suffix(".jsonl.plan.json").exists())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_suffix(".jsonl.metadata.json").exists())

    def test_plan_change_fails_before_candidates(self):
        real_compile = paths.compile_polynomial
        def compile_changed(*args, **kwargs):
            probe = real_compile(*args, **kwargs)
            self.output.with_suffix(".jsonl.plan.json").write_text('{}\n')
            return probe
        with mock.patch.object(paths, "compile_polynomial", side_effect=compile_changed):
            with self.assertRaisesRegex(ValueError, "frozen plan changed"):
                self.run_extract()
        self.assertFalse(self.output.exists())

    def test_invalid_label_dictionary_leaves_no_plan(self):
        (self.tokenizer / "tokenizer.json").write_text('{"model":{"vocab":{"bad":0}}}')
        with self.assertRaisesRegex(ValueError, "token labels"):
            self.run_extract()
        self.assertFalse(self.output.with_suffix(".jsonl.plan.json").exists())

    def test_compiled_coefficient_change_fails_before_candidates(self):
        real_compile = paths.compile_polynomial
        def compile_changed(*args, **kwargs):
            probe = real_compile(*args, **kwargs)
            def changed(*a, **k):
                result = probe.optimize(*a, **k)
                # Simulate an accidental mutation despite the read-only view.
                # Only this tiny test-owned coefficient is made writable.
                probe.g7.setflags(write=True)
                probe.g7[0] += 1
                probe.g7.setflags(write=False)
                return result
            return SimpleNamespace(**{name: getattr(probe, name) for name in
                                      ("token_ids", "direct", "features", "readout", "neuron_readout", "g7", "h7", "alpha", "metadata", "score_components")}, optimize=changed)
        with mock.patch.object(paths, "compile_polynomial", side_effect=compile_changed):
            with self.assertRaisesRegex(ValueError, "coefficients changed"):
                self.run_extract()
        self.assertTrue(self.output.with_suffix(".jsonl.plan.json").exists())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_suffix(".jsonl.metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
