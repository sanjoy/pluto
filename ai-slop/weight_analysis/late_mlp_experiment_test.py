"""Synthetic integrity tests: no real checkpoint, corpus, or candidate inputs."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import late_mlp_experiment as experiment
from .checkpoint import GPT2Config, SOURCE_FILES, sha256_file, tensor_manifest


class ExperimentTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.spec = experiment.ExperimentSpec(size=4, length=4, restarts=2, sweeps=2)
        tokenizer = self.directory / "tokenizer.json"
        tokenizer.write_text("{}\n")
        sources = {name: experiment.file_record(Path(experiment.__file__).with_name(name))
                   for name in experiment.EXTRACTOR_SOURCES}
        sources.update(tokenizer=experiment.file_record(tokenizer),
                       production_gelu=experiment.file_record(experiment.ROOT / "src/llm/layers/gelu.cc"),
                       protocol=experiment.file_record(experiment.PROTOCOL))
        config = GPT2Config()
        tensors = tensor_manifest(config)
        checkpoint = {
            "config": asdict(config), "head_dim": 64,
            "checkpoint_directory": str(self.directory / "step_13030"),
            "format": "pluto-gpt2-raw-unique-fp32-weights", "storage_dtype": "<f4",
            "unique_weight_files": 100, "tensors": [item.to_dict() for item in tensors],
            "checkpoint_bytes": sum(item.nbytes for item in tensors),
            "parameter_count": sum(item.nbytes // 4 for item in tensors),
            "weight_sha256": {f"weight_{index}.bin": "0" * 64 for index in range(100)},
            "source_sha256": {name: sha256_file(experiment.ROOT / name) for name in SOURCE_FILES}}
        selected = np.arange(self.spec.size)
        rng = np.random.Generator(np.random.PCG64(self.spec.seed))
        initial = selected[rng.integers(0, self.spec.size,
                           size=(self.spec.restarts, self.spec.length), dtype=np.int64)]
        for arm, mode in zip(experiment.ARMS, experiment.MODES):
            candidate, plan_path, metadata_path = self.paths(arm)
            current_checkpoint = deepcopy(checkpoint)
            if arm == "early_full":
                current_checkpoint["checkpoint_directory"] = str(self.directory / "step_10")
                current_checkpoint["weight_sha256"]["weight_0.bin"] = "1" * 64
            parameters = {"mode": mode, "vocabulary_size": self.spec.size,
                          "length": self.spec.length, "restarts": self.spec.restarts,
                          "sweeps": self.spec.sweeps, "seed": self.spec.seed,
                          "alpha": self.spec.alpha, "label": arm, "blocks": [6, 7]}
            plan = {"schema_version": 1, "stage": "frozen_weight_only_polynomial_plan",
                    "parameters": parameters, "checkpoint": current_checkpoint,
                    "selection_checkpoint": checkpoint, "sources": sources,
                    "output": str(candidate), "runtime_environment": {"synthetic": True},
                    "selection": {"method": "final:centroid_distance",
                                  "ranked_ids": selected[::-1].tolist(),
                                  "ranked_scores": [4, 3, 2, 1], "sorted_ids": selected.tolist(),
                                  "full_logical_score_sha256": "a" * 64,
                                  "full_logical_centroid_sha256": "b" * 64,
                                  "rule": "descending norm(E[t]-mean_ALL_LOGICAL(E)); ascending ID ties"},
                    "initial_token_ids": initial.tolist(),
                    "initial_ids_sha256": experiment.digest(initial.astype("<i4").tobytes())}
            self.write_json(plan_path, plan)
            # Every toy restart ends at [0,0,0,0]. Duplicates intentionally remain.
            rows, trace = initial.copy(), []
            for index in range(self.spec.sweeps * self.spec.length):
                sweep, offset = divmod(index, self.spec.length)
                position = offset if sweep % 2 == 0 else self.spec.length - offset - 1
                changed = np.flatnonzero(rows[:, position] != 0).tolist()
                rows[:, position] = 0
                trace.append({"sweep": sweep, "position": position,
                              "direction": "forward" if sweep % 2 == 0 else "reverse",
                              "scores_before": [float(index)] * self.spec.restarts,
                              "scores_after": [float(index + 1)] * self.spec.restarts,
                              "selected_token_ids": [0] * self.spec.restarts,
                              "changed_rows": changed, "roundoff_retained_rows": [],
                              "max_score_discrepancy": 0.0})
            final_score = float(len(trace))
            method = "late_mlp_" + arm
            records = [{"candidate_id": f"{method}:restart{index}", "method": method,
                        "token_ids": rows[index].tolist(), "vocabulary_labels": ["a"] * self.spec.length,
                        "initial_token_ids": initial[index].tolist(), "initial_score": 0.0,
                        "score": final_score, "linear_score": final_score, "quadratic_score": 0.0,
                        "provenance": {"plan_file": str(plan_path), "restart": index}}
                       for index in range(self.spec.restarts)]
            candidate.write_text("".join(json.dumps(row) + "\n" for row in records))
            arrays = {name: {"sha256": "0" * 64} for name in
                      ("token_ids", "direct", "features", "readout", "neuron_readout", "g7", "h7")}
            metadata = {"schema_version": 1, "stage": "completed_weight_only_polynomial_extraction",
                        "complete": True, "parameters": parameters,
                        "plan": experiment.file_record(plan_path),
                        "candidates": experiment.file_record(candidate),
                        "checkpoint_and_sources_unchanged": True,
                        "compiled_arrays_unchanged": True, "compiled_arrays": arrays,
                        "compiled_arrays_after": arrays, "runtime_environment": {"synthetic": True},
                        "compiler": {key: parameters[key] for key in ("mode", "blocks", "length", "alpha")},
                        "candidate_count": self.spec.restarts, "distinct_candidates": 1,
                        "optimization_trace": trace,
                        "final_components": {"total": [final_score] * self.spec.restarts,
                                             "linear": [final_score] * self.spec.restarts,
                                             "quadratic": [0.0] * self.spec.restarts}}
            self.write_json(metadata_path, metadata)

    def paths(self, arm="final_full"):
        candidate = self.directory / f"{arm}.jsonl"
        return candidate, Path(str(candidate) + ".plan.json"), Path(str(candidate) + ".metadata.json")

    def write_json(self, path, value):
        path.write_text(json.dumps(value) + "\n")

    def edit_json(self, path, edit):
        value = json.loads(path.read_text())
        edit(value)
        self.write_json(path, value)

    def refresh(self, arm="final_full"):
        candidate, plan, metadata = self.paths(arm)
        self.edit_json(metadata, lambda value: value.update(
            plan=experiment.file_record(plan), candidates=experiment.file_record(candidate)))

    def freeze(self):
        return experiment.freeze(self.directory, self.spec)

    def test_freeze_preserves_all_bytes_and_never_reads_corpus(self):
        expected = b"".join(self.paths(arm)[0].read_bytes() for arm in experiment.ARMS)
        with mock.patch("numpy.memmap", side_effect=AssertionError("corpus accessed")):
            frozen_path = self.freeze()
            record, candidates = experiment.checked_freeze(self.directory, self.spec)
        self.assertEqual(frozen_path, Path(record["path"]))
        self.assertEqual((self.directory / "combined.jsonl").read_bytes(), expected)
        self.assertEqual(len(candidates), 10)
        self.assertEqual(json.loads(frozen_path.read_text())["arm_order"], list(experiment.ARMS))
        self.assertFalse(json.loads(frozen_path.read_text())["corpus_read_by_this_command"])

    def test_no_overwrite_or_partial_combined_when_manifest_exists(self):
        (self.directory / "frozen.json").write_text("keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.freeze()
        self.assertFalse((self.directory / "combined.jsonl").exists())
        self.assertEqual((self.directory / "frozen.json").read_text(), "keep")

    def test_candidate_hash_tampering(self):
        candidate = self.paths()[0]
        candidate.write_bytes(candidate.read_bytes().replace(b'"score": 8.0', b'"score": 9.0'))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.freeze()

    def test_wrong_parameters_even_with_updated_plan_hash(self):
        self.edit_json(self.paths()[1], lambda plan: plan["parameters"].update(alpha=0.25))
        self.refresh()
        with self.assertRaisesRegex(ValueError, "nonprotocol parameters"):
            self.freeze()

    def test_seed_initials_tampering(self):
        self.edit_json(self.paths()[1], lambda plan: plan["initial_token_ids"][0].__setitem__(0, 3))
        self.refresh()
        with self.assertRaisesRegex(ValueError, "initial strings"):
            self.freeze()

    def test_selection_and_checkpoint_identity(self):
        self.edit_json(self.paths("final_affine")[1],
                       lambda plan: plan["selection_checkpoint"]["weight_sha256"].update({"weight_0.bin": "f" * 64}))
        self.refresh("final_affine")
        with self.assertRaisesRegex(ValueError, "do not share"):
            self.freeze()

    def test_wrong_checkpoint_step(self):
        self.edit_json(self.paths("early_full")[1],
                       lambda plan: plan["checkpoint"].update(checkpoint_directory="/tmp/step_20"))
        self.refresh("early_full")
        with self.assertRaisesRegex(ValueError, "wrong checkpoint step"):
            self.freeze()

    def test_incomplete_trace(self):
        self.edit_json(self.paths()[2], lambda metadata: metadata["optimization_trace"].pop())
        with self.assertRaisesRegex(ValueError, "incomplete optimization trace"):
            self.freeze()

    def test_trace_choice_replay(self):
        self.edit_json(self.paths()[2],
                       lambda metadata: metadata["optimization_trace"][-1]["selected_token_ids"].__setitem__(0, 1))
        with self.assertRaisesRegex(ValueError, "changed rows"):
            self.freeze()

    def test_changed_source_before_freeze(self):
        (self.directory / "tokenizer.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "source/protocol/tokenizer changed"):
            self.freeze()

    def test_postfreeze_tampering_stops_before_corpus_access(self):
        self.freeze()
        combined = self.directory / "combined.jsonl"
        combined.write_bytes(combined.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "frozen inputs"):
            experiment.summarize(self.directory, self.directory / "nonexistent_corpus", 4, self.spec)
        self.assertFalse((self.directory / "summary.json").exists())

    def test_manifest_tampering(self):
        path = self.freeze()
        self.edit_json(path, lambda frozen: frozen.update(candidate_count=9))
        with self.assertRaisesRegex(ValueError, "frozen inputs"):
            experiment.checked_freeze(self.directory, self.spec)

    def test_tiny_summary_keeps_duplicate_denominators_and_split_scope(self):
        self.freeze()
        native = self.directory / "toy.u32"
        native.write_bytes(np.array([0, 0, 0, 0, 1, 2, 3, 1], dtype="<u4").tobytes())
        destination = self.directory / "custom_summary.json"
        output = experiment.summarize(self.directory, native, 4, self.spec, destination)
        report = json.loads(output.read_text())
        for arm in experiment.ARMS:
            method = "late_mlp_" + arm
            for scope in ("full_corpus", "current_prefix", "current_suffix"):
                self.assertEqual(report["summaries"][scope][method]["candidates"], 2)
                self.assertEqual(report["summaries"][scope][method]["distinct_candidate_sequences"], 1)
            self.assertEqual(report["summaries"]["current_prefix"][method]["whole_candidate_matches"], 2)
            self.assertEqual(report["summaries"]["current_suffix"][method]["whole_candidate_matches"], 0)
        self.assertEqual(report["split_token_index"], 4)
        self.assertIn("historical", report["limitations"][0])
        with self.assertRaisesRegex(ValueError, "already exists"):
            experiment.summarize(self.directory, native, 4, self.spec, destination)

    def test_nonfinite_and_duplicate_json_rejected(self):
        for text in ('{"a": 1, "a": 2}', '{"a": NaN}', '{"a": 1e999}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                experiment.strict_json(text)


if __name__ == "__main__":
    unittest.main()
