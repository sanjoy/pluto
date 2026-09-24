#!/usr/bin/env python3
"""CPU-only checks of dataset-to-weights experiment identity and safety."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import itertools
import json
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run_dataset_weights as experiment
from verify_predictions import TSV_HEADER


class DatasetWeightsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def checkpoint(self, path, value=0):
        """Write the exact logical FP32 tensor shapes of the fixed model."""
        sizes = [4475 * 10, 27 * 10]
        sizes += [10, 10, 10 * 30, 30, 10 * 10, 10,
                  10, 10, 10 * 20, 20, 20 * 10, 10] * 4
        sizes += [10, 10]
        self.assertEqual(len(sizes), 52)
        self.assertEqual(sum(sizes), 48680)
        path.mkdir(parents=True)
        for index, size in enumerate(sizes):
            (path / f"weight_{index}.bin").write_bytes(struct.pack("<f", value) * size)

    def predictions(self, path, rows, ids, *, wrong=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as output:
            output.write("\t".join(TSV_HEADER) + "\n")
            for line, row in enumerate(rows, 1):
                targets = [ids[token] for token in row[5:]] + [50256]
                for position, target in enumerate(targets, 5):
                    predicted = 1 if wrong and line == 1 and position == 5 else target
                    output.write(f"{line}\t{position}\t{target}\t{predicted}\t0.01\n")

    def test_identity_permutation_and_eos(self):
        self.assertEqual(experiment.permutation(7, 3, 0, 1337), list(range(7)))

    def test_permutation_is_bijective_with_exact_support(self):
        for vocabulary, eos in [(7, 0), (7, 3), (4475, 4474)]:
            for support in [2, vocabulary - 1]:
                for seed in range(10):
                    with self.subTest(vocabulary=vocabulary, eos=eos,
                                      support=support, seed=seed):
                        mapping = experiment.permutation(vocabulary, eos, support, seed)
                        self.assertEqual(sorted(mapping), list(range(vocabulary)))
                        self.assertEqual(mapping[eos], eos)
                        self.assertEqual(sum(old != new for old, new in enumerate(mapping)),
                                         support)

    def test_permutation_is_reproducible_without_changing_global_randomness(self):
        import random
        before = random.getstate()
        first = experiment.permutation(4475, 4474, 512, 321)
        self.assertEqual(first, experiment.permutation(4475, 4474, 512, 321))
        self.assertNotEqual(first, experiment.permutation(4475, 4474, 512, 322))
        self.assertEqual(before, random.getstate())

    def test_invalid_permutation_support_and_eos_rejected(self):
        for vocabulary, eos, support in [(7, -1, 2), (7, 7, 2), (7, 6, -1),
                                          (7, 6, 1), (7, 6, 7), (7, 6, 8)]:
            with self.subTest(vocabulary=vocabulary, eos=eos, support=support):
                with self.assertRaises(ValueError):
                    experiment.permutation(vocabulary, eos, support, 0)

    def test_plan_pins_reproducibility_controls_and_holdouts(self):
        plan = list(itertools.islice(experiment.trial_plan(), 23))
        self.assertEqual([trial["id"] for trial in plan[:2]],
                         ["baseline", "baseline_repeat"])
        self.assertEqual([trial["support"] for trial in plan[:2]], [0, 0])
        self.assertEqual(plan[1]["split"], "control")
        self.assertEqual({trial["support"] for trial in plan[2:9]},
                         {2, 8, 32, 128, 512, 2048, 4474})
        self.assertTrue(all(trial["split"] == "train" for trial in plan[2:9]))
        self.assertTrue(all(trial["split"] == "test" for trial in plan[9:16]))
        self.assertTrue(all(trial["split"] == "train" for trial in plan[16:23]))
        self.assertEqual(len({trial["permutation_seed"] for trial in plan[2:]}), 21)

    def test_identical_embedding_plan_is_baseline_then_exactly_three_permutations(self):
        plan = list(experiment.identical_embedding_trial_plan())
        self.assertEqual([trial["id"] for trial in plan],
                         ["baseline", "rename_000_0002", "rename_000_0512", "rename_000_4474"])
        self.assertEqual([trial["support"] for trial in plan], [0, 2, 512, 4474])
        self.assertEqual([trial["split"] for trial in plan], ["train", "train", "test", "test"])
        self.assertEqual([trial["permutation_seed"] for trial in plan],
                         [0, 810002, 810512, 814474])

    def test_identical_embedding_check_includes_last_eos_row(self):
        checkpoint = self.root / "checkpoint"
        self.checkpoint(checkpoint)
        row = struct.pack("<10f", *[i * 0.01 for i in range(10)])
        embedding = checkpoint / "weight_0.bin"
        embedding.write_bytes(row * 4475)
        experiment.check_identical_embedding_rows(checkpoint)
        # Equal norms or approximate equality are insufficient: bitwise row
        # identity is the initialization symmetry under study.
        embedding.write_bytes(row * 4474 + row[:-4] + struct.pack("<f", 0.091))
        with self.assertRaisesRegex(ValueError, "rows are not identical"):
            experiment.check_identical_embedding_rows(checkpoint)

    def test_identical_embedding_check_rejects_wrong_shape(self):
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "weight_0.bin").write_bytes(b"\x00" * (4474 * 10 * 4))
        with self.assertRaisesRegex(ValueError, "embedding shape"):
            experiment.check_identical_embedding_rows(checkpoint)

    def test_renaming_preserves_sentence_boundaries_and_repeated_tokens(self):
        rows = [[1, 2, 1], [2, 3], [0]]
        self.assertEqual(experiment.renamed_rows(rows, [0, 3, 1, 2]),
                         [[3, 1, 3], [1, 2], [0]])
        self.assertEqual(rows, [[1, 2, 1], [2, 3], [0]])

    def test_raw_integer_format_is_little_endian_and_adds_no_tokens(self):
        path = self.root / "tokens.i32"
        experiment.write_int32(path, iter([0, 1, 4474]))
        self.assertEqual(path.read_bytes(), struct.pack("<iii", 0, 1, 4474))
        with self.assertRaises(FileExistsError):
            experiment.write_int32(path, [])

    def test_text_rows_preserve_lengths_and_refuse_overwrite(self):
        path = self.root / "tokens.tsv"
        experiment.write_rows(path, [[1, 2], [3]])
        self.assertEqual(path.read_text(), "1 2\n3\n")
        with self.assertRaises(FileExistsError):
            experiment.write_rows(path, [[9]])

    def test_embedding_symmetry_uses_forward_not_inverse_permutation(self):
        source = self.root / "source"
        source.mkdir()
        embedding = struct.pack("<ffffff", 10, 11, 20, 21, 30, 31)
        (source / "weight_0.bin").write_bytes(embedding)
        (source / "weight_1.bin").write_bytes(b"unchanged positions")
        (source / "manifest.txt").write_text("unchanged metadata")
        destination = self.root / "mapped"
        # A three-cycle distinguishes pi from inverse(pi), unlike a swap.
        experiment.row_permuted_checkpoint(source, destination, [1, 2, 0], width=2)
        self.assertEqual(struct.unpack("<ffffff", (destination / "weight_0.bin").read_bytes()),
                         (30, 31, 10, 11, 20, 21))
        self.assertEqual((destination / "weight_1.bin").read_bytes(), b"unchanged positions")
        self.assertEqual((destination / "manifest.txt").read_text(), "unchanged metadata")
        self.assertEqual((source / "weight_0.bin").read_bytes(), embedding)

    def test_embedding_symmetry_rejects_bad_mapping_or_shape_before_writing(self):
        source = self.root / "source"
        source.mkdir()
        (source / "weight_0.bin").write_bytes(struct.pack("<ff", 1, 2))
        for mapping, width in [([0, 0], 1), ([1, 0], 0), ([1, 0], 2)]:
            with self.subTest(mapping=mapping, width=width):
                with self.assertRaises(ValueError):
                    experiment.row_permuted_checkpoint(source, self.root / "output",
                                                        mapping, width)
                self.assertFalse((self.root / "output").exists())

    def test_embedding_symmetry_refuses_existing_destination(self):
        source = self.root / "source"
        source.mkdir()
        (source / "weight_0.bin").write_bytes(struct.pack("<ff", 1, 2))
        destination = self.root / "output"
        destination.mkdir()
        with self.assertRaises(FileExistsError):
            experiment.row_permuted_checkpoint(source, destination, [1, 0], 1)

    def test_checkpoint_flattening_is_numeric_not_lexicographic(self):
        checkpoint = self.root / "checkpoint"
        self.checkpoint(checkpoint)
        (checkpoint / "weight_2.bin").write_bytes(struct.pack("<f", 2) * 10)
        flattened = experiment.checkpoint_bytes(checkpoint)
        self.assertEqual(len(flattened), 48680 * 4)
        self.assertEqual(flattened[(44750 + 270) * 4:(44750 + 280) * 4],
                         struct.pack("<f", 2) * 10)

    def test_checkpoint_rejects_missing_extra_and_truncated_tensors(self):
        checkpoint = self.root / "checkpoint"
        self.checkpoint(checkpoint)
        extra = checkpoint / "weight_52.bin"
        extra.write_bytes(b"extra")
        with self.assertRaises(ValueError):
            experiment.checkpoint_bytes(checkpoint)
        extra.unlink()
        (checkpoint / "weight_51.bin").write_bytes(b"")
        with self.assertRaises(ValueError):
            experiment.checkpoint_bytes(checkpoint)
        (checkpoint / "weight_51.bin").unlink()
        with self.assertRaises(ValueError):
            experiment.checkpoint_bytes(checkpoint)

    def test_audit_uses_canonical_compact_to_original_ids_after_renaming(self):
        ids = [100, 200, 300, 400, 500, 600, 50256]
        mapping = [1, 2, 3, 4, 5, 0, 6]
        original_rows = [[0, 1, 2, 3, 4, 5]] * 1024
        renamed = experiment.renamed_rows(original_rows, mapping)
        path = self.root / "predictions.tsv"
        self.predictions(path, renamed, ids)
        audit = experiment.audit_predictions(path, renamed, ids)
        self.assertEqual(audit["errors"], 0)
        self.assertEqual(audit["targets"], 2048)
        self.assertEqual(audit["exact_sentences"], 1024)
        # Inverting pi again while auditing would erroneously accept the old
        # target. The native output IDs are canonical labels of renamed IDs.
        with self.assertRaisesRegex(ValueError, "independent tokenization"):
            experiment.audit_predictions(path, original_rows, ids)

    def test_audit_counts_wrong_predictions_without_calling_them_malformed(self):
        ids = [100, 200, 300, 400, 500, 600, 50256]
        rows = [[0, 1, 2, 3, 4, 5]] * 1024
        path = self.root / "predictions.tsv"
        self.predictions(path, rows, ids, wrong=True)
        audit = experiment.audit_predictions(path, rows, ids)
        self.assertEqual(audit["errors"], 1)
        self.assertEqual(audit["exact_sentences"], 1023)

    def test_training_schedule_uses_fixed_endpoint_without_early_stop(self):
        self.assertEqual(experiment.SCHEDULE["steps"], 120000)
        self.assertEqual(experiment.SCHEDULE["batch_size"], 32)
        self.assertEqual(experiment.SCHEDULE["seed"], 1337)
        self.assertEqual(experiment.SCHEDULE["learning_rate"], 0.0012)
        self.assertEqual(experiment.SCHEDULE["training_seconds"], 0)
        self.assertFalse(experiment.SCHEDULE["stop_when_memorized"])
        self.assertEqual(experiment.MODEL["layers"], 4)
        self.assertEqual(experiment.MODEL["model_width"], 10)
        self.assertEqual(experiment.MODEL["feed_forward_width"], 20)
        self.assertEqual(experiment.MODEL["context_length"], 27)

    def test_snapshot_pins_binary_runtime_scripts_and_exact_unpadded_tokens(self):
        binary = self.root / "native"
        binary.write_bytes(b"pinned executable")
        runtime = self.root / "libcudart.so.13"
        runtime.write_bytes(b"pinned CUDA runtime")
        corpus = self.root / "corpus.txt"
        corpus.write_text("".join(f"fact {i}\n" for i in range(1024)))
        tokenizer_dir = self.root / "tokenizer"
        tokenizer_dir.mkdir()
        (tokenizer_dir / "tokenizer.json").write_text("pinned tokenizer")
        lengths = [14] * 786 + [13] * 238
        flat = [i % 4474 for i in range(14098)]
        encoded = []
        start = 0
        for length in lengths:
            encoded.append(flat[start:start + length])
            start += length
        tokenizer = mock.Mock()
        tokenizer.encode.side_effect = lambda line, **kwargs: SimpleNamespace(
            ids=encoded[int(line.split()[1])])
        tokenizer_class = mock.Mock()
        tokenizer_class.from_file.return_value = tokenizer
        args = argparse.Namespace(run_dir=self.root / "run", binary=binary,
                                  corpus=corpus, tokenizer=tokenizer_dir)
        with mock.patch.dict(sys.modules, {"tokenizers": SimpleNamespace(Tokenizer=tokenizer_class)}):
            with mock.patch.object(experiment.subprocess, "check_output",
                                   return_value=f"libcudart.so.13 => {runtime} (0xabc)\n"):
                rows, ids, hashes = experiment.snapshot(args)
                with self.assertRaises(FileExistsError):
                    experiment.snapshot(args)
        self.assertEqual(rows, encoded)
        self.assertEqual(ids, list(range(4474)) + [50256])
        self.assertEqual((args.run_dir / "inputs/base_tokens.i32").stat().st_size, 14098 * 4)
        self.assertEqual((args.run_dir / "inputs/sentence_lengths.i32").read_bytes(),
                         struct.pack("<1024i", *lengths))
        self.assertEqual((args.run_dir / "bin/libcudart.so.13").read_bytes(), runtime.read_bytes())
        self.assertEqual((args.run_dir / "bin/memorize_general_facts").read_bytes(), binary.read_bytes())
        for name in experiment.SOURCE_NAMES:
            self.assertIn("scripts/" + name, hashes)
        for relative, digest in hashes.items():
            self.assertEqual(experiment._sha256(args.run_dir / relative), digest)
        tokenizer.no_padding.assert_called_once()
        tokenizer.no_truncation.assert_called_once()

    def test_run_command_launches_only_owned_process_group(self):
        process = mock.Mock(pid=1234)
        process.wait.return_value = 0
        with mock.patch.object(experiment.subprocess, "Popen", return_value=process) as launch:
            with mock.patch.object(experiment.time, "monotonic", return_value=100):
                code = experiment.run_command(["fake"], self.root / "log", deadline=112,
                                               environment={"PINNED": "yes"})
        self.assertEqual(code, 0)
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        self.assertEqual(launch.call_args.kwargs["env"], {"PINNED": "yes"})
        process.wait.assert_called_once_with(timeout=12)

    def test_run_command_deadline_terminates_and_reaps_owned_group(self):
        process = mock.Mock(pid=1234)
        process.wait.side_effect = [subprocess.TimeoutExpired("fake", 12),
                                    subprocess.TimeoutExpired("fake", 5), -9]
        with mock.patch.object(experiment.subprocess, "Popen", return_value=process):
            with mock.patch.object(experiment.os, "killpg") as kill:
                with self.assertRaises(subprocess.TimeoutExpired):
                    experiment.run_command(["fake"], self.root / "log", deadline=0,
                                           environment={})
        self.assertEqual(kill.call_args_list,
                         [mock.call(1234, signal.SIGTERM), mock.call(1234, signal.SIGKILL)])
        self.assertEqual(process.wait.call_count, 3)

    def test_verify_trial_rejects_native_audit_success_disagreement(self):
        trial = dict(id="test", token_corpus="tokens.tsv")
        (self.root / "test").mkdir()
        with mock.patch.object(experiment, "run_command", return_value=0):
            with mock.patch.object(experiment, "audit_predictions", return_value={"errors": 1}):
                with self.assertRaisesRegex(ValueError, "disagrees"):
                    experiment.verify_trial(self.root, trial, [], [], {}, 100,
                                            self.root / "checkpoint", "verification")

    def test_report_passes_explicit_input_and_output_paths(self):
        with mock.patch.object(experiment.subprocess, "run",
                               return_value=SimpleNamespace(returncode=0)) as report:
            experiment.update_report(self.root, {"PINNED": "yes"})
        self.assertEqual(report.call_args.args[0], [
            sys.executable, str(self.root / "scripts/analyze_dataset_weights.py"),
            "--summary", str(self.root / "summary.json"),
            "--output", str(self.root / "analysis.html")])
        self.assertEqual(report.call_args.kwargs["env"], {"PINNED": "yes"})

    def test_report_errors_do_not_abort_gpu_collection(self):
        for error in [OSError("missing analyzer"), subprocess.TimeoutExpired("analyzer", 60)]:
            with self.subTest(error=error):
                with mock.patch.object(experiment.subprocess, "run", side_effect=error):
                    with redirect_stdout(io.StringIO()) as output:
                        experiment.update_report(self.root, {})
                    self.assertIn("WARNING", output.getvalue())

    def run_fixture(self, *, changed_initial=False, changed_input=False,
                    training_timeout=False, identical=False, first_memorized_step=256,
                    baseline_final_wrong=False, missing_first_checkpoint=False,
                    first_checkpoint_wrong=False, nonidentical_rows=False):
        """Mock GPU execution but keep all file audits and gates real."""
        run = self.root / "run"
        rows = [[0, 1, 2, 3, 4, 5]] * 1024
        ids = list(range(4474)) + [50256]
        commands = []
        self.commands = commands

        def snapshot(args):
            run.mkdir()
            (run / "inputs").mkdir()
            (run / "inputs/corpus.txt").write_text("pinned corpus")
            return rows, ids, {"inputs/corpus.txt": experiment._sha256(run / "inputs/corpus.txt")}

        def command(argv, log_path, *, deadline, environment):
            commands.append(argv)
            options = dict(argument[2:].split("=", 1) for argument in argv[1:])
            transformed = [[int(token) for token in line.split()]
                           for line in Path(options["token_corpus"]).read_text().splitlines()]
            output = Path(options["output_dir"])
            trial_id = Path(options["token_corpus"]).parent.name
            wrong = baseline_final_wrong and trial_id == "baseline"
            if output.name == "memorization_verification":
                wrong = first_checkpoint_wrong
            if options["mode"] == "train_model":
                if training_timeout:
                    raise subprocess.TimeoutExpired(argv, 600)
                output /= "layers_4"
                checkpoints = Path(options["checkpoint_dir"]) / "layers_4"
                initial = checkpoints / "step_0"
                initial_value = 1 if changed_initial and "baseline_repeat" in str(initial) else 0
                self.checkpoint(initial, initial_value)
                if nonidentical_rows:
                    embedding = initial / "weight_0.bin"
                    embedding.write_bytes(embedding.read_bytes()[:-4] + struct.pack("<f", 1))
                final = checkpoints / "step_120000"
                self.checkpoint(final)
                if identical and first_memorized_step >= 0 and not missing_first_checkpoint:
                    first = checkpoints / f"step_{first_memorized_step}"
                    if not first.exists():
                        self.checkpoint(first, 1)
                output.mkdir(parents=True)
                (output / "result.txt").write_text(
                    f"step=120000\nparameters=48680\nsuccess={int(not wrong)}\nerrors={int(wrong)}\n"
                    f"first_memorized_step={first_memorized_step}\ncheckpoint={final}\n")
            self.predictions(output / "final_predictions.tsv", transformed, ids, wrong=wrong)
            if changed_input and options["mode"] == "infer_model":
                (run / "inputs/corpus.txt").write_text("modified after capture")
            return 2 if wrong else 0

        args = argparse.Namespace(run_dir=run, duration_seconds=600,
                                  max_trials=0 if identical else 3,
                                  identical_token_embeddings=identical)
        with mock.patch.object(experiment, "snapshot", side_effect=snapshot):
            with mock.patch.object(experiment, "run_command", side_effect=command):
                with mock.patch.object(experiment.subprocess, "run") as report:
                    with redirect_stdout(io.StringIO()):
                        result = experiment.run(args)
                    completed = sum(trial["status"] in ("verified", "not_memorized")
                                    for trial in result["trials"])
                    self.assertEqual(report.call_count, completed + 1)
        return result, commands

    def test_end_to_end_controls_pairs_and_fresh_audits(self):
        result, commands = self.run_fixture()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["trials"]), 3)
        baseline, repeat, renamed = result["trials"]
        self.assertTrue(repeat["identical_to_baseline"])
        self.assertEqual({trial["initial_sha256"] for trial in result["trials"]},
                         {baseline["initial_sha256"]})
        self.assertEqual(renamed["symmetry_errors"], 0)
        self.assertTrue(all(trial["status"] == "verified" for trial in result["trials"]))
        self.assertEqual(len(commands), 7)  # 3 training + 3 fresh + 1 symmetry.
        training = [argv for argv in commands if "--mode=train_model" in argv]
        for argv in training:
            self.assertIn("--stop_when_memorized=false", argv)
            self.assertIn("--steps=120000", argv)
            self.assertIn("--training_seconds=0", argv)
            self.assertIn("--batch_size=32", argv)
        saved = json.loads((self.root / "run/summary.json").read_text())
        self.assertEqual(saved["status"], "complete")
        self.assertEqual(saved["trials"][2]["step"], 120000)

    def test_identical_initialization_gate_then_exactly_three_training_permutations(self):
        original_schedule = dict(experiment.SCHEDULE)
        result, commands = self.run_fixture(identical=True)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["baseline_memorization_verified"])
        self.assertEqual(len(result["trials"]), 4)
        training = [argv for argv in commands if "--mode=train_model" in argv]
        self.assertEqual(len(training), 4)
        for argv in training:
            self.assertIn("--identical_token_embeddings=true", argv)
            self.assertIn("--steps=120000", argv)
            self.assertIn("--stop_when_memorized=false", argv)
        for argv in commands:
            if "--mode=infer_model" in argv:
                self.assertFalse(any(arg.startswith("--identical_token_embeddings") for arg in argv))
        self.assertTrue(result["schedule"]["identical_token_embeddings"])
        self.assertEqual(experiment.SCHEDULE, original_schedule)
        baseline = result["trials"][0]
        self.assertEqual(baseline["memorization_audit"]["errors"], 0)
        self.assertTrue(baseline["memorization_checkpoint"].endswith("/step_256"))
        first_gate = next(i for i, argv in enumerate(commands)
                          if any(arg.endswith("/baseline/memorization_verification") for arg in argv))
        first_permutation = next(i for i, argv in enumerate(commands)
                                 if any(arg.endswith("/rename_000_0002/tokens.tsv") for arg in argv))
        self.assertLess(first_gate, first_permutation)

    def test_identical_initialization_stops_if_baseline_never_memorized(self):
        result, commands = self.run_fixture(identical=True, first_memorized_step=-1,
                                            baseline_final_wrong=True)
        self.assertEqual(result["status"], "baseline_not_memorized")
        self.assertFalse(result["baseline_memorization_verified"])
        self.assertEqual(len(result["trials"]), 1)
        self.assertEqual(result["trials"][0]["status"], "not_memorized")
        self.assertEqual(sum("--mode=train_model" in argv for argv in commands), 1)
        self.assertFalse(any("rename_" in arg for argv in commands for arg in argv))

    def test_early_memorization_allows_later_regression_but_keeps_fixed_endpoints(self):
        result, commands = self.run_fixture(identical=True, baseline_final_wrong=True)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["baseline_memorization_verified"])
        baseline = result["trials"][0]
        self.assertEqual(baseline["status"], "not_memorized")
        self.assertEqual(baseline["audit"]["errors"], 1)
        self.assertEqual(baseline["memorization_audit"]["errors"], 0)
        self.assertTrue(baseline["final_checkpoint"].endswith("/step_120000"))
        # First-perfect weights are all ones in this fixture; final weights
        # are zeros. Symmetry comparisons must transform the latter only.
        final_bytes = experiment.checkpoint_bytes(Path(baseline["final_checkpoint"]))
        self.assertNotEqual(final_bytes, experiment.checkpoint_bytes(
            Path(baseline["memorization_checkpoint"])))
        for trial in result["trials"][1:]:
            self.assertEqual(experiment.checkpoint_bytes(Path(trial["symmetry_checkpoint"])),
                             final_bytes)
        self.assertEqual(sum("--mode=train_model" in argv for argv in commands), 4)

    def test_missing_first_perfect_checkpoint_prevents_permutation_training(self):
        with self.assertRaisesRegex(ValueError, "expected exactly 52 canonical weights"):
            self.run_fixture(identical=True, missing_first_checkpoint=True)
        self.assertEqual(sum("--mode=train_model" in argv for argv in self.commands), 1)
        saved = json.loads((self.root / "run/summary.json").read_text())
        self.assertFalse(saved["baseline_memorization_verified"])
        self.assertEqual(saved["status"], "error")

    def test_first_perfect_checkpoint_is_freshly_audited_before_gate_passes(self):
        with self.assertRaisesRegex(ValueError, "first-perfect checkpoint failed"):
            self.run_fixture(identical=True, first_checkpoint_wrong=True)
        self.assertEqual(sum("--mode=train_model" in argv for argv in self.commands), 1)
        saved = json.loads((self.root / "run/summary.json").read_text())
        self.assertFalse(saved["baseline_memorization_verified"])
        self.assertEqual(saved["status"], "error")

    def test_runner_rejects_nonidentical_initial_rows_before_memorization_gate(self):
        with self.assertRaisesRegex(ValueError, "rows are not identical"):
            self.run_fixture(identical=True, nonidentical_rows=True)
        self.assertEqual(sum("--mode=train_model" in argv for argv in self.commands), 1)
        self.assertEqual(len(self.commands), 1)
        saved = json.loads((self.root / "run/summary.json").read_text())
        self.assertFalse(saved["baseline_memorization_verified"])

    def test_run_refuses_initialization_changes_between_trials(self):
        with self.assertRaisesRegex(ValueError, "initial weights differ"):
            self.run_fixture(changed_initial=True)
        saved = json.loads((self.root / "run/summary.json").read_text())
        self.assertEqual(saved["status"], "error")
        self.assertEqual(saved["trials"][-1]["status"], "incomplete")

    def test_run_refuses_mutation_of_pinned_inputs(self):
        with self.assertRaisesRegex(ValueError, "pinned experiment input changed"):
            self.run_fixture(changed_input=True)
        saved = json.loads((self.root / "run/summary.json").read_text())
        self.assertEqual(len(saved["trials"]), 1)
        self.assertEqual(saved["trials"][0]["status"], "verified")

    def test_deadline_does_not_mislabel_an_incomplete_endpoint_as_fixed_step_data(self):
        result, commands = self.run_fixture(training_timeout=True)
        self.assertEqual(result["status"], "deadline_reached")
        self.assertEqual(len(result["trials"]), 1)
        self.assertEqual(result["trials"][0]["status"], "incomplete")
        self.assertNotIn("final_checkpoint", result["trials"][0])
        self.assertEqual(len(commands), 1)

    def test_run_requires_healthy_disk_before_snapshot(self):
        args = argparse.Namespace(run_dir=self.root / "run")
        with mock.patch.object(experiment.shutil, "disk_usage", return_value=mock.Mock(free=1)):
            with mock.patch.object(experiment, "snapshot") as snapshot:
                with self.assertRaisesRegex(ValueError, "10 GiB"):
                    experiment.run(args)
                snapshot.assert_not_called()

    def test_cli_defaults_and_invalid_deadlines(self):
        base = [f"--binary={self.root / 'binary'}", f"--corpus={self.root / 'corpus'}",
                f"--tokenizer={self.root / 'tokenizer'}", f"--run_dir={self.root / 'run'}"]
        args = experiment.parse_args(base)
        self.assertEqual(args.duration_seconds, 10800)
        self.assertEqual(args.max_trials, 0)
        self.assertFalse(args.identical_token_embeddings)
        self.assertTrue(experiment.parse_args(base + ["--identical_token_embeddings"])
                        .identical_token_embeddings)
        for option in ["--duration_seconds=nan", "--duration_seconds=inf",
                       "--duration_seconds=599", "--max_trials=-1"]:
            with self.subTest(option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    experiment.parse_args(base + [option])

    def test_cli_refuses_artifacts_inside_repository(self):
        repository = Path(experiment.__file__).resolve().parents[2]
        for target in [repository, repository / "runs"]:
            with self.subTest(target=target), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    experiment.parse_args(["--binary=/tmp/binary", "--corpus=/tmp/corpus",
                                           "--tokenizer=/tmp/tokenizer", f"--run_dir={target}"])


if __name__ == "__main__":
    unittest.main()
