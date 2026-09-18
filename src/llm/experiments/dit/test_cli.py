"""Real train/replay/HVP/inference round trips through the command interface.

Only architecture defaults are reduced; training, differentiation, checkpoint
replay, tokenization, report writing, and finite deletion retraining are real.
The separate model tests check the production GPT-2 architecture dimensions.
"""

from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from src.llm.experiments.dit import dit
from src.llm.experiments.dit.model import GPT2Config, Gpt2
from src.llm.experiments.dit.trajectory import Trajectory


def _small_default_config(**values):
    if values:
        return GPT2Config(**values)
    return GPT2Config(
        vocab_size=7,
        padded_vocab_size=8,
        context_length=4,
        n_layers=1,
        d_model=4,
        n_heads=2,
        d_ff=8,
    )


class CliTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tokenizer_dir = self.root / "tokenizer"
        self.tokenizer_dir.mkdir()
        tokenizer = Tokenizer(
            models.WordLevel(
                {
                    "<unk>": 0,
                    "to": 1,
                    "be": 2,
                    "or": 3,
                    "not": 4,
                    "the": 5,
                    "<|endoftext|>": 6,
                },
                unk_token="<unk>",
            )
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(str(self.tokenizer_dir / "tokenizer.json"))
        self.corpus = self.root / "shakespeare.txt"
        self.corpus.write_text(
            "to be or not to be to be or not to be", encoding="utf-8"
        )
        self.run_dir = self.root / "run"
        patcher = mock.patch.object(
            dit, "GPT2Config", side_effect=_small_default_config
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.train_arguments = [
            "train",
            "--run_dir",
            str(self.run_dir),
            "--corpus",
            str(self.corpus),
            "--tokenizer_dir",
            str(self.tokenizer_dir),
            "--device",
            "cpu",
            "--dtype",
            "float64",
            "--steps",
            "3",
            "--batch_size",
            "2",
            "--context_tokens",
            "2",
            "--max_examples",
            "4",
            "--learning_rate",
            ".0001",
            "--checkpoint_interval",
            "2",
            "--seed",
            "19",
        ]

    def invoke(self, arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            dit.main(arguments)
        return output.getvalue()

    def train(self, extra=()):
        return self.invoke(self.train_arguments + list(extra))

    def infer(self, name, extra=()):
        output = self.root / f"{name}.json"
        arguments = ["infer", "--run_dir", str(self.run_dir), "--output", str(output)]
        stdout = self.invoke(arguments + list(extra))
        return json.loads(output.read_text()), stdout

    def edit_manifest(self, mutate):
        path = self.run_dir / "manifest.json"
        manifest = json.loads(path.read_text())
        mutate(manifest)
        path.write_text(json.dumps(manifest))

    def test_train_infer_all_query_modes_and_generate(self):
        exported = self.root / "native_checkpoint"
        stdout = self.train(["--export_checkpoint", str(exported)])
        self.assertIn("Completed DIT trajectory", stdout)
        self.assertIn("step 3:", stdout)
        run = Trajectory.open(self.run_dir)
        self.assertEqual(run.sample_count, 4)
        self.assertEqual([c["step"] for c in run.manifest["checkpoints"]], [0, 2, 3])
        # The same exported bytes can be consumed by native-compatible loading.
        imported = Gpt2(_small_default_config(), dtype=torch.float32)
        imported.load_pluto_checkpoint(exported)
        original = Gpt2(_small_default_config(), dtype=torch.float64)
        run.load_state(original, 3, dit.training_loss)
        for expected, actual in zip(original.parameters(), imported.parameters()):
            torch.testing.assert_close(expected.float(), actual, rtol=0, atol=0)
        for query in ("logit", "probability", "loss", "parameter"):
            with self.subTest(query=query):
                extra = [
                    "--query",
                    query,
                    "--t1",
                    "1",
                    "--t2",
                    "3",
                    "--sample_ids",
                    "0,2",
                    "--validate_loo",
                ]
                if query == "parameter":
                    extra += [
                        "--parameter",
                        "token_embedding.weight",
                        "--coordinate",
                        "1,0",
                    ]
                else:
                    extra += ["--prompt", "to be", "--target_id", "3"]
                report, printed = self.infer(query, extra)
                self.assertEqual(report["window"], [1, 3])
                self.assertEqual(report["query"]["kind"], query)
                self.assertEqual(
                    {row["example_id"] for row in report["examples"]}, {0, 2}
                )
                self.assertTrue(math.isfinite(report["query_start"]))
                self.assertTrue(math.isfinite(report["query_end"]))
                self.assertIn("Validating with finite leave-one-out", printed)
                for row in report["examples"]:
                    self.assertTrue(math.isfinite(row["dit_score"]))
                    self.assertTrue(math.isfinite(row["finite_loo"]))
                    self.assertEqual(
                        row["token_ids"], run.examples[row["example_id"]].tolist()
                    )
                    self.assertEqual(row["token_offset"], row["example_id"] * 3)
                    self.assertEqual(
                        row["absolute_error"], abs(row["dit_score"] - row["finite_loo"])
                    )
        generated = self.invoke(
            [
                "generate",
                "--run_dir",
                str(self.run_dir),
                "--prompt",
                "to be",
                "--max_new_tokens",
                "2",
            ]
        )
        self.assertTrue(generated.strip().startswith("to be"))
        # Step 1 is deliberately not checkpointed: this exercises actual SGD
        # replay before text generation, rather than only loading stored bytes.
        generated = self.invoke(
            [
                "generate",
                "--run_dir",
                str(self.run_dir),
                "--step",
                "1",
                "--prompt",
                "to be",
                "--max_new_tokens",
                "0",
            ]
        )
        self.assertEqual(generated.strip(), "to be")

    def test_saved_run_relocates_without_original_corpus_or_tokenizer_path(self):
        self.train()
        moved = self.root / "relocated_run"
        self.run_dir.rename(moved)
        self.run_dir = moved
        tokenizer_copy = self.root / "relocated_tokenizer"
        self.tokenizer_dir.rename(tokenizer_copy)
        self.corpus.unlink()
        report, _ = self.infer(
            "portable",
            [
                "--tokenizer_dir",
                str(tokenizer_copy),
                "--prompt",
                "to be",
                "--sample_ids",
                "1",
            ],
        )
        self.assertEqual(report["examples"][0]["example_id"], 1)
        self.assertEqual(report["examples"][0]["token_ids"], [4, 1, 2])
        self.assertEqual(report["run_dir"], str(moved.resolve()))

    def test_native_initial_checkpoint_defines_trajectory_theta_zero(self):
        exported = self.root / "initial_weights"
        self.train(["--export_checkpoint", str(exported)])
        resumed = self.root / "initialized_from_native"
        self.train(
            [
                "--run_dir",
                str(resumed),
                "--initial_checkpoint",
                str(exported),
                "--steps",
                "1",
                "--seed",
                "999",
            ]
        )
        run = Trajectory.open(resumed)
        self.assertEqual(run.metadata["initial_checkpoint"], str(exported.resolve()))
        expected = Gpt2(_small_default_config(), dtype=torch.float64)
        expected.load_pluto_checkpoint(exported)
        actual = Gpt2(_small_default_config(), seed=999, dtype=torch.float64)
        run.load_state(actual, 0, dit.training_loss)
        for left, right in zip(expected.parameters(), actual.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_training_run_and_inference_reports_are_never_overwritten(self):
        self.train()
        manifest_before = (self.run_dir / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.train()
        self.assertEqual((self.run_dir / "manifest.json").read_bytes(), manifest_before)
        self.infer("preserved", ["--prompt", "to be"])
        output = self.root / "preserved.json"
        original = output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.infer("preserved", ["--prompt", "to be"])
        self.assertEqual(output.read_bytes(), original)

    def test_runtime_source_and_tokenizer_mismatch_are_rejected(self):
        self.train()
        path = self.run_dir / "manifest.json"
        original_manifest = path.read_bytes()
        for field in ("runtime", "source_sha256"):
            with self.subTest(field=field):
                self.edit_manifest(
                    lambda m: m["metadata"][field].update(unexpected="changed")
                )
                with self.assertRaisesRegex(ValueError, "runtime/source"):
                    self.infer(field, ["--prompt", "to be"])
                self.assertFalse((self.root / f"{field}.json").exists())
                path.write_bytes(original_manifest)
        tokenizer_path = self.tokenizer_dir / "tokenizer.json"
        tokenizer_path.write_text(tokenizer_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "tokenizer differs"):
            self.infer("tokenizer_changed", ["--prompt", "to be"])

    def test_generation_saved_checkpoint_allows_changed_runtime_but_replay_does_not(
        self,
    ):
        self.train()
        self.edit_manifest(
            lambda m: m["metadata"]["runtime"].update(python="different-version")
        )
        generated = self.invoke(
            [
                "generate",
                "--run_dir",
                str(self.run_dir),
                "--step",
                "2",
                "--prompt",
                "to be",
                "--max_new_tokens",
                "0",
            ]
        )
        self.assertEqual(generated.strip(), "to be")
        with self.assertRaisesRegex(ValueError, "runtime/source"):
            self.invoke(
                [
                    "generate",
                    "--run_dir",
                    str(self.run_dir),
                    "--step",
                    "1",
                    "--prompt",
                    "to be",
                    "--max_new_tokens",
                    "0",
                ]
            )

    def test_automatic_target_is_selected_once_and_not_changed_by_model(self):
        self.train()
        for query_kind in ("logit", "probability"):
            with self.subTest(query=query_kind):
                args = dit.make_parser().parse_args(
                    [
                        "infer",
                        "--run_dir",
                        str(self.run_dir),
                        "--output",
                        str(self.root / "unused.json"),
                        "--query",
                        query_kind,
                        "--prompt",
                        "to be",
                    ]
                )
                run, model, tokenizer = dit._load_run(args)
                query, description = dit._query(args, run, model, tokenizer, 3)
                tokens = dit._prompt_tokens(model, tokenizer, "to be")
                selected = description["target_id"]
                self.assertEqual(selected, int(model(tokens)[0, -1].argmax()))
                replacement = (selected + 1) % model.config.vocab_size
                # Force a different winner by making the final hidden vector
                # constant and assigning another tied embedding row a score.
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.zero_()
                    model.final_norm.bias.fill_(1)
                    model.token_embedding.weight[replacement].fill_(2)
                logits = model(tokens)[0, -1]
                self.assertEqual(int(logits.argmax()), replacement)
                expected = (
                    logits[selected]
                    if query_kind == "logit"
                    else logits.softmax(-1)[selected]
                )
                torch.testing.assert_close(query(model), expected, rtol=0, atol=0)
                self.assertEqual(description["target_id"], selected)

    def test_invalid_queries_windows_and_prompts_leave_no_report(self):
        self.train()
        cases = [
            ["--query", "loss", "--prompt", "to"],
            ["--prompt", "to", "--target_id", "7"],
            ["--prompt", ""],
            ["--prompt", "to be or not to"],
            ["--query", "parameter", "--parameter", "missing", "--coordinate", "0"],
            [
                "--query",
                "parameter",
                "--parameter",
                "token_embedding.weight",
                "--coordinate",
                "1,-1",
            ],
            [
                "--query",
                "parameter",
                "--parameter",
                "token_embedding.weight",
                "--coordinate",
                "1,0",
                "--prompt",
                "to",
            ],
            ["--prompt", "to", "--parameter", "token_embedding.weight"],
            ["--prompt", "to", "--t1", "2", "--t2", "2"],
            ["--prompt", "to", "--t2", "4"],
            ["--prompt", "to", "--sample_ids", "4"],
        ]
        for index, extra in enumerate(cases):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.infer(f"bad_{index}", extra)
            self.assertFalse((self.root / f"bad_{index}.json").exists())

    def test_module_help_subprocess_documents_modes(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.llm.experiments.dit.dit", "--help"],
            cwd=Path(__file__).resolve().parents[4],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("train", result.stdout)
        self.assertIn("infer", result.stdout)
        self.assertIn("generate", result.stdout)


if __name__ == "__main__":
    unittest.main()
