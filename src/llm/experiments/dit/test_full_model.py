"""Opt-in full 51M-parameter Shakespeare/GPU training and influence test.

Unlike the fast analytic tests, this uses the unmodified production dimensions,
real GPT-2 tokens, actual checkpoint I/O, and full-parameter Hessian propagation.
It needs roughly 1 GB of temporary disk space and a CUDA GPU.
"""

import math
import os
from pathlib import Path
import tempfile
import unittest

import torch

from src.llm.experiments.dit.data import load_text_examples, load_tokenizer
from src.llm.experiments.dit.dit import configure_runtime, training_loss
from src.llm.experiments.dit.influence import compute_influence, validate_leave_one_out
from src.llm.experiments.dit.model import Gpt2
from src.llm.experiments.dit.trajectory import train_trajectory


@unittest.skipUnless(
    os.environ.get("PLUTO_DIT_FULL_MODEL") == "1",
    "set PLUTO_DIT_FULL_MODEL=1 for the full GPU integration test",
)
class FullModelTest(unittest.TestCase):
    def test_full_gpt2_shakespeare_train_replay_infer_and_finite_loo(self):
        # An explicitly requested integration run must fail, not silently skip,
        # when prerequisites are missing.
        self.assertTrue(torch.cuda.is_available())
        tokenizer_dir = os.environ["PLUTO_GPT2_TOKENIZER_DIR"]
        configure_runtime("cuda")
        corpus = Path(__file__).resolve().parents[4] / "testdata/shakespeare.txt"
        examples = load_text_examples(corpus, tokenizer_dir, 8, max_examples=4)
        tokenizer = load_tokenizer(tokenizer_dir)
        model = Gpt2(device="cuda")
        self.assertEqual(sum(p.numel() for p in model.parameters()), 51_483_648)
        prompt = torch.tensor(
            [tokenizer.encode("First Citizen:", add_special_tokens=False).ids],
            dtype=torch.int64,
            device="cuda",
        )
        query = lambda m: -m(prompt)[0, -1].log_softmax(dim=-1)[198]
        with tempfile.TemporaryDirectory() as directory:
            run = train_trajectory(
                model,
                examples.tokens,
                Path(directory) / "run",
                steps=3,
                batch_size=2,
                learning_rate=1e-5,
                seed=123,
                checkpoint_interval=2,
                loss_fn=training_loss,
            )
            # Step 1 is unsaved; the late-window query must really replay it.
            result = compute_influence(
                model, run, training_loss, query, t1=1, t2=3, sample_ids=[0, 1, 2, 3]
            )
            repeated = compute_influence(
                model, run, training_loss, query, t1=1, t2=3, sample_ids=[0, 1, 2, 3]
            )
            self.assertEqual(result, repeated)
            actual = validate_leave_one_out(
                model, run, training_loss, query, t1=1, t2=3, sample_ids=[0, 1, 2, 3]
            )
            for sample, estimate in result.scores.items():
                self.assertTrue(math.isfinite(estimate))
                # This short, small-rate trajectory is deliberately in a local
                # regime. This tolerance is NOT an accuracy guarantee for long
                # runs/full-strength deletion; analytic tests establish the
                # exact derivative identity independently.
                self.assertLess(abs(estimate - actual[sample]), 0.002)
            self.assertTrue(any(abs(value) > 1e-5 for value in result.scores.values()))
            print(
                "full GPT-2 late-window DIT:",
                result.scores,
                "finite LOO:",
                actual,
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
