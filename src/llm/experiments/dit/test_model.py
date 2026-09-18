"""Architecture, checkpoint, causality, and complete-model second derivatives."""

from pathlib import Path
import tempfile
import unittest

import torch
from torch.func import functional_call

from src.llm.experiments.dit.model import GPT2Config, Gpt2


def tiny_config(**overrides):
    values = dict(
        vocab_size=7,
        padded_vocab_size=8,
        context_length=4,
        n_layers=1,
        d_model=4,
        n_heads=2,
        d_ff=8,
    )
    values.update(overrides)
    return GPT2Config(**values)


class Gpt2Test(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_full_recipe_dimensions_and_unique_parameter_count(self):
        model = Gpt2()
        self.assertEqual(len(list(model.parameters())), 100)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 51_483_648)
        self.assertEqual(model.token_embedding.weight.shape, (50_272, 512))
        self.assertEqual(model.position_embedding.weight.shape, (1_024, 512))
        self.assertEqual(len(model.blocks), 8)
        self.assertEqual(model.blocks[0].qkv.weight.shape, (1_536, 512))
        self.assertEqual(model.blocks[0].mlp_input.weight.shape, (2_048, 512))

    def test_seed_reproducibility_preserves_rng_and_bias_initialization(self):
        torch.manual_seed(555)
        before = torch.get_rng_state()
        one = Gpt2(tiny_config(), seed=12)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        two = Gpt2(tiny_config(), seed=12)
        three = Gpt2(tiny_config(), seed=13)
        self.assertTrue(
            all(torch.equal(a, b) for a, b in zip(one.parameters(), two.parameters()))
        )
        self.assertFalse(
            torch.equal(one.token_embedding.weight, three.token_embedding.weight)
        )
        for name, value in one.named_parameters():
            if name.endswith(".bias"):
                self.assertEqual(torch.count_nonzero(value).item(), 0)
            if "norm.weight" in name:
                self.assertTrue(torch.equal(value, torch.ones_like(value)))

    def test_future_tokens_and_other_samples_cannot_change_prefix(self):
        model = Gpt2(tiny_config(), dtype=torch.float64)
        first = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
        second = torch.tensor([[1, 2, 6, 0], [6, 0, 1, 2]])
        one, two = model(first), model(second)
        self.assertEqual(one.shape, (2, 4, 7))
        torch.testing.assert_close(one[0, :2], two[0, :2], rtol=0, atol=0)
        torch.testing.assert_close(
            one[0, :2], model(first[:1, :2])[0], rtol=0, atol=1e-16
        )
        self.assertFalse(torch.equal(one[0, 2:], two[0, 2:]))

    def test_per_example_loss_uses_shifted_tokens_and_mean_over_positions(self):
        model = Gpt2(tiny_config(), dtype=torch.float64)
        examples = torch.tensor([[1, 2, 3], [3, 4, 5]])
        logits = model(examples[:, :-1])
        manual = (
            -logits.log_softmax(-1)
            .gather(-1, examples[:, 1:, None])
            .squeeze(-1)
            .mean(-1)
        )
        torch.testing.assert_close(model.per_example_loss(examples), manual)
        torch.testing.assert_close(model.loss(examples), manual.mean())
        # A padded embedding class cannot enter the denominator.
        with torch.no_grad():
            model.token_embedding.weight[7].fill_(1e20)
        torch.testing.assert_close(model.per_example_loss(examples), manual)

    def test_gradgradcheck_complete_transformer_and_tied_embedding(self):
        model = Gpt2(tiny_config(), dtype=torch.float64)
        names, parameters = zip(*model.named_parameters())
        examples = torch.tensor([[1, 2, 3]])

        def loss(*values):
            logits = functional_call(model, dict(zip(names, values)), examples[:, :-1])
            return torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), examples[:, 1:]
            )

        self.assertTrue(torch.autograd.gradcheck(loss, parameters, fast_mode=True))
        self.assertTrue(torch.autograd.gradgradcheck(loss, parameters, fast_mode=True))
        grads = torch.autograd.grad(loss(*parameters), parameters)
        embedding_grad = grads[0]
        # Row 6 is only used by the head, row 1 also has the lookup path.
        self.assertGreater(embedding_grad[6].norm().item(), 0)
        self.assertGreater(embedding_grad[1].norm().item(), 0)
        self.assertEqual(embedding_grad[7].abs().max().item(), 0)

    def test_hessian_vector_product_matches_gradient_finite_difference(self):
        model = Gpt2(tiny_config(), dtype=torch.float64)
        examples = torch.tensor([[1, 2, 3], [2, 3, 4]])
        parameters = tuple(model.parameters())
        generator = torch.Generator().manual_seed(987)
        direction = tuple(
            torch.randn(p.shape, generator=generator, dtype=p.dtype) * 0.02
            for p in parameters
        )
        gradients = torch.autograd.grad(
            model.loss(examples), parameters, create_graph=True
        )
        hvp = torch.autograd.grad(
            sum((g * v).sum() for g, v in zip(gradients, direction)), parameters
        )
        base = [p.detach().clone() for p in parameters]
        epsilon = 1e-5
        numerical = []
        for sign in (1, -1):
            with torch.no_grad():
                for parameter, initial, vector in zip(parameters, base, direction):
                    parameter.copy_(initial + sign * epsilon * vector)
            numerical.append(torch.autograd.grad(model.loss(examples), parameters))
        for actual, positive, negative in zip(hvp, *numerical):
            torch.testing.assert_close(
                actual, (positive - negative) / (2 * epsilon), rtol=2e-5, atol=2e-7
            )

    def test_native_checkpoint_order_transposes_and_roundtrip(self):
        model = Gpt2(tiny_config())
        layout = model.pluto_weight_layout()
        self.assertEqual(
            [name for name, _, _ in layout],
            [
                "token_embedding.weight",
                "position_embedding.weight",
                "blocks.0.attention_norm.weight",
                "blocks.0.attention_norm.bias",
                "blocks.0.qkv.weight",
                "blocks.0.qkv.bias",
                "blocks.0.attention_projection.weight",
                "blocks.0.attention_projection.bias",
                "blocks.0.mlp_norm.weight",
                "blocks.0.mlp_norm.bias",
                "blocks.0.mlp_input.weight",
                "blocks.0.mlp_input.bias",
                "blocks.0.mlp_output.weight",
                "blocks.0.mlp_output.bias",
                "final_norm.weight",
                "final_norm.bias",
            ],
        )
        with torch.no_grad():
            for index, (_, value, _) in enumerate(layout):
                value.copy_(
                    torch.arange(value.numel()).reshape(value.shape) / 100 + index
                )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "checkpoint"
            model.write_pluto_checkpoint(directory)
            qkv_raw = torch.frombuffer(
                bytearray((directory / "weight_4.bin").read_bytes()),
                dtype=torch.float32,
            ).reshape(4, 12)
            torch.testing.assert_close(
                qkv_raw, model.blocks[0].qkv.weight.t(), rtol=0, atol=0
            )
            other = Gpt2(tiny_config(), seed=987)
            other.load_pluto_checkpoint(directory)
            for a, b in zip(model.parameters(), other.parameters()):
                self.assertTrue(torch.equal(a, b))
            with self.assertRaises(FileExistsError):
                other.write_pluto_checkpoint(directory)
            (directory / "weight_15.bin").write_bytes(b"oops")
            before = [p.clone() for p in other.parameters()]
            with self.assertRaisesRegex(ValueError, "size"):
                other.load_pluto_checkpoint(directory)
            self.assertTrue(
                all(torch.equal(a, b) for a, b in zip(before, other.parameters()))
            )

    def test_checkpoint_rejects_extra_missing_and_nonfinite_files(self):
        model = Gpt2(tiny_config())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "checkpoint"
            model.write_pluto_checkpoint(directory)
            (directory / "weight_999.bin").write_bytes(b"junk")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                model.load_pluto_checkpoint(directory)
            (directory / "weight_999.bin").unlink()
            (directory / "weight_0.bin").write_bytes(
                torch.full_like(model.token_embedding.weight, torch.nan)
                .numpy()
                .tobytes()
            )
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                model.load_pluto_checkpoint(directory)
            (directory / "weight_15.bin").unlink()
            with self.assertRaisesRegex(ValueError, "mismatch"):
                model.load_pluto_checkpoint(directory)

    def test_invalid_config_tokens_targets_and_precision(self):
        for options in (
            dict(n_layers=0),
            dict(d_model=3),
            dict(padded_vocab_size=6),
            dict(layer_norm_epsilon=float("nan")),
            dict(n_heads=True),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                tiny_config(**options)
        with self.assertRaises(ValueError):
            Gpt2(tiny_config(), dtype=torch.bfloat16)
        model = Gpt2(tiny_config())
        for tokens in (
            torch.tensor([[7]]),
            torch.tensor([[-1]]),
            torch.ones(1, 2),
            torch.ones(0, 2, dtype=torch.int64),
            torch.ones(1, 5, dtype=torch.int64),
        ):
            with self.subTest(shape=tokens.shape), self.assertRaises(ValueError):
                model(tokens)
        with self.assertRaises(ValueError):
            model.loss(torch.tensor([[1, 7]]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
    def test_cuda_second_derivative_matches_cpu(self):
        rng_before = torch.cuda.get_rng_state()
        cpu = Gpt2(tiny_config(), dtype=torch.float64)
        gpu = Gpt2(tiny_config(), dtype=torch.float64, device="cuda")
        self.assertTrue(torch.equal(torch.cuda.get_rng_state(), rng_before))
        examples = torch.tensor([[1, 2, 3]])
        results = []
        for model, batch in ((cpu, examples), (gpu, examples.cuda())):
            parameters = tuple(model.parameters())
            gradient = torch.autograd.grad(
                model.loss(batch), parameters, create_graph=True
            )
            results.append(
                torch.autograd.grad(sum(g.sum() for g in gradient), parameters)
            )
        for cpu_hvp, gpu_hvp in zip(*results):
            torch.testing.assert_close(cpu_hvp, gpu_hvp.cpu(), rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
