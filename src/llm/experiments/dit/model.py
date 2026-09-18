"""Differentiable GPT-2 recipe used by dynamic influence tracking.

This is the architecture in llm/recipes/gpt2.cc, expressed in ordinary PyTorch
operations so Hessian-vector products differentiate the *entire* model. In
particular, attention deliberately avoids FlashAttention/fused SDPA, whose
backward does not generally support a second derivative. FP32 is the default;
FP64 is useful for numerical checks. This is not a bitwise emulation of Pluto's
FP16/BF16 matrix instructions, and seeded initialization uses PyTorch's RNG,
not the native C++ normal sampler. Native FP32 master weights can be imported.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPT2Config:
    vocab_size: int = 50_257
    padded_vocab_size: int = 50_272
    context_length: int = 1_024
    n_layers: int = 8
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2_048
    layer_norm_epsilon: float = 1e-5

    def __post_init__(self):
        for name in (
            "vocab_size",
            "padded_vocab_size",
            "context_length",
            "n_layers",
            "d_model",
            "n_heads",
            "d_ff",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.padded_vocab_size < self.vocab_size:
            raise ValueError("padded_vocab_size must cover the logical vocabulary")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not math.isfinite(self.layer_norm_epsilon) or self.layer_norm_epsilon <= 0:
            raise ValueError("layer_norm_epsilon must be finite and positive")


class _LayerNorm(nn.Module):
    """Literal population-variance LayerNorm, including its second derivative."""

    def __init__(self, config, **factory):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.d_model, **factory))
        self.bias = nn.Parameter(torch.zeros(config.d_model, **factory))
        self.epsilon = config.layer_norm_epsilon

    def forward(self, x):
        centered = x - x.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1, keepdim=True)
        return centered * torch.rsqrt(variance + self.epsilon) * self.weight + self.bias


class _TransformerBlock(nn.Module):
    def __init__(self, config, **factory):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        # Registration order mirrors native checkpoint traversal: attention
        # norm, QKV, attention projection, MLP norm, expansion, contraction.
        self.attention_norm = _LayerNorm(config, **factory)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, **factory)
        self.attention_projection = nn.Linear(config.d_model, config.d_model, **factory)
        self.mlp_norm = _LayerNorm(config, **factory)
        self.mlp_input = nn.Linear(config.d_model, config.d_ff, **factory)
        self.mlp_output = nn.Linear(config.d_ff, config.d_model, **factory)

    def forward(self, x):
        batch, length, width = x.shape
        # Q/K/V occupy consecutive width-sized portions of the native QKV
        # projection. Heads partition each portion into contiguous features.
        q, k, v = self.qkv(self.attention_norm(x)).chunk(3, dim=-1)
        q, k, v = (
            t.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
            for t in (q, k, v)
        )
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        probabilities = scores.masked_fill(future, -torch.inf).softmax(dim=-1)
        attended = (probabilities @ v).transpose(1, 2).reshape(batch, length, width)
        x = x + self.attention_projection(attended)
        hidden = self.mlp_input(self.mlp_norm(x))
        # Native GELU uses this tanh approximation (not erf GELU).
        hidden = (
            0.5
            * hidden
            * (1 + torch.tanh(0.7978845608 * (hidden + 0.044715 * hidden.pow(3))))
        )
        return x + self.mlp_output(hidden)


class Gpt2(nn.Module):
    """Pre-LayerNorm GPT-2, tied token embedding/head, and no dropout.

    The physical embedding table includes native alignment padding. Padding
    rows are never valid input/target tokens or softmax classes. The head uses
    the same Parameter as the embedding lookup, so both gradient paths add to
    one trainable tensor rather than double counting it in influence vectors.
    """

    def __init__(self, config=GPT2Config(), seed=123, device=None, dtype=torch.float32):
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("DIT model requires float32 or float64 compute")
        self.config = config
        factory = dict(device=device, dtype=dtype)
        # Build and seed on CPU first. A local generator makes construction
        # reproducible without perturbing the caller's random stream; then
        # move to the requested device once initialization is complete.
        factory["device"] = "cpu"
        with torch.random.fork_rng(devices=[]):
            # manual_seed() also changes CUDA RNGs. Initialization is on CPU,
            # so seed only the CPU generator that fork_rng restores afterward.
            torch.random.default_generator.manual_seed(seed)
            self.token_embedding = nn.Embedding(
                config.padded_vocab_size, config.d_model, **factory
            )
            self.position_embedding = nn.Embedding(
                config.context_length, config.d_model, **factory
            )
            self.blocks = nn.ModuleList(
                _TransformerBlock(config, **factory) for _ in range(config.n_layers)
            )
            self.final_norm = _LayerNorm(config, **factory)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            with torch.no_grad():
                for name, parameter in self.named_parameters():
                    if name.endswith(".bias"):
                        parameter.zero_()
                    elif "norm.weight" in name:
                        parameter.fill_(1)
                    else:
                        residual = (
                            "attention_projection.weight" in name
                            or "mlp_output.weight" in name
                        )
                        stddev = (
                            0.02 / math.sqrt(2 * config.n_layers) if residual else 0.02
                        )
                        parameter.normal_(std=stddev, generator=generator)
        self.to(device=device, dtype=dtype)

    def forward(self, tokens):
        if tokens.dtype != torch.int64 or tokens.ndim != 2:
            raise ValueError("tokens must be int64 [batch, context] indices")
        batch, length = tokens.shape
        if batch <= 0 or not 1 <= length <= self.config.context_length:
            raise ValueError("nonempty batch and context within model limits required")
        if bool(((tokens < 0) | (tokens >= self.config.vocab_size)).any()):
            raise ValueError("input token is outside the logical vocabulary")
        x = self.token_embedding(tokens)
        positions = torch.arange(length, device=tokens.device)
        x = x + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return F.linear(
            self.final_norm(x), self.token_embedding.weight[: self.config.vocab_size]
        )

    def per_example_loss(self, examples):
        """Mean next-token CE over each row's context, never over batch size.

        Each [context+1] row is one independently removable training example;
        rows do not attend to each other. The outer training loop averages
        these values over its batch, keeping sample removal scaling explicit.
        """
        if examples.dtype != torch.int64 or examples.ndim != 2 or examples.shape[1] < 2:
            raise ValueError("examples must be int64 [batch, context+1] indices")
        if bool(((examples < 0) | (examples >= self.config.vocab_size)).any()):
            raise ValueError("example token is outside the logical vocabulary")
        logits = self(examples[:, :-1])
        losses = F.cross_entropy(
            logits.transpose(1, 2), examples[:, 1:], reduction="none"
        )
        return losses.mean(dim=1)

    def loss(self, examples):
        return self.per_example_loss(examples).mean()

    def pluto_weight_layout(self):
        """(name, Parameter, transpose) in unique native weight_N.bin order."""
        return [
            (
                name,
                parameter,
                name.endswith(".weight")
                and any(
                    f".{layer}." in f".{name}"
                    for layer in (
                        "qkv",
                        "attention_projection",
                        "mlp_input",
                        "mlp_output",
                    )
                ),
            )
            for name, parameter in self.named_parameters()
        ]

    def load_pluto_checkpoint(self, directory):
        """Import raw native FP32 weights, validating all files before mutation.

        Native fully-connected weights are [input, output], while nn.Linear
        stores [output, input]. Other tensors have identical layout. Native
        checkpoint directories do not encode architecture; callers must select
        the correct config. Extra or missing weight files are rejected.
        """
        directory = Path(directory)
        layout = self.pluto_weight_layout()
        expected = {f"weight_{index}.bin" for index in range(len(layout))}
        found = {path.name for path in directory.glob("weight_*.bin")}
        if found != expected:
            raise ValueError(
                f"checkpoint weight files mismatch: missing {sorted(expected-found)}, extra {sorted(found-expected)}"
            )
        loaded = []
        for index, (name, parameter, transpose) in enumerate(layout):
            path = directory / f"weight_{index}.bin"
            if path.stat().st_size != parameter.numel() * 4:
                raise ValueError(f"incorrect checkpoint size for {name}: {path}")
            raw = bytearray(path.read_bytes())
            if sys.byteorder != "little":
                raise ValueError(
                    "native checkpoint import requires a little-endian host"
                )
            value = torch.frombuffer(raw, dtype=torch.float32).clone()
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"nonfinite checkpoint weight: {path}")
            shape = tuple(reversed(parameter.shape)) if transpose else parameter.shape
            value = value.reshape(shape)
            loaded.append(value.t() if transpose else value)
        with torch.no_grad():
            for (_, parameter, _), value in zip(layout, loaded):
                parameter.copy_(value)

    def write_pluto_checkpoint(self, directory):
        """Export native-compatible FP32 weights to a new directory only.

        FP64 research weights are explicitly converted to native FP32 storage.
        This is a weights-only artifact, not a resumable DIT training trajectory.
        """
        directory = Path(directory)
        if sys.byteorder != "little":
            raise ValueError("native checkpoint export requires a little-endian host")
        directory.mkdir(parents=False, exist_ok=False)
        for index, (_, parameter, transpose) in enumerate(self.pluto_weight_layout()):
            value = parameter.detach().t() if transpose else parameter.detach()
            value = value.to(device="cpu", dtype=torch.float32).contiguous()
            (directory / f"weight_{index}.bin").write_bytes(value.numpy().tobytes())
