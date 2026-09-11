"""Independent scalar softmax oracles for the static polynomial, on toys only.

The extractor must not execute attention softmax. These tests deliberately do:
two synthetic positions, scalar projection loops, and an actual two-way softmax
provide an independent reference for its zeroth and first Taylor coefficients.
No checkpoint files, training text, tokenizer, or real model are accessed.
"""

import copy
import math
import unittest

import numpy as np

from .checkpoint import GPT2Config, tensor_manifest
from .closed_trigram import RECIPE_EPSILON


class ToyCheckpoint:
    """Tiny, entirely in-memory implementation of the read-only weight API."""

    def __init__(self, heads=2, seed=271):
        self.config = GPT2Config(vocab_size=6, padded_vocab_size=8,
                                 context_length=3, n_layers=1,
                                 d_model=2 * heads, n_heads=heads, d_ff=4 * heads)
        random = np.random.default_rng(seed)
        self.tensors = {spec.name: random.normal(0, .35, spec.shape)
                        for spec in tensor_manifest(self.config)}
        self.tensors['blocks.0.ln1.scale'] += 1

    def __getitem__(self, name):
        return self.tensors[name]

    @property
    def token_embedding(self):
        return self.tensors['token_embedding.weight'][:self.config.vocab_size]

    def qkv(self, block, component, head=None):
        width = self.config.d_model
        start = 'qkv'.index(component) * width
        count = width
        if head is not None:
            start += head * self.config.head_dim
            count = self.config.head_dim
        return self.tensors[f'blocks.{block}.attn.qkv.weight'][:, start:start + count]

    def qkv_bias(self, block, component, head=None):
        width = self.config.d_model
        start = 'qkv'.index(component) * width
        count = width
        if head is not None:
            start += head * self.config.head_dim
            count = self.config.head_dim
        return self.tensors[f'blocks.{block}.attn.qkv.bias'][start:start + count]

    def attention_output_head(self, block, head):
        start = head * self.config.head_dim
        return self.tensors[f'blocks.{block}.attn.output.weight'][
            start:start + self.config.head_dim]


def scalar_input(checkpoint, token, position):
    """Population LayerNorm calculated independently with Python scalar sums."""
    row = [float(e) + float(p) for e, p in zip(
        checkpoint.token_embedding[token], checkpoint['position_embedding.weight'][position])]
    mean = math.fsum(row) / len(row)
    variance = math.fsum((value - mean) ** 2 for value in row) / len(row)
    denominator = math.sqrt(variance + RECIPE_EPSILON)
    return [(value - mean) / denominator * float(gamma) + float(beta)
            for value, gamma, beta in zip(row, checkpoint['blocks.0.ln1.scale'],
                                          checkpoint['blocks.0.ln1.bias'])]


def scalar_project(row, matrix, bias):
    """Row-vector projection without using the production vectorized products."""
    return [math.fsum(float(row[i]) * float(matrix[i, j])
                      for i in range(len(row))) + float(bias[j])
            for j in range(matrix.shape[1])]


def scalar_attention(checkpoint, previous, current, scale, pairing=None):
    """True two-position attention with BOTH logits multiplied by scale.

    Even keys/values that cancel algebraically are explicitly constructed with
    their biases here. Each head projects the softmax mixture through its own
    content output block; the global output bias is added once, not per head.
    A broken control moves value projection, value bias, and output together.
    """
    heads, width = checkpoint.config.n_heads, checkpoint.config.d_model
    channels = checkpoint.config.head_dim
    if pairing is None:
        pairing = list(range(heads))
    x0, x1 = scalar_input(checkpoint, previous, 0), scalar_input(checkpoint, current, 1)
    head_writes = []
    for routing, content in enumerate(pairing):
        query = scalar_project(x1, checkpoint.qkv(0, 'q', routing),
                               checkpoint.qkv_bias(0, 'q', routing))
        keys = [scalar_project(x, checkpoint.qkv(0, 'k', routing),
                               checkpoint.qkv_bias(0, 'k', routing)) for x in (x0, x1)]
        values = [scalar_project(x, checkpoint.qkv(0, 'v', content),
                                 checkpoint.qkv_bias(0, 'v', content)) for x in (x0, x1)]
        logits = [scale * math.fsum(q * k for q, k in zip(query, key)) /
                  math.sqrt(channels) for key in keys]
        largest = max(logits)
        exponentials = [math.exp(value - largest) for value in logits]
        denominator = math.fsum(exponentials)
        probabilities = [value / denominator for value in exponentials]
        mixed = [math.fsum(probabilities[p] * values[p][channel] for p in (0, 1))
                 for channel in range(channels)]
        head_writes.append(scalar_project(
            mixed, checkpoint.attention_output_head(0, content), np.zeros(width)))
    return np.array([math.fsum(head[j] for head in head_writes) +
                     float(checkpoint['blocks.0.attn.output.bias'][j])
                     for j in range(width)])


class LazyPolynomialOracleTest(unittest.TestCase):
    # These two adapters name only the public production API. The expected
    # results above never reuse its contractions or projected dictionaries.
    def probe(self, checkpoint, pairing=None, component='first_order'):
        from .lazy_trigram import LazyTrigramProbe
        return LazyTrigramProbe(checkpoint, np.arange(checkpoint.config.vocab_size),
                                component=component, pairing=pairing)

    def components(self, probe, previous, current):
        result = probe.components_for_pair(previous, current)
        return result['uniform'], result['derivative']

    def test_uniform_and_derivative_match_scalar_softmax_for_heads_and_biases(self):
        for heads in (1, 2, 3):
            for bias_mode in ('all', 'none', 'only_query', 'only_key_value'):
                checkpoint = ToyCheckpoint(heads=heads, seed=900 + heads)
                if bias_mode == 'none':
                    checkpoint['blocks.0.attn.qkv.bias'][:] = 0
                    checkpoint['blocks.0.attn.output.bias'][:] = 0
                elif bias_mode == 'only_query':
                    checkpoint.qkv_bias(0, 'k')[:] = 0
                    checkpoint.qkv_bias(0, 'v')[:] = 0
                    checkpoint['blocks.0.attn.output.bias'][:] = 0
                elif bias_mode == 'only_key_value':
                    checkpoint.qkv_bias(0, 'q')[:] = 0
                pairings = [list(range(heads))]
                if heads > 1:
                    pairings.append([(h + 1) % heads for h in range(heads)])
                for pairing in pairings:
                    probe = self.probe(checkpoint, pairing)
                    for previous, current in ((0, 0), (1, 4), (5, 2)):
                        with self.subTest(heads=heads, biases=bias_mode,
                                          pairing=pairing, pair=(previous, current)):
                            uniform, derivative = self.components(probe, previous, current)
                            expected_uniform = scalar_attention(checkpoint, previous, current,
                                                                0, pairing)
                            # For a centered difference, truncation is O(h^2)
                            # and scalar cancellation is O(machine_eps/h).
                            # h=1e-5 balances both for these O(1) toy tensors.
                            h = 1e-5
                            expected_derivative = (
                                scalar_attention(checkpoint, previous, current, h, pairing) -
                                scalar_attention(checkpoint, previous, current, -h, pairing)) / (2 * h)
                            np.testing.assert_allclose(uniform, expected_uniform,
                                                       atol=2e-12, rtol=2e-12)
                            np.testing.assert_allclose(derivative, expected_derivative,
                                                       atol=3e-9, rtol=3e-7)

    def test_first_order_residual_is_cubic_not_quadratic(self):
        checkpoint = ToyCheckpoint(heads=1)
        for tensor in checkpoint.tensors.values():
            tensor[:] = 0
        checkpoint['blocks.0.ln1.scale'][:] = 1
        checkpoint.token_embedding[0] = [1, -1]
        checkpoint.token_embedding[1] = [-1, 1]
        checkpoint.qkv(0, 'q')[0, 0] = 1
        checkpoint.qkv(0, 'k')[0, 0] = .5
        checkpoint.qkv(0, 'v')[0, 0] = 1
        checkpoint['blocks.0.attn.output.weight'][:] = np.eye(2)
        checkpoint.qkv_bias(0, 'v')[:] = [.3, -.4]
        checkpoint['blocks.0.attn.output.bias'][:] = [.1, .2]
        uniform, derivative = self.components(self.probe(checkpoint), 0, 1)
        errors = []
        for scale in (.2, .1, .05):
            positive = scalar_attention(checkpoint, 0, 1, scale)
            negative = scalar_attention(checkpoint, 0, 1, -scale)
            np.testing.assert_allclose((positive + negative) / 2, uniform,
                                       atol=2e-15, rtol=2e-15)
            errors.append(np.linalg.norm(positive - (uniform + scale * derivative)))
        self.assertGreater(min(errors), 1e-8)
        # Halving t reduces a nonzero cubic remainder by approximately eight.
        for ratio in (errors[0] / errors[1], errors[1] / errors[2]):
            self.assertGreater(ratio, 7.8)
            self.assertLess(ratio, 8.2)

    def test_complete_content_pairing_leaves_uniform_term_unchanged(self):
        checkpoint = ToyCheckpoint(heads=3, seed=313)
        intact = self.probe(checkpoint)
        pairing = [1, 2, 0]
        broken = self.probe(checkpoint, pairing)
        for previous, current in ((0, 1), (2, 2), (4, 3)):
            base, derivative = self.components(intact, previous, current)
            broken_base, broken_derivative = self.components(broken, previous, current)
            np.testing.assert_allclose(base, broken_base, atol=2e-12, rtol=2e-12)
            np.testing.assert_allclose(broken_base,
                                       scalar_attention(checkpoint, previous, current, 0, pairing),
                                       atol=2e-12, rtol=2e-12)
            self.assertGreater(np.linalg.norm(derivative - broken_derivative), 1e-5)

    def test_key_value_output_biases_do_not_change_derivative(self):
        checkpoint = ToyCheckpoint(heads=2, seed=144)
        altered = copy.deepcopy(checkpoint)
        altered.qkv_bias(0, 'k')[:] += [.4, -.8, 1.7, -.3]
        altered.qkv_bias(0, 'v')[:] += [1.3, .7, -2.2, .9]
        altered['blocks.0.attn.output.bias'][:] += [.1, .5, -.2, .3]
        original_probe, altered_probe = self.probe(checkpoint), self.probe(altered)
        for previous, current in ((1, 3), (5, 5)):
            uniform, derivative = self.components(original_probe, previous, current)
            new_uniform, new_derivative = self.components(altered_probe, previous, current)
            np.testing.assert_allclose(new_derivative, derivative, atol=2e-12, rtol=2e-12)
            self.assertGreater(np.linalg.norm(new_uniform - uniform), .1)
            np.testing.assert_allclose(new_uniform,
                                       scalar_attention(altered, previous, current, 0),
                                       atol=2e-12, rtol=2e-12)
        query_changed = copy.deepcopy(checkpoint)
        query_changed.qkv_bias(0, 'q')[:] += [.6, -.5, 1.1, .4]
        _, changed_derivative = self.components(self.probe(query_changed), 1, 3)
        _, original_derivative = self.components(original_probe, 1, 3)
        self.assertGreater(np.linalg.norm(changed_derivative - original_derivative), 1e-4)

    def test_centered_raw_destination_order_and_compensated_translation(self):
        checkpoint = ToyCheckpoint(heads=2, seed=454)
        translated = copy.deepcopy(checkpoint)
        shift = np.array([.2, -.7, 1.1, .9])
        translated['token_embedding.weight'][:] += shift
        translated['position_embedding.weight'][:] -= shift
        original_probe, shifted_probe = self.probe(checkpoint), self.probe(translated)
        original_mean = checkpoint.token_embedding.mean(axis=0)
        shifted_mean = translated.token_embedding.mean(axis=0)
        for previous, current in ((0, 2), (3, 3)):
            base, derivative = self.components(original_probe, previous, current)
            shifted_base, shifted_derivative = self.components(shifted_probe, previous, current)
            for component, write, shifted_write in (
                    ('uniform', base, shifted_base),
                    ('derivative', derivative, shifted_derivative),
                    ('first_order', base + derivative, shifted_base + shifted_derivative)):
                np.testing.assert_allclose(write, shifted_write, atol=2e-12, rtol=2e-12)
                raw = checkpoint.token_embedding @ write
                centered = (checkpoint.token_embedding - original_mean) @ write / np.linalg.norm(write)
                shifted = (translated.token_embedding - shifted_mean) @ shifted_write / np.linalg.norm(shifted_write)
                np.testing.assert_allclose(centered, shifted, atol=2e-12, rtol=2e-12)
                self.assertEqual(np.argsort(-raw).tolist(), np.argsort(-centered).tolist())
                edges = self.probe(checkpoint, component=component).edges(previous, current)
                shifted_edges = self.probe(translated, component=component).edges(previous, current)
                expected = np.argsort(-raw)[:4].tolist()
                self.assertEqual([edge['token_ids'][2] for edge in edges], expected)
                self.assertEqual([edge['token_ids'][2] for edge in shifted_edges], expected)
                np.testing.assert_allclose([edge['score'] for edge in edges], centered[expected],
                                           atol=2e-12, rtol=2e-12)
                np.testing.assert_allclose([edge['score'] for edge in shifted_edges], centered[expected],
                                           atol=2e-12, rtol=2e-12)


if __name__ == '__main__':
    unittest.main()
