"""Synthetic-only oracles for the short-prefix transformer diagnostic.

The scalar implementation intentionally does not call the forward, LayerNorm,
GELU, attention, or BF16 helpers under test. No real checkpoint, tokenizer,
corpus, CUDA device, or model inference binary is used by these tests.
"""

import math
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from .checkpoint import GPT2Checkpoint, GPT2Config, tensor_manifest
from . import phrase_reference as reference


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def scalar_bf16(value):
    """Independent tie handling: compare discarded bits, then increment."""
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    retained, discarded = bits >> 16, bits & 65535
    if discarded > 32768 or (discarded == 32768 and retained % 2):
        retained += 1
    return struct.unpack('<f', struct.pack('<I', retained << 16))[0]


def scalar_forward(checkpoint, tokens, mode):
    """Obvious nested loops, including all heads, biases, and two residuals."""
    config = checkpoint.config
    store = scalar_bf16 if mode == 'bf16' else f32
    w = {spec.name: checkpoint[spec.name].tolist()
         for spec in tensor_manifest(config)}
    stages = {}

    def norm(rows, gamma, beta):
        result = []
        for row in rows:
            mean = f32(math.fsum(row) / len(row))
            centered = [f32(x - mean) for x in row]
            variance = f32(math.fsum(f32(x*x) for x in centered) / len(row))
            inv = f32(1 / math.sqrt(f32(variance + f32(1e-5))))
            result.append([store(f32(f32(f32(x*inv)*g) + b))
                           for x, g, b in zip(centered, gamma, beta)])
        return result

    def dense(rows, matrix, bias):
        # Double-precision fsum gives an independent accumulation order. The
        # comparison allows FP32 roundoff, not changed equations or data types.
        return [[store(f32(math.fsum(
            store(x) * store(matrix[i][j]) for i, x in enumerate(row)) + bias[j]))
                 for j in range(len(bias))] for row in rows]

    def add(left, right):
        return [[store(f32(a+b)) for a, b in zip(x, y)]
                for x, y in zip(left, right)]

    x = [[store(value) for value in w['token_embedding.weight'][token]]
         for token in tokens]
    stages['embedding'] = x
    x = add(x, w['position_embedding.weight'][:len(tokens)])
    stages['positioned'] = x
    attention = []
    for block in range(config.n_layers):
        p = f'blocks.{block}'
        n = norm(x, w[p+'.ln1.scale'], w[p+'.ln1.bias'])
        stages[p+'.ln1'] = n
        qkv = dense(n, w[p+'.attn.qkv.weight'], w[p+'.attn.qkv.bias'])
        stages[p+'.qkv'] = qkv
        context = [[0.0]*config.d_model for _ in tokens]
        probabilities = [[[0.0]*len(tokens) for _ in tokens]
                         for _ in range(config.n_heads)]
        scale = f32(1 / math.sqrt(config.head_dim))
        for head in range(config.n_heads):
            offset = head * config.head_dim
            for query in range(len(tokens)):
                scores = []
                for key in range(query+1):
                    scores.append(f32(scale * f32(math.fsum(
                        qkv[query][offset+d] *
                        qkv[key][config.d_model+offset+d]
                        for d in range(config.head_dim)))))
                maximum = max(scores)
                exps = [math.exp(score-maximum) for score in scores]
                denom = math.fsum(exps)
                probs = [e/denom for e in exps]
                probabilities[head][query][:query+1] = probs
                for d in range(config.head_dim):
                    context[query][offset+d] = store(f32(math.fsum(
                        probs[key] * qkv[key][2*config.d_model+offset+d]
                        for key in range(query+1))))
        stages[p+'.attention'] = context
        attention.append(np.asarray(probabilities))
        branch = dense(context, w[p+'.attn.output.weight'],
                       w[p+'.attn.output.bias'])
        stages[p+'.attention_projected'] = branch
        x = add(x, branch)
        stages[p+'.after_attention'] = x
        n = norm(x, w[p+'.ln2.scale'], w[p+'.ln2.bias'])
        stages[p+'.ln2'] = n
        fc1 = dense(n, w[p+'.mlp.input.weight'], w[p+'.mlp.input.bias'])
        stages[p+'.fc1'] = fc1
        features = []
        for row in fc1:
            feature_row = []
            for value in row:
                cube_term = f32(f32(f32(f32(.044715)*value)*value)*value)
                inner = f32(f32(.7978845608) * f32(value+cube_term))
                feature_row.append(store(f32(f32(.5*value) *
                                            f32(1+f32(math.tanh(inner))))))
            features.append(feature_row)
        stages[p+'.gelu'] = features
        branch = dense(features, w[p+'.mlp.output.weight'],
                       w[p+'.mlp.output.bias'])
        stages[p+'.mlp_projected'] = branch
        x = add(x, branch)
        stages[p+'.after_mlp'] = x
    final = norm(x, w['final_norm.scale'], w['final_norm.bias'])
    stages['final_norm'] = final
    logits = [[f32(math.fsum(store(x)*store(y) for x, y in
                            zip(row, w['token_embedding.weight'][token])))
               for token in range(config.vocab_size)] for row in final]
    return stages, attention, np.asarray(logits)


class PhraseReferenceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = GPT2Config(vocab_size=9, padded_vocab_size=12,
                                 context_length=8, n_layers=2, d_model=4,
                                 n_heads=2, d_ff=6)
        self.rng = np.random.default_rng(198)
        self.weights = {}
        for spec in tensor_manifest(self.config):
            values = self.rng.normal(0, .15, spec.shape).astype('<f4')
            if spec.name.endswith('.scale'):
                values += 1
            self.weights[spec.name] = values
        self.checkpoint = self.write_checkpoint()

    def write_checkpoint(self):
        for spec in tensor_manifest(self.config):
            self.weights[spec.name].astype('<f4').tofile(
                self.directory / spec.filename)
        return GPT2Checkpoint(self.directory, self.config, check_finite=True)

    def test_bf16_golden_ties_subnormals_and_signed_zero(self):
        inputs = np.array([0, 0x80000000, 0x3f808000, 0x3f818000,
                           0xbf808000, 0xbf818000, 0x00008000,
                           0x00018000, 0x80008000], dtype=np.uint32)
        expected = np.array([0, 0x80000000, 0x3f800000, 0x3f820000,
                             0xbf800000, 0xbf820000, 0x00000000,
                             0x00020000, 0x80000000], dtype=np.uint32)
        before = inputs.copy()
        actual = reference.bf16(inputs.view(np.float32))
        np.testing.assert_array_equal(actual.view(np.uint32), expected)
        np.testing.assert_array_equal(inputs, before)
        self.assertEqual(reference.bf16(np.float32(1.1)).shape, ())
        np.testing.assert_array_equal(reference.bf16(actual).view(np.uint32), expected)

    def test_bf16_independent_scalar_oracle(self):
        values = self.rng.normal(size=(17, 5)).astype(np.float32)
        expected = np.array([scalar_bf16(float(x)) for x in values.flat],
                            dtype=np.float32).reshape(values.shape)
        np.testing.assert_array_equal(reference.bf16(values), expected)
        np.testing.assert_array_equal(reference.bf16(values[:, ::2]),
                                      expected[:, ::2])

    def test_bf16_rejects_nonfinite_overflow_and_nonreal(self):
        for value in [np.inf, -np.inf, np.nan, np.finfo(np.float32).max,
                      [complex(1, 2)], ['1'], [True]]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                reference.bf16(value)

    def test_layer_norm_population_variance_affine_and_rounding(self):
        x = np.array([[1, 2, 3, 4], [7, 7, 7, 7]], dtype=np.float32)
        gamma = np.array([.7, 1.003, -2, .1], dtype=np.float32)
        beta = np.array([.0013, .17, -.31, 2.001], dtype=np.float32)
        expected = np.array([[(float(value)-2.5) / math.sqrt(1.25+f32(1e-5)) * g+b
                              for value, g, b in zip(x[0], gamma, beta)], beta],
                             dtype=np.float32)
        np.testing.assert_allclose(reference.layer_norm(x, gamma, beta, 'fp32'),
                                   expected, rtol=2e-7, atol=2e-7)
        actual = reference.layer_norm(x, gamma, beta, 'bf16')
        np.testing.assert_array_equal(actual, reference.bf16(expected))
        # Gamma and beta are not rounded independently to BF16.
        wrong = reference.layer_norm(x, reference.bf16(gamma),
                                     reference.bf16(beta), 'bf16')
        self.assertFalse(np.array_equal(actual, wrong))

    def test_gelu_tanh_formula_and_output_rounding(self):
        x = np.array([-5, -2.13, -.07, 0, .19, 1.71, 6], dtype=np.float32)
        expected = np.array([.5*float(v)*(1+math.tanh(
            f32(.7978845608)*(float(v)+f32(.044715)*float(v)**3)))
                             for v in x], dtype=np.float32)
        np.testing.assert_allclose(reference.gelu(x, 'fp32'), expected,
                                   atol=1e-7, rtol=1e-6)
        np.testing.assert_allclose(reference.gelu(x, 'bf16'),
                                   reference.bf16(expected), atol=1e-7)
        self.assertEqual(float(reference.gelu(np.float32(0))), 0)
        with self.assertRaises(ValueError):
            reference.gelu([np.nan])

    def test_full_forward_against_independent_scalar_oracle(self):
        tokens = [2, 5, 1, 7]
        for mode in ['fp32', 'bf16']:
            actual = reference.forward(self.checkpoint, tokens, mode)
            stages, attention, logits = scalar_forward(self.checkpoint, tokens, mode)
            self.assertEqual(set(actual['stages']), set(stages))
            for name, expected in stages.items():
                with self.subTest(mode=mode, stage=name):
                    np.testing.assert_allclose(actual['stages'][name], expected,
                                               rtol=3e-6, atol=4e-7)
            for block, probs in enumerate(attention):
                np.testing.assert_allclose(actual['attention'][block]['probabilities'],
                                           probs, rtol=3e-6, atol=4e-7)
            np.testing.assert_allclose(actual['logits'], logits,
                                       rtol=3e-6, atol=4e-7)

    def test_future_token_invariance_and_prefix_equivalence(self):
        for mode in ['fp32', 'bf16']:
            full = reference.forward(self.checkpoint, [1, 2, 3, 4, 5], mode)
            changed = reference.forward(self.checkpoint, [1, 2, 3, 8, 0], mode)
            short = reference.forward(self.checkpoint, [1, 2, 3], mode)
            for other in [changed, short]:
                for name, value in full['stages'].items():
                    np.testing.assert_allclose(value[:3], other['stages'][name][:3],
                                               atol=1e-6, rtol=1e-6)
                np.testing.assert_allclose(full['logits'][:3], other['logits'][:3],
                                           atol=1e-6, rtol=1e-6)

    def test_causal_probabilities_and_head_contribution_accounting(self):
        for mode in ['fp32', 'bf16']:
            result = reference.forward(self.checkpoint, [0, 1, 0, 1], mode)
            for block, record in enumerate(result['attention']):
                probabilities = record['probabilities']
                self.assertEqual(probabilities.shape, (2, 4, 4))
                self.assertEqual(record['context'].shape, (4, 2, 2))
                self.assertEqual(record['head_contributions'].shape, (2, 4, 4))
                np.testing.assert_array_equal(np.triu(probabilities, 1), 0)
                self.assertTrue(np.all(probabilities >= 0))
                np.testing.assert_allclose(probabilities.sum(axis=2), 1,
                                           atol=2e-7)
                np.testing.assert_allclose(
                    record['head_contributions'].sum(axis=0)+record['output_bias'],
                    record['projection_before_rounding'], atol=1e-7, rtol=1e-6)
                projected = record['projection_before_rounding']
                if mode == 'bf16':
                    projected = reference.bf16(projected)
                np.testing.assert_array_equal(projected, result['stages'][
                    f'blocks.{block}.attention_projected'])

    def test_zero_checkpoint_uniform_attention_tied_logits(self):
        for values in self.weights.values():
            values.fill(0)
        checkpoint = self.write_checkpoint()
        for mode in ['fp32', 'bf16']:
            result = reference.forward(checkpoint, [1, 2, 3, 4], mode)
            np.testing.assert_array_equal(result['logits'], np.zeros((4, 9)))
            # np.argmax supplies the smallest-ID tie behavior; forward itself
            # returns all logical logits and does not sample or pick a token.
            np.testing.assert_array_equal(np.argmax(result['logits'], axis=1), 0)
            for record in result['attention']:
                for row in range(4):
                    np.testing.assert_allclose(record['probabilities'][:, row, :row+1],
                                               1/(row+1), atol=1e-7)

    def test_all_stored_bf16_stages_and_unrounded_logits(self):
        result = reference.forward(self.checkpoint, [1, 5, 7], 'bf16')
        for name, values in result['stages'].items():
            self.assertEqual(values.dtype, np.float32, name)
            self.assertTrue(np.isfinite(values).all(), name)
            np.testing.assert_array_equal(values.view(np.uint32) & 65535, 0)
        self.assertEqual(result['logits'].dtype, np.float32)
        self.assertTrue(np.any(result['logits'].view(np.uint32) & 65535))
        fp32 = reference.forward(self.checkpoint, [1, 5, 7], 'fp32')
        self.assertFalse(np.array_equal(fp32['logits'], result['logits']))

    def test_position_parameters_not_rounded_before_addition(self):
        # Half-ulp positional details matter: BF16(1 + p) != BF16(1 + BF16(p)).
        self.weights['token_embedding.weight'][1] = 1
        self.weights['position_embedding.weight'][0] = np.float32(.01169)
        checkpoint = self.write_checkpoint()
        result = reference.forward(checkpoint, [1])
        expected = reference.bf16(np.ones((1, 4), dtype=np.float32) +
                                  self.weights['position_embedding.weight'][:1])
        wrong = reference.bf16(np.ones((1, 4), dtype=np.float32) +
                               reference.bf16(self.weights['position_embedding.weight'][:1]))
        np.testing.assert_array_equal(result['stages']['positioned'], expected)
        self.assertFalse(np.array_equal(expected, wrong))

    def test_bf16_matrix_rounding_fp32_bias_and_tied_head(self):
        result = reference.forward(self.checkpoint, [1, 5, 7])
        stages = result['stages']
        matrix = reference.bf16(self.weights['blocks.0.attn.qkv.weight'])
        bias = self.weights['blocks.0.attn.qkv.bias']
        expected = reference.bf16(stages['blocks.0.ln1'] @ matrix + bias)
        np.testing.assert_array_equal(expected, stages['blocks.0.qkv'])
        wrong = reference.bf16(stages['blocks.0.ln1'] @ matrix + reference.bf16(bias))
        self.assertFalse(np.array_equal(expected, wrong))
        expected_logits = (stages['final_norm'] @ reference.bf16(
            self.weights['token_embedding.weight'][:self.config.vocab_size]).T)
        np.testing.assert_array_equal(result['logits'], expected_logits)
        self.assertEqual(result['logits'].shape[1], self.config.vocab_size)

    def test_valid_lengths_input_unchanged_and_exact_token_identity(self):
        tokens = np.array([3], dtype=np.int32)
        result = reference.forward(self.checkpoint, tokens)
        np.testing.assert_array_equal(tokens, [3])
        np.testing.assert_array_equal(result['tokens'], [3])
        tokens[0] = 4
        np.testing.assert_array_equal(result['tokens'], [3])
        self.assertEqual(result['logits'].shape, (1, 9))
        result = reference.forward(self.checkpoint, [0]*self.config.context_length)
        self.assertEqual(result['logits'].shape, (8, 9))
        self.assertFalse(self.checkpoint['token_embedding.weight'].flags.writeable)

    def test_invalid_tokens_modes_shapes_and_parameters(self):
        invalid = [[], [True], [1.0], ['1'], [[1]], -1, [9], [-1],
                   [0]*9, np.array([2**64-1], dtype=np.uint64)]
        for tokens in invalid:
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                reference.forward(self.checkpoint, tokens)
        for mode in ['', 'FP32', 'fp16', None]:
            with self.assertRaises(ValueError):
                reference.forward(self.checkpoint, [1], mode)
        for x, gamma, beta in [([], [], []), ([1, 2], [1, 1], [0, 0]),
                               ([[1, 2]], [1], [0, 0]),
                               ([[1, 2]], [1, 1], [[0, 0]]),
                               ([[np.nan, 2]], [1, 1], [0, 0])]:
            with self.assertRaises(ValueError):
                reference.layer_norm(x, gamma, beta)
        self.weights['blocks.1.ln1.scale'][0] = np.nan
        for spec in tensor_manifest(self.config):
            self.weights[spec.name].tofile(self.directory / spec.filename)
        checkpoint = GPT2Checkpoint(self.directory, self.config)
        with self.assertRaisesRegex(ValueError, 'finite'):
            reference.forward(checkpoint, [1])


if __name__ == '__main__':
    unittest.main()
