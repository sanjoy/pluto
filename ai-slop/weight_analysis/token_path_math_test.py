"""Synthetic tests for explicit sampled-token margins, positions and gates."""

import math
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from .checkpoint import GPT2Checkpoint, GPT2Config, tensor_manifest
from .phrase_reference import forward
from .phrase_reference_test import scalar_bf16
from .phrase_trace_analysis import margin_accounting, reconstructed_attention
from .token_path_math import neuron_operations, query_attention, target_ledger


class TokenPathMathTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        config = GPT2Config(vocab_size=7, padded_vocab_size=8, context_length=6,
                            n_layers=2, d_model=4, n_heads=2, d_ff=6)
        rng = np.random.default_rng(771)
        for spec in tensor_manifest(config):
            values = rng.normal(0, .2, spec.shape).astype('<f4')
            if spec.name.endswith('.scale'):
                values += 1
            values.tofile(Path(self.temp.name)/spec.filename)
        self.cp = GPT2Checkpoint(self.temp.name, config, check_finite=True)
        self.result = forward(self.cp, [1, 5, 2, 6], 'bf16')
        self.stages = self.result['stages']

    def test_each_row_matches_full_winner_ledger_including_nonzero_positions(self):
        full = margin_accounting(self.cp, self.stages, self.result['logits'])
        for row in range(4):
            result = target_ledger(self.cp, self.stages, row,
                                   int(full['winners'][row]), int(full['runners'][row]),
                                   self.result['logits'][row])
            for key, value in result['terms'].items():
                self.assertAlmostEqual(value, full['terms'][key][row], places=12)
            self.assertLess(abs(result['closure_error']), 1e-12)

    def test_nonwinning_target_and_swapped_margin_have_opposite_terms(self):
        row = 3
        order = np.argsort(self.result['logits'][row])
        a, b = int(order[0]), int(order[-1])
        first = target_ledger(self.cp, self.stages, row, a, b, self.result['logits'][row])
        second = target_ledger(self.cp, self.stages, row, b, a, self.result['logits'][row])
        self.assertLess(first['native_margin'], 0)
        for key in first['terms']:
            self.assertAlmostEqual(first['terms'][key], -second['terms'][key], places=12)
        self.assertAlmostEqual(math.fsum(first['terms'].values()), first['native_margin'], places=12)

    def test_query_only_attention_matches_full_causal_math_and_ignores_future(self):
        qkv = self.stages['blocks.0.qkv']
        probabilities, values, context = reconstructed_attention(qkv, 2)
        for row in range(len(qkv)):
            p, v, c = query_attention(qkv, 2, row)
            np.testing.assert_allclose(p, probabilities[:, row, :row+1], atol=1e-15)
            np.testing.assert_array_equal(v, values[:, :row+1])
            np.testing.assert_allclose(c, context[row], atol=1e-15)
            altered = qkv.copy(); altered[row+1:] = 999
            np.testing.assert_array_equal(query_attention(altered, 2, row)[0], p)

    def test_neuron_gates_key_products_and_byte_addresses(self):
        row, block, neuron = 2, 1, 3
        ledger = target_ledger(self.cp, self.stages, row, 0, 6, self.result['logits'][row])
        result = neuron_operations(self.cp, self.stages, row, block, neuron, ledger['direction'])
        p = 'blocks.1'
        scalar_products = [float(self.stages[p+'.ln2'][row, i]) *
                           scalar_bf16(self.cp[p+'.mlp.input.weight'][i, neuron]) for i in range(4)]
        np.testing.assert_array_equal(result['input_product_terms'], scalar_products)
        self.assertAlmostEqual(result['analytical_pre_gelu'], math.fsum(scalar_products) +
                               float(self.cp[p+'.mlp.input.bias'][neuron]), places=15)
        self.assertAlmostEqual(result['margin_contribution'], ledger['neurons'][p][neuron], places=15)
        self.assertEqual(result['input_weight_file'], 'weight_22.bin')
        self.assertEqual(result['input_column_first_byte'], 12)
        self.assertEqual(result['input_column_byte_stride'], 24)
        self.assertEqual(result['output_weight_file'], 'weight_24.bin')
        self.assertEqual(result['output_row_byte_offset'], 48)

    def test_selected_position_term_is_independent_of_position_zero(self):
        row = 3
        result = target_ledger(self.cp, self.stages, row, 0, 6,
                               self.result['logits'][row])
        selected = self.cp['position_embedding.weight'][row]
        expected = math.fsum(float(value)*float(direction)
                             for value, direction in zip(selected, result['direction']))
        wrong_position_zero = math.fsum(float(value)*float(direction) for value, direction in
                                        zip(self.cp['position_embedding.weight'][0], result['direction']))
        self.assertAlmostEqual(result['terms']['position_embedding'], expected, places=14)
        self.assertGreater(abs(expected-wrong_position_zero), 1e-5)

    def test_neuron_reported_addresses_read_the_exact_raw_master_weights(self):
        row, block, neuron = 1, 0, 4
        ledger = target_ledger(self.cp, self.stages, row, 0, 6,
                               self.result['logits'][row])
        result = neuron_operations(self.cp, self.stages, row, block, neuron,
                                    ledger['direction'])
        first = Path(self.temp.name)/result['input_weight_file']
        last = Path(self.temp.name)/result['output_weight_file']
        first_bytes, last_bytes = first.read_bytes(), last.read_bytes()
        for coordinate in range(self.cp.config.d_model):
            column_address = (result['input_column_first_byte'] +
                              coordinate*result['input_column_byte_stride'])
            row_address = result['output_row_byte_offset'] + 4*coordinate
            input_value = struct.unpack_from('<f', first_bytes, column_address)[0]
            output_value = struct.unpack_from('<f', last_bytes, row_address)[0]
            self.assertEqual(input_value,
                             float(self.cp['blocks.0.mlp.input.weight'][coordinate, neuron]))
            self.assertEqual(output_value,
                             float(self.cp['blocks.0.mlp.output.weight'][neuron, coordinate]))
            self.assertEqual(result['input_product_terms'][coordinate],
                             float(self.stages['blocks.0.ln2'][row, coordinate])*scalar_bf16(input_value))
        bias_bytes = (Path(self.temp.name)/result['input_bias_file']).read_bytes()
        self.assertEqual(result['input_bias'], struct.unpack_from(
            '<f', bias_bytes, result['input_bias_byte_offset'])[0])

    def test_neuron_operations_rejects_broadcastable_stage_shapes(self):
        direction = np.ones(self.cp.config.d_model)
        for name in ['blocks.0.ln2', 'blocks.0.fc1', 'blocks.0.gelu', 'positioned']:
            changed = dict(self.stages)
            changed[name] = np.zeros((4, 1), dtype=np.float32)
            with self.subTest(name=name), self.assertRaises(ValueError):
                neuron_operations(self.cp, changed, 1, 0, 2, direction)
        for direction in [np.ones(1), np.array([1, 2, 3, np.nan])]:
            with self.assertRaises(ValueError):
                neuron_operations(self.cp, self.stages, 1, 0, 2, direction)

    def test_analytical_gelu_uses_native_float32_constants_and_captured_input(self):
        stages = dict(self.stages)
        stages['blocks.0.fc1'] = self.stages['blocks.0.fc1'].copy()
        # Deliberately replace the captured pre-value: the diagnostic must
        # apply GELU to that observation, not its independently rebuilt dot.
        stages['blocks.0.fc1'][1, 2] = np.float32(3.1875)
        result = neuron_operations(self.cp, stages, 1, 0, 2,
                                    np.ones(self.cp.config.d_model))
        x = 3.1875
        c, a = float(np.float32(.7978845608)), float(np.float32(.044715))
        expected = .5*x*(1+math.tanh(c*(x+a*x**3)))
        decimal_constants = .5*x*(1+math.tanh(.7978845608*(x+.044715*x**3)))
        self.assertNotEqual(expected, decimal_constants)
        self.assertEqual(result['native_pre_gelu'], x)
        self.assertEqual(result['analytical_gelu_of_native_pre'], expected)
        self.assertEqual(result['native_post_minus_analytical'],
                         result['native_post_gelu']-expected)

    def test_invalid_targets_rows_and_geometry_fail(self):
        for row, target, other in [(-1, 0, 1), (4, 0, 1), (0, -1, 1), (0, 0, 7), (0, 2, 2), (0, True, 2)]:
            with self.assertRaises(ValueError):
                target_ledger(self.cp, self.stages, row, target, other, self.result['logits'][0])
        for heads, row in [(0, 0), (3, 0), (2, -1), (2, 4)]:
            with self.assertRaises(ValueError):
                query_attention(self.stages['blocks.0.qkv'], heads, row)


if __name__ == '__main__':
    unittest.main()
