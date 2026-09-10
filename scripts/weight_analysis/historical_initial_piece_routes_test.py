"""Small independent numerical fixtures, not substitutes for native measurements."""

import math
import unittest

import numpy as np

from .historical_initial_piece_routes import (
    decode_bf16, fixed_direction, indentation_groups, rounded_bf16, source_route)


class InitialPieceRoutesTest(unittest.TestCase):
    def test_bf16_decode_signed_zero_and_round_to_nearest_even(self):
        words = np.array([0x0000, 0x8000, 0x3f80, 0xc000], dtype='<u2')
        values = decode_bf16(words.tobytes(), (2, 2))
        np.testing.assert_array_equal(values, [[0., -0.], [1., -2.]])
        self.assertTrue(np.signbit(values[0, 1]))
        bits = np.array([0x3f808000, 0x3f818000], dtype=np.uint32)
        np.testing.assert_array_equal(rounded_bf16(bits.view(np.float32)), [1., 1.015625])
        with self.assertRaises(ValueError):
            decode_bf16(words.tobytes(), (3, 2))
        with self.assertRaises(ValueError):
            decode_bf16(np.array([0x7f80], dtype='<u2').tobytes(), (1,))

    def test_direction_preserves_fp32_gamma_and_centers(self):
        gamma = np.array([1.001, 2.003], dtype=np.float32)
        direction, std = fixed_direction([[1., 0.], [0., 1.]], gamma, [1., -1.])
        expected = np.array([float(gamma[0]), -float(gamma[1])]) / math.sqrt(1 + 1e-5)
        expected -= expected.mean()
        np.testing.assert_array_equal(direction, expected)
        self.assertEqual(std, math.sqrt(1 + 1e-5))
        self.assertEqual(float(direction.sum()), 0.)
        with self.assertRaises(ValueError):
            fixed_direction([[1., 0.]], gamma, [1., -1.])

    def test_uniform_attention_and_scalar_source_contractions(self):
        # Two-dimensional head; Q/K=0 makes both probabilities exactly 1/2.
        qkv = np.array([[0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 0, 4]], dtype=float)
        result = source_route(qkv, [1., 2.], np.eye(2), [1., -1.],
            head=0, n_heads=1, row=1, groups={'a': (0, 1), 'b': (1, 2)})
        self.assertEqual(result['reconstructed_attention_probabilities'], [.5, .5])
        self.assertEqual(result['reconstructed_fixed_margin_source_terms'], [1., -2.])
        self.assertEqual(result['native_context_fixed_margin_term'], -1.)
        self.assertEqual(result['reconstructed_context_max_abs_error'], 0.)
        self.assertEqual(result['groups']['a']['partial_context_norm'], 1.)
        self.assertEqual(result['groups']['b']['partial_context_norm'], 2.)

    def test_query_scale_causal_truncation_and_head_selection(self):
        qkv = np.zeros((3, 12))  # Width4, two heads, dimension2.
        qkv[1, 2:4] = [math.sqrt(2), 0]  # Head1 query at row1.
        qkv[0, 6:8] = [0, 0]
        qkv[1, 6:8] = [math.log(3), 0]
        qkv[2, 6:8] = [1e6, 0]  # Future key MUST NOT participate.
        qkv[0, 10:12] = [2, 0]
        qkv[1, 10:12] = [0, 4]
        result = source_route(qkv, [0, 0, .5, 3], np.eye(4), [0, 0, 1, -1],
            head=1, n_heads=2, row=1, groups={'past': (0, 1), 'self': (1, 2)})
        np.testing.assert_allclose(result['reconstructed_attention_probabilities'], [.25, .75], atol=1e-15)
        np.testing.assert_allclose(result['reconstructed_fixed_margin_source_terms'], [.5, -3.], atol=1e-15)
        self.assertLess(result['reconstructed_context_max_abs_error'], 1e-14)
        self.assertEqual(len(result['scaled_qk_scores']), 2)

    def test_group_norm_ratio_can_exceed_one_without_being_a_probability(self):
        qkv = np.array([[0, 0, 0, 0, 4, 0], [0, 0, 0, 0, -2, 0]], dtype=float)
        result = source_route(qkv, [1, 0], np.eye(2), [1, 0], head=0,
            n_heads=1, row=1, groups={'positive': (0, 1), 'negative': (1, 2)})
        self.assertEqual(result['groups']['positive']['ratio_to_native_context_norm'], 2.)
        self.assertEqual(result['groups']['positive']['reconstructed_attention_mass'], .5)

    def test_native_rounding_difference_is_retained(self):
        qkv = np.zeros((2, 6))
        qkv[:, 4:] = [1., 2.]
        result = source_route(qkv, [1.25, 2], np.eye(2), [1, 0], head=0,
            n_heads=1, row=1, groups={'all': (0, 2)})
        self.assertEqual(result['reconstructed_context_max_abs_error'], .25)
        self.assertEqual(result['reconstructed_source_sum_minus_native_term'], -.25)

    def test_exact_indentation_boundaries_and_token_bytes(self):
        ids = [1]*1024
        ids[944] = ids[960] = 198
        ids[961:] = [220]*63
        token_bytes = {1: b'x', 198: b'\n', 220: b' '}
        self.assertEqual(indentation_groups(ids, token_bytes)['current_indent'], (961, 1024))
        for position in (944, 960, 970, 1023):
            bad = ids.copy()
            bad[position] = 1
            with self.assertRaises(ValueError):
                indentation_groups(bad, token_bytes)
        bad = ids.copy()
        bad[950] = 198
        with self.assertRaises(ValueError):
            indentation_groups(bad, token_bytes)

    def test_invalid_geometry_and_nonpartitioning_groups(self):
        qkv = np.zeros((2, 6))
        for groups in ({'missing': (0, 1)}, {'a': (0, 2), 'b': (1, 2)},
                       {'out': (0, 3)}, {'empty': (0, 0)}):
            with self.assertRaises(ValueError):
                source_route(qkv, [0, 0], np.eye(2), [1, 0], head=0,
                             n_heads=1, row=1, groups=groups)
        for head in (-1, 1, True):
            with self.assertRaises(ValueError):
                source_route(qkv, [0, 0], np.eye(2), [1, 0], head=head,
                             n_heads=1, row=1, groups={'all': (0, 2)})
        with self.assertRaises(ValueError):
            source_route(qkv, [0, np.nan], np.eye(2), [1, 0], head=0,
                         n_heads=1, row=1, groups={'all': (0, 2)})


if __name__ == '__main__':
    unittest.main()
