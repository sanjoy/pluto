"""Synthetic weight-change and Gram-spectrum checks; no pretrained model."""

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from . import checkpoint_delta_audit as audit


class SpectrumTest(unittest.TestCase):
    def test_known_singular_values_tall_wide_and_rotated(self):
        rng = np.random.default_rng(918)
        left, _ = np.linalg.qr(rng.normal(size=(9, 4)))
        right, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        expected = np.array([7., 3., .5, .1])
        matrix = left @ np.diag(expected) @ right.T
        for value in (matrix, matrix.T):
            result = audit.gram_spectrum(value)
            np.testing.assert_allclose(result['singular_values_sqrt_clipped_gram_eigenvalues'], expected, rtol=1e-11)
            self.assertEqual(result['gram_shape'], [4, 4])
            self.assertAlmostEqual(result['stable_rank'], np.sum(expected**2)/49)
            self.assertEqual(len(result['gram_eigenvalues_descending']), 4)

    def test_zero_matrix_does_not_invent_rank_or_energy(self):
        result = audit.gram_spectrum(np.zeros((7, 3)))
        self.assertEqual(result['singular_values_sqrt_clipped_gram_eigenvalues'], [0., 0., 0.])
        self.assertEqual(result['stable_rank'], 0)
        self.assertEqual(result['entropy_effective_rank'], 0)
        self.assertEqual(result['singular_values_above_diagnostic_floor'], 0)
        self.assertEqual(result['components_for_energy_fraction']['0.99'], 0)

    def test_tiny_true_singular_value_is_not_a_rank_guarantee(self):
        # The matrix really has rank two, but its second singular direction is
        # below the declared Gram precision floor. Do not call this exact rank1.
        result = audit.gram_spectrum(np.diag([1., 1e-10]))
        self.assertEqual(result['singular_values_above_diagnostic_floor'], 1)
        self.assertIn('NOT exact rank', result['rank_scope'])
        self.assertAlmostEqual(result['singular_values_sqrt_clipped_gram_eigenvalues'][1], 1e-10)

    def test_small_negative_gram_noise_retained_but_large_negative_rejected(self):
        with mock.patch.object(np.linalg, 'eigvalsh', return_value=np.array([-1e-16, 1.])):
            result = audit.gram_spectrum(np.eye(2))
        self.assertEqual(result['negative_gram_eigenvalue_count'], 1)
        self.assertEqual(result['gram_eigenvalues_descending'][-1], -1e-16)
        self.assertEqual(result['singular_values_sqrt_clipped_gram_eigenvalues'][-1], 0)
        with mock.patch.object(np.linalg, 'eigvalsh', return_value=np.array([-.1, 1.])):
            with self.assertRaises(ArithmeticError): audit.gram_spectrum(np.eye(2))


class MatrixTest(unittest.TestCase):
    def test_raw_difference_norms_exact_changes_and_row_ids(self):
        before = np.array([[1., 2.], [3., 4.], [0., 0.]], dtype='<f4')
        after = before + np.array([[0., 0.], [3., 4.], [0., 1.]], dtype='<f4')
        result = audit.audit_matrix(before, after, top_rows=2)
        self.assertEqual(result['maximum_absolute_delta'], 4)
        self.assertAlmostEqual(result['delta_frobenius_norm'], np.sqrt(26))
        self.assertAlmostEqual(result['delta_over_before_frobenius_norm'], np.sqrt(26/30))
        self.assertEqual(result['numerically_unchanged_elements'], 3)
        self.assertEqual(result['bitwise_unchanged_elements'], 3)
        self.assertEqual(result['exactly_unchanged_rows'], 1)
        self.assertEqual([x['row_id'] for x in result['top_rows_by_norm']], [1, 2])
        expected = np.array([0., 5., 1.], dtype='<f8')
        self.assertEqual(result['row_norms_sha256_float64_le'], hashlib.sha256(expected.tobytes()).hexdigest())
        self.assertIn('no decay correction', result['delta_definition'])

    def test_numeric_and_bitwise_zero_are_distinct_and_zero_norm_ratio_undefined(self):
        before = np.array([[0., -0.]], dtype='<f4')
        after = np.array([[-0., 0.]], dtype='<f4')
        result = audit.audit_matrix(before, after)
        self.assertEqual(result['numerically_unchanged_fraction'], 1)
        self.assertEqual(result['bitwise_unchanged_fraction'], 0)
        self.assertIsNone(result['delta_over_before_frobenius_norm'])
        self.assertTrue(result['relative_norm_zero_denominator'])

    def test_row_ties_use_ascending_ids(self):
        result = audit.audit_matrix(np.zeros((4, 2)), np.ones((4, 2)), top_rows=3)
        self.assertEqual([x['row_id'] for x in result['top_rows_by_norm']], [0, 1, 2])

    def test_progress_formats_undefined_zero_baseline_ratio(self):
        self.assertEqual(audit._format_relative_norm(None), 'undefined (zero baseline)')
        self.assertEqual(audit._format_relative_norm(.125), '0.125')

    def test_bad_shapes_values_and_options_rejected(self):
        for before, after in ((np.zeros((0, 2)), np.zeros((0, 2))),
                              (np.ones((2, 3)), np.ones((3, 2))),
                              (np.array([[np.nan]]), np.zeros((1, 1))),
                              (np.array([[1]]), np.array([[2]]))):
            with self.assertRaises(ValueError): audit.audit_matrix(before, after)
        with self.assertRaises(ValueError): audit.audit_matrix(np.eye(2), np.eye(2), top_rows=True)

    def test_selected_embedding_excludes_padding(self):
        class ToyCheckpoint:
            config = SimpleNamespace(n_layers=2, vocab_size=3)
            def __init__(self):
                self.table = np.zeros((5, 2), dtype='<f4')
                self.table[3:] = 1000
                names = ['token_embedding.weight', 'blocks.0.attn.qkv.weight',
                         'blocks.0.mlp.input.weight', 'blocks.0.mlp.output.weight',
                         'blocks.1.mlp.input.weight', 'blocks.1.mlp.output.weight']
                self.specs = {name: SimpleNamespace(filename=f'weight_{i}.bin',
                                                    shape=(5, 2) if i == 0 else (2, 2))
                              for i, name in enumerate(names)}
            @property
            def token_embedding(self): return self.table[:self.config.vocab_size]
            def __getitem__(self, name): return np.zeros((2, 2), dtype='<f4')
        selected = list(audit.selected_matrices(ToyCheckpoint()))
        self.assertEqual(len(selected), 6)
        _, embedding, address = selected[0]
        self.assertEqual(embedding.shape, (3, 2))
        self.assertTrue(np.all(embedding == 0))
        self.assertEqual(address['selected_byte_range'], [0, 24])
        self.assertTrue(address['embedding_padding_excluded'])
        self.assertEqual(audit.audit_matrix(embedding, embedding)['delta_frobenius_norm'], 0)

    def test_refuses_existing_output_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'audit.json'
            output.write_text('preserve me')
            with self.assertRaises(FileExistsError):
                audit.audit_checkpoints('/not/a/checkpoint', [1, 2], output)
            self.assertEqual(output.read_text(), 'preserve me')


if __name__ == '__main__':
    unittest.main()
