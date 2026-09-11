"""Small CPU fixtures for the *operand* diagnostic, not model predictions."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import checkpoint, paired_compute_weight_diff as diff


class ComputeWeightDiffTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.a, self.b = self.root / 'a', self.root / 'b'
        self.config = checkpoint.GPT2Config(vocab_size=3, padded_vocab_size=4,
            context_length=2, n_layers=1, d_model=2, n_heads=1, d_ff=4)
        self.specs = checkpoint.tensor_manifest(self.config)
        for directory in (self.a, self.b):
            directory.mkdir()
            for spec in self.specs:
                np.ones(spec.shape, dtype='<f4').tofile(directory / spec.filename)

    def change(self, name, value, index=0):
        spec = next(spec for spec in self.specs if spec.name == name)
        array = np.memmap(self.b / spec.filename, mode='r+', dtype='<f4', shape=spec.shape)
        array.flat[index] = value
        array.flush()

    def compare(self, **kwargs):
        return diff.compare(self.a, self.b, config=self.config, **kwargs)

    def test_identical_and_tied_dictionary_counted_once(self):
        result = self.compare()
        self.assertEqual(result['totals']['stored_bits_changed'], 0)
        self.assertIsNone(result['totals']['hidden_fraction_of_stored_bit_changes'])
        self.assertEqual(result['totals']['elements'], sum(spec.nbytes // 4 for spec in self.specs) - 2)
        self.assertEqual(result['vocabulary_padding']['elements'], 2)

    def test_matrix_changes_can_be_hidden_but_fp32_position_bias_norm_cannot(self):
        for spec in self.specs:
            self.change(spec.name, 1.0001)
        result = self.compare(chunk_elements=3)
        for spec, tensor in zip(self.specs, result['tensors']):
            expected_cast = spec.name == 'token_embedding.weight' or '.weight' in spec.name and spec.name.startswith('blocks.')
            self.assertEqual(tensor['stored_bits_changed'], 1)
            self.assertEqual(tensor['operand_bits_changed'], 0 if expected_cast else 1)
            self.assertEqual(tensor['stored_changes_hidden_by_cast'], int(expected_cast))

    def test_effective_change_can_exceed_master_change_at_rounding_boundary(self):
        spec = self.specs[0]
        a = np.ones(spec.shape, dtype='<f4')
        a.flat[0] = 1.0039
        a.tofile(self.a / spec.filename)
        self.change(spec.name, 1.004)
        result = self.compare(selected_rows=[0])['selected_embedding_rows']
        self.assertEqual(result['operand_bits_changed'], 1)
        self.assertGreater(result['operand_delta_l2'], result['stored_delta_l2'])

    def test_padding_excluded_and_selected_rows_nonadditive(self):
        self.change('token_embedding.weight', 2., index=0)
        self.change('token_embedding.weight', 3., index=6)
        result = self.compare(selected_rows=[0], chunk_elements=5)
        self.assertEqual(result['totals']['operand_bits_changed'], 1)
        self.assertEqual(result['vocabulary_padding']['operand_bits_changed'], 1)
        self.assertEqual(result['selected_embedding_rows']['operand_bits_changed'], 1)
        self.assertEqual(result['left']['weights_sha256']['weight_0.bin'],
                         checkpoint.sha256_file(self.a / 'weight_0.bin'))

    def test_chunk_size_does_not_change_counts(self):
        for spec in self.specs:
            self.change(spec.name, .015625)
        self.assertEqual(self.compare(chunk_elements=1), self.compare(chunk_elements=1000))

    def test_signed_zero_and_subnormals_are_explicit(self):
        left = np.array([0., np.nextafter(np.float32(0), np.float32(1))], dtype='<f4')
        right = np.array([-0., 0.], dtype='<f4')
        result = diff._measure(left, right, 'bf16')
        self.assertEqual(result['stored_bits_changed'], 2)
        self.assertEqual(result['stored_values_changed'], 1)
        self.assertEqual(result['operand_bits_changed'], 1)
        self.assertEqual(result['operand_values_changed'], 0)
        self.assertEqual(result['stored_subnormal_values'], 1)
        self.assertEqual(result['operand_subnormal_values'], 0)

    def test_nonfinite_padding_and_rounding_overflow_rejected(self):
        for value in (np.nan, np.inf, np.finfo(np.float32).max):
            with self.subTest(value=value):
                self.change('token_embedding.weight', value, index=7)
                with self.assertRaises(ValueError):
                    self.compare()

    def test_invalid_rows_and_chunk_sizes(self):
        for rows in ([1, 0], [0, 0], [3], [-1], [True]):
            with self.assertRaises(ValueError):
                self.compare(selected_rows=rows)
        for size in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                self.compare(chunk_elements=size)

    def test_replaced_source_rejected(self):
        original = diff.paired_weight_patch._snapshot
        calls = 0
        def snapshot(*args):
            nonlocal calls
            calls += 1
            result = original(*args)
            if calls == 3:
                result['directory_identity'] = [-1, -1]
            return result
        with mock.patch.object(diff.paired_weight_patch, '_snapshot', side_effect=snapshot):
            with self.assertRaisesRegex(ValueError, 'changed during'):
                self.compare()

    def test_symlink_weight_and_wrong_size_rejected(self):
        weight = self.b / 'weight_0.bin'
        weight.unlink()
        weight.symlink_to(self.a / 'weight_0.bin')
        with self.assertRaises(ValueError):
            self.compare()
        weight.unlink()
        weight.write_bytes(b'')
        with self.assertRaises(ValueError):
            self.compare()


if __name__ == '__main__':
    unittest.main()
