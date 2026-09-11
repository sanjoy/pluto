"""CPU fixtures for MLP row doses; these tests do not establish model effects."""

import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

from weight_analysis.checkpoint import (
    GPT2Checkpoint, GPT2Config, sha256_file, tensor_manifest)
from weight_analysis import neuron_weight_patch as patcher


class NeuronWeightPatchTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='neuron-weight-patch-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source' / 'step_7'
        self.output = self.root / 'output' / 'step_7'
        self.source.mkdir(parents=True)
        self.output.parent.mkdir()
        self.config = GPT2Config(vocab_size=5, padded_vocab_size=8, context_length=4,
                                 n_layers=2, d_model=4, n_heads=2, d_ff=8)
        self.specs = tensor_manifest(self.config)
        self.tensor = 'blocks.1.mlp.output.weight'
        self.weight = next(spec for spec in self.specs if spec.name == self.tensor)
        for spec in self.specs:
            data = (np.arange(np.prod(spec.shape), dtype='<f4') +
                    np.float32(1 + spec.index * 100)).reshape(spec.shape)
            data.reshape(-1)[::2] *= -1
            if spec.name == self.tensor:
                # Include both signs of zero, a negative scalar, and a smallest
                # subnormal: half-dose rounds its original FP32 value once.
                data[1] = np.array([0x80000000, 0x00000000,
                                    0xC0400000, 0x80000001], dtype='<u4').view('<f4')
            data.tofile(self.source / spec.filename)
        self.initial_hashes = self.hashes(self.source)

    def hashes(self, path):
        return {spec.filename: sha256_file(path / spec.filename) for spec in self.specs}

    def patch(self, **kwargs):
        parameters = dict(block=1, neurons=[7, 1, 0], scale=0.5, config=self.config)
        parameters.update(kwargs)
        return patcher.create_patch(self.source, self.output, **parameters)

    def assert_dose(self, scale, rows):
        source = GPT2Checkpoint(self.source, self.config)
        actual = GPT2Checkpoint(self.output, self.config, check_finite=True)
        for spec in self.specs:
            expected = np.array(source[spec.name], copy=True)
            if spec.name == self.tensor:
                for row in rows:
                    if scale == 0:
                        expected[row] = np.zeros(self.config.d_model, dtype='<f4')
                    elif scale == 0.5:
                        expected[row] = np.multiply(expected[row], np.float32(0.5),
                                                    dtype=np.float32)
            self.assertEqual(actual[spec.name].tobytes(), expected.tobytes(), spec.name)
            output_info = (self.output / spec.filename).stat()
            self.assertEqual(output_info.st_nlink, 1)
            for other in self.specs:
                source_info = (self.source / other.filename).stat()
                self.assertNotEqual((output_info.st_dev, output_info.st_ino),
                                    (source_info.st_dev, source_info.st_ino))
        self.assertEqual(self.hashes(self.source), self.initial_hashes)

    def test_half_dose_exact_rows_bias_preservation_and_artifact(self):
        result = self.patch()
        self.assert_dose(0.5, [0, 1, 7])
        self.assertEqual(result['format'], 'pluto-neuron-weight-patch-v1')
        self.assertEqual(result['selection']['neurons'], [0, 1, 7])
        self.assertEqual(result['selection']['filename'], 'weight_24.bin')
        self.assertEqual(result['selection']['row_byte_ranges'],
                         [[0, 16], [16, 32], [112, 128]])
        self.assertEqual(result['output']['weights_sha256'], self.hashes(self.output))
        self.assertEqual(result['source']['weights_sha256'], self.initial_hashes)
        self.assertEqual(result['source']['step'], 7)
        self.assertTrue(all(result['validation'].values()))
        self.assertFalse(any(result['measurement'].values()))
        self.assertEqual(json.loads((self.output / 'patch.json').read_text()), result)
        self.assertEqual({p.name for p in self.output.iterdir()},
                         {spec.filename for spec in self.specs} | {'patch.json'})
        for path, digest in result['implementation']['sources_sha256'].items():
            self.assertEqual(sha256_file(path), digest)

    def test_zero_dose_uses_positive_zero_even_for_negative_zero_and_values(self):
        self.patch(scale=0)
        self.assert_dose(0, [0, 1, 7])
        actual = GPT2Checkpoint(self.output, self.config)[self.tensor]
        self.assertEqual(actual[[0, 1, 7]].tobytes(), bytes(3 * 4 * 4))

    def test_one_dose_is_bitwise_copy_including_signed_zero(self):
        self.patch(scale=1)
        self.assert_dose(1, [0, 1, 7])
        self.assertEqual(self.hashes(self.output), self.initial_hashes)
        actual = GPT2Checkpoint(self.output, self.config)[self.tensor]
        self.assertEqual(actual[1].view('<u4')[0], 0x80000000)

    def test_half_dose_extreme_finite_values_matches_independent_scalar_rounding(self):
        bits = np.array([0x00000000, 0x80000000, 0x00000001, 0x80000001,
                         0x00000003, 0x80000003, 0x00800000, 0x80800000,
                         0x7F7FFFFF, 0xFF7FFFFF], dtype='<u4')
        values = bits.view('<f4')
        # FP64 represents both FP32 and its product with 0.5 exactly; struct
        # independently rounds once back to FP32, including subnormal ties.
        expected = b''.join(struct.pack('<f', float(value) * 0.5) for value in values)
        self.assertEqual(patcher._scaled_row(values, 0.5).tobytes(), expected)

    def test_donor_patch_validators_reject_neuron_dose_format(self):
        from weight_analysis import embedding_factorial_readout
        from weight_analysis import paired_branch_readout

        self.patch()
        path = self.output / 'patch.json'
        with self.assertRaisesRegex(ValueError, 'invalid patch provenance'):
            paired_branch_readout._patch(path, {}, self.config)
        with self.assertRaisesRegex(ValueError, 'unsupported/incomplete patch'):
            embedding_factorial_readout._validate_patch(path, [], {}, {}, self.config)

    def test_empty_selection_is_independent_copy_control(self):
        result = self.patch(neurons=[], scale=0)
        self.assert_dose(0, [])
        self.assertEqual(self.hashes(self.output), self.initial_hashes)
        self.assertEqual(result['selection']['row_byte_ranges'], [])

    def test_numpy_integer_selections_are_canonicalized(self):
        result = self.patch(block=np.int64(1), neurons=[np.int64(2), np.int32(1)])
        self.assertEqual(result['selection']['neurons'], [1, 2])
        self.assertIs(type(result['selection']['block']), int)
        self.assert_dose(0.5, [1, 2])

    def test_invalid_blocks_doses_and_neurons_do_not_create_output(self):
        cases = ([{'block': value} for value in [-1, 2, True, np.bool_(0), 1.0, '1']] +
                 [{'scale': value} for value in [-1, 0.25, 2, np.nan, np.inf,
                                                 -np.inf, True, np.bool_(0), '0.5']] +
                 [{'neurons': values} for values in [[-1], [8], [1, 1], [1.0],
                                                      [True], [np.bool_(0)], ['1'],
                                                      [None], [np.int64(1), 1]]])
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.patch(**arguments)
            self.assertFalse(self.output.exists())

    def test_existing_output_file_directory_and_symlink_are_preserved(self):
        for kind in ('file', 'directory', 'symlink'):
            with self.subTest(kind=kind):
                if kind == 'file':
                    self.output.write_bytes(b'preserve')
                elif kind == 'directory':
                    self.output.mkdir()
                else:
                    self.output.symlink_to(self.root / 'absent')
                with self.assertRaises((FileExistsError, ValueError)):
                    self.patch()
                if kind == 'file':
                    self.assertEqual(self.output.read_bytes(), b'preserve')
                elif kind == 'symlink':
                    self.assertTrue(self.output.is_symlink())
                    self.assertFalse((self.root / 'absent').exists())
                if kind == 'directory':
                    self.output.rmdir()
                else:
                    self.output.unlink()

    def test_source_destination_overlap_is_rejected(self):
        for output in [self.source, self.source / 'step_7', self.source.parent]:
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'overlap'):
                patcher.create_patch(self.source, output, block=1, neurons=[1],
                                     scale=0, config=self.config)
        self.assertEqual(self.hashes(self.source), self.initial_hashes)

    def test_step_basename_and_integer_range_and_existing_parent_required(self):
        for output in [self.output.parent / 'checkpoint', self.output.parent / 'step_8',
                       self.root / 'absent' / 'step_7']:
            with self.subTest(output=output), self.assertRaises(ValueError):
                patcher.create_patch(self.source, output, block=1, neurons=[1],
                                     scale=0, config=self.config)
            self.assertFalse(output.exists())
        for basename in ['checkpoint', 'step_-7', 'step_2147483648', 'step_７']:
            renamed = self.source.with_name(basename)
            self.source.rename(renamed)
            try:
                with self.subTest(basename=basename), self.assertRaises(ValueError):
                    patcher.create_patch(renamed, self.output.parent / basename,
                                         block=1, neurons=[1], scale=0,
                                         config=self.config)
            finally:
                renamed.rename(self.source)

    def test_symlinked_source_output_ancestors_and_dotdot_are_rejected(self):
        alias = self.root / 'alias'
        alias.symlink_to(self.source.parent, target_is_directory=True)
        cases = [(alias / 'step_7', self.output),
                 (self.source, alias / 'another' / 'step_7'),
                 (alias / '..' / 'source' / 'step_7', self.output)]
        for source, output in cases:
            with self.subTest(source=source, output=output), self.assertRaisesRegex(
                    ValueError, 'symlink'):
                patcher.create_patch(source, output, block=1, neurons=[1],
                                     scale=0, config=self.config)
        self.assertFalse(self.output.exists())

    def test_symlinked_weight_and_sidecar_are_rejected(self):
        path = self.source / 'weight_0.bin'
        outside = self.root / 'outside'
        path.rename(outside)
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'regular'):
            self.patch()
        path.unlink()
        outside.rename(path)
        (self.source / 'sidecar').symlink_to(self.root / 'missing')
        with self.assertRaisesRegex(ValueError, 'regular'):
            self.patch()
        self.assertFalse(self.output.exists())

    def test_regular_sidecar_is_not_copied(self):
        (self.source / 'notes.txt').write_text('source metadata, not a weight')
        self.patch()
        self.assertFalse((self.output / 'notes.txt').exists())
        self.assertEqual((self.source / 'notes.txt').read_text(),
                         'source metadata, not a weight')

    def test_missing_extra_malformed_and_wrong_size_weights_are_rejected(self):
        path = self.source / 'weight_0.bin'
        saved = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'file set'):
            self.patch()
        path.write_bytes(saved[:-4])
        with self.assertRaisesRegex(ValueError, 'regular'):
            self.patch()
        path.write_bytes(saved)
        for basename in ['weight_999.bin', 'weight_01.bin', 'weight_bad.bin']:
            extra = self.source / basename
            extra.write_bytes(b'')
            with self.subTest(basename=basename), self.assertRaisesRegex(ValueError, 'file set'):
                self.patch()
            extra.unlink()
        self.assertFalse(self.output.exists())

    def test_nonfinite_weights_including_unselected_padding_are_rejected(self):
        path = self.source / 'weight_0.bin'
        saved = path.read_bytes()
        for value in [np.nan, np.inf, -np.inf]:
            data = np.frombuffer(saved, dtype='<f4').copy()
            data[-1] = value
            data.tofile(path)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.patch()
            self.assertFalse(self.output.exists())
        path.write_bytes(saved)

    def test_corrupted_unselected_output_retains_partial_without_marker(self):
        real_copy = patcher.shutil.copyfileobj

        def corrupt(incoming, outgoing, **kwargs):
            real_copy(incoming, outgoing, **kwargs)
            if Path(incoming.name).name == self.weight.filename:
                outgoing.seek(2 * self.config.d_model * 4)
                outgoing.write(np.array([123.0], dtype='<f4').tobytes())

        with mock.patch.object(patcher.shutil, 'copyfileobj', side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, 'unselected MLP'):
                self.patch()
        self.assertTrue(self.output.is_dir())
        self.assertFalse((self.output / 'patch.json').exists())
        self.assertEqual(self.hashes(self.source), self.initial_hashes)

    def test_corrupted_selected_output_retains_partial_without_marker(self):
        real_scale = patcher._scaled_row
        calls = 0

        def corrupt_once(row, scale):
            nonlocal calls
            calls += 1
            result = real_scale(row, scale)
            if calls == 1:
                result[0] = np.float32(123.0)
            return result

        with mock.patch.object(patcher, '_scaled_row', side_effect=corrupt_once):
            with self.assertRaisesRegex(ValueError, 'selected MLP'):
                self.patch()
        self.assertFalse((self.output / 'patch.json').exists())
        self.assertEqual(self.hashes(self.source), self.initial_hashes)

    def test_source_change_detected_even_with_snapshot_check_mocked_unchanged(self):
        before = patcher._source_snapshot(self.source, self.specs)
        real_copy = patcher.shutil.copyfileobj

        def change_after_copy(incoming, outgoing, **kwargs):
            real_copy(incoming, outgoing, **kwargs)
            if Path(incoming.name).name == 'weight_0.bin':
                with (self.source / 'weight_0.bin').open('r+b') as stream:
                    stream.write(np.array([123.0], dtype='<f4').tobytes())

        with mock.patch.object(patcher, '_source_snapshot', return_value=before):
            with mock.patch.object(patcher.shutil, 'copyfileobj', side_effect=change_after_copy):
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    self.patch()
        self.assertTrue(self.output.is_dir())
        self.assertFalse((self.output / 'patch.json').exists())

    def test_marker_is_last_and_not_overwritten(self):
        original = patcher._publish_marker

        def verify_and_publish(path, provenance):
            self.assertEqual({p.name for p in self.output.iterdir()},
                             {spec.filename for spec in self.specs})
            self.assert_dose(0.5, [0, 1, 7])
            original(path, provenance)

        with mock.patch.object(patcher, '_publish_marker', side_effect=verify_and_publish):
            self.patch()
        marker = self.output / 'patch.json'
        saved = marker.read_bytes()
        with self.assertRaises(FileExistsError):
            original(marker, {'complete': False})
        self.assertEqual(marker.read_bytes(), saved)

    def test_marker_write_failure_removes_incomplete_marker(self):
        marker = self.root / 'patch.json'
        with mock.patch.object(patcher.os, 'fsync', side_effect=OSError('injected failure')):
            with self.assertRaisesRegex(OSError, 'injected failure'):
                patcher._publish_marker(marker, {'complete': True})
        self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()
