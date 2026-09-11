"""CPU-only tests of the exact checkpoint map; no GPU or real data required.

Run from the repository root:
    python -m unittest weight_analysis.checkpoint_test

Every tensor in the synthetic fixture contains unique increasing numbers.
Consequently tests catch transposes, wrong block offsets, Q/K/V interleaving,
and accidentally including vocabulary padding, not just compatible shapes.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import struct
import tempfile
import unittest

import numpy as np

from weight_analysis.checkpoint import (
    GPT2Checkpoint, GPT2Config, SOURCE_FILES, latest_checkpoint_directory,
    sha256_file, tensor_manifest,
)


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / 'step_7'
        self.directory.mkdir()
        self.config = GPT2Config(vocab_size=5, padded_vocab_size=8,
                                 context_length=3, n_layers=2, d_model=4,
                                 n_heads=2, d_ff=8)
        self.expected = {}
        for spec in tensor_manifest(self.config):
            values = (np.arange(np.prod(spec.shape), dtype='<f4') +
                      np.float32(1000 * spec.index)).reshape(spec.shape)
            values.tofile(self.directory / spec.filename)
            self.expected[spec.name] = values

    def load(self, **kwargs):
        return GPT2Checkpoint(self.directory, self.config, **kwargs)

    def test_manifest_follows_cpp_weight_order(self):
        manifest = tensor_manifest(self.config)
        self.assertEqual(len(manifest), 4 + 12 * self.config.n_layers)
        self.assertEqual(manifest[0].name, 'token_embedding.weight')
        self.assertEqual(manifest[1].name, 'position_embedding.weight')
        expected_block = [
            'ln1.scale', 'ln1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
            'attn.output.weight', 'attn.output.bias', 'ln2.scale', 'ln2.bias',
            'mlp.input.weight', 'mlp.input.bias', 'mlp.output.weight',
            'mlp.output.bias',
        ]
        for block in range(self.config.n_layers):
            for index, suffix in enumerate(expected_block):
                spec = manifest[2 + 12 * block + index]
                self.assertEqual(spec.name, f'blocks.{block}.{suffix}')
                self.assertEqual(spec.filename, f'weight_{spec.index}.bin')
        self.assertEqual(manifest[-2].name, 'final_norm.scale')
        self.assertEqual(manifest[-1].name, 'final_norm.bias')

    def test_all_arrays_are_exact_read_only_memmaps(self):
        checkpoint = self.load(check_finite=True)
        for spec in checkpoint:
            with self.subTest(name=spec.name):
                actual = checkpoint[spec.name]
                self.assertIsInstance(actual, np.memmap)
                self.assertTrue(actual.flags.c_contiguous)
                self.assertFalse(actual.flags.writeable)
                np.testing.assert_array_equal(actual, self.expected[spec.name])
                with self.assertRaises(ValueError):
                    actual.flat[0] = 7
                with self.assertRaises(ValueError):
                    actual.setflags(write=True)
        with self.assertRaises(TypeError):
            checkpoint.tensors['anything'] = np.zeros(1)
        with self.assertRaises(KeyError):
            checkpoint['typo']

    def test_embedding_padding_and_tied_lm_head(self):
        checkpoint = self.load()
        full = checkpoint['token_embedding.weight']
        self.assertIs(checkpoint['lm_head.weight'], full)
        self.assertEqual(checkpoint.token_embedding.shape, (5, 4))
        np.testing.assert_array_equal(checkpoint.token_embedding, full[:5])
        self.assertTrue(np.shares_memory(full, checkpoint.token_embedding))
        self.assertEqual(checkpoint.manifest[0].shared_with, ('lm_head.weight',))

    def test_qkv_component_and_head_views(self):
        checkpoint = self.load()
        for block in range(self.config.n_layers):
            weight = self.expected[f'blocks.{block}.attn.qkv.weight']
            bias = self.expected[f'blocks.{block}.attn.qkv.bias']
            for component_index, component in enumerate(('q', 'k', 'v')):
                for head in (None, 0, 1):
                    with self.subTest(block=block, component=component, head=head):
                        offset = component_index * 4 + (0 if head is None else head * 2)
                        width = 4 if head is None else 2
                        actual = checkpoint.qkv(block, component, head)
                        np.testing.assert_array_equal(actual, weight[:, offset:offset + width])
                        np.testing.assert_array_equal(checkpoint.qkv_bias(block, component, head),
                                                      bias[offset:offset + width])
                        self.assertTrue(np.shares_memory(
                            actual, checkpoint[f'blocks.{block}.attn.qkv.weight']))
                        self.assertFalse(actual.flags.writeable)

    def test_attention_output_uses_rows_not_columns(self):
        checkpoint = self.load()
        for block in range(self.config.n_layers):
            weight = self.expected[f'blocks.{block}.attn.output.weight']
            for head in range(self.config.n_heads):
                np.testing.assert_array_equal(
                    checkpoint.attention_output_head(block, head),
                    weight[head * 2:(head + 1) * 2, :])

    def test_mlp_keys_and_values_share_neuron_index(self):
        checkpoint = self.load()
        for block in range(self.config.n_layers):
            keys = checkpoint.mlp_keys(block)
            values = checkpoint.mlp_values(block)
            self.assertEqual(keys.shape, (8, 4))
            self.assertEqual(values.shape, (8, 4))
            for neuron in range(8):
                np.testing.assert_array_equal(keys[neuron],
                    self.expected[f'blocks.{block}.mlp.input.weight'][:, neuron])
                np.testing.assert_array_equal(values[neuron],
                    self.expected[f'blocks.{block}.mlp.output.weight'][neuron])
            self.assertTrue(np.shares_memory(keys, checkpoint[f'blocks.{block}.mlp.input.weight']))

    def test_byte_ranges_are_local_exact_c_order(self):
        checkpoint = self.load()
        for spec in checkpoint.manifest:
            self.assertEqual(spec.byte_range, (0, spec.nbytes))
            for indices in np.ndindex(spec.shape):
                start, end = spec.element_byte_range(*indices)
                self.assertEqual(end - start, 4)
                with (self.directory / spec.filename).open('rb') as stream:
                    stream.seek(start)
                    value = struct.unpack('<f', stream.read(4))[0]
                self.assertEqual(value, float(checkpoint[spec.name][indices]))
        spec = checkpoint.manifest[0]
        for indices in [(), (0,), (-1, 0), (0, 4), (8, 0), (0.5, 0)]:
            with self.subTest(indices=indices), self.assertRaises(IndexError):
                spec.element_byte_range(*indices)

    def test_explicit_little_endian_float32(self):
        path = self.directory / 'weight_0.bin'
        with path.open('r+b') as stream:
            stream.write(struct.pack('<f', 1.25))
        self.assertEqual(float(self.load()['token_embedding.weight'][0, 0]), 1.25)

    def test_missing_weight_rejected(self):
        (self.directory / 'weight_4.bin').unlink()
        with self.assertRaisesRegex(ValueError, 'file set mismatch'):
            self.load()

    def test_extra_weight_rejected(self):
        (self.directory / 'weight_28.bin').write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'extra='):
            self.load()

    def test_noncanonical_weight_filename_rejected(self):
        (self.directory / 'weight_00.bin').write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'weight_00.bin'):
            self.load()

    def test_non_weight_metadata_file_ignored(self):
        (self.directory / 'train.log').write_text('synthetic fixture\n')
        self.load()

    def test_truncated_or_oversized_weight_rejected(self):
        path = self.directory / 'weight_4.bin'
        for size in (4, 196):
            path.write_bytes(b'\0' * size)
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, 'expected 192 bytes'):
                self.load()

    def test_directory_masquerading_as_weight_rejected(self):
        path = self.directory / 'weight_4.bin'
        path.unlink()
        path.mkdir()
        with self.assertRaisesRegex(ValueError, 'not a regular file'):
            self.load()

    def test_sae_four_file_checkpoint_rejected(self):
        for spec in tensor_manifest(self.config)[4:]:
            (self.directory / spec.filename).unlink()
        with self.assertRaisesRegex(ValueError, 'file set mismatch'):
            self.load()

    def test_optional_nonfinite_validation_including_padding(self):
        path = self.directory / 'weight_0.bin'
        for value in (float('nan'), float('inf'), float('-inf')):
            with path.open('r+b') as stream:
                stream.seek(7 * 4 * 4)  # A padding token row, not logical vocab.
                stream.write(struct.pack('<f', value))
            checkpoint = self.load()  # No implicit full scan without opt-in.
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'non-finite'):
                checkpoint.validate_finite(chunk_elements=3)
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                self.load(check_finite=True)

    def test_invalid_indices_and_scan_chunk_rejected(self):
        checkpoint = self.load()
        for block in (-1, 2, 0.5, True):
            with self.subTest(block=block), self.assertRaises(IndexError):
                checkpoint.mlp_keys(block)
        for head in (-1, 2, '1', True):
            with self.subTest(head=head), self.assertRaises(IndexError):
                checkpoint.qkv(0, 'q', head)
        with self.assertRaises(ValueError):
            checkpoint.qkv(0, 'query')
        for chunk in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                checkpoint.validate_finite(chunk)

    def test_invalid_config_rejected(self):
        for field in ('vocab_size', 'padded_vocab_size', 'context_length',
                      'n_layers', 'd_model', 'n_heads', 'd_ff'):
            for value in (0, -1, 2.0, True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(self.config, **{field: value})
        with self.assertRaises(ValueError):
            replace(self.config, vocab_size=9)
        with self.assertRaises(ValueError):
            replace(self.config, n_heads=3)

    def test_missing_checkpoint_directory_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not a directory'):
            GPT2Checkpoint(self.directory / 'missing', self.config)

    def test_latest_directory_uses_numeric_steps_not_archives(self):
        parent = Path(self.temporary.name)
        (parent / 'step_9').mkdir()
        (parent / 'step_11').mkdir()
        (parent / 'step_invalid').mkdir()
        (parent / 'step_100.tar.gz').write_bytes(b'not an archive')
        (parent / 'step_999').write_bytes(b'not a directory')
        self.assertEqual(latest_checkpoint_directory(parent), parent / 'step_11')
        with self.assertRaises(FileNotFoundError):
            latest_checkpoint_directory(self.directory)

    def test_provenance_hashes_bytes_without_claiming_historical_source(self):
        checkpoint = self.load()
        result = checkpoint.provenance(hash_weights=True)
        json.dumps(result)  # Report is directly serializable without NumPy hooks.
        self.assertEqual(result['config']['d_model'], 4)
        self.assertEqual(result['unique_weight_files'], 28)
        self.assertEqual(result['checkpoint_bytes'], sum(s.nbytes for s in checkpoint))
        self.assertEqual(result['parameter_count'], result['checkpoint_bytes'] // 4)
        self.assertEqual(set(result['source_sha256']), set(SOURCE_FILES))
        self.assertIn('not authenticated', result['source_hash_scope'])
        self.assertIn('caller assumptions', result['checkpoint_metadata'])
        path = self.directory / 'weight_0.bin'
        self.assertEqual(result['weight_sha256']['weight_0.bin'],
                         hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(sha256_file(path), result['weight_sha256']['weight_0.bin'])
        self.assertNotIn('weight_sha256', checkpoint.provenance())

    def test_default_config_still_matches_cpp_recipe(self):
        # Fail loudly if the fixed C++ recipe changes while this map does not.
        # Tensor order is separately asserted above against the audited builder.
        root = Path(__file__).resolve().parents[2]
        source = (root / 'src/llm/recipes/gpt2.h').read_text()
        config = GPT2Config()
        fields = {'VocabularySize': 'vocab_size',
                  'PaddedVocabularySize': 'padded_vocab_size',
                  'ContextLength': 'context_length',
                  'TransformerBlockCount': 'n_layers',
                  'ModelWidth': 'd_model', 'AttentionHeads': 'n_heads',
                  'AttentionHeadDimension': 'head_dim',
                  'FeedForwardWidth': 'd_ff'}
        for cpp_suffix, field in fields.items():
            match = re.search(r'\bkGpt2' + cpp_suffix + r"\s*=\s*([0-9']+)\s*;", source)
            self.assertIsNotNone(match, cpp_suffix)
            self.assertEqual(int(match[1].replace("'", '')), getattr(config, field))
        self.assertEqual(len(tensor_manifest()), 100)


if __name__ == '__main__':
    unittest.main()
