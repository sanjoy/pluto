"""Tiny CPU fixtures for exact, non-mutating paired checkpoint patches."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from weight_analysis.checkpoint import (
    GPT2Checkpoint, GPT2Config, sha256_file, tensor_manifest)
from weight_analysis import paired_weight_patch
from weight_analysis.paired_weight_patch import create_patch


class PairedWeightPatchTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='paired-weight-patch-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original = self.root / 'original'
        self.replacement = self.root / 'replacement'
        self.output = self.root / 'patched'
        self.config = GPT2Config(vocab_size=5, padded_vocab_size=8, context_length=4,
                                 n_layers=1, d_model=4, n_heads=2, d_ff=8)
        self.specs = tensor_manifest(self.config)
        for path, shift in [(self.original, 0), (self.replacement, -1000)]:
            path.mkdir()
            for spec in self.specs:
                data = (np.arange(np.prod(spec.shape), dtype='<f4') +
                        np.float32(spec.index * 100 + shift)).reshape(spec.shape)
                if spec.index == 0 and shift:
                    data[1, 0] = np.float32(-0.0)
                data.tofile(path / spec.filename)
        self.initial_hashes = {str(path): self.hashes(path)
                               for path in (self.original, self.replacement)}

    def hashes(self, path):
        return {spec.filename: sha256_file(path / spec.filename) for spec in self.specs}

    def patch(self, **kwargs):
        return create_patch(self.original, self.replacement, self.output,
                            config=self.config, **kwargs)

    def assert_expected(self, tensors=(), rows=()):
        a = GPT2Checkpoint(self.original, self.config)
        b = GPT2Checkpoint(self.replacement, self.config)
        result = GPT2Checkpoint(self.output, self.config, check_finite=True)
        for spec in self.specs:
            expected = np.array(b[spec.name] if spec.name in tensors else a[spec.name])
            if spec.name == 'token_embedding.weight' and rows:
                expected[list(rows)] = b[spec.name][list(rows)]
            self.assertEqual(result[spec.name].tobytes(), expected.tobytes(), spec.name)
            for source in (self.original, self.replacement):
                before, after = (source / spec.filename).stat(), (self.output / spec.filename).stat()
                self.assertNotEqual((before.st_dev, before.st_ino), (after.st_dev, after.st_ino))
        for source in (self.original, self.replacement):
            self.assertEqual(self.hashes(source), self.initial_hashes[str(source)])

    def test_rows_only_copy_exact_bits_and_leave_padding_and_sources_unchanged(self):
        provenance = self.patch(embedding_rows=[4, 1])
        self.assert_expected(rows=[1, 4])
        self.assertEqual(provenance['selection']['embedding_rows'], [1, 4])
        self.assertEqual(provenance['selection']['embedding_row_byte_ranges'], [[16, 32], [64, 80]])
        self.assertEqual(provenance['output']['weights_sha256'], self.hashes(self.output))
        for name, source in [('original', self.original), ('replacement', self.replacement)]:
            self.assertEqual(provenance['sources'][name]['weights_sha256'], self.hashes(source))
        self.assertEqual(json.loads((self.output / 'patch.json').read_text()), provenance)
        self.assertEqual({p.name for p in self.output.iterdir()},
                         {spec.filename for spec in self.specs} | {'patch.json'})
        self.assertTrue(all(provenance['validation'].values()))

    def test_whole_tensors_and_rows_can_be_combined(self):
        tensors = ['blocks.0.attn.qkv.weight', 'final_norm.bias']
        self.patch(tensors=tensors, embedding_rows=[2])
        self.assert_expected(tensors=tensors, rows=[2])

    def test_tied_head_alias_copies_whole_embedding(self):
        provenance = self.patch(tensors=['lm_head.weight'])
        self.assertEqual(provenance['selection']['tensors'], ['token_embedding.weight'])
        self.assert_expected(tensors=['token_embedding.weight'])

    def test_empty_selection_is_independent_exact_copy_control(self):
        self.patch()
        self.assert_expected()

    def test_invalid_or_duplicate_tensor_names(self):
        for names in [['missing'], ['final_norm.bias'] * 2, [42],
                      ['token_embedding.weight', 'lm_head.weight']]:
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.patch(tensors=names)
            self.assertFalse(self.output.exists())

    def test_invalid_or_duplicate_embedding_rows(self):
        for rows in [[-1], [5], [8], [1, 1], [1.0], [True], [np.bool_(False)]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.patch(embedding_rows=rows)
            self.assertFalse(self.output.exists())

    def test_whole_embedding_overlap_rejected(self):
        for tensor in ['token_embedding.weight', 'lm_head.weight']:
            with self.subTest(tensor=tensor), self.assertRaisesRegex(ValueError, 'overlap'):
                self.patch(tensors=[tensor], embedding_rows=[0])
        self.assertFalse(self.output.exists())

    def test_existing_file_directory_and_dangling_symlink_are_not_overwritten(self):
        file_path = self.root / 'existing_file'
        file_path.write_bytes(b'preserve')
        dangling = self.root / 'dangling'
        dangling.symlink_to(self.root / 'absent')
        for output in (file_path, self.original, dangling):
            with self.subTest(output=output), self.assertRaises(FileExistsError):
                create_patch(self.original, self.replacement, output, config=self.config)
        self.assertEqual(file_path.read_bytes(), b'preserve')
        self.assertTrue(dangling.is_symlink())
        self.assertFalse((self.root / 'absent').exists())

    def test_output_inside_either_source_and_symlinked_parent_rejected(self):
        alias = self.root / 'source_alias'
        alias.symlink_to(self.replacement, target_is_directory=True)
        for parent in (self.original, self.replacement, alias):
            output = parent / 'nested'
            with self.subTest(parent=parent), self.assertRaisesRegex(ValueError, 'inside'):
                create_patch(self.original, self.replacement, output, config=self.config)
            self.assertFalse(output.exists())

    def test_missing_extra_and_wrong_sized_weights_rejected(self):
        path = self.replacement / self.specs[2].filename
        original_bytes = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'file set'):
            self.patch(embedding_rows=[1])
        path.write_bytes(original_bytes[:-4])
        with self.assertRaisesRegex(ValueError, 'regular'):
            self.patch(embedding_rows=[1])
        path.write_bytes(original_bytes)
        extra = self.replacement / 'weight_999.bin'
        extra.write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'file set'):
            self.patch(embedding_rows=[1])
        self.assertFalse(self.output.exists())

    def test_nonfinite_sources_even_unselected_padding_rejected(self):
        for source in (self.original, self.replacement):
            path = source / 'weight_0.bin'
            original_bytes = path.read_bytes()
            for nonfinite in (np.nan, np.inf):
                data = np.frombuffer(original_bytes, dtype='<f4').copy()
                data[-1] = nonfinite
                data.tofile(path)
                with self.subTest(source=source, value=nonfinite), self.assertRaises(ValueError):
                    self.patch(tensors=['final_norm.bias'])
                self.assertFalse(self.output.exists())
            path.write_bytes(original_bytes)

    def test_symlinked_source_weight_rejected(self):
        path = self.replacement / 'weight_0.bin'
        target = self.root / 'outside_weight'
        path.rename(target)
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'regular'):
            self.patch(embedding_rows=[1])
        self.assertFalse(self.output.exists())

    def test_unselected_output_corruption_prevents_complete_marker(self):
        real_copy = paired_weight_patch.shutil.copyfileobj

        def corrupt_output(incoming, outgoing, **kwargs):
            real_copy(incoming, outgoing, **kwargs)
            if Path(incoming.name).name == 'weight_0.bin':
                outgoing.seek(self.specs[0].nbytes - 4)
                outgoing.write(np.array([123.0], dtype='<f4').tobytes())

        with mock.patch.object(paired_weight_patch.shutil, 'copyfileobj', side_effect=corrupt_output):
            with self.assertRaisesRegex(ValueError, 'unselected embedding'):
                self.patch(embedding_rows=[1])
        self.assertFalse((self.output / 'patch.json').exists())
        for source in (self.original, self.replacement):
            self.assertEqual(self.hashes(source), self.initial_hashes[str(source)])

    def test_source_change_during_copy_prevents_complete_marker(self):
        real_copy = paired_weight_patch.shutil.copyfileobj
        changed = False

        def change_source(incoming, outgoing, **kwargs):
            nonlocal changed
            real_copy(incoming, outgoing, **kwargs)
            if not changed:
                changed = True
                path = self.replacement / self.specs[-1].filename
                with path.open('r+b') as stream:
                    stream.write(np.array([123.0], dtype='<f4').tobytes())

        with mock.patch.object(paired_weight_patch.shutil, 'copyfileobj', side_effect=change_source):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                self.patch(embedding_rows=[1])
        self.assertTrue(self.output.is_dir())
        self.assertFalse((self.output / 'patch.json').exists())


if __name__ == '__main__':
    unittest.main()
