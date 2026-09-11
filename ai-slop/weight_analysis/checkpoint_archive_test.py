"""Small CPU fixtures exercise the reader, never real checkpoints or a GPU."""

import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import checkpoint_archive
from .checkpoint import GPT2Config, sha256_file, tensor_manifest
from .checkpoint_archive import read_embedding


class CheckpointArchiveTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / 'step_7'
        self.directory.mkdir()
        self.archive = self.root / 'step_7.tar.gz'
        self.config = GPT2Config(vocab_size=3, padded_vocab_size=4,
                                 context_length=3, n_layers=1, d_model=2,
                                 n_heads=1, d_ff=4)
        self.manifest = tensor_manifest(self.config)
        self.payloads = {}
        for spec in self.manifest:
            array = (np.arange(np.prod(spec.shape), dtype='<f4') +
                     np.float32(spec.index * 100)).reshape(spec.shape)
            self.payloads[spec.filename] = array.tobytes()
            (self.directory / spec.filename).write_bytes(array.tobytes())

    def members(self):
        top = tarfile.TarInfo('step_7')
        top.type = tarfile.DIRTYPE
        top.mtime = 123.25
        members = [(top, b'')]
        for spec in self.manifest:
            info = tarfile.TarInfo('step_7/' + spec.filename)
            data = self.payloads[spec.filename]
            info.size = len(data)
            info.mtime = 125.5
            members.append((info, data))
        return members

    def write_archive(self, members=None):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w', format=tarfile.PAX_FORMAT) as archive:
            for info, data in self.members() if members is None else members:
                archive.addfile(info, io.BytesIO(data))
        self.archive.write_bytes(gzip.compress(stream.getvalue(), mtime=0))
        return self.archive

    def load(self, path=None):
        return read_embedding(self.archive if path is None else path, 7, self.config)

    def assert_result(self, actual, provenance):
        expected = np.frombuffer(self.payloads['weight_0.bin'], dtype='<f4').reshape(4, 2)
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(actual.flags.writeable)
        self.assertEqual(actual.dtype.str, '<f4')
        self.assertEqual(provenance['physical_embedding_shape'], [4, 2])
        self.assertEqual(provenance['logical_vocab_size'], 3)
        self.assertEqual(provenance['embedding_sha256'],
                         hashlib.sha256(self.payloads['weight_0.bin']).hexdigest())
        self.assertTrue(provenance['input_files_unchanged_during_read'])
        self.assertFalse(provenance['other_weight_finiteness_checked'])
        json.dumps(provenance, allow_nan=False)
        with self.assertRaises(ValueError):
            actual[0, 0] = 999
        with self.assertRaises(ValueError):
            actual.setflags(write=True)

    def test_valid_directory_full_hashes_and_input_unchanged(self):
        before = {p.name: sha256_file(p) for p in self.directory.iterdir()}
        actual, provenance = self.load(self.directory)
        self.assert_result(actual, provenance)
        self.assertEqual(provenance['source_kind'], 'directory')
        self.assertEqual(len(provenance['input_files']), len(self.manifest))
        self.assertEqual(provenance['input_files'],
                         [checkpoint_archive._content_record(record)
                          for record in provenance['weight_files_after']])
        self.assertTrue(all(set(record) == {'path', 'bytes', 'sha256'}
                            for record in provenance['input_files']))
        self.assertEqual(before, {p.name: sha256_file(p) for p in self.directory.iterdir()})

    def test_valid_pax_archive_full_hash_and_no_extraction(self):
        self.write_archive()
        before = self.archive.read_bytes()
        directory_before = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        actual, provenance = self.load()
        self.assert_result(actual, provenance)
        self.assertTrue(provenance['gzip_eof_and_crc_verified'])
        self.assertTrue(provenance['trailing_tar_bytes_zero_only'])
        self.assertEqual(provenance['archive_sha256'], hashlib.sha256(before).hexdigest())
        self.assertEqual(provenance['archive_bytes'], len(before))
        self.assertEqual(len(provenance['members']), len(self.manifest) + 1)
        self.assertEqual(provenance['members'][0]['pax_headers']['mtime'], '123.25')
        self.assertEqual(provenance['input_files'],
                         [checkpoint_archive._content_record(provenance['archive_after'])])
        self.assertEqual(provenance['archive_before'], provenance['archive_after'])
        self.assertEqual(self.archive.read_bytes(), before)
        self.assertEqual({p.name: p.read_bytes() for p in self.directory.iterdir()}, directory_before)
        self.assertEqual({p.name for p in self.root.iterdir()}, {'step_7', 'step_7.tar.gz'})

    def test_member_order_not_assumed(self):
        self.write_archive(list(reversed(self.members())))
        self.assert_result(*self.load())

    def test_wrong_path_name_and_step_rejected(self):
        for step in (-1, True, 7.0, 8):
            with self.subTest(step=step), self.assertRaises(ValueError):
                read_embedding(self.directory, step, self.config)
        with self.assertRaisesRegex(ValueError, 'config'):
            read_embedding(self.directory, 7, {})

    def test_directory_extra_log_missing_wrong_size(self):
        extra = self.directory / 'log.txt'
        extra.write_text('extra')
        with self.assertRaisesRegex(ValueError, 'extra='):
            self.load(self.directory)
        extra.unlink()
        weight = self.directory / 'weight_1.bin'
        weight.unlink()
        with self.assertRaisesRegex(ValueError, 'missing='):
            self.load(self.directory)
        weight.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError, 'expected'):
            self.load(self.directory)

    def test_directory_symlink_or_directory_weight_rejected(self):
        weight = self.directory / 'weight_1.bin'
        weight.unlink()
        weight.symlink_to(self.directory / 'weight_0.bin')
        with self.assertRaisesRegex(ValueError, 'non-symlink'):
            self.load(self.directory)
        weight.unlink()
        weight.mkdir()
        with self.assertRaisesRegex(ValueError, 'regular'):
            self.load(self.directory)

    def test_top_path_symlink_rejected_for_both_kinds(self):
        outside = self.root / 'links'
        outside.mkdir()
        for target in (self.directory, self.write_archive()):
            link = outside / target.name
            link.symlink_to(target)
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'non-symlink'):
                self.load(link)

    def test_missing_archive_weight_or_top_directory(self):
        members = self.members()
        for removed in (0, 1, len(members) - 1):
            self.write_archive(members[:removed] + members[removed + 1:])
            with self.subTest(removed=removed), self.assertRaisesRegex(ValueError, 'missing archive'):
                self.load()

    def test_duplicate_archive_weight_or_directory(self):
        for index in (0, 1):
            members = self.members()
            members.append(members[index])
            self.write_archive(members)
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'duplicate'):
                self.load()

    def test_wrong_member_size(self):
        for index in (1, 2):
            members = self.members()
            member, data = members[index]
            member.size -= 4
            members[index] = (member, data[:-4])
            self.write_archive(members)
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'expected'):
                self.load()

    def test_unknown_traversal_absolute_or_nested_names(self):
        for name in ('step_7/../escape', '/step_7/weight_0.bin',
                     'step_7/nested/weight_0.bin', 'step_7/weight_00.bin',
                     'step_7/log.txt', 'step_8/weight_0.bin'):
            members = self.members()
            members[1][0].name = name
            self.write_archive(members)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'unexpected'):
                self.load()
        self.assertFalse((self.root / 'escape').exists())

    def test_symlinks_hardlinks_devices_and_directory_weights_rejected(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE,
                     tarfile.BLKTYPE, tarfile.FIFOTYPE, tarfile.DIRTYPE):
            members = self.members()
            member, _ = members[1]
            member.type = kind
            member.size = 0
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                member.linkname = '../escape'
            members[1] = member, b''
            self.write_archive(members)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'plain regular'):
                self.load()

    def test_top_directory_type_rejected(self):
        members = self.members()
        members[0][0].type = tarfile.REGTYPE
        self.write_archive(members)
        with self.assertRaisesRegex(ValueError, 'empty directory'):
            self.load()

    def test_regular_file_link_metadata_rejected(self):
        members = self.members()
        members[1][0].linkname = 'not-used-by-regular-file'
        self.write_archive(members)
        with self.assertRaisesRegex(ValueError, 'plain regular'):
            self.load()

    def test_sparse_extension_metadata_rejected(self):
        members = self.members()
        members[1][0].pax_headers['GNU.sparse.unsupported'] = '1'
        self.write_archive(members)
        with self.assertRaisesRegex(ValueError, 'plain regular'):
            self.load()

    def test_nonfinite_embedding_including_padding_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            data = np.frombuffer(self.payloads['weight_0.bin'], dtype='<f4').copy()
            data[-1] = value  # Deliberately in a padding row.
            self.payloads['weight_0.bin'] = data.tobytes()
            (self.directory / 'weight_0.bin').write_bytes(data.tobytes())
            self.write_archive()
            for path in (self.directory, self.archive):
                with self.subTest(value=value, path=path), self.assertRaisesRegex(ValueError, 'non-finite'):
                    self.load(path)

    def test_gzip_crc_corruption_detected_after_tar_end(self):
        self.write_archive()
        data = bytearray(self.archive.read_bytes())
        data[-8] ^= 1
        self.archive.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'gzip/tar'):
            self.load()

    def test_gzip_trailer_truncation_detected_after_tar_end(self):
        self.write_archive()
        data = self.archive.read_bytes()
        for removed in (1, 4, 8, len(data) // 2):
            self.archive.write_bytes(data[:-removed])
            with self.subTest(removed=removed), self.assertRaises(ValueError):
                self.load()

    def test_invalid_tar_checksum_rejected(self):
        self.write_archive()
        data = bytearray(gzip.decompress(self.archive.read_bytes()))
        data[10] ^= 1
        self.archive.write_bytes(gzip.compress(data))
        with self.assertRaisesRegex(ValueError, 'gzip/tar'):
            self.load()

    def test_nonzero_trailing_data_or_second_tar_rejected(self):
        self.write_archive()
        tar_bytes = gzip.decompress(self.archive.read_bytes())
        for suffix in (b'not padding', tar_bytes):
            self.archive.write_bytes(gzip.compress(tar_bytes + suffix))
            with self.subTest(suffix_length=len(suffix)), self.assertRaisesRegex(ValueError, 'terminator'):
                self.load()

    def test_changed_file_during_hash_rejected(self):
        self.write_archive()
        original = checkpoint_archive.sha256_file

        def hash_then_mutate(path):
            result = original(path)
            with Path(path).open('ab') as stream:
                stream.write(b'change')
            return result

        with mock.patch.object(checkpoint_archive, 'sha256_file', side_effect=hash_then_mutate):
            with self.assertRaisesRegex(ValueError, 'changed while hashing'):
                self.load()

    def test_changed_input_after_embedding_materialized_rejected(self):
        self.write_archive()
        original = checkpoint_archive._embedding
        for input_path, changed_path in ((self.directory, self.directory / 'weight_1.bin'),
                                         (self.archive, self.archive)):
            saved = changed_path.read_bytes()

            def materialize_then_mutate(payload, spec):
                result = original(payload, spec)
                data = bytearray(changed_path.read_bytes())
                data[-1] ^= 1  # Keep its length unchanged.
                changed_path.write_bytes(data)
                return result

            with self.subTest(input_path=input_path):
                with mock.patch.object(checkpoint_archive, '_embedding',
                                       side_effect=materialize_then_mutate):
                    with self.assertRaisesRegex(ValueError, 'changed during read'):
                        self.load(input_path)
                changed_path.write_bytes(saved)


if __name__ == '__main__':
    unittest.main()
