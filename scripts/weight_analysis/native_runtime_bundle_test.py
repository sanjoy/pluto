"""CPU fixture tests; ldd is mocked and no native/GPU code is executed."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from . import native_runtime_bundle as bundle


class ParserTest(unittest.TestCase):
    def test_loader_and_virtual_mapping(self):
        self.assertEqual(bundle._parse_ldd('linux-vdso.so.1 (0x123)\n'
            'liba.so => /lib/liba.so (0xabc)\n/lib/ld-linux.so.1 (0x42)\n'),
            {'liba.so': '/lib/liba.so', 'ld-linux.so.1': '/lib/ld-linux.so.1'})

    def test_malformed_unresolved_duplicate_or_empty(self):
        for text in ('', 'statically linked', 'liba.so => not found',
                     'liba.so => relative/liba.so (0x1)', 'nonsense',
                     '../liba.so => /lib/liba.so (0x1)',
                     'liba.so => /a (0x1)\nliba.so => /b (0x2)',
                     '/lib/ld.so (0x1)\n/lib/ld.so (0x2)'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                bundle._parse_ldd(text)


class BundleTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / 'bazel-bin'
        self.solib = self.bin / '_solib_aarch64'
        self.solib.mkdir(parents=True)
        self.reference = self.bin / 'original'
        self.relocated = self.root / 'relocated'
        self.reference.write_bytes(b'trusted CPU fixture, not executable')
        self.relocated.write_bytes(self.reference.read_bytes())
        self.library = self.bin / 'libimplementation.so'
        self.library.write_bytes(b'project library')
        self.loader_name = 'libproject_Slibimplementation.so'
        (self.solib / self.loader_name).symlink_to(self.library)
        self.system = self.root / 'libsystem.so.1'
        self.system.write_bytes(b'system library')
        self.loader = self.root / 'ld-linux.so.1'
        self.loader.write_bytes(b'ELF loader')
        self.cuda = self.root / 'libcudart.so.13'
        self.cuda.write_bytes(b'CUDA runtime, never executed')
        (self.solib / 'cuda_toolchain').mkdir()
        (self.solib / 'cuda_toolchain/libcudart.so.13').symlink_to(self.cuda)
        self.destination = self.root / 'runtime'
        self.environment = mock.patch.dict(os.environ, {'PATH': '/usr/bin:/bin',
            'LD_LIBRARY_PATH': '/old/unrelated', 'PRIVATE_TEST_SECRET': 'do-not-serialize'}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.calls = []

    def output(self, after=False):
        project = self.destination / self.loader_name if after else self.solib / self.loader_name
        cuda = self.cuda if after else self.solib / 'cuda_toolchain/libcudart.so.13'
        return ('linux-vdso.so.1 (0x1)\n'
                f'{self.loader_name} => {project} (0x2)\n'
                f'libsystem.so.1 => {self.system} (0x3)\n'
                f'libcudart.so.13 => {cuda} (0x4)\n'
                f'{self.loader} (0x5)\n')

    def ldd(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, self.output(len(self.calls) == 2), '')

    def freeze(self, side_effect=None):
        with mock.patch.object(bundle.subprocess, 'run', side_effect=side_effect or self.ldd):
            return bundle.freeze_runtime(self.reference, self.relocated, self.destination)

    def test_success_preserves_executables_and_pins_entire_closure(self):
        before = self.relocated.read_bytes()
        result = self.freeze()
        self.assertEqual(self.relocated.read_bytes(), before)
        self.assertEqual(self.reference.read_bytes(), before)
        self.assertEqual(result['environment'], {'LD_LIBRARY_PATH': str(self.destination)})
        self.assertNotIn('PRIVATE_TEST_SECRET', str(result))
        self.assertNotIn('do-not-serialize', str(result))
        self.assertNotIn('LD_LIBRARY_PATH', self.calls[0][1]['env'])
        self.assertEqual(self.calls[1][1]['env']['LD_LIBRARY_PATH'], str(self.destination))
        self.assertEqual(set(self.destination.iterdir()), {self.destination / self.loader_name})
        self.assertEqual((self.destination / self.loader_name).read_bytes(), self.library.read_bytes())
        self.assertFalse((self.destination / self.loader_name).is_symlink())
        self.assertEqual(sum(r['bundled'] for r in result['libraries']), 1)
        self.assertEqual(len(result['libraries']), 4)
        self.assertEqual(len(result['frozen_records']), 7)
        self.assertFalse(result['runtime_dlopen_covered'])
        bundle._verify(result['frozen_records'])

    def test_existing_destination_is_untouched(self):
        self.destination.mkdir()
        marker = self.destination / 'existing'
        marker.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError): self.freeze()
        self.assertEqual(marker.read_bytes(), b'preserve')
        self.assertEqual(self.calls, [])

    def test_dangling_destination_symlink_is_rejected(self):
        self.destination.symlink_to(self.root / 'absent')
        with self.assertRaises(FileExistsError): self.freeze()

    def test_executable_identity_mismatch(self):
        self.relocated.write_bytes(b'different')
        with self.assertRaisesRegex(ValueError, 'bytes differ'): self.freeze()
        self.assertFalse(self.destination.exists())
        self.assertEqual(self.calls, [])

    def test_relocated_symlink_is_rejected(self):
        self.relocated.unlink()
        self.relocated.symlink_to(self.reference)
        with self.assertRaisesRegex(ValueError, 'symlink'): self.freeze()

    def test_preload_and_audit_even_empty_are_rejected(self):
        for key in ('LD_PRELOAD', 'LD_AUDIT'):
            for value in ('', '/evil.so'):
                with self.subTest(key=key, value=value), mock.patch.dict(os.environ, {key: value}):
                    with self.assertRaisesRegex(ValueError, key): self.freeze()
        self.assertEqual(self.calls, [])

    def test_unresolved_reference(self):
        def missing(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, 'libmissing.so => not found\n', '')
        with self.assertRaisesRegex(ValueError, 'unresolved'): self.freeze(missing)
        self.assertFalse(self.destination.exists())

    def test_ldd_error_or_warning_rejected(self):
        for returncode, stderr in [(1, ''), (0, 'unexpected warning')]:
            with self.subTest(returncode=returncode, stderr=stderr):
                with self.assertRaisesRegex(ValueError, 'ldd failed'):
                    self.freeze(lambda argv, **kw: subprocess.CompletedProcess(argv, returncode, self.output(), stderr))
        self.assertFalse(self.destination.exists())

    def test_bazel_symlink_cannot_escape_build_tree(self):
        (self.solib / self.loader_name).unlink()
        (self.solib / self.loader_name).symlink_to(self.system)
        with self.assertRaisesRegex(ValueError, 'escapes build tree'): self.freeze()
        self.assertFalse(self.destination.exists())

    def test_directory_dependency_is_rejected(self):
        self.library.unlink()
        self.library.mkdir()
        with self.assertRaisesRegex(ValueError, 'regular file'): self.freeze()

    def test_changed_source_detected_after_copy(self):
        def mutated(argv, **kwargs):
            result = self.ldd(argv, **kwargs)
            if len(self.calls) == 2: self.library.write_bytes(b'mutated after copy')
            return result
        with self.assertRaisesRegex(ValueError, 'runtime input changed'): self.freeze(mutated)

    def test_changed_executable_detected(self):
        def mutated(argv, **kwargs):
            result = self.ldd(argv, **kwargs)
            if len(self.calls) == 2: self.relocated.write_bytes(b'mutated executable')
            return result
        with self.assertRaisesRegex(ValueError, 'runtime input changed'): self.freeze(mutated)

    def test_changed_copy_detected(self):
        def mutated(argv, **kwargs):
            result = self.ldd(argv, **kwargs)
            if len(self.calls) == 2:
                target = self.destination / self.loader_name
                target.chmod(0o644)
                target.write_bytes(b'changed copy')
            return result
        with self.assertRaisesRegex(ValueError, 'dependency bytes differ'): self.freeze(mutated)

    def test_relocated_fallback_to_bazel_rejected(self):
        def fallback(argv, **kwargs):
            result = self.ldd(argv, **kwargs)
            if len(self.calls) == 2: result.stdout = self.output(False)
            return result
        with self.assertRaisesRegex(ValueError, 'outside its pinned path'): self.freeze(fallback)

    def test_same_bytes_different_system_path_rejected(self):
        other = self.root / 'other-system.so'
        other.write_bytes(self.system.read_bytes())
        def changed(argv, **kwargs):
            result = self.ldd(argv, **kwargs)
            if len(self.calls) == 2: result.stdout = result.stdout.replace(str(self.system), str(other))
            return result
        with self.assertRaisesRegex(ValueError, 'outside its pinned path'): self.freeze(changed)

    def test_relocated_extra_dependency_rejected(self):
        def extra(argv, **kwargs):
            result = self.ldd(argv, **kwargs)
            if len(self.calls) == 2: result.stdout += f'libextra.so => {self.system} (0x6)\n'
            return result
        with self.assertRaisesRegex(ValueError, 'dependency names differ'): self.freeze(extra)

    def test_new_directory_collision_or_unexpected_file_rejected(self):
        def extra(argv, **kwargs):
            result = self.ldd(argv, **kwargs)
            if len(self.calls) == 2: (self.destination / 'injected').write_bytes(b'extra')
            return result
        with self.assertRaisesRegex(ValueError, 'unexpected files'): self.freeze(extra)


if __name__ == '__main__':
    unittest.main()
