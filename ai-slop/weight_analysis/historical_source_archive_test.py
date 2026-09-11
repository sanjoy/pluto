"""Synthetic CPU fixtures; no real archive, checkpoint, or GPU is modified."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import historical_source_archive as archive


class SourceArchiveTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.snapshot = self.root / 'snapshot'
        self.sources = self.snapshot / 'repository/scripts/weight_analysis'
        self.sources.mkdir(parents=True)
        self.a, self.b = self.sources / 'a.py', self.sources / 'b.py'
        self.a.write_bytes(b'old source A\n')
        self.b.write_bytes(b'old source B\n')
        self.external = self.root / 'evidence.bin'
        self.external.write_bytes(b'fixed external evidence')
        self.rows = []
        for path in (self.a, self.b):
            physical = archive.file_record(path)
            original = dict(physical, path=str(self.repo / 'scripts/weight_analysis' / path.name))
            self.rows.append(dict(original=original, archived=physical))
        self.original = self.rows[0]['original']
        self.external_record = archive.file_record(self.external)
        self.request = self.root / 'run/request.json'
        self.request.parent.mkdir()
        self.request.write_text(json.dumps({'frozen_inputs': [self.original, self.external_record]}))
        self.owner = archive.file_record(self.request)
        self.manifest_path = self.snapshot / 'manifest.json'
        self.document = dict(format=archive.FORMAT, files=self.rows,
            repository=str(self.repo), package_file_count=2, frozen_repository_input_count=1,
            owning_run_request=self.owner, all_frozen_repository_inputs_preserved=True,
            original_paths_retained_in_metadata=True, original_sources_moved=False,
            analysis_completion_claimed=False, source_bytes_unchanged=True)
        self.publish_manifest()

    def publish_manifest(self):
        self.manifest_path.write_text(json.dumps(self.document))
        self.manifest_record = archive.file_record(self.manifest_path)

    def make(self, **kwargs):
        arguments = dict(expected_manifest_record=self.manifest_record,
                         owning_request_record=self.owner)
        arguments.update(kwargs)
        return archive.SourceArchive(self.manifest_path, **arguments)

    def test_exact_original_ids_and_physical_bindings(self):
        verifier = self.make()
        originals = [dict(self.original), dict(self.external_record)]
        expected = copy.deepcopy(originals)
        bindings = verifier.verify(originals)
        self.assertEqual(originals, expected)
        self.assertEqual(bindings, [dict(original=self.original, physical=self.rows[0]['archived']),
                                    dict(original=self.external_record, physical=self.external_record)])
        self.assertFalse(Path(self.original['path']).exists())
        self.assertEqual(verifier.physical_records(originals), [r['physical'] for r in bindings])

    def test_pins_and_results_are_copies(self):
        verifier = self.make()
        verifier.manifest_record['sha256'] = '0' * 64
        verifier.owning_request_record['bytes'] = 0
        result = verifier.verify([self.original])
        result[0]['original']['bytes'] = 0
        result[0]['physical']['bytes'] = 0
        self.assertEqual(verifier.manifest_record, self.manifest_record)
        self.assertEqual(verifier.owning_request_record, self.owner)
        self.assertEqual(verifier.verify([self.original])[0]['original'], self.original)

    def test_manifest_pin_and_owning_request_pin_are_required(self):
        with self.assertRaises(TypeError): archive.SourceArchive(self.manifest_path)
        with self.assertRaises(TypeError):
            archive.SourceArchive(self.manifest_path, expected_manifest_record=self.manifest_record)

    def test_wrong_manifest_pin_or_path(self):
        with self.assertRaises(ValueError):
            self.make(expected_manifest_record=dict(self.manifest_record, sha256='0' * 64))
        identical = self.snapshot / 'same-bytes.json'
        identical.write_bytes(self.manifest_path.read_bytes())
        with self.assertRaisesRegex(ValueError, 'manifest path'):
            self.make(expected_manifest_record=archive.file_record(identical))

    def test_wrong_owning_request_even_with_identical_bytes(self):
        identical = self.root / 'different-request.json'
        identical.write_bytes(self.request.read_bytes())
        with self.assertRaisesRegex(ValueError, 'owning request'):
            self.make(owning_request_record=archive.file_record(identical))
        self.request.write_bytes(b'changed request')
        with self.assertRaises(ValueError): self.make()

    def test_unknown_repo_record_never_uses_current_source(self):
        current = self.repo / 'ai-slop/weight_analysis/a.py'
        current.parent.mkdir(parents=True)
        current.write_bytes(self.a.read_bytes())
        with self.assertRaisesRegex(ValueError, 'unknown historical repository'):
            self.make().verify([archive.file_record(current)])

    def test_same_size_wrong_logical_identity_or_hash(self):
        wrong = dict(self.original, path=self.rows[1]['original']['path'])
        with self.assertRaisesRegex(ValueError, 'archived original'):
            self.make().verify([wrong])
        with self.assertRaises(ValueError):
            self.make().verify([dict(self.original, sha256='0' * 64)])

    def test_conflicting_and_malformed_input_records(self):
        verifier = self.make()
        with self.assertRaisesRegex(ValueError, 'conflicting historical'):
            verifier.verify([self.original, dict(self.original, sha256='0' * 64)])
        for value in (dict(self.original, bytes=True), dict(self.original, extra=1),
                      dict(self.original, path='relative.py'), dict(self.original, sha256='ABC')):
            with self.subTest(value=value), self.assertRaises(ValueError): verifier.verify([value])

    def test_identical_duplicates_are_verified_once_per_call(self):
        verifier = self.make()
        with mock.patch.object(archive, '_read', wraps=archive._read) as reader:
            bindings = verifier.verify([self.original, self.original])
        self.assertEqual(bindings[0], bindings[1])
        self.assertEqual(sum(call.args[0] == str(self.a) for call in reader.call_args_list), 1)
        self.a.write_bytes(b'changed')
        with self.assertRaises(ValueError): verifier.verify([self.original])

    def test_malformed_mapping_and_declarations(self):
        base = copy.deepcopy(self.document)
        changes = [
            lambda d: d.update(format='other'),
            lambda d: d.update(package_file_count=3),
            lambda d: d.update(frozen_repository_input_count=2),
            lambda d: d.update(source_bytes_unchanged=False),
            lambda d: d.update(analysis_completion_claimed=True),
            lambda d: d['files'][0].pop('archived'),
            lambda d: d['files'][0].update(extra='unexpected'),
            lambda d: d['files'][0]['archived'].update(bytes=0),
            lambda d: d['files'][0]['archived'].update(sha256='0' * 64),
            lambda d: d['files'][0]['archived'].update(path=str(self.external)),
            lambda d: d['files'][0]['archived'].update(path='relative.py'),
            lambda d: d['files'][0]['original'].update(path=str(self.root/'outside.py')),
            lambda d: d['files'][0]['original'].update(path=str(self.repo/'scripts')+'/../a.py'),
            lambda d: d['files'][0]['original'].update(path=str(self.repo)+'//a.py'),
            lambda d: d['files'].__setitem__(1, copy.deepcopy(d['files'][0])),
            lambda d: d['files'][1].update(archived=copy.deepcopy(d['files'][0]['archived'])),
        ]
        for change in changes:
            self.document = copy.deepcopy(base)
            change(self.document)
            self.publish_manifest()
            with self.subTest(document=self.document), self.assertRaises(ValueError): self.make()

    def test_frozen_repository_inputs_must_be_covered_exactly(self):
        self.document['files'] = [self.rows[1]]
        self.document['package_file_count'] = 1
        self.publish_manifest()
        with self.assertRaisesRegex(ValueError, 'not preserved exactly'): self.make()

    def test_duplicate_owning_request_input(self):
        self.request.write_text(json.dumps({'frozen_inputs': [self.original, self.original]}))
        self.owner = archive.file_record(self.request)
        self.document['owning_run_request'] = self.owner
        self.publish_manifest()
        with self.assertRaisesRegex(ValueError, 'duplicate owning-request'): self.make()

    def test_duplicate_json_keys_and_nonfinite_values(self):
        for contents in ('{"format":"a","format":"b"}', '{"unused":NaN}'):
            self.manifest_path.write_text(contents)
            self.manifest_record = archive.file_record(self.manifest_path)
            with self.subTest(contents=contents), self.assertRaises(ValueError): self.make()

    def test_archived_source_hash_is_checked_on_demand(self):
        self.a.write_bytes(b'old source X\n')
        verifier = self.make()
        with self.assertRaisesRegex(ValueError, 'physical file differs'):
            verifier.verify([self.original])

    def test_source_replaced_by_symlink_after_construction(self):
        verifier = self.make()
        target = self.root / 'same-source.py'
        target.write_bytes(self.a.read_bytes())
        self.a.unlink()
        self.a.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'symlink'): verifier.verify([self.original])

    def test_archive_parent_symlink_before_and_after_construction(self):
        verifier = self.make()
        moved = self.sources.with_name('actual_sources')
        self.sources.rename(moved)
        self.sources.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'): self.make()
        with self.assertRaisesRegex(ValueError, 'symlink'): verifier.verify([self.original])

    def test_manifest_and_owner_parent_symlinks(self):
        verifier = self.make()
        moved = self.request.parent.with_name('actual_run')
        self.request.parent.rename(moved)
        self.request.parent.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'): verifier.verify([])
        with self.assertRaisesRegex(ValueError, 'symlink'): self.make()

    def test_manifest_parent_symlink(self):
        verifier = self.make()
        moved = self.snapshot.with_name('actual_snapshot')
        self.snapshot.rename(moved)
        self.snapshot.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'): verifier.verify([])

    def test_nonrepository_file_keeps_its_original_location(self):
        verifier = self.make()
        self.external.write_bytes(b'changed evidence')
        with self.assertRaisesRegex(ValueError, 'physical file differs'):
            verifier.verify([self.external_record])

    def test_source_edit_during_hashing(self):
        real_hash = hashlib.sha256
        source = self.a

        class EditingHash:
            def __init__(self): self.digest = real_hash()
            def update(self, block):
                self.digest.update(block)
                source.write_bytes(b'old source X\n')
            def hexdigest(self): return self.digest.hexdigest()

        with mock.patch.object(archive.hashlib, 'sha256', EditingHash):
            with self.assertRaisesRegex(ValueError, 'changed while hashing'):
                archive.file_record(source)

    def test_equal_size_source_edit_with_coalesced_timestamps(self):
        real_hash, real_signature = hashlib.sha256, archive._signature
        source = self.a
        original_signature = real_signature(source.stat())

        class EditingHash:
            def __init__(self): self.digest = real_hash()
            def update(self, block):
                self.digest.update(block)
                source.write_bytes(b'old source X\n')
            def hexdigest(self): return self.digest.hexdigest()

        def coalesced_timestamps(info):
            values = real_signature(info)
            return values[:-2] + original_signature[-2:]

        with mock.patch.object(archive.hashlib, 'sha256', EditingHash), \
             mock.patch.object(archive, '_signature', side_effect=coalesced_timestamps):
            with self.assertRaisesRegex(ValueError, 'content changed while hashing'):
                archive.file_record(source)
        self.assertEqual(source.stat().st_size, original_signature[3])

    def test_second_streaming_pass_keeps_first_pass_json_once(self):
        path = self.root / 'multichunk.json'
        contents = json.dumps({'value': 'a' * (1024 * 1024 + 11)}).encode()
        path.write_bytes(contents)
        record, retained, _ = archive._read(path, retain=True)
        self.assertEqual(retained, contents)
        self.assertEqual(record, dict(path=str(path), bytes=len(contents),
                                     sha256=hashlib.sha256(contents).hexdigest()))

    def test_same_bytes_replacement_during_hashing(self):
        real_hash = hashlib.sha256
        source, replacement = self.a, self.a.with_name('replacement.py')

        class ReplacingHash:
            def __init__(self): self.digest = real_hash()
            def update(self, block):
                self.digest.update(block)
                replacement.write_bytes(block)
                replacement.replace(source)
            def hexdigest(self): return self.digest.hexdigest()

        with mock.patch.object(archive.hashlib, 'sha256', ReplacingHash):
            with self.assertRaisesRegex(ValueError, 'changed while hashing'):
                archive.file_record(source)

    def test_source_edit_later_in_verification_pass(self):
        verifier, read = self.make(), archive._read

        def change_after_external(path, **kwargs):
            result = read(path, **kwargs)
            if str(path) == str(self.external): self.a.write_bytes(b'old source X\n')
            return result

        with mock.patch.object(archive, '_read', side_effect=change_after_external):
            with self.assertRaisesRegex(ValueError, 'changed during verification'):
                verifier.verify([self.original, self.external_record])

    def test_manifest_or_owner_changed_during_verification(self):
        for target in (self.manifest_path, self.request):
            verifier, read = self.make(), archive._read
            contents = target.read_bytes()

            def change_after_source(path, **kwargs):
                result = read(path, **kwargs)
                if str(path) == str(self.a): target.write_bytes(contents + b' ')
                return result

            with mock.patch.object(archive, '_read', side_effect=change_after_source):
                with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'pinned file changed'):
                    verifier.verify([self.original])
            target.write_bytes(contents)

    def test_missing_or_nonregular_file(self):
        with self.assertRaises(ValueError): archive.file_record(self.root / 'missing')
        with self.assertRaises(ValueError): archive.file_record(self.root)
        pipe = self.root / 'pipe'
        os.mkfifo(pipe)
        with self.assertRaises(ValueError): archive.file_record(pipe)


if __name__ == '__main__':
    unittest.main()
