"""CPU integration fixtures for explicit historical-source reader mode."""

import copy
from pathlib import Path
import unittest
from unittest import mock

from . import historical_source_archive_test as archive_fixtures
from . import paired_branch_readout as readout
from .paired_branch_readout_test import Fixture, write_json


class BranchArchiveTest(unittest.TestCase):
    def setUp(self):
        # Compose the fixture without inheriting or rediscovering its tests.
        self.source_fixture = archive_fixtures.SourceArchiveTest(methodName='runTest')
        self.source_fixture.setUp()
        self.addCleanup(self.source_fixture.doCleanups)
        self.archive = self.source_fixture.make()
        self.original = dict(self.source_fixture.original)
        self.physical = dict(self.source_fixture.rows[0]['archived'])
        root = self.source_fixture.root / 'branch'
        root.mkdir()
        self.fixture = Fixture(root, baseline_copies=False)
        self.bind_cases()
        self.execution = self.fixture.executions['patched']['main']
        self.directory = self.fixture.scores['patched']['main']
        self.metadata = readout._json(self.directory / 'metadata.json')
        self.run = readout._json(self.execution)
        self.run['inputs_before'].append(dict(self.original))
        self.run['inputs_after'] = copy.deepcopy(self.run['inputs_before'])
        write_json(self.execution, self.run)
        self.required_inputs = [item['path'] for item in self.run['inputs_before']]
        self.required_outputs = [item['path'] for item in self.run['outputs']]

    def bind_cases(self, *, amended=False):
        if amended:
            self.fixture.amend_capitalization()
        for suite in readout.SUITES:
            plan = self.fixture.plans[suite]
            plan['sources' if amended else 'source_inputs'] = [dict(self.original)]
            if amended:
                plan['provenance'] = [dict(self.source_fixture.external_record)]
            if suite == 'supplemental':
                plan['source_word_cases'] = readout._record(self.fixture.cases['main'])
            write_json(self.fixture.cases[suite], plan)
        for role in readout.ROLES:
            for suite in readout.SUITES:
                self.fixture.refresh_execution(role, suite)

    def load_execution(self, *, source_archive=True, inputs=None, outputs=None, records=None):
        return readout._execution(self.execution, self.metadata, self.directory,
            self.required_inputs if inputs is None else inputs,
            self.required_outputs if outputs is None else outputs,
            {} if records is None else records,
            source_archive=self.archive if source_archive else None)

    def publish_run(self):
        write_json(self.execution, self.run)

    def assert_archive_inventory(self, records):
        self.assertNotIn(self.original['path'], records)
        for expected in (self.physical, self.archive.manifest_record,
                         self.archive.owning_request_record):
            self.assertEqual(records[expected['path']], expected)

    def test_historical_register_preserves_identity_and_records_physical_provenance(self):
        records, expected = {}, copy.deepcopy(self.original)
        result = readout._historical_register(records, self.archive, expected['path'], expected)
        self.assertEqual(result, self.original)
        self.assertEqual(expected, self.original)
        self.assert_archive_inventory(records)
        self.assertEqual(len(records), 3)
        self.assertFalse(Path(self.original['path']).exists())

    def test_historical_register_rejects_wrong_requested_path(self):
        with self.assertRaisesRegex(ValueError, 'requested identity'):
            readout._historical_register({}, self.archive, self.physical['path'], self.original)

    def test_historical_register_rejects_conflicting_physical_inventory(self):
        records = {self.physical['path']: dict(self.physical, sha256='0' * 64)}
        with self.assertRaisesRegex(ValueError, 'changed during validation'):
            readout._historical_register(records, self.archive, self.original['path'], self.original)

    def test_legacy_cases_map_missing_producer_without_rewriting_saved_cases(self):
        for suite in readout.SUITES:
            path = self.fixture.cases[suite]
            before, records = path.read_bytes(), {}
            loaded = readout._cases(path, suite, records, self.fixture.config,
                                   source_archive=self.archive)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(loaded['plan'], self.fixture.plans[suite])
            self.assertEqual(loaded['plan']['source_inputs'], [self.original])
            self.assert_archive_inventory(records)

    def test_amended_cases_map_sources_and_keep_external_provenance(self):
        self.bind_cases(amended=True)
        for suite in readout.SUITES:
            path = self.fixture.cases[suite]
            before, records = path.read_bytes(), {}
            loaded = readout._cases(path, suite, records, self.fixture.config,
                                   source_archive=self.archive)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(loaded['plan']['sources'], [self.original])
            self.assertEqual(loaded['plan']['provenance'], [self.source_fixture.external_record])
            self.assertEqual(records[self.source_fixture.external_record['path']],
                             self.source_fixture.external_record)
            self.assert_archive_inventory(records)

    def test_cases_default_mode_rejects_missing_old_producer(self):
        with self.assertRaises(ValueError):
            readout._cases(self.fixture.cases['main'], 'main', {}, self.fixture.config)

    def test_cases_reject_archived_source_tampering(self):
        self.source_fixture.a.write_bytes(b'changed archived source')
        with self.assertRaisesRegex(ValueError, 'physical file differs'):
            readout._cases(self.fixture.cases['main'], 'main', {}, self.fixture.config,
                           source_archive=self.archive)

    def test_execution_keeps_old_required_memberships_and_physical_inventory(self):
        before, records = self.execution.read_bytes(), {}
        result = self.load_execution(records=records)
        self.assertEqual(result, readout._record(self.execution))
        self.assertEqual(self.execution.read_bytes(), before)
        self.assert_archive_inventory(records)
        for path in self.required_outputs:
            self.assertEqual(records[path], readout._record(path))

    def test_execution_default_mode_rejects_missing_old_source(self):
        with self.assertRaises(ValueError): self.load_execution(source_archive=False)

    def test_execution_batches_archive_audit_and_registers_pins_once(self):
        records = {}
        with mock.patch.object(self.archive, 'verify', wraps=self.archive.verify) as verify, \
                mock.patch.object(readout, '_register', wraps=readout._register) as register:
            self.load_execution(records=records)
        verify.assert_called_once_with(self.run['inputs_before'] + self.run['outputs'])
        paths = [str(call.args[1]) for call in register.call_args_list]
        for expected in (self.physical, self.archive.manifest_record,
                         self.archive.owning_request_record):
            self.assertEqual(paths.count(expected['path']), 1)
        self.assert_archive_inventory(records)

    def test_execution_rejects_missing_extra_and_reordered_archive_bindings(self):
        bindings = self.archive.verify(self.run['inputs_before'] + self.run['outputs'])
        cases = [(bindings[:-1], 'inventory length'),
                 (bindings + [bindings[0]], 'inventory length'),
                 (list(reversed(bindings)), 'historical identity')]
        for malformed, message in cases:
            with self.subTest(message=message), \
                    mock.patch.object(self.archive, 'verify', return_value=malformed) as verify:
                with self.assertRaisesRegex(ValueError, message):
                    self.load_execution()
                verify.assert_called_once()

    def test_execution_freshly_checks_physical_bytes_after_archive_audit(self):
        verify = self.archive.verify
        def change_after_verify(records):
            bindings = verify(records)
            self.source_fixture.a.write_bytes(b'changed after archive audit')
            return bindings
        with mock.patch.object(self.archive, 'verify', side_effect=change_after_verify) as checked:
            with self.assertRaisesRegex(ValueError, 'hash/size/path mismatch'):
                self.load_execution()
            checked.assert_called_once()

    def test_execution_physical_archive_path_cannot_replace_required_old_id(self):
        with self.assertRaisesRegex(ValueError, 'omits required inputs_before'):
            self.load_execution(inputs=[self.physical['path']])
        self.run['inputs_before'][-1] = dict(self.physical)
        self.run['inputs_after'] = copy.deepcopy(self.run['inputs_before'])
        self.publish_run()
        with self.assertRaisesRegex(ValueError, 'omits required inputs_before'):
            self.load_execution()

    def test_execution_rejects_missing_required_old_source(self):
        self.run['inputs_before'].pop()
        self.run['inputs_after'] = copy.deepcopy(self.run['inputs_before'])
        self.publish_run()
        with self.assertRaisesRegex(ValueError, 'omits required inputs_before'):
            self.load_execution()

    def test_execution_rejects_duplicate_historical_input(self):
        self.run['inputs_before'].append(dict(self.original))
        self.run['inputs_after'] = copy.deepcopy(self.run['inputs_before'])
        self.publish_run()
        with mock.patch.object(self.archive, 'verify', wraps=self.archive.verify) as verify:
            with self.assertRaisesRegex(ValueError, 'duplicate execution evidence'):
                self.load_execution()
            verify.assert_called_once_with(self.run['inputs_before'] + self.run['outputs'])

    def test_execution_rejects_duplicate_output(self):
        self.run['outputs'].append(dict(self.run['outputs'][0]))
        self.publish_run()
        with self.assertRaisesRegex(ValueError, 'duplicate execution evidence'):
            self.load_execution()

    def test_execution_rejects_before_after_identity_changes(self):
        self.run['inputs_after'][-1] = dict(self.original, sha256='0' * 64)
        self.publish_run()
        with self.assertRaisesRegex(ValueError, 'input hashes changed'):
            self.load_execution()

    def test_execution_rejects_wrong_historical_source_hash(self):
        self.run['inputs_before'][-1] = dict(self.original, sha256='0' * 64)
        self.run['inputs_after'] = copy.deepcopy(self.run['inputs_before'])
        self.publish_run()
        with mock.patch.object(self.archive, 'verify', wraps=self.archive.verify) as verify:
            with self.assertRaisesRegex(ValueError, 'archived original'):
                self.load_execution()
            verify.assert_called_once_with(self.run['inputs_before'] + self.run['outputs'])

    def test_execution_rejects_changed_native_command_paths(self):
        original = list(self.run['command'])
        for index, flag in ((1, 'checkpoint'), (2, 'batch'), (3, 'output_dir')):
            self.run['command'] = list(original)
            self.run['command'][index] = '--' + flag + '=/different/path'
            self.publish_run()
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'command path mismatch'):
                self.load_execution()

    def test_execution_rejects_binary_relabeling_even_with_same_bytes(self):
        other = self.fixture.root / 'identical_binary'
        other.write_bytes(self.fixture.binary.read_bytes())
        self.run['command'][0] = str(other)
        self.publish_run()
        with self.assertRaisesRegex(ValueError, 'execution binary mismatch'):
            self.load_execution()

    def test_execution_rejects_duplicate_unknown_and_changed_batching_flags(self):
        original = list(self.run['command'])
        cases = [(original + [original[2]], 'duplicate execution argument'),
                 (original + ['--unknown=1'], 'unexpected probe flags'),
                 (original[:-1] + ['--batch_sequences=2'], 'batching mismatch')]
        for command, message in cases:
            self.run['command'] = command
            self.publish_run()
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, message):
                self.load_execution()

    def test_execution_rejects_missing_output_inventory_member(self):
        self.run['outputs'].pop()
        self.publish_run()
        with self.assertRaisesRegex(ValueError, 'omits required outputs'):
            self.load_execution()

    def test_execution_rejects_changed_output_bytes(self):
        path = self.directory / 'losses.f32.bin'
        data = path.read_bytes()
        path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        with self.assertRaisesRegex(ValueError, 'physical file differs'):
            self.load_execution()

    def test_default_execution_still_checks_only_actual_current_paths(self):
        self.fixture.refresh_execution('patched', 'main')
        run = readout._json(self.execution)
        result = self.load_execution(source_archive=False,
                                     inputs=[row['path'] for row in run['inputs_before']])
        self.assertEqual(result, readout._record(self.execution))


if __name__ == '__main__':
    unittest.main()
