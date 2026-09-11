"""CPU-only runner tests: no real model execution or historical writes."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import paired_early_branches as runner
from . import paired_complement_localization as core
from . import paired_complement_localization_test as patch_tests

CONFIG = patch_tests.CONFIG


class IdentityArchive:
    """Local fixture files need no relocation; real archive behavior is separate."""
    def verify(self, records):
        runner.training.verify_records(records)
        return [dict(original=record, physical=record) for record in records]


class HistoricalModelTest(unittest.TestCase):
    # Reuse setup explicitly, without collecting inherited or imported tests.
    hashes = patch_tests.ComplementPatchTest.hashes

    def setUp(self):
        patch_tests.ComplementPatchTest.setUp(self)
        self.old = core.build_model(self.a, self.d, self.root/'old', ['H'], self.rows, CONFIG)
        self.fresh = core.build_model(self.a, self.d, self.root/'fresh', ['A', 'M'], self.rows,
                                      CONFIG, layout=runner.LAYOUT)

    def test_historical_adapter_preserves_paths_and_old_contract(self):
        result = runner.historical_model(self.fresh, self.old, IdentityArchive(), config=CONFIG)
        self.assertEqual(result['paths'], self.old['paths'])
        self.assertEqual(result['original_model_contract'], self.old)
        self.assertTrue(result['historical_only'])
        self.assertNotEqual(result['paths']['patched'], self.fresh['paths']['patched'])

    def test_adapter_rejects_wrong_selection_hashes_or_endpoints(self):
        mutations = [lambda value: value['selection'].pop(),
                     lambda value: value['hashes']['patched'].update({'weight_0.bin': '0'*64}),
                     lambda value: value['paths'].update(recipient='/another/model')]
        for change in mutations:
            value = copy.deepcopy(self.fresh)
            change(value)
            with self.assertRaises(ValueError):
                runner.historical_model(value, self.old, IdentityArchive(), config=CONFIG)

    def test_adapter_rejects_source_mutation(self):
        (Path(self.old['paths']['patched'])/'weight_0.bin').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            runner.historical_model(self.fresh, self.old, IdentityArchive(), config=CONFIG)

    def test_adapter_rejects_same_directory_or_shared_inode(self):
        with self.assertRaises(ValueError):
            runner.historical_model(self.old, self.old, IdentityArchive(), config=CONFIG)
        target = Path(self.fresh['paths']['patched'])/'weight_0.bin'
        target.unlink()
        os.link(Path(self.old['paths']['patched'])/'weight_0.bin', target)
        with self.assertRaisesRegex(ValueError, 'inode'):
            runner.historical_model(self.fresh, self.old, IdentityArchive(), config=CONFIG)


class InventoryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.file = self.root/'evidence'
        self.file.write_bytes(b'data')
        self.marker = self.root/'summary.json'
        self.marker.write_text('{}')
        self.records = [runner.history.file_record(self.file)]

    def test_inventory_and_caller_pin(self):
        runner._inventory(self.root, self.marker, self.records)
        self.assertEqual(runner._pinned(self.file, self.records[0]['sha256']), self.records[0])
        with self.assertRaises(ValueError): runner._pinned(self.file, '0'*64)

    def test_extra_missing_duplicate_or_linked_output_rejected(self):
        for records in ([], self.records*2):
            with self.assertRaises(ValueError): runner._inventory(self.root, self.marker, records)
        link = self.root/'link'
        link.symlink_to(self.file)
        with self.assertRaises(ValueError): runner._inventory(self.root, self.marker, self.records)


class RunnerTest(unittest.TestCase):
    """Mock native work only; use real files, state publication and hash gates."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='early-branch-runner-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.previous = self.root/'previous'
        self.previous.mkdir()
        self.output = self.root/'new'
        self.summary = self.file('previous/summary.json', {})
        self.archive_record = self.file('archive.json', {})
        self.binary = self.file('probe', {})
        self.batch = self.file('batch', {})
        self.source = self.file('executing_source.py', {})
        self.cases = {suite: self.file(suite+'.json', dict(packed_batch=self.batch)) for suite in runner.SUITES}
        self.archive = IdentityArchive()
        self.archive.manifest_record = self.archive_record
        self.archive.owning_request_record = self.summary
        references = {cell: {name: self.summary for name in ('main', 'word_next_native', 'shared_piece')}
                      for cell in ('E', 'EC')}
        self.plan = dict(cases=self.cases, gpu={'test': True}, binaries={'loss_probe': self.binary},
            runtime=dict(complete=True, runtime_dlopen_covered=False, environment={'LD_LIBRARY_PATH': '/fixture'}),
            directions={name: dict(recipient={'path': str(self.root/'recipient')},
                donor={'path': str(self.root/'donor')}, rows=[1, 2], references=references)
                for name in runner.DIRECTIONS})
        self.handoff = dict(plan=self.plan, archive=self.archive, bindings=[],
            physical=[self.summary, self.archive_record, self.binary, self.batch, *self.cases.values()],
            old_models={name: {old: {'label': old} for old in runner.ANCHORS.values()} for name in runner.DIRECTIONS},
            summary_record=self.summary, archive_record=self.archive_record)
        self.commands = []
        self.readout_change = None
        self.baseline_change = False
        self.load_calls = []
        patches = [mock.patch.dict(os.environ),
                   mock.patch.object(runner, 'IMPORTED_SOURCES', [self.source]),
                   mock.patch.object(runner, 'TEST_NAMES', ()),
                   mock.patch.object(runner, 'validate_handoff', return_value=self.handoff),
                   mock.patch.object(core, 'build_model', side_effect=self.build),
                   mock.patch.object(runner, 'historical_model', side_effect=lambda fresh, old, archive: fresh),
                   mock.patch.object(runner.screen, 'run_native', side_effect=self.execute),
                   mock.patch.object(core, 'load_native', side_effect=self.load),
                   mock.patch.object(core, 'compare_fp64_reference', return_value={'checked_predictions': 3}),
                   mock.patch.object(core, '_cross_suite'),
                   mock.patch.object(core, 'analyze_cube', side_effect=self.readout)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        os.environ.pop('LD_PRELOAD', None)
        os.environ.pop('LD_AUDIT', None)

    def file(self, relative, value):
        path = self.root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return runner.history.file_record(path)

    def build(self, recipient, donor, output, subset, rows, *, layout):
        self.assertEqual(layout, runner.LAYOUT)
        output.mkdir()
        patch = output/'patch.json'
        patch.write_text('{}')
        return dict(paths=dict(recipient=recipient, donor=donor, patched=str(output)),
                    records=[runner.training.record(patch)], subset=list(subset))

    def execute(self, command, inputs, output, log, ledger, gpu):
        self.assertEqual(gpu, self.plan['gpu'])
        self.assertIn(self.binary['path'], inputs)
        self.assertIn(self.batch['path'], inputs)
        self.assertIn(str(self.output/'request.json'), inputs)
        self.assertIn(self.source['path'], inputs)
        self.commands.append(command)
        output.mkdir()
        log.write_text('fixture native execution, not a GPU run\n')
        ledger.write_text(json.dumps(dict(command=command, inputs=inputs)))

    def load(self, model, cases_path, suite, output, execution, **kwargs):
        historical = str(output).startswith(str(self.previous))
        self.assertIs(kwargs['case_archive'], self.archive)
        self.assertEqual('execution_archive' in kwargs, historical)
        self.load_calls.append((historical, suite))
        return dict(records=[], cases={'record': self.cases[suite]},
                    losses=np.zeros((2, 4), dtype='<f4'), argmax=np.zeros((2, 4), dtype='<i4'))

    def readout(self, models, cases, scores, ledgers, output, **kwargs):
        self.assertEqual(kwargs['layout'], runner.LAYOUT)
        self.assertIs(kwargs['case_archive'], self.archive)
        value = dict(format=core.EARLY_BRANCHES_FORMAT, layout=runner.LAYOUT, complete=True,
                     baseline_controls_certified=False, files=[],
                     cells={key:list(subset) for key,subset in core.cube_subsets(layout=runner.LAYOUT).items()})
        if self.readout_change:
            self.readout_change(value)
        runner.training.publish(output, value)
        if self.baseline_change:
            (output.parent/'baseline_controls.json').write_text('{"forged":true}')
        return value

    def run_experiment(self):
        runner.run(self.previous, self.archive_record['path'], self.summary['sha256'],
                   self.archive_record['sha256'], self.output)

    def test_complete_run_scores_all_cells_and_controls_before_new_cells(self):
        self.run_experiment()
        self.assertEqual(len(self.commands), 32)
        self.assertEqual(sum(historical for historical, _ in self.load_calls), 16)
        names = [Path(next(arg.split('=', 1)[1] for arg in cmd if arg.startswith('--checkpoint='))).parent.name
                 for cmd in self.commands]
        self.assertEqual(names, [cell for _ in runner.DIRECTIONS for cell in runner.CELL_ORDER for _ in runner.SUITES])
        result = runner.native._json(self.output/'summary.json')
        self.assertTrue(result['complete'])
        self.assertEqual(len(set(result['completed'])), 32)
        self.assertFalse(result['training_restarted'])
        self.assertFalse(result['goal_completion_claimed'])
        self.assertFalse((self.output/'failure.json').exists())
        for name in runner.DIRECTIONS:
            control = runner.native._json(self.output/name/'baseline_controls.json')
            self.assertEqual(set(control['controls']), set(runner.ANCHORS))

    def test_existing_or_nonsibling_output_refused_before_handoff(self):
        self.output.mkdir()
        with self.assertRaises(ValueError): self.run_experiment()
        runner.validate_handoff.assert_not_called()

    def test_mutated_executing_source_during_handoff_refused_before_creation(self):
        def mutate(*args):
            Path(self.source['path']).write_text('changed during handoff')
            return self.handoff
        runner.validate_handoff.side_effect = mutate
        with self.assertRaises(ValueError): self.run_experiment()
        self.assertFalse(self.output.exists())

    def test_native_failure_is_not_retried_or_marked_complete(self):
        runner.screen.run_native.side_effect = RuntimeError('native failed')
        with self.assertRaises(RuntimeError): self.run_experiment()
        self.assertEqual(runner.screen.run_native.call_count, 1)
        failure = runner.native._json(self.output/'failure.json')
        self.assertEqual(failure['completed'], [])
        self.assertTrue(failure['no_automatic_restart'])
        self.assertFalse((self.output/'summary.json').exists())

    def test_inherited_parity_failure_stops_before_next_measurement(self):
        with mock.patch.object(core, 'assert_native_copy_parity', side_effect=ValueError('not equal')):
            with self.assertRaises(ValueError): self.run_experiment()
        self.assertEqual(len(self.commands), 1)
        self.assertFalse((self.output/'summary.json').exists())

    def test_control_file_change_during_readout_cannot_be_certified(self):
        self.baseline_change = True
        with self.assertRaises(ValueError): self.run_experiment()
        self.assertFalse((self.output/runner.DIRECTIONS[0]/'complete.json').exists())
        self.assertFalse((self.output/'summary.json').exists())

    def test_partial_or_wrong_layout_readout_cannot_be_certified(self):
        self.readout_change = lambda result: result.update(complete=False)
        with self.assertRaisesRegex(ValueError, 'incomplete or misrouted'): self.run_experiment()
        self.assertFalse((self.output/'summary.json').exists())

    def test_injected_runtime_rejected_before_handoff(self):
        os.environ['LD_PRELOAD'] = '/untrusted.so'
        with self.assertRaises(ValueError): self.run_experiment()
        runner.validate_handoff.assert_not_called()


if __name__ == '__main__':
    unittest.main()
