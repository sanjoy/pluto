"""CPU-only owner, copy-control, and runner failure tests; no native GPU work."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import paired_mlp_affine_run as runner
from . import paired_complement_localization_test as fixtures


class TemporaryFiles(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='mlp-affine-runner-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def file(self, relative, value):
        path = self.root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return runner.history.file_record(path)


class CompletionTest(TemporaryFiles):
    def setUp(self):
        super().setUp()
        self.previous = self.root/'previous'
        self.owner = dict(pid=123, start_ticks=456)
        self.complete = [f'{direction}/{cell}/{suite}' for direction in runner.DIRECTIONS
                         for cell in runner.previous_runner.CELL_ORDER for suite in runner.SUITES]
        self.archive = self.file('archive.json', {})
        self.prior = self.file('prior.json', {})
        self.request = dict(format=runner.previous_runner.FORMAT, runner_identity=self.owner,
            native_measurement_count=32, source_archive=self.archive, previous_summary=self.prior,
            frozen_inputs=[self.archive, self.prior])
        self.owner_record = self.file('previous/request.json', self.request)
        self.state = dict(format=runner.previous_runner.FORMAT, phase='complete', completed=self.complete,
                          runner_pid=123)
        self.file('previous/state.json', self.state)
        self.summary = dict(format=runner.previous_runner.FORMAT, complete=True,
            baseline_controls_certified=True, native_measurement_count=32, completed=self.complete,
            source_archive=self.archive, previous_summary=self.prior,
            frozen_inputs=[self.owner_record, self.archive, self.prior],
            training_restarted=False, goal_completion_claimed=False)
        self.republish()
        patch = mock.patch.object(runner.training, 'process_live', return_value=False)
        self.live = patch.start()
        self.addCleanup(patch.stop)

    def republish(self):
        self.summary['artifacts'] = [runner.history.file_record(p) for p in sorted(self.previous.rglob('*'))
                                     if p.is_file() and p.name != 'summary.json']
        self.summary_record = self.file('previous/summary.json', self.summary)

    def validate(self):
        return runner.validate_completed_run(self.previous, self.summary_record['sha256'])

    def test_complete_inventory_and_absent_exact_owner(self):
        result = self.validate()
        self.assertEqual(result['request'], self.request)
        self.live.assert_called_once_with(self.owner)

    def test_live_owner_rejected_despite_summary(self):
        self.live.return_value = True
        with self.assertRaisesRegex(RuntimeError, 'still live'):
            self.validate()

    def test_wrong_pin_extra_file_and_mutated_file_rejected(self):
        with self.assertRaises(ValueError):
            runner.validate_completed_run(self.previous, '0'*64)
        extra = self.previous/'extra'
        extra.write_text('unexpected')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            self.validate()
        extra.unlink()
        (self.previous/'state.json').write_text(json.dumps(dict(self.state, updated='changed')))
        with self.assertRaises(ValueError):
            self.validate()

    def test_incomplete_duplicate_or_wrong_owner_rejected_even_if_rehashed(self):
        changes = [lambda: self.summary.update(complete=False),
                   lambda: self.summary.update(completed=self.complete[:-1]),
                   lambda: self.summary.update(completed=self.complete[:-1]+[self.complete[0]]),
                   lambda: self.summary.update(native_measurement_count=16),
                   lambda: self.file('previous/state.json', dict(self.state, runner_pid=124)),
                   lambda: self.file('previous/state.json', dict(self.state, phase='native_measurement'))]
        for change in changes:
            with self.subTest(change=change):
                saved = copy.deepcopy(self.summary)
                change()
                self.republish()
                with self.assertRaises(ValueError):
                    self.validate()
                self.summary = saved
                self.file('previous/state.json', self.state)
                self.republish()

    def test_missing_frozen_owner_and_failure_rejected(self):
        self.summary['frozen_inputs'].remove(self.owner_record)
        self.republish()
        with self.assertRaisesRegex(ValueError, 'omitted'):
            self.validate()
        self.summary['frozen_inputs'].append(self.owner_record)
        self.file('previous/failure.json', {'error': 'failed'})
        self.republish()
        with self.assertRaises(ValueError):
            self.validate()


class CurrentCopyTest(unittest.TestCase):
    hashes = fixtures.ComplementPatchTest.hashes

    def setUp(self):
        fixtures.ComplementPatchTest.setUp(self)
        self.prior = runner.models.build_model(self.a, self.d, self.root/'prior', 'M', self.rows, fixtures.CONFIG)
        self.fresh = runner.models.build_model(self.a, self.d, self.root/'fresh', 'M', self.rows, fixtures.CONFIG)

    def test_current_sources_are_not_sent_to_historical_archive(self):
        self.assertIn(runner.models.SOURCE_RECORDS[0], self.prior['records'])
        with mock.patch.object(runner.history.SourceArchive, 'verify', side_effect=AssertionError('unknown historical path')):
            runner.validate_current_copy(self.fresh, self.prior, fixtures.CONFIG)

    def test_changed_selection_hash_source_or_same_path_rejected(self):
        mutations = [lambda x: x.update(selection=[]),
                     lambda x: x['hashes']['patched'].update(weight_0='bad'),
                     lambda x: x['paths'].update(donor='/wrong'),
                     lambda x: x['paths'].update(patched=self.prior['paths']['patched'])]
        for change in mutations:
            value = copy.deepcopy(self.fresh)
            change(value)
            with self.assertRaises(ValueError):
                runner.validate_current_copy(value, self.prior, fixtures.CONFIG)

    def test_hardlinked_copy_rejected_even_with_matching_bytes(self):
        path = Path(self.fresh['paths']['patched'])/'weight_0.bin'
        path.unlink()
        os.link(Path(self.prior['paths']['patched'])/'weight_0.bin', path)
        with self.assertRaisesRegex(ValueError, 'inode'):
            runner.validate_current_copy(self.fresh, self.prior, fixtures.CONFIG)


class RunnerTest(TemporaryFiles):
    """Mock native work and model construction, retaining actual file hash gates."""
    def setUp(self):
        super().setUp()
        self.previous = self.root/'previous'
        self.previous.mkdir()
        self.output = self.root/'new'
        self.summary = self.file('previous/summary.json', {})
        self.archive_record = self.file('archive.json', {})
        self.binary = self.file('probe', {})
        self.batch = self.file('batch', {})
        self.source = self.file('executing_source.py', {})
        self.cases = {suite: self.file(suite+'.json', dict(packed_batch=self.batch)) for suite in runner.SUITES}
        self.archive = object()
        references = {'E': {name: self.summary for name in ('main', 'word_next_native', 'shared_piece')}}
        self.plan = dict(cases=self.cases, gpu={'test': True}, binaries={'loss_probe': self.binary},
            runtime=dict(complete=True, runtime_dlopen_covered=False, environment={'LD_LIBRARY_PATH': '/fixture'}),
            source_archive=self.archive_record,
            directions={name: dict(recipient={'path': str(self.root/'recipient')},
                donor={'path': str(self.root/'donor')}, rows=[1, 2], references=references) for name in runner.DIRECTIONS})
        self.handoff = dict(request=self.plan, archive=self.archive, summary_record=self.summary,
            references={name: {cell: dict(model={'cell': cell},
                measured={suite: self.measured(suite) for suite in runner.SUITES}) for cell in runner.ANCHORS}
                for name in runner.DIRECTIONS},
            records=[self.summary, self.archive_record, self.binary, self.batch, *self.cases.values()])
        self.commands = []
        self.fail_cell = None
        self.selected = [{'fixture': 'native cases'}]
        patches = [mock.patch.dict(os.environ),
            mock.patch.object(runner, 'IMPORTED_SOURCES', [self.source]),
            mock.patch.object(runner, 'TEST_NAMES', ()),
            mock.patch.object(runner, 'validate_handoff', return_value=self.handoff),
            mock.patch.object(runner.models, 'build_model', side_effect=self.build),
            mock.patch.object(runner, 'validate_current_copy'),
            mock.patch.object(runner.screen, 'run_native', side_effect=self.execute),
            mock.patch.object(runner.core, 'load_native', side_effect=self.load),
            mock.patch.object(runner.core, 'compare_fp64_reference', return_value={'checked_predictions': 3}),
            mock.patch.object(runner.core, '_cross_suite'),
            mock.patch.object(runner.previous_reader.previous_partial, 'selected_cases', return_value=self.selected),
            mock.patch.object(runner.readout, 'aggregate', return_value=[{'checked': True}])]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        os.environ.pop('LD_PRELOAD', None)
        os.environ.pop('LD_AUDIT', None)

    def measured(self, suite):
        return dict(records=[], cases={'record': self.cases[suite]},
                    losses=np.zeros((2, 4), dtype='<f4'), argmax=np.zeros((2, 4), dtype='<i4'))

    def build(self, recipient, donor, output, cell, rows):
        output.mkdir()
        patch = output/'patch.json'
        patch.write_text('{}')
        return dict(paths=dict(recipient=recipient, donor=donor, patched=str(output)),
                    records=[runner.history.file_record(patch)], cell=cell)

    def execute(self, command, inputs, output, log, ledger, gpu):
        self.assertEqual(gpu, self.plan['gpu'])
        for path in (self.binary['path'], self.batch['path'], str(self.output/'request.json'), self.source['path']):
            self.assertIn(path, inputs)
        self.commands.append(command)
        output.mkdir()
        log.write_text('fixture only: no native GPU run\n')
        ledger.write_text(json.dumps({'command': command, 'inputs': inputs}))

    def load(self, model, cases, suite, output, execution, *, case_archive):
        self.assertIs(case_archive, self.archive)
        result = self.measured(suite)
        if model['cell'] == self.fail_cell:
            result['argmax'][1, 3] = 1  # Padding-only mismatch must still fail.
        return result

    def run_experiment(self):
        runner.run(self.previous, self.summary['sha256'], self.output)

    def test_full_run_checks_controls_before_new_cells(self):
        self.run_experiment()
        cells = [Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--checkpoint='))).parent.name
                 for command in self.commands]
        self.assertEqual(cells, [cell for _ in runner.DIRECTIONS for cell in runner.CELL_ORDER for _ in runner.SUITES])
        summary = runner.native._json(self.output/'summary.json')
        self.assertEqual(summary['native_measurement_count'], 16)
        self.assertEqual(len(set(summary['completed'])), 16)
        self.assertTrue(summary['complete'])
        self.assertFalse(summary['goal_completion_claimed'])
        self.assertFalse(summary['training_restarted'])
        self.assertFalse((self.output/'failure.json').exists())
        self.assertEqual(runner.validate_current_copy.call_count, 4)
        self.assertEqual(runner.core.compare_fp64_reference.call_count, 6)
        for direction in runner.DIRECTIONS:
            report = runner.native._json(self.output/direction/'readout.json')
            self.assertEqual(set(report['controls']), set(runner.ANCHORS))
            self.assertEqual(report['per_case'], self.selected)
        runner.readout.aggregate.assert_called_with(self.selected)

    def test_control_padding_mismatch_stops_before_new_cells(self):
        self.fail_cell = 'M'
        with self.assertRaisesRegex(ValueError, 'copy control differs'):
            self.run_experiment()
        self.assertEqual(len(self.commands), 3)  # E suites, then failing M/main.
        self.assertFalse((self.output/'summary.json').exists())
        self.assertTrue(runner.native._json(self.output/'failure.json')['no_automatic_restart'])

    def test_cross_suite_failure_stops_without_summary(self):
        runner.core._cross_suite.side_effect = ValueError('cross-suite differs')
        with self.assertRaisesRegex(ValueError, 'cross-suite'):
            self.run_experiment()
        self.assertEqual(len(self.commands), 2)
        self.assertFalse((self.output/'summary.json').exists())

    def test_existing_output_refused_before_handoff(self):
        self.output.mkdir()
        with self.assertRaises(ValueError):
            self.run_experiment()
        runner.validate_handoff.assert_not_called()

    def test_live_predecessor_failure_creates_no_output(self):
        runner.validate_handoff.side_effect = RuntimeError('previous process still live')
        with self.assertRaises(RuntimeError):
            self.run_experiment()
        self.assertFalse(self.output.exists())
        self.assertEqual(self.commands, [])

    def test_source_mutated_during_handoff_refused(self):
        def mutate(*args):
            Path(self.source['path']).write_text('mutated')
            return self.handoff
        runner.validate_handoff.side_effect = mutate
        with self.assertRaises(ValueError):
            self.run_experiment()
        self.assertFalse(self.output.exists())

    def test_native_failure_retains_partial_evidence_without_retry(self):
        runner.screen.run_native.side_effect = RuntimeError('native failed')
        with self.assertRaisesRegex(RuntimeError, 'native failed'):
            self.run_experiment()
        self.assertEqual(runner.screen.run_native.call_count, 1)
        self.assertFalse((self.output/'summary.json').exists())
        self.assertTrue((self.output/'failure.json').exists())

    def test_readout_failure_does_not_publish_success(self):
        runner.readout.aggregate.side_effect = ValueError('invalid causal score')
        with self.assertRaisesRegex(ValueError, 'invalid causal score'):
            self.run_experiment()
        self.assertEqual(len(self.commands), 8)
        self.assertFalse((self.output/'summary.json').exists())
        self.assertTrue((self.output/'failure.json').exists())


if __name__ == '__main__':
    unittest.main()
