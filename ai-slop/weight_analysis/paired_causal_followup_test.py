"""CPU safety tests: never invoke a CUDA test or native probe here."""

import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from . import paired_causal_followup as followup


class CausalFollowupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.summary = self.root / 'summary.json'
        self.identity = {'pid': 123, 'start_ticks': 456,
                         'argv': ['python', '-m', 'weight_analysis.paired_analysis']}
        self.live_check = mock.patch.object(followup, 'process_still_live', return_value=False)
        self.live_check.start()
        self.addCleanup(self.live_check.stop)

    def complete(self):
        followup.write_json(self.summary, {'format': 'pluto-paired-analysis-v1', 'complete': True})

    def test_waits_for_actual_scorer_exit_even_if_summary_is_present(self):
        self.complete()
        with mock.patch.object(followup.paired_analysis, '_process_identity',
                               side_effect=[self.identity, FileNotFoundError(), FileNotFoundError()]), \
             mock.patch.object(followup.time, 'sleep') as sleep:
            self.assertTrue(followup.wait_for_scorer(self.summary, self.identity)['complete'])
            self.assertEqual(sleep.call_args_list, [mock.call(30), mock.call(30)])

    def test_missing_handle_rechecked_but_never_restarted(self):
        with mock.patch.object(followup.paired_analysis, '_process_identity',
                               side_effect=FileNotFoundError()) as identity, \
             mock.patch.object(followup.time, 'sleep') as sleep, \
             mock.patch.object(followup.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(RuntimeError, 'missing'):
                followup.wait_for_scorer(self.summary, self.identity)
            self.assertEqual(identity.call_count, 4)
            self.assertEqual(sleep.call_count, 3)
            process.assert_not_called()

    def test_reused_pid_is_not_the_upstream_scorer(self):
        self.complete()
        for changed in ({'start_ticks': 999}, {'argv': ['unrelated']}):
            with self.subTest(changed=changed), mock.patch.object(
                    followup.paired_analysis, '_process_identity',
                    return_value={**self.identity, **changed}):
                with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                    followup.wait_for_scorer(self.summary, self.identity)

    def test_missing_handle_can_finish_during_observation_race(self):
        def sleep(_):
            if not self.summary.exists():
                self.complete()
        with mock.patch.object(followup.paired_analysis, '_process_identity',
                               side_effect=FileNotFoundError()), \
             mock.patch.object(followup.time, 'sleep', side_effect=sleep):
            self.assertTrue(followup.wait_for_scorer(self.summary, self.identity)['complete'])

    def test_one_failed_identity_read_does_not_mean_the_process_exited(self):
        self.complete()
        with mock.patch.object(followup.paired_analysis, '_process_identity',
                               side_effect=FileNotFoundError()), \
             mock.patch.object(followup, 'process_still_live', side_effect=[True, False, False]) as check, \
             mock.patch.object(followup.time, 'sleep') as sleep:
            self.assertTrue(followup.wait_for_scorer(self.summary, self.identity)['complete'])
        self.assertEqual(check.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_transient_observation_error_retries_the_same_handle(self):
        self.complete()
        with mock.patch.object(followup.paired_analysis, '_process_identity',
                               side_effect=[OSError('temporary'), FileNotFoundError(),
                                            FileNotFoundError()]) as identity, \
             mock.patch.object(followup.time, 'sleep') as sleep:
            self.assertTrue(followup.wait_for_scorer(self.summary, self.identity)['complete'])
        self.assertEqual(identity.call_args_list, [mock.call(123)] * 3)
        self.assertEqual(sleep.call_count, 2)

    def test_terminal_failure_is_not_success_even_with_summary(self):
        self.complete()
        followup.write_json(self.root / 'failure.json', {'error': 'bad hash'})
        with self.assertRaisesRegex(RuntimeError, 'upstream analysis failed'):
            followup.wait_for_scorer(self.summary, None)

    def test_completed_mode_and_invalid_summary_or_poll(self):
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            followup.wait_for_scorer(self.summary, None)
        followup.write_json(self.summary, {'format': 'pluto-paired-analysis-v1', 'complete': False})
        with self.assertRaisesRegex(ValueError, 'not complete'):
            followup.wait_for_scorer(self.summary, None)
        for interval in (0, -1, 61, float('nan')):
            with self.assertRaises(ValueError):
                followup.wait_for_scorer(self.summary, None, poll_seconds=interval)

    def test_execution_record_binds_bytes_and_output(self):
        incoming = self.root / 'input.bin'
        incoming.write_bytes(b'known input')
        output = self.root / 'outputs'
        output.mkdir()
        (output / 'metadata.json').write_bytes(b'{"complete":true}')
        fake = {'pid': 100, 'returncode': 0, 'command': ['fake'],
                'started_utc': 'start', 'finished_utc': 'finish'}
        with mock.patch.object(followup, 'execute', return_value=fake):
            result = followup.execute_recorded(['fake'], [incoming, incoming], output,
                self.root / 'run.log', self.root / 'execution.json')
        self.assertEqual(result['inputs_before'], result['inputs_after'])
        self.assertEqual(len(result['inputs_before']), 1)
        self.assertEqual(result['outputs'], [followup.record(output / 'metadata.json')])

    def test_execution_record_rejects_changed_input_and_creates_no_marker(self):
        incoming = self.root / 'input.bin'
        incoming.write_bytes(b'original')
        def change(*_):
            incoming.write_bytes(b'changed!')
            return {'pid': 100, 'returncode': 0}
        with mock.patch.object(followup, 'execute', side_effect=change):
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                followup.execute_recorded(['fake'], [incoming], self.root,
                    self.root / 'run.log', self.root / 'execution.json')
        self.assertFalse((self.root / 'execution.json').exists())

    def test_failed_child_and_interruption_are_reaped(self):
        failed = mock.Mock(pid=100)
        failed.wait.return_value = 2
        with mock.patch.object(followup.subprocess, 'Popen', return_value=failed):
            with self.assertRaisesRegex(RuntimeError, 'exited 2'):
                followup.execute(['fake'], self.root / 'failed.log')
        interrupted = mock.Mock(pid=101)
        interrupted.wait.side_effect = [KeyboardInterrupt(), subprocess.TimeoutExpired('fake', 30), 0]
        with mock.patch.object(followup.subprocess, 'Popen', return_value=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                followup.execute(['fake'], self.root / 'interrupted.log')
        interrupted.terminate.assert_called_once()
        interrupted.kill.assert_called_once()
        self.assertEqual(interrupted.wait.call_count, 3)

    def test_gpu_test_requires_actual_completed_cases_not_skips(self):
        result = {'tests': 1, 'failures': 0, 'errors': 0, 'disabled': 0,
                  'testsuites': [{'testsuite': [{'status': 'RUN', 'result': 'COMPLETED'}]}]}
        for index, variant in enumerate((result, {**result, 'tests': 0},
            {**result, 'disabled': 1}, {**result, 'testsuites': []},
            {**result, 'testsuites': [{'testsuite': [{'status': 'RUN', 'result': 'SKIPPED'}]}]})):
            path = self.root / f'test_{index}.json'
            followup.write_json(path, variant)
            if index == 0:
                followup.require_gpu_test_success(path)
            else:
                with self.assertRaises(ValueError):
                    followup.require_gpu_test_success(path)

    def runner_fixture(self):
        root = self.root / 'experiment'
        root.mkdir()
        followup.write_json(root / 'manifest.json', {})
        followup.write_json(root / 'state.json', {'phase': 'training_complete'})
        for name in ('word_cases', 'supplemental_cases'):
            directory = root / name
            directory.mkdir()
            (directory / 'packed.bin').write_bytes(b'fixture')
            followup.write_json(directory / 'cases.json',
                                {'packed_batch': followup.record(directory / 'packed.bin')})
        analysis = root / 'analysis_trajectory'
        analysis.mkdir()
        scores = {}
        for name in ('losses', 'argmax'):
            path = analysis / f'{name}.bin'
            path.write_bytes(b'baseline')
            scores[name] = followup.record(path)
        followup.write_json(analysis / 'behavior.json', {'scores': scores})
        checkpoint = {'step': 100, 'path': str(root / 'original/checkpoints/step_100'), 'sha256': {}}
        summary = {'format': 'pluto-paired-analysis-v1', 'complete': True,
                   'behavior': {checkpoint['path']: {'report': followup.record(analysis / 'behavior.json')}}}
        followup.write_json(analysis / 'summary.json', summary)
        plan = {'interventions': [
            {'name': 'copy', 'recipient_checkpoint': checkpoint, 'donor_checkpoint': checkpoint,
             'tensors': [], 'embedding_rows': []},
            {'name': 'rows', 'recipient_checkpoint': checkpoint, 'donor_checkpoint': checkpoint,
             'tensors': [], 'embedding_rows': [45]},
            {'name': 'later_branch', 'tensors': ['not-executed'], 'embedding_rows': []}],
            'exports': {}, 'analysis_summary': followup.record(analysis / 'summary.json')}
        for name, count in (('main', 3), ('word_next_native', 4)):
            path = root / f'{name}.rows'
            path.write_bytes(b'rows')
            plan['exports'][name] = {'rows_per_case': count, 'selected_rows': followup.record(path),
                'cases_json': followup.record(root / 'word_cases/cases.json'),
                'packed_batch': followup.record(root / 'supplemental_cases/packed.bin')}
        binary = self.root / 'fake_binary'
        binary.write_bytes(b'never execute this')
        args = SimpleNamespace(root=root, output=self.root / 'followup', observer_pid=None,
            observer_start_ticks=None, loss_probe=binary, factorial_probe=binary, gpu_test=binary)
        return args, plan

    def test_full_embedding_stage_has_gate_copy_controls_and_measured_baselines(self):
        from . import paired_intervention_plan, embedding_factorial_readout
        args, plan = self.runner_fixture()
        calls = []

        def prepare(summary, main, supplemental, output):
            calls.append('plan')
            output.mkdir()
            followup.write_json(output / 'plan.json', plan)
            return plan

        def patch(recipient, donor, output, **kwargs):
            self.assertEqual(output.name, 'step_100')
            output.mkdir()
            followup.write_json(output / 'patch.json', kwargs)
            return kwargs

        def execute(command, inputs, output, log, execution):
            calls.append(Path(command[0]).name)
            output.mkdir(exist_ok=True)
            if Path(command[0]).name == 'gpu_test':
                followup.write_json(output / 'gtest.json', {'tests': 1, 'failures': 0,
                    'errors': 0, 'disabled': 0, 'testsuites': [{'testsuite': [
                        {'status': 'RUN', 'result': 'COMPLETED'}]}]})
            elif Path(command[0]).name == 'loss_probe':
                checkpoint_flag = next(arg for arg in command if arg.startswith('--checkpoint='))
                self.assertEqual(Path(checkpoint_flag.split('=', 1)[1]).name, 'step_100')
                (output / 'losses.f32.bin').write_bytes(b'baseline')
                (output / 'argmax.i32.bin').write_bytes(b'baseline')
            followup.write_json(execution, {'returncode': 0})
            return {'returncode': 0}

        def summarize(cases, scores, output, *args, **kwargs):
            followup.write_json(output, {'ok': True})

        with mock.patch.object(followup.paired_training, 'verify_frozen_inputs'), \
             mock.patch.object(followup.paired_analysis, 'validate_completion',
                               return_value={'initial': {}}) as completion, \
             mock.patch.object(followup.paired_analysis, 'validate_determinism_gate') as gate, \
             mock.patch.object(followup.paired_analysis, '_verify_checkpoint'), \
             mock.patch.object(paired_intervention_plan, 'prepare', side_effect=prepare), \
             mock.patch.object(followup.paired_weight_patch, 'create_patch', side_effect=patch), \
             mock.patch.object(followup, 'execute_recorded', side_effect=execute), \
             mock.patch.object(followup.paired_supplemental_cases, 'summarize', side_effect=summarize) as supplemental, \
             mock.patch.object(embedding_factorial_readout, 'analyze', side_effect=
                 lambda cases, scores, patch, output, **kwargs:
                     summarize(cases, scores, output)) as factorial:
            result = followup.run(args)
        self.assertEqual(calls, ['plan', 'gpu_test', 'loss_probe', 'loss_probe',
                                 'factorial_probe', 'factorial_probe', 'loss_probe'])
        completion.assert_called_once()
        gate.assert_called_once()
        self.assertEqual(supplemental.call_count, 2)  # Unmodified AND patched baseline.
        self.assertEqual(factorial.call_count, 2)
        self.assertEqual(factorial.call_args_list[1].kwargs['case_kind'], 'word_next_native')
        self.assertTrue(result['complete'])
        self.assertFalse(result['goal_completion_claimed'])
        self.assertEqual(len(result['deferred_interventions']), 1)
        self.assertTrue(result['completed'][0]['copy_control_all_scores_byte_equal'])

    def test_failed_upstream_cannot_reach_gpu_execution(self):
        args, _ = self.runner_fixture()
        with mock.patch.object(followup.paired_training, 'verify_frozen_inputs'), \
             mock.patch.object(followup, 'wait_for_scorer', side_effect=RuntimeError('failed upstream')), \
             mock.patch.object(followup, 'execute_recorded') as execute:
            with self.assertRaisesRegex(RuntimeError, 'failed upstream'):
                followup.run(args)
            execute.assert_not_called()
        self.assertTrue((args.output / 'failure.json').is_file())
        self.assertFalse((args.output / 'summary.json').exists())


if __name__ == '__main__':
    unittest.main()
