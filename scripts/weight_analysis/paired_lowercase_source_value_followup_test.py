"""CPU-only amended handoff tests; native/GPU execution is always mocked.

Tiny immutable artifact trees exercise the new queue adapter. The existing
planner's extensive tests own native corpus/checkpoint validation; here its
fresh regeneration is mocked explicitly, never mistaken for real training.
The unchanged historical readout has its own independent numerical tests.
"""

from contextlib import ExitStack, contextmanager
import copy
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from . import paired_lowercase_source_value_followup as followup

training = followup.training


def identity(root, upstream):
    return dict(pid=123, start_ticks=456, state='S', parent_pid=100,
                argv=['python', '-m', followup.MODULE, '--root', str(root),
                      '--output', str(upstream)])


def complete_summary():
    return dict(format=followup.combined.FORMAT, complete=True,
                goal_completion_claimed=False)


class WaitingTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.path = self.root / 'summary.json'
        self.identity = identity(self.root, self.root / 'causal')
        self.sleep = self.enterContext(mock.patch.object(followup.time, 'sleep'))
        self.spawn = self.enterContext(mock.patch.object(followup.historical.causal, 'execute'))

    def test_completion_waits_for_two_consecutive_exits(self):
        training.publish(self.path, complete_summary())
        with mock.patch.object(training, 'process_live', side_effect=[True, False, True, False, False]) as live:
            self.assertEqual(followup.wait_for_causal(self.path, self.identity), complete_summary())
        self.assertEqual(live.call_count, 5)
        self.spawn.assert_not_called()

    def test_transient_error_resets_exit_count(self):
        training.publish(self.path, complete_summary())
        with mock.patch.object(training, 'process_live', side_effect=[False, OSError('read'), False, False]) as live:
            followup.wait_for_causal(self.path, self.identity)
        self.assertEqual(live.call_count, 4)

    def test_missing_summary_or_repeated_observation_failures_fail_closed(self):
        for observed in (False, OSError('read'), ValueError('bad identity')):
            with self.subTest(observed=observed), mock.patch.object(training, 'process_live',
                    **({'side_effect': observed} if isinstance(observed, Exception) else {'return_value': observed})) as live:
                with self.assertRaises(RuntimeError):
                    followup.wait_for_causal(self.path, self.identity)
                self.assertEqual(live.call_count, 4)
        self.spawn.assert_not_called()

    def test_pid_reuse_and_failure_marker_never_unlock_gpu(self):
        training.publish(self.path, complete_summary())
        with mock.patch.object(training, 'process_live', side_effect=RuntimeError('PID reused')):
            with self.assertRaisesRegex(RuntimeError, 'PID reused'):
                followup.wait_for_causal(self.path, self.identity)
        training.publish(self.root / 'failure.json', {})
        with mock.patch.object(training, 'process_live') as live:
            with self.assertRaisesRegex(RuntimeError, 'causal stage failed'):
                followup.wait_for_causal(self.path, self.identity)
            live.assert_not_called()
        self.spawn.assert_not_called()

    def test_wrong_summary_and_bad_poll_interval_rejected(self):
        for change in ({'format': 'legacy'}, {'complete': 1}, {'goal_completion_claimed': True}):
            training.publish(self.path, {**complete_summary(), **change}, exclusive=False)
            with mock.patch.object(training, 'process_live', return_value=False), self.assertRaises(ValueError):
                followup.wait_for_causal(self.path, self.identity)
        for interval in (0, -1, 61, float('nan')):
            with self.assertRaises(ValueError):
                followup.wait_for_causal(self.path, self.identity, poll_seconds=interval)


class Fixture(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / 'lowercase_amendment'
        self.root.mkdir()
        self.upstream = self.root / 'causal_stage'
        self.upstream.mkdir()
        self.identity = identity(self.root, self.upstream)
        self.device = dict(index='0', uuid='GPU-fixture', name='fixture', driver_version='fixture')
        training.publish(self.root / 'request.json', dict(format=training.FORMAT,
            amendment_root=str(self.root), legacy_root=str(self.root.parent),
            frozen_inputs=[], gpu=self.device))
        self.bound = self.root / 'bound.txt'
        self.bound.write_bytes(b'frozen source fixture')
        self.request = dict(format=followup.combined.FORMAT, stage='embedding_and_branches',
            amendment_root=str(self.root), runner_identity=self.identity,
            frozen_inputs=[training.record(self.bound)], goal_completion_claimed=False)
        training.publish(self.upstream / 'request.json', self.request)

    def prepare_completed(self):
        self.plan = dict(format=followup.planner.FORMAT, complete=True,
            goal_completion_claimed=False, patches_materialized=False,
            arm_roots={'original': str(self.root.parent), 'replacement': str(self.root)},
            frozen_inputs=[training.record(self.bound)], implementation=[], interventions=[
                dict(name='copy', kind='copy_control', tensors=[]),
                dict(name='rows', kind='embedding_rows', tensors=[]),
                dict(name='branch', kind='branch', tensors=[4])])
        (self.upstream / 'plan').mkdir()
        training.publish(self.upstream / 'plan' / 'plan.json', self.plan)
        completed, artifacts = [], []
        for intervention in self.plan['interventions']:
            directory = self.upstream / intervention['name']
            directory.mkdir()
            (directory / 'result.bin').write_bytes(intervention['name'].encode())
            item = dict(intervention=intervention,
                        artifacts=[training.record(directory / 'result.bin')])
            training.publish(directory / 'complete.json', item)
            completed.append(item)
            artifacts.extend([*item['artifacts'], training.record(directory / 'complete.json')])
        tests = self.upstream / 'gpu_validation'
        tests.mkdir()
        self.write_gtest(tests / 'gtest.json')
        log = self.upstream / 'gpu_test.log'
        log.write_text('synthetic success, not a GPU run')
        test = dict(returncode=0, inputs_before=[training.record(self.bound)],
                    outputs=[training.record(tests / 'gtest.json')], log=training.record(log))
        training.publish(self.upstream / 'gpu_test_execution.json', test)
        self.summary = dict(**complete_summary(), plan=training.record(self.upstream / 'plan' / 'plan.json'),
            frozen_inputs=[*self.request['frozen_inputs'], training.record(self.upstream / 'plan' / 'plan.json')],
            completed=completed, completed_artifact_records=artifacts, gpu_test=test)
        training.publish(self.upstream / 'summary.json', self.summary)

    @staticmethod
    def write_gtest(path, skipped=False):
        case = dict(status='RUN', result='COMPLETED')
        if skipped:
            case['skipped'] = {}
        training.publish(path, dict(tests=1, failures=0, errors=0, disabled=0,
                                    testsuites=[dict(testsuite=[case])]))

    def regenerate(self, root, summary, exports, output):
        self.assertEqual(root, self.root)
        self.assertEqual(summary, self.root / 'analysis_trajectory' / 'summary.json')
        self.assertEqual(exports, self.root / 'causal_cases' / 'exports.json')
        output.mkdir()
        training.publish(output / 'plan.json', self.plan)
        return copy.deepcopy(self.plan)


class IdentityTest(Fixture):
    def test_source_freeze_includes_adapter_when_executed_as_main(self):
        with mock.patch.dict(sys.modules):
            sys.modules.pop(followup.__name__)
            self.assertNotIn(followup.__name__, sys.modules)
            self.assertIn(Path(followup.__file__).resolve(), followup.source_paths())

    def test_exact_identity_root_output_and_request_bound(self):
        with mock.patch.object(training, 'process_identity', return_value=self.identity):
            self.assertEqual(followup.observer_identity(self.root, self.upstream, 123, 456, self.request), self.identity)
            for request in ({**self.request, 'stage': 'branches'},
                            {**self.request, 'amendment_root': str(self.root.parent)},
                            {**self.request, 'runner_identity': {**self.identity, 'start_ticks': 999}}):
                with self.assertRaises(ValueError):
                    followup.observer_identity(self.root, self.upstream, 123, 456, request)

    def test_changed_live_identity_or_malformed_argv_rejected(self):
        for changed in ({'start_ticks': 999}, {'state': 'Z'}, {'argv': ['python', '-m']},
                        {'argv': self.identity['argv'] + ['--root', str(self.root)]},
                        {'argv': self.identity['argv'][:-1]},
                        {'argv': self.identity['argv'][:-1] + [str(self.root / 'wrong')]}):
            actual = {**self.identity, **changed}
            request = {**self.request, 'runner_identity': actual}
            with self.subTest(changed=changed), mock.patch.object(training, 'process_identity', return_value=actual), self.assertRaises(ValueError):
                followup.observer_identity(self.root, self.upstream, 123, 456, request)


class UpstreamTest(Fixture):
    def setUp(self):
        super().setUp()
        self.prepare_completed()
        self.output = self.root / 'revalidated'
        self.regeneration = self.enterContext(mock.patch.object(followup.planner, 'prepare', side_effect=self.regenerate))
        self.native = self.enterContext(mock.patch.object(followup.historical.causal, 'execute'))

    def validate(self):
        # Deliberately modified manifests in rejection tests are the actual
        # published input, not an unbound replacement of the parsed object.
        training.publish(self.upstream / 'summary.json', self.summary, exclusive=False)
        return followup.validate_upstream(self.root, self.upstream, self.summary, self.output)

    def test_changed_summary_since_handoff_is_rejected_before_regeneration(self):
        training.publish(self.upstream / 'summary.json', {'changed': True}, exclusive=False)
        with self.assertRaisesRegex(ValueError, 'changed since confirmed handoff'):
            followup.validate_upstream(self.root, self.upstream, self.summary, self.output)
        self.regeneration.assert_not_called()

    def test_complete_handoff_revalidates_plan_and_all_actual_artifacts(self):
        records = self.validate()
        training.verify_records(records)
        self.assertIn(training.record(self.upstream / 'summary.json'), records)
        self.assertIn(training.record(self.output / 'plan.json'), records)
        self.regeneration.assert_called_once()
        self.native.assert_not_called()

    def test_completed_request_may_only_append_its_plan_record(self):
        self.summary['frozen_inputs'] = self.request['frozen_inputs']
        with self.assertRaisesRegex(ValueError, 'frozen request and plan'):
            self.validate()
        self.regeneration.assert_not_called()

    def test_fresh_training_failure_and_changed_plan_fail_closed(self):
        for effect in (ValueError('training budget incomplete'), lambda *a: {**self.plan, 'selected_steps': [999]}):
            self.regeneration.side_effect = effect
            with self.assertRaises(ValueError):
                self.validate()
        self.native.assert_not_called()

    def test_source_root_relabeling_is_rejected_even_with_equal_fresh_plan(self):
        self.plan['arm_roots']['original'] = str(self.root)
        training.publish(self.upstream / 'plan' / 'plan.json', self.plan, exclusive=False)
        updated = training.record(self.upstream / 'plan' / 'plan.json')
        self.summary.update(plan=updated, frozen_inputs=[*self.request['frozen_inputs'], updated])
        with self.assertRaisesRegex(ValueError, 'source roots'):
            self.validate()

    def test_missing_or_reordered_completed_interventions_rejected(self):
        self.summary['completed'] = self.summary['completed'][::-1]
        with self.assertRaisesRegex(ValueError, 'prescribed ordered screen'):
            self.validate()

    def test_changed_bytes_extra_files_or_symlinks_are_rejected(self):
        (self.upstream / 'copy' / 'result.bin').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact inventory'):
            self.validate()

    def test_extra_unrecorded_stage_file_is_rejected(self):
        (self.upstream / 'copy' / 'extra').write_bytes(b'unrecorded')
        with self.assertRaisesRegex(ValueError, 'artifact inventory'):
            self.validate()

    def test_symlink_in_stage_is_rejected(self):
        (self.upstream / 'copy' / 'link').symlink_to(self.bound)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.validate()

    def test_changed_per_intervention_marker_is_rejected(self):
        training.publish(self.upstream / 'copy' / 'complete.json', {'changed': True}, exclusive=False)
        with self.assertRaisesRegex(ValueError, 'per-intervention completion'):
            self.validate()

    def test_changed_completion_marker_or_artifact_ledger_is_rejected(self):
        self.summary['completed_artifact_records'].pop()
        with self.assertRaisesRegex(ValueError, 'ledger differs'):
            self.validate()

    def test_frozen_input_mutation_stops_before_fresh_plan(self):
        self.bound.write_bytes(b'mutated')
        with self.assertRaisesRegex(ValueError, 'frozen evidence changed'):
            self.validate()
        self.regeneration.assert_not_called()


class RunnerTest(Fixture):
    def setUp(self):
        super().setUp()
        self.prepare_completed()
        history = self.root.parent / 'historical'
        history.mkdir()
        checkpoint = history / 'step_13030'
        checkpoint.mkdir()
        self.assay = dict(historical_checkpoint=str(checkpoint), block=1, head=2,
            query=1023, target_id=2797, sources=[1022, 889, 1023, 1021, 512])
        for name in ('prefix', 'expected_logits', 'expected_qkv', 'expected_context'):
            (history / name).write_bytes(b'fixture ' + name.encode())
            self.assay[name] = training.record(history / name)
        self.historical_records = [self.assay[name] for name in ('prefix', 'expected_logits', 'expected_qkv', 'expected_context')]
        self.binaries = {}
        for name in ('probe', 'gpu_test'):
            path = self.root.parent / name
            path.write_bytes(b'not executable; native code must be mocked: ' + name.encode())
            self.binaries[name] = path
        self.args = SimpleNamespace(root=self.root, causal_output=self.upstream,
            output=self.root / 'source_stage', historical_root=history,
            historical_checkpoint=checkpoint, historical_manifest=history / 'manifest.json',
            observer_pid=123, observer_start_ticks=456, **self.binaries)

    @contextmanager
    def mocks(self, failure=None):
        calls, commands = [], []

        def wait(*args):
            calls.append('wait')
            if failure == 'wait':
                raise RuntimeError('upstream failed')
            if failure == 'binary_changed':
                self.binaries['probe'].write_bytes(b'changed while waiting')
            if failure == 'copied_source_changed':
                copy_path = self.args.output / 'source' / Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[2])
                copy_path.chmod(0o644)
                copy_path.write_text('changed source copy')
            return self.summary

        def validate(*args):
            calls.append('validate')
            if failure == 'training':
                raise ValueError('training incomplete')
            return [training.record(self.upstream / 'summary.json')]

        def idle(*args):
            calls.append('idle')
            if failure == 'busy':
                raise RuntimeError('GPU busy')

        def test_run(command, inputs, output, log, record_path):
            calls.append('gpu_test'); commands.append(command)
            self.write_gtest(output / 'gtest.json', skipped=failure == 'skipped')
            log.write_text('synthetic GPU test')
            result = dict(command=command, returncode=0, inputs_before=[training.record(p) for p in inputs],
                          outputs=[training.record(output / 'gtest.json')], log=training.record(log))
            training.publish(record_path, result)
            if failure == 'after_test_upstream':
                training.publish(self.upstream / 'summary.json', {'changed': True}, exclusive=False)
            return result

        def probe(command, log):
            calls.append('probe'); commands.append(command)
            directory = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--output_dir=')))
            directory.mkdir(); (directory / 'fixture').write_bytes(b'native output mocked')
            log.write_text('synthetic source probe')
            return dict(command=command, returncode=0, pid=999, log=training.record(log))

        def summarize(directory, path, assay):
            calls.append('readout')
            self.assertEqual(assay, self.assay)
            self.assertEqual(training.read_json(path)['returncode'], 0)
            return dict(complete=True, new_paired_model_result=False,
                        goal_completion_claimed=False, synthetic_test_fixture=True)

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(training, 'process_identity', return_value=self.identity))
            stack.enter_context(mock.patch.object(training, 'gpu_snapshot', return_value={'device': self.device}))
            stack.enter_context(mock.patch.object(followup, 'source_paths', return_value=[Path(__file__).resolve()]))
            stack.enter_context(mock.patch.object(followup.historical, 'authenticate_assay', return_value=(self.assay, self.historical_records)))
            stack.enter_context(mock.patch.object(followup, 'wait_for_causal', side_effect=wait))
            stack.enter_context(mock.patch.object(followup, 'validate_upstream', side_effect=validate))
            stack.enter_context(mock.patch.object(training, 'require_idle_gpu', side_effect=idle))
            stack.enter_context(mock.patch.object(followup.historical.causal, 'execute_recorded', side_effect=test_run))
            stack.enter_context(mock.patch.object(followup.historical.causal, 'execute', side_effect=probe))
            stack.enter_context(mock.patch.object(followup.readout, 'summarize', side_effect=summarize))
            spawn = stack.enter_context(mock.patch.object(followup.historical.subprocess, 'Popen', side_effect=AssertionError('real process forbidden')))
            shell = stack.enter_context(mock.patch.object(followup.historical.subprocess, 'run', side_effect=AssertionError('real subprocess forbidden')))
            yield calls, commands
            spawn.assert_not_called(); shell.assert_not_called()

    def test_ordered_success_keeps_historical_scope_and_exact_old_assay(self):
        with self.mocks() as (calls, commands):
            result = followup.run(self.args)
        self.assertEqual(calls, ['wait', 'validate', 'idle', 'gpu_test', 'idle', 'probe', 'readout'])
        self.assertEqual(len(commands), 2)
        self.assertIn('--sources=1022,889,1023,1021,512', commands[1])
        self.assertIn('--checkpoint=' + self.assay['historical_checkpoint'], commands[1])
        self.assertIn('--query=1023', commands[1]); self.assertIn('--target_id=2797', commands[1])
        self.assertFalse(result['new_paired_model_result'])
        self.assertFalse(result['initial_piece_assay'])
        self.assertFalse(result['goal_completion_claimed'])
        self.assertEqual(result['stage'], 'historical_source_value')
        training.verify_records(result['frozen_inputs'])
        self.assertIsNone(training.read_json(self.args.output / 'validated_upstream.json')['observer_exit_code'])
        self.assertEqual(result, training.read_json(self.args.output / 'summary.json'))

    def test_wait_failure_never_starts_native_child(self):
        with self.mocks('wait') as (calls, commands), self.assertRaises(RuntimeError):
            followup.run(self.args)
        self.assertEqual(calls, ['wait']); self.assertFalse(commands)
        self.assertFalse(training.read_json(self.args.output / 'failure.json')['training_restarted'])

    def test_training_gate_failure_never_starts_native_child(self):
        with self.mocks('training') as (calls, commands), self.assertRaises(ValueError):
            followup.run(self.args)
        self.assertEqual(calls, ['wait', 'validate']); self.assertFalse(commands)

    def test_busy_gpu_stops_before_any_native_child(self):
        with self.mocks('busy') as (calls, commands), self.assertRaises(RuntimeError):
            followup.run(self.args)
        self.assertEqual(calls, ['wait', 'validate', 'idle']); self.assertFalse(commands)

    def test_changed_binary_while_queued_fails_before_training_validation(self):
        with self.mocks('binary_changed') as (calls, commands), self.assertRaises(ValueError):
            followup.run(self.args)
        self.assertEqual(calls, ['wait']); self.assertFalse(commands)

    def test_changed_copied_source_while_queued_fails_closed(self):
        with self.mocks('copied_source_changed') as (calls, commands), self.assertRaises(ValueError):
            followup.run(self.args)
        self.assertEqual(calls, ['wait']); self.assertFalse(commands)

    def test_skipped_gpu_test_prevents_historical_probe(self):
        with self.mocks('skipped') as (calls, commands), self.assertRaises(ValueError):
            followup.run(self.args)
        self.assertEqual(len(commands), 1); self.assertNotIn('probe', calls)

    def test_upstream_mutation_during_test_prevents_historical_probe(self):
        with self.mocks('after_test_upstream') as (calls, commands), self.assertRaises(ValueError):
            followup.run(self.args)
        self.assertEqual(len(commands), 1); self.assertNotIn('probe', calls)

    def test_existing_output_and_nonchild_output_are_rejected(self):
        self.args.output = self.upstream
        with self.assertRaises(FileExistsError):
            followup.run(self.args)
        self.args.output = self.root.parent / 'wrong'
        with self.assertRaises(ValueError):
            followup.run(self.args)


if __name__ == '__main__':
    unittest.main()
