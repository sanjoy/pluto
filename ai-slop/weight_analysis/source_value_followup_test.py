"""CPU-only queue and provenance tests for the historical source-value assay.

No test executes a CUDA program or queries the real GPU. Native producers are
replaced by small file writers; byte hashing, immutable copies, ordering gates,
and non-skipped GoogleTest validation remain real where practical.
"""

from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from . import source_value_followup as followup


class SourceValueFollowupTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.summary_path = self.root / 'summary.json'
        self.identity = {
            'pid': 123, 'start_ticks': 456,
            'argv': ['python', '-m', 'weight_analysis.paired_branch_followup',
                     '--root', str(self.root), '--output', str(self.root / 'branches')],
        }

    @staticmethod
    def completed_summary():
        return {'format': 'pluto-paired-branch-followup-v1', 'complete': True,
                'stage': 'branches', 'goal_completion_claimed': False}

    def publish_summary(self, **changes):
        value = {**self.completed_summary(), **changes}
        self.summary_path.write_text(json.dumps(value))
        return value

    def test_summary_does_not_bypass_live_observer_or_two_confirmed_exits(self):
        expected_summary = self.publish_summary()
        with mock.patch.object(followup.paired_analysis, '_process_identity',
                side_effect=[self.identity, self.identity, FileNotFoundError(),
                             ProcessLookupError()]) as inspect, \
             mock.patch.object(followup.causal, 'process_still_live', return_value=False) as live, \
             mock.patch.object(followup.time, 'sleep') as sleep, \
             mock.patch.object(followup.causal.subprocess, 'Popen') as spawn:
            self.assertEqual(followup.wait_for_branches(self.summary_path, self.identity),
                             expected_summary)
        self.assertEqual(inspect.call_args_list, [mock.call(123)] * 4)
        self.assertEqual(live.call_count, 2)
        self.assertEqual(sleep.call_args_list, [mock.call(30)] * 3)
        spawn.assert_not_called()

    def test_transient_read_error_and_signal_zero_live_do_not_imply_exit(self):
        self.publish_summary()
        with mock.patch.object(followup.paired_analysis, '_process_identity', side_effect=[
                OSError('temporary proc read failure'), FileNotFoundError(),
                self.identity, FileNotFoundError(), FileNotFoundError()]) as inspect, \
             mock.patch.object(followup.causal, 'process_still_live',
                               side_effect=[True, False, False]) as live, \
             mock.patch.object(followup.time, 'sleep') as sleep, \
             mock.patch.object(followup.causal.subprocess, 'Popen') as spawn:
            followup.wait_for_branches(self.summary_path, self.identity, poll_seconds=1)
        self.assertEqual(inspect.call_count, 5)
        self.assertEqual(live.call_count, 3)
        self.assertEqual(sleep.call_args_list, [mock.call(1)] * 4)
        spawn.assert_not_called()

    def test_signal_zero_live_resets_missing_observation_count(self):
        self.publish_summary()
        with mock.patch.object(followup.paired_analysis, '_process_identity',
                               side_effect=FileNotFoundError()) as inspect, \
             mock.patch.object(followup.causal, 'process_still_live',
                               side_effect=[False, True, False, False]), \
             mock.patch.object(followup.time, 'sleep') as sleep:
            followup.wait_for_branches(self.summary_path, self.identity)
        self.assertEqual(inspect.call_count, 4)
        self.assertEqual(sleep.call_count, 3)

    def test_transient_error_resets_missing_count_for_consecutive_exit_policy(self):
        self.publish_summary()
        with mock.patch.object(followup.paired_analysis, '_process_identity', side_effect=[
                FileNotFoundError(), OSError('temporary error'), FileNotFoundError(),
                FileNotFoundError()]) as inspect, \
             mock.patch.object(followup.causal, 'process_still_live', return_value=False), \
             mock.patch.object(followup.time, 'sleep') as sleep:
            followup.wait_for_branches(self.summary_path, self.identity)
        self.assertEqual(inspect.call_count, 4)
        self.assertEqual(sleep.call_count, 3)

    def test_missing_observer_without_summary_gets_four_confirmed_observations(self):
        with mock.patch.object(followup.paired_analysis, '_process_identity',
                               side_effect=FileNotFoundError()) as inspect, \
             mock.patch.object(followup.causal, 'process_still_live', return_value=False), \
             mock.patch.object(followup.time, 'sleep') as sleep, \
             mock.patch.object(followup.causal.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'missing without completed summary'):
                followup.wait_for_branches(self.summary_path, self.identity)
        self.assertEqual(inspect.call_count, 4)
        self.assertEqual(sleep.call_count, 3)
        spawn.assert_not_called()

    def test_persistent_observation_errors_fail_closed_without_restart(self):
        self.publish_summary()
        with mock.patch.object(followup.paired_analysis, '_process_identity',
                               side_effect=OSError('unreadable')) as inspect, \
             mock.patch.object(followup.time, 'sleep'), \
             mock.patch.object(followup.causal.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'cannot verify branch observer'):
                followup.wait_for_branches(self.summary_path, self.identity)
        self.assertEqual(inspect.call_count, 4)
        spawn.assert_not_called()

    def test_pid_reuse_and_changed_argv_are_rejected(self):
        self.publish_summary()
        for change in ({'pid': 124}, {'start_ticks': 457}, {'argv': ['different program']}):
            with self.subTest(change=change), \
                 mock.patch.object(followup.paired_analysis, '_process_identity',
                                   return_value={**self.identity, **change}), \
                 mock.patch.object(followup.causal.subprocess, 'Popen') as spawn:
                with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                    followup.wait_for_branches(self.summary_path, self.identity)
                spawn.assert_not_called()

    def test_failure_marker_wins_over_completed_summary_and_live_process(self):
        self.publish_summary()
        followup.write_json(self.root / 'failure.json', {'error': 'upstream failure'})
        with mock.patch.object(followup.paired_analysis, '_process_identity') as inspect:
            with self.assertRaisesRegex(RuntimeError, 'upstream branch analysis failed'):
                followup.wait_for_branches(self.summary_path, self.identity)
        inspect.assert_not_called()

    def test_bad_summary_cannot_unlock_work(self):
        for change in ({'format': 'wrong'}, {'stage': 'embedding'}, {'complete': False},
                       {'complete': 1}, {'goal_completion_claimed': True},
                       {'goal_completion_claimed': None}):
            with self.subTest(change=change):
                self.publish_summary(**change)
                with self.assertRaisesRegex(ValueError, 'not a completed branch stage'):
                    followup.wait_for_branches(self.summary_path, None)
        self.summary_path.write_text('{ not JSON')
        with self.assertRaises(ValueError):
            followup.wait_for_branches(self.summary_path, None)

    def test_already_completed_mode_requires_summary_and_does_not_poll(self):
        with mock.patch.object(followup.paired_analysis, '_process_identity') as inspect, \
             mock.patch.object(followup.time, 'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'missing without completed summary'):
                followup.wait_for_branches(self.summary_path, None)
            value = self.publish_summary()
            self.assertEqual(followup.wait_for_branches(self.summary_path, None), value)
        inspect.assert_not_called()
        sleep.assert_not_called()

    def test_bad_poll_interval_is_rejected_before_observation(self):
        for interval in (0, -1, 60.1, float('nan'), float('inf')):
            with self.subTest(interval=interval), \
                 mock.patch.object(followup.paired_analysis, '_process_identity') as inspect:
                with self.assertRaises(ValueError):
                    followup.wait_for_branches(self.summary_path, self.identity,
                                              poll_seconds=interval)
                inspect.assert_not_called()

    def test_observer_identity_binds_module_root_output_and_ticks(self):
        with mock.patch.object(followup.paired_analysis, '_process_identity',
                               return_value=self.identity):
            self.assertEqual(followup.observer_identity(123, 456, self.root,
                                                       self.root / 'branches'), self.identity)
            for ticks, root, output in ((999, self.root, self.root / 'branches'),
                                        (456, self.root / 'other', self.root / 'branches'),
                                        (456, self.root, self.root / 'other')):
                with self.subTest(ticks=ticks, root=root, output=output):
                    with self.assertRaises(ValueError):
                        followup.observer_identity(123, ticks, root, output)
        for argv in (['python'], ['python', '-m'], ['python', '-m', 'wrong'],
                     self.identity['argv'] + ['-m', 'another.module'],
                     self.identity['argv'] + ['--root', str(self.root)],
                     self.identity['argv'][:-2]):
            with self.subTest(argv=argv), \
                 mock.patch.object(followup.paired_analysis, '_process_identity',
                                   return_value={**self.identity, 'argv': argv}):
                with self.assertRaises(ValueError):
                    followup.observer_identity(123, 456, self.root, self.root / 'branches')

    def test_completed_mode_requires_both_identity_read_and_signal_zero_to_show_absence(self):
        with mock.patch.object(followup.paired_analysis, '_process_identity',
                               side_effect=FileNotFoundError()) as inspect, \
             mock.patch.object(followup.os, 'kill', side_effect=ProcessLookupError()) as signal_zero:
            followup.require_observer_absent(123)
        inspect.assert_called_once_with(123)
        signal_zero.assert_called_once_with(123, 0)

    def test_completed_mode_rejects_live_identity_or_failed_read_with_existing_pid(self):
        for identity in (self.identity, FileNotFoundError()):
            with self.subTest(identity=identity), \
                 mock.patch.object(followup.paired_analysis, '_process_identity',
                     **({'side_effect': identity} if isinstance(identity, Exception)
                        else {'return_value': identity})), \
                 mock.patch.object(followup.os, 'kill') as signal_zero:
                with self.assertRaisesRegex(ValueError, 'observer still exists'):
                    followup.require_observer_absent(123)
                if isinstance(identity, Exception):
                    signal_zero.assert_called_once_with(123, 0)
                else:
                    signal_zero.assert_not_called()

    def test_completed_mode_invalid_pid_or_unverifiable_identity_never_assumes_exit(self):
        for pid in (None, True, 0, -1, '123', 123.0):
            with self.subTest(pid=pid), \
                 mock.patch.object(followup.paired_analysis, '_process_identity') as inspect:
                with self.assertRaisesRegex(ValueError, 'valid observer PID'):
                    followup.require_observer_absent(pid)
                inspect.assert_not_called()
        for error in (OSError('proc unavailable'), ValueError('malformed stat')):
            with self.subTest(error=error), \
                 mock.patch.object(followup.paired_analysis, '_process_identity', side_effect=error), \
                 mock.patch.object(followup.os, 'kill') as signal_zero:
                with self.assertRaises(type(error)):
                    followup.require_observer_absent(123)
                signal_zero.assert_not_called()

    def test_recursive_outputs_records_every_nested_file_in_stable_order(self):
        output = self.root / 'native'
        (output / 'dose_zero' / 'head').mkdir(parents=True)
        (output / 'dose_zero' / 'head' / 'context.bin').write_bytes(b'context')
        (output / 'metadata.json').write_text('{}')
        (output / 'dose_zero' / 'logits.bin').write_bytes(b'logits')
        files = followup.recursive_outputs(output)
        self.assertEqual(files, [followup.record(path) for path in sorted((
            output / 'dose_zero' / 'head' / 'context.bin',
            output / 'dose_zero' / 'logits.bin', output / 'metadata.json'))])

    def test_recursive_outputs_rejects_empty_missing_and_file_roots(self):
        for name in ('missing', 'empty', 'nested_empty', 'regular_file'):
            path = self.root / name
            if name == 'empty':
                path.mkdir()
            elif name == 'nested_empty':
                (path / 'only_directory' / 'empty').mkdir(parents=True)
            elif name == 'regular_file':
                path.write_text('not a directory')
            with self.subTest(name=name), self.assertRaises(ValueError):
                followup.recursive_outputs(path)

    def test_recursive_outputs_rejects_symlink_and_special_file_anywhere(self):
        for kind in ('root_link', 'file_link', 'directory_link', 'fifo'):
            with self.subTest(kind=kind):
                base = self.root / kind
                native = base / 'native'
                (native / 'nested').mkdir(parents=True)
                (native / 'metadata.json').write_text('{}')
                if kind == 'root_link':
                    link = base / 'link'
                    link.symlink_to(native, target_is_directory=True)
                    native = link
                elif kind == 'file_link':
                    (native / 'nested' / 'link').symlink_to(native / 'metadata.json')
                elif kind == 'directory_link':
                    (native / 'nested' / 'link').symlink_to(native, target_is_directory=True)
                else:
                    os.mkfifo(native / 'nested' / 'pipe')
                with self.assertRaises(ValueError):
                    followup.recursive_outputs(native)

    def execution_fixture(self):
        incoming = self.root / 'input.bin'
        incoming.write_bytes(b'unchanged input')
        output = self.root / 'native'
        log = self.root / 'native.log'
        execution = self.root / 'execution.json'
        return incoming, output, log, execution

    @staticmethod
    def fake_native_output(output, log):
        (output / 'source_1022' / 'scale_0').mkdir(parents=True)
        (output / 'source_1022' / 'scale_0' / 'logits.f32').write_bytes(b'native output')
        log.write_text('fake native process finished\n')

    def test_execute_tree_records_deduplicated_inputs_and_nested_outputs(self):
        incoming, output, log, execution = self.execution_fixture()

        def execute(command, log_path):
            self.fake_native_output(output, log_path)
            return {'command': command, 'pid': 321, 'returncode': 0,
                    'log': followup.record(log_path)}

        with mock.patch.object(followup.causal, 'execute', side_effect=execute) as run:
            result = followup.execute_tree(['mock-probe'], [incoming, incoming],
                                          output, log, execution)
        run.assert_called_once_with(['mock-probe'], log)
        self.assertEqual(result, followup.read_json(execution))
        self.assertEqual(result['format'], 'pluto-source-value-execution-v1')
        self.assertEqual(result['inputs_before'], [followup.record(incoming)])
        self.assertEqual(result['inputs_after'], result['inputs_before'])
        self.assertEqual(result['outputs'], followup.recursive_outputs(output))

    def test_execute_tree_input_mutation_fails_without_modifying_native_outputs(self):
        incoming, output, log, execution = self.execution_fixture()

        def execute(command, log_path):
            self.fake_native_output(output, log_path)
            incoming.write_bytes(b'mutated during native execution')
            return {'command': command, 'pid': 321, 'returncode': 0,
                    'log': followup.record(log_path)}

        with mock.patch.object(followup.causal, 'execute', side_effect=execute):
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                followup.execute_tree(['mock-probe'], [incoming], output, log, execution)
        self.assertEqual((output / 'source_1022' / 'scale_0' / 'logits.f32').read_bytes(),
                         b'native output')
        self.assertEqual(incoming.read_bytes(), b'mutated during native execution')
        # Failed validation must not publish a success-shaped execution record.
        self.assertFalse(execution.exists())

    def test_execute_tree_failed_native_preserves_partial_nested_outputs(self):
        incoming, output, log, execution = self.execution_fixture()
        before = followup.record(incoming)

        def execute(command, log_path):
            self.fake_native_output(output, log_path)
            raise RuntimeError('analysis child exited 17')

        with mock.patch.object(followup.causal, 'execute', side_effect=execute):
            with self.assertRaisesRegex(RuntimeError, 'exited 17'):
                followup.execute_tree(['mock-probe'], [incoming], output, log, execution)
        self.assertEqual(followup.record(incoming), before)
        self.assertEqual((output / 'source_1022' / 'scale_0' / 'logits.f32').read_bytes(),
                         b'native output')
        self.assertIn('finished', log.read_text())
        self.assertFalse(execution.exists())

    def test_gpu_identity_accepts_only_single_recorded_device_and_matching_visibility(self):
        for visible in (None, '0', 'GPU-fixture'):
            with self.subTest(visible=visible), mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(followup.subprocess, 'run', return_value=SimpleNamespace(
                     stdout='GPU-fixture\n')) as query:
                if visible is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = visible
                self.assertEqual(followup.gpu_identity(), 'GPU-fixture')
                self.assertEqual(query.call_args.args[0],
                    ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'])
                self.assertEqual(query.call_args.kwargs['timeout'], 15)

    def test_gpu_identity_rejects_disabled_wrong_or_multiple_devices(self):
        for stdout, visible in (('', None), ('not-a-uuid\n', None),
                                ('GPU-first\nGPU-second\n', None), ('GPU-fixture\n', ''),
                                ('GPU-fixture\n', '1'), ('GPU-fixture\n', 'GPU-other'),
                                ('GPU-fixture\n', '0,1')):
            with self.subTest(stdout=stdout, visible=visible), \
                 mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(followup.subprocess, 'run',
                                   return_value=SimpleNamespace(stdout=stdout)):
                if visible is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = visible
                with self.assertRaises(ValueError):
                    followup.gpu_identity()

    def test_idle_gpu_check_never_starts_cuda_and_rejects_busy_or_changed_gpu(self):
        with mock.patch.object(followup, 'gpu_identity', return_value='GPU-fixture'), \
             mock.patch.object(followup.subprocess, 'run',
                               return_value=SimpleNamespace(stdout='')) as query, \
             mock.patch.object(followup.subprocess, 'Popen') as spawn:
            followup.require_idle_gpu('GPU-fixture')
            self.assertEqual(query.call_args.args[0],
                ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])
            query.return_value.stdout = '1727028\n'
            with self.assertRaisesRegex(RuntimeError, 'another compute process'):
                followup.require_idle_gpu('GPU-fixture')
        spawn.assert_not_called()
        with mock.patch.object(followup, 'gpu_identity', return_value='GPU-other'), \
             mock.patch.object(followup.subprocess, 'run') as query:
            with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                followup.require_idle_gpu('GPU-fixture')
        query.assert_not_called()

    def test_gpu_metadata_query_failure_is_not_silently_treated_as_idle(self):
        with mock.patch.object(followup, 'gpu_identity', return_value='GPU-fixture'), \
             mock.patch.object(followup.subprocess, 'run',
                               side_effect=subprocess.TimeoutExpired('nvidia-smi', 15)):
            with self.assertRaises(subprocess.TimeoutExpired):
                followup.require_idle_gpu('GPU-fixture')

    def runner_fixture(self):
        root = self.root / 'experiment'
        root.mkdir()
        upstream = root / 'branches'
        upstream.mkdir()
        followup.write_json(upstream / 'request.json', {
            'stage': 'branches', 'root': str(root), 'frozen_inputs': [], 'runner_pid': 123})
        followup.write_json(upstream / 'summary.json', self.completed_summary())
        followup.write_json(root / 'manifest.json', {})
        historical = self.root / 'historical' / 'step_13030'
        historical.mkdir(parents=True)
        evidence = self.root / 'historical_evidence'
        evidence.mkdir()
        for name in ('prefix.i32', 'logits.f32', 'qkv.bf16', 'context.bf16'):
            (evidence / name).write_bytes(b'frozen historical ' + name.encode())
        followup.write_json(evidence / 'manifest.json', {'complete': True})
        assay = {
            'historical_checkpoint': str(historical), 'block': 1, 'head': 2,
            'query': 1023, 'sources': [1022, 889, 1023, 1021, 512], 'target_id': 2797,
            'prefix': followup.record(evidence / 'prefix.i32'),
            'expected_logits': followup.record(evidence / 'logits.f32'),
            'expected_qkv': followup.record(evidence / 'qkv.bf16'),
            'expected_context': followup.record(evidence / 'context.bf16'),
        }
        native = self.root / 'mock_native_binary'
        native.write_bytes(b'never executed; both native producers are mocked')
        args = SimpleNamespace(root=root, branch_output=upstream, output=root / 'source_value',
            historical_manifest=evidence / 'manifest.json', historical_root=evidence,
            historical_checkpoint=historical, probe=native, gpu_test=native,
            observer_pid=None, observer_start_ticks=None)
        source = Path(__file__).resolve()  # Safe real source with repository-relative path.
        records = [followup.record(path) for path in sorted(evidence.iterdir())]
        return args, assay, records, source

    @contextmanager
    def runner_mocks(self, args, assay, historical_records, source, calls, *,
                     failure=None):
        """Mock expensive producers, retaining byte checks and test-result gate."""
        native_commands = []
        real_test_gate = followup.causal.require_gpu_test_success

        def wait(*_):
            calls.append('wait')
            if failure == 'upstream':
                raise RuntimeError('upstream failed')
            if failure == 'interrupt':
                raise KeyboardInterrupt('observer interrupted')
            if failure == 'source':
                repo = Path(followup.__file__).resolve().parents[2]
                frozen_source = args.output / 'source' / source.relative_to(repo)
                frozen_source.chmod(0o644)
                frozen_source.write_text('# changed while queued\n')
            return self.completed_summary()

        def validate(*_):
            calls.append('validate')
            if failure == 'gate':
                raise ValueError('completed training gate failed')
            return [followup.record(args.branch_output / 'summary.json')]

        def idle(*_):
            calls.append('idle')
            if failure == 'busy':
                raise RuntimeError('GPU has another compute process')
            if failure == 'metadata':
                raise subprocess.TimeoutExpired('nvidia-smi', 15)

        def gpu_test(command, inputs, output, log, execution):
            calls.append('gpu_test')
            native_commands.append(command)
            report = {'tests': 1, 'failures': 0, 'errors': 0, 'disabled': 0,
                'testsuites': [{'testsuite': [{'status': 'RUN', 'result': 'COMPLETED'}]}]}
            if failure == 'skipped':
                report['testsuites'][0]['testsuite'][0]['skipped'] = {}
            followup.write_json(output / 'gtest.json', report)
            log.write_text('mock GoogleTest output\n')
            result = {'returncode': 0, 'command': command}
            followup.write_json(execution, result)
            if failure == 'after_test_source':
                repo = Path(followup.__file__).resolve().parents[2]
                copied_source = args.output / 'source' / source.relative_to(repo)
                copied_source.chmod(0o644)
                copied_source.write_text('# changed during GPU validation\n')
            elif failure == 'after_test_upstream':
                (args.branch_output / 'summary.json').write_text('{"changed": true}')
            return result

        def test_gate(path):
            calls.append('non_skipped_gate')
            return real_test_gate(path)

        def probe(command, log):
            calls.append('probe')
            native_commands.append(command)
            output = Path(next(arg.split('=', 1)[1] for arg in command
                               if arg.startswith('--output_dir=')))
            self.fake_native_output(output, log)
            return {'returncode': 0, 'command': command, 'pid': 987,
                    'log': followup.record(log)}

        def readout(native, execution, supplied_assay):
            calls.append('readout')
            self.assertEqual(supplied_assay, assay)
            self.assertEqual(native, args.output / 'native')
            self.assertEqual(execution, args.output / 'probe_execution.json')
            self.assertEqual(followup.read_json(execution)['returncode'], 0)
            return {'complete': True, 'interpretation': 'mock readout, not a measured effect'}

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(followup.importlib, 'import_module',
                return_value=SimpleNamespace(summarize=readout)))
            stack.enter_context(mock.patch.object(followup, 'authenticate_assay',
                return_value=(assay, historical_records)))
            stack.enter_context(mock.patch.object(followup, 'source_paths', return_value=[source]))
            stack.enter_context(mock.patch.object(followup.paired_analysis, '_process_identity',
                side_effect=FileNotFoundError()))
            stack.enter_context(mock.patch.object(followup.os, 'kill', side_effect=ProcessLookupError()))
            stack.enter_context(mock.patch.object(followup.paired_training, 'verify_frozen_inputs'))
            stack.enter_context(mock.patch.object(followup, 'gpu_identity', return_value='GPU-fixture'))
            stack.enter_context(mock.patch.object(followup, 'wait_for_branches', side_effect=wait))
            stack.enter_context(mock.patch.object(followup, 'validate_upstream', side_effect=validate))
            stack.enter_context(mock.patch.object(followup, 'require_idle_gpu', side_effect=idle))
            stack.enter_context(mock.patch.object(followup.causal, 'execute_recorded', side_effect=gpu_test))
            stack.enter_context(mock.patch.object(followup.causal, 'require_gpu_test_success', side_effect=test_gate))
            stack.enter_context(mock.patch.object(followup.causal, 'execute', side_effect=probe))
            # Guard against accidental unmocked GPU commands throughout the fixture.
            process = stack.enter_context(mock.patch.object(followup.subprocess, 'Popen',
                side_effect=AssertionError('unexpected external process in CPU test')))
            query = stack.enter_context(mock.patch.object(followup.subprocess, 'run',
                side_effect=AssertionError('unexpected metadata query in mocked runner')))
            yield native_commands
            process.assert_not_called()
            query.assert_not_called()

    def test_full_run_requires_queue_gate_non_skipped_gpu_test_probe_then_readout(self):
        args, assay, records, source = self.runner_fixture()
        calls = []
        with self.runner_mocks(args, assay, records, source, calls) as commands:
            result = followup.run(args)
            with self.assertRaises(FileExistsError):
                followup.run(args)  # Completed output may not be reused or overwritten.
        self.assertEqual(calls, ['wait', 'validate', 'idle', 'gpu_test',
                                 'non_skipped_gate', 'idle', 'probe', 'readout'])
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0][0], str(args.output / 'bin' / 'gpu_test'))
        self.assertEqual(commands[1][0], str(args.output / 'bin' / 'probe'))
        self.assertIn('--sources=1022,889,1023,1021,512', commands[1])
        self.assertIn('--query=1023', commands[1])
        self.assertIn('--expected_logits=' + assay['expected_logits']['path'], commands[1])
        self.assertEqual(result, followup.read_json(args.output / 'summary.json'))
        self.assertTrue(result['complete'])
        self.assertFalse(result['new_paired_model_result'])
        self.assertFalse(result['goal_completion_claimed'])
        self.assertEqual(result['stage'], 'historical_source_value')
        self.assertEqual(followup.read_json(args.output / 'validated_upstream.json')
                         ['observer_exit_code'], None)
        self.assertFalse((args.output / 'failure.json').exists())
        followup.verify_records(result['frozen_inputs'])
        self.assertTrue(any('/source/ai-slop/weight_analysis/' in item['path']
                            for item in result['frozen_inputs']))

    def test_failed_queue_gate_changed_source_or_skipped_tests_prevent_probe(self):
        for failure in ('upstream', 'gate', 'source', 'skipped', 'interrupt',
                        'after_test_source', 'after_test_upstream'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(dir=self.root) as temporary:
                parent = self.root
                self.root = Path(temporary)
                args, assay, records, source = self.runner_fixture()
                self.root = parent
                calls = []
                with self.runner_mocks(args, assay, records, source, calls, failure=failure):
                    with self.assertRaises((ValueError, RuntimeError, KeyboardInterrupt)):
                        followup.run(args)
                self.assertNotIn('probe', calls)
                self.assertNotIn('readout', calls)
                if failure not in ('skipped', 'after_test_source', 'after_test_upstream'):
                    self.assertNotIn('gpu_test', calls)
                else:
                    self.assertIn('non_skipped_gate', calls)
                failed = followup.read_json(args.output / 'failure.json')
                self.assertFalse(failed['training_restarted'])
                self.assertFalse(failed['goal_completion_claimed'])
                self.assertFalse((args.output / 'summary.json').exists())

    def test_busy_gpu_or_failed_metadata_preserves_failure_without_cuda(self):
        for failure in ('busy', 'metadata'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(dir=self.root) as temporary:
                parent = self.root
                self.root = Path(temporary)
                args, assay, records, source = self.runner_fixture()
                self.root = parent
                calls = []
                with self.runner_mocks(args, assay, records, source, calls, failure=failure) as commands:
                    with self.assertRaises((RuntimeError, subprocess.TimeoutExpired)):
                        followup.run(args)
                self.assertEqual(calls, ['wait', 'validate', 'idle'])
                self.assertEqual(commands, [])
                failed = followup.read_json(args.output / 'failure.json')
                self.assertIn(failed['type'], ('RuntimeError', 'TimeoutExpired'))
                self.assertFalse(failed['training_restarted'])
                self.assertFalse((args.output / 'summary.json').exists())

    def test_output_must_be_new_direct_child_not_reused_nested_or_symlinked(self):
        args, _, _, _ = self.runner_fixture()
        symlink = args.root / 'linked_output'
        symlink.symlink_to(self.root / 'nonexistent_target')
        choices = (args.root, args.branch_output, args.branch_output / 'child',
                   args.root / 'missing_parent' / 'child', self.root / 'outside', symlink)
        with mock.patch.object(followup.importlib, 'import_module', return_value=SimpleNamespace()), \
             mock.patch.object(followup, 'authenticate_assay') as auth, \
             mock.patch.object(followup, 'gpu_identity') as gpu:
            for output in choices:
                with self.subTest(output=output):
                    args.output = output
                    with self.assertRaises((FileExistsError, ValueError)):
                        followup.run(args)
        auth.assert_not_called()
        gpu.assert_not_called()

    def test_observer_pid_and_start_ticks_are_required_as_a_pair(self):
        args, _, _, _ = self.runner_fixture()
        for pid, ticks in ((123, None), (None, 456)):
            args.observer_pid, args.observer_start_ticks = pid, ticks
            with self.subTest(pid=pid, ticks=ticks), \
                 mock.patch.object(followup.importlib, 'import_module', return_value=SimpleNamespace()), \
                 mock.patch.object(followup, 'observer_identity') as identity:
                with self.assertRaisesRegex(ValueError, 'supplied together'):
                    followup.run(args)
                identity.assert_not_called()
        self.assertFalse(args.output.exists())

    def test_full_run_completed_summary_cannot_skip_live_observer_exit(self):
        args, _, _, _ = self.runner_fixture()
        with mock.patch.object(followup.importlib, 'import_module', return_value=SimpleNamespace()), \
             mock.patch.object(followup.paired_analysis, '_process_identity',
                               return_value=self.identity), \
             mock.patch.object(followup, 'authenticate_assay') as auth, \
             mock.patch.object(followup, 'gpu_identity') as gpu:
            with self.assertRaisesRegex(ValueError, 'observer still exists'):
                followup.run(args)
        auth.assert_not_called()
        gpu.assert_not_called()
        self.assertFalse(args.output.exists())

    def test_live_observer_request_must_match_runner_pid_root_and_stage(self):
        args, _, _, _ = self.runner_fixture()
        args.observer_pid, args.observer_start_ticks = 123, 456
        expected = {**self.identity, 'argv': ['mock branch observer']}
        for changes in ({'runner_pid': 999}, {'root': str(self.root)}, {'stage': 'embedding'}):
            request = {'runner_pid': 123, 'root': str(args.root), 'stage': 'branches',
                       'frozen_inputs': [], **changes}
            (args.branch_output / 'request.json').write_text(json.dumps(request))
            with self.subTest(changes=changes), \
                 mock.patch.object(followup.importlib, 'import_module', return_value=SimpleNamespace()), \
                 mock.patch.object(followup, 'observer_identity', return_value=expected), \
                 mock.patch.object(followup, 'authenticate_assay') as auth:
                with self.assertRaisesRegex(ValueError, 'request disagrees'):
                    followup.run(args)
                auth.assert_not_called()
        self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
