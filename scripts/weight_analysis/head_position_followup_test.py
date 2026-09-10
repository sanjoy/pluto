"""CPU-only controller tests; every native child and heavy readout is synthetic.

Real immutable file copying, record verification, publication, and recursive
inventory checks are retained. These tests never certify a GPU result: the
separate native suite must run successfully after the real queue handoff.
"""

from contextlib import ExitStack, contextmanager
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from . import head_position_followup as followup

training = followup.training
TEST_NAMES = {
    'NativeScopesDosesAndIndependentProjectionZeroAgree',
    'OtherSequenceAndFutureOnlyQueriesCannotChangeLogits',
    'RejectsWrongIdentityMalformedInputsAndOriginalMutation',
}


def write_gtest(path, *, skipped=False):
    cases = [dict(name=name, status='RUN', result='COMPLETED') for name in sorted(TEST_NAMES)]
    if skipped:
        cases[0]['skipped'] = {}
    value = dict(tests=3, failures=0, errors=0, disabled=0,
                 testsuites=[dict(name='HeadContextGpuTest', testsuite=cases)])
    training.publish(path, value)
    return value


def native_tree(directory):
    """Correct file *inventory* only, explicitly not valid numerical evidence."""
    directory.mkdir()
    for name in ('metadata.json', 'tokens.i32', 'padded_tokens.i32',
                 'original_before.bf16', 'original_context.bf16',
                 'original_qkv.bf16', 'clean_logits.f32'):
        (directory / name).write_bytes(b'synthetic inventory, not native data')
    names = ['calibration_all_queries_zero',
             *[f'{scope}_{label}' for scope in followup.readout.oracle.SCOPES
               for label, _ in followup.readout.DOSES]]
    for name in names:
        arm = directory / name
        arm.mkdir()
        for filename in ('context.bf16', 'logits.f32'):
            (arm / filename).write_bytes(b'synthetic arm bytes')


class FileFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.binary = self.base / 'probe'
        self.binary.write_bytes(b'never execute this synthetic binary')
        self.input = self.base / 'input'
        self.input.write_bytes(b'bound bytes')
        self.spawn = self.enterContext(mock.patch.object(followup.common.causal.subprocess, 'Popen',
            side_effect=AssertionError('real native subprocess forbidden')))
        self.shell = self.enterContext(mock.patch.object(followup.common.causal.subprocess, 'run',
            side_effect=AssertionError('real subprocess forbidden')))


class HelpersTest(FileFixture):
    def test_source_freeze_explicitly_includes_main_without_canonical_alias(self):
        with mock.patch.dict(sys.modules):
            sys.modules.pop(followup.__name__, None)
            self.assertNotIn(followup.__name__, sys.modules)
            self.assertIn(Path(followup.__file__).resolve(), followup.source_paths())

    def test_exact_native_command_has_only_nine_explicit_flags(self):
        case = dict(block=3, head=6, query=370, target_id=4490,
            prefix=dict(path='/archive/prefix'), calibration=dict(
                clean_logits=dict(path='/archive/clean'),
                all_query_weight_zero_logits=dict(path='/archive/zero')))
        command = followup.command_for_case(self.binary, dict(checkpoint_directory='/weights'), case, '/new/native')
        self.assertEqual(command, [str(self.binary), '--checkpoint=/weights',
            '--tokens_file=/archive/prefix', '--output_dir=/new/native', '--block=3',
            '--head=6', '--query=370', '--target_id=4490',
            '--expected_clean_logits=/archive/clean', '--expected_all_query_zero_logits=/archive/zero'])

    def test_native_test_gate_requires_all_three_actual_names(self):
        path = self.base / 'gtest.json'
        good = write_gtest(path)
        followup.require_native_tests(path)
        for kind in ('filtered', 'unrelated', 'duplicate', 'suite', 'skipped', 'failed'):
            value = copy.deepcopy(good)
            cases = value['testsuites'][0]['testsuite']
            if kind == 'filtered':
                cases.pop(); value['tests'] = 2
            elif kind == 'unrelated': cases[0]['name'] = 'UnrelatedPassingTest'
            elif kind == 'duplicate': cases[0]['name'] = cases[1]['name']
            elif kind == 'suite': value['testsuites'][0]['name'] = 'OtherGpuTest'
            elif kind == 'skipped': cases[0]['skipped'] = {}
            elif kind == 'failed': value['failures'] = 1
            training.publish(path, value, exclusive=False)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                followup.require_native_tests(path)
        self.spawn.assert_not_called(); self.shell.assert_not_called()


class ExecuteTreeTest(FileFixture):
    def setUp(self):
        super().setUp()
        self.native = self.base / 'native'
        self.log = self.base / 'probe.log'
        self.execution = self.base / 'execution.json'
        self.command = [str(self.binary), '--output_dir=' + str(self.native)]

    def child(self, command, log):
        native_tree(self.native)
        log.write_text('synthetic child log')
        return dict(command=command, pid=42, returncode=0,
            started_utc='2026-09-10T00:00:00+00:00', finished_utc='2026-09-10T00:00:01+00:00',
            log=training.record(log))

    def execute(self):
        return followup.execute_tree(self.command, [self.input, self.binary, self.input],
                                     self.native, self.log, self.execution)

    def test_records_real_inputs_and_complete_recursive_outputs(self):
        with mock.patch.object(followup.common.causal, 'execute', side_effect=self.child) as execute:
            result = self.execute()
        self.assertEqual(result, training.read_json(self.execution))
        self.assertEqual(result['format'], followup.readout.EXECUTION_FORMAT)
        self.assertEqual(result['inputs_before'], result['inputs_after'])
        self.assertEqual(len(result['inputs_before']), 2)
        self.assertEqual(len(result['outputs']), 33)
        training.verify_records([*result['inputs_before'], *result['outputs'], result['log']])
        execute.assert_called_once_with(self.command, self.log)

    def test_rejects_existing_directory_before_execution(self):
        self.native.mkdir(); (self.native / 'keep').write_text('preserved')
        with mock.patch.object(followup.common.causal, 'execute') as execute, self.assertRaises(FileExistsError):
            self.execute()
        execute.assert_not_called()
        self.assertEqual((self.native / 'keep').read_text(), 'preserved')

    def test_rejects_existing_execution_record_before_execution(self):
        self.execution.write_text('preserved old execution')
        with mock.patch.object(followup.common.causal, 'execute') as execute, self.assertRaises(FileExistsError):
            self.execute()
        execute.assert_not_called()
        self.assertEqual(self.execution.read_text(), 'preserved old execution')

    def test_missing_binary_input_and_linked_input_cannot_launch(self):
        with mock.patch.object(followup.common.causal, 'execute') as execute:
            with self.assertRaises(ValueError):
                followup.execute_tree(self.command, [self.input], self.native, self.log, self.execution)
            linked = self.base / 'linked'; linked.symlink_to(self.input)
            with self.assertRaises(ValueError):
                followup.execute_tree(self.command, [self.binary, linked], self.native, self.log, self.execution)
        execute.assert_not_called()

    def test_failed_child_keeps_partial_tree_and_log_without_success_record(self):
        def fail(command, log):
            self.native.mkdir(); (self.native / 'partial').write_text('keep')
            log.write_text('failure detail')
            raise RuntimeError('synthetic child failed')
        with mock.patch.object(followup.common.causal, 'execute', side_effect=fail), self.assertRaises(RuntimeError):
            self.execute()
        self.assertFalse(self.execution.exists())
        self.assertEqual((self.native / 'partial').read_text(), 'keep')
        self.assertEqual(self.log.read_text(), 'failure detail')

    def test_changed_inputs_are_rejected_after_native_exit(self):
        def mutate(command, log):
            result = self.child(command, log); self.input.write_bytes(b'changed')
            return result
        with mock.patch.object(followup.common.causal, 'execute', side_effect=mutate), self.assertRaisesRegex(ValueError, 'inputs changed'):
            self.execute()
        self.assertTrue(self.native.exists()); self.assertFalse(self.execution.exists())

    def test_incomplete_inventory_cannot_publish_success(self):
        def incomplete(command, log):
            result = self.child(command, log)
            (self.native / 'unexpected').write_text('extra')
            return result
        with mock.patch.object(followup.common.causal, 'execute', side_effect=incomplete), self.assertRaisesRegex(ValueError, 'inventory'):
            self.execute()
        self.assertFalse(self.execution.exists())


class RunnerTest(FileFixture):
    def setUp(self):
        super().setUp()
        self.root = self.base / 'amendment'; self.root.mkdir()
        self.upstream = self.root / 'source_stage'; self.upstream.mkdir()
        self.output = self.root / 'head_position'
        self.identity = dict(pid=123, start_ticks=456, argv=['python', '-m', 'synthetic'], state='S', parent_pid=1)
        self.device = dict(index='0', uuid='GPU-fixture', name='fixture', driver_version='fixture')
        training.publish(self.root / 'request.json', dict(format=training.FORMAT,
            amendment_root=str(self.root), gpu=self.device, frozen_inputs=[training.record(self.input)]))
        training.publish(self.upstream / 'request.json', dict(frozen_inputs=[training.record(self.input)]))
        training.publish(self.upstream / 'summary.json', dict(complete=True, synthetic_test_fixture=True))
        self.descriptor = self.base / 'cases.json'
        self.document = dict(checkpoint_directory='/synthetic/weights', case_count=2,
            cases=[dict(case_index=i, word=word, block=0, head=i, query=3, target_id=2,
                        prefix=dict(path=str(self.input)), calibration=dict(
                            clean_logits=dict(path=str(self.input)),
                            all_query_weight_zero_logits=dict(path=str(self.input))))
                   for i, word in enumerate(('Exeunt', 'grandam'))])
        training.publish(self.descriptor, self.document)
        self.gpu_test = self.base / 'gpu_test'; self.gpu_test.write_bytes(b'mock tests only')
        self.args = SimpleNamespace(root=self.root, source_output=self.upstream,
            output=self.output, cases=self.descriptor, probe=self.binary, gpu_test=self.gpu_test,
            observer_pid=123, observer_start_ticks=456)

    @contextmanager
    def mocks(self, failure=None):
        calls, commands = [], []

        def descriptor(path, ledger, config, allow_synthetic):
            if failure == 'descriptor': raise ValueError('invalid descriptor')
            ledger.add(training.record(self.input))
            return self.document, ledger.file(path)

        def wait(path, expected):
            calls.append('wait')
            request = training.read_json(self.output / 'request.json')
            training.verify_records(request['frozen_inputs'])
            self.assertEqual(expected, self.identity)
            self.assertEqual(request['binaries']['probe']['sha256'], training.record(self.binary)['sha256'])
            self.assertTrue((self.output / 'bin/gpu_test').is_file())
            source_copy = self.output / 'source' / Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[2])
            self.assertEqual(source_copy.read_bytes(), Path(__file__).read_bytes())
            self.assertFalse(commands)
            if failure == 'wait': raise RuntimeError('source observer failed')
            if failure == 'live': raise KeyboardInterrupt('source observer remains live; stop synthetic wait')
            if failure == 'binary_changed': self.binary.write_bytes(b'changed during wait')
            if failure == 'descriptor_changed': self.descriptor.write_text('changed during wait')
            if failure == 'source_copy_changed':
                source_copy.chmod(0o644); source_copy.write_text('changed frozen source')
            return training.read_json(path)

        def validate(*args):
            calls.append('validate')
            if failure == 'upstream': raise ValueError('invalid upstream evidence')
            return [training.record(self.upstream / 'summary.json')]

        def idle(*args):
            calls.append('idle')
            if failure == 'busy': raise RuntimeError('GPU is busy')

        def tests(command, inputs, directory, log, execution_path):
            calls.append('gpu_test'); commands.append(command)
            write_gtest(directory / 'gtest.json', skipped=failure == 'skipped')
            log.write_text('synthetic GPU test result, never a real GPU run')
            records = [training.record(p) for p in inputs]
            result = dict(command=command, pid=999, returncode=0,
                inputs_before=records, inputs_after=records,
                outputs=[training.record(directory / 'gtest.json')], log=training.record(log))
            training.publish(execution_path, result)
            if failure == 'after_test_upstream':
                training.publish(self.upstream / 'summary.json', {'changed': True}, exclusive=False)
            return result

        def probe(command, log):
            calls.append('probe'); commands.append(command)
            directory = Path(next(a.split('=', 1)[1] for a in command if a.startswith('--output_dir=')))
            log.write_text('synthetic probe output')
            if failure == 'second_probe' and directory.parent.name == 'case_01':
                directory.mkdir(); (directory / 'partial').write_text('second failure preserved')
                raise RuntimeError('second probe failed')
            native_tree(directory)
            return dict(command=command, pid=1000, returncode=0, log=training.record(log),
                started_utc='2026-09-10T00:00:00+00:00', finished_utc='2026-09-10T00:00:01+00:00')

        def read_case(path, index, directory, execution):
            calls.append('readout')
            self.assertEqual(len(training.read_json(execution)['outputs']), 33)
            if failure == 'second_readout' and index == 1: raise ValueError('second readout invalid')
            return dict(complete=True, synthetic_test_fixture=True, case_index=index)

        def aggregate(path, completed):
            calls.append('aggregate')
            self.assertEqual([c['case_index'] for c in completed], [0, 1])
            return dict(complete=True, synthetic_test_fixture=True)

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(followup.handoff, 'observer_identity',
                **({'side_effect': ValueError('wrong observer identity')} if failure == 'identity' else {'return_value': self.identity})))
            stack.enter_context(mock.patch.object(training, 'process_identity', return_value=self.identity))
            stack.enter_context(mock.patch.object(training, 'gpu_snapshot', return_value={'device': self.device}))
            stack.enter_context(mock.patch.object(followup, 'source_paths', return_value=[Path(__file__).resolve()]))
            stack.enter_context(mock.patch.object(followup.readout, 'validate_descriptor', side_effect=descriptor))
            stack.enter_context(mock.patch.object(followup.handoff, 'wait_for_source', side_effect=wait))
            stack.enter_context(mock.patch.object(followup.handoff, 'validate_upstream', side_effect=validate))
            stack.enter_context(mock.patch.object(training, 'require_idle_gpu', side_effect=idle))
            stack.enter_context(mock.patch.object(followup.common.causal, 'execute_recorded', side_effect=tests))
            stack.enter_context(mock.patch.object(followup.common.causal, 'execute', side_effect=probe))
            stack.enter_context(mock.patch.object(followup.readout, 'analyze_case', side_effect=read_case))
            stack.enter_context(mock.patch.object(followup.readout, 'analyze_all', side_effect=aggregate))
            yield calls, commands
            self.spawn.assert_not_called(); self.shell.assert_not_called()

    def test_success_orders_handoff_gates_and_preserves_per_case_progress(self):
        original = self.input.read_bytes()
        with self.mocks() as (calls, commands): result = followup.run(self.args)
        self.assertEqual(calls, ['wait', 'validate', 'idle', 'gpu_test', 'idle', 'probe', 'readout',
                                 'idle', 'probe', 'readout', 'aggregate'])
        self.assertEqual(len(commands), 3)
        self.assertTrue(result['historical_only']); self.assertFalse(result['new_paired_model_result'])
        self.assertFalse(result['full_word_probability_claimed']); self.assertFalse(result['goal_completion_claimed'])
        self.assertEqual([c['case_index'] for c in result['completed']], [0, 1])
        self.assertEqual(result, training.read_json(self.output / 'summary.json'))
        self.assertEqual(training.read_json(self.output / 'state.json')['phase'], 'complete')
        self.assertIsNone(training.read_json(self.output / 'validated_upstream.json')['observer_exit_code'])
        self.assertEqual(self.input.read_bytes(), original)
        training.verify_records([*result['frozen_inputs'], *result['completed_artifact_records']])

    def assert_no_gpu(self, failure, error):
        with self.mocks(failure) as (calls, commands), self.assertRaises(error): followup.run(self.args)
        self.assertFalse(commands)
        return calls

    def test_live_observer_never_unlocks_gpu(self):
        self.assertEqual(self.assert_no_gpu('live', KeyboardInterrupt), ['wait'])

    def test_failed_observer_never_unlocks_gpu(self):
        self.assertEqual(self.assert_no_gpu('wait', RuntimeError), ['wait'])
        self.assertFalse(training.read_json(self.output / 'failure.json')['training_restarted'])

    def test_invalid_identity_rejected_before_creating_output(self):
        self.assertEqual(self.assert_no_gpu('identity', ValueError), [])
        self.assertFalse(self.output.exists())

    def test_invalid_descriptor_rejected_before_creating_output(self):
        self.assertEqual(self.assert_no_gpu('descriptor', ValueError), [])
        self.assertFalse(self.output.exists())

    def test_upstream_validation_failure_prevents_any_gpu_execution(self):
        self.assertEqual(self.assert_no_gpu('upstream', ValueError), ['wait', 'validate'])

    def test_busy_gpu_prevents_even_test_execution(self):
        self.assertEqual(self.assert_no_gpu('busy', RuntimeError), ['wait', 'validate', 'idle'])

    def test_changed_original_binary_while_waiting_fails_closed(self):
        self.assertEqual(self.assert_no_gpu('binary_changed', ValueError), ['wait'])

    def test_changed_descriptor_while_waiting_fails_closed(self):
        self.assertEqual(self.assert_no_gpu('descriptor_changed', ValueError), ['wait'])

    def test_changed_frozen_source_copy_while_waiting_fails_closed(self):
        self.assertEqual(self.assert_no_gpu('source_copy_changed', ValueError), ['wait'])

    def test_skipped_native_test_prevents_all_probe_cases(self):
        with self.mocks('skipped') as (calls, commands), self.assertRaises(ValueError): followup.run(self.args)
        self.assertEqual(len(commands), 1); self.assertNotIn('probe', calls)

    def test_upstream_mutation_after_tests_prevents_probe(self):
        with self.mocks('after_test_upstream') as (calls, commands), self.assertRaises(ValueError): followup.run(self.args)
        self.assertEqual(len(commands), 1); self.assertNotIn('probe', calls)

    def assert_partial_failure(self, kind, error):
        with self.mocks(kind) as (calls, commands), self.assertRaises(error): followup.run(self.args)
        failed = training.read_json(self.output / 'failure.json')
        self.assertEqual([c['case_index'] for c in failed['completed']], [0])
        self.assertFalse(failed['training_restarted']); self.assertTrue(failed['no_automatic_restart'])
        self.assertTrue((self.output / 'case_00/complete.json').is_file())
        self.assertTrue((self.output / 'case_01/probe.log').is_file())
        self.assertFalse((self.output / 'case_01/complete.json').exists())
        self.assertFalse((self.output / 'summary.json').exists())
        self.assertNotIn('aggregate', calls)

    def test_second_native_failure_keeps_first_success_and_partial_native_files(self):
        self.assert_partial_failure('second_probe', RuntimeError)
        self.assertEqual((self.output / 'case_01/native/partial').read_text(), 'second failure preserved')
        self.assertFalse((self.output / 'case_01/execution.json').exists())

    def test_second_readout_failure_keeps_native_evidence_and_first_success(self):
        self.assert_partial_failure('second_readout', ValueError)
        self.assertTrue((self.output / 'case_01/execution.json').is_file())
        self.assertEqual(len(followup.readout.native_inventory(self.output / 'case_01/native')), 33)

    def test_existing_output_outside_root_and_linked_root_are_rejected(self):
        with self.mocks() as (calls, commands):
            self.args.output = self.upstream
            with self.assertRaises(FileExistsError): followup.run(self.args)
            self.args.output = self.base / 'outside'
            with self.assertRaises(ValueError): followup.run(self.args)
            linked = self.base / 'linked_root'; linked.symlink_to(self.root, target_is_directory=True)
            self.args.root = linked
            with self.assertRaises(ValueError): followup.run(self.args)
        self.assertFalse(calls); self.assertFalse(commands)


if __name__ == '__main__':
    unittest.main()
