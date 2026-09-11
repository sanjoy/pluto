"""CPU-only handoff tests: no native process is launched by these fixtures.

Real tiny files exercise byte inventories and recursive artifact discovery.
Only the already separately tested historical assay/readout and expensive full
paired-chain validator are mocked. Their exact delegation and failure behavior
are tested here; successful flags alone never stand in for those validators.
"""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import head_position_handoff as handoff


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    handoff.training.publish(path, value, exclusive=False)


def historical_kind():
    return dict(format=handoff.source.FORMAT, stage='historical_source_value',
                new_paired_model_result=False, initial_piece_assay=False,
                goal_completion_claimed=False)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / 'amendment'
        self.upstream = self.root / 'historical_source_value_stage'
        self.causal = self.root / 'causal_stage'
        self.output = self.root / 'new_revalidation'
        self.historical_root = self.base / 'historical'
        self.checkpoint = self.base / 'checkpoint'
        for path in (self.upstream, self.causal, self.historical_root, self.checkpoint):
            path.mkdir(parents=True)
        self.manifest = self.historical_root / 'manifest.json'
        write(self.manifest, {'authenticated_fixture': True})
        self.prefix = self.historical_root / 'prefix.i32'
        self.prefix.write_bytes(b'prefix fixture')
        self.logits = self.historical_root / 'logits.f32'
        self.logits.write_bytes(b'logit fixture')
        self.historical_records = [handoff.training.record(path) for path in
                                   (self.manifest, self.prefix, self.logits)]
        self.assay = dict(block=1, head=2, query=1023, target_id=2797,
                          sources=[1022, 889, 1023, 1021, 512],
                          historical_checkpoint=str(self.checkpoint),
                          prefix=handoff.training.record(self.prefix),
                          expected_logits=handoff.training.record(self.logits))
        write(self.root / 'request.json', dict(format=handoff.training.FORMAT,
              amendment_root=str(self.root), gpu={'fixture': 'gpu identity'}))
        self.combined_identity = dict(pid=41, start_ticks=101, state='S', parent_pid=1,
            argv=['python', '-m', handoff.source.MODULE, '--root', str(self.root),
                  '--output', str(self.causal)])
        write(self.causal / 'request.json', dict(runner_identity=self.combined_identity))
        self.causal_summary = {'complete': True, 'fixture': 'delegated strong validator'}
        write(self.causal / 'summary.json', self.causal_summary)
        self.plan = {'immutable_plan': ['first', 'second']}
        self.old_plan = self.upstream / 'upstream_revalidation_plan' / 'plan.json'
        write(self.old_plan, self.plan)
        self.chain_prefix = [handoff.training.record(self.causal / name)
                             for name in ('request.json', 'summary.json')]
        self.ledger = dict(inputs=[*self.chain_prefix, handoff.training.record(self.old_plan)],
                           observer_identity=self.combined_identity, observer_exit_code=None)
        write(self.upstream / 'validated_upstream.json', self.ledger)
        originals, binaries = {}, {}
        for name in ('probe', 'gpu_test'):
            original = self.base / name
            original.write_bytes(('non-executable fixture: ' + name).encode())
            copied = self.upstream / 'bin' / name
            copied.parent.mkdir(exist_ok=True)
            copied.write_bytes(original.read_bytes())
            originals[name] = handoff.training.record(original)
            binaries[name] = handoff.training.record(copied)
        self.identity = dict(pid=42, start_ticks=102, state='S', parent_pid=1,
            argv=['python', '-m', handoff.MODULE, '--root', str(self.root),
                  '--output', str(self.upstream), '--causal-output', str(self.causal),
                  '--historical-root', str(self.historical_root),
                  '--historical-manifest', str(self.manifest),
                  '--historical-checkpoint', str(self.checkpoint),
                  '--probe', originals['probe']['path'],
                  '--gpu-test', originals['gpu_test']['path']])
        # Historical collectors can repeat identical files. Native execution
        # records deliberately use sorted UNIQUE paths, unlike request order.
        frozen = [handoff.training.record(self.root / 'request.json'),
                  handoff.training.record(self.causal / 'request.json'),
                  *self.historical_records, *originals.values(), *binaries.values(),
                  self.historical_records[0]]
        self.request = dict(**historical_kind(), runner_identity=self.identity,
            upstream_identity=self.combined_identity, amendment_root=str(self.root),
            causal_output=str(self.causal), gpu={'fixture': 'gpu identity'},
            assay=self.assay, binaries=binaries, frozen_inputs=frozen)
        write(self.upstream / 'request.json', self.request)
        write(self.upstream / 'state.json', dict(format=handoff.source.FORMAT, phase='complete'))
        self.gtest = self.upstream / 'gpu_validation' / 'gtest.json'
        write(self.gtest, dict(tests=1, failures=0, errors=0, disabled=0,
            testsuites=[dict(testsuite=[dict(name='Runs', status='RUN', result='COMPLETED')])]))
        self.native = self.upstream / 'native'
        (self.native / 'arm').mkdir(parents=True)
        (self.native / 'metadata.json').write_bytes(b'fixture metadata')
        (self.native / 'arm' / 'logits.f32').write_bytes(b'nested native output')
        self.test_command = [binaries['gpu_test']['path'], f'--gtest_output=json:{self.gtest}']
        self.native_command = [binaries['probe']['path'],
            f'--checkpoint={self.checkpoint}', f'--tokens_file={self.prefix}',
            f'--output_dir={self.native}', '--block=1', '--head=2', '--query=1023',
            '--sources=1022,889,1023,1021,512', '--target_id=2797',
            f'--expected_logits={self.logits}']
        self.test_execution = self.execution('gpu_test', 'pluto-paired-probe-execution-v1',
            self.test_command, self.gtest.parent, '2026-09-10T12:00:00+00:00',
            '2026-09-10T12:01:00+00:00')
        self.probe_execution = self.execution('probe', 'pluto-source-value-execution-v1',
            self.native_command, self.native, '2026-09-10T12:02:00+00:00',
            '2026-09-10T12:03:00+00:00')
        self.readout = {'complete': True, 'fixture': 'independent regenerated readout'}
        write(self.upstream / 'readout.json', self.readout)
        self.summary = dict(**historical_kind(), complete=True)
        self.refresh_summary()
        self.auth = self.start_patch('historical.authenticate_assay',
                                    return_value=(self.assay, self.historical_records))
        self.regen_readout = self.start_patch('readout.summarize', return_value=self.readout)
        self.chain = self.start_patch('source.validate_upstream', side_effect=self.regenerate_chain)
        # A regression that executes or probes hardware must fail immediately.
        self.start_patch('historical.execute_tree', side_effect=AssertionError('native execution'))
        self.start_patch('historical.causal.execute_recorded', side_effect=AssertionError('native test'))
        self.start_patch('training.require_idle_gpu', side_effect=AssertionError('GPU access'))

    def start_patch(self, attribute, **kwargs):
        patcher = mock.patch('weight_analysis.head_position_handoff.' + attribute, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def execution(self, stem, format_, command, output, start, finish):
        log = self.upstream / (stem + '.log')
        log.write_bytes(b'fixture log')
        unique = {record['path']: record for record in self.request['frozen_inputs']}
        inputs = [unique[path] for path in sorted(unique)]
        value = dict(format=format_, pid=52, command=command, returncode=0,
            started_utc=start, finished_utc=finish, log=handoff.training.record(log),
            inputs_before=inputs, inputs_after=copy.deepcopy(inputs),
            outputs=handoff.historical.recursive_outputs(output))
        write(self.upstream / (stem + '_execution.json'), value)
        return value

    def refresh_summary(self):
        self.summary.update(frozen_inputs=self.request['frozen_inputs'], assay=self.request['assay'],
            gpu_test=handoff.training.read_json(self.upstream / 'gpu_test_execution.json'),
            execution=handoff.training.record(self.upstream / 'probe_execution.json'),
            readout=handoff.training.record(self.upstream / 'readout.json'),
            upstream=handoff.training.record(self.upstream / 'validated_upstream.json'))
        write(self.upstream / 'summary.json', self.summary)

    def regenerate_chain(self, root, upstream, summary, output):
        self.assertEqual((root, upstream, summary), (self.root, self.causal, self.causal_summary))
        self.assertFalse(output.exists())
        write(output / 'plan.json', self.plan)
        return [*self.chain_prefix, handoff.training.record(output / 'plan.json')]

    def validate(self):
        return handoff.validate_upstream(self.root, self.upstream, self.summary, self.output)

    def change_execution(self, stem, value):
        write(self.upstream / (stem + '_execution.json'), value)
        self.refresh_summary()


class IdentityTest(Fixture):
    def test_exact_live_identity(self):
        with mock.patch.object(handoff.training, 'process_identity', return_value=self.identity) as process:
            self.assertEqual(handoff.observer_identity(self.root, self.upstream, 42, 102,
                                                       self.request), self.identity)
        process.assert_called_once_with(42)

    def test_rejects_process_reuse_or_changed_command(self):
        for changes in ({'pid': 43}, {'start_ticks': 103}, {'state': 'Z'},
                        {'argv': self.identity['argv'] + ['--extra']}):
            with self.subTest(changes=changes), mock.patch.object(handoff.training,
                    'process_identity', return_value={**self.identity, **changes}):
                with self.assertRaises(ValueError):
                    handoff.observer_identity(self.root, self.upstream, 42, 102, self.request)

    def test_rejects_wrong_saved_request_kind(self):
        for key, value in (('format', 'other'), ('stage', 'initial_piece'),
                           ('amendment_root', str(self.base)), ('initial_piece_assay', True),
                           ('new_paired_model_result', True), ('goal_completion_claimed', True)):
            request = {**self.request, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                handoff.observer_identity(self.root, self.upstream, 42, 102, request)

    def test_saved_argv_must_bind_exact_module_and_paths(self):
        for old, new in ((handoff.MODULE, handoff.source.MODULE),
                         (str(self.upstream), str(self.causal)),
                         (str(self.causal), str(self.upstream))):
            request = copy.deepcopy(self.request)
            argv = request['runner_identity']['argv']
            argv[argv.index(old)] = new
            with self.subTest(old=old), self.assertRaises(ValueError):
                handoff.observer_identity(self.root, self.upstream, 42, 102, request)

    def test_duplicate_or_missing_flag(self):
        for argv in (['--root', 'a', '--root', 'a'], ['--root'], []):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                handoff.flag(argv, '--root')


class WaitTest(Fixture):
    def wait(self, states):
        with mock.patch.object(handoff.training, 'process_live', side_effect=states) as process, \
             mock.patch.object(handoff.time, 'sleep'):
            result = handoff.wait_for_source(self.upstream / 'summary.json', self.identity,
                                             poll_seconds=1)
        return result, process.call_count

    def test_summary_does_not_replace_two_actual_consecutive_exits(self):
        result, calls = self.wait([True, False, True, False, False])
        self.assertEqual(result, self.summary)
        self.assertEqual(calls, 5)

    def test_transient_read_error_is_not_an_exit_and_resets_exit_count(self):
        result, calls = self.wait([False, OSError('transient'), False, False])
        self.assertEqual(result, self.summary)
        self.assertEqual(calls, 4)

    def test_repeated_handle_read_errors_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'cannot verify'):
            self.wait([ValueError('malformed proc')] * 4)

    def test_pid_reuse_fails_immediately(self):
        with self.assertRaisesRegex(RuntimeError, 'PID reused'):
            self.wait([RuntimeError('PID reused')])

    def test_failure_marker_even_with_summary(self):
        write(self.upstream / 'failure.json', {'failure': True})
        with self.assertRaisesRegex(RuntimeError, 'stage failed'):
            self.wait([])

    def test_no_summary_after_exit(self):
        (self.upstream / 'summary.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'without completed evidence'):
            self.wait([False] * 4)

    def test_incomplete_summary_and_initial_piece_relabel_rejected(self):
        for change in ({'complete': False}, {'initial_piece_assay': True}):
            write(self.upstream / 'summary.json', {**self.summary, **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.wait([False, False])

    def test_invalid_poll(self):
        for poll in (0, -1, 61):
            with self.subTest(poll=poll), self.assertRaises(ValueError):
                handoff.wait_for_source(self.upstream / 'summary.json', self.identity,
                                         poll_seconds=poll)


class ValidateTest(Fixture):
    def test_complete_native_and_full_chain_revalidated(self):
        before = {str(path): path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        records = self.validate()
        self.chain.assert_called_once_with(self.root, self.causal, self.causal_summary, self.output)
        self.auth.assert_called_once_with(self.manifest, self.historical_root, self.checkpoint)
        self.regen_readout.assert_called_once_with(self.native,
            str(self.upstream / 'probe_execution.json'), self.assay)
        by_path = {record['path']: record for record in records}
        self.assertEqual(len(records), len(by_path))
        for path in (self.old_plan, self.output / 'plan.json',
                     self.native / 'arm' / 'logits.f32', self.upstream / 'validated_upstream.json'):
            self.assertEqual(by_path[str(path)], handoff.training.record(path))
        for path, data in before.items():
            self.assertEqual(Path(path).read_bytes(), data)

    def test_existing_or_overlapping_output_not_written(self):
        for path in (self.upstream, self.upstream / 'new-child', self.root):
            self.output = path
            with self.subTest(path=path), self.assertRaises((ValueError, FileExistsError)):
                self.validate()
        self.chain.assert_not_called()

    def test_failure_marker_rejected(self):
        write(self.upstream / 'failure.json', {'failed': True})
        with self.assertRaisesRegex(ValueError, 'failure marker'):
            self.validate()

    def test_summary_must_equal_saved_bytes(self):
        self.summary = {**self.summary, 'not_saved': True}
        with self.assertRaisesRegex(ValueError, 'changed after handoff'):
            self.validate()

    def test_state_must_be_complete(self):
        write(self.upstream / 'state.json', dict(format=handoff.source.FORMAT, phase='running'))
        with self.assertRaisesRegex(ValueError, 'complete state'):
            self.validate()

    def test_summary_cannot_relabel_upstream_as_initial_piece(self):
        self.summary['initial_piece_assay'] = True
        write(self.upstream / 'summary.json', self.summary)
        with self.assertRaisesRegex(ValueError, 'historical suffix'):
            self.validate()

    def test_request_frozen_ledger_is_exact_in_summary(self):
        self.summary['frozen_inputs'] = self.summary['frozen_inputs'][:-1]
        write(self.upstream / 'summary.json', self.summary)
        with self.assertRaisesRegex(ValueError, 'summary/request disagree'):
            self.validate()

    def test_frozen_input_mutation_rejected(self):
        self.prefix.write_bytes(b'changed input')
        with self.assertRaises(ValueError):
            self.validate()
        self.chain.assert_not_called()

    def test_historical_assay_is_reauthenticated(self):
        self.auth.return_value = ({**self.assay, 'target_id': 1475}, self.historical_records)
        with self.assertRaisesRegex(ValueError, 'authenticated fixed historical'):
            self.validate()

    def test_native_command_cannot_change_target_or_scope(self):
        self.probe_execution['command'] = [*self.native_command[:-2], '--target_id=1475',
                                            self.native_command[-1]]
        self.change_execution('probe', self.probe_execution)
        with self.assertRaisesRegex(ValueError, 'identity/command'):
            self.validate()

    def test_execution_must_include_all_frozen_input_paths(self):
        self.probe_execution['inputs_before'] = self.probe_execution['inputs_before'][1:]
        self.probe_execution['inputs_after'] = self.probe_execution['inputs_before']
        self.change_execution('probe', self.probe_execution)
        with self.assertRaisesRegex(ValueError, 'input ledger differs'):
            self.validate()

    def test_execution_cannot_mutate_inputs(self):
        self.probe_execution['inputs_after'] = self.probe_execution['inputs_after'][1:]
        self.change_execution('probe', self.probe_execution)
        with self.assertRaisesRegex(ValueError, 'input ledger differs'):
            self.validate()

    def test_unrecorded_recursive_native_artifact_rejected(self):
        (self.native / 'arm' / 'unrecorded.f32').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'output/log ledger'):
            self.validate()

    def test_changed_recorded_recursive_native_artifact_rejected(self):
        (self.native / 'arm' / 'logits.f32').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'output/log ledger'):
            self.validate()

    def test_skipped_gtest_is_not_a_successful_native_validation(self):
        test = handoff.training.read_json(self.gtest)
        test['testsuites'][0]['testsuite'][0]['result'] = 'SKIPPED'
        write(self.gtest, test)
        self.test_execution['outputs'] = handoff.historical.recursive_outputs(self.gtest.parent)
        self.change_execution('gpu_test', self.test_execution)
        with self.assertRaises(ValueError):
            self.validate()
        self.regen_readout.assert_not_called()

    def test_test_must_finish_before_native_probe(self):
        self.test_execution['finished_utc'] = '2026-09-10T12:04:00+00:00'
        self.change_execution('gpu_test', self.test_execution)
        with self.assertRaisesRegex(ValueError, 'record/order'):
            self.validate()

    def test_failed_child_record_rejected_even_if_summary_copied(self):
        self.probe_execution['returncode'] = 1
        self.change_execution('probe', self.probe_execution)
        with self.assertRaisesRegex(ValueError, 'identity/command'):
            self.validate()

    def test_naive_execution_timestamp_rejected(self):
        self.probe_execution['started_utc'] = '2026-09-10T12:02:00'
        self.change_execution('probe', self.probe_execution)
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            self.validate()

    def test_saved_readout_must_match_independent_regeneration(self):
        self.regen_readout.return_value = {**self.readout, 'changed': True}
        with self.assertRaisesRegex(ValueError, 'independently regenerated'):
            self.validate()

    def test_readout_validation_error_propagates(self):
        self.regen_readout.side_effect = ValueError('native calibration failed')
        with self.assertRaisesRegex(ValueError, 'native calibration failed'):
            self.validate()
        self.chain.assert_not_called()

    def test_no_invented_outside_child_exit_code(self):
        self.ledger['observer_exit_code'] = 0
        write(self.upstream / 'validated_upstream.json', self.ledger)
        self.refresh_summary()
        with self.assertRaisesRegex(ValueError, 'outside-child exit'):
            self.validate()

    def test_actual_combined_observer_is_bound(self):
        changed = copy.deepcopy(self.combined_identity)
        changed['start_ticks'] += 1
        self.request['upstream_identity'] = changed
        write(self.upstream / 'request.json', self.request)
        self.ledger['observer_identity'] = changed
        write(self.upstream / 'validated_upstream.json', self.ledger)
        self.refresh_summary()
        with self.assertRaisesRegex(ValueError, 'actual combined publisher'):
            self.validate()

    def test_full_chain_revalidation_failure_propagates(self):
        self.chain.side_effect = ValueError('paired-training checkpoint mismatch')
        with self.assertRaisesRegex(ValueError, 'paired-training checkpoint mismatch'):
            self.validate()

    def test_regenerated_plan_must_have_identical_bytes(self):
        self.plan = {'immutable_plan': ['changed']}
        with self.assertRaisesRegex(ValueError, 'regenerated causal plans differ'):
            self.validate()

    def test_saved_chain_ledger_must_preserve_exact_order(self):
        self.ledger['inputs'].reverse()
        write(self.upstream / 'validated_upstream.json', self.ledger)
        self.refresh_summary()
        with self.assertRaisesRegex(ValueError, 'revalidated causal-chain evidence'):
            self.validate()

    def test_saved_chain_cannot_drop_an_authenticated_record(self):
        self.ledger['inputs'] = self.ledger['inputs'][1:]
        write(self.upstream / 'validated_upstream.json', self.ledger)
        self.refresh_summary()
        with self.assertRaisesRegex(ValueError, 'revalidated causal-chain evidence'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
