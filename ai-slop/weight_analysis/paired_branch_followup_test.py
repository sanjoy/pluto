"""CPU-only observer/safety tests; all model invocations are mocked."""

import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from . import paired_branch_followup as branch


class BranchFollowupTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.summary_path = self.root / 'summary.json'
        self.identity = {'pid': 123, 'start_ticks': 456,
                         'argv': ['python', '-m', 'weight_analysis.paired_causal_followup',
                                  '--root', str(self.root), '--output', str(self.root / 'embedding')]}

    def complete(self):
        branch.write_json(self.summary_path, {'format': 'pluto-paired-causal-followup-v1',
            'complete': True, 'stage': 'embedding', 'goal_completion_claimed': False})

    def test_waits_for_exact_observer_exit_even_after_summary_is_present(self):
        self.complete()
        with mock.patch.object(branch.paired_analysis, '_process_identity',
                side_effect=[self.identity, FileNotFoundError(), FileNotFoundError()]), \
             mock.patch.object(branch.causal, 'process_still_live', return_value=False), \
             mock.patch.object(branch.time, 'sleep') as sleep:
            result = branch.wait_for_embedding(self.summary_path, self.identity)
        self.assertTrue(result['complete'])
        self.assertEqual(sleep.call_count, 2)

    def test_observation_timeout_or_transient_error_does_not_restart_or_assume_exit(self):
        self.complete()
        with mock.patch.object(branch.paired_analysis, '_process_identity', side_effect=[
                OSError('transient'), FileNotFoundError(), FileNotFoundError(), FileNotFoundError()]) as identity, \
             mock.patch.object(branch.causal, 'process_still_live', side_effect=[True, False, False]), \
             mock.patch.object(branch.time, 'sleep') as sleep, \
             mock.patch.object(branch.causal.subprocess, 'Popen') as process:
            branch.wait_for_embedding(self.summary_path, self.identity)
        self.assertEqual(identity.call_args_list, [mock.call(123)] * 4)
        self.assertEqual(sleep.call_count, 3)
        process.assert_not_called()

    def test_missing_handle_without_summary_is_rechecked_four_times(self):
        with mock.patch.object(branch.paired_analysis, '_process_identity', side_effect=FileNotFoundError()) as identity, \
             mock.patch.object(branch.causal, 'process_still_live', return_value=False), \
             mock.patch.object(branch.time, 'sleep'), \
             mock.patch.object(branch.causal.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(RuntimeError, 'missing'):
                branch.wait_for_embedding(self.summary_path, self.identity)
        self.assertEqual(identity.call_count, 4)
        process.assert_not_called()

    def test_pid_reuse_or_changed_command_is_rejected(self):
        for changes in ({'start_ticks': 999}, {'argv': ['other']}):
            with mock.patch.object(branch.paired_analysis, '_process_identity',
                                   return_value={**self.identity, **changes}):
                with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                    branch.wait_for_embedding(self.summary_path, self.identity)

    def test_failed_or_wrong_stage_completion_cannot_unlock_gpu(self):
        self.complete()
        branch.write_json(self.root / 'failure.json', {'error': 'failure'})
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            branch.wait_for_embedding(self.summary_path, None)
        (self.root / 'failure.json').unlink()
        for changes in ({'stage': 'training'}, {'complete': False}, {'goal_completion_claimed': True}):
            value = branch.read_json(self.summary_path)
            value.update(changes)
            self.summary_path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                branch.wait_for_embedding(self.summary_path, None)
        for interval in (0, -1, 61, float('nan')):
            with self.assertRaises(ValueError):
                branch.wait_for_embedding(self.summary_path, None, poll_seconds=interval)

    def test_queue_identity_checks_module_root_output_and_ticks(self):
        with mock.patch.object(branch.paired_analysis, '_process_identity', return_value=self.identity):
            self.assertEqual(branch.observer_identity(123, 456, self.root, self.root / 'embedding'), self.identity)
            for ticks, root, output in ((999, self.root, self.root / 'embedding'),
                                        (456, self.root / 'other', self.root / 'embedding'),
                                        (456, self.root, self.root / 'other')):
                with self.assertRaises(ValueError):
                    branch.observer_identity(123, ticks, root, output)
        for argv in (['python', '-m'], ['python', '-m', 'wrong'],
                     self.identity['argv'] + ['--root', str(self.root)]):
            with mock.patch.object(branch.paired_analysis, '_process_identity',
                                   return_value={**self.identity, 'argv': argv}):
                with self.assertRaises(ValueError):
                    branch.observer_identity(123, 456, self.root, self.root / 'embedding')

    def runner_fixture(self):
        root = self.root / 'experiment'
        root.mkdir()
        embedding = root / 'arbitrary_embedding_output_name'
        embedding.mkdir()
        branch.write_json(embedding / 'request.json', {'stage': 'embedding', 'root': str(root)})
        branch.write_json(root / 'manifest.json', {})
        for name in ('word_cases', 'supplemental_cases'):
            directory = root / name
            directory.mkdir()
            (directory / 'batch.bin').write_bytes(b'unchanged batch')
            branch.write_json(directory / 'cases.json', {'packed_batch': branch.record(directory / 'batch.bin')})
        (root / 'analysis_trajectory').mkdir()
        branch.write_json(root / 'analysis_trajectory' / 'summary.json', {'complete': True})
        recipient = {'path': str(root / 'original' / 'checkpoints' / 'step_100'), 'step': 100, 'sha256': {}}
        donor = {'path': str(root / 'replacement' / 'checkpoints' / 'step_100'), 'step': 100, 'sha256': {}}
        selected = [{'name': 'whole', 'step': 100, 'kind': 'attention_whole_branch',
                     'recipient_checkpoint': recipient, 'donor_checkpoint': donor,
                     'recipient_arm': 'original', 'donor_arm': 'replacement',
                     'tensors': ['blocks.0.ln1.scale'], 'embedding_rows': []},
                    {'name': 'writes', 'step': 100, 'kind': 'mlp_output_write',
                     'recipient_checkpoint': donor, 'donor_checkpoint': recipient,
                     'recipient_arm': 'replacement', 'donor_arm': 'original',
                     'tensors': ['blocks.0.mlp.output.weight'], 'embedding_rows': []}]
        plan = {'interventions': selected,
                'analysis_summary': branch.record(root / 'analysis_trajectory' / 'summary.json')}
        branch.write_json(embedding / 'plan.json', plan)
        summary = {'format': 'pluto-paired-causal-followup-v1', 'complete': True,
                   'stage': 'embedding', 'goal_completion_claimed': False,
                   'plan': branch.record(embedding / 'plan.json'), 'completed_artifact_records': []}
        branch.write_json(embedding / 'summary.json', summary)
        baselines = {}
        for path in (recipient['path'], donor['path']):
            directory = embedding / Path(path).parent.parent.name
            directory.mkdir()
            branch.write_json(directory / 'complete.json', {'complete': True})
            baselines[path] = {'scores': {name: directory / name for name in ('main', 'supplemental')},
                               'executions': {name: directory / (name + '.json') for name in ('main', 'supplemental')},
                               'completion': branch.record(directory / 'complete.json')}
        binary = self.root / 'fake_binary'
        binary.write_bytes(b'not an executable GPU program')
        source = self.root / 'fake_source.py'
        source.write_text('# frozen fixture source\n')
        args = SimpleNamespace(root=root, embedding_output=embedding, output=root / 'branches',
                               loss_probe=binary, observer_pid=None, observer_start_ticks=None)
        return args, plan, selected, baselines, source

    def test_sequential_full_screen_runs_only_after_gate_and_uses_step_directories(self):
        args, plan, selected, baselines, source = self.runner_fixture()
        calls, executed = [], []

        def gate(*_):
            calls.append('gate')
            return plan, selected, baselines

        def patch(recipient, donor, output, **kwargs):
            calls.append('patch')
            self.assertEqual(output.name, 'step_100')  # Native parser rejects plain "checkpoint".
            output.mkdir()
            (output / 'weight_0.bin').write_bytes(b'patched bytes')
            branch.write_json(output / 'patch.json', kwargs)

        def execute(command, inputs, output, log_path, execution_path):
            calls.append('execute')
            executed.append(command)
            self.assertIn('--batch_sequences=1', command)
            self.assertTrue(any(arg.endswith('/step_100') for arg in command if arg.startswith('--checkpoint=')))
            output.mkdir()
            branch.write_json(output / 'metadata.json', {'complete': True})
            log_path.write_text('fake child completed\n')
            branch.write_json(execution_path, {'returncode': 0})

        def analyze(item, patch_path, main, supplemental, scores, output, *, executions):
            calls.append('readout')
            self.assertEqual(set(scores), {'recipient', 'donor', 'patched'})
            self.assertEqual(set(executions['patched']), {'main', 'supplemental'})
            branch.write_json(output, {'complete': True, 'intervention': item})

        with mock.patch.object(branch, '_readout_module', return_value=SimpleNamespace(analyze=analyze)), \
             mock.patch.object(branch, '_source_paths', return_value=[source]), \
             mock.patch.object(branch.paired_training, 'verify_frozen_inputs'), \
             mock.patch.object(branch, '_validate_upstream', side_effect=gate), \
             mock.patch.object(branch.paired_analysis, '_verify_checkpoint'), \
             mock.patch.object(branch.paired_weight_patch, 'create_patch', side_effect=patch), \
             mock.patch.object(branch, 'execute_recorded', side_effect=execute):
            result = branch.run(args)
        self.assertEqual(calls, ['gate', 'patch', 'execute', 'execute', 'readout',
                                 'patch', 'execute', 'execute', 'readout'])
        self.assertEqual(len(executed), 4)
        self.assertEqual(len(result['completed']), 2)
        self.assertTrue(result['complete'])
        self.assertFalse(result['goal_completion_claimed'])
        self.assertEqual(result['stage'], 'branches')
        self.assertFalse((args.output / 'failure.json').exists())
        self.assertTrue((args.output / 'validated_upstream.json').exists())
        self.assertEqual(branch.read_json(args.output / 'validated_upstream.json')['observer_exit_code'], None)

    def test_failed_gate_source_change_or_interrupt_never_launches_gpu(self):
        for mode in ('gate', 'source', 'interrupt'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=self.root) as temporary:
                # Isolate each runner fixture without reusing a previous output.
                previous_root = self.root
                self.root = Path(temporary)
                args, plan, selected, baselines, source = self.runner_fixture()
                self.root = previous_root
                def wait(*_):
                    if mode == 'source':
                        source.write_text('# changed while queued\n')
                    if mode == 'interrupt':
                        raise KeyboardInterrupt()
                    return branch.read_json(args.embedding_output / 'summary.json')
                with mock.patch.object(branch, '_readout_module', return_value=SimpleNamespace()), \
                     mock.patch.object(branch, '_source_paths', return_value=[source]), \
                     mock.patch.object(branch.paired_training, 'verify_frozen_inputs'), \
                     mock.patch.object(branch, 'wait_for_embedding', side_effect=wait), \
                     mock.patch.object(branch, '_validate_upstream', side_effect=RuntimeError('gate failed')), \
                     mock.patch.object(branch, 'execute_recorded') as execute, \
                     mock.patch.object(branch.paired_weight_patch, 'create_patch') as patch:
                    with self.assertRaises((ValueError, RuntimeError, KeyboardInterrupt)):
                        branch.run(args)
                execute.assert_not_called()
                patch.assert_not_called()
                self.assertTrue((args.output / 'failure.json').exists())
                self.assertFalse((args.output / 'summary.json').exists())

    def test_reused_completed_output_and_input_overlaps_are_refused(self):
        args, _, _, _, source = self.runner_fixture()
        with mock.patch.object(branch, '_readout_module', return_value=SimpleNamespace()):
            for output in (args.root, args.embedding_output, args.root / 'word_cases',
                           args.embedding_output / 'child', args.root / 'word_cases' / 'child'):
                args.output = output
                with self.assertRaises((FileExistsError, ValueError)):
                    branch.run(args)

    def test_execution_cleanup_terminates_and_reaps_only_its_owned_child(self):
        incoming = self.root / 'input'
        incoming.write_bytes(b'input')
        child = mock.Mock(pid=123)
        child.wait.side_effect = [KeyboardInterrupt(), subprocess.TimeoutExpired('fake', 30), 0]
        with mock.patch.object(branch.causal.subprocess, 'Popen', return_value=child):
            with self.assertRaises(KeyboardInterrupt):
                branch.execute_recorded(['not-executed'], [incoming], self.root / 'outputs',
                                         self.root / 'child.log', self.root / 'execution.json')
        child.terminate.assert_called_once()
        child.kill.assert_called_once()
        self.assertEqual(child.wait.call_count, 3)
        self.assertFalse((self.root / 'execution.json').exists())

    def test_execution_validation_checks_actual_bytes_not_success_label(self):
        incoming, output, log = (self.root / name for name in ('input', 'output', 'log'))
        incoming.write_bytes(b'input')
        output.write_bytes(b'output')
        log.write_bytes(b'log')
        execution_path = self.root / 'execution.json'
        branch.write_json(execution_path, {'format': 'pluto-paired-probe-execution-v1', 'returncode': 0,
            'inputs_before': [branch.record(incoming)], 'inputs_after': [branch.record(incoming)],
            'outputs': [branch.record(output)], 'log': branch.record(log)})
        branch._validate_execution(execution_path)
        output.write_bytes(b'changed output')
        with self.assertRaisesRegex(ValueError, 'changed'):
            branch._validate_execution(execution_path)

    def validation_fixture(self):
        """Real hashed evidence graph; only training/GPU producers are mocked."""
        root = self.root / 'validation'
        root.mkdir()
        embedding = root / 'embedding_v2'
        embedding.mkdir()
        trajectory_dir = root / 'analysis_trajectory'
        trajectory_dir.mkdir()
        for name in ('manifest.json', 'state.json'):
            branch.write_json(root / name, {})
        pair = {'name': 'final', **{arm: {'step': 100,
                    'path': str(root / arm / 'checkpoints' / 'step_100'), 'sha256': {}}
                    for arm in ('original', 'replacement')}}
        completion = {'pairs': [pair], 'all_matched_steps': True,
                      'matched_steps': [100], 'matched_step': 100, 'initial': {}}
        trajectory = {'format': 'pluto-paired-analysis-v1', 'complete': True,
                      'plan': completion, 'behavior': {}}
        for arm in ('original', 'replacement'):
            directory = trajectory_dir / arm
            directory.mkdir()
            scores = {}
            for key, filename in (('losses', 'losses.f32.bin'), ('argmax', 'argmax.i32.bin')):
                (directory / filename).write_bytes(b'complete native baseline ' + arm.encode())
                scores[key] = branch.record(directory / filename)
            branch.write_json(directory / 'readout.json', {'scores': scores})
            trajectory['behavior'][pair[arm]['path']] = {'report': branch.record(directory / 'readout.json')}
        branch.write_json(trajectory_dir / 'summary.json', trajectory)
        # The helper returns tuples in specs; JSON-normalization must not make
        # the validated plan falsely disagree with the immutable serialized one.
        interventions = branch.paired_intervention_plan._interventions(
            pair, branch.checkpoint.tensor_manifest(), [45, 68, 303, 400, 1475, 2797, 3109, 21733, 45177])
        interventions = json.loads(json.dumps(interventions))
        plan = {'format': branch.paired_intervention_plan.FORMAT, 'complete': True,
                'patches_materialized': False, 'interventions': interventions,
                'word_piece_ids': [45, 68, 303, 400, 1475, 2797, 3109, 21733, 45177],
                'analysis_summary': branch.record(trajectory_dir / 'summary.json'), 'frozen_inputs': []}
        branch.write_json(embedding / 'plan.json', plan)
        binary = root / 'binary'
        binary.write_bytes(b'fixture executable identity, never executed')
        completed, artifacts = [], []

        def execution(path, outputs):
            log = path.with_suffix('.log')
            log.write_bytes(b'completed child')
            value = {'format': 'pluto-paired-probe-execution-v1', 'returncode': 0,
                     'inputs_before': [branch.record(binary)], 'inputs_after': [branch.record(binary)],
                     'outputs': [branch.record(output) for output in outputs], 'log': branch.record(log)}
            branch.write_json(path, value)
            return value

        for item in interventions:
            if item['tensors']:
                continue
            stage = embedding / item['name']
            stage.mkdir()
            checkpoint = stage / 'step_100'
            checkpoint.mkdir()
            branch.write_json(checkpoint / 'patch.json', {'selection': {'tensors': [], 'embedding_rows': []}})
            evidence = {'intervention': item, 'patch': branch.record(checkpoint / 'patch.json')}
            if not item['embedding_rows']:
                suite = stage / 'copy_scores'
                suite.mkdir()
                arm = item['recipient_arm']
                outputs = []
                for filename in ('losses.f32.bin', 'argmax.i32.bin'):
                    target = suite / filename
                    target.write_bytes((trajectory_dir / arm / filename).read_bytes())
                    outputs.append(target)
                execution(stage / 'copy_execution.json', outputs)
                suite = stage / 'supplemental_scores'
                suite.mkdir()
                (suite / 'scores.bin').write_bytes(b'supplemental bytes')
                execution(stage / 'supplemental_execution.json', [suite / 'scores.bin'])
                branch.write_json(stage / 'supplemental_readout.json', {'complete': True})
                evidence['supplemental'] = branch.record(stage / 'supplemental_readout.json')
                evidence['copy_control_all_scores_byte_equal'] = True
            evidence['artifacts'] = [branch.record(path) for path in sorted(stage.rglob('*')) if path.is_file()]
            branch.write_json(stage / 'complete.json', evidence)
            completed.append(evidence)
            artifacts += [*evidence['artifacts'], branch.record(stage / 'complete.json')]
        tests = embedding / 'gpu_validation'
        tests.mkdir()
        result = {'tests': 1, 'failures': 0, 'errors': 0, 'disabled': 0,
                  'testsuites': [{'testsuite': [{'status': 'RUN', 'result': 'COMPLETED'}]}]}
        branch.write_json(tests / 'gtest.json', result)
        gpu_test = execution(embedding / 'gpu_execution.json', [tests / 'gtest.json'])
        summary = {'format': 'pluto-paired-causal-followup-v1', 'complete': True,
                   'stage': 'embedding', 'goal_completion_claimed': False,
                   'frozen_inputs': [], 'completed_artifact_records': artifacts,
                   'plan': branch.record(embedding / 'plan.json'), 'completed': completed,
                   'gpu_test': gpu_test, 'deferred_interventions': [item for item in interventions if item['tensors']]}
        return root, embedding, summary, completion

    def validate(self, root, embedding, summary, completion):
        with mock.patch.object(branch.paired_training, 'verify_frozen_inputs'), \
             mock.patch.object(branch.paired_analysis, 'validate_completion', return_value=completion), \
             mock.patch.object(branch.paired_analysis, 'validate_determinism_gate'):
            return branch._validate_upstream(root, embedding, summary)

    def test_full_upstream_graph_validates_every_stage_and_serialized_tensor_specs(self):
        root, embedding, summary, completion = self.validation_fixture()
        plan, selected, baselines = self.validate(root, embedding, summary, completion)
        self.assertEqual(len(selected), 64)  # One positive step, both directions, all 8 blocks.
        self.assertEqual(len(baselines), 2)
        self.assertEqual({item['kind'] for item in selected}, {
            'attention_whole_branch', 'attention_output_write', 'mlp_whole_branch', 'mlp_output_write'})
        self.assertTrue(all(value['scores']['main'].name == 'copy_scores' for value in baselines.values()))

    def test_upstream_corrupted_artifact_missing_completion_and_deferred_plan_are_rejected(self):
        root, embedding, summary, completion = self.validation_fixture()
        original = json.loads(json.dumps(summary))
        path = Path(summary['completed_artifact_records'][0]['path'])
        contents = path.read_bytes()
        path.write_bytes(contents + b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.validate(root, embedding, summary, completion)
        path.write_bytes(contents)
        summary['completed'] = summary['completed'][1:]
        with self.assertRaisesRegex(ValueError, 'every predeclared'):
            self.validate(root, embedding, summary, completion)
        summary = json.loads(json.dumps(original))
        summary['deferred_interventions'] = summary['deferred_interventions'][:-1]
        with self.assertRaisesRegex(ValueError, 'different branch screen'):
            self.validate(root, embedding, summary, completion)

    def test_actual_copy_scores_must_still_match_trajectory_even_if_artifacts_rehashed(self):
        root, embedding, summary, completion = self.validation_fixture()
        evidence = next(item for item in summary['completed'] if not item['intervention']['embedding_rows'])
        stage = embedding / evidence['intervention']['name']
        loss = stage / 'copy_scores' / 'losses.f32.bin'
        loss.write_bytes(b'not the trajectory baseline')
        evidence['artifacts'] = [branch.record(item['path']) for item in evidence['artifacts']]
        (stage / 'complete.json').write_text(json.dumps(evidence))
        summary['completed_artifact_records'] = [branch.record(item['path'])
                                                 for item in summary['completed_artifact_records']]
        with self.assertRaisesRegex(ValueError, 'full trajectory scores'):
            self.validate(root, embedding, summary, completion)

    def test_gpu_test_completion_cannot_be_all_skipped(self):
        root, embedding, summary, completion = self.validation_fixture()
        test_path = Path(summary['gpu_test']['outputs'][0]['path'])
        test_path.write_text(json.dumps({'tests': 0, 'failures': 0, 'errors': 0, 'disabled': 0}))
        summary['gpu_test']['outputs'][0] = branch.record(test_path)
        with self.assertRaisesRegex(ValueError, 'did not pass'):
            self.validate(root, embedding, summary, completion)

    def test_cli_identity_pairing_and_dispatch(self):
        argv = ['--root', 'root', '--embedding-output', 'embedding', '--output', 'new', '--loss-probe', 'probe']
        with mock.patch.object(branch, 'run') as run, mock.patch.object(branch.signal, 'signal'):
            branch.main(argv + ['--observer-pid', '123', '--observer-start-ticks', '456'])
            self.assertEqual(run.call_args.args[0].observer_pid, 123)
            self.assertEqual(run.call_args.args[0].observer_start_ticks, 456)
            with self.assertRaises(SystemExit):
                branch.main(argv + ['--observer-pid', '123'])


if __name__ == '__main__':
    unittest.main()
