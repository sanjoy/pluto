"""CPU mocks for completion planning, live-handle gates, and safe orchestration."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from . import paired_analysis as analysis


def dump(path, value):
    path.write_text(json.dumps(value))


def identity(pid, root, *, child=False, parent=10, ticks=100):
    return {'pid': pid, 'parent_pid': parent, 'start_ticks': ticks,
            'argv': ['trainer', '--arg=1'] if child else
                    ['python', '-u', '-m', 'scripts.weight_analysis.paired_training',
                     'run', '--output', str(root)], 'started_unix': 1000.}


def live_state():
    timestamp = datetime.fromtimestamp(1000, timezone.utc).isoformat()
    return {'phase': 'original', 'runner_pid': 10, 'started_utc': timestamp,
            'runs': {'original': {'pid': 20, 'started_utc': timestamp,
                                  'command': ['trainer', '--arg=1']}}}


class PairedAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def completed_fixture(self, original_step=200, replacement_step=230):
        manifest = {'root': str(self.root), 'seconds_per_arm': 14400,
                    'flags': {'seed': 17, 'mode': 'train_model'},
                    'binaries': {'trainer': {'path': '/frozen/trainer'}},
                    'inputs': {f'{arm}.full': {'text': f'/frozen/{arm}.txt'}
                               for arm in ('original', 'replacement')}}
        state = {'phase': 'training_complete', 'runs': {}}
        inventories = {}
        for arm, last in (('initial', 0), ('original', original_step), ('replacement', replacement_step)):
            directory = self.root / arm
            directory.mkdir()
            checkpoints = directory / 'checkpoints'
            checkpoints.mkdir()
            steps = sorted({0, last} | ({100, 200} if last >= 200 else set()))
            records = []
            for step in steps:
                path = checkpoints / f'step_{step}'
                path.mkdir()
                records.append({'step': step, 'path': str(path),
                                'sha256': {'weight_0.bin': 'initial' if step == 0 else f'{arm}:{step}'}})
            dump(directory / 'checkpoints.json', records)
            inventories[arm] = {item['step']: item for item in records}
            log = (f'training stopped at step: {last}\ntraining stop reason: time_limit\n'
                   'training elapsed seconds: 14400.2\nfinal training loss: 1.2\nfinal test loss: 2.3\n')
            (directory / 'train.log').write_text(log)
            state['runs'][arm] = {'returncode': 0, 'initial_weights_match': True,
                                  'command': analysis.paired_training.training_command(manifest, arm,
                                               'original' if arm == 'initial' else arm, seconds=14400),
                                  'final_step': last, 'stop_reason': 'time_limit',
                                  'training_elapsed_seconds': 14400.2, 'training_loss': 1.2,
                                  'test_loss': 2.3, 'final_checkpoint': records[-1]['path']}
        dump(self.root / 'manifest.json', manifest)
        dump(self.root / 'state.json', state)
        case_dir = self.root / 'word_cases'
        case_dir.mkdir()
        packed = case_dir / 'packed_cases.bin'
        packed.write_bytes(b'fake fixture batch')
        cases = case_dir / 'cases.json'
        dump(cases, {'manifest': analysis._record(self.root / 'manifest.json'),
                     'packed_batch': analysis._record(packed)})
        probe = self.root / 'frozen_probe'
        probe.write_bytes(b'not executable: all GPU calls must be mocked')
        return manifest, state, inventories, cases, probe

    def test_plan_common_positive_step_and_checkpoint_deduplication(self):
        _, state, inventories, _, _ = self.completed_fixture()
        result = analysis.plan_comparisons(inventories['original'], inventories['replacement'], 200, 230)
        self.assertEqual(result['matched_step'], 200)
        self.assertFalse(result['matched_is_final_pair'])
        self.assertEqual([pair['name'] for pair in result['pairs']], ['final', 'matched_step'])
        # Original step200 participates in both comparisons, but is scored once.
        self.assertEqual(len(result['checkpoints']), 3)

    def test_equal_final_steps_need_one_pair_and_two_probes(self):
        _, _, inventories, _, _ = self.completed_fixture(replacement_step=200)
        result = analysis.plan_comparisons(inventories['original'], inventories['replacement'], 200, 200)
        self.assertTrue(result['matched_is_final_pair'])
        self.assertEqual(len(result['pairs']), 1)
        self.assertEqual(len(result['checkpoints']), 2)

    def test_all_matched_steps_include_initialization_and_reuse_endpoint_scores(self):
        _, _, inventories, _, _ = self.completed_fixture()
        result = analysis.plan_comparisons(inventories['original'], inventories['replacement'],
                                          200, 230, all_matched_steps=True)
        self.assertEqual(result['matched_steps'], [0, 100, 200])
        self.assertEqual([pair['name'] for pair in result['pairs']],
                         ['final', 'initial', 'matched_step_100', 'matched_step_200'])
        self.assertEqual(len(result['checkpoints']), 6)
        self.assertEqual(result['behavior_aliases'], {
            inventories['replacement'][0]['path']: inventories['original'][0]['path']})
        same_endpoint = analysis.plan_comparisons(
            inventories['original'], inventories['replacement'], 200, 200,
            all_matched_steps=True)
        self.assertEqual(same_endpoint['matched_steps'], [0, 100, 200])
        self.assertEqual([pair['name'] for pair in same_endpoint['pairs']],
                         ['final', 'initial', 'matched_step_100'])
        self.assertEqual(len(same_endpoint['checkpoints']), 5)

    def test_all_matched_steps_cli_is_opt_in(self):
        arguments = [value for name in ('root', 'cases', 'probe', 'output')
                     for value in ('--' + name, '/unused')]
        with mock.patch.object(analysis, 'analyze') as analyze, \
                mock.patch.object(analysis.signal, 'signal'):
            analysis.main(arguments)
            self.assertFalse(analyze.call_args.kwargs['all_matched_steps'])
            analysis.main(arguments + ['--all-matched-steps'])
            self.assertTrue(analyze.call_args.kwargs['all_matched_steps'])

    def test_no_common_positive_step_is_reported_not_fabricated(self):
        result = analysis.plan_comparisons({0: {}, 3: {'path': 'a'}},
                                            {0: {}, 5: {'path': 'b'}}, 3, 5)
        self.assertIsNone(result['matched_step'])
        self.assertEqual(len(result['pairs']), 1)
        with self.assertRaisesRegex(ValueError, 'absent'):
            analysis.plan_comparisons({0: {}}, {0: {}}, 1, 2)

    def test_complete_state_revalidated_against_logs_commands_and_initial_hashes(self):
        manifest, state, _, _, _ = self.completed_fixture()
        result = analysis.validate_completion(self.root, state, manifest)
        self.assertEqual(result['initial']['step'], 0)
        self.assertEqual(result['matched_step'], 200)
        for key, bad in (('returncode', 1), ('initial_weights_match', False),
                          ('stop_reason', 'step_limit'), ('training_elapsed_seconds', 14399.),
                          ('training_loss', float('nan')), ('command', ['wrong trainer'])):
            changed = json.loads(json.dumps(state))
            changed['runs']['original'][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                analysis.validate_completion(self.root, changed, manifest)
        records_path = self.root / 'replacement' / 'checkpoints.json'
        records = json.loads(records_path.read_text())
        records[0]['sha256']['weight_0.bin'] = 'wrong initialization'
        dump(records_path, records)
        with self.assertRaisesRegex(ValueError, 'shared initialization'):
            analysis.validate_completion(self.root, state, manifest)

    def test_short_or_wrong_budget_and_stale_final_step_fail(self):
        manifest, state, _, _, _ = self.completed_fixture()
        with self.assertRaises(ValueError):
            analysis.validate_completion(self.root, state, dict(manifest, seconds_per_arm=60))
        path = self.root / 'original' / 'train.log'
        path.write_text(path.read_text().replace('14400.2', '14399.9'))
        with self.assertRaisesRegex(ValueError, 'budget'):
            analysis.validate_completion(self.root, state, manifest)

    def test_inventory_rejects_duplicate_steps_and_foreign_paths(self):
        _, _, _, _, _ = self.completed_fixture()
        path = self.root / 'original' / 'checkpoints.json'
        records = json.loads(path.read_text())
        dump(path, records + [records[0]])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            analysis._inventory(self.root, 'original')
        records[1]['path'] = '/another/checkpoint'
        dump(path, records)
        with self.assertRaisesRegex(ValueError, 'path'):
            analysis._inventory(self.root, 'original')

    def test_live_identity_checks_commands_parent_and_process_start(self):
        state, seen = live_state(), {}
        supervisor, child = identity(10, self.root), identity(20, self.root, child=True)
        with mock.patch.object(analysis, '_process_identity', side_effect=[supervisor, child]):
            analysis._validate_live(self.root, state, seen)
        self.assertEqual(set(seen), {'10', '20'})
        for bad in (dict(child, argv=['wrong']), dict(child, parent_pid=999),
                     dict(child, started_unix=2000.), dict(child, start_ticks=999)):
            with mock.patch.object(analysis, '_process_identity', side_effect=[supervisor, bad]):
                with self.assertRaises(ValueError):
                    analysis._validate_live(self.root, state, seen)

    def test_missing_child_while_supervisor_hashes_remains_verified_live_wait(self):
        with mock.patch.object(analysis, '_process_identity', side_effect=[
                identity(10, self.root), FileNotFoundError('reaped child')]):
            seen = {}
            analysis._validate_live(self.root, live_state(), seen)
        self.assertEqual(set(seen), {'10'})

    def test_wait_transitions_to_complete_without_gpu(self):
        states = [live_state(), {'phase': 'training_complete'}]
        with mock.patch.object(analysis, '_json', side_effect=states), \
                mock.patch.object(analysis, '_process_identity', side_effect=[identity(10, self.root),
                     identity(20, self.root, child=True)]), \
                mock.patch.object(analysis.time, 'sleep') as sleep, \
                mock.patch.object(analysis.subprocess, 'Popen') as process:
            result, seen = analysis.wait_for_training(self.root, wait=True)
        self.assertEqual(result['phase'], 'training_complete')
        self.assertEqual(set(seen), {'10', '20'})
        sleep.assert_called_once_with(30)
        process.assert_not_called()

    def test_missing_supervisor_rechecked_before_failure_never_restarted(self):
        with mock.patch.object(analysis, '_json', return_value=live_state()), \
                mock.patch.object(analysis, '_process_identity', side_effect=FileNotFoundError) as read, \
                mock.patch.object(analysis.time, 'sleep') as sleep, \
                mock.patch.object(analysis.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(RuntimeError, 'after rechecking'):
                analysis.wait_for_training(self.root, wait=True)
        self.assertEqual(read.call_count, 4)
        self.assertEqual(sleep.call_count, 3)
        process.assert_not_called()

    def test_transient_missing_handle_is_repolled_without_restart(self):
        with mock.patch.object(analysis, '_json', side_effect=[live_state(), live_state(),
                {'phase': 'training_complete'}]), \
                mock.patch.object(analysis, '_process_identity', side_effect=[FileNotFoundError,
                     identity(10, self.root), identity(20, self.root, child=True)]), \
                mock.patch.object(analysis.time, 'sleep'), \
                mock.patch.object(analysis.subprocess, 'Popen') as process:
            result, _ = analysis.wait_for_training(self.root, wait=True)
        self.assertEqual(result['phase'], 'training_complete')
        process.assert_not_called()

    def test_failed_or_nonwaiting_incomplete_state_never_sleeps_or_spawns(self):
        with mock.patch.object(analysis.time, 'sleep') as sleep, \
                mock.patch.object(analysis.subprocess, 'Popen') as process:
            for state, wait in ((live_state(), False), ({'phase': 'failed', 'error': 'boom'}, True)):
                with mock.patch.object(analysis, '_json', return_value=state):
                    with self.assertRaises(RuntimeError):
                        analysis.wait_for_training(self.root, wait=wait)
        sleep.assert_not_called()
        process.assert_not_called()
        for value in (0, 61, float('nan')):
            with self.assertRaises(ValueError):
                analysis.wait_for_training(self.root, poll_seconds=value)

    def test_probe_child_terminated_and_reaped_on_interruption(self):
        child = mock.Mock(pid=30)
        child.wait.side_effect = [KeyboardInterrupt(), -15]
        with mock.patch.object(analysis.subprocess, 'Popen', return_value=child):
            with self.assertRaises(KeyboardInterrupt):
                analysis._run_probe(Path('/probe'), '/checkpoint', '/batch', self.root / 'scores',
                                     self.root / 'probe.log')
        child.terminate.assert_called_once()
        self.assertEqual(child.wait.call_count, 2)
        child.kill.assert_not_called()

    def test_probe_stubborn_child_killed_after_bounded_wait(self):
        child = mock.Mock(pid=30)
        child.wait.side_effect = [KeyboardInterrupt(), subprocess.TimeoutExpired('probe', 30), -9]
        with mock.patch.object(analysis.subprocess, 'Popen', return_value=child):
            with self.assertRaises(KeyboardInterrupt):
                analysis._run_probe(Path('/probe'), '/checkpoint', '/batch', self.root / 'scores',
                                     self.root / 'probe.log')
        child.kill.assert_called_once()
        self.assertEqual(child.wait.call_count, 3)

    def test_probe_failure_is_not_retried(self):
        child = mock.Mock(pid=30)
        child.wait.return_value = 2
        with mock.patch.object(analysis.subprocess, 'Popen', return_value=child) as spawn:
            with self.assertRaisesRegex(RuntimeError, 'exited 2'):
                analysis._run_probe(Path('/probe'), '/checkpoint', '/batch', self.root / 'scores',
                                     self.root / 'probe.log')
        spawn.assert_called_once()

    def test_child_exit_race_still_reaps_and_preserves_interruption(self):
        child = mock.Mock(pid=30)
        child.wait.side_effect = [KeyboardInterrupt(), -15]
        child.terminate.side_effect = ProcessLookupError()
        with mock.patch.object(analysis.subprocess, 'Popen', return_value=child):
            with self.assertRaises(KeyboardInterrupt):
                analysis._run_probe(Path('/probe'), '/checkpoint', '/batch', self.root / 'scores',
                                     self.root / 'probe.log')
        self.assertEqual(child.wait.call_count, 2)

    def test_broken_symlink_output_is_not_followed(self):
        output = self.root / 'existing_symlink'
        output.symlink_to(self.root / 'missing_destination')
        with self.assertRaises(FileExistsError):
            analysis.analyze(self.root, '/cases', '/probe', output)
        self.assertTrue(output.is_symlink())
        self.assertFalse((self.root / 'missing_destination').exists())

    def test_pipeline_no_gpu_before_complete_and_preserves_failure(self):
        _, state, _, cases, probe = self.completed_fixture()
        state['phase'] = 'original'
        dump(self.root / 'state.json', state)
        output = self.root / 'new_analysis'
        with mock.patch.object(analysis.paired_training, 'verify_frozen_inputs'), \
                mock.patch.object(analysis, '_run_probe') as run:
            with self.assertRaisesRegex(RuntimeError, 'not complete'):
                analysis.analyze(self.root, cases, probe, output)
        run.assert_not_called()
        self.assertTrue((output / 'request.json').exists())
        self.assertTrue((output / 'failure.json').exists())
        self.assertFalse((output / 'summary.json').exists())
        with self.assertRaises(FileExistsError):
            analysis.analyze(self.root, cases, probe, output)

    def test_full_pipeline_plans_three_unique_probes_and_provenance_reports(self):
        _, _, _, cases, probe = self.completed_fixture()
        output = self.root / 'new_analysis'
        scored = {}

        def fake_probe(binary, checkpoint_path, batch, score_dir, log):
            scored[str(score_dir)] = checkpoint_path
            return {'command': [str(binary)], 'returncode': 0}

        def fake_summary(case_path, score_dir, summary_path):
            result = {'probe_metadata': {'checkpoint_directory': scored[str(score_dir)]},
                      'groups': [{'mean_token_nll': 1.}]}
            dump(summary_path, result)
            return result

        with mock.patch.object(analysis.paired_training, 'verify_frozen_inputs'), \
                mock.patch.object(analysis, '_verify_checkpoint', side_effect=lambda item:
                     {'path': item['path'], 'weight_sha256': item['sha256']}) as verify, \
                mock.patch.object(analysis.paired_weight_diff, 'compare_checkpoints',
                     return_value={'model': {'delta_l2': 2.}}) as compare, \
                mock.patch.object(analysis, '_run_probe', side_effect=fake_probe) as run, \
                mock.patch.object(analysis.paired_word_cases, 'summarize', side_effect=fake_summary):
            result = analysis.analyze(self.root, cases, probe, output)
        self.assertTrue(result['complete'])
        self.assertEqual(run.call_count, 3)
        self.assertEqual(compare.call_count, 2)
        self.assertEqual(verify.call_count, 12)  # 3 initial copies +3 endpoints, before/after.
        self.assertEqual(set(result['weights']), {'final', 'matched_step'})
        self.assertEqual(len(result['behavior']), 3)
        self.assertTrue((output / 'summary.json').exists())
        self.assertFalse((output / 'failure.json').exists())
        for record in result['frozen_inputs']:
            self.assertEqual(record, analysis._record(record['path']))

    def test_corrupted_initial_checkpoint_blocks_every_probe(self):
        _, _, _, cases, probe = self.completed_fixture()
        with mock.patch.object(analysis.paired_training, 'verify_frozen_inputs'), \
                mock.patch.object(analysis, '_verify_checkpoint', side_effect=ValueError('nonfinite initial')), \
                mock.patch.object(analysis, '_run_probe') as run:
            with self.assertRaisesRegex(ValueError, 'nonfinite initial'):
                analysis.analyze(self.root, cases, probe, self.root / 'new_analysis')
        run.assert_not_called()

    def deterministic_fixture(self):
        manifest, state, inventories, cases, probe = self.completed_fixture()
        manifest['determinism'] = {'required': True}
        for arm in ('control_a', 'control_b'):
            directory = self.root / arm / 'checkpoints'
            directory.mkdir(parents=True)
            inventories[arm] = {}
            for step in range(3):
                path = directory / f'step_{step}'
                path.mkdir()
                inventories[arm][step] = {'step': step, 'path': str(path), 'sha256': {}}
        for arm, inventory in inventories.items():
            for step, item in inventory.items():
                path = Path(item['path']) / 'weight_0.bin'
                value = 'initial' if step == 0 else f'{"control" if arm.startswith("control_") else arm}:{step}'
                path.write_text(value)
                item['sha256'] = {'weight_0.bin': analysis.checkpoint.sha256_file(path)}
            dump(self.root / arm / 'checkpoints.json', list(inventory.values()))
        evidence = analysis.paired_training.compare_determinism_controls(
            list(inventories['control_a'].values()), list(inventories['control_b'].values()))
        state['determinism_gate'] = evidence
        dump(self.root / 'determinism_gate.json', evidence)
        dump(self.root / 'manifest.json', manifest)
        dump(self.root / 'state.json', state)
        case_data = json.loads(cases.read_text())
        case_data['manifest'] = analysis._record(self.root / 'manifest.json')
        dump(cases, case_data)
        return manifest, state, inventories, cases, probe

    def fake_checkpoint(self, directory, **kwargs):
        # Tiny checkpoint layout, but exercise the real file-hash verifier.
        return mock.Mock(directory=Path(directory),
                         manifest=[mock.Mock(filename='weight_0.bin')])

    def test_deterministic_gate_independently_hashes_every_control_weight(self):
        manifest, state, inventories, _, _ = self.deterministic_fixture()
        with mock.patch.object(analysis.checkpoint, 'GPT2Checkpoint', side_effect=self.fake_checkpoint):
            result = analysis.validate_determinism_gate(
                self.root, state, manifest, inventories['initial'][0])
            self.assertEqual(len(result['checkpoints']), 6)
            self.assertEqual(result['evidence'], state['determinism_gate'])
            path = Path(inventories['control_b'][1]['path']) / 'weight_0.bin'
            path.write_text('changed after training')
            with self.assertRaisesRegex(ValueError, 'differs from completed inventory'):
                analysis.validate_determinism_gate(
                    self.root, state, manifest, inventories['initial'][0])

    def test_missing_control_step_or_modified_gate_blocks_validation(self):
        manifest, state, inventories, _, _ = self.deterministic_fixture()
        with mock.patch.object(analysis.checkpoint, 'GPT2Checkpoint', side_effect=self.fake_checkpoint):
            evidence = state['determinism_gate']
            evidence['checkpoints']['control_b'][1]['sha256']['weight_0.bin'] = 'fabricated'
            dump(self.root / 'determinism_gate.json', evidence)
            with self.assertRaisesRegex(ValueError, 'gate evidence disagrees with current'):
                analysis.validate_determinism_gate(
                    self.root, state, manifest, inventories['initial'][0])
            dump(self.root / 'control_b' / 'checkpoints.json',
                 [item for step, item in inventories['control_b'].items() if step != 1])
            with self.assertRaisesRegex(ValueError, 'exactly control steps'):
                analysis.validate_determinism_gate(
                    self.root, state, manifest, inventories['initial'][0])

    def test_changed_control_prevents_all_gpu_scoring(self):
        _, _, inventories, cases, probe = self.deterministic_fixture()
        (Path(inventories['control_a'][2]['path']) / 'weight_0.bin').write_text('corrupt')
        with mock.patch.object(analysis.paired_training, 'verify_frozen_inputs'), \
                mock.patch.object(analysis.checkpoint, 'GPT2Checkpoint', side_effect=self.fake_checkpoint), \
                mock.patch.object(analysis, '_run_probe') as run:
            with self.assertRaisesRegex(ValueError, 'differs from completed inventory'):
                analysis.analyze(self.root, cases, probe, self.root / 'new_analysis',
                                 all_matched_steps=True)
        run.assert_not_called()

    def test_full_matched_pipeline_retains_gate_evidence_and_shared_initial_score(self):
        _, state, inventories, cases, probe = self.deterministic_fixture()
        output = self.root / 'new_analysis'
        scored = {}

        def fake_probe(binary, checkpoint_path, batch, score_dir, log):
            scored[str(score_dir)] = checkpoint_path
            return {'command': [str(binary)], 'returncode': 0}

        def fake_summary(case_path, score_dir, summary_path):
            result = {'probe_metadata': {'checkpoint_directory': scored[str(score_dir)]},
                      'groups': [{'mean_token_nll': 1.}]}
            dump(summary_path, result)
            return result

        with mock.patch.object(analysis.paired_training, 'verify_frozen_inputs'), \
                mock.patch.object(analysis.checkpoint, 'GPT2Checkpoint', side_effect=self.fake_checkpoint), \
                mock.patch.object(analysis.paired_weight_diff, 'compare_checkpoints',
                                  return_value={'model': {'delta_l2': 2.}}) as compare, \
                mock.patch.object(analysis, '_run_probe', side_effect=fake_probe) as run, \
                mock.patch.object(analysis.paired_word_cases, 'summarize', side_effect=fake_summary):
            result = analysis.analyze(self.root, cases, probe, output, all_matched_steps=True)
        self.assertEqual(run.call_count, 6)
        self.assertEqual(compare.call_count, 4)
        self.assertEqual(result['plan']['matched_steps'], [0, 100, 200])
        self.assertEqual(result['determinism_verification']['evidence'], state['determinism_gate'])
        original_zero = inventories['original'][0]['path']
        replacement_zero = inventories['replacement'][0]['path']
        self.assertEqual(result['behavior'][replacement_zero]['shared_score_from'], original_zero)
        self.assertEqual(result['behavior'][replacement_zero]['report'],
                         result['behavior'][original_zero]['report'])
        self.assertIn(str(self.root / 'determinism_gate.json'),
                      [record['path'] for record in result['terminal_records']])


if __name__ == '__main__':
    unittest.main()
