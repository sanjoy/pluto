"""CPU-only tests for the paired experiment's alignment and run controls."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.weight_analysis.paired_training import (
    compare_determinism_controls, freeze_sources, main, parse_training_result, run,
    sha256_file, split_boundary, training_command, verify_alignment,
    verify_frozen_inputs, write_json)


MODULE = 'scripts.weight_analysis.paired_training'


class PairedTrainingTest(unittest.TestCase):
    def test_line_aligned_split_is_unchanged(self):
        original = b'first line\n Exeunt\nlast line\n'
        changed = original.replace(b'Exeunt', b'Nuveth')
        self.assertEqual(split_boundary(original, .4), split_boundary(changed, .4))
        self.assertEqual(original[split_boundary(original, .4) - 1], ord('\n'))

    def test_split_without_newline_preserves_utf8_boundary(self):
        content = 'aa\u20acb'.encode()
        self.assertEqual(split_boundary(content, .5), 5)

    def test_invalid_split(self):
        for fraction in [0, 1, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                split_boundary(b'a\nb\n', fraction)

    def arrays(self):
        # A space-prefixed occurrence, with an unrelated preceding/following
        # token. Replacement subword boundaries need not equal the original.
        return (b'x Exeunt!', b'x Nuveth!', np.array([9, 1475, 68, 2797, 8]),
                np.array([9, 21733, 303, 400, 8]),
                np.array([0, 1, 4, 5, 8, 9]), np.array([0, 1, 4, 6, 8, 9]))

    def test_three_slots_are_replaced_without_shifting_neighbors(self):
        result = verify_alignment(*self.arrays())
        self.assertEqual(result['replacements'], 1)
        self.assertEqual(result['changed_token_ids'], 3)
        self.assertTrue(result['outside_replacement_tokens_identical'])

    def test_unchanged_split_with_no_occurrences(self):
        result = verify_alignment(b'abc', b'abc', np.array([1]), np.array([1]),
                                  np.array([0, 3]), np.array([0, 3]))
        self.assertEqual(result['replacements'], 0)

    def test_rejects_unrelated_token_change(self):
        args = list(self.arrays())
        args[3][0] = 77
        with self.assertRaisesRegex(ValueError, 'outside'):
            verify_alignment(*args)

    def test_rejects_unrelated_byte_change(self):
        args = list(self.arrays())
        args[1] = b'y Nuveth!'
        with self.assertRaisesRegex(ValueError, 'unrelated'):
            verify_alignment(*args)

    def test_rejects_token_count_change(self):
        args = list(self.arrays())
        args[3] = args[3][:-1]
        with self.assertRaisesRegex(ValueError, 'counts'):
            verify_alignment(*args)

    def test_rejects_outer_boundary_change(self):
        args = list(self.arrays())
        args[5][1] = 2
        with self.assertRaisesRegex(ValueError, 'boundary'):
            verify_alignment(*args)

    def test_commands_share_settings_and_initial_checkpoint(self):
        manifest = {'root': '/experiment', 'binaries': {'trainer': {'path': '/frozen/trainer'}},
                    'flags': {'seed': 17, 'batch_size': 10, 'checkpoint_initial': True},
                    'inputs': {'original.full': {'text': '/original.txt'},
                               'replacement.full': {'text': '/replacement.txt'}}}
        a = training_command(manifest, 'original', 'original', seconds=14400)
        b = training_command(manifest, 'replacement', 'replacement', seconds=14400)
        varying = ('--corpus=', '--checkpoint_dir=', '--log_file=')
        self.assertEqual([v for v in a if not v.startswith(varying)],
                         [v for v in b if not v.startswith(varying)])
        self.assertIn('--resume_from=/experiment/initial/checkpoints', a)
        self.assertIn('--training_seconds=14400', a)
        self.assertFalse(any(v.startswith('--steps=') for v in a))

    def test_deterministic_controls_save_every_step_only_when_requested(self):
        manifest = {'root': '/experiment', 'seconds_per_arm': 14400,
                    'binaries': {'trainer': {'path': '/frozen/trainer'}},
                    'flags': {'checkpoint_every': 100},
                    'inputs': {'original.full': {'text': '/original.txt'}}}
        legacy = training_command(manifest, 'control_a', 'original', steps=2)
        self.assertIn('--checkpoint_every=100', legacy)
        manifest['determinism'] = {'required': True}
        control = training_command(manifest, 'control_a', 'original', steps=2)
        self.assertIn('--checkpoint_every=1', control)
        timed = training_command(manifest, 'original', 'original', seconds=14400)
        self.assertIn('--checkpoint_every=100', timed)

    def test_require_determinism_cli_is_optional(self):
        arguments = ['prepare']
        for name in ('output', 'corpus', 'tokenizer-dir', 'trainer',
                     'tokenizer-binary', 'sampler-binary'):
            arguments.extend(['--' + name, '/unused'])
        with mock.patch(MODULE + '.prepare') as prepare:
            main(arguments)
            self.assertFalse(prepare.call_args.args[0].require_determinism)
            main(arguments + ['--require-determinism'])
            self.assertTrue(prepare.call_args.args[0].require_determinism)

    def control_inventory(self, arm='control_a'):
        return [{'step': step, 'path': f'/{arm}/step_{step}',
                 'sha256': {'weight_0.bin': f'a{step}', 'weight_1.bin': f'b{step}'}}
                for step in range(3)]

    def test_exact_control_evidence_includes_every_step_and_weight(self):
        a = self.control_inventory()
        b = self.control_inventory('control_b')
        evidence = compare_determinism_controls(a, b)
        self.assertEqual(evidence['status'], 'verified')
        self.assertEqual(evidence['steps'], [0, 1, 2])
        self.assertEqual(evidence['checkpoints']['control_a'], a)
        self.assertEqual(evidence['checkpoints']['control_b'], b)

    def test_intermediate_weight_mismatch_fails_even_if_endpoint_matches(self):
        a = self.control_inventory()
        b = copy.deepcopy(a)
        b[1]['sha256']['weight_1.bin'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'mismatch at step 1.*weight_1.bin'):
            compare_determinism_controls(a, b)

    def test_missing_control_checkpoint_or_weight_cannot_pass(self):
        a = self.control_inventory()
        for missing_step in range(3):
            with self.subTest(step=missing_step):
                b = [item for item in a if item['step'] != missing_step]
                with self.assertRaisesRegex(ValueError, 'exactly control steps'):
                    compare_determinism_controls(a, b)
        b = copy.deepcopy(a)
        del b[1]['sha256']['weight_1.bin']
        with self.assertRaisesRegex(ValueError, 'mismatch at step 1'):
            compare_determinism_controls(a, b)
        with self.assertRaisesRegex(ValueError, 'exactly control steps'):
            compare_determinism_controls(a, a + [a[1]])
        for item in a:
            item['sha256'] = {}
        with self.assertRaisesRegex(ValueError, 'no weight hashes'):
            compare_determinism_controls(a, a)

    def test_failed_control_gate_prevents_long_arms_and_cannot_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {'root': str(root), 'seconds_per_arm': 14400,
                        'binaries': {'trainer': {'path': '/frozen/trainer'}},
                        'flags': {}, 'determinism': {'required': True},
                        'inputs': {'original.full': {'text': '/original.txt'},
                                   'replacement.full': {'text': '/replacement.txt'}}}
            write_json(root / 'manifest.json', manifest)
            write_json(root / 'state.json', {'phase': 'prepared'})

            def inventory(path):
                name = path.parent.name
                value = self.control_inventory(name)
                if name == 'initial':
                    value = value[:1]
                if name == 'control_b':
                    value[1]['sha256']['weight_0.bin'] = 'changed'
                return value

            def launch(command, **kwargs):
                log_path = next(arg.split('=', 1)[1] for arg in command
                                if arg.startswith('--log_file='))
                Path(log_path).write_text('mock training result')
                return mock.Mock(pid=123, wait=mock.Mock(return_value=0))

            def terminal(text, *, seconds, expected_steps):
                return {'final_step': expected_steps, 'stop_reason': 'step_limit',
                        'training_elapsed_seconds': 1.0,
                        'training_loss': 1.0, 'test_loss': 1.0}

            with mock.patch(MODULE + '.verify_frozen_inputs'), \
                    mock.patch(MODULE + '.checkpoint_inventory', side_effect=inventory), \
                    mock.patch(MODULE + '.parse_training_result', side_effect=terminal), \
                    mock.patch(MODULE + '.subprocess.Popen', side_effect=launch) as process:
                with self.assertRaisesRegex(ValueError, 'mismatch at step 1'):
                    run(root)
                self.assertEqual(process.call_count, 3)
                self.assertFalse((root / 'original').exists())
                self.assertFalse((root / 'replacement').exists())
                self.assertFalse((root / 'determinism_gate.json').exists())
                state = json.loads((root / 'state.json').read_text())
                self.assertEqual(state['phase'], 'failed')
                self.assertIn('mismatch at step 1', state['error'])
                with self.assertRaisesRegex(ValueError, 'already started'):
                    run(root)
                self.assertEqual(process.call_count, 3)

    def test_verified_gate_is_published_before_long_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {'root': str(root), 'seconds_per_arm': 14400,
                        'binaries': {'trainer': {'path': '/frozen/trainer'}},
                        'flags': {}, 'determinism': {'required': True},
                        'inputs': {'original.full': {'text': '/original.txt'},
                                   'replacement.full': {'text': '/replacement.txt'}}}
            write_json(root / 'manifest.json', manifest)
            write_json(root / 'state.json', {'phase': 'prepared'})

            def inventory(path):
                value = self.control_inventory(path.parent.name)
                return value[:1] if path.parent.name == 'initial' else value

            def launch(command, **kwargs):
                log_path = Path(next(arg.split('=', 1)[1] for arg in command
                                     if arg.startswith('--log_file=')))
                if log_path.parent.name in ('original', 'replacement'):
                    gate = json.loads((root / 'determinism_gate.json').read_text())
                    self.assertEqual(gate['status'], 'verified')
                    self.assertEqual(gate['steps'], [0, 1, 2])
                log_path.write_text('mock training result')
                return mock.Mock(pid=123, wait=mock.Mock(return_value=0))

            def terminal(text, *, seconds, expected_steps):
                return {'final_step': 2 if expected_steps is None else expected_steps,
                        'stop_reason': 'time_limit' if seconds else 'step_limit',
                        'training_elapsed_seconds': seconds or 1.0,
                        'training_loss': 1.0, 'test_loss': 1.0}

            with mock.patch(MODULE + '.verify_frozen_inputs'), \
                    mock.patch(MODULE + '.checkpoint_inventory', side_effect=inventory), \
                    mock.patch(MODULE + '.parse_training_result', side_effect=terminal), \
                    mock.patch(MODULE + '.subprocess.Popen', side_effect=launch) as process:
                run(root)
                self.assertEqual(process.call_count, 6)
                state = json.loads((root / 'state.json').read_text())
                self.assertEqual(state['phase'], 'training_complete')
                self.assertEqual(state['determinism_gate']['status'], 'verified')
                self.assertEqual(state['determinism_gate'], json.loads(
                    (root / 'determinism_gate.json').read_text()))
                with self.assertRaisesRegex(ValueError, 'already started'):
                    run(root)
                self.assertEqual(process.call_count, 6)

    def test_exclusive_provenance_and_atomic_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            write_json(path, {'a': 1}, exclusive=True)
            with self.assertRaises(FileExistsError):
                write_json(path, {}, exclusive=True)
            write_json(path, {'a': 2})
            self.assertIn('2', path.read_text())
            self.assertEqual([p.name for p in path.parent.iterdir()], ['state.json'])

    def log(self, elapsed='14400.2', reason='time_limit'):
        return (f'[2026-09-10 12:34:56 UTC] training stopped at step: 123\n'
                f'training stop reason: {reason}\ntraining elapsed seconds: {elapsed}\n'
                'final training loss: 1.2\nfinal test loss: 2.4\n')

    def test_timestamped_terminal_budget(self):
        result = parse_training_result(self.log(), seconds=14400)
        self.assertEqual(result['final_step'], 123)
        self.assertEqual(result['test_loss'], 2.4)

    def test_incomplete_or_nonfinite_budget_rejected(self):
        for log in [self.log(elapsed='14399'), self.log(reason='step_limit'),
                    self.log(elapsed='nan'), '']:
            with self.assertRaises(ValueError):
                parse_training_result(log, seconds=14400)

    def test_short_run_requires_requested_steps(self):
        parse_training_result(self.log(reason='step_limit'), expected_steps=123)
        with self.assertRaises(ValueError):
            parse_training_result(self.log(), expected_steps=124)

    def test_changed_tokenizer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tokenizer.json'
            path.write_text('changed')
            manifest = {'binaries': {}, 'inputs': {},
                        'tokenizer_files': {'json': {'path': str(path), 'sha256': 'wrong'}},
                        'sampling': {'prefix_path': str(path), 'prefix_sha256': 'wrong'}}
            with self.assertRaisesRegex(ValueError, 'input changed'):
                verify_frozen_inputs(manifest)

    def test_sources_are_frozen_and_later_edits_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'launcher.py'
            source.write_text('print("original")\n')
            digest = sha256_file(source)
            with mock.patch(MODULE + '.SOURCE_PATHS', {'launcher.py': source}), \
                    mock.patch(MODULE + '.IMPORTED_SOURCE_HASHES', {'launcher.py': digest}):
                records = freeze_sources(root)
                frozen = Path(records['launcher.py']['path'])
                self.assertEqual(frozen.read_bytes(), source.read_bytes())
                self.assertEqual(frozen.stat().st_mode & 0o777, 0o444)
                manifest = {'binaries': {}, 'inputs': {}, 'tokenizer_files': {},
                            'source_files': records,
                            'sampling': {'prefix_path': str(source), 'prefix_sha256': digest}}
                verify_frozen_inputs(manifest)
                with mock.patch(MODULE + '.IMPORTED_SOURCE_HASHES',
                                {'launcher.py': 'different loaded source'}):
                    with self.assertRaisesRegex(ValueError, 'loaded experiment source differs'):
                        verify_frozen_inputs(manifest)
                frozen.chmod(0o644)
                frozen.write_text('modified frozen source')
                with self.assertRaisesRegex(ValueError, 'frozen experiment input changed'):
                    verify_frozen_inputs(manifest)
                source.write_text('modified since import')
                with self.assertRaisesRegex(ValueError, 'source changed since import'):
                    freeze_sources(root)


if __name__ == '__main__':
    unittest.main()
