"""CPU-only tests for the paired experiment's alignment and run controls."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.weight_analysis.paired_training import (
    parse_training_result, split_boundary, training_command, verify_alignment,
    verify_frozen_inputs, write_json)


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


if __name__ == '__main__':
    unittest.main()
