"""Synthetic follow-through extraction checks; no actual checkpoint reads."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import delta_followthrough as follow
from . import restart_delta as previous
from .checkpoint import GPT2Config


class ArithmeticTest(unittest.TestCase):
    def test_fixed_centers_and_no_overlap_with_original_experiment(self):
        self.assertEqual(follow.checkpoint_steps(580), [670, 680, 690, 700])
        current = {step for boundary in follow.BOUNDARIES for step in follow.checkpoint_steps(boundary)}
        old = {step for boundary in previous.BOUNDARIES for step in previous.checkpoint_steps(boundary)}
        self.assertEqual(len(current), 20)
        self.assertFalse(current & old)
        self.assertEqual(len(follow.expected_names()), 12)
        for invalid in (-10, True, 581, '580'):
            with self.assertRaises(ValueError):
                follow.checkpoint_steps(invalid)

    def test_center_positive_neighbors_negative_and_inputs_unchanged(self):
        for index, expected in ((0, -1), (1, 2), (2, -1)):
            values = np.ones((3, 9))
            values[index, 0] = np.exp(2)
            original = values.tobytes()
            result = follow.triplet_score(values)
            self.assertAlmostEqual(result[0], expected)
            np.testing.assert_array_equal(result[1:], 0)
            self.assertEqual(values.tobytes(), original)
        np.testing.assert_array_equal(follow.triplet_score(np.zeros((3, 9))), np.zeros(9))

    def test_shared_midrank_mean_and_exact_ties(self):
        values = np.array([[3., 1, 2, 2]] * 5)
        original = values.tobytes()
        np.testing.assert_array_equal(follow.shared_scores(values), [.875, .125, .5, .5])
        self.assertEqual(values.tobytes(), original)
        np.testing.assert_array_equal(follow.shared_scores(np.ones((5, 4))), [.5] * 4)

    def test_invalid_arithmetic_inputs(self):
        for values in (np.ones((2, 5)), np.ones(3), np.full((3, 5), np.nan),
                       np.full((3, 5), -1), np.ones((3, 0))):
            with self.assertRaises(ValueError):
                follow.triplet_score(values)
        for values in (np.ones((4, 5)), np.ones(5), np.full((5, 3), np.inf)):
            with self.assertRaises(ValueError):
                follow.shared_scores(values)


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='pluto-followthrough-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoints, self.output = self.root / 'checkpoints', self.root / 'output'
        self.checkpoints.mkdir()
        self.protocol, self.inventory = self.root / 'protocol.md', self.root / 'inventory.json'
        self.protocol.write_text('Synthetic protocol identity only; no real experiment.\n')
        self.inventory.write_text('{}\n')
        self.steps = [step for boundary in follow.BOUNDARIES for step in follow.checkpoint_steps(boundary)]
        for step in self.steps:
            (self.checkpoints / f'step_{step}.tar.gz').write_bytes(f'synthetic checkpoint {step}'.encode())
        self.config = GPT2Config(vocab_size=132, padded_vocab_size=136, d_model=4,
                                 n_heads=1, d_ff=8, n_layers=1, context_length=4)
        self.reads = []

    def synthetic_read(self, path, step):
        # Source/path planning may inspect metadata before this callback, but
        # the exclusive plan must precede any checkpoint payload access.
        self.assertTrue((self.output / 'plan.json').is_file())
        self.assertFalse((self.output / 'frozen.json').exists())
        self.reads.append(step)
        values = np.random.default_rng(step).normal(size=(136, 4)).astype('<f4')
        values[132:] = 1e12  # Deliberately large padding must never become tokens.
        values.setflags(write=False)
        return values, {'step': step, 'input_files': [follow.file_record(path)],
                        'input_files_unchanged_during_read': True}

    def run_pipeline(self, reader=None):
        # Only the model dimensions and actual archive payload reader are
        # replaced. All output files, hashes, numerical helpers, and freeze
        # ordering are real; archive-parser behavior has its separate tests.
        with mock.patch.object(follow, 'GPT2Config', return_value=self.config), \
                mock.patch.object(follow, 'read_embedding', side_effect=reader or self.synthetic_read), \
                contextlib.redirect_stdout(io.StringIO()):
            return follow.run(self.checkpoints, self.output, self.protocol, self.inventory)

    def test_complete_pipeline_shapes_addresses_scores_shuffles_and_freeze(self):
        result = self.run_pipeline()
        self.assertEqual(self.reads, self.steps)
        self.assertEqual(len(result['checkpoint_provenance']), 20)
        self.assertEqual(len(result['intervals']), 15)
        self.assertEqual(len(result['files']), 18)
        self.assertEqual(json.loads((self.output / 'frozen.json').read_text()), result)
        candidates = json.loads((self.output / 'candidates.json').read_text())['candidate_lists']
        self.assertEqual(set(candidates), follow.expected_names())
        permutation = np.random.Generator(np.random.PCG64(follow.SEED)).permutation(132)
        with np.load(self.output / 'scores.npz', allow_pickle=False) as arrays:
            self.assertEqual(set(arrays.files), follow.expected_names() | {'identity_permutation'})
            np.testing.assert_array_equal(arrays['identity_permutation'], permutation)
            for name, candidate in candidates.items():
                values = arrays[name]
                self.assertEqual(values.shape, (132,))
                order = np.lexsort((np.arange(132), -values))[:128]
                self.assertEqual(candidate['token_ids'], order.tolist())
                self.assertEqual(candidate['scores'], values[order].tolist())
                self.assertEqual(candidate['identity_shuffled_token_ids'], permutation[order].tolist())
                if name.startswith('shared_'):
                    variant = name.removeprefix('shared_followthrough_')
                    components = np.asarray([arrays[f'followthrough_{variant}_{boundary}'][order]
                                             for boundary in follow.BOUNDARIES])
                    self.assertEqual(candidate['component_scores'], components.tolist())
                    self.assertEqual(candidate['positive_component_count'], (components > 0).sum(axis=0).tolist())
        for boundary in follow.BOUNDARIES:
            steps = follow.checkpoint_steps(boundary)
            intervals = []
            for before, after in zip(steps[:-1], steps[1:]):
                with np.load(self.output / f'interval_{before}_{after}.npz', allow_pickle=False) as values:
                    self.assertEqual(set(values.files), {'raw', 'adjusted'})
                    self.assertEqual(values['raw'].shape, (132,))
                    self.assertEqual(values['adjusted'].shape, (132,))
                    intervals.append({kind: values[kind] for kind in follow.VARIANTS})
            with np.load(self.output / 'scores.npz', allow_pickle=False) as arrays:
                for kind in follow.VARIANTS:
                    expected = follow.triplet_score([values[kind] for values in intervals])
                    np.testing.assert_array_equal(arrays[f'followthrough_{kind}_{boundary}'], expected)
        for record in [*result['sources'].values(), *result['files'].values()]:
            self.assertEqual(follow.file_record(record['path']), record)
        with self.assertRaises(FileExistsError):
            self.run_pipeline()

    def test_missing_or_ambiguous_input_fails_before_payload_read(self):
        (self.checkpoints / f'step_{self.steps[0]}').mkdir()
        with self.assertRaisesRegex(ValueError, 'missing or ambiguous'):
            self.run_pipeline()
        self.assertEqual(self.reads, [])
        self.assertFalse(self.output.exists())

    def test_output_inside_checkpoint_rejected_before_payload_read(self):
        self.output = self.checkpoints / 'analysis'
        with self.assertRaisesRegex(ValueError, 'outside checkpoint root'):
            self.run_pipeline()
        self.assertEqual(self.reads, [])

    def test_reader_failure_leaves_plan_but_no_completion(self):
        def failed(path, step):
            self.synthetic_read(path, step)
            raise ValueError('synthetic corrupt archive')
        with self.assertRaisesRegex(ValueError, 'synthetic corrupt archive'):
            self.run_pipeline(failed)
        self.assertTrue((self.output / 'plan.json').is_file())
        self.assertFalse((self.output / 'frozen.json').exists())

    def test_changed_checkpoint_final_hash_prevents_freeze(self):
        def changed(path, step):
            value = self.synthetic_read(path, step)
            if step == self.steps[-1]:
                first = self.checkpoints / f'step_{self.steps[0]}.tar.gz'
                first.write_bytes(b'changed synthetic checkpoint')
            return value
        with self.assertRaisesRegex(ValueError, 'checkpoint input changed'):
            self.run_pipeline(changed)
        self.assertFalse((self.output / 'frozen.json').exists())

    def test_changed_protocol_or_prior_activity_prevents_freeze(self):
        for target in ('protocol', 'activity'):
            with self.subTest(target=target):
                self.output = self.root / ('output_' + target)
                def changed(path, step):
                    value = self.synthetic_read(path, step)
                    if step == self.steps[-1]:
                        destination = (self.protocol if target == 'protocol' else
                                       self.output / f'interval_{self.steps[0]}_{self.steps[1]}.npz')
                        destination.write_bytes(b'changed synthetic evidence')
                    return value
                with self.assertRaisesRegex(ValueError, 'source or output evidence changed'):
                    self.run_pipeline(changed)
                self.assertFalse((self.output / 'frozen.json').exists())

    def test_unverified_reader_record_or_wrong_layout_prevents_freeze(self):
        for variant in ('unverified', 'wrong_shape'):
            with self.subTest(variant=variant):
                self.output = self.root / ('output_' + variant)
                def broken(path, step):
                    values, record = self.synthetic_read(path, step)
                    if variant == 'unverified':
                        record['input_files_unchanged_during_read'] = False
                    else:
                        values = values[:-1]
                    return values, record
                with self.assertRaises(ValueError):
                    self.run_pipeline(broken)
                self.assertFalse((self.output / 'frozen.json').exists())


if __name__ == '__main__':
    unittest.main()
