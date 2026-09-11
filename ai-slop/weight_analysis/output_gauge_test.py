"""Corpus-free tests of exact translation algebra and gauge-audit safeguards."""

from dataclasses import asdict
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from .checkpoint import GPT2Checkpoint, GPT2Config, sha256_file, tensor_manifest
from .closed_trigram import ClosedTrigramProbe, RECIPE_EPSILON
from .output_gauge import (_hash_array, audit_translation, destination_scores,
                            main, probe_from_metadata)
from .trigram import weight_provenance


class OutputGaugeTest(unittest.TestCase):
    def test_scalar_common_shift_proof(self):
        writes = np.array([[2.], [-3.], [0.]])
        targets = np.array([[1.], [4.], [-2.]])
        shift = np.array([5.])
        original = destination_scores(writes, targets)['raw']
        shifted = destination_scores(writes, targets + shift)['raw']
        np.testing.assert_array_equal(shifted - original,
                                       np.broadcast_to(writes @ shift[:, None], original.shape))
        report = audit_translation(writes, targets, shift)
        self.assertEqual(report['raw_dot']['top1_exact_id_agreement_fraction'], 1)
        self.assertEqual(report['zero_write_rows_excluded_from_rank_agreement'], 1)
        self.assertEqual(report['raw_translation_identity']['max_absolute_residual'], 0)

    def test_cosine_rank_reversal_with_nonzero_targets(self):
        # Original B is better aligned with +x; after the declared mean shift,
        # A is better aligned. Raw A>B stays A>B in both coordinate systems.
        writes, targets = np.array([[1., 0.]]), np.array([[2., 2.], [1., .1]])
        shifted = targets - targets.mean(axis=0)
        self.assertTrue(np.all(np.linalg.norm(targets, axis=1) > 0))
        self.assertTrue(np.all(np.linalg.norm(shifted, axis=1) > 0))
        report = audit_translation(writes, targets, -targets.mean(axis=0))
        self.assertEqual(report['raw_dot']['top1_exact_id_agreement_fraction'], 1)
        self.assertEqual(report['cosine']['top1_exact_id_agreement_fraction'], 0)

    def test_input_position_compensation_and_probability_identity(self):
        rng = np.random.default_rng(7)
        embedding, positions = rng.normal(size=(5, 3)), rng.normal(size=(2, 3))
        shift = -embedding.mean(axis=0)
        hidden = rng.normal(size=(4, 3))
        np.testing.assert_allclose(embedding[:, None] + positions,
                                   (embedding + shift)[:, None] + (positions - shift), atol=1e-15)
        original, moved = hidden @ embedding.T, hidden @ (embedding + shift).T
        def softmax(x):
            exp = np.exp(x - x.max(axis=1, keepdims=True))
            return exp / exp.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(softmax(original), softmax(moved), atol=1e-15)

    def test_random_raw_ranks_and_top4_sets_invariant(self):
        rng = np.random.default_rng(19)
        writes, targets = rng.normal(size=(17, 8)), rng.normal(size=(23, 8))
        before_writes, before_targets = writes.copy(), targets.copy()
        report = audit_translation(writes, targets, -targets.mean(axis=0))
        self.assertEqual(report['raw_dot']['top_k_exact_set_agreement_fraction'], 1)
        self.assertEqual(report['raw_dot']['top1_disagreements_beyond_roundoff'], 0)
        self.assertEqual(report['raw_translation_identity']['rows_exceeding_roundoff_bound'], 0)
        np.testing.assert_array_equal(writes, before_writes)
        np.testing.assert_array_equal(targets, before_targets)

    def test_roundoff_disagreement_is_not_called_a_true_tie(self):
        targets = np.array([[1., 1.], [np.nextafter(1., 2.), 2.]])
        report = audit_translation([[1., 0.]], targets, [2. ** 54, 0.], top_k=1)
        self.assertEqual(report['raw_dot']['top1_exact_id_agreement_fraction'], 0)
        self.assertEqual(report['raw_dot']['top1_disagreements_compatible_with_roundoff'], 1)
        self.assertEqual(report['raw_dot']['top1_disagreements_beyond_roundoff'], 0)

    def test_zero_directions_and_exact_tie_ids(self):
        scores = destination_scores([[1., 0.], [0., 0.]], [[0., 0.], [1., 0.]])
        self.assertEqual(scores['raw'][0, 0], 0)
        self.assertTrue(np.isneginf(scores['cosine'][0, 0]))
        self.assertTrue(np.isneginf(scores['cosine'][1]).all())
        report = audit_translation([[1., 0.]], [[0., 0.], [1., 0.]], [-1., 0.], [9, 2])
        self.assertEqual(report['original_zero_target_rows'], 1)
        self.assertEqual(report['shifted_zero_target_rows'], 1)
        self.assertEqual(report['cosine']['top_k_exact_set_agreement_fraction'], 0)
        empty = audit_translation([[1., 0.]], [[0., 0.]], [0., 0.])
        self.assertIsNone(empty['cosine']['top1_exact_id_agreement_fraction'])
        self.assertEqual(empty['cosine']['both_rankings_empty_nonzero_write_rows'], 1)
        ties = audit_translation([[1., 0.]], [[1., 0.], [1., 0.]], [0., 1.], [9, 2], 1)
        self.assertEqual(ties['raw_dot']['top1_exact_id_agreement_fraction'], 1)
        json.dumps(empty, allow_nan=False)

    def test_invalid_input(self):
        for writes, targets, shift in [([], [[1.]], [0.]), ([[1., 2.]], [[1.]], [0.]),
                                      ([[1.]], [[np.nan]], [0.]), ([[1.]], [[1.]], [np.inf]),
                                      ([[1.]], [[1.]], [0., 0.])]:
            with self.assertRaises(ValueError):
                audit_translation(writes, targets, shift)
        for ids in [[0, 0], [0., 1.], [True, False], [-1, 0]]:
            with self.assertRaises(ValueError):
                audit_translation([[1.]], [[1.], [2.]], [0.], ids)

    def fixture(self, root):
        config = GPT2Config(vocab_size=5, padded_vocab_size=8, context_length=3,
                             n_layers=1, d_model=4, n_heads=2, d_ff=8)
        directory = root / 'checkpoint'
        directory.mkdir()
        rng = np.random.default_rng(8)
        for spec in tensor_manifest(config):
            values = rng.normal(size=spec.shape).astype('<f4')
            if spec.name == 'token_embedding.weight':
                values[config.vocab_size:] = 1e6  # Padding must not affect shift.
            values.tofile(directory / spec.filename)
        cp = GPT2Checkpoint(directory, config, check_finite=True)
        ids, pairing = np.array([0, 2, 4]), [0, 1]
        probe = ClosedTrigramProbe(cp, ids, 'normalized', pairing)
        metadata = {
            'stage': 'corpus_blind_extraction',
            'protocol': 'fixed final-output-offset vocabulary with normalized first-order closed trigram paths',
            'checkpoint': {'config': asdict(config), 'weight_sha256': {
                spec.filename: sha256_file(directory / spec.filename) for spec in cp}},
            'parameters': {'input_geometry': 'normalized', 'broken_routing_control': False},
            'geometry_provenance': {'input_geometry': 'normalized', 'previous_position': 0,
                                    'current_position': 1, 'epsilon': RECIPE_EPSILON},
            'vocabulary_selection': {'ids_ascending': ids.tolist(),
                                      'ids_little_endian_uint32_sha256': _hash_array(ids, '<u4')},
            'routing_weight_provenance': weight_provenance(cp, 0, pairing),
            'numerical_geometry_hashes': {
                'X0_little_endian_float64_sha256': _hash_array(probe.inputs0),
                'X1_little_endian_float64_sha256': _hash_array(probe.inputs1)},
            'source_sha256': {name: sha256_file(Path(__file__).with_name(name))
                              for name in ('offset_paths.py', 'closed_trigram.py', 'trigram.py', 'checkpoint.py')},
        }
        return cp, metadata

    def test_metadata_rejects_changes_and_cli_uses_full_logical_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cp, metadata = self.fixture(root)
            for edit in [lambda m: m['parameters'].update(input_geometry='raw'),
                         lambda m: m['parameters'].update(broken_routing_control=True),
                         lambda m: m['source_sha256'].update({'trigram.py': 'bad'}),
                         lambda m: m['checkpoint']['weight_sha256'].update({'weight_0.bin': 'bad'}),
                         lambda m: m['vocabulary_selection'].update(ids_ascending=[0, 1, 4]),
                         lambda m: m['numerical_geometry_hashes'].update(X0_little_endian_float64_sha256='bad')]:
                changed = copy.deepcopy(metadata)
                edit(changed)
                with self.assertRaises(ValueError):
                    probe_from_metadata(cp, changed)
            path, output = root / 'metadata.json', root / 'audit.json'
            path.write_text(json.dumps(metadata))
            # Count only actual derivative calls: exactly one per current S ID.
            original = ClosedTrigramProbe.interaction_writes
            calls = []
            def tracked(probe, current):
                calls.append(current)
                return original(probe, current)
            with mock.patch.object(ClosedTrigramProbe, 'interaction_writes', tracked):
                main(['--checkpoint', str(cp.directory), '--candidates-metadata', str(path),
                      '--output', str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(calls, [0, 1, 2])
            self.assertEqual(report['stage'], 'weight_only_gauge_audit_not_extraction')
            expected = -np.mean(cp.token_embedding, axis=0, dtype=np.float64)
            self.assertEqual(report['shift']['float64_sha256'], _hash_array(expected))
            self.assertEqual(report['vocabulary_ids_ascending'], [0, 2, 4])
            self.assertFalse(report['frozen_metadata']['candidate_file_read'])
            self.assertEqual(report['raw_dot']['top1_disagreements_beyond_roundoff'], 0)

    def test_no_overwrite_before_metadata_access(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'audit.json'
            output.write_text('preserve me')
            args = ['--checkpoint', '/missing', '--candidates-metadata', '/missing',
                    '--output', str(output)]
            with self.assertRaises(FileExistsError):
                main(args)
            self.assertEqual(output.read_text(), 'preserve me')
            output.unlink()
            output.symlink_to(Path(directory) / 'missing')
            with self.assertRaises(FileExistsError):
                main(args)
            self.assertTrue(output.is_symlink())


if __name__ == '__main__':
    unittest.main()
