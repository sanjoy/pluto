"""Independent scalar and algebraic checks; no real corpus or GPU required."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .checkpoint import GPT2Checkpoint, GPT2Config, tensor_manifest
from .vocabulary import extract, ranking, weight_scores, write_report


class VocabularyTest(unittest.TestCase):
    def test_scalar_score_oracle(self):
        e = np.array([[1., 2., 3.], [-2., 1., 0.], [3., 1., -1.]])
        beta = np.array([2., -3., 1.])
        result = weight_scores(e, beta)
        mean = [sum(row[j] for row in e) / len(e) for j in range(3)]
        for token, row in enumerate(e):
            self.assertAlmostEqual(result['embedding_norm'][token], sum(x*x for x in row)**.5)
            self.assertAlmostEqual(result['centroid_distance'][token],
                                   sum((x-m)**2 for x, m in zip(row, mean))**.5)
            self.assertEqual(result['output_offset'][token], sum(x*b for x, b in zip(row, beta)))

    def test_output_offset_exact_decomposition_in_real_arithmetic(self):
        rng = np.random.default_rng(17)
        e, hidden, gamma, beta = [rng.normal(size=shape) for shape in [(7, 4), (9, 4), (4,), (4,)]]
        centered = hidden - hidden.mean(axis=1, keepdims=True)
        normalized = centered / np.sqrt(np.mean(centered**2, axis=1, keepdims=True) + 1e-5)
        full = (normalized * gamma + beta) @ e.T
        contextual = (normalized * gamma) @ e.T
        offset = weight_scores(e, beta)['output_offset']
        np.testing.assert_allclose(full, contextual + offset, rtol=1e-13, atol=1e-13)

    def test_simultaneous_orthogonal_rotation_preserves_scores(self):
        rng = np.random.default_rng(29)
        e, beta = rng.normal(size=(8, 4)), rng.normal(size=4)
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        for key, values in weight_scores(e, beta).items():
            np.testing.assert_allclose(values, weight_scores(e @ q, beta @ q)[key], atol=1e-14)

    def test_common_row_translation_preserves_centered_distances_and_offset_order(self):
        e = np.array([[1., 0.], [0., 2.], [-3., 2.]])
        a, b = weight_scores(e, [1., 2.]), weight_scores(e + [2., 3.], [1., 2.])
        np.testing.assert_allclose(a['centroid_distance'], b['centroid_distance'])
        np.testing.assert_allclose(b['output_offset'] - a['output_offset'], 8.)
        self.assertFalse(np.allclose(a['embedding_norm'], b['embedding_norm']))

    def test_tie_order_and_negative_scores(self):
        actual = ranking('example', [-2., 0., 0., -1.])
        self.assertEqual(actual['token_ids'], [1, 2, 3, 0])
        self.assertEqual(actual['scores'], [0., 0., -1., -2.])

    def test_invalid_inputs_and_overflow(self):
        for e, beta in [([], []), ([[1., 2.]], [1.]), ([[np.nan]], [0.]), ([[0.]], [np.inf]),
                        ([[1e308]], [1e308])]:
            with self.subTest(e=e), self.assertRaises(ValueError):
                weight_scores(e, beta)
        for scores in [[], [[1.]], [np.inf], [np.nan]]:
            with self.assertRaises(ValueError):
                ranking('bad', scores)

    def test_physical_padding_excluded_and_manifest_complete(self):
        config = GPT2Config(vocab_size=3, padded_vocab_size=4, context_length=2,
                            n_layers=1, d_model=2, n_heads=1, d_ff=4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for spec in tensor_manifest(config):
                values = np.zeros(spec.shape, dtype='<f4')
                if spec.name == 'token_embedding.weight':
                    values[:] = [[1., 0.], [0., 2.], [2., 0.], [1e10, 1e10]]
                if spec.name == 'final_norm.bias':
                    values[:] = [1., -1.]
                values.tofile(root / spec.filename)
            cp = GPT2Checkpoint(root, config)
            before = cp.provenance(hash_weights=True)['weight_sha256']
            report = extract(cp, cp)
            self.assertEqual(len(report['rankings']), 6)
            for item in report['rankings']:
                self.assertEqual(sorted(item['token_ids']), [0, 1, 2])
                expected = weight_scores([[1., 0.], [0., 2.], [2., 0.]], [1., -1.])
                np.testing.assert_allclose(item['scores'], expected[item['method'].split(':')[1]][item['token_ids']])
            self.assertEqual(report['weight_dependencies']['row_bytes'], 8)
            self.assertEqual(report['weight_dependencies']['output_offset_shared_beta_file'], 'weight_15.bin')
            self.assertEqual(before, cp.provenance(hash_weights=True)['weight_sha256'])

    def test_write_once_and_no_symlink_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'scores.json'
            write_report(path, {'answer': 42})
            self.assertEqual(json.loads(path.read_text()), {'answer': 42})
            with self.assertRaises(FileExistsError):
                write_report(path, {})
            link = Path(tmp) / 'link'
            link.symlink_to(Path(tmp) / 'absent')
            with self.assertRaises(FileExistsError):
                write_report(link, {})
            self.assertFalse((Path(tmp) / 'absent').exists())


if __name__ == '__main__':
    unittest.main()
