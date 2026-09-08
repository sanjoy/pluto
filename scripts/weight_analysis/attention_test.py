"""Synthetic tests: directed weight associations are not model inference."""

import unittest

import numpy as np

from scripts.weight_analysis.attention import (
    _head_provenance, _parse_blocks, _top_indices, extract_head_edges,
)
from scripts.weight_analysis.checkpoint import GPT2Config, tensor_manifest


class StaticAttentionTest(unittest.TestCase):
    def test_known_directed_association(self):
        # A rank-one head maps token 0's x direction to token 1's y direction.
        # WV@WO is deliberately non-symmetric. Reversing multiplication, using
        # Q/K, or interpreting WO's head dimension as columns would fail here.
        embedding = np.eye(3, dtype=np.float32)
        value = np.array([[2], [0], [0]], dtype=np.float32)
        output = np.array([[0, 3, 0]], dtype=np.float32)
        edges = list(extract_head_edges(embedding, value, output,
                                       source_count=3, top_k=1, chunk_size=1))
        self.assertEqual(len(edges), 1)
        self.assertEqual((edges[0]['source'], edges[0]['target']), (0, 1))
        self.assertAlmostEqual(edges[0]['score'], 1.0)
        self.assertAlmostEqual(edges[0]['raw_dot_product'], 6.0)
        self.assertAlmostEqual(edges[0]['source_write_norm'], 6.0)

    def test_factorization_matches_dense_reference_and_chunks(self):
        rng = np.random.default_rng(91)
        embedding = rng.normal(size=(19, 5)).astype(np.float32)
        value = rng.normal(size=(5, 2)).astype(np.float32)
        output = rng.normal(size=(2, 5)).astype(np.float32)
        writes = embedding @ value @ output
        norms = np.linalg.norm(writes, axis=1)
        sources = np.argsort(-norms)[:4]
        raw = writes @ embedding.T
        for mode in ['raw', 'cosine']:
            scores = raw.copy()
            if mode == 'cosine':
                scores /= norms[:, None] * np.linalg.norm(embedding, axis=1)[None, :]
            expected = [(int(source), int(target)) for source in sources
                        for target in np.argsort(-scores[source])[:3]]
            for chunk_size in [1, 4, 19, 100]:
                with self.subTest(mode=mode, chunk_size=chunk_size):
                    edges = list(extract_head_edges(
                        embedding, value, output, source_count=4, top_k=3,
                        score_mode=mode, chunk_size=chunk_size))
                    self.assertEqual([(e['source'], e['target']) for e in edges], expected)
                    for edge in edges:
                        s, t = edge['source'], edge['target']
                        np.testing.assert_allclose(edge['score'], scores[s, t],
                                                   rtol=1e-5, atol=1e-6)
                        np.testing.assert_allclose(edge['raw_dot_product'], raw[s, t],
                                                   rtol=1e-5, atol=1e-6)

    def test_chunk_ties_are_resolved_by_token_id(self):
        embedding = np.ones((7, 2), dtype=np.float32)
        edges = list(extract_head_edges(embedding, np.eye(2), np.eye(2),
                                       source_count=2, top_k=3, chunk_size=2))
        self.assertEqual([(e['source'], e['target']) for e in edges],
                         [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])

    def test_zero_norms_and_self_exclusion(self):
        embedding = np.array([[1, 0], [0, 0], [0, 1]], dtype=np.float32)
        edges = list(extract_head_edges(embedding, np.eye(2), np.eye(2),
                                       source_count=8, top_k=8, exclude_self=True))
        self.assertEqual([(e['source'], e['target']) for e in edges], [(0, 2), (2, 0)])
        empty = list(extract_head_edges(embedding, np.zeros((2, 1)),
                                       np.ones((1, 2))))
        self.assertEqual(empty, [])

    def test_broken_pairing_destroys_planted_association(self):
        embedding = np.eye(3, dtype=np.float32)
        value = np.array([[1], [0], [0]], dtype=np.float32)
        output_0 = np.array([[0, 1, 0]], dtype=np.float32)
        output_1 = np.array([[0, 0, 1]], dtype=np.float32)
        intact = list(extract_head_edges(embedding, value, output_0, top_k=1))
        broken = list(extract_head_edges(embedding, value, output_1, top_k=1))
        self.assertEqual(intact[0]['target'], 1)
        self.assertEqual(broken[0]['target'], 2)

    def test_inverse_channel_reparameterization_preserves_edges(self):
        # A basis permutation inside a head cannot change its OV product.
        rng = np.random.default_rng(3)
        embedding, value, output = [rng.normal(size=shape) for shape in
                                   [(7, 4), (4, 2), (2, 4)]]
        intact = list(extract_head_edges(embedding, value, output, top_k=2))
        permuted = list(extract_head_edges(embedding, value[:, ::-1],
                                          output[::-1, :], top_k=2))
        self.assertEqual([(e['source'], e['target']) for e in intact],
                         [(e['source'], e['target']) for e in permuted])
        np.testing.assert_allclose([e['score'] for e in intact],
                                   [e['score'] for e in permuted], atol=1e-12)

    def test_validation_and_top_indices(self):
        self.assertEqual(_top_indices([2, 2, 1, np.nan], 2, [5, 1, 0, 8]).tolist(),
                         [1, 0])
        for options in [{'source_count': 0}, {'top_k': -1}, {'chunk_size': 0},
                        {'score_mode': 'probability'}]:
            with self.assertRaises(ValueError):
                list(extract_head_edges(np.eye(2), np.eye(2), np.eye(2), **options))
        with self.assertRaises(ValueError):
            list(extract_head_edges(np.eye(2), np.ones((3, 1)), np.ones((1, 2))))
        with self.assertRaises(ValueError):
            list(extract_head_edges(np.array([[np.nan, 0]]), np.eye(2), np.eye(2)))

    def test_blocks_and_exact_head_slices(self):
        self.assertEqual(_parse_blocks('all', 2), [0, 1])
        self.assertEqual(_parse_blocks('1,0', 2), [1, 0])
        for bad in ['2', '-1', '0,0', 'x', '']:
            with self.assertRaises(ValueError):
                _parse_blocks(bad, 2)
        config = GPT2Config(vocab_size=3, padded_vocab_size=4, context_length=2,
                            n_layers=1, d_model=4, n_heads=2, d_ff=8)
        class Fixture:
            pass
        fixture = Fixture()
        fixture.config = config
        fixture.specs = {spec.name: spec for spec in tensor_manifest(config)}
        provenance = _head_provenance(fixture, 0, 1, 0)
        self.assertEqual(provenance['tensors']['value_projection']['columns'], [10, 12])
        self.assertEqual(provenance['tensors']['output_projection']['rows'], [0, 2])
        self.assertEqual(provenance['tensors']['embedding']['rows'], [0, 3])
        self.assertEqual(provenance['tensors']['value_projection']['filename'], 'weight_4.bin')


if __name__ == '__main__':
    unittest.main()
