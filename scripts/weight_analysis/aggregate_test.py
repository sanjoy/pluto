"""Synthetic, corpus-free tests of signed aggregate operator extraction."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from .aggregate import (aggregate_operator, extract_operator_edges, gate_vector,
                        main, neuron_contributions)
from .mlp import decode_paths


class AggregateTest(unittest.TestCase):
    def test_signed_cancellation_is_preserved(self):
        # The first two neurons individually support/opppose 0 -> 1 equally.
        # A positive-only or strongest-neuron probe would invent that edge.
        w1 = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.float64)
        w2 = np.array([[0, 9], [0, -9], [2, 0]], dtype=np.float64)
        gates = np.ones(3)
        operator = aggregate_operator(w1, w2, gates)
        np.testing.assert_array_equal(operator, [[2, 0], [0, 0]])
        terms = neuron_contributions([1, 0], [0, 1], w1, w2, gates)
        np.testing.assert_array_equal(terms, [9, -9, 0])
        self.assertEqual(float(terms.sum()), 0)
        edge = list(extract_operator_edges(np.eye(2), operator, top_k=1))[0]
        self.assertEqual((edge['source'], edge['target']), (0, 0))

    def test_planted_directed_sequence_without_corpus(self):
        # Static shift operator contains 0->1->2->3; no text is supplied.
        operator = np.diag(np.ones(3), 1)
        rows = list(extract_operator_edges(np.eye(4), operator, source_count=4, top_k=1))
        edges = {(row['source'], row['target']):
                 {'score': row['score'], 'provenance': {'pair': [row['source'], row['target']]}}
                 for row in rows}
        paths = list(decode_paths(edges, length=4, starts=1))
        self.assertEqual(paths[0]['token_ids'], [0, 1, 2, 3])

    def test_dense_factorized_and_terms_agree(self):
        rng = np.random.default_rng(55)
        embedding = rng.normal(size=(13, 5))
        w1, w2 = rng.normal(size=(5, 7)), rng.normal(size=(7, 5))
        gates = rng.normal(size=7)
        pairing = np.roll(np.arange(7), 1)
        for selected in [None, pairing]:
            operator = aggregate_operator(w1, w2, gates, selected)
            values = w2 if selected is None else w2[selected]
            dense = ((embedding @ w1) * gates) @ values @ embedding.T
            np.testing.assert_allclose(embedding @ operator @ embedding.T, dense, atol=1e-13)
            for source, target in [(0, 1), (4, 7), (12, 8)]:
                terms = neuron_contributions(embedding[source], embedding[target],
                                              w1, w2, gates, selected)
                self.assertAlmostEqual(terms.sum(), dense[source, target], places=11)

    def test_chunked_ranking_matches_dense_cosine(self):
        rng = np.random.default_rng(17)
        embedding, operator = rng.normal(size=(17, 4)), rng.normal(size=(4, 4))
        writes = embedding @ operator
        norms, target_norms = np.linalg.norm(writes, axis=1), np.linalg.norm(embedding, axis=1)
        raw = writes @ embedding.T
        scores = raw / (norms[:, None] * target_norms[None, :])
        sources = np.argsort(-norms)[:5]
        expected = [(int(s), int(t)) for s in sources for t in np.argsort(-scores[s])[:3]]
        for chunk_size in [1, 4, 17, 40]:
            rows = list(extract_operator_edges(embedding, operator, source_count=5,
                                               top_k=3, chunk_size=chunk_size))
            self.assertEqual([(r['source'], r['target']) for r in rows], expected)
            for row in rows:
                s, t = row['source'], row['target']
                self.assertAlmostEqual(row['score'], scores[s, t], places=12)
                self.assertAlmostEqual(row['raw_dot_product'], raw[s, t], places=12)

    def test_gate_derivative_has_correct_sign_and_matches_finite_difference(self):
        bias = np.array([-3., -1., 0., 1., 3.])
        def gelu(x):
            return 0.5*x*(1 + np.tanh(0.7978845608*(x + 0.044715*x**3)))
        expected = (gelu(bias + 1e-5) - gelu(bias - 1e-5)) / 2e-5
        np.testing.assert_allclose(gate_vector(bias, 'bias_gelu'), expected, atol=1e-9)
        self.assertLess(gate_vector(bias, 'bias_gelu')[1], 0)
        self.assertEqual(gate_vector(bias, 'bias_gelu')[2], 0.5)
        np.testing.assert_array_equal(gate_vector(bias), np.ones(5))

    def test_signed_scaling_and_paired_permutation_preserve_operator(self):
        rng = np.random.default_rng(2)
        w1, w2, gates = rng.normal(size=(3, 4)), rng.normal(size=(4, 3)), rng.normal(size=4)
        operator = aggregate_operator(w1, w2, gates)
        scales = np.array([2., -3., 0.5, -0.1])
        np.testing.assert_allclose(aggregate_operator(w1 * scales, w2 / scales[:, None], gates),
                                   operator, atol=1e-12)
        perm = np.array([2, 0, 3, 1])
        np.testing.assert_allclose(aggregate_operator(w1[:, perm], w2[perm], gates[perm]),
                                   operator, atol=1e-12)
        self.assertFalse(np.allclose(aggregate_operator(w1, w2, gates, perm), operator))

    def test_zeros_ties_and_invalid_inputs(self):
        self.assertEqual(list(extract_operator_edges(np.eye(2), np.zeros((2, 2)))), [])
        rows = list(extract_operator_edges(np.ones((4, 2)), np.eye(2),
                                           source_count=2, top_k=2, chunk_size=1))
        self.assertEqual([(r['source'], r['target']) for r in rows],
                         [(0, 0), (0, 1), (1, 0), (1, 1)])
        for pairing in [[0, 0], [0.0, 1.0], [0, 2]]:
            with self.assertRaises(ValueError):
                aggregate_operator(np.eye(2), np.eye(2), np.ones(2), pairing)
        with self.assertRaises(ValueError):
            aggregate_operator(np.eye(2), np.eye(3), np.ones(2))
        with self.assertRaises(ValueError):
            gate_vector(np.ones(2), 'contextual')
        with self.assertRaises(ValueError):
            list(extract_operator_edges(np.eye(2), np.eye(2), source_count=0))
        with self.assertRaises(ValueError):
            list(extract_operator_edges(np.eye(2), np.full((2, 2), np.nan)))

    def test_no_overwrite_before_checkpoint_access(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'candidates.jsonl'
            sidecar = output.with_suffix('.jsonl.metadata.json')
            for occupied in [output, sidecar]:
                occupied.write_text('preserve me')
                with self.assertRaises(FileExistsError):
                    main(['--checkpoint', '/does/not/exist', '--tokenizer-dir', '/also/missing',
                          '--output', str(output)])
                self.assertEqual(occupied.read_text(), 'preserve me')
                occupied.unlink()


if __name__ == '__main__':
    unittest.main()
