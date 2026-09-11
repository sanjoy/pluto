import math
import unittest

import numpy as np

from .token_weight_affinities import (actual_contribution, bf16_values,
                                      output_affinities, select_features)


class TokenWeightAffinitiesTest(unittest.TestCase):
    def test_scalar_affinity_formula_with_nonconstant_gamma(self):
        embedding = np.array([[1., 2., -1.], [0., -2., 3.], [-1., 1., 2.]])
        gamma = np.array([2., .5, -1.])
        row = np.array([3., -1., 2.])
        mean = math.fsum(row) / len(row)
        expected = [math.fsum(float(e[j]) * float(gamma[j]) * (float(row[j]) - mean)
                              for j in range(3)) for e in embedding]
        np.testing.assert_allclose(output_affinities(embedding, gamma, row), expected,
                                   rtol=0, atol=2e-15)
        # Centering gamma*row instead would be mathematically wrong.
        wrong = embedding @ (gamma * row - np.mean(gamma * row))
        self.assertFalse(np.allclose(expected, wrong))

    def test_per_target_projection_identity_and_negative_gate(self):
        random = np.random.default_rng(51)
        embedding = random.normal(size=(13, 16))
        gamma = random.normal(size=16)
        row = random.normal(size=16)
        scores = output_affinities(embedding, gamma, row)
        for gate in (-.07, 0., 1.3):
            for target, competitor in ((0, 1), (9, 4), (12, 8)):
                std = .83
                direction = (embedding[target] - embedding[competitor]) * gamma / std
                direction -= direction.mean()
                expected = gate * math.fsum(float(row[j]) * float(direction[j]) for j in range(16))
                self.assertAlmostEqual(actual_contribution(scores, target, competitor, gate, std),
                                       expected, places=13)

    def test_constant_decoder_direction_is_removed_by_centering(self):
        embedding = np.arange(15).reshape(3, 5)
        gamma = np.arange(5)
        np.testing.assert_array_equal(output_affinities(embedding, gamma, np.ones(5)), np.zeros(3))

    def test_bf16_rounding_ties(self):
        values = np.array([1., 1. + 2**-8, 1. + 3 * 2**-8, -1. - 2**-8], dtype=np.float32)
        np.testing.assert_array_equal(bf16_values(values), [1., 1., 1. + 2**-6, -1.])

    def test_invalid_geometry_and_nonfinite_rejected(self):
        for gamma, row in ((np.ones(2), np.ones(3)), (np.ones(3), np.ones(2)),
                           (np.array([1., math.nan, 1.]), np.ones(3))):
            with self.assertRaises(ValueError):
                output_affinities(np.ones((4, 3)), gamma, row)
        for args in ((np.ones(3), 0, 0, 1, 1), (np.ones(3), 0, 3, 1, 1),
                     (np.ones(3), 0, 1, math.nan, 1), (np.ones(3), 0, 1, 1, 0)):
            with self.assertRaises(ValueError):
                actual_contribution(*args)

    def test_signed_selection_ties_and_fixed_B4_extra_rule(self):
        documents = {}
        for step in (51, 52, 53, 54):
            values = {f'blocks.{block}': [0.] * 2048 for block in range(8)}
            values['blocks.0'][7] = 4
            values['blocks.1'][1] = 4
            values['blocks.2'][9] = -5
            values['blocks.4'][3] = 2
            values['blocks.4'][4] = 1
            documents[step] = {'ledger': {'neurons': values}}
        selected = select_features(documents)
        self.assertEqual(len(selected), 16)
        for step in (51, 52, 53, 54):
            entries = [entry for entry in selected if entry['generation_step'] == step]
            self.assertEqual([(entry['block'], entry['neuron']) for entry in entries[:3]],
                             [(0, 7), (1, 1), (2, 9)])
            self.assertEqual(len(entries), 5 if step in (53, 54) else 3)
        documents[51]['ledger']['neurons']['blocks.0'][0] = math.nan
        with self.assertRaises(ValueError):
            select_features(documents)


if __name__ == '__main__':
    unittest.main()
