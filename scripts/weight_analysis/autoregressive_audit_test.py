import math
import unittest

import numpy as np

from .autoregressive_audit import (Mt19937, canonical_uniform, scalar_distribution,
                                   scalar_sample, sequential_sum, validate_context)


class AutoregressiveAuditTest(unittest.TestCase):
    def test_reference_mt_words(self):
        random = Mt19937(5489)
        self.assertEqual([random.draw() for _ in range(5)],
                         [3499211612, 581869302, 3890346734, 3586334585, 545404204])

    def test_mt_multiple_twists_against_independent_numpy_legacy_engine(self):
        for seed in (0, 18, 5489, 2**32 - 1):
            random = Mt19937(seed)
            expected = np.random.RandomState(seed).randint(0, 2**32, 1500, dtype=np.uint32)
            self.assertEqual([random.draw() for _ in range(1500)], expected.tolist())

    def test_uniform_word_order_and_rounded_one(self):
        self.assertEqual(canonical_uniform(0, 0), 0)
        self.assertEqual(canonical_uniform(0, 2**31), .5)
        self.assertEqual(canonical_uniform(1, 0), 2**-64)
        self.assertEqual(canonical_uniform(2**32 - 1, 2**32 - 1), math.nextafter(1, 0))
        for words in ((-1, 0), (0, 2**32), (True, 0)):
            with self.assertRaises(ValueError):
                canonical_uniform(*words)

    def test_sequential_not_compensated_sum(self):
        values = [1., 2**-53, 2**-53]
        self.assertEqual(sequential_sum(values), 1.)
        self.assertEqual(math.fsum(values), 1. + 2**-52)

    def test_float_subtraction_happens_before_double_division(self):
        row = np.array([2**25, 1], dtype=np.float32)
        probabilities, _, total = scalar_distribution(row, 1e7)
        expected = math.exp(float(np.float32(1 - 2**25)) / 1e7)
        wrong = math.exp((1. - 2**25) / 1e7)
        self.assertNotEqual(expected, wrong)
        self.assertEqual(total, 1. + expected)
        self.assertEqual(probabilities[1], expected / total)

    def test_lower_bound_boundary_convention(self):
        row = np.array([0, 0], dtype=np.float32)
        self.assertEqual(scalar_sample(row, .8, .5)['token_id'], 0)
        self.assertEqual(scalar_sample(row, .8, math.nextafter(.5, 1))['token_id'], 1)
        zero = np.array([-1000, 0, -1000, 0], dtype=np.float32)
        self.assertEqual(scalar_sample(zero, .8, 0)['token_id'], 0)
        self.assertEqual(scalar_sample(zero, .8, .5)['token_id'], 1)

    def test_invalid_rows_temperatures_and_uniforms(self):
        for row in (np.array([], dtype=np.float32), np.array([0], dtype=np.float32),
                    np.array([0, 1], dtype=np.float64), np.array([[0, 1]], dtype=np.float32),
                    np.array([0, np.nan], dtype=np.float32), np.array([0, np.inf], dtype=np.float32)):
            with self.assertRaises(ValueError):
                scalar_distribution(row, .8)
        row = np.array([0, 1], dtype=np.float32)
        for temperature in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                scalar_distribution(row, temperature)
        for uniform in (-1, 1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                scalar_sample(row, .8, uniform)

    def test_context_crops_exact_ids_and_rejects_off_by_one(self):
        ids = list(range(20))
        event = {'index': 8, 'step': 8, 'absolute_token_index': 11,
                 'context_start': 7, 'context_length': 4, 'output_row': 3, 'token_id': 11}
        self.assertEqual(validate_context(event, 3, 8, ids, 4), [7, 8, 9, 10])
        for key in event:
            changed = dict(event)
            changed[key] += 1
            with self.assertRaises(ValueError):
                validate_context(changed, 3, 8, ids, 4)


if __name__ == '__main__':
    unittest.main()
