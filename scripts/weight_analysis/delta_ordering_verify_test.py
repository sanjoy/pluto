import unittest

import numpy as np

from .delta_ordering_verify import seed_matches
from .verify import CorpusIndex


class SeedMatchesTest(unittest.TestCase):
    def test_counts_and_end_guard(self):
        index = CorpusIndex(np.array([1, 2, 3, 1, 2, 4, 2, 3, 1, 2]), 5)
        result = seed_matches(index, dict(token_ids=[1, 2, 3], seed_position=1, seed_id=2))
        self.assertEqual(result['predecessor_pair_occurrences'], 3)
        self.assertEqual(result['successor_pair_occurrences'], 2)
        self.assertEqual(result['seed_triple_occurrences'], 1)
        self.assertEqual(result['seed_triple_token_starts'], [0])

    def test_absent_and_invalid_seed(self):
        index = CorpusIndex(np.array([1, 2, 3]), 5)
        result = seed_matches(index, dict(token_ids=[4, 2, 1], seed_position=1, seed_id=2))
        self.assertEqual(result['seed_triple_occurrences'], 0)
        with self.assertRaises(ValueError):
            seed_matches(index, dict(token_ids=[1, 2, 3], seed_position=1, seed_id=4))


if __name__ == '__main__':
    unittest.main()
