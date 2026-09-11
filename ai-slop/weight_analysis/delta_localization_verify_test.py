"""Independent set-of-position checks for retrieval/replay overlap metrics."""

import unittest

import numpy as np

from .delta_localization_verify import merge_intervals, overlap_metrics, replay_union_arrays


class IntervalMetricsTest(unittest.TestCase):
    def test_union_touching_nested_and_disjoint(self):
        self.assertEqual(merge_intervals([(8, 10), (1, 5), (2, 3), (5, 7)]),
                         [[1, 7], [8, 10]])

    def test_half_open_overlap(self):
        starts = np.array([[0, 2, 6], [3, 4, 8]])
        ends = replay_union_arrays(starts, 3, 3)
        result = overlap_metrics([(0, 3), (3, 5)], *ends, 12)
        self.assertEqual(result['intersection_tokens'], [5, 2])
        self.assertEqual(result['target_union_tokens'], [8, 7])
        self.assertEqual(result['precision'], [1.0, .4])

    def test_random_against_sets(self):
        rng = np.random.default_rng(12)
        for _ in range(20):
            starts = rng.integers(0, 40, size=(7, 8))
            retrieved = [(int(s), int(s) + 5) for s in rng.integers(0, 40, 6)]
            output = overlap_metrics(retrieved, *replay_union_arrays(starts, 8, 5), 45)
            found = {p for s, e in retrieved for p in range(s, e)}
            for i, row in enumerate(starts):
                target = {p for s in row for p in range(s, s + 5)}
                self.assertEqual(output['intersection_tokens'][i], len(found & target))
                self.assertEqual(output['target_union_tokens'][i], len(target))
                self.assertEqual(output['recall'][i], len(found & target) / len(target))

    def test_empty_retrieval_precision_undefined(self):
        result = overlap_metrics([], np.array([[0]]), np.array([[3]]), 5)
        self.assertEqual(result['precision'], [None])
        self.assertEqual(result['recall'], [0.0])

    def test_reject_invalid_union_geometry(self):
        with self.assertRaises(ValueError):
            merge_intervals([(2, 2)])
        with self.assertRaises(ValueError):
            overlap_metrics([(0, 6)], np.array([[0]]), np.array([[3]]), 5)
        with self.assertRaises(ValueError):
            overlap_metrics([], np.array([[0, 1]]), np.array([[3, 4]]), 5)


if __name__ == '__main__':
    unittest.main()
