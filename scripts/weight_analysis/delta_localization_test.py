"""Synthetic retrieval/phase-integrity tests; never read real corpus or weights."""

from contextlib import ExitStack
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import delta_localization as dl


class DistinctWindowTest(unittest.TestCase):
    def test_exhaustive_small_windows_match_obvious_set_oracle(self):
        for length in range(1, 7):
            for values in itertools.product(range(2), repeat=length):
                index = dl.CorpusIndex(np.array(values), 3)
                for window in range(1, length + 1):
                    for ids in ([0], [1], [2], [0, 1], [1, 2]):
                        expected = [len(set(values[s:s + window]) & set(ids))
                                    for s in range(length - window + 1)]
                        actual = dl.distinct_window_scores(index, ids, window)
                        np.testing.assert_array_equal(actual, expected)
                        self.assertEqual(actual.dtype, np.dtype('<u2'))

    def test_random_multiclass_and_inverted_positions(self):
        rng = np.random.default_rng(81)
        for _ in range(30):
            tokens = rng.integers(0, 20, size=120)
            ids = rng.choice(25, size=12, replace=False)
            index = dl.CorpusIndex(tokens, 25)
            for token in range(25):
                np.testing.assert_array_equal(index.occurrences(token), np.flatnonzero(tokens == token))
            for window in (1, 9, 29, 120):
                expected = [len(set(tokens[s:s + window]) & set(ids))
                            for s in range(len(tokens) - window + 1)]
                np.testing.assert_array_equal(dl.distinct_window_scores(index, ids, window), expected)

    def test_last_window_and_repetition_and_counts_above_uint8(self):
        index = dl.CorpusIndex([0, 0, 0, 1, 2], 3)
        np.testing.assert_array_equal(dl.distinct_window_scores(index, [0, 1, 2], 3), [1, 2, 3])
        index = dl.CorpusIndex(np.arange(300), 301)
        np.testing.assert_array_equal(dl.distinct_window_scores(index, np.arange(300), 300), [300])

    def test_invalid_ids_corpus_and_window(self):
        for tokens in ([], [-1], [3], [0.0], [True], [[0]]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                dl.CorpusIndex(tokens, 3)
        index = dl.CorpusIndex([0, 1, 2], 3)
        for ids in ([], [0, 0], [-1], [3], [0.0], [True]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                dl.distinct_window_scores(index, ids, 1)
        for window in (0, 4, -1, True, 1.0):
            with self.subTest(window=window), self.assertRaises(ValueError):
                dl.distinct_window_scores(index, [0], window)


class SelectionAndSupportTest(unittest.TestCase):
    def test_nms_matches_sorted_greedy_bruteforce_and_preserves_scores(self):
        rng = np.random.default_rng(2)
        for _ in range(30):
            scores = rng.integers(0, 6, size=50)
            original = scores.copy()
            for width in (1, 2, 7, 100):
                expected = []
                for start in sorted(range(len(scores)), key=lambda s: (-scores[s], s)):
                    if scores[start] == 0:
                        continue
                    if all(abs(start - previous) >= width for previous in expected):
                        expected.append(start)
                        if len(expected) == 8:
                            break
                selected = dl.select_windows(scores, width, 8)
                self.assertEqual([item['start'] for item in selected], expected)
                np.testing.assert_array_equal(scores, original)

    def test_ties_adjacent_windows_and_plateaus_are_half_open_start_ranges(self):
        scores = np.array([4, 4, 4, 4, 4, 2, 2, 0])
        selected = dl.select_windows(scores, 2, 10)
        self.assertEqual([item['start'] for item in selected], [0, 2, 4, 6])
        self.assertEqual([(item['plateau_start'], item['plateau_end']) for item in selected],
                         [(0, 5), (0, 5), (0, 5), (5, 7)])
        self.assertEqual(dl.score_plateau(scores, 4), (0, 5))
        self.assertEqual(dl.score_plateau(scores, 7), (7, 8))
        self.assertEqual(dl.select_windows(np.zeros(8, dtype=int), 2), [])

    def test_support_preserves_occurrences_and_native_byte_boundaries(self):
        index = dl.CorpusIndex([0, 1, 1, 2, 0, 3], 4)
        ids = [1, 0, 2]
        scores = dl.distinct_window_scores(index, ids, 4)
        selected = dl.select_windows(scores, 4)
        offsets = np.array([0, 1, 3, 5, 8, 9, 13])
        result = dl.add_window_support(index, ids, selected, offsets)
        self.assertEqual(result[0]['start'], 0)
        self.assertEqual(result[0]['end'], 4)
        self.assertEqual(result[0]['byte_start'], 0)
        self.assertEqual(result[0]['byte_end'], 8)
        self.assertEqual(result[0]['score'], 3)
        self.assertEqual(result[0]['support'], [
            dict(token_id=1, token_positions=[1, 2], byte_ranges=[[1, 3], [3, 5]]),
            dict(token_id=0, token_positions=[0], byte_ranges=[[0, 1]]),
            dict(token_id=2, token_positions=[3], byte_ranges=[[5, 8]])])

    def test_score_offset_and_selection_bounds_validation(self):
        for scores in ([], [1.0], [-1], [65536], [True], [[1]]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                dl.select_windows(scores)
        for start in (-1, 2, True, 0.0):
            with self.assertRaises(ValueError):
                dl.score_plateau([1, 2], start)
        for window, limit in ((0, 1), (1, 0), (True, 1), (1, 10001)):
            with self.assertRaises(ValueError):
                dl.select_windows([1], window, limit)
        index = dl.CorpusIndex([0, 1, 0], 2)
        for offsets in ([0, 1, 2], [0, 1, 1, 3], [0., 1., 2., 3.], [1, 2, 3, 4]):
            with self.assertRaises(ValueError):
                dl.add_window_support(index, [0], [], offsets)
        with self.assertRaisesRegex(ValueError, 'distinct support'):
            dl.add_window_support(index, [0], [dict(start=0, end=2, score=2)], np.arange(4))


class FrequencyControlTest(unittest.TestCase):
    def test_exact_integer_bucket_boundaries(self):
        counts = np.array([0, 1, 2, 3, 4, 7, 8, 2**53-1, 2**53, 2**63], dtype=np.uint64)
        np.testing.assert_array_equal(dl.frequency_buckets(counts), [-1, 0, 1, 1, 2, 2, 3, 52, 53, 63])

    def test_controls_are_deterministic_unique_and_range_not_exact_matched(self):
        counts = np.array([0, 0, 1, 1, 2, 3, 2, 3, 4, 7, 5, 6])
        ids = [0, 2, 4, 5, 8, 9]
        a = dl.frequency_range_controls(counts, ids, 10)
        b = dl.frequency_range_controls(counts, ids, 10)
        buckets = dl.frequency_buckets(counts)
        for first, second in zip(a, b):
            np.testing.assert_array_equal(first, second)
            self.assertEqual(len(set(first)), len(ids))
            np.testing.assert_array_equal(buckets[first], buckets[ids])
        self.assertTrue(any(np.any(counts[value] != counts[ids]) for value in a))
        # Independent scalar mapping construction verifies the exact generator
        # stream order across all buckets, including buckets with no selected ID.
        rng = np.random.Generator(np.random.PCG64(20260909))
        mapping = {}
        for bucket in sorted(set(map(int, buckets))):
            group = [i for i, value in enumerate(buckets) if value == bucket]
            mapping.update(zip(group, map(int, rng.permutation(group))))
        self.assertEqual(a[0].tolist(), [mapping[token] for token in ids])

    def test_singleton_fixed_points_retained_and_invalid_inputs(self):
        result = dl.frequency_range_controls([0, 1, 2, 4], [0, 1, 2, 3], 2)
        for ids in result:
            np.testing.assert_array_equal(ids, [0, 1, 2, 3])
        for counts in ([], [-1], [1.0], [True], [[1]]):
            with self.assertRaises(ValueError):
                dl.frequency_buckets(counts)
        for count, seed in ((0, 17), (1001, 17), (1, -1), (1, True), (1, 2**128)):
            with self.assertRaises(ValueError):
                dl.frequency_range_controls([1, 2], [0], count, seed)


class RunIntegrityTest(unittest.TestCase):
    def fake_run(self, root, *, mutate=None):
        """Mock only upstream authentication/native decoding on tiny fixtures.

        The actual plan/output ordering, source/input hashing, counting,
        permutations, full-array writes, and final rechecks remain exercised.
        """
        data = np.array([0, 1, 2, 0, 3, 1, 4, 0, 2, 3, 1, 0], dtype='<i4')
        paths = {name: root/name for name in ('corpus', 'tokens', 'offsets', 'protocol',
                                             'restart', 'followthrough')}
        paths['corpus'].write_bytes(bytes(65 + int(t) for t in data))
        paths['tokens'].write_bytes(data.tobytes())
        paths['offsets'].write_bytes(np.arange(len(data)+1, dtype='<u8').tobytes())
        for name in ('protocol', 'restart', 'followthrough'):
            paths[name].write_text(name)
        (root/'tokenizer.json').write_text('{}')
        sources = {'protocol': dl.file_record(paths['protocol'])}
        raw_candidate = dict(token_ids=[0, 1, 2], identity_shuffled_token_ids=[5, 6, 7])
        restart = {'candidate_lists': {name: dict(raw_candidate) for name in dl.RESTART_NAMES}}
        followthrough = {'candidate_lists': {name: dict(raw_candidate) for name in dl.FOLLOWTHROUGH_NAMES}}
        upstream = {'sources': {}, 'files': {}}
        real_file_record = dl.file_record
        output = root/'out'

        def guarded_hash(path):
            if Path(path) in (paths['corpus'], paths['tokens'], paths['offsets'], root/'tokenizer.json'):
                self.assertTrue((output/'plan.json').is_file(), 'input hashed before plan')
            return real_file_record(path)

        def native(*args):
            self.assertFalse((output/'frozen.json').exists())
            plan = json.loads((output/'plan.json').read_text())
            self.assertTrue(plan['no_sampler_access'])
            self.assertEqual(len(plan['panel_lists']), 18)
            if mutate:
                paths[mutate].write_bytes(paths[mutate].read_bytes() + b'changed')
            return paths['corpus'].read_bytes(), data, np.arange(len(data)+1), {
                token: bytes([65+token]) for token in range(8)}, 8, 8

        with ExitStack() as stack:
            for name, value in (('VOCABULARY', 8), ('TOP_K', 3), ('WINDOW', 3),
                                ('MAX_WINDOWS', 2), ('CONTROL_COUNT', 2)):
                stack.enter_context(mock.patch.object(dl, name, value))
            stack.enter_context(mock.patch.object(dl, '_sources', return_value=sources))
            stack.enter_context(mock.patch.object(dl, 'file_record', side_effect=guarded_hash))
            stack.enter_context(mock.patch.object(dl.earlier, 'authenticate', return_value=(
                upstream, restart, real_file_record(paths['restart']))))
            stack.enter_context(mock.patch.object(dl.later, 'authenticate', return_value=(
                upstream, followthrough, real_file_record(paths['followthrough']))))
            stack.enter_context(mock.patch.object(dl.earlier, 'load_native', side_effect=native))
            stack.enter_context(mock.patch('builtins.print'))
            return dl.run(paths['restart'], paths['followthrough'], paths['corpus'], paths['tokens'],
                          paths['offsets'], root, paths['protocol'], output)

    def test_plan_precedes_corpus_access_complete_arrays_then_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.fake_run(root)
            self.assertEqual(len(result['lists']), 20)
            self.assertEqual(result['corpus']['prefix_tokens'], 8)
            frozen = json.loads((root/'out/frozen.json').read_text())
            self.assertTrue(frozen['complete'])
            self.assertEqual(len(frozen['files']), 23)
            for record in frozen['files'].values():
                self.assertEqual(dl.file_record(record['path']), record)
            for item in result['lists'].values():
                with np.load(item['score_file']['path'], allow_pickle=False) as arrays:
                    self.assertEqual(arrays.files, ['scores'])
                    self.assertEqual(arrays['scores'].shape, (10,))
                    for window in item['selected_windows']:
                        self.assertEqual(int(arrays['scores'][window['start']]), window['score'])
            plan = json.loads((root/'out/plan.json').read_text())
            self.assertEqual(len(plan['panel_lists']), 18)  # Not mutated by later control creation.

    def test_changed_corpus_or_protocol_cannot_emit_completion(self):
        for field in ('corpus', 'protocol'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaisesRegex(ValueError, 'changed'):
                    self.fake_run(root, mutate=field)
                self.assertTrue((root/'out/plan.json').exists())
                self.assertFalse((root/'out/frozen.json').exists())
                self.assertFalse((root/'out/localization.json').exists())

    def test_existing_output_is_not_reused_and_bad_freeze_never_loads_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'out').mkdir()
            with mock.patch.object(dl.earlier, 'authenticate') as authentication:
                with self.assertRaises(FileExistsError):
                    dl.run(*(root/'missing' for _ in range(7)), root/'out')
                authentication.assert_not_called()
            with mock.patch.object(dl.earlier, 'load_native') as loader:
                with self.assertRaises(FileNotFoundError):
                    dl.run(*(root/'missing' for _ in range(7)), root/'new')
                loader.assert_not_called()
                self.assertFalse((root/'new/plan.json').exists())


if __name__ == '__main__':
    unittest.main()
