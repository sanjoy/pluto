"""Synthetic fixed-ranking verification tests; no real corpus/checkpoint use."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import restart_delta_verify as verifier


class CountsAndReplayTest(unittest.TestCase):
    def test_split_counts_and_ranked_frequency_order(self):
        tokens = np.array([0, 1, 1, 2, 3, 3, 3], dtype=np.int32)
        counts = verifier.frequencies(tokens, 4, 5)
        np.testing.assert_array_equal(counts['full'], [1, 2, 1, 3, 0])
        np.testing.assert_array_equal(counts['current_prefix'], [1, 2, 1, 0, 0])
        np.testing.assert_array_equal(counts['current_suffix'], [0, 0, 0, 3, 0])
        actual = verifier.summarize_frequencies(counts['full'], [4, 1, 3])
        self.assertEqual(actual['counts_in_rank_order'], [0, 2, 3])
        self.assertEqual(actual['ranked_token_ids_present'], 2)
        self.assertEqual(actual['selected_token_occurrences'], 5)
        self.assertEqual(actual['selected_share_of_corpus_tokens'], 5 / 7)

    def test_union_contains_first_input_and_final_target_no_duplicate_counts(self):
        tokens = np.array([0, 1, 2, 3, 4, 4])
        starts = np.array([[0, 1], [2, 3]])
        actual = verifier.window_presence(tokens, starts, 2, 1, 6)
        np.testing.assert_array_equal(np.flatnonzero(actual[0]), [0, 1, 2])
        np.testing.assert_array_equal(np.flatnonzero(actual[1]), [2, 3, 4])
        both = verifier.window_presence(tokens, starts, 2, 2, 6)
        np.testing.assert_array_equal(np.flatnonzero(both[0]), [0, 1, 2, 3])
        np.testing.assert_array_equal(np.flatnonzero(both[1]), [2, 3, 4])

    def test_overlaps_keep_all_alternatives_and_descriptive_ties(self):
        presence = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0],
                             [1, 1, 0, 0], [0, 0, 0, 0]], dtype=bool)
        actual = verifier.summarize_overlap(presence, [2, 0, 1])
        self.assertEqual(actual['seed17_overlap'], 2)
        self.assertEqual(actual['alternative_overlap_counts_in_seed_order'], [3, 1, 2, 0])
        self.assertEqual(actual['alternatives_greater_or_equal_to_seed17'], 2)
        self.assertEqual(actual['seed17_descriptive_rank_among_all_seeds'], 2)
        self.assertEqual(actual['alternative_mean'], 1.5)
        self.assertEqual(actual['alternative_histogram'], {'0': 1, '1': 1, '2': 1, '3': 1})

    def test_invalid_bounds_and_duplicate_ids_fail_closed(self):
        for starts, context, count in (([[-1]], 2, 1), ([[4]], 2, 1),
                                       ([[0]], 0, 1), ([[0]], 2, 2), ([[0.]], 2, 1)):
            with self.subTest(starts=starts, context=context):
                with self.assertRaises(ValueError):
                    verifier.window_presence(np.arange(6), np.array(starts), context, count, 6)
        with self.assertRaises(ValueError):
            verifier.summarize_frequencies(np.ones(6, dtype=int), [1, 1])
        with self.assertRaises(ValueError):
            verifier.summarize_overlap(np.zeros((2, 6), dtype=bool), [1, 1])
        with self.assertRaises(ValueError):
            verifier.frequencies(np.arange(6), 6, 6)

    def test_sampler_cli_parser_checks_seed_order_shape_and_range(self):
        with mock.patch.object(verifier, 'WINDOW_COUNTS', (1, 2)):
            valid = subprocess.CompletedProcess([], 0, '17 0 3\n18 1 2\n', '')
            with mock.patch.object(subprocess, 'run', return_value=valid):
                starts, record = verifier.sampler_rows('/fake/sampler', 6, 2, 17, 2)
            np.testing.assert_array_equal(starts, [[0, 3], [1, 2]])
            self.assertEqual(record['seed_count'], 2)
            for bad in ('17 0 4\n18 1 2\n', '18 0 3\n17 1 2\n', '17 0\n18 1 2\n',
                        '17 +0 3\n18 1 2\n', '17 0 3\n'):
                with mock.patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, bad, '')):
                    with self.assertRaises(ValueError):
                        verifier.sampler_rows('/fake/sampler', 6, 2, 17, 2)

    def test_expected_list_set_includes_all_comparison_arms(self):
        names = verifier.expected_names()
        self.assertEqual(len(names), 42)
        for name in ('boundary_raw_580', 'ordinary_before_adjusted_8160',
                     'shared_boundary_adjusted', 'shared_ordinary_after_raw',
                     'endpoint_norm_7230', 'shared_endpoint_norm'):
            self.assertIn(name, names)


class FrozenGateTest(unittest.TestCase):
    def test_missing_freeze_cannot_open_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(verifier, 'load_native', side_effect=AssertionError('corpus opened')):
                with self.assertRaises(FileNotFoundError):
                    verifier.verify_files(root / 'missing_frozen.json', root / 'corpus', root / 'tokens',
                                          root / 'offsets', root, root / 'sampler', root / 'report.json')
            self.assertFalse((root / 'report_start.json').exists())

    def test_incomplete_freeze_cannot_open_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / 'frozen.json'
            frozen.write_text(json.dumps({'schema_version': 1, 'complete': False}))
            with mock.patch.object(verifier, 'load_native', side_effect=AssertionError('corpus opened')):
                with self.assertRaisesRegex(ValueError, 'completely frozen'):
                    verifier.verify_files(frozen, root / 'corpus', root / 'tokens', root / 'offsets',
                                          root, root / 'sampler', root / 'report.json')
            self.assertFalse((root / 'report_start.json').exists())

    def test_existing_output_rejected_before_any_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'report.json'
            output.write_text('preserve me')
            with mock.patch.object(verifier, 'authenticate', side_effect=AssertionError('unexpected input read')):
                with self.assertRaises(FileExistsError):
                    verifier.verify_files(root / 'freeze', root / 'corpus', root / 'tokens', root / 'offsets',
                                          root, root / 'sampler', output)
            self.assertEqual(output.read_text(), 'preserve me')


if __name__ == '__main__':
    unittest.main()
