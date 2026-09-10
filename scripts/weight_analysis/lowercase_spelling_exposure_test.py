"""Independent synthetic tests; no native executable or model is launched."""

import unittest

from .lowercase_spelling_exposure import read_starts, surviving_positions
from .paired_training_exposure import exposure


class LowercaseSpellingExposureTest(unittest.TestCase):
    def test_native_survivor_not_uppercase_or_isolated_suffix(self):
        ids = [1475, 68, 2797, 409, 68, 2797, 1086, 2797]
        replacement = [21733, 45177, 400, 409, 68, 2797, 1086, 2797]
        pieces = [b' Ex', b'e', b'unt', b' ex', b'e', b'unt', b' Bl', b'unt']
        offsets = [0]
        for p in pieces:
            offsets.append(offsets[-1] + len(p))
        self.assertEqual(surviving_positions(ids, replacement, offsets, b''.join(pieces)), [3])

    def test_claimed_native_ids_require_exact_piece_bytes(self):
        with self.assertRaises(ValueError):
            surviving_positions([409, 68, 2797], [409, 68, 2797], [0, 3, 4, 7], b' Exeunt')
        with self.assertRaises(ValueError):
            surviving_positions([409, 68, 2797], [1, 68, 2797], [0, 3, 4, 7], b' exeunt')

    def test_empty_and_too_short_inputs(self):
        self.assertEqual(surviving_positions([1], [1], [0, 1], b'x'), [])
        self.assertEqual(surviving_positions([1, 2], [1, 2], [0, 1, 2], b'xy'), [])

    def test_invalid_offsets_and_shapes(self):
        for a, b, offsets, text in [([1], [1, 2], [0, 1], b'a'),
                                    ([1], [1], [0, 0], b''),
                                    ([1.0], [1], [0, 1], b'a')]:
            with self.subTest(a=a), self.assertRaises(ValueError):
                surviving_positions(a, b, offsets, text)

    def test_sampler_seed_count_and_inclusive_final_start(self):
        self.assertEqual(read_starts(b'17\n0\n5\n', seed=17, count=2,
                                     token_count=10, context_length=4), [0, 5])
        self.assertEqual(read_starts(b'17\n', seed=17, count=0,
                                     token_count=10, context_length=4), [])
        for payload in (b'18 0 5', b'17 0', b'17 0 6', b'17 -1 0', b'17 1.5 0'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                read_starts(payload, seed=17, count=2, token_count=10, context_length=4)

    def test_last_target_may_lie_outside_input(self):
        # For input [0,4), targets are [1,5); the word at [2,5) is fully
        # supervised, although its last token is not an input in this window.
        result = exposure([0, 2], [2], token_count=8, context_length=4)
        self.assertEqual(result['complete_target_counts'], [1])
        self.assertEqual(result['partial_target_window_counts'], [1])
        self.assertEqual(result['piece_target_counts'], [[1, 2, 2]])


if __name__ == '__main__':
    unittest.main()
