"""CLI tests for current-runtime conditional sampling, without corpus access.

The scalar Python oracle separately implements MT19937-64 and the current
libstdc++ 13 multiply/reject integer mapping. This is an alternate-language
algorithm check, not evidence about the implementation used during training.
Portable shape/range/validation tests apply on other standard libraries too.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest


MASK64 = (1 << 64) - 1


def mt19937_64(seed):
    """Scalar recurrence with Python integers and explicit uint64 truncation.

    Constants and recurrence are checked against installed libstdc++ 13's
    bits/random.h and bits/random.tcc; no C++ sampler output seeds this oracle.
    In-place state updates matter after the middle of each 312-word twist.
    """
    state = [seed & MASK64]
    for index in range(1, 312):
        previous = state[-1]
        state.append((6364136223846793005 * (previous ^ (previous >> 62)) + index) & MASK64)
    while True:
        for index in range(312):
            combined = (state[index] & 0xFFFFFFFF80000000) | (state[(index + 1) % 312] & 0x7FFFFFFF)
            state[index] = state[(index + 156) % 312] ^ (combined >> 1)
            if combined & 1:
                state[index] ^= 0xB5026F5AA96619E9
        for value in state:
            value ^= (value >> 29) & 0x5555555555555555
            value ^= (value << 17) & 0x71D67FFFEDA60000
            value ^= (value << 37) & 0xFFF7EEE000000000
            value ^= value >> 43
            yield value & MASK64


def libstdcxx13_starts(total_tokens, context, count, seed):
    """Unbiased [0, total-context) draw using explicit 128-bit arithmetic.

    This is deliberately not modulo reduction: low products below the
    remainder threshold must be rejected, matching uniform_int_dist.h.
    """
    extent = total_tokens - context
    threshold = (1 << 64) % extent
    engine = mt19937_64(seed)
    output = []
    while len(output) < count:
        product = next(engine) * extent
        if (product & MASK64) >= threshold:
            output.append(product >> 64)
    return output


class ReplaySamplerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('g++')
        if compiler is None:
            raise unittest.SkipTest('standalone replay_sampler tests require g++')
        cls.temporary = tempfile.TemporaryDirectory(prefix='pluto-replay-test-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.binary = Path(cls.temporary.name) / 'replay_sampler'
        subprocess.run([compiler, '-O2', '-std=c++20', '-Wall', '-Wextra', '-Werror',
                        '-pedantic', str(Path(__file__).with_name('replay_sampler.cc')),
                        '-o', str(cls.binary)], check=True, capture_output=True, text=True)
        cls.implementation = subprocess.check_output([str(cls.binary), '--implementation'], text=True)

    def run_sampler(self, *arguments, success=True):
        result = subprocess.run([str(self.binary), *map(str, arguments)],
                                capture_output=True, text=True, timeout=10)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, '')
        else:
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, '')
            self.assertIn('replay_sampler:', result.stderr)
        return result.stdout

    def rows(self, *arguments):
        return [list(map(int, row.split())) for row in self.run_sampler(*arguments).splitlines()]

    def test_documented_cli_and_implementation(self):
        self.assertIn('conditional replay', self.run_sampler('--help'))
        self.assertIn('size_t_bits=', self.implementation)
        self.assertIn('stdlib=', self.implementation)

    def test_deterministic_row_counts_and_bounds(self):
        first = self.rows(1650781, 1024, 100, 10000, 1000)
        self.assertEqual(first, self.rows(1650781, 1024, 100, 10000, 1000))
        self.assertEqual(len(first), 1000)
        for index, row in enumerate(first):
            self.assertEqual(row[0], 10000 + index)
            self.assertEqual(len(row), 101)
            self.assertTrue(all(0 <= start <= 1650781 - 1024 - 1 for start in row[1:]))

    def test_ten_draws_are_prefix_of_hundred_and_seeds_are_independent(self):
        short = self.rows(1650781, 1024, 10, 17, 3)
        longer = self.rows(1650781, 1024, 100, 17, 3)
        for small, large in zip(short, longer):
            self.assertEqual(small, large[:11])
            self.assertEqual(small, self.rows(1650781, 1024, 10, small[0], 1)[0])

    def test_only_valid_start_zero_and_largest_seed(self):
        self.assertEqual(self.rows(1025, 1024, 5, MASK64, 1), [[MASK64, 0, 0, 0, 0, 0]])
        self.assertEqual(self.rows(2, 1, 2, 0, 1), [[0, 0, 0]])

    def test_invalid_syntax_and_argument_counts(self):
        for arguments in ((), (1,), (2, 1, 1, 0, 1, 3), ('--bad',)):
            with self.subTest(arguments=arguments):
                self.run_sampler(*arguments, success=False)
        for malformed in ('-1', '+1', ' 1', '1 ', '1x', '1.0', '', str(1 << 64)):
            for index in range(5):
                arguments = [1025, 1024, 1, 0, 1]
                arguments[index] = malformed
                with self.subTest(malformed=malformed, index=index):
                    self.run_sampler(*arguments, success=False)

    def test_invalid_ranges_overflow_and_output_caps(self):
        for arguments in ((0, 1, 1, 0, 1), (1024, 1024, 1, 0, 1),
                          (1025, 0, 1, 0, 1), (1 << 33, 1 << 31, 1, 0, 1),
                          (MASK64, 1, 1, 0, 1), (1025, 1024, 0, 0, 1),
                          (1025, 1024, 1, 0, 0), (1025, 1024, 10001, 0, 1),
                          (1025, 1024, 1, 0, 10001), (1025, 1024, 10000, 0, 10000),
                          (1025, 1024, 1, MASK64, 2)):
            with self.subTest(arguments=arguments):
                self.run_sampler(*arguments, success=False)

    def test_seed17_matches_independent_current_libstdcxx_algorithm(self):
        if 'stdlib=libstdc++' not in self.implementation or 'stdlib_release=13' not in self.implementation:
            self.skipTest('oracle intentionally targets current libstdc++ 13 mapping')
        for total, context, count, seed in ((1650781, 1024, 1000, 17),
                                            (1650781, 1024, 100, 10000),
                                            ((1 << 61) + 2, 1, 1000, 17)):
            with self.subTest(total=total, seed=seed):
                self.assertEqual(self.rows(total, context, count, seed, 1)[0],
                                 [seed, *libstdcxx13_starts(total, context, count, seed)])


if __name__ == '__main__':
    unittest.main()
