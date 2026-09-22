#!/usr/bin/env python3
"""Exact language/output equivalence for compiled attention control flow."""

import copy
from dataclasses import FrozenInstanceError
import itertools
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest

from discretize_attention_logic import build_attention, evaluate, render_attention


def histories(alphabet, maximum):
    return [key for length in range(maximum + 1)
            for key in itertools.product(alphabet, repeat=length)]


class AttentionLogicTest(unittest.TestCase):
    def test_shared_suffixes_minimize_only_when_outputs_and_future_edges_match(self):
        rows = [[[1], 10], [[1, 2], 20], [[3], 10], [[3, 2], 20]]
        program = build_attention(rows)
        self.assertEqual(program.trie_nodes, 5)
        self.assertEqual(len(program.nodes), 3)
        self.assertEqual(evaluate(program, [1, 2]), 20)
        self.assertEqual(evaluate(program, [3, 2]), 20)
        rows[-1][1] = 21
        distinct = build_attention(rows)
        self.assertEqual(len(distinct.nodes), 5)
        self.assertEqual(evaluate(distinct, [3, 2]), 21)

    def test_exact_domain_rejects_truncations_extensions_and_wrong_order(self):
        rows = [[[1, 2], 0], [[3, 2], 0xffffffff]]
        program = build_attention(rows)
        self.assertEqual(evaluate(program, [1, 2]), 0)
        self.assertEqual(evaluate(program, [3, 2]), 0xffffffff)
        for invalid in ([], [1], [3], [2, 1], [1, 2, 3], [9, 2], [-1], [True]):
            with self.subTest(history=invalid):
                self.assertIsNone(evaluate(program, invalid))

    def test_accepting_zero_cannot_merge_with_a_nonaccepting_prefix(self):
        program = build_attention([[[1], 0], [[1, 3], 9], [[2, 3], 9]])
        self.assertEqual(evaluate(program, [1]), 0)
        self.assertIsNone(evaluate(program, [2]))
        self.assertEqual(evaluate(program, [2, 3]), 9)

    def test_random_models_match_exact_dictionary_on_every_small_history(self):
        randomizer = random.Random(7041)
        candidates = histories(range(3), 5)
        for trial in range(20):
            selected = randomizer.sample(candidates[1:], 40)
            table = {key: randomizer.randrange(12) for key in selected}
            program = build_attention([[list(key), output] for key, output in table.items()])
            for key in candidates:
                self.assertEqual(evaluate(program, key), table.get(key), (trial, key))
            for key in selected:
                self.assertIsNone(evaluate(program, key + (99,)))

    def test_determinism_duplicate_rows_and_immutable_program(self):
        rows = [[[1, 2], 5], [[1], 3], [[4, 2], 5], [[4], 3]]
        original = copy.deepcopy(rows)
        program = build_attention(rows)
        source, stats = render_attention("Attention0", rows)
        reordered = list(reversed(rows)) + [rows[0]]
        self.assertEqual(program, build_attention(reordered))
        self.assertEqual((source, stats), render_attention("Attention0", reordered))
        self.assertEqual(rows, original)
        with self.assertRaises(FrozenInstanceError):
            program.root = 0
        with self.assertRaises(FrozenInstanceError):
            program.nodes[0].output = 7

    def test_malformed_rows_conflicts_and_cpp_names_are_rejected(self):
        for rows in ([[[1], 2], [[1], 3]], [[[], 0]], [[[-1], 1]],
                     [[[True], 1]], [[[1], 0x100000000]], [["12", 4]], [[1]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                build_attention(rows)
        for name in ("class", "Bad::Name", "1Bad", "_Reserved", "Bad__Name", "x;}"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                render_attention(name, [])
        for size in (0, -1, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                render_attention("Attention", [], chunk_size=size)
        with self.assertRaises(ValueError):
            render_attention("Attention", [], strategy="approximate")

    def test_empty_table_and_generated_control_flow(self):
        program = build_attention([])
        self.assertIsNone(evaluate(program, []))
        self.assertIsNone(evaluate(program, [1]))
        source, stats = render_attention("Attention", [[[1], 10], [[1, 2], 20]], chunk_size=1)
        self.assertIn("PLUTO_ATTN_MATCH", source)
        self.assertIn("goto n", source)
        self.assertGreater(stats["helpers"], 1)
        self.assertEqual(stats["source_bytes"], len(source.encode()))
        self.assertEqual(stats["flat_scalars"], 9)
        self.assertNotIn("kKeys[]", source)
        self.assertNotIn("AttentionRow", source)
        self.assertNotIn("float", source)

    def test_hybrid_uses_narrow_literal_runs_without_changing_hidden_ids(self):
        chain = tuple(range(65000, 65012))
        rows = [[list(chain[:length]), 65535 - length]
                for length in range(1, len(chain) + 1)]
        source, stats = render_attention("NarrowAttention", rows)
        _, pure = render_attention("PureAttention", rows, strategy="control_flow")
        self.assertGreater(stats["literal_sequence_steps"], 4)
        self.assertEqual(stats["literal_sequence_word_bits"], 16)
        self.assertIn("std::uint16_t symbol, output", source)
        self.assertIn("65001u", source)
        self.assertIn("65534u", source)
        self.assertEqual(pure["literal_sequence_patterns"], 0)

    def test_cpp_execution_matches_all_outputs_and_unsupported_histories(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("a C++20 compiler is unavailable")
        randomizer = random.Random(1059)
        candidates = histories(range(3), 5)
        selected = randomizer.sample(candidates[1:], 80)
        table = {key: randomizer.randrange(20) for key in selected}
        # Large StateIds and the supported zero output must survive emission.
        table[(0xffffffff,)] = 0
        table[(0xffffffff, 7)] = 0xffffffff
        chain = (70000, 8, 9, 10, 11, 12, 13, 14, 0xffffffff)
        for length in range(1, len(chain) + 1):
            table[chain[:length]] = 0xfffffffe if length == 5 else length
        rows = [[list(key), output] for key, output in table.items()]
        generated, stats = render_attention("CompiledAttention", rows, chunk_size=7)
        pure, _ = render_attention("PureAttention", rows, chunk_size=7, strategy="control_flow")
        empty, _ = render_attention("EmptyAttention", [])
        narrow_rows = [[list(range(1, length + 1)), length - 1]
                       for length in range(1, 9)]
        narrow, narrow_stats = render_attention("NarrowAttention", narrow_rows)
        self.assertEqual(narrow_stats["literal_sequence_word_bits"], 16)
        self.assertGreater(narrow_stats["literal_sequence_patterns"], 0)
        self.assertGreater(stats["helpers"], 1)
        self.assertGreater(stats["literal_sequence_patterns"], 0)
        self.assertEqual(stats["literal_sequence_word_bits"], 32)
        source = """#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <span>
#include <vector>
namespace absl { template<class T> using Span = std::span<T>; }
using StateId = std::uint32_t;
using TransitionResult = std::optional<StateId>;
""" + generated + pure + empty + narrow + """
int main() {
  std::vector<StateId> narrow_key;
  if (NarrowAttention(narrow_key).has_value()) return 4;
  for (StateId length = 1; length <= 8; ++length) {
    narrow_key.push_back(length);
    if (NarrowAttention(narrow_key) != TransitionResult{length - 1}) return 5;
  }
  narrow_key.push_back(9);
  if (NarrowAttention(narrow_key).has_value()) return 6;
  std::size_t count;
  while (std::cin >> count) {
    std::vector<StateId> key(count);
    for (auto& symbol : key) std::cin >> symbol;
    if (EmptyAttention(key).has_value()) return 2;
    auto result = CompiledAttention(key);
    auto original = PureAttention(key);
    if (result != original) return 3;
    if (result.has_value()) std::cout << *result << '\\n';
    else std::cout << "unsupported\\n";
  }
}
"""
        checked = candidates + [(0xffffffff,), (0xffffffff, 7), (0xffffffff, 7, 1)]
        checked += [key + (99,) for key in selected]
        checked += [chain[:length] for length in range(1, len(chain) + 1)]
        checked += [key + (99,) for key in table]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attention.cc"
            executable = Path(directory) / "attention"
            path.write_text(source, encoding="utf-8")
            built = subprocess.run([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                                    str(path), "-o", str(executable)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            request = "".join(str(len(key)) + " " + " ".join(map(str, key)) + "\n" for key in checked)
            ran = subprocess.run([str(executable)], input=request, capture_output=True, text=True)
            self.assertEqual(ran.returncode, 0, ran.stderr)
            self.assertEqual(ran.stdout.splitlines(),
                             [str(table[key]) if key in table else "unsupported" for key in checked])


if __name__ == "__main__":
    unittest.main()
