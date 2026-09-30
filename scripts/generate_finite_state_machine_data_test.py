#!/usr/bin/env python3
"""Independent interpreter checks of generated and committed FSM data."""

from collections import defaultdict
from pathlib import Path
import re
import string
import unittest

import generate_finite_state_machine_data as generator


def parse_and_check(line: str) -> tuple[tuple, str, str]:
    """Check each answer without calling the generator's evaluator."""
    assert line.count(">") == 1
    description_and_input, answer = line.split(">")
    fields = description_and_input.split(";")
    assert len(fields) >= 2
    input_text = fields[-1]
    assert re.fullmatch(r"[A-Z]{1,24}", input_text), line
    assert re.fullmatch(r"[0-9]{3}|ERR", answer), line
    transitions, states = {}, set()
    for edge in fields[:-1]:
        assert re.fullmatch(r"[0-9]{3}[A-Z][0-9]{3}", edge), edge
        source, letter, destination = edge[:3], edge[3], edge[4:]
        assert (source, letter) not in transitions, edge
        transitions[source, letter] = destination
        states.update((source, destination))
    assert 4 <= len(states) <= 16
    assert "000" in states
    assert {c for _, c in transitions} == set(string.ascii_uppercase)
    state = "000"
    for letter in input_text:
        if (state, letter) not in transitions:
            state = "ERR"
            break
        state = transitions[state, letter]
    assert state == answer, (line, state)
    return tuple(sorted(transitions.items())), input_text, answer


class FiniteStateMachineDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.training, cls.test = generator.generate()

    def test_sentences_and_split_independence(self) -> None:
        signatures = []
        for lines, size in ((self.training, 4096), (self.test, 128)):
            self.assertEqual(len(lines), size)
            self.assertEqual(len(set(lines)), size)
            machines = defaultdict(list)
            for line in lines:
                signature, input_text, answer = parse_and_check(line)
                machines[signature].append((input_text, answer))
            self.assertEqual(len(machines), size // 2)
            for pair in machines.values():
                self.assertEqual(len(pair), 2)
                self.assertEqual(sum(y == "ERR" for _, y in pair), 1)
                left, right = pair[0][0], pair[1][0]
                self.assertEqual(len(left), len(right))
                self.assertEqual(sum(a != b for a, b in zip(left, right)), 1)
            signatures.append(set(machines))
        self.assertFalse(signatures[0] & signatures[1])

    def test_committed_files_reproduce_exactly(self) -> None:
        directory = Path(__file__).resolve().parents[1] / "testdata"
        for name, lines in (("training", self.training), ("test", self.test)):
            path = directory / f"finite_state_machine_{name}_data.txt"
            self.assertEqual(
                path.read_text(encoding="ascii"), "\n".join(lines) + "\n"
            )

    def test_evaluator_edge_cases(self) -> None:
        transitions = {(0, "A"): 31, (31, "B"): 0, (31, "C"): 31}
        for input_text, expected in (
            ("", "000"),
            ("A", "031"),
            ("AB", "000"),
            ("ACC", "031"),
            ("X", "ERR"),
            ("AX", "ERR"),
            ("ABX", "ERR"),
        ):
            with self.subTest(input_text=input_text):
                self.assertEqual(generator.evaluate(transitions, input_text), expected)

    def test_variety(self) -> None:
        lengths, letters, answers = set(), set(), set()
        for line in self.training:
            _, input_text, answer = parse_and_check(line)
            lengths.add(len(input_text))
            letters.update(input_text)
            answers.add(answer)
        self.assertEqual(lengths, set(range(1, 25)))
        self.assertEqual(letters, set(string.ascii_uppercase))
        self.assertIn("000", answers)
        self.assertGreater(len(answers), 500)


if __name__ == "__main__":
    unittest.main()
