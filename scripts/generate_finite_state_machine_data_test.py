#!/usr/bin/env python3
"""Independent interpreter checks of generated and committed FSM data."""

from collections import defaultdict
from pathlib import Path
import re
import string
import subprocess
import sys
import tempfile
import unittest

import generate_finite_state_machine_data as generator


def parse_and_check(line: str, full_trace: bool = False) -> tuple[tuple, str, str]:
    """Check each answer without calling the generator's evaluator."""
    assert line.count(">") == 1
    description_and_input, answer = line.split(">")
    fields = description_and_input.split(";")
    assert len(fields) >= 2
    input_text = fields[-1]
    assert re.fullmatch(r"[A-Z]{1,24}", input_text), line
    if full_trace:
        assert re.fullmatch(r"000(?: [0-9]{3})*(?: ERR)?", answer), line
    else:
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
    visited = [state]
    for letter in input_text:
        if (state, letter) not in transitions:
            state = "ERR"
            visited.append(state)
            break
        state = transitions[state, letter]
        visited.append(state)
    expected_answer = " ".join(visited) if full_trace else state
    assert expected_answer == answer, (line, expected_answer)
    return tuple(sorted(transitions.items())), input_text, state


class FiniteStateMachineDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.training, cls.test = generator.generate()
        cls.full_training, cls.full_test = generator.generate(full_trace=True)

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

    def test_full_trace_sentences_match_original_prompts(self) -> None:
        signatures = []
        for original, full, size in (
            (self.training, self.full_training, 4096),
            (self.test, self.full_test, 128),
        ):
            self.assertEqual(len(full), size)
            self.assertEqual(len(set(full)), size)
            machines = set()
            errors = 0
            for original_line, full_line in zip(original, full):
                self.assertEqual(original_line.split(">")[0], full_line.split(">")[0])
                signature, _, final_state = parse_and_check(full_line, full_trace=True)
                self.assertEqual(final_state, original_line.split(">")[1])
                machines.add(signature)
                errors += final_state == "ERR"
            self.assertEqual(errors, size // 2)
            self.assertEqual(len(machines), size // 2)
            signatures.append(machines)
        self.assertFalse(signatures[0] & signatures[1])

    def test_full_trace_committed_files_reproduce_exactly(self) -> None:
        directory = Path(__file__).resolve().parents[1] / "testdata"
        for name, lines in (
            ("training", self.full_training), ("test", self.full_test)
        ):
            path = directory / f"finite_state_machine_full_{name}_data.txt"
            self.assertEqual(
                path.read_text(encoding="ascii"), "\n".join(lines) + "\n"
            )

    def test_full_trace_exact_examples(self) -> None:
        transitions = {(0, "X"): 93, (93, "A"): 44}
        self.assertEqual(
            generator.execution_trace(transitions, "XA"), ["000", "093", "044"]
        )
        self.assertEqual(
            generator.execution_trace(transitions, "XB"), ["000", "093", "ERR"]
        )

    def test_full_trace_cycles_and_failure_timing(self) -> None:
        transitions = {(0, "A"): 31, (31, "B"): 0, (31, "C"): 31}
        for input_text, expected in (
            ("", ["000"]),
            ("A", ["000", "031"]),
            ("ABAB", ["000", "031", "000", "031", "000"]),
            ("ACC", ["000", "031", "031", "031"]),
            ("XABA", ["000", "ERR"]),
            ("AXAB", ["000", "031", "ERR"]),
            ("ABX", ["000", "031", "000", "ERR"]),
        ):
            with self.subTest(input_text=input_text):
                self.assertEqual(
                    generator.execution_trace(transitions, input_text), expected
                )

    def test_full_trace_cli_does_not_overwrite_final_answer_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("training", "test"):
                (directory / f"finite_state_machine_{name}_data.txt").write_text(
                    "preserve this final-answer dataset\n", encoding="ascii"
                )
            subprocess.run(
                [sys.executable, generator.__file__, "--full_trace", "--output_dir", temporary],
                check=True, capture_output=True, text=True,
            )
            for name, lines in (
                ("training", self.full_training), ("test", self.full_test)
            ):
                self.assertEqual(
                    (directory / f"finite_state_machine_{name}_data.txt").read_text(
                        encoding="ascii"
                    ),
                    "preserve this final-answer dataset\n",
                )
                self.assertEqual(
                    (directory / f"finite_state_machine_full_{name}_data.txt").read_text(
                        encoding="ascii"
                    ),
                    "\n".join(lines) + "\n",
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
