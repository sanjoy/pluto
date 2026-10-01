#!/usr/bin/env python3
"""Generate reproducible finite-state-machine interpretation datasets.

Format: transition[;transition...];input>output, without headers.
Transitions have three source digits, one A-Z letter, and three destination
digits. Execution starts at 000; output is the final state or ERR at the first
missing edge. With --full_trace, output lists 000 followed by each visited state,
separated by spaces for readability, ending in ERR on the first missing edge.
There are no accepting states: every completed walk is valid.

Machines have 4-16 states. Each supplies one successful walk of 1-24 letters and
one same-length input with one letter changed to force failure. Transition and
sentence order are shuffled. Train/test machines are disjoint, but share the
alphabet and state-label range. All 26 letters occur somewhere in each machine,
preventing globally absent symbols from giving away failures. This is an
in-distribution test, not extrapolation to longer inputs or larger machines.

Run from the repository root:
  python3 scripts/generate_finite_state_machine_data.py
  python3 scripts/generate_finite_state_machine_data.py --full_trace
  python3 -m unittest discover -s scripts -p 'generate_finite_state_machine_data_test.py'
"""

import argparse
from pathlib import Path
import random
import string

ALPHABET = string.ascii_uppercase
DEFAULT_SEED = 20260930
TransitionMap = dict[tuple[int, str], int]


def execution_trace(transitions: TransitionMap, input_text: str) -> list[str]:
    """Return 000 and every visited state, stopping at the first missing edge."""
    state = 0
    trace = ["000"]
    for letter in input_text:
        destination = transitions.get((state, letter))
        if destination is None:
            trace.append("ERR")
            break
        state = destination
        trace.append(f"{state:03d}")
    return trace


def evaluate(transitions: TransitionMap, input_text: str) -> str:
    """Follow transitions from 000 until failure or end of input."""
    return execution_trace(transitions, input_text)[-1]


def make_machine(rng: random.Random) -> TransitionMap:
    """Make a partial DFA with random labels, cycles, branches and self-loops.

    A backbone cycle makes every state reachable from 000. Extra edges provide
    distractors. This deliberately structured sampling is not uniform over DFAs.
    """
    states = [0] + rng.sample(range(1, 1000), rng.randint(4, 16) - 1)
    transitions = {}
    for index, state in enumerate(states):
        letters = rng.sample(ALPHABET, rng.randint(3, 8))
        transitions[state, letters[0]] = states[(index + 1) % len(states)]
        for letter in letters[1:]:
            transitions[state, letter] = rng.choice(states)

    # Failures must depend on the current state, not globally absent letters.
    # Keep at least one missing letter per state for the paired ERR input.
    used = {letter for _, letter in transitions}
    for letter in ALPHABET:
        if letter not in used:
            candidates = [
                state
                for state in states
                if sum(source == state for source, _ in transitions) < 25
            ]
            transitions[rng.choice(candidates), letter] = rng.choice(states)
    return transitions


def make_pair(
    rng: random.Random, transitions: TransitionMap, full_trace: bool = False
) -> tuple[str, str]:
    """Pair a valid walk with one mutation at a uniformly drawn input position."""
    state = 0
    visited, letters = [], []
    for _ in range(rng.randint(1, 24)):
        visited.append(state)
        outgoing = sorted(c for source, c in transitions if source == state)
        letter = rng.choice(outgoing)
        letters.append(letter)
        state = transitions[state, letter]
    failed_letters = letters.copy()
    index = rng.randrange(len(letters))
    missing = [c for c in ALPHABET if (visited[index], c) not in transitions]
    failed_letters[index] = rng.choice(missing)

    edges = [f"{s:03d}{c}{d:03d}" for (s, c), d in transitions.items()]
    rng.shuffle(edges)
    description = ";".join(edges)
    input_text, failed_input = "".join(letters), "".join(failed_letters)
    assert evaluate(transitions, input_text) == f"{state:03d}"
    assert evaluate(transitions, failed_input) == "ERR"
    if full_trace:
        answer = " ".join(execution_trace(transitions, input_text))
        failed_answer = " ".join(execution_trace(transitions, failed_input))
    else:
        answer, failed_answer = f"{state:03d}", "ERR"
    return (
        f"{description};{input_text}>{answer}",
        f"{description};{failed_input}>{failed_answer}",
    )


def generate(
    seed: int = DEFAULT_SEED, full_trace: bool = False
) -> tuple[list[str], list[str]]:
    """Return 4096 training and 128 test sentences with disjoint machines.

    Full traces do not consume randomness: a given seed produces exactly the
    same machines, inputs and sentence order in either output format.
    """
    rng = random.Random(seed)
    seen = set()

    def split(size: int) -> list[str]:
        lines = []
        while len(lines) < size:
            transitions = make_machine(rng)
            signature = tuple(sorted(transitions.items()))
            if signature in seen:
                continue
            seen.add(signature)
            lines.extend(make_pair(rng, transitions, full_trace))
        rng.shuffle(lines)
        return lines

    return split(4096), split(128)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--full_trace", action="store_true",
        help="Write full_ files with every visited state; leave final-answer files alone.",
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "testdata",
    )
    args = parser.parse_args()
    training, test = generate(args.seed, args.full_trace)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, lines in (("training", training), ("test", test)):
        prefix = "full_" if args.full_trace else ""
        path = args.output_dir / f"finite_state_machine_{prefix}{name}_data.txt"
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        print(f"{path}: {len(lines)} sentences, {len(lines) // 2} machines")


if __name__ == "__main__":
    main()
