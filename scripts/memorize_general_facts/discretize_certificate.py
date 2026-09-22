#!/usr/bin/env python3
"""Independently certify pairwise irreducibility of immutable integer tables.

This checker imports no reducer code and trusts no search history or rejection
cache. It works backward: distinct fixed vocabulary outputs cannot merge;
injective MLP tables inherit that distinction; identifying two attention-input
symbols is forbidden when it collides on two already-distinguishable outputs.

A proof covers every same-boundary pair in the supplied quotient. It does not
prove a global minimum across alternative partitions, native-checkpoint
equivalence, or corpus accuracy; those require their separate checks. If this
sufficient backward argument cannot be completed, the result is inconclusive.

CLI exit codes: 0 proven, 2 inconclusive, 1 malformed input or I/O error.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path


class CertificateError(ValueError):
    """Malformed state or transition shape, rather than an incomplete proof."""


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise CertificateError(f"{name} must be an integer >= {minimum}")
    return value


def _list(value, name, length=None):
    if not isinstance(value, list) or (length is not None and len(value) != length):
        suffix = "" if length is None else f" of length {length}"
        raise CertificateError(f"{name} must be a list{suffix}")
    return value


def _validate(model):
    if not isinstance(model, dict) or type(model.get("schema")) is not int or model["schema"] != 1:
        raise CertificateError("expected model schema 1")
    width = _integer(model.get("width"), "width", 1)
    layers = _integer(model.get("layers"), "layers")
    vocabulary = _integer(model.get("vocab_size"), "vocab_size", 1)
    by_stage = [[] for _ in range(2 * layers + 1)]
    state_stages = {}
    for row in _list(model.get("states"), "states"):
        if not isinstance(row, dict):
            raise CertificateError("state must be an object")
        state = _integer(row.get("id"), "state ID", vocabulary)
        stage = _integer(row.get("stage"), "state stage")
        if stage >= len(by_stage) or state in state_stages:
            raise CertificateError("duplicate state ID or invalid boundary")
        for word in _list(row.get("bits"), "state bits", width):
            if _integer(word, "BF16 bits") > 65535:
                raise CertificateError("BF16 bits exceed uint16")
        state_stages[state] = stage
        by_stage[stage].append(state)
    if any(not states for states in by_stage):
        raise CertificateError("every boundary must contain at least one state")
    for states in by_stage:
        states.sort()

    def state_at(value, stage):
        _integer(value, "referenced state", vocabulary)
        if state_stages.get(value) != stage:
            raise CertificateError(f"state {value} does not belong to boundary {stage}")

    entry_keys = set()
    for row in _list(model.get("entry"), "entry"):
        token, position, output = _list(row, "entry row", 3)
        if _integer(token, "entry token") >= vocabulary:
            raise CertificateError("entry token is outside vocabulary")
        _integer(position, "entry position")
        state_at(output, 0)
        if (token, position) in entry_keys:
            raise CertificateError("duplicate entry key")
        entry_keys.add((token, position))
    if not entry_keys:
        raise CertificateError("entry table must not be empty")

    attention, mlp = [], []
    attention_rows = _list(model.get("attention"), "attention", layers)
    mlp_rows = _list(model.get("mlp"), "MLP", layers)
    for layer in range(layers):
        table = {}
        for row in _list(attention_rows[layer], "attention table"):
            prefix, output = _list(row, "attention row", 2)
            _list(prefix, "attention prefix")
            if not prefix:
                raise CertificateError("attention prefix must not be empty")
            for state in prefix:
                state_at(state, 2 * layer)
            state_at(output, 2 * layer + 1)
            key = tuple(prefix)
            if key in table:
                raise CertificateError("duplicate attention key")
            table[key] = output
        attention.append(table)
        table = {}
        for row in _list(mlp_rows[layer], "MLP table"):
            source, output = _list(row, "MLP row", 2)
            state_at(source, 2 * layer + 1)
            state_at(output, 2 * layer + 2)
            if source in table:
                raise CertificateError("duplicate MLP key")
            table[source] = output
        mlp.append(table)

    snap = {}
    for row in _list(model.get("snap"), "snap"):
        state, token = _list(row, "snap row", 2)
        state_at(state, 2 * layers)
        if _integer(token, "snap token") >= vocabulary:
            raise CertificateError("snap token is outside vocabulary")
        if state in snap:
            raise CertificateError("duplicate snap key")
        snap[state] = token
    return by_stage, attention, mlp, snap


def _duplicate_output(table):
    seen = {}
    for state, output in table.items():
        if output in seen:
            return [seen[output], state]
        seen[output] = state
    return None


def _attention_pair_witnesses(states, original):
    """Yield (first state, second state, has contradictory collision).

    Byte encoding makes normalization fast for small alphabets. Tuples provide
    the identical argument for larger alphabets without a size restriction.
    Only keys containing the first symbol can change. Their normalized forms
    are compared with the original table and with other rewritten keys; every
    original key containing the first symbol is automatically unreachable by
    such a normalized lookup, which contains only the second symbol.
    """
    numbers = {state: index for index, state in enumerate(states)}
    use_bytes = len(states) <= 256
    encode = bytes if use_bytes else tuple
    table = {encode(numbers[state] for state in key): output
             for key, output in original.items()}
    uses = [[] for _ in states]
    for key, output in table.items():
        for state in set(key):
            uses[state].append((key, output))
    for first, second in itertools.combinations(range(len(states)), 2):
        translation = bytes.maketrans(bytes([first]), bytes([second])) if use_bytes else None
        rewritten = {}
        witnessed = False
        for key, output in uses[first]:
            normalized = (key.translate(translation) if use_bytes else
                          tuple(second if value == first else value for value in key))
            other = rewritten.get(normalized, table.get(normalized))
            if other is not None and other != output:
                witnessed = True
                break
            rewritten[normalized] = output
        yield states[first], states[second], witnessed


def certify_model(model):
    """Return a proof report or an honest inconclusive result; never mutate input.

    Raises CertificateError for malformed shapes. A missing readout/MLP row,
    repeated output label, or attention pair without a collision leaves this
    backward proof incomplete. No search metadata is read.
    """
    by_stage, attention, mlp, snap = _validate(model)
    pair_count = lambda states: len(states) * (len(states) - 1) // 2
    report = {
        "certificate_schema": 1,
        "status": "inconclusive",
        "method": "independent_backward_table_collisions",
        "states": sum(map(len, by_stage)),
        "boundaries": len(by_stage),
        "same_boundary_pairs": sum(map(pair_count, by_stage)),
        "proven_pairs": 0,
        "attention_pairs_checked": 0,
        "pairwise_irreducible_proven": False,
        "global_minimum_proven": False,
        "stages": [],
    }

    def inconclusive(reason, stage, pair=None):
        report.update(reason=reason, unresolved_stage=stage)
        if pair is not None:
            report["unresolved_pair"] = pair
        return report

    def proven(stage, argument):
        count = pair_count(by_stage[stage])
        report["proven_pairs"] += count
        report["stages"].append({"stage": stage, "states": len(by_stage[stage]),
                                  "pairs": count, "argument": argument})

    final_stage = len(by_stage) - 1
    if set(snap) != set(by_stage[final_stage]):
        return inconclusive("not every final state has a fixed readout label", final_stage)
    duplicate = _duplicate_output(snap)
    if duplicate is not None:
        return inconclusive("two final states have the same fixed token label", final_stage, duplicate)
    proven(final_stage, "distinct fixed token labels")

    for layer in reversed(range(len(attention))):
        stage = 2 * layer + 1
        if set(mlp[layer]) != set(by_stage[stage]):
            return inconclusive("MLP table does not cover every input state", stage)
        duplicate = _duplicate_output(mlp[layer])
        if duplicate is not None:
            return inconclusive("MLP is not injective into distinguishable outputs", stage, duplicate)
        proven(stage, "injective MLP into already-distinguishable downstream states")
        stage -= 1
        for first, second, witnessed in _attention_pair_witnesses(by_stage[stage], attention[layer]):
            report["attention_pairs_checked"] += 1
            if not witnessed:
                return inconclusive("state identification has no contradictory attention-key collision",
                                    stage, [first, second])
        proven(stage, "every pair collides on already-distinguishable attention outputs")
    report.update(status="proven", pairwise_irreducible_proven=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path,
                        help="Saved immutable integer-model JSON")
    args = parser.parse_args(argv)
    try:
        encoded = args.model.read_bytes()
        model = json.loads(encoded)
        certificate = certify_model(model)
        certificate["model_sha256"] = hashlib.sha256(encoded).hexdigest()
        print(json.dumps(certificate, indent=2, sort_keys=True))
        return 0 if certificate["status"] == "proven" else 2
    except (OSError, ValueError) as error:
        print(json.dumps({"certificate_schema": 1, "status": "error",
                          "reason": str(error), "pairwise_irreducible_proven": False,
                          "global_minimum_proven": False}, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
