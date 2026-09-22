#!/usr/bin/env python3
"""Extract and reduce a finite, causal integer quotient of captured GPT-2.

The guarantee is the supplied prompt trajectories, including EOS. No sentence
identity or expected continuation participates in a transition lookup. BF16
vectors are exact extraction keys and search coordinates, never runtime math.
"""

from collections import deque
import hashlib
import json
import math
from pathlib import Path
import signal
import struct
import tempfile
import threading
import time


class ModelError(ValueError):
    """Malformed capture, nondeterministic table, or failed continuation."""


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ModelError(f"{name} must be an integer >= {minimum}")
    return value


def _put(table, key, value, name):
    if key in table and table[key] != value:
        raise ModelError(f"conflicting {name} key {key}: {table[key]} vs {value}")
    table[key] = value


def build_model(capture_path, *, expected_samples=None):
    """Read one metadata JSON line followed by independent captured samples."""
    source = Path(capture_path)
    with source.open(encoding="utf-8") as stream:
        try:
            header = json.loads(next(stream))
        except (StopIteration, json.JSONDecodeError) as error:
            raise ModelError("capture must begin with a metadata JSON object") from error
        if header.get("schema") != 1:
            raise ModelError("unsupported capture schema")
        width = _integer(header.get("width"), "width", 1)
        layers = _integer(header.get("layers"), "layers", 1)
        vocab_size = _integer(header.get("vocab_size"), "vocab_size", 1)
        prompt_tokens = _integer(header.get("prompt_tokens"), "prompt_tokens", 1)
        eos_token = _integer(header.get("eos_token"), "eos_token")
        vocabulary = header.get("vocabulary")
        if eos_token >= vocab_size or not isinstance(vocabulary, list) or len(vocabulary) != vocab_size:
            raise ModelError("invalid EOS or vocabulary length")
        original_ids = set()
        for row in vocabulary:
            original = _integer(row.get("original_id"), "original_id")
            if original in original_ids:
                raise ModelError("duplicate original vocabulary ID")
            original_ids.add(original)
            try:
                bytes.fromhex(row["hex"])
            except (KeyError, TypeError, ValueError) as error:
                raise ModelError("invalid vocabulary token bytes") from error
        states, samples = [], []
        intern = [{} for _ in range(2 * layers + 1)]
        entry, snap = {}, {}
        attention = [{} for _ in range(layers)]
        mlp = [{} for _ in range(layers)]
        targets = 0
        for line_number, line in enumerate(stream, 2):
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                raise ModelError(f"invalid JSON at capture line {line_number}") from error
            tokens = sample.get("tokens")
            if not isinstance(tokens, list) or len(tokens) < prompt_tokens:
                raise ModelError(f"invalid token sequence at capture line {line_number}")
            for token in tokens:
                if _integer(token, "token") >= vocab_size:
                    raise ModelError("token outside compact vocabulary")
            predictions = sample.get("predictions")
            if not isinstance(predictions, list) or len(predictions) != len(tokens):
                raise ModelError("one prediction is required per real token")
            for prediction in predictions:
                if _integer(prediction, "prediction") >= vocab_size:
                    raise ModelError("prediction outside compact vocabulary")
            required = tokens[prompt_tokens:] + [eos_token]
            if predictions[prompt_tokens - 1:] != required:
                raise ModelError(f"reference suffix/EOS is incorrect at capture line {line_number}")
            boundaries = sample.get("boundaries")
            if not isinstance(boundaries, list) or len(boundaries) != len(intern):
                raise ModelError("wrong number of captured residual boundaries")
            encoded = []
            for stage, vectors in enumerate(boundaries):
                if not isinstance(vectors, list) or len(vectors) != len(tokens):
                    raise ModelError("one boundary vector is required per real token")
                row_ids = []
                for bits in vectors:
                    if not isinstance(bits, list) or len(bits) != width:
                        raise ModelError("wrong boundary vector width")
                    for word in bits:
                        if _integer(word, "BF16 bits") > 65535 or word & 0x7f80 == 0x7f80:
                            raise ModelError("invalid or nonfinite BF16 boundary vector")
                    key = tuple(bits)
                    state = intern[stage].get(key)
                    if state is None:
                        state = vocab_size + len(states)
                        intern[stage][key] = state
                        states.append({"id": state, "stage": stage, "bits": bits})
                    row_ids.append(state)
                encoded.append(row_ids)
            for position, token in enumerate(tokens):
                _put(entry, (token, position), encoded[0][position], "entry")
                for layer in range(layers):
                    _put(attention[layer], tuple(encoded[2 * layer][:position + 1]),
                         encoded[2 * layer + 1][position], "attention")
                    _put(mlp[layer], encoded[2 * layer + 1][position],
                         encoded[2 * layer + 2][position], "MLP")
                if position >= prompt_tokens - 1:
                    _put(snap, encoded[-1][position], predictions[position], "snap")
            targets += len(required)
            samples.append({"tokens": tokens})
    if not samples or (expected_samples is not None and len(samples) != expected_samples):
        raise ModelError(f"unexpected sample count: {len(samples)}")
    model = {key: header[key] for key in
             ("schema", "width", "layers", "vocab_size", "eos_token", "prompt_tokens", "vocabulary")}
    model.update(states=states, samples=samples,
                 entry=[[token, position, out] for (token, position), out in sorted(entry.items())],
                 attention=[[[list(key), out] for key, out in sorted(table.items())] for table in attention],
                 mlp=[[[key, out] for key, out in sorted(table.items())] for table in mlp],
                 snap=[[key, out] for key, out in sorted(snap.items())],
                 stats={"captured_samples": len(samples), "scored_targets": targets,
                        "exact_states": len(states), "capture_sha256": _file_hash(source)})
    model["stats"]["verification"] = evaluate_model(model)
    return model


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class IntegerModel:
    """Compiled dictionaries for integer-only causal inference."""

    def __init__(self, model):
        self.model = model
        self.entry = {}
        self.attention = [{} for _ in range(model["layers"])]
        self.mlp = [{} for _ in range(model["layers"])]
        self.snap = {}
        for token, position, state in model["entry"]:
            _put(self.entry, (token, position), state, "entry")
        for layer in range(model["layers"]):
            for prefix, output in model["attention"][layer]:
                _put(self.attention[layer], tuple(prefix), output, "attention")
            for source, output in model["mlp"][layer]:
                _put(self.mlp[layer], source, output, "MLP")
        for state, token in model["snap"]:
            _put(self.snap, state, token, "snap")

    def predict(self, tokens):
        if not tokens:
            raise ModelError("cannot predict from an empty prefix")
        try:
            current = [self.entry[token, position] for position, token in enumerate(tokens)]
            for attention, mlp in zip(self.attention, self.mlp):
                after_attention = [attention[tuple(current[:position + 1])]
                                   for position in range(len(current))]
                current = [mlp[state] for state in after_attention]
            return self.snap[current[-1]]
        except KeyError as error:
            raise ModelError(f"undefined discrete transition for prefix {list(tokens)}: {error}") from error


def evaluate_model(model):
    """Feed back predictions and require all expected tokens followed by EOS."""
    runtime = IntegerModel(model)
    targets = 0
    for number, sample in enumerate(model["samples"], 1):
        expected = sample["tokens"]
        prefix = list(expected[:model["prompt_tokens"]])
        for token in expected[model["prompt_tokens"]:] + [model["eos_token"]]:
            predicted = runtime.predict(prefix)
            targets += 1
            if predicted != token:
                raise ModelError(f"sample {number}, token {len(prefix)}: expected {token}, got {predicted}")
            if token != model["eos_token"]:
                prefix.append(predicted)
    return {"samples": len(model["samples"]), "targets": targets,
            "errors": 0, "explicit_eos": len(model["samples"])}


def save_model(model, path):
    """Atomically replace a JSON artifact; the previous file survives errors."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix="." + destination.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(model, stream, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_model(path, *, verify=True):
    with Path(path).open(encoding="utf-8") as stream:
        model = json.load(stream)
    if model.get("schema") != 1:
        raise ModelError("unsupported model schema")
    if verify:
        evaluate_model(model)
    return model


def restore_membership(model, original_model):
    """Recover exact original-state membership from a quotient's integer tables.

    This also upgrades older checkpoints that did not save membership. It checks
    the homomorphism at every boundary and cannot infer memberships merely from
    the surviving representative vectors. Neither input dictionary is modified.
    """
    for key in ("schema", "width", "layers", "vocab_size", "eos_token", "prompt_tokens", "vocabulary"):
        if model[key] != original_model[key]:
            raise ModelError(f"membership source disagrees on {key}")
    quotient = IntegerModel(model)
    mapping = {}
    try:
        for token, position, state in original_model["entry"]:
            _put(mapping, state, quotient.entry[token, position], "state membership")
        for layer in range(model["layers"]):
            for inputs, output in original_model["attention"][layer]:
                transformed = tuple(mapping[state] for state in inputs)
                _put(mapping, output, quotient.attention[layer][transformed], "state membership")
            for source, output in original_model["mlp"][layer]:
                _put(mapping, output, quotient.mlp[layer][mapping[source]], "state membership")
        for state, token in original_model["snap"]:
            if quotient.snap[mapping[state]] != token:
                raise ModelError("membership source disagrees on a required token label")
    except KeyError as error:
        raise ModelError(f"model is not a complete quotient of the membership source: {error}") from error
    members = {row["id"]: [] for row in model["states"]}
    stages = {row["id"]: row["stage"] for row in model["states"]}
    for row in original_model["states"]:
        target = mapping.get(row["id"])
        if target not in members or row["stage"] != stages[target]:
            raise ModelError("membership state is missing or crosses a boundary")
        members[target].extend(row.get("members", [row["id"]]))
    if any(not group for group in members.values()):
        raise ModelError("quotient contains a state without an original member")
    result = dict(model)
    result["states"] = [dict(row, members=sorted(members[row["id"]]),
                             member_count=len(members[row["id"]])) for row in model["states"]]
    result["stats"] = dict(model.get("stats", {}), membership_complete=True,
                           membership_original_states=sum(len(group) for group in members.values()))
    return result


class QuotientReducer:
    """Incremental congruence closure with complete rollback of rejected seeds.

    Terms are immutable. A root indexes the terms mentioning any of its member
    states. Only terms affected by a union need to be rehashed. Distinct fixed
    token labels are an immediate contradiction; tokens themselves never merge.
    """

    def __init__(self, model):
        self.model = model
        self.states = model["states"]
        self.index = {row["id"]: i for i, row in enumerate(self.states)}
        if len(self.index) != len(self.states):
            raise ModelError("duplicate state IDs")
        n = len(self.states)
        self.parent = list(range(n))
        self.size = [1] * n
        self.label = [-1] * n
        self.stage = [row["stage"] for row in self.states]
        self.uses = [set() for _ in range(n)]
        self.terms = []
        self.signatures = {}
        self.term_signatures = []
        self.attempted = 0
        self.accepted = 0
        self.unions = 0
        self.rejected_pairs = set()
        self.cached_rejections = 0
        self.accepted_merges = []
        for layer in range(model["layers"]):
            for inputs, output in model["attention"][layer]:
                self._add_term(2 * layer + 1, inputs, output)
            for source, output in model["mlp"][layer]:
                self._add_term(2 * layer + 2, [source], output)
        for state, token in model["snap"]:
            index = self.index[state]
            if self.label[index] not in (-1, token):
                raise ModelError("conflicting snap labels")
            self.label[index] = token

    def _add_term(self, stage, inputs, output):
        args = tuple(self.index[state] for state in inputs)
        out = self.index[output]
        if not args or self.stage[out] != stage or any(self.stage[arg] != stage - 1 for arg in args):
            raise ModelError("transition crosses incorrect state boundaries")
        signature = (stage,) + args
        if signature in self.signatures:
            other = self.terms[self.signatures[signature]][2]
            if other != out:
                raise ModelError("conflicting original transition")
            return
        term = len(self.terms)
        self.terms.append((stage, args, out))
        self.term_signatures.append(signature)
        self.signatures[signature] = term
        for arg in set(args):
            self.uses[arg].add(term)

    def find(self, index):
        # No path compression: union by size gives log depth and cheap rollback.
        while self.parent[index] != index:
            index = self.parent[index]
        return index

    def roots(self, stage=None):
        return [i for i in range(len(self.states))
                if self.parent[i] == i and (stage is None or self.stage[i] == stage)]

    def terminal_label(self, root):
        """Necessary label constraint through the final pointwise MLP only."""
        if self.stage[root] == 2 * self.model["layers"]:
            return self.label[root]
        if self.stage[root] == 2 * self.model["layers"] - 1:
            for term in self.uses[root]:
                label = self.label[self.find(self.terms[term][2])]
                if label >= 0:
                    return label
        return -1

    def try_merge(self, first_id, second_id):
        """Commit a sound quotient or return False with every table restored."""
        first, second = self.index[first_id], self.index[second_id]
        if self.stage[first] != self.stage[second]:
            raise ModelError("states from different boundaries cannot merge")
        first, second = self.find(first), self.find(second)
        if first == second:
            return True
        pair = (min(first, second), max(first, second))
        if pair in self.rejected_pairs:
            self.cached_rejections += 1
            return False
        seed_stage = self.stage[first]
        seed_ids = [self.states[first]["id"], self.states[second]["id"]]
        seed_distance = math.sqrt(_distance(_vector(self.states[first]["bits"]),
                                            _vector(self.states[second]["bits"])))
        self.attempted += 1
        pending = deque([(first, second)])
        undo = []
        count = 0
        while pending:
            first, second = (self.find(index) for index in pending.popleft())
            if first == second:
                continue
            first_label, second_label = self.terminal_label(first), self.terminal_label(second)
            if first_label >= 0 and second_label >= 0 and first_label != second_label:
                self._rollback(undo)
                self.rejected_pairs.add(pair)
                return False
            if (self.size[first], -first) < (self.size[second], -second):
                first, second = second, first
            affected = sorted(self.uses[second])
            for term in affected:
                signature = self.term_signatures[term]
                if self.signatures.get(signature) == term:
                    undo.append(("dictionary", signature, term))
                    del self.signatures[signature]
            undo.append(("union", first, second, self.size[first], self.label[first]))
            self.parent[second] = first
            self.size[first] += self.size[second]
            if self.label[first] < 0:
                self.label[first] = self.label[second]
            added = self.uses[second] - self.uses[first]
            self.uses[first].update(added)
            undo.append(("uses", first, added))
            count += 1
            for term in affected:
                stage, args, out = self.terms[term]
                signature = (stage,) + tuple(self.find(arg) for arg in args)
                undo.append(("term", term, self.term_signatures[term]))
                self.term_signatures[term] = signature
                other = self.signatures.get(signature)
                if other is None:
                    undo.append(("dictionary", signature, None))
                    self.signatures[signature] = term
                else:
                    # Follow one implication toward terminal labels promptly.
                    # Breadth-first closure can expand thousands of doomed
                    # intermediate unions before reaching the first conflict.
                    pending.appendleft((out, self.terms[other][2]))
        if count:
            self.accepted += 1
            self.unions += count
            self.accepted_merges.append({"stage": seed_stage, "seed_ids": seed_ids,
                                         "euclidean_distance": seed_distance,
                                         "induced_unions": count - 1})
        return True

    def _rollback(self, undo):
        for operation in reversed(undo):
            if operation[0] == "dictionary":
                _, key, previous = operation
                if previous is None:
                    self.signatures.pop(key, None)
                else:
                    self.signatures[key] = previous
            elif operation[0] == "term":
                _, term, previous = operation
                self.term_signatures[term] = previous
            elif operation[0] == "uses":
                _, root, added = operation
                self.uses[root].difference_update(added)
            else:
                _, first, second, size, label = operation
                self.parent[second] = second
                self.size[first] = size
                self.label[first] = label

    def export(self):
        """Emit deduplicated quotient tables with dense, global internal IDs."""
        roots = sorted(self.roots(), key=lambda root: (self.stage[root], self.states[root]["id"]))
        renumber = {root: self.model["vocab_size"] + i for i, root in enumerate(roots)}
        def state_id(old):
            return renumber[self.find(self.index[old])]
        result = {key: self.model[key] for key in
                  ("schema", "width", "layers", "vocab_size", "eos_token", "prompt_tokens", "vocabulary", "samples")}
        result["states"] = [{"id": renumber[root], "stage": self.stage[root],
                              "bits": self.states[root]["bits"]} for root in roots]
        # Old reduced checkpoints need restore_membership once; treating their
        # newly numbered representatives as original IDs would invent provenance.
        complete_membership = (not self.model.get("stats", {}).get("state_unions", 0)
                               or all("members" in row for row in self.states))
        if complete_membership:
            members = {root: [] for root in roots}
            for index, row in enumerate(self.states):
                members[self.find(index)].extend(row.get("members", [row["id"]]))
            for row, root in zip(result["states"], roots):
                row["members"] = sorted(members[root])
                row["member_count"] = len(members[root])
        result["entry"] = [[token, position, state_id(out)] for token, position, out in self.model["entry"]]
        result["attention"], result["mlp"] = [], []
        for layer in range(self.model["layers"]):
            attn, mlp = {}, {}
            for inputs, output in self.model["attention"][layer]:
                _put(attn, tuple(state_id(value) for value in inputs), state_id(output), "quotient attention")
            for source, output in self.model["mlp"][layer]:
                _put(mlp, state_id(source), state_id(output), "quotient MLP")
            result["attention"].append([[list(key), output] for key, output in sorted(attn.items())])
            result["mlp"].append([[key, output] for key, output in sorted(mlp.items())])
        snap = {}
        for state, token in self.model["snap"]:
            _put(snap, state_id(state), token, "quotient snap")
        result["snap"] = [[key, output] for key, output in sorted(snap.items())]
        result["stats"] = dict(self.model.get("stats", {}))
        previous = self.model.get("stats", {})
        result["stats"].update(states=len(roots),
            membership_complete=complete_membership,
            membership_original_states=(sum(row["member_count"] for row in result["states"])
                                        if complete_membership else None),
            states_per_stage=[sum(self.stage[root] == stage for root in roots)
                              for stage in range(2 * self.model["layers"] + 1)],
            attempted_seeds=previous.get("attempted_seeds", 0) + self.attempted,
            accepted_seeds=previous.get("accepted_seeds", 0) + self.accepted,
            state_unions=previous.get("state_unions", 0) + self.unions,
            cached_rejections=previous.get("cached_rejections", 0) + self.cached_rejections,
            accepted_merges=previous.get("accepted_merges", []) + self.accepted_merges)
        return result

    def save_checkpoint(self, path, *, search=None):
        """Persist a portable quotient; resume requires no original GPU capture."""
        result = self.export()
        result["stats"]["verification"] = evaluate_model(result)
        if search is not None:
            result["stats"]["search"] = search
        save_model(result, path)


def _vector(bits):
    return tuple(struct.unpack("!f", struct.pack("!I", word << 16))[0] for word in bits)


def _distance(first, second):
    return math.fsum((a - b) ** 2 for a, b in zip(first, second))


def _candidate_pairs(reducer, stage, neighbors, *, exhaustive=False):
    """Yield distance-sorted pairs; scipy accelerates true nearest neighbors.

    Without scipy, coordinate-sorted windows provide an explicitly approximate
    shortlist. All-pair mode is exact and intended for small remaining stages.
    """
    roots = reducer.roots(stage)
    coordinates = [_vector(reducer.states[root]["bits"]) for root in roots]
    if len(roots) < 2:
        return []
    pairs = set()
    labels = [reducer.terminal_label(root) for root in roots]
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    if exhaustive:
        for label, group in groups.items():
            pairs.update((min(i, j), max(i, j)) for rank, i in enumerate(group) for j in group[rank + 1:])
        for first in groups.get(-1, []):
            for second in range(len(roots)):
                if labels[second] != -1:
                    pairs.add((min(first, second), max(first, second)))
    else:
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            cKDTree = None
        for label, group in groups.items():
            # Known different terminal labels can never merge. Searching within
            # a compatible label group is exact pruning, not a heuristic loss.
            pool = list(range(len(roots))) if label == -1 else group
            if len(pool) < 2:
                continue
            if cKDTree is None:
                dimensions = min(4, reducer.model["width"])
                selected = set(group)
                for dimension in range(dimensions):
                    order = sorted(pool, key=lambda i: (coordinates[i][dimension], roots[i]))
                    for rank, first in enumerate(order):
                        if first not in selected:
                            continue
                        for second in order[max(0, rank - neighbors):rank + 1 + neighbors]:
                            if first != second:
                                pairs.add((min(first, second), max(first, second)))
            else:
                tree = cKDTree([coordinates[i] for i in pool])
                k = min(neighbors + 1, len(pool))
                _, indices = tree.query([coordinates[i] for i in group], k=k, workers=1)
                for first, row in zip(group, indices):
                    for index in row:
                        second = pool[int(index)]
                        if first != second:
                            pairs.add((min(first, second), max(first, second)))
    result = [(_distance(coordinates[first], coordinates[second]),
               reducer.states[roots[first]]["id"], reducer.states[roots[second]]["id"])
              for first, second in pairs]
    result.sort()
    return result


def _eligible_pair_count(reducer):
    total = 0
    for stage in range(2 * reducer.model["layers"] + 1):
        counts = {}
        for root in reducer.roots(stage):
            label = reducer.terminal_label(root)
            counts[label] = counts.get(label, 0) + 1
        unknown = counts.pop(-1, 0)
        total += unknown * (unknown - 1) // 2 + unknown * sum(counts.values())
        total += sum(count * (count - 1) // 2 for count in counts.values())
    return total


class _CooperativeInterrupt:
    """Defer SIGINT until a trial union has committed or completely rolled back."""

    def __init__(self):
        self.requested = False
        self.checkpoint = None
        self.previous = None
        self.installed = False

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            self.previous = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._request)
            self.installed = True
        return self

    def _request(self, signum, frame):
        self.requested = True

    def check(self):
        if self.requested:
            self.requested = False
            if self.checkpoint is not None:
                self.checkpoint()
            raise KeyboardInterrupt("reduction interrupted at a safe checkpoint boundary")

    def __exit__(self, exc_type, exc_value, traceback):
        if self.installed:
            signal.signal(signal.SIGINT, self.previous)


def reduce_model(model, *, neighbors=4, max_passes=10, max_attempts=None,
                 exhaustive_pair_limit=100_000, checkpoint_path=None,
                 checkpoint_seconds=60, progress=None):
    """Reduce a model; SIGINT atomically saves a verified checkpoint if supplied.

    Distances order candidate attempts but never restrict the final exhaustive
    sweep. Resume by loading the saved model and calling this function again.
    """
    with _CooperativeInterrupt() as interrupt:
        result = _reduce_model(model, neighbors=neighbors, max_passes=max_passes,
            max_attempts=max_attempts, exhaustive_pair_limit=exhaustive_pair_limit,
            checkpoint_path=checkpoint_path, checkpoint_seconds=checkpoint_seconds,
            progress=progress, interrupt=interrupt)
        interrupt.check()
        return result


def _reduce_model(model, *, neighbors, max_passes, max_attempts,
                  exhaustive_pair_limit, checkpoint_path,
                  checkpoint_seconds, progress, interrupt):
    """Reduce nearby states and honestly report the achieved stopping condition.

    A saved checkpoint is a complete verified model and can be passed back here
    to resume. Search restarts from its surviving representative vectors. The
    final all-pair sweep runs only when its pair count fits the explicit limit.
    """
    if neighbors < 1 or max_passes < 1 or exhaustive_pair_limit < 0:
        raise ValueError("invalid reduction search limits")
    reducer = QuotientReducer(model)
    start = last_checkpoint = last_report_time = time.monotonic()
    last_report_attempt = 0
    history = []
    if checkpoint_path is not None:
        interrupt.checkpoint = lambda: reducer.save_checkpoint(checkpoint_path,
            search={"history": history, "interrupted": True,
                    "pairwise_irreducible": False, "global_minimum_proven": False})
    stop = "pass_limit"
    pairwise_irreducible = False
    def report(phase, pass_number):
        nonlocal last_checkpoint, last_report_attempt, last_report_time
        interrupt.check()
        roots = reducer.roots()
        counts = [0] * (2 * model["layers"] + 1)
        for root in roots:
            counts[reducer.stage[root]] += 1
        info = {"phase": phase, "pass": pass_number, "states": len(roots),
                "states_per_stage": counts,
                "attempted": reducer.attempted, "accepted": reducer.accepted,
                "unions": reducer.unions, "seconds": time.monotonic() - start}
        if progress is not None:
            progress(info)
        if checkpoint_path is not None and time.monotonic() - last_checkpoint >= checkpoint_seconds:
            reducer.save_checkpoint(checkpoint_path, search={"history": history, "current": info})
            last_checkpoint = time.monotonic()
        last_report_attempt = reducer.attempted
        last_report_time = time.monotonic()
        return info
    def report_due():
        now = time.monotonic()
        return (reducer.attempted >= last_report_attempt + 1000
                or now - last_report_time >= 10
                or (checkpoint_path is not None and now - last_checkpoint >= checkpoint_seconds))
    for pass_number in range(1, max_passes + 1):
        previous = reducer.unions
        report("nearest", pass_number)
        exhausted_budget = False
        for stage in range(2 * model["layers"] + 1):
            for _, first, second in _candidate_pairs(reducer, stage, neighbors):
                interrupt.check()
                if max_attempts is not None and reducer.attempted >= max_attempts:
                    exhausted_budget = True
                    break
                if reducer.find(reducer.index[first]) != reducer.find(reducer.index[second]):
                    reducer.try_merge(first, second)
                interrupt.check()
                if report_due():
                    report("nearest", pass_number)
            report("nearest", pass_number)
            if exhausted_budget:
                break
        history.append(report("nearest_pass_complete", pass_number))
        if exhausted_budget:
            stop = "attempt_limit"
            break
        if reducer.unions == previous:
            stop = "nearest_candidates_exhausted"
            break
    remaining_pairs = _eligible_pair_count(reducer)
    if remaining_pairs <= exhaustive_pair_limit and stop != "attempt_limit":
        sweep = 0
        while True:
            sweep += 1
            previous = reducer.unions
            exhausted_budget = False
            for stage in range(2 * model["layers"] + 1):
                for _, first, second in _candidate_pairs(reducer, stage, neighbors, exhaustive=True):
                    interrupt.check()
                    if max_attempts is not None and reducer.attempted >= max_attempts:
                        exhausted_budget = True
                        break
                    if reducer.find(reducer.index[first]) != reducer.find(reducer.index[second]):
                        reducer.try_merge(first, second)
                    interrupt.check()
                    if report_due():
                        report("exhaustive", sweep)
                report("exhaustive", sweep)
                if exhausted_budget:
                    break
            history.append(report("exhaustive_pass_complete", sweep))
            if exhausted_budget:
                stop = "attempt_limit"
                break
            if reducer.unions == previous:
                pairwise_irreducible = True
                stop = "no_compatible_pair"
                break
    result = reducer.export()
    result["stats"]["verification"] = evaluate_model(result)
    result["stats"]["search"] = {"stopping_reason": stop,
        "pairwise_irreducible": pairwise_irreducible, "global_minimum_proven": False,
        "nearest_neighbors": neighbors, "exhaustive_pair_limit": exhaustive_pair_limit,
        "remaining_pairs_before_sweep": remaining_pairs, "history": history,
        "seconds": time.monotonic() - start}
    if checkpoint_path is not None:
        save_model(result, checkpoint_path)
    return result
