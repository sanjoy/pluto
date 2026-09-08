"""Verification-only bigram-preserving controls for frozen static graph paths.

A sequence assembled from token-pair scores may match prose simply because its
pairs are ordinary language. A global token-label permutation is too weak a
control for that: it destroys even common spaces, punctuation, and BPE pairs.
Here we randomize the CORPUS while retaining every directed adjacent pair,
including its multiplicity, and both endpoint tokens. No training text is used
to generate, rank, or change candidates. This module is not a decompressor.

The shuffle is a randomized Euler trail of the corpus's directed multigraph.
It is not uniform over all trails and does not yield calibrated p-values.
Small or rigid graphs can admit no different sequence at all. Keeping bigrams
also keeps some real learned information, so this is specifically a diagnostic
of evidence for order BEYOND token pairs, not a test for all forms of learning.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .checkpoint import sha256_file
from .verify import CorpusIndex, validate_candidates


LENGTHS = (3, 4, 5, 8)


def checked_tokens(tokens, vocab_size):
    tokens = np.asarray(tokens)
    if type(vocab_size) is not int or not 0 < vocab_size <= 2**31:
        raise ValueError("vocab_size must be a positive int32 vocabulary size")
    if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError("tokens must be a one-dimensional integer array")
    if len(tokens) and (tokens.min() < 0 or tokens.max() >= vocab_size):
        raise ValueError("token ID outside vocabulary")
    return tokens.astype(np.int32, copy=False)


def bigram_digest(tokens):
    """Hash a canonical multiset, preserving directed edge multiplicities.

    Unlike a sum/XOR of individual edge hashes, sorting cannot cancel duplicate
    edges. Each pair is an unambiguous uint64 made from two uint32 token IDs.
    The hash is an implementation audit, not a memorization significance test.
    """
    edges = (tokens[:-1].astype(np.uint64) << 32) | tokens[1:].astype(np.uint64)
    edges.sort()
    return hashlib.sha256(edges.astype("<u8").tobytes()).hexdigest()


def bigram_shuffle(tokens, vocab_size, seed):
    """Shuffle outgoing edges, then construct one valid Euler trail in O(N).

    The input itself witnesses that an Euler trail exists. Hierholzer's method
    follows unused edges until stuck, then appends vertices while backtracking;
    reversing that list splices all cycles into the original endpoint-to-
    endpoint trail. A naive random walk could get stuck BEFORE using all edges.

    Compact NumPy arrays hold edges, traversal stack, and output. Sorting the
    edges by source adds O(N log N) preprocessing. The main traversal is O(N).
    Each source's outgoing order is independently randomized with PCG64.
    """
    tokens = checked_tokens(tokens, vocab_size)
    if len(tokens) < 2:
        return tokens.copy()
    counts = np.bincount(tokens[:-1], minlength=vocab_size)
    bounds = np.concatenate(([0], np.cumsum(counts)))
    destinations = tokens[1:][np.argsort(tokens[:-1], kind="stable")].copy()
    rng = np.random.Generator(np.random.PCG64(seed))
    for source in np.flatnonzero(counts > 1):
        rng.shuffle(destinations[bounds[source]:bounds[source + 1]])
    cursor = bounds[:-1].copy()
    stack = np.empty(len(tokens), dtype=np.int32)
    output = np.empty_like(stack)
    stack[0] = tokens[0]
    top = 0
    written = 0
    while top >= 0:
        source = int(stack[top])
        edge = cursor[source]
        if edge < bounds[source + 1]:
            cursor[source] += 1
            top += 1
            stack[top] = destinations[edge]
        else:
            output[written] = source
            written += 1
            top -= 1
    if written != len(tokens) or np.any(cursor != bounds[1:]):
        raise AssertionError("Euler traversal failed to consume every edge")
    result = output[::-1].copy()
    if result[0] != tokens[0] or result[-1] != tokens[-1]:
        raise AssertionError("Euler traversal changed an endpoint")
    return result


def summarize_paths(tokens, candidates, vocab_size):
    """Score frozen paths; every method retains its original denominator.

    Report distinct matching substrings alongside record counts, since many
    paths can share the same newline/punctuation sequence. Whole-candidate
    matches are kept separate from matches to any internal substring.
    """
    index = CorpusIndex(tokens, vocab_size)
    cache = {}
    methods = defaultdict(list)
    for candidate in candidates:
        sequence = tuple(candidate["token_ids"])
        if sequence not in cache:
            cache[sequence] = index.analyze(sequence)["longest_match"]
        methods[candidate["method"]].append((sequence, cache[sequence]))
    output = {}
    for method, records in sorted(methods.items()):
        selected = []
        for sequence, match in records:
            if match["length"]:
                start = match["candidate_token_start"]
                selected.append(sequence[start:start + match["length"]])
        lengths = [match["length"] for _, match in records]
        output[method] = {
            "candidates": len(records),
            "distinct_candidate_sequences": len({seq for seq, _ in records}),
            "max_longest_match_tokens": max(lengths, default=0),
            "mean_longest_match_tokens": sum(lengths) / len(records),
            "whole_candidate_matches": sum(len(seq) == match["length"]
                                            for seq, match in records),
            "candidates_with_match_at_least": {
                str(length): sum(n >= length for n in lengths) for length in LENGTHS
            },
            "distinct_selected_longest_substrings_at_least": {
                str(length): len({seq for seq in selected if len(seq) >= length})
                for length in LENGTHS
            },
        }
    return output


def evaluate_controls(tokens, candidates, vocab_size, seeds=(17, 29, 43)):
    """Compare fixed candidates to actual and pair-preserving shuffled corpus.

    Pairs are omitted because their occurrence counts are identical by design.
    This saves time and prevents an uninformative pair count from dominating a
    summary aimed specifically at multi-token passage order.
    """
    tokens = checked_tokens(tokens, vocab_size)
    validate_candidates(candidates, vocab_size)
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(
        type(seed) is not int or seed < 0 for seed in seeds
    ):
        raise ValueError("provide distinct nonnegative integer seeds")
    paths = [candidate for candidate in candidates if len(candidate["token_ids"]) >= 3]
    original_digest = bigram_digest(tokens)
    original_counts = np.bincount(tokens, minlength=vocab_size)
    original_hash = hashlib.sha256(tokens.astype("<i4").tobytes()).hexdigest()
    controls = []
    for seed in seeds:
        shuffled = bigram_shuffle(tokens, vocab_size, seed)
        digest = bigram_digest(shuffled)
        if digest != original_digest or not np.array_equal(
            original_counts, np.bincount(shuffled, minlength=vocab_size)
        ):
            raise AssertionError("shuffle changed unigram or directed bigram counts")
        controls.append({
            "seed": seed,
            "token_sha256": hashlib.sha256(shuffled.astype("<i4").tobytes()).hexdigest(),
            "bigram_multiset_sha256": digest,
            "positions_different_from_original": int(np.count_nonzero(tokens != shuffled)),
            "by_method": summarize_paths(shuffled, paths, vocab_size),
        })
    return {
        "schema_version": 1,
        "stage": "verification_only",
        "control": "randomized_euler_trail_preserving_directed_bigram_counts",
        "limitations": [
            "Not uniform over Euler trails; not a p-value or calibrated null.",
            "Only tests order beyond pairs, not whether pair associations were learned.",
            "Corpus used only in verification; candidate extraction must be separately audited.",
            "Preserving pairs can force some longer substrings; rigid graphs may remain unchanged.",
        ],
        "numpy_version": np.__version__,
        "generator": "numpy.random.PCG64",
        "vocab_size": vocab_size,
        "corpus_token_count": len(tokens),
        "corpus_token_sha256": original_hash,
        "bigram_multiset_sha256": original_digest,
        "input_candidates": len(candidates),
        "omitted_candidates_shorter_than_three": len(candidates) - len(paths),
        "eligible_candidates_by_method": dict(Counter(c["method"] for c in paths)),
        "real": summarize_paths(tokens, paths, vocab_size),
        "shuffled_corpora": controls,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-token-ids", type=Path, required=True,
                        help="Previously validated native uint32 little-endian export")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite control report")
    if args.corpus_token_ids.stat().st_size % 4:
        raise ValueError("native token file has an incomplete uint32")
    tokens = np.fromfile(args.corpus_token_ids, dtype="<u4")
    with args.candidates.open() as source:
        candidates = [json.loads(line) for line in source if line.strip()]
    report = evaluate_controls(tokens, candidates, args.vocab_size, args.seeds)
    report["sources"] = {
        "candidates": {"path": str(args.candidates.resolve()), "sha256": sha256_file(args.candidates)},
        "corpus_token_ids": {"path": str(args.corpus_token_ids.resolve()),
                             "sha256": sha256_file(args.corpus_token_ids)},
        "code_sha256": {name: sha256_file(Path(__file__).with_name(name))
                        for name in ("controls.py", "verify.py", "checkpoint.py")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"real": report["real"], "shuffled_corpora": report["shuffled_corpora"]}, indent=2))


if __name__ == "__main__":
    main()
