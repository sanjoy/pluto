"""Verification-only controls preserving every ordered corpus trigram exactly.

A path assembled from three-token scores may match language because it contains
ordinary triples, not because it recovers passage-specific order. We randomize
the CORPUS, never the frozen candidates, while preserving the multiplicity of
every trigram, bigram, and token, and the first and last token pairs.

This is a randomized Euler trail on observed token-pair nodes. It is NOT uniform
over possible trails and gives no calibrated p-value. Rigid graphs may force the
original sequence or some longer substrings. Preserving triples deliberately
retains real learned local information: the question is whether matches provide
evidence BEYOND those triples. This module is validation, not extraction.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .checkpoint import sha256_file
from .controls import checked_tokens
from .verify import CorpusIndex, validate_candidates


LENGTHS = (4, 5, 8, 12, 16, 32)
DEFAULT_SEEDS = (17, 29, 43)


def _checked_seeds(seeds):
    try:
        seeds = tuple(seeds)
    except TypeError as error:
        raise ValueError("provide distinct nonnegative integer seeds") from error
    # Validate types BEFORE constructing a set; malformed nested inputs must
    # produce a useful validation error rather than an unhashable-type error.
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("provide distinct nonnegative integer seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("provide distinct nonnegative integer seeds")
    return seeds


def trigram_shuffle(tokens, vocab_size, seed):
    """Construct one endpoint-preserving Euler trail through all trigram edges.

    A corpus a,b,c,d describes pair nodes (a,b),(b,c),(c,d), with one edge
    per triple. Permuting a node's outgoing edges changes ordering without
    changing those triples. Hierholzer's stack/backtracking algorithm consumes
    all edges; a simple random walk could strand unused edges in a cycle.

    Only OBSERVED pairs are allocated, not a vocabulary-squared table. Sorting
    pairs/edges costs O(N log N); traversal and array storage are O(N). Pair IDs
    and the traversal stack use int64 independently of the int32 token format.
    PCG64 shuffles each node's destinations in canonical numeric node order.
    """
    tokens = checked_tokens(tokens, vocab_size)
    _checked_seeds((seed,))
    if len(tokens) < 3:
        return tokens.copy()

    # A pair fits in uint64, including the largest supported int32 token ID.
    pair_codes = ((tokens[:-1].astype(np.uint64) << 32)
                  | tokens[1:].astype(np.uint64))
    nodes, pair_ids = np.unique(pair_codes, return_inverse=True)
    sources = pair_ids[:-1]
    counts = np.bincount(sources, minlength=len(nodes))
    bounds = np.concatenate(([0], np.cumsum(counts)))
    destinations = pair_ids[1:][np.argsort(sources, kind="stable")].copy()
    rng = np.random.Generator(np.random.PCG64(seed))
    for source in np.flatnonzero(counts > 1):
        rng.shuffle(destinations[bounds[source]:bounds[source + 1]])
    cursor = bounds[:-1].copy()
    stack = np.empty(len(pair_ids), dtype=np.int64)
    reverse_trail = np.empty_like(stack)
    stack[0] = pair_ids[0]
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
            reverse_trail[written] = source
            written += 1
            top -= 1
    if written != len(pair_ids) or np.any(cursor != bounds[1:]):
        raise AssertionError("Euler traversal failed to consume every trigram")

    trail = nodes[reverse_trail[::-1]]
    left = trail >> 32
    right = trail & np.uint64(0xFFFFFFFF)
    if not np.array_equal(right[:-1], left[1:]):
        raise AssertionError("Euler trail contains nonoverlapping pair nodes")
    result = np.empty(len(tokens), dtype=np.int32)
    result[0] = left[0]
    result[1:] = right
    if not np.array_equal(result[:2], tokens[:2]) or not np.array_equal(
        result[-2:], tokens[-2:]
    ):
        raise AssertionError("Euler traversal changed an endpoint pair")
    return result


def _canonical_ngrams(tokens, width):
    """Sorted rows retain order WITHIN each n-gram and duplicate multiplicity."""
    count = max(0, len(tokens) - width + 1)
    rows = np.empty((count, width), dtype="<u4")
    for column in range(width):
        rows[:, column] = tokens[column:column + count]
    # np.lexsort's last key is primary. Explicit columns avoid host byte-order
    # dependence and ensure ordinary numeric lexicographic row ordering.
    order = np.lexsort(tuple(rows[:, column]
                             for column in reversed(range(width))))
    return rows[order]


def _array_digest(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def trigram_digest(tokens):
    """Hash lexicographically sorted little-endian uint32 triple rows."""
    tokens = checked_tokens(tokens, 2**31)
    return _array_digest(_canonical_ngrams(tokens, 3))


def summarize_paths(tokens, candidates, vocab_size):
    """Separate full candidates, internal substrings, and duplicate proposals.

    CorpusIndex's longest-match tie rule selects the earliest candidate offset;
    distinct-substring counts refer to those selected strings, not all tied
    substrings. The cache spans methods as well as records: repeated proposals
    are compared once but still contribute to their original method denominator.
    """
    if not candidates:
        return {}
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
        lengths = [match["length"] for _, match in records]
        selected = set()
        whole = []
        for sequence, match in records:
            if match["length"]:
                start = match["candidate_token_start"]
                selected.add(sequence[start:start + match["length"]])
            if len(sequence) == match["length"]:
                whole.append(sequence)
        output[method] = {
            "candidates": len(records),
            "distinct_candidate_sequences": len({seq for seq, _ in records}),
            "candidate_length_histogram": dict(sorted(
                Counter(str(len(seq)) for seq, _ in records).items(),
                key=lambda item: int(item[0]))),
            "longest_match_length_histogram": dict(sorted(
                Counter(str(length) for length in lengths).items(),
                key=lambda item: int(item[0]))),
            "max_longest_match_tokens": max(lengths),
            "mean_longest_match_tokens": sum(lengths) / len(records),
            "whole_candidate_matches": len(whole),
            "distinct_whole_candidate_matches": len(set(whole)),
            "candidates_with_match_at_least": {
                str(length): sum(n >= length for n in lengths)
                for length in LENGTHS
            },
            "distinct_selected_longest_substrings_at_least": {
                str(length): sum(len(seq) >= length for seq in selected)
                for length in LENGTHS
            },
        }
    return output


def evaluate_controls(tokens, candidates, vocab_size, seeds=DEFAULT_SEEDS):
    """Verify fixed length>=4 candidates against real and shuffled corpora.

    Length<=3 candidates are uninformative under exact triple preservation and
    are omitted, but still validated. Each shuffle is checked with elementwise
    equality of sorted n-gram rows, not merely by comparing their hashes. Seeds
    are never retried to obtain a more different or more favorable shuffle.
    """
    tokens = checked_tokens(tokens, vocab_size)
    validate_candidates(candidates, vocab_size)
    seeds = _checked_seeds(seeds)
    paths = [record for record in candidates if len(record["token_ids"]) >= 4]
    canonical = {width: _canonical_ngrams(tokens, width)
                 for width in (1, 2, 3)}
    names = {1: "unigram", 2: "bigram", 3: "trigram"}
    digests = {names[width] + "_multiset_sha256": _array_digest(rows)
               for width, rows in canonical.items()}
    controls = []
    for seed in seeds:
        shuffled = trigram_shuffle(tokens, vocab_size, seed)
        checks = {
            "token_count": len(shuffled) == len(tokens),
            "first_pair": np.array_equal(shuffled[:2], tokens[:2]),
            "last_pair": np.array_equal(shuffled[-2:], tokens[-2:]),
        }
        shuffled_digests = {}
        for width, original in canonical.items():
            rows = _canonical_ngrams(shuffled, width)
            checks[names[width] + "_multiplicities"] = np.array_equal(
                rows, original)
            shuffled_digests[names[width] + "_multiset_sha256"] = (
                _array_digest(rows))
        if not all(checks.values()):
            raise AssertionError("shuffle violated exact preservation: "
                                 + repr(checks))
        changed = int(np.count_nonzero(tokens != shuffled))
        controls.append({
            "seed": seed,
            "token_sha256": _array_digest(shuffled.astype("<u4")),
            **shuffled_digests,
            "preservation_checks": checks,
            "positions_different_from_original": changed,
            "fraction_positions_different": changed / len(tokens) if len(tokens) else 0.0,
            "unchanged_sequence": changed == 0,
            "by_method": summarize_paths(shuffled, paths, vocab_size),
        })
    return {
        "schema_version": 1,
        "stage": "verification_only",
        "control": "randomized_euler_trail_preserving_ordered_trigram_counts",
        "preserved_order": 3,
        "minimum_candidate_length": 4,
        "limitations": [
            "Nonuniform over Euler trails; not a p-value or calibrated null.",
            "Tests order beyond triples, not whether triple associations were learned.",
            "Corpus is used only in verification; extraction needs its own leakage audit.",
            "Rigid graphs may remain unchanged or force longer matching substrings.",
            "Distinct substring counts select the earliest candidate offset among longest ties.",
        ],
        "numpy_version": np.__version__,
        "generator": "numpy.random.PCG64",
        "ngram_hash_format": "numeric lexicographic rows of little-endian uint32 IDs; duplicates retained",
        "vocab_size": vocab_size,
        "corpus_token_count": len(tokens),
        "corpus_token_sha256": _array_digest(tokens.astype("<u4")),
        **digests,
        "first_pair": tokens[:2].tolist(),
        "last_pair": tokens[-2:].tolist(),
        "input_candidates": len(candidates),
        "omitted_candidates_shorter_than_four": len(candidates) - len(paths),
        "eligible_candidates_by_method": dict(sorted(
            Counter(record["method"] for record in paths).items())),
        "real": summarize_paths(tokens, paths, vocab_size),
        "shuffled_corpora": controls,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-token-ids", type=Path, required=True,
                        help="Previously validated native little-endian uint32 export")
    parser.add_argument("--candidates", type=Path, required=True,
                        help="Frozen candidate JSONL; extraction must precede verification")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite higher-order control report")

    # Read each input once: the hashes then describe the exact bytes analyzed,
    # even if another process later replaces a source file.
    token_bytes = args.corpus_token_ids.read_bytes()
    candidate_bytes = args.candidates.read_bytes()
    if len(token_bytes) % 4:
        raise ValueError("native token file has an incomplete uint32")
    tokens = np.frombuffer(token_bytes, dtype="<u4")
    candidates = [json.loads(line) for line in candidate_bytes.splitlines()
                  if line.strip()]
    sources = {
        "candidates": {
            "path": str(args.candidates.resolve()),
            "sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        },
        "corpus_token_ids": {
            "path": str(args.corpus_token_ids.resolve()),
            "sha256": hashlib.sha256(token_bytes).hexdigest(),
        },
        "code_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("higher_order_controls.py", "controls.py",
                         "verify.py", "checkpoint.py")
        },
    }
    report = evaluate_controls(tokens, candidates, args.vocab_size, args.seeds)
    report["sources"] = sources
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"real": report["real"],
                      "shuffled_corpora": report["shuffled_corpora"]}, indent=2))


if __name__ == "__main__":
    main()
