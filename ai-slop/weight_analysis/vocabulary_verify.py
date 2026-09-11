"""Verify frozen weight-only vocabulary rankings; never select an extractor.

Input JSON contains schema_version=1, vocab_size, and rankings, a list of
{method, token_ids, scores}. Each token_ids list is a full vocabulary permutation;
scores are aligned, descending, with ascending token ID breaking exact ties.
Additional extraction provenance stays in the hashed input artifact.

The actual ranking is consumed exactly as supplied. Fixed prefix sizes and a
seeded score-label permutation are diagnostics, not fitting or reranking using
the corpus. Coverage does not demonstrate passage extraction or authenticated
historical training membership. Never feed these results back into a frozen
extraction experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TOP_K = (128, 512, 2048, 8192)
WINDOW_LENGTHS = (2, 3, 4, 8, 12)
CONTROL_SEED = 17


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _integers(values, label):
    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(label + " must be a one-dimensional integer array")
    return values


def _vocab_size(value):
    if type(value) is not int or not 0 < value <= 2**31:
        raise ValueError("vocab_size must be a positive int32 vocabulary size")
    return value


def checked_tokens(tokens, vocab_size):
    _vocab_size(vocab_size)
    tokens = _integers(tokens, "corpus tokens")
    if len(tokens) and (tokens.min() < 0 or tokens.max() >= vocab_size):
        raise ValueError("corpus token outside vocabulary")
    return tokens.astype(np.int32, copy=False)


def validate_rankings(artifact):
    """Reject malformed/reranked inputs, rather than silently repairing them."""
    if not isinstance(artifact, dict) or type(artifact.get("schema_version")) is not int:
        raise ValueError("expected ranking artifact with schema_version=1")
    if artifact["schema_version"] != 1:
        raise ValueError("unsupported ranking artifact schema")
    vocab_size = _vocab_size(artifact.get("vocab_size"))
    rankings = artifact.get("rankings")
    if not isinstance(rankings, list) or not rankings:
        raise ValueError("rankings must be a nonempty list")
    methods = set()
    for ranking in rankings:
        if not isinstance(ranking, dict):
            raise ValueError("ranking must be an object")
        method = ranking.get("method")
        if not isinstance(method, str) or not method or method in methods:
            raise ValueError("method names must be unique nonempty strings")
        methods.add(method)
        ids, scores = ranking.get("token_ids"), ranking.get("scores")
        if (not isinstance(ids, list) or len(ids) != vocab_size
                or any(type(token) is not int for token in ids)
                or set(ids) != set(range(vocab_size))):
            raise ValueError("token_ids must be a full vocabulary permutation")
        if (not isinstance(scores, list) or len(scores) != vocab_size
                or any(type(score) not in (int, float) for score in scores)):
            raise ValueError("scores must be numeric and aligned with token_ids")
        try:
            values = np.asarray(scores, dtype=np.float64)
        except (ValueError, OverflowError) as error:
            raise ValueError("scores must be finite float64 numbers") from error
        if not np.isfinite(values).all():
            raise ValueError("scores must be finite float64 numbers")
        indices = np.asarray(ids, dtype=np.int64)
        if (np.any(values[:-1] < values[1:])
                or np.any((values[:-1] == values[1:])
                          & (indices[:-1] > indices[1:]))):
            raise ValueError("ranking must descend by score, ties ascending token ID")
    return vocab_size


def average_ranks(values):
    """One-based ascending ranks, averaging exact ties without breaking them.

    The artifact's token-ID tie rule determines top-k membership, but must NOT
    turn tied scores into artificial Spearman correlation.
    """
    values = np.asarray(values)
    if (values.ndim != 1 or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all()):
        raise ValueError("rank values must be a finite numeric vector")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    boundaries = np.concatenate(([0], np.flatnonzero(
        sorted_values[1:] != sorted_values[:-1]) + 1, [len(values)]))
    ranks = np.empty(len(values), dtype=np.float64)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        ranks[order[start:end]] = (start + 1 + end) / 2
    return ranks


def spearman(scores, frequencies):
    """Score/count correlation over all vocabulary IDs, including absent IDs."""
    first, second = average_ranks(scores), average_ranks(frequencies)
    if len(first) != len(second):
        raise ValueError("score and frequency lengths differ")
    if not len(first):
        return None
    first -= first.mean()
    second -= second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    # Constant vectors have undefined correlation, not zero correlation.
    return float(np.clip(np.dot(first, second) / denominator, -1, 1)) if denominator else None


def coverage(tokens, selected_ids, vocab_size):
    """Count exact all-selected windows, including overlapping occurrences.

    Precision is distinct selected IDs present / selected IDs. Recall is
    selected IDs present / all distinct corpus IDs. Token-position coverage
    instead weights each occurrence. These answer different questions.
    """
    tokens = checked_tokens(tokens, vocab_size)
    selected = _integers(selected_ids, "selected IDs")
    if (len(selected) and (selected.min() < 0 or selected.max() >= vocab_size)
            or len(np.unique(selected)) != len(selected)):
        raise ValueError("selected IDs must be distinct and in vocabulary")
    included = np.zeros(vocab_size, dtype=np.bool_)
    included[selected] = True
    mask = included[tokens]
    # Runs of True suffice: an r-token run contains max(r-k+1,0) windows.
    padded = np.concatenate(([False], mask, [False]))
    boundaries = np.flatnonzero(padded[1:] != padded[:-1])
    run_lengths = boundaries[1::2] - boundaries[::2]
    present = np.bincount(tokens, minlength=vocab_size) > 0
    selected_present = int(np.count_nonzero(present[selected]))
    distinct_present = int(np.count_nonzero(present))
    return {
        "selected_ids": len(selected),
        "selected_ids_present": selected_present,
        "corpus_distinct_ids": distinct_present,
        "distinct_vocabulary_precision": selected_present / len(selected) if len(selected) else None,
        "distinct_vocabulary_recall": selected_present / distinct_present if distinct_present else None,
        "covered_token_positions": int(np.count_nonzero(mask)),
        "token_position_coverage": float(mask.mean()) if len(mask) else None,
        "all_selected_windows": {
            str(k): int(np.maximum(run_lengths - k + 1, 0).sum())
            for k in WINDOW_LENGTHS
        },
        "longest_all_selected_run": int(run_lengths.max()) if len(run_lengths) else 0,
    }


def evaluate(artifact, tokens):
    """Apply the fixed protocol only; no corpus-dependent choice is exposed."""
    vocab_size = validate_rankings(artifact)
    tokens = checked_tokens(tokens, vocab_size)
    frequencies = np.bincount(tokens, minlength=vocab_size)
    ids = np.arange(vocab_size, dtype=np.int32)
    permutation = np.random.Generator(np.random.PCG64(CONTROL_SEED)).permutation(vocab_size)
    methods = {}
    for ranking in artifact["rankings"]:
        ranked_ids = np.asarray(ranking["token_ids"], dtype=np.int32)
        aligned_scores = np.asarray(ranking["scores"], dtype=np.float64)
        scores_by_id = np.empty(vocab_size, dtype=np.float64)
        scores_by_id[ranked_ids] = aligned_scores
        # Use one shared label permutation across methods, independent of corpus
        # counts. Exact score ties still obey ascending token ID, as declared.
        control_scores = scores_by_id[permutation]
        control_ids = np.lexsort((ids, -control_scores)).astype(np.int32)

        def summarize(order, scores):
            prefixes = {}
            for k in TOP_K:
                prefix = order[:k]
                prefixes[str(k)] = {
                    "requested_top_k": k,
                    "effective_top_k": len(prefix),
                    "selected_ids_sha256": _digest(prefix.astype("<u4").tobytes()),
                    **coverage(tokens, prefix, vocab_size),
                }
            return {
                "ranking_ids_sha256": _digest(order.astype("<u4").tobytes()),
                "scores_by_token_id_sha256": _digest(scores.astype("<f8").tobytes()),
                "score_frequency_spearman": spearman(scores, frequencies),
                "top_k": prefixes,
            }

        methods[ranking["method"]] = {
            "frozen_ranking": summarize(ranked_ids, scores_by_id),
            "permuted_score_control": summarize(control_ids, control_scores),
        }
    return {
        "schema_version": 1,
        "stage": "verification_only",
        "vocab_size": vocab_size,
        "corpus_token_count": len(tokens),
        "corpus_token_ids_sha256": _digest(tokens.astype("<u4").tobytes()),
        "frequency_counts_sha256": _digest(frequencies.astype("<u8").tobytes()),
        "top_k": list(TOP_K),
        "window_lengths": list(WINDOW_LENGTHS),
        "spearman_definition": "weight score versus token count across all vocabulary IDs; averaged ties; positive means higher scores associate with higher counts; null if undefined",
        "control": {
            "generator": "numpy.random.PCG64",
            "numpy_version": np.__version__,
            "seed": CONTROL_SEED,
            "permutation_sha256": _digest(permutation.astype("<u4").tobytes()),
            "definition": "control_score[token_id] = original_score[permutation[token_id]]; one permutation shared by all methods; descending scores then ascending token ID",
        },
        "limitations": [
            "Verification only; rankings must have been frozen without corpus access.",
            "Coverage is necessary but not sufficient for passage extraction.",
            "Full source corpus membership does not authenticate historical training membership.",
            "Fixed score permutation is a descriptive control, not a calibrated p-value.",
            "Exact ties use token IDs for top-k membership, but averaged ranks for Spearman.",
            "Top-k is capped at vocabulary size for small fixtures; effective size is reported.",
        ],
        "by_method": methods,
    }


def write_report(path, report):
    """Write a fresh artifact exclusively; never replace a prior result."""
    path = Path(path)
    # Serialize first so invalid JSON values cannot leave a partial new report.
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as output:
        output.write(encoded)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", required=True, type=Path)
    parser.add_argument("--corpus-token-ids", required=True, type=Path,
                        help="Previously validated native little-endian uint32 tokens")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite vocabulary verification report")
    ranking_bytes = args.rankings.read_bytes()
    token_bytes = args.corpus_token_ids.read_bytes()
    if len(token_bytes) % 4:
        raise ValueError("native corpus file contains an incomplete uint32")
    report = evaluate(json.loads(ranking_bytes), np.frombuffer(token_bytes, dtype="<u4"))
    report["sources"] = {
        "rankings": {"path": str(args.rankings.resolve()), "sha256": _digest(ranking_bytes)},
        "corpus_token_ids": {"path": str(args.corpus_token_ids.resolve()), "sha256": _digest(token_bytes)},
        "verifier": {"path": str(Path(__file__).resolve()),
                     "sha256": _digest(Path(__file__).read_bytes())},
    }
    write_report(args.output, report)
    print(json.dumps({"output": str(args.output), "methods": list(report["by_method"])}))


if __name__ == "__main__":
    main()
