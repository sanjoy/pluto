#!/usr/bin/env python3
"""Compare A3 neighborhoods with exact puzzle-readout successes (CPU only).

Requires NumPy and SciPy. Reads the standalone capture HTML and prediction TSV;
prints JSON without modifying either input. Neighbors exclude the query row.
The lexical analysis removes EOS and punctuation from BOTH query and candidate
pools, using analyze_puzzle_predictions' shared token-category heuristic.
These are associations on the training corpus, not causal or held-out results.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from analyze_puzzle_predictions import frequency_bin, load_rows


def load_coordinates(path, rows):
    """Join by fact/position and reject incompatible captures or token IDs."""
    text = Path(path).read_text()
    marker = "const data = "
    start = text.index(marker) + len(marker)
    data, _ = json.JSONDecoder().raw_decode(text[start:])
    points = {}
    for point in data["points"]:
        if point["target_token"] < 0:
            continue
        key = point["fact_index"], point["position"]
        if key in points:
            raise ValueError("duplicate scored capture row")
        points[key] = point
    if len(rows) != len(points) or set(points) != {
            (row["fact_index"], row["position"]) for row in rows}:
        raise ValueError("capture and predictions have different scored rows")
    coordinates = []
    for row in rows:
        point = points[row["fact_index"], row["position"]]
        if (point["target_token"] != row["target"] or
                point["input_token"] != row["input_token"]):
            raise ValueError("capture and prediction token IDs disagree")
        fact = data["facts"][row["fact_index"]]
        if not fact.startswith(row["prefix"]) or (
                row["category"] == "EOS" and fact != row["prefix"]):
            raise ValueError("capture and prediction fact text disagree")
        coordinates.append(point["coordinates"])
    # HTML max_digits10 numbers round-trip the original BF16 values via FP32.
    vectors = np.asarray(coordinates, dtype=np.float32).astype(np.float64)
    if vectors.shape != (len(rows), data["model_width"]) or not np.isfinite(vectors).all():
        raise ValueError("invalid capture coordinate shape or nonfinite value")
    return vectors


def neighbor_features(vectors, targets, facts):
    """Exact Euclidean neighbors; self is excluded even for duplicate vectors."""
    count = len(targets)
    if count < 2:
        raise ValueError("geometry requires at least two candidate rows")
    # Enough neighbors for 25-row purity and a neighbor from another fact.
    query_count = min(count, max(26, max(Counter(facts).values()) + 1))
    distances, indices = cKDTree(vectors).query(vectors, k=query_count, workers=2)
    neighbor_ids = np.empty((count, query_count - 1), dtype=np.int64)
    neighbor_distances = np.empty_like(neighbor_ids, dtype=float)
    for row in range(count):
        keep = indices[row] != row
        neighbor_ids[row] = indices[row][keep][:query_count - 1]
        neighbor_distances[row] = distances[row][keep][:query_count - 1]
    agreement = targets[neighbor_ids] == targets[:, None]
    different_fact = facts[neighbor_ids] != facts[:, None]
    different_fact_agreement = np.full(count, np.nan)
    valid = different_fact.any(axis=1)
    first = different_fact.argmax(axis=1)
    different_fact_agreement[valid] = agreement[np.arange(count)[valid], first[valid]]
    return {
        "nn_same_target": agreement[:, 0].astype(float),
        "nn_distance": neighbor_distances[:, 0],
        "knn5_purity": agreement[:, :5].mean(axis=1),
        "knn25_purity": agreement[:, :25].mean(axis=1),
        "different_fact_nn_same_target": different_fact_agreement,
        "same_fact_nearest_neighbor": (~different_fact[:, 0]).astype(float),
    }


def correlation(x, y, ranked=True):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None
    return float(spearmanr(x, y).statistic if ranked else np.corrcoef(x, y)[0, 1])


def within_target_correlation(feature, correct, targets):
    """Pearson association after subtracting each target ID's own mean."""
    _, groups = np.unique(targets, return_inverse=True)
    counts = np.bincount(groups)
    x = feature - (np.bincount(groups, weights=feature) / counts)[groups]
    y = correct - (np.bincount(groups, weights=correct) / counts)[groups]
    return correlation(x, y, ranked=False)


def mean(values):
    return float(values.mean()) if len(values) else None


def describe(geometry, correct, targets, frequencies, mask=None):
    if mask is None:
        mask = np.ones(len(correct), dtype=bool)
    result = {"count": int(mask.sum()), "correct": int(correct[mask].sum()),
              "accuracy": mean(correct[mask]),
              "target_frequency_rho_correct": correlation(frequencies[mask], correct[mask])}
    for name, features in geometry.items():
        result[name] = {}
        for feature_name, feature in features.items():
            valid = mask & np.isfinite(feature)
            result[name][feature_name] = {
                "count": int(valid.sum()), "all_mean": mean(feature[valid]),
                "correct_mean": mean(feature[valid & correct]),
                "wrong_mean": mean(feature[valid & ~correct]),
                "rho_correct": correlation(feature[valid], correct[valid]),
            }
        same = features["nn_same_target"] == 1
        result[name]["accuracy_by_nn_agreement"] = {
            "agree": mean(correct[mask & same]),
            "disagree": mean(correct[mask & ~same]),
            "agree_count": int((mask & same).sum()),
            "disagree_count": int((mask & ~same).sum()),
            "agree_correct": int(correct[mask & same].sum()),
            "disagree_correct": int(correct[mask & ~same].sum()),
        }
        result[name]["within_target_pearson"] = {
            feature: within_target_correlation(features[feature][mask], correct[mask], targets[mask])
            for feature in ("nn_same_target", "knn5_purity", "knn25_purity")
        }
    return result


def summarize(vectors, rows):
    targets = np.array([row["target"] for row in rows])
    correct = np.array([row["correct"] for row in rows])
    facts = np.array([row["fact_index"] for row in rows])
    categories = np.array([row["category"] for row in rows])
    counts = Counter(targets)
    frequencies = np.array([counts[target] for target in targets])
    centered = vectors - vectors.mean(axis=1, keepdims=True)
    spaces = {"raw": vectors,
              "layer_normalized": centered / np.sqrt(np.mean(centered ** 2, axis=1, keepdims=True) + 1e-5)}
    geometry = {name: neighbor_features(x, targets, facts) for name, x in spaces.items()}
    lexical = ~np.isin(categories, ("EOS", "Punctuation"))
    if lexical.sum() < 2:
        raise ValueError("lexical geometry requires at least two non-EOS/non-punctuation rows")
    lexical_geometry = {
        name: neighbor_features(x[lexical], targets[lexical], facts[lexical])
        for name, x in spaces.items()
    }
    lexical_describe = lambda mask=None: describe(
        lexical_geometry, correct[lexical], targets[lexical], frequencies[lexical], mask)
    bins = np.array([frequency_bin(int(count)) for count in frequencies[lexical]])
    return {
        "method": {
            "distance": "exact Euclidean; query row excluded; equal-distance ties follow SciPy tree ordering",
            "layer_normalization": "row mean subtraction, then division by sqrt(row variance + 1e-5); no learned affine parameters",
            "purity": "fraction of neighbors sharing the target; k capped at available candidate count minus one",
            "lexical_pool": "EOS and punctuation removed from both queries and candidates using shared category heuristic",
            "correlations": "rho is Spearman; within_target_pearson demeans correctness and feature by target ID",
            "interpretation": "descriptive training-corpus association, not causality or held-out prediction",
        },
        "all_scored": describe(geometry, correct, targets, frequencies),
        "lexical_only": lexical_describe(),
        "category_queries_all_candidates": {
            category: describe(geometry, correct, targets, frequencies, categories == category)
            for category in ("EOS", "Punctuation") if np.any(categories == category)
        },
        "lexical_frequency_bins": {key: lexical_describe(bins == key) for key in dict.fromkeys(bins)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.predictions)
    vectors = load_coordinates(args.html, rows)
    print(json.dumps(summarize(vectors, rows), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
