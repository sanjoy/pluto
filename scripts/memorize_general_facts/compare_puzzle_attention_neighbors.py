#!/usr/bin/env python3
"""Compare A3/A4 target-label neighborhoods on exactly the same scored rows.

Consumes the optional state TSV from puzzle_prediction_probe, without training.
Requires NumPy and SciPy. Singleton targets have no possible matching neighbor;
reporting a repeated-target query subset makes this limitation explicit.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from analyze_puzzle_geometry import load_coordinates, neighbor_features
from analyze_puzzle_predictions import load_rows


def load_states(path, rows):
    with Path(path).open() as stream:
        reader = csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        width = sum(name.startswith("a3_") for name in reader.fieldnames)
        expected = ["fact_index", "position", "target", "input_token"]
        expected += [f"{boundary}_{i}" for boundary in ("a3", "a4") for i in range(width)]
        if width == 0 or reader.fieldnames != expected:
            raise ValueError("unexpected A3/A4 state TSV header")
        by_key = {}
        for record in reader:
            key = int(record["fact_index"]), int(record["position"])
            if key in by_key:
                raise ValueError("duplicate state row")
            by_key[key] = record
    if set(by_key) != {(row["fact_index"], row["position"]) for row in rows}:
        raise ValueError("prediction/state row coverage differs")
    vectors = {boundary: [] for boundary in ("a3", "a4")}
    for row in rows:
        record = by_key[row["fact_index"], row["position"]]
        if int(record["target"]) != row["target"] or int(record["input_token"]) != row["input_token"]:
            raise ValueError("prediction/state token IDs differ")
        for boundary in vectors:
            vectors[boundary].append([float(record[f"{boundary}_{i}"]) for i in range(width)])
    vectors = {name: np.asarray(values, dtype=np.float32).astype(np.float64)
               for name, values in vectors.items()}
    if not all(np.isfinite(values).all() for values in vectors.values()):
        raise ValueError("nonfinite state coordinate")
    return vectors


def normalize(vectors):
    centered = vectors - vectors.mean(axis=1, keepdims=True)
    return centered / np.sqrt(np.mean(centered ** 2, axis=1, keepdims=True) + 1e-5)


def summarize_pool(vectors, rows):
    targets = np.array([row["target"] for row in rows])
    facts = np.array([row["fact_index"] for row in rows])
    counts = Counter(targets)
    frequency = np.array([counts[target] for target in targets])
    correct = np.array([row["correct"] for row in rows])
    groups = {"all_queries": np.ones(len(rows), dtype=bool),
              "repeated_target_queries": frequency > 1,
              "a3_readout_correct_queries": correct,
              "a3_readout_wrong_queries": ~correct}
    features = {
        boundary: {space: neighbor_features(values, targets, facts)
                   for space, values in (("raw", x), ("normalized", normalize(x)))}
        for boundary, x in vectors.items()
    }
    report = {"rows": len(rows), "target_types": len(counts),
              "singleton_target_rows": int((frequency == 1).sum()), "groups": {}}
    for group, mask in groups.items():
        if not mask.any():
            continue
        result = {"queries": int(mask.sum()),
                  "chance_neighbor_agreement": float(((frequency[mask] - 1) / (len(rows) - 1)).mean()),
                  "maximum_purity": {
                      str(k): float((np.minimum(frequency[mask] - 1, min(k, len(rows) - 1)) /
                                     min(k, len(rows) - 1)).mean()) for k in (1, 5, 25)}}
        for boundary in features:
            result[boundary] = {}
            for space, feature in features[boundary].items():
                result[boundary][space] = {
                    "nearest_same_target": int(feature["nn_same_target"][mask].sum()),
                    "nearest_agreement": float(feature["nn_same_target"][mask].mean()),
                    "purity_5": float(feature["knn5_purity"][mask].mean()),
                    "purity_25": float(feature["knn25_purity"][mask].mean()),
                    "different_fact_nearest_agreement": float(np.nanmean(feature["different_fact_nn_same_target"][mask])),
                    "nearest_same_fact_fraction": float(feature["same_fact_nearest_neighbor"][mask].mean()),
                }
        result["paired_nearest_agreement"] = {}
        for space in ("raw", "normalized"):
            a3 = features["a3"][space]["nn_same_target"][mask].astype(bool)
            a4 = features["a4"][space]["nn_same_target"][mask].astype(bool)
            result["paired_nearest_agreement"][space] = {
                "both": int((a3 & a4).sum()), "a3_only": int((a3 & ~a4).sum()),
                "a4_only": int((~a3 & a4).sum()), "neither": int((~a3 & ~a4).sum())}
        report["groups"][group] = result
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--a3_capture_html", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.predictions)
    vectors = load_states(args.states, rows)
    prior_a3 = load_coordinates(args.a3_capture_html, rows)
    if not np.array_equal(prior_a3, vectors["a3"]):
        raise ValueError("new A3 capture differs from the original puzzle capture")
    lexical = np.array([r["category"] not in ("EOS", "Punctuation") for r in rows])
    lexical_rows = [r for r, keep in zip(rows, lexical) if keep]
    report = {
        "method": {
            "boundaries": "post-attention residuals of blocks 3/4 (one-based), before each block's MLP",
            "normalization": "subtract row mean; divide by sqrt(row variance + 1e-5); no learned affine parameters",
            "neighbors": "Euclidean; query row excluded; each boundary has the same candidate rows",
            "lexical_pool": "EOS/punctuation removed from both queries and candidates",
            "a3_capture_matches_prior_exactly": True,
            "a3_readout_correct_queries": "subsets refer only to the fixed A3 readout, not an A4 trained readout",
        },
        "all_scored": summarize_pool(vectors, rows),
        "lexical_only": summarize_pool({name: x[lexical] for name, x in vectors.items()}, lexical_rows),
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
