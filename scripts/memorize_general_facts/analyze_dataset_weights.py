#!/usr/bin/env python3
"""Held-out, CPU-only analysis of controlled dataset-to-weight training runs.

Token IDs are categorical labels, never numerical regression features. Models
predict FP32 training deltas; a tiny error on mostly unchanged initialized
weights must not masquerade as explaining learning. Predictions use only the
preassigned training split. Kernel hyperparameters use leave-one-training-run-
out validation, never the held-out runs. NumPy is the only extra dependency.
"""

import argparse
from dataclasses import dataclass
import hashlib
import html
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Tensor:
    """One unique FP32 checkpoint tensor in native serialization order."""

    name: str
    shape: tuple
    start: int
    stop: int


def tensor_layout(model):
    """Follow CreateGpt2's actual storage, including tied embedding deduping.

    Dense matrices store input-major [input_dim, output_dim] FP32 masters.
    Q/K/V are adjacent output column groups, not three checkpoint files.
    """
    width = int(model["model_width"])
    feed_forward = int(model["feed_forward_width"])
    layers = int(model["layers"])
    vocab = int(model["vocabulary_size"])
    context = int(model["context_length"])
    if min(width, feed_forward, vocab, context) <= 0 or layers < 0:
        raise ValueError("invalid model shape")
    shapes = [("token_embedding", (vocab, width)),
              ("position_embedding", (context, width))]
    for block in range(layers):
        for name, shape in (
            ("attention.layer_norm.scale", (width,)),
            ("attention.layer_norm.bias", (width,)),
            ("attention.qkv.matrix", (width, 3 * width)),
            ("attention.qkv.bias", (3 * width,)),
            ("attention.output.matrix", (width, width)),
            ("attention.output.bias", (width,)),
            ("mlp.layer_norm.scale", (width,)),
            ("mlp.layer_norm.bias", (width,)),
            ("mlp.expand.matrix", (width, feed_forward)),
            ("mlp.expand.bias", (feed_forward,)),
            ("mlp.contract.matrix", (feed_forward, width)),
            ("mlp.contract.bias", (width,)),
        ):
            shapes.append((f"block_{block}.{name}", shape))
    shapes.extend([("final_layer_norm.scale", (width,)),
                   ("final_layer_norm.bias", (width,))])
    result, start = [], 0
    for name, shape in shapes:
        stop = start + int(np.prod(shape))
        result.append(Tensor(name, shape, start, stop))
        start = stop
    return result


def read_checkpoint(path, layout):
    """Reject missing, extra, non-FP32-sized, or nonfinite parameter tensors."""
    path = Path(path)
    expected = {f"weight_{i}.bin" for i in range(len(layout))}
    found = {p.name for p in path.glob("weight_*.bin")}
    if found != expected:
        raise ValueError(f"{path}: checkpoint weight filenames do not match model")
    arrays, digest = [], hashlib.sha256()
    for index, tensor in enumerate(layout):
        contents = (path / f"weight_{index}.bin").read_bytes()
        if len(contents) != 4 * (tensor.stop - tensor.start):
            raise ValueError(f"{path}/weight_{index}.bin: wrong FP32 tensor size")
        data = np.frombuffer(contents, dtype="<f4").astype(np.float64)
        if not np.isfinite(data).all():
            raise ValueError(f"{path}/weight_{index}.bin: nonfinite weights")
        arrays.append(data)
        digest.update(contents)
    return np.concatenate(arrays), digest.hexdigest()


def read_tokens(path):
    """Keep sentence lengths in addition to the unpadded categorical vector."""
    rows = [line.split() for line in Path(path).read_text().splitlines()
            if line.strip()]
    if not rows:
        raise ValueError("empty token corpus")
    tokens = np.array([int(token) for row in rows for token in row], dtype=np.int64)
    if (tokens < 0).any() or (tokens > np.iinfo(np.int32).max).any():
        raise ValueError("token IDs must be nonnegative int32")
    return tokens, tuple(map(len, rows))


def hamming(left, right):
    """Fraction of differing token positions; invariant to shared ID renaming."""
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Hamming inputs require matching token-vector dimensions")
    # Avoid materializing [runs, runs, 14k tokens] for larger future sweeps.
    return np.array([[np.mean(a != b) for b in right] for a in left])


def metrics(target, prediction):
    """Normalized RMSE is ||prediction-target|| / ||target||, on deltas."""
    target, prediction = np.asarray(target), np.asarray(prediction)
    error = prediction - target
    norm = float(np.linalg.norm(target))
    predicted_norm = float(np.linalg.norm(prediction))
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "normalized_rmse": float(np.linalg.norm(error) / norm) if norm else None,
        "cosine": float(np.dot(target, prediction) / (norm * predicted_norm))
        if norm and predicted_norm else None,
        "target_l2": norm,
    }


def weight_difference(reference, observed):
    """Compare FP32 parameter values, not bytes or training-update vectors.

    Exact changes use numeric inequality (positive and negative zero compare
    equal). The second count applies a strict absolute threshold of 0.001.
    Relative L2 divides by the reference checkpoint's norm, not the norm of a
    training delta, so it answers a different question from prediction NRMSE.
    """
    difference = np.asarray(observed) - np.asarray(reference)
    norm = float(np.linalg.norm(reference))
    distance = float(np.linalg.norm(difference))
    return {"l2": distance, "relative_l2": distance / norm if norm else None,
            "reference_l2": norm,
            "changed_parameters": int(np.count_nonzero(difference)),
            "abs_gt_1e3_count": int(np.count_nonzero(np.abs(difference) > .001))}


def memorization_summary(trial):
    """A once-perfect model may no longer be perfect at the fixed endpoint."""
    first = trial.get("first_memorized_step")
    ever = bool(first > 0) if first is not None else (True if trial["success"] else None)
    audit = trial.get("memorization_audit")
    verified = bool(trial.get("memorization_checkpoint") and isinstance(audit, dict)
                    and audit.get("errors") == 0)
    return {"ever_memorized": ever, "first_memorized_step": first,
            "memorization_checkpoint": trial.get("memorization_checkpoint"),
            "memorization_independently_verified": verified,
            "fixed_endpoint_success": trial["success"]}


def kernel_weights(train_distances, query_distances, bandwidth, ridge):
    """Kernel ridge with an unpenalized constant mean (universal kriging).

    An augmented linear system fits that mean using training data only. Its
    inverse also gives exact leave-one-out errors without refitting 50k output
    dimensions for every candidate hyperparameter.
    """
    n = len(train_distances)
    system = np.ones((n + 1, n + 1), dtype=np.float64)
    system[:n, :n] = np.exp(-train_distances / bandwidth) + ridge * np.eye(n)
    system[n, n] = 0.0
    inverse = np.linalg.inv(system)
    query = np.column_stack([np.exp(-query_distances / bandwidth),
                             np.ones(len(query_distances))])
    return query @ inverse[:, :n], inverse[:n, :n]


def fit_predict(tokens, deltas, query_tokens):
    """Return held-out predictions and exclusively training-selected settings."""
    if len(tokens) < 2:
        raise ValueError("at least two training datasets are required")
    distances = hamming(tokens, tokens)
    queries = hamming(query_tokens, tokens)
    predictions = {
        "initialization_only": np.zeros((len(query_tokens), deltas.shape[1])),
        "training_mean_delta": np.repeat(deltas.mean(axis=0)[None, :],
                                          len(query_tokens), axis=0),
        "nearest_dataset": deltas[queries.argmin(axis=1)],
    }
    settings = {"nearest_training_indices": queries.argmin(axis=1).tolist()}
    if len(tokens) < 3:
        settings["kernel_status"] = "need at least three training runs for tuning"
        return predictions, settings

    # The unpenalized intercept annihilates constant rows. Center first to
    # avoid subtracting large common training-update energies in LOOCV.
    centered = deltas - deltas.mean(axis=0)
    gram = centered @ centered.T
    positive = distances[distances > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    best = None
    for bandwidth in sorted(set([scale / 16, scale / 4, scale, scale * 4, 4.0])):
        for ridge in (1e-5, 0.001, 0.1, 1.0, 10.0):
            weights, inverse = kernel_weights(distances, queries, bandwidth, ridge)
            diagonal = np.diag(inverse)
            residual_norms = np.einsum("ij,jk,ik->i", inverse, gram, inverse)
            loo_mse = float(np.mean(np.maximum(residual_norms, 0) /
                                    (diagonal * diagonal)) / deltas.shape[1])
            candidate = (loo_mse, bandwidth, ridge)
            if best is None or candidate < best[0]:
                best = candidate, weights
    (loo_mse, bandwidth, ridge), weights = best
    predictions["hamming_kernel_ridge"] = weights @ deltas
    settings.update({"bandwidth": bandwidth, "ridge": ridge,
                     "training_leave_one_out_mse": loo_mse})
    return predictions, settings


def align_embeddings(weights, permutation, layout):
    """Undo vocabulary renaming: aligned E[old] = renamed E[pi(old)]."""
    aligned = weights.copy()
    embedding = layout[0]
    aligned[:embedding.stop] = weights[:embedding.stop].reshape(
        embedding.shape)[permutation].reshape(-1)
    return aligned


def _read_permutation(path, vocabulary):
    path = Path(path)
    values = (np.load(path, allow_pickle=False) if path.suffix == ".npy" else
              np.array([int(value) for value in path.read_text().split()]))
    if (values.ndim != 1 or len(values) != vocabulary or
            values.dtype.kind not in "iu" or
            not np.array_equal(np.sort(values), np.arange(vocabulary))):
        raise ValueError("permutation must be an old-to-new vocabulary bijection")
    return values.astype(np.int64)


def _spectral_summary(deltas):
    centered = deltas - deltas.mean(axis=0)
    eigenvalues = np.maximum(np.linalg.eigvalsh(centered @ centered.T), 0)[::-1]
    total = float(eigenvalues.sum())
    ratios = eigenvalues / total if total else np.zeros(len(eigenvalues))
    return {"training_runs": len(deltas), "max_possible_centered_rank": len(deltas) - 1,
            "variance_energy_fractions": ratios.tolist(),
            "rank_90_percent": int(np.searchsorted(np.cumsum(ratios), .9) + 1)
            if total else 0,
            "rank_99_percent": int(np.searchsorted(np.cumsum(ratios), .99) + 1)
            if total else 0}


def _summarize_predictions(predictions, targets, trials, layout):
    result = {}
    for name, predicted in predictions.items():
        per_trial = []
        for trial, actual, guess in zip(trials, targets, predicted):
            tensors = []
            for tensor in layout:
                section = slice(tensor.start, tensor.stop)
                tensors.append({"name": tensor.name, "shape": tensor.shape,
                                **metrics(actual[section], guess[section])})
            per_trial.append({"id": trial["id"], "success": trial["success"],
                              "errors": trial.get("errors"),
                              "support": trial.get("support"),
                              "overall": metrics(actual, guess), "tensors": tensors})
        result[name] = {"overall": metrics(targets.reshape(-1), predicted.reshape(-1)),
                        "trials": per_trial}
    return result


def analyze(summary_path):
    """Read an immutable endpoint snapshot; never pool incomparable endpoints."""
    summary_path = Path(summary_path).resolve()
    root = summary_path.parent
    summary = json.loads(summary_path.read_text())
    layout = tensor_layout(summary["model"])
    vocabulary = summary["model"]["vocabulary_size"]
    identical_embeddings = bool(summary.get("schedule", {}).get("identical_token_embeddings", False))
    groups, skipped, seen_ids = {}, [], set()
    cache = {}

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else root / path

    def checkpoint(value):
        path = resolve(value)
        if path not in cache:
            cache[path] = read_checkpoint(path, layout)
        return cache[path]

    for original in summary.get("trials", []):
        trial = dict(original)
        if trial["id"] in seen_ids:
            raise ValueError("trial IDs must be unique")
        seen_ids.add(trial["id"])
        if (trial.get("status") not in ("verified", "not_memorized") or
                not trial.get("final_checkpoint") or not trial.get("initial_checkpoint")):
            skipped.append({"id": trial["id"], "reason": "no finalized verified checkpoint pair"})
            continue
        if trial.get("split") not in ("train", "test", "control"):
            raise ValueError("preassigned train/test/control split is required")
        fingerprint = trial.get("training_fingerprint", summary.get("training_fingerprint"))
        if not fingerprint:
            raise ValueError("training_fingerprint is required for comparability")
        if type(trial.get("step")) is not int or trial["step"] <= 0:
            raise ValueError("positive integer endpoint step is required")
        if type(trial.get("success")) is not bool:
            raise ValueError("endpoint success must be explicit bool")
        trial["tokens"], lengths = read_tokens(resolve(trial["token_corpus"]))
        if trial["tokens"].max() >= vocabulary:
            raise ValueError("corpus token outside model vocabulary")
        trial["initial"], initial_hash = checkpoint(trial["initial_checkpoint"])
        if identical_embeddings:
            embedding = trial["initial"][:layout[0].stop].reshape(layout[0].shape)
            if not np.all(embedding == embedding[0]):
                raise ValueError("identical-token-embedding experiment has unequal initial rows")
        trial["weights"], trial["final_sha256"] = checkpoint(trial["final_checkpoint"])
        if trial.get("parameters", layout[-1].stop) != layout[-1].stop:
            raise ValueError("trial parameter count disagrees with model")
        trial["permutation_values"] = (_read_permutation(resolve(trial["permutation"]), vocabulary)
                                        if trial.get("permutation") else None)
        key = (trial["family"], trial["step"], initial_hash, str(fingerprint), lengths)
        groups.setdefault(key, []).append(trial)

    report = {"model": summary["model"], "parameters": layout[-1].stop,
              "source_summary": str(summary_path), "skipped": skipped, "groups": [],
              "identical_token_embeddings": identical_embeddings,
              "baseline_memorization_verified": summary.get("baseline_memorization_verified"),
              "limitations": [
                  "A few dozen samples cannot identify an unrestricted function from datasets to weights.",
                  "Vocabulary renamings preserve every sentence's structure; this does not test learning arbitrary new facts.",
                  "Token IDs are categorical; Hamming distance ignores the numerical label values.",
                  "All regression metrics concern training deltas, not raw weights dominated by initialization.",
                  "Only predetermined held-out endpoints test generalization; tuning uses training runs only.",
                  ("Every initial token-embedding row, including EOS, is identical and checked; "
                   "vocabulary renaming therefore preserves initialization. Finite-precision "
                   "reductions can still break exact training-trajectory equivariance."
                   if identical_embeddings else
                   "Same seed with renamed data is not permutation-equivariant initialization."),
                  "Low parameter error does not establish correct generated completions; native evaluation is separate.",
                  "Observed low rank is bounded by the number of training runs, not an intrinsic-dimension proof.",
                  "All, memorized-only, and unmemorized-only cohorts are reported separately; endpoint selection is observational.",
                  "Memorized cohorts refer to the fixed endpoint. First reaching perfect accuracy "
                  "and retaining it through the endpoint are separate observations.",
              ]}
    for key, trials in groups.items():
        family, step, initial_hash, fingerprint, _ = key
        group = {"family": family, "step": step, "initial_sha256": initial_hash,
                 "training_fingerprint": fingerprint, "cohorts": [], "controls": [],
                 "baseline_differences": []}
        identities, noncontrol_identities = {}, set()
        for trial in trials:
            token_hash = hashlib.sha256(trial["tokens"].tobytes()).hexdigest()
            previous = identities.get(token_hash)
            if trial["split"] != "control":
                if token_hash in noncontrol_identities:
                    raise ValueError("duplicate dataset across non-control trials: split leakage/reweighting")
                noncontrol_identities.add(token_hash)
            if previous:
                difference = trial["weights"] - previous["weights"]
                group["controls"].append({"left": previous["id"], "right": trial["id"],
                                           "identical_weights": previous["final_sha256"] == trial["final_sha256"],
                                           "max_abs_difference": float(np.max(np.abs(difference)))})
            else:
                identities[token_hash] = trial
        baseline = next((trial for trial in trials if trial["id"] == "baseline"), None)
        if baseline:
            for trial in trials:
                permutation = trial["permutation_values"]
                if permutation is not None and not np.array_equal(
                        trial["tokens"], permutation[baseline["tokens"]]):
                    raise ValueError("corpus does not match declared token permutation")
                differences = {"id": trial["id"], "success": trial["success"],
                               "errors": trial.get("errors"),
                               **memorization_summary(trial),
                               "symmetry_errors": trial.get("symmetry_errors"),
                               "token_fraction_changed": float(np.mean(
                                   trial["tokens"] != baseline["tokens"])),
                               **{f"raw_{name}": value for name, value in
                                  weight_difference(baseline["weights"], trial["weights"]).items()}}
                if permutation is not None:
                    aligned = align_embeddings(trial["weights"], permutation, layout)
                    differences.update({f"embedding_aligned_{name}": value for name, value in
                                        weight_difference(baseline["weights"], aligned).items()})
                    differences["tensors"] = [{"name": tensor.name, "shape": tensor.shape,
                        **{f"raw_{name}": value for name, value in weight_difference(
                            baseline["weights"][tensor.start:tensor.stop],
                            trial["weights"][tensor.start:tensor.stop]).items()},
                        **{f"embedding_aligned_{name}": value for name, value in weight_difference(
                            baseline["weights"][tensor.start:tensor.stop],
                            aligned[tensor.start:tensor.stop]).items()}}
                        for tensor in layout]
                group["baseline_differences"].append(differences)

        for cohort, select in (("all", lambda trial: True),
                               ("memorized", lambda trial: trial["success"]),
                               ("not_memorized", lambda trial: not trial["success"])):
            train = [trial for trial in trials if trial["split"] == "train" and select(trial)]
            test = [trial for trial in trials if trial["split"] == "test" and select(trial)]
            entry = {"cohort": cohort, "training_ids": [t["id"] for t in train],
                     "heldout_ids": [t["id"] for t in test], "representations": {}}
            if len(train) < 2 or not test:
                entry["status"] = "insufficient comparable train/held-out endpoints"
                group["cohorts"].append(entry)
                continue
            entry["status"] = "evaluated"
            for representation in ("raw", "embedding_aligned"):
                if representation == "embedding_aligned" and any(
                        trial["permutation_values"] is None for trial in train + test):
                    continue

                def output(trial):
                    delta = trial["weights"] - trial["initial"]
                    # Align the whole delta, including its initialization. This
                    # avoids crediting a predictor for a known initialization
                    # row permutation rather than predicting training updates.
                    return (align_embeddings(delta, trial["permutation_values"], layout)
                            if representation == "embedding_aligned" else delta)

                train_outputs = np.array([output(t) for t in train])
                test_outputs = np.array([output(t) for t in test])
                predictions, settings = fit_predict(np.array([t["tokens"] for t in train]),
                                                     train_outputs,
                                                     np.array([t["tokens"] for t in test]))
                if baseline and any(trial["id"] == baseline["id"] for trial in train):
                    guesses = []
                    for trial in test:
                        permutation = trial["permutation_values"]
                        if permutation is None:
                            break
                        # Inverse alignment relabels the final baseline weights.
                        final = align_embeddings(baseline["weights"],
                                                  np.argsort(permutation), layout)
                        guess = final - trial["initial"]
                        if representation == "embedding_aligned":
                            guess = align_embeddings(guess, permutation, layout)
                        guesses.append(guess)
                    if len(guesses) == len(test):
                        predictions["permuted_baseline_final"] = np.array(guesses)
                entry["representations"][representation] = {
                    "settings": settings,
                    "training_delta_spectrum": _spectral_summary(train_outputs),
                    "predictions": _summarize_predictions(predictions, test_outputs, test, layout)}
            group["cohorts"].append(entry)
        report["groups"].append(group)
    return report


def render_html(report):
    """A self-contained report with expandable per-tensor, per-heldout results."""
    escape = lambda text: html.escape(str(text))

    def number(value):
        return "undefined (zero norm)" if value is None else f"{value:.6g}"

    parts = ["<!doctype html><meta charset='utf-8'><title>Dataset to weights</title>",
             "<style>body{font:16px system-ui;max-width:1300px;margin:2em auto;padding:1em}"
             "table{border-collapse:collapse;margin:1em 0;width:100%}th,td{padding:.4em;text-align:left;"
             "border:1px solid #bbb}h2{margin-top:2em}details{margin:.6em 0}code{overflow-wrap:anywhere}"
             "</style><h1>Dataset → trained weights: held-out analysis</h1>",
             f"<p>{report['parameters']:,} FP32 parameters. Source: <code>"
             f"{escape(report['source_summary'])}</code>.</p>",
             "<p>Normalized RMSE = ‖predicted training delta − actual training delta‖₂ / "
             "‖actual training delta‖₂. Lower is better; 1 equals predicting no learning. "
             "Cosine compares delta direction. Embedding alignment undoes the known token "
             "renaming; it does not align arbitrary hidden-unit symmetries.</p><ul>"]
    parts.extend(f"<li>{escape(item)}</li>" for item in report["limitations"])
    parts.append("</ul>")
    if report["baseline_memorization_verified"] is not None:
        parts.append("<p>Baseline memorization independently verified before permutations: "
                     f"{escape(report['baseline_memorization_verified'])}.</p>")
    for group in report["groups"]:
        parts.append(f"<h2>{escape(group['family'])}, step {group['step']:,}</h2>")
        for control in group["controls"]:
            parts.append(f"<p>Determinism control {escape(control['left'])} / "
                         f"{escape(control['right'])}: byte-equivalent finite weights "
                         f"{control['identical_weights']}; maximum difference "
                         f"{control['max_abs_difference']:.6g}.</p>")
        if group["baseline_differences"]:
            parts.append("<h3>Differences from the trained identity-corpus baseline</h3>"
                         "<p>These are measured weight distances, not prediction/generalization scores. "
                         "Symmetry errors evaluate an exactly row-permuted baseline checkpoint, "
                         "not weights learned from the renamed corpus. Relative aligned L2 divides "
                         "by the norm of the baseline's final weights. Exact changes count unequal "
                         "numeric values; the threshold count requires an absolute change greater "
                         "than 0.001.</p><table><tr><th>Run</th><th>Fixed endpoint memorized</th>"
                         "<th>First perfect step</th><th>Memorization checkpoint audited</th>"
                         "<th>Token positions changed</th><th>Raw weight L2</th>"
                         "<th>Aligned weight L2</th><th>Relative aligned L2</th>"
                         "<th>Aligned exact changes</th><th>Aligned changes &gt; 0.001</th>"
                         "<th>Symmetry errors</th></tr>")
            for trial in group["baseline_differences"]:
                parts.append(f"<tr><td>{escape(trial['id'])}</td><td>{trial['success']}</td>"
                             f"<td>{escape(trial['first_memorized_step'])}</td>"
                             f"<td>{trial['memorization_independently_verified']}</td>"
                             f"<td>{trial['token_fraction_changed']:.2%}</td>"
                             f"<td>{trial['raw_l2']:.6g}</td>"
                             f"<td>{number(trial.get('embedding_aligned_l2'))}</td>"
                             f"<td>{number(trial.get('embedding_aligned_relative_l2'))}</td>"
                             f"<td>{escape(trial.get('embedding_aligned_changed_parameters'))}</td>"
                             f"<td>{escape(trial.get('embedding_aligned_abs_gt_1e3_count'))}</td>"
                             f"<td>{escape(trial['symmetry_errors'])}</td></tr>")
            parts.append("</table>")
            parts.append("<details><summary>Weight differences by tensor</summary>")
            for trial in group["baseline_differences"]:
                parts.append(f"<details><summary>{escape(trial['id'])}</summary><table>"
                             "<tr><th>Tensor</th><th>Shape</th><th>Raw L2 difference</th>"
                             "<th>Aligned L2 difference</th><th>Relative aligned L2</th>"
                             "<th>Raw exact changes</th><th>Aligned exact changes</th>"
                             "<th>Raw changes &gt; 0.001</th><th>Aligned changes &gt; 0.001</th></tr>")
                for tensor in trial.get("tensors", []):
                    parts.append(f"<tr><td>{escape(tensor['name'])}</td>"
                                 f"<td>{escape(tensor['shape'])}</td>"
                                 f"<td>{tensor['raw_l2']:.6g}</td>"
                                 f"<td>{tensor['embedding_aligned_l2']:.6g}</td>"
                                 f"<td>{number(tensor['embedding_aligned_relative_l2'])}</td>"
                                 f"<td>{tensor['raw_changed_parameters']}</td>"
                                 f"<td>{tensor['embedding_aligned_changed_parameters']}</td>"
                                 f"<td>{tensor['raw_abs_gt_1e3_count']}</td>"
                                 f"<td>{tensor['embedding_aligned_abs_gt_1e3_count']}</td></tr>")
                parts.append("</table></details>")
            parts.append("</details>")
        for cohort in group["cohorts"]:
            parts.append(f"<h3>{escape(cohort['cohort'])}: {len(cohort['training_ids'])} training, "
                         f"{len(cohort['heldout_ids'])} held-out runs</h3>")
            if cohort["status"] != "evaluated":
                parts.append(f"<p>{escape(cohort['status'])}</p>")
                continue
            for representation, result in cohort["representations"].items():
                spectrum = result["training_delta_spectrum"]
                parts.append(f"<h4>{escape(representation)}</h4><p>Training-only settings: "
                             f"<code>{escape(result['settings'])}</code></p><p>Centered training-delta "
                             f"rank needed for 90% / 99% energy: {spectrum['rank_90_percent']} / "
                             f"{spectrum['rank_99_percent']} (sample-count ceiling "
                             f"{spectrum['max_possible_centered_rank']}).</p>"
                             "<table><tr><th>Predictor</th><th>Held-out normalized RMSE</th>"
                             "<th>Held-out cosine</th></tr>")
                for name, prediction in result["predictions"].items():
                    metric = prediction["overall"]
                    parts.append(f"<tr><td>{escape(name)}</td><td>{number(metric['normalized_rmse'])}"
                                 f"</td><td>{number(metric['cosine'])}</td></tr>")
                parts.append("</table>")
                for name, prediction in result["predictions"].items():
                    parts.append(f"<details><summary>{escape(name)}: per-run and tensor detail</summary>")
                    for trial in prediction["trials"]:
                        parts.append(f"<details><summary>{escape(trial['id'])}, fixed endpoint memorized="
                                     f"{trial['success']}, errors={escape(trial['errors'])}, "
                                     f"normalized RMSE={number(trial['overall']['normalized_rmse'])}"
                                     "</summary><table><tr><th>Tensor</th><th>Shape</th>"
                                     "<th>Normalized RMSE</th><th>Cosine</th></tr>")
                        for tensor in trial["tensors"]:
                            parts.append(f"<tr><td>{escape(tensor['name'])}</td><td>"
                                         f"{escape(tensor['shape'])}</td><td>"
                                         f"{number(tensor['normalized_rmse'])}</td><td>"
                                         f"{number(tensor['cosine'])}</td></tr>")
                        parts.append("</table></details>")
                    parts.append("</details>")
    if report["skipped"]:
        parts.append(f"<h2>Incomplete endpoints skipped</h2><pre>{escape(report['skipped'])}</pre>")
    return "\n".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="self-contained HTML report")
    args = parser.parse_args(argv)
    report = analyze(args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(report))
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output} and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
