"""Freeze token-row candidates from candidate restart checkpoint deltas.

No corpus, tokenizer, model forward, backward, or optimizer is used. Ranked
IDs are UNORDERED token candidates. The timestamp gaps are not authenticated
restarts and the fitted geometric coefficient is not known AdamW decay.
"""

import argparse
import datetime
import os
from pathlib import Path
import time

import numpy as np

from .checkpoint import GPT2Config
from .late_mlp_paths import file_record, write_exclusive


BOUNDARIES = (580, 1780, 4280, 7230, 8160)
SEED = 20260909
TOP_K = 128
VARIANTS = ("raw", "adjusted")
CENTERS = {"ordinary_before": 1, "boundary": 3, "ordinary_after": 5}


def checkpoint_steps(boundary):
    if type(boundary) is not int or boundary < 40 or boundary % 10:
        raise ValueError("boundary must be a multiple of ten >=40")
    return list(range(boundary - 30, boundary + 41, 10))


def finite_matrix(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < 1 or not np.isfinite(value).all():
        raise ValueError("expected finite nonempty matrix")
    return value


def activities(before, after):
    """Row L2 changes, with a separate translation/radial-fit diagnostic.

    Only logical vocabulary rows may be passed. A fit to all physical rows
    would give unused embedding padding an unintended statistical role.
    All arithmetic is FP64 on the stored FP32 endpoints, not gradient recovery.
    """
    before, after = finite_matrix(before), finite_matrix(after)
    if before.shape != after.shape:
        raise ValueError("endpoint shapes differ")
    delta = after - before
    raw = np.linalg.norm(delta, axis=1)
    mean = delta.mean(axis=0)
    centered_before = before - before.mean(axis=0)
    residual = delta - mean
    denominator = float(np.sum(centered_before * centered_before))
    coefficient = float(np.sum(centered_before * residual) / denominator) if denominator else 0.
    residual -= coefficient * centered_before
    adjusted = np.linalg.norm(residual, axis=1)
    if not np.isfinite(raw).all() or not np.isfinite(adjusted).all() or not np.isfinite(coefficient):
        raise ValueError("nonfinite activity arithmetic")
    return {"raw": raw, "adjusted": adjusted}, {
        "shape": list(before.shape), "shared_translation": mean.tolist(),
        "fitted_radial_coefficient_not_authenticated_decay": coefficient,
        "raw_squared_norm": float(np.sum(raw * raw)),
        "adjusted_squared_norm": float(np.sum(adjusted * adjusted)),
        "centered_before_squared_norm": denominator,
    }


def log_activity(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError("activity must be finite and nonnegative")
    base = float(np.median(value))
    if base == 0:
        base = float(value.max())
    if base == 0:
        return np.zeros_like(value)
    floor = base * 1e-12
    if floor == 0:
        raise ValueError("activity scale too small for the declared FP64 floor")
    # Subtract logs instead of dividing first, avoiding an unnecessary ratio
    # overflow when testing well-separated but finite activity magnitudes.
    return np.log(np.maximum(value, floor)) - np.log(base)


def transient_scores(intervals):
    intervals = np.asarray(intervals, dtype=np.float64)
    if intervals.ndim != 2 or intervals.shape[0] != 7:
        raise ValueError("need exactly seven consecutive interval activities")
    logs = np.stack([log_activity(row) for row in intervals])
    return {name: logs[center] - (logs[center - 1] + logs[center + 1]) / 2
            for name, center in CENTERS.items()}


def percentile_midrank(values):
    """Ascending percentile midranks: identical values get identical scores."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("midranks require a finite vector")
    order = np.argsort(values, kind="stable")
    _, first, counts = np.unique(values[order], return_index=True, return_counts=True)
    ordered = np.repeat((first + counts / 2) / len(values), counts)
    result = np.empty_like(values)
    result[order] = ordered
    return result


def ranked_candidates(scores, permutation, k=TOP_K):
    scores = np.asarray(scores, dtype=np.float64)
    permutation = np.asarray(permutation)
    if (scores.ndim != 1 or not np.isfinite(scores).all() or type(k) is not int
            or not 0 < k <= len(scores) or permutation.dtype.kind not in "iu"
            or permutation.shape != scores.shape
            or not np.array_equal(np.sort(permutation), np.arange(len(scores)))):
        raise ValueError("invalid scores, K, or identity permutation")
    ids = np.lexsort((np.arange(len(scores)), -scores))[:k]
    return {"token_ids": ids.tolist(), "scores": scores[ids].tolist(),
            "identity_shuffled_token_ids": permutation[ids].tolist(),
            "interpretation": "unordered token pieces, not a generated sequence"}


def write_npz(path, **arrays):
    with Path(path).open("xb") as output:
        np.savez(output, **arrays)
    return file_record(path)


def run(checkpoint_root, output_dir, protocol, inventory):
    from .checkpoint_archive import read_embedding

    start = time.monotonic()
    checkpoint_root = Path(checkpoint_root).resolve()
    requested_output = Path(output_dir).absolute()
    if requested_output.exists() or requested_output.is_symlink():
        raise FileExistsError(requested_output)
    output = requested_output.resolve()
    if output == checkpoint_root or checkpoint_root in output.parents:
        raise ValueError("outputs must be outside checkpoint root")
    source_dir = Path(__file__).resolve().parent
    sources = {name: file_record(source_dir / name) for name in
               ("restart_delta.py", "checkpoint_archive.py", "checkpoint.py", "late_mlp_paths.py")}
    sources["protocol"] = file_record(protocol)
    sources["history_inventory"] = file_record(inventory)
    paths = {}
    for boundary in BOUNDARIES:
        for step in checkpoint_steps(boundary):
            candidates = [checkpoint_root / f"step_{step}.tar.gz", checkpoint_root / f"step_{step}"]
            present = [path for path in candidates if path.exists() or path.is_symlink()]
            if len(present) != 1:
                raise ValueError(f"missing or ambiguous checkpoint {step}")
            paths[step] = present[0]
    output.mkdir()
    write_exclusive(output / "plan.json", {
        "created_utc_before_weight_reads": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sources": sources, "boundaries": list(BOUNDARIES), "checkpoint_paths": {str(k): str(v) for k, v in paths.items()},
        "logical_vocab_size": GPT2Config().vocab_size, "seed": SEED, "top_k": TOP_K,
        "stage": "weight_only_candidate_plan_no_corpus_or_model",
    })
    vocabulary = GPT2Config().vocab_size
    permutation = np.random.Generator(np.random.PCG64(SEED)).permutation(vocabulary)
    candidates, scores, evidence, files, provenance, endpoint_norms = {}, {}, [], {}, {}, []
    collected = {variant: {kind: [] for kind in CENTERS} for variant in VARIANTS}
    for boundary in BOUNDARIES:
        sequence = checkpoint_steps(boundary)
        vectors = {variant: [] for variant in VARIANTS}
        previous = None
        for step in sequence:
            embedding, record = read_embedding(paths[step], step)
            provenance[str(step)] = record
            logical = embedding[:vocabulary]
            if step == boundary:
                norm = np.linalg.norm(logical.astype(np.float64), axis=1)
                endpoint_norms.append(norm)
                key = f"endpoint_norm_{boundary}"
                scores[key] = norm
                candidates[key] = ranked_candidates(norm, permutation)
            if previous is not None:
                interval_vectors, summary = activities(previous[:vocabulary], logical)
                for variant in VARIANTS:
                    vectors[variant].append(interval_vectors[variant])
                key = f"interval_{step - 10}_{step}"
                files[key] = write_npz(output / (key + ".npz"), **interval_vectors)
                evidence.append({"before_step": step - 10, "after_step": step, **summary, "activity_file": key})
            previous = embedding
            print(f"Read step {step}; {len(provenance)}/{len(paths)} checkpoints, no corpus access", flush=True)
        for variant in VARIANTS:
            for kind, score in transient_scores(vectors[variant]).items():
                key = f"{kind}_{variant}_{boundary}"
                scores[key] = score
                collected[variant][kind].append(score)
                candidates[key] = ranked_candidates(score, permutation)
    for variant in VARIANTS:
        for kind in CENTERS:
            components = np.asarray(collected[variant][kind])
            score = np.mean([percentile_midrank(row) for row in components], axis=0)
            key = f"shared_{kind}_{variant}"
            scores[key] = score
            candidates[key] = ranked_candidates(score, permutation)
            ids = candidates[key]["token_ids"]
            candidates[key]["positive_component_count"] = (components[:, ids] > 0).sum(axis=0).tolist()
            candidates[key]["component_scores"] = components[:, ids].tolist()
    score = np.mean([percentile_midrank(row) for row in endpoint_norms], axis=0)
    scores["shared_endpoint_norm"] = score
    candidates["shared_endpoint_norm"] = ranked_candidates(score, permutation)
    files["complete_scores"] = write_npz(output / "scores.npz", **scores, identity_permutation=permutation)
    for record in provenance.values():
        if record.get("input_files_unchanged_during_read") is not True:
            raise ValueError("checkpoint reader did not verify input integrity")
        for item in record["input_files"]:
            if file_record(item["path"]) != item:
                raise ValueError("checkpoint input changed before completion")
    for record in sources.values():
        if file_record(record["path"]) != record:
            raise ValueError("analysis source changed during extraction")
    write_exclusive(output / "candidates.json", {
        "stage": "frozen_unordered_token_candidates_from_weight_deltas",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "top_k": TOP_K, "candidate_lists": candidates,
        "no_corpus_or_tokenizer_access": True, "no_model_execution": True,
    })
    files["candidates"] = file_record(output / "candidates.json")
    files["plan"] = file_record(output / "plan.json")
    for record in files.values():
        if file_record(record["path"]) != record:
            raise ValueError("output evidence changed before freezing")
    result = {"schema_version": 1, "complete": True,
              "stage": "checkpoint_delta_token_candidate_freeze_not_sequence_recovery",
              "frozen_utc_before_corpus_verification": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "sources": sources, "files": files, "checkpoint_provenance": provenance,
              "intervals": evidence, "elapsed_seconds": time.monotonic() - start,
              "runtime_environment": {"numpy": np.__version__,
                                      "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                                      "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
              "limitations": ["Timestamp gaps are not authenticated restarts.",
                              "Activity/transient scores are not recovered raw gradients.",
                              "Fitted radial subtraction is not authenticated AdamW decay.",
                              "Ranked token IDs do not specify word order or historical batch membership."]}
    write_exclusive(output / "frozen.json", result)
    print(f"Frozen {len(candidates)} real/control lists plus their identity shuffles; no corpus opened.", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint-root", "output-dir", "protocol", "inventory"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.checkpoint_root, args.output_dir, args.protocol, args.inventory)


if __name__ == "__main__":
    main()
