"""Freeze and optimize a weight-only late-MLP polynomial, not a model forward.

All token positions are optimized jointly by coordinate ascent over a compiled
degree-three score. Only fixed coefficient contractions see candidate IDs.
No corpus, prompt, attention, or candidate-dependent GELU/LayerNorm is used.
Choosing the last two MLPs was informed by the separate corpus-assisted causal
experiment. That choice is not an independent, corpus-blind discovery.

One exclusive plan is written before compilation/search, followed by every
restart's final candidate and a completion record. Repetitions and duplicates
are retained. A separate verifier must check these frozen candidates.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from .checkpoint import GPT2Checkpoint, sha256_file
from .late_mlp_polynomial import compile_polynomial
from .path_diagnostics import analyze as analyze_structure


PROTOCOL = (Path(__file__).resolve().parents[2] /
            "ai-slop/research/weight_memorization/LATE_MLP_POLYNOMIAL_PROTOCOL.md")
MODES = ("full", "affine", "no_cross", "broken")


def json_value(value):
    """Convert numerical containers without hiding NaN/Inf serialization errors."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def encoded(value):
    return json.dumps(json_value(value), indent=2, allow_nan=False) + "\n"


def write_exclusive(path, value):
    text = encoded(value)
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(text)


def array_record(array, dtype, chunk_bytes=8 * 1024 * 1024):
    """Hash canonical little-endian C-order bytes without a whole-array copy.

    Compiled dictionaries can be several GB. Native contiguous arrays are
    hashed through a memoryview, not tobytes(). A strided array or one needing
    a dtype/byte-order conversion is traversed in bounded C-order chunks.
    Equal logical arrays therefore have identical records regardless of source
    strides, contiguity, or endianness. Shape and dtype accompany the raw-byte
    digest so reshapes cannot silently pass coefficient-integrity checks.
    """
    array = np.asarray(array)
    canonical = np.dtype(dtype).newbyteorder("<")
    if (canonical.kind not in "iuf" or canonical.hasobject
            or type(chunk_bytes) is not int or chunk_bytes <= 0):
        raise ValueError("array hashing requires a numerical dtype and positive chunk size")
    digest = hashlib.sha256()
    if array.size:
        if array.flags.c_contiguous and array.dtype == canonical:
            raw = memoryview(array).cast("B")
            for first in range(0, len(raw), chunk_bytes):
                digest.update(raw[first:first + chunk_bytes])
        else:
            # buffered+external_loop bounds temporary storage for transposes,
            # negative strides, and non-native byte order.
            iterator = np.nditer(
                array, flags=["external_loop", "buffered", "zerosize_ok"],
                op_flags=["readonly"], op_dtypes=[canonical], order="C",
                casting="unsafe", buffersize=max(1, chunk_bytes // canonical.itemsize))
            for chunk in iterator:
                contiguous = np.ascontiguousarray(chunk, dtype=canonical)
                digest.update(memoryview(contiguous).cast("B"))
    return {"dtype": canonical.str, "shape": list(array.shape),
            "bytes": int(array.size) * canonical.itemsize, "sha256": digest.hexdigest()}


def array_hash(array, dtype):
    return array_record(array, dtype)["sha256"]


def compiled_array_records(probe):
    """Capture all coefficient dictionaries, including gates and logical IDs."""
    return {name: array_record(getattr(probe, name), "<i8" if name == "token_ids" else "<f8")
            for name in ("token_ids", "direct", "features", "readout",
                         "neuron_readout", "g7", "h7")}


def verify_compiled_arrays(probe, before):
    after = compiled_array_records(probe)
    if after != before:
        raise ValueError("compiled polynomial coefficients changed during optimization")
    return after


def runtime_environment():
    """Record runtime identity and requested BLAS/OpenMP thread settings.

    Environment variables are requested settings, not a measurement of actual
    library thread-pool sizes. Unset variables are recorded as null.
    """
    return {"python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "python_executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(), "machine": platform.machine(),
            "environment": {name: os.environ.get(name)
                            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}}


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def select_vocabulary(embedding, size):
    """Full logical-vocabulary centroid distance, never corpus frequencies.

    Center using ALL logical rows, not the selected subset. Distances match
    the earlier frozen centroid selector; both sides of an early/final
    comparison receive the same final-trained vocabulary prior explicitly.
    """
    embedding = np.asarray(embedding, dtype=np.float64)
    if (embedding.ndim != 2 or not all(embedding.shape)
            or not np.isfinite(embedding).all()
            or type(size) is not int or not 0 < size <= len(embedding)):
        raise ValueError("invalid finite logical embedding or vocabulary size")
    mean = embedding.mean(axis=0)
    scores = np.linalg.norm(embedding - mean, axis=1)
    if not np.isfinite(scores).all():
        raise ValueError("vocabulary geometry exceeded finite FP64 range")
    ranked = np.lexsort((np.arange(len(scores)), -scores))[:size]
    return np.sort(ranked), {
        "method": "final:centroid_distance", "ranked_ids": ranked,
        "ranked_scores": scores[ranked], "sorted_ids": np.sort(ranked),
        "full_logical_score_sha256": array_hash(scores, "<f8"),
        "full_logical_centroid_sha256": array_hash(mean, "<f8"),
        "rule": "descending norm(E[t]-mean_ALL_LOGICAL(E)); ascending ID ties",
    }


def initial_strings(ids, restarts, length, seed):
    ids = np.asarray(ids)
    if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu"
            or np.any(ids < 0) or np.any(ids[1:] <= ids[:-1])
            or any(type(x) is not int or x <= 0 for x in (restarts, length))
            or type(seed) is not int or seed < 0):
        raise ValueError("invalid sorted IDs, initial-string dimensions, or seed")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(ids), size=(restarts, length), dtype=np.int64)
    return ids[indices].astype(np.int64)


def mixed_order_audit(probe):
    """Inspect ordered mixed coefficients, not whole-sequence reversal accuracy.

    A scalar Hessian alone is symmetric in its source vectors. Different
    frozen position transports can break source-CONTENT swap symmetry. Check
    fixed small index sets rather than assume distinct transports suffice.
    Position-independent features give zero antisymmetry, but the entire
    prefix/target objective can still distinguish sequence order.
    """
    source_count = min(8, len(probe.token_ids))
    target_count = min(16, len(probe.token_ids))
    length = probe.direct.shape[0]
    if length < 3:
        return {"source_ids": [], "target_ids": [], "pairs": []}
    pairs = [(0, 1, 2)]
    if length > 3:
        pairs.append((0, length - 2, length - 1))
    targets = probe.neuron_readout[:target_count] * probe.h7
    results = []
    for left, right, target_position in pairs:
        products = (probe.features[left, :source_count, None, :]
                    * probe.features[right, None, :source_count, :])
        mixed = np.einsum("abm,um->abu", products, targets,
                          optimize=True) * (probe.alpha / target_position) ** 2
        swapped = mixed.transpose(1, 0, 2)
        difference = mixed - swapped
        denominator = np.abs(mixed) + np.abs(swapped)
        relative = np.zeros_like(denominator)
        np.divide(np.abs(difference), denominator, out=relative,
                  where=denominator > 0)
        if not all(np.isfinite(x).all() for x in (mixed, difference, relative)):
            raise ValueError("mixed-order audit exceeded finite FP64 range")
        results.append({
            "source_positions": [left, right], "target_position": target_position,
            "mixed_coefficients": mixed,
            "maximum_absolute_coefficient": float(np.abs(mixed).max()),
            "maximum_absolute_content_swap_difference": float(np.abs(difference).max()),
            "maximum_relative_content_swap_difference": float(relative.max()),
            "zero_denominator_cases": int(np.count_nonzero(denominator == 0)),
            "tested_source_pairs": source_count ** 2,
            "tested_targets": target_count,
        })
    return {"source_ids": probe.token_ids[:source_count],
            "target_ids": probe.token_ids[:target_count], "pairs": results,
            "scope": "fixed polynomial mixed coefficients; ordinary FP64, not an interval proof or model routing measurement"}


def extract(checkpoint_path, selection_checkpoint_path, tokenizer_dir, output,
            *, mode="full", size=8192, length=16, restarts=64, sweeps=8,
            seed=20260909, alpha=0.125, label=None):
    started = time.monotonic()
    output = Path(output)
    plan_path = output.with_suffix(output.suffix + ".plan.json")
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    for path in (output, plan_path, metadata_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    if (mode not in MODES or type(sweeps) is not int or sweeps <= 0
            or not np.isfinite(alpha) or alpha <= 0):
        raise ValueError("invalid mode, sweeps, or positive finite amplitude")
    checkpoint_path = Path(checkpoint_path).resolve()
    selection_checkpoint_path = Path(selection_checkpoint_path).resolve()
    for source in (checkpoint_path, selection_checkpoint_path):
        if output.resolve().is_relative_to(source):
            raise ValueError("outputs must not be inside a checkpoint directory")
    checkpoint = GPT2Checkpoint(checkpoint_path, check_finite=True)
    selection_checkpoint = (checkpoint if selection_checkpoint_path == checkpoint_path
                            else GPT2Checkpoint(selection_checkpoint_path, check_finite=True))
    if checkpoint.config != selection_checkpoint.config:
        raise ValueError("evaluation and selection architectures differ")
    if not 2 <= length <= checkpoint.config.context_length:
        raise ValueError("phrase length outside checkpoint context")
    ids, selection = select_vocabulary(selection_checkpoint.token_embedding, size)
    initial = initial_strings(ids, restarts, length, seed)
    tokenizer_path = Path(tokenizer_dir) / "tokenizer.json"
    tokenizer_record = file_record(tokenizer_path)
    vocabulary = json.loads(tokenizer_path.read_text())["model"]["vocab"]
    if (len(vocabulary) != checkpoint.config.vocab_size
            or set(vocabulary.values()) != set(range(checkpoint.config.vocab_size))
            or any(type(value) is not int for value in vocabulary.values())):
        raise ValueError("token labels do not match logical vocabulary")
    labels = {value: key for key, value in vocabulary.items()}
    source_paths = {name: Path(__file__).with_name(name) for name in
                    ("late_mlp_paths.py", "late_mlp_polynomial.py", "checkpoint.py",
                     "path_diagnostics.py", "verify.py")}
    sources = {name: file_record(path) for name, path in source_paths.items()}
    sources["production_gelu"] = file_record(
        Path(__file__).resolve().parents[2] / "src/llm/layers/gelu.cc")
    sources["tokenizer"] = tokenizer_record
    sources["protocol"] = file_record(PROTOCOL)
    before = checkpoint.provenance(hash_weights=True)
    selection_before = selection_checkpoint.provenance(hash_weights=True)
    parameters = {"mode": mode, "vocabulary_size": size, "length": length,
                  "restarts": restarts, "sweeps": sweeps, "seed": seed,
                  "alpha": alpha, "label": label,
                  "blocks": [checkpoint.config.n_layers - 2, checkpoint.config.n_layers - 1]}
    runtime = runtime_environment()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_exclusive(plan_path, {
        "schema_version": 1, "stage": "frozen_weight_only_polynomial_plan",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": parameters, "checkpoint": before,
        "selection_checkpoint": selection_before, "selection": selection,
        "initial_token_ids": initial, "initial_ids_sha256": array_hash(initial, "<i4"),
        "numpy_version": np.__version__, "rng": "PCG64; integers [0,S) int64 mapped through sorted IDs",
        "runtime_environment": runtime,
        "sources": sources, "output": str(output.resolve()),
        "limitations": [
            "Last-two-MLP selection was informed by a corpus-assisted ablation experiment.",
            "This is a declared virtual-path polynomial, not the Taylor polynomial or execution of GPT-2.",
            "Known vocabulary/position anchors are not contextual late-layer activation statistics.",
            "Early checkpoint is already trained and receives the final-selected vocabulary prior.",
            "No corpus, prompt, or candidate-dependent nonlinear layer execution is used.",
        ],
    })
    plan_before = file_record(plan_path)
    print(f"Frozen plan: {plan_path}", flush=True)
    probe = compile_polynomial(checkpoint, ids, length, mode=mode, alpha=alpha)
    compiled_before = compiled_array_records(probe)
    order_audit = mixed_order_audit(probe)
    print(f"Compiled {mode}; optimizing {restarts} complete strings", flush=True)
    optimized = probe.optimize(initial, sweeps=sweeps)
    compiled_after = verify_compiled_arrays(probe, compiled_before)
    parts = probe.score_components(optimized.token_ids)
    # Verify input identities AFTER computation and BEFORE emitting candidates.
    if (checkpoint.provenance(hash_weights=True) != before
            or selection_checkpoint.provenance(hash_weights=True) != selection_before):
        raise ValueError("checkpoint or production source changed during extraction")
    if any(file_record(record["path"]) != record for record in sources.values()):
        raise ValueError("extractor, protocol, or tokenizer changed during extraction")
    if file_record(plan_path) != plan_before:
        raise ValueError("frozen plan changed during extraction")
    method = "late_mlp_" + (label or mode)
    records = []
    for index, tokens in enumerate(optimized.token_ids):
        records.append({
            "candidate_id": f"{method}:restart{index}", "method": method,
            "token_ids": tokens.tolist(), "vocabulary_labels": [labels[int(t)] for t in tokens],
            "initial_token_ids": initial[index].tolist(),
            "initial_score": float(optimized.initial_scores[index]),
            "score": float(parts["total"][index]),
            "linear_score": float(parts["linear"][index]),
            "quadratic_score": float(parts["quadratic"][index]),
            "score_definition": "fixed signed polynomial, not probability or model logit",
            "provenance": {"plan_file": str(plan_path.resolve()), "restart": index},
        })
    # Serialize completely before opening, then use exclusive creation.
    serialized = "".join(json.dumps(record, allow_nan=False) + "\n" for record in records)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    write_exclusive(metadata_path, {
        "schema_version": 1, "stage": "completed_weight_only_polynomial_extraction",
        "complete": True, "plan": plan_before, "candidates": file_record(output),
        "parameters": parameters, "compiler": probe.metadata,
        "runtime_environment": runtime,
        "compiled_arrays": compiled_before, "compiled_arrays_after": compiled_after,
        "compiled_arrays_unchanged": True,
        "mixed_order_audit": order_audit, "optimization_trace": optimized.trace,
        "candidate_count": len(records),
        "distinct_candidates": len({tuple(record["token_ids"]) for record in records}),
        "candidate_structure": analyze_structure(records, checkpoint.config.vocab_size),
        "final_components": parts, "checkpoint_and_sources_unchanged": True,
        "elapsed_seconds": time.monotonic() - started,
        "search_limit": "coordinate-local search with fixed sweep cap, not a certified global optimum",
    })
    print(encoded({"output": str(output), "candidate_sha256": sha256_file(output),
                   "candidates": len(records), "elapsed_seconds": time.monotonic() - started}), flush=True)
    return metadata_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "selection-checkpoint", "tokenizer-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default="full")
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--restarts", type=int, default=64)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--alpha", type=float, default=0.125)
    parser.add_argument("--label")
    args = parser.parse_args(argv)
    extract(args.checkpoint, args.selection_checkpoint, args.tokenizer_dir, args.output,
            mode=args.mode, size=args.size, length=args.length, restarts=args.restarts,
            sweeps=args.sweeps, seed=args.seed, alpha=args.alpha, label=args.label)


if __name__ == "__main__":
    main()
