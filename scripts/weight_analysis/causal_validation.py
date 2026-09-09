"""Freeze byte-addressed passages and summarize causal forward validation.

This tool is NOT a weight-only extractor. Corpus tokens are explicit model
inputs during the later forward probe. Passage locations, interventions, and
metrics are fixed before losses are observed; neither planner nor reporter
selects examples by difficulty. The current suffix is not an authenticated
historical holdout. Ablation damage measures reliance, not uniquely stored or
independently recoverable memories.

The planner uses the production native export without retokenizing windows.
Its exclusive output directory contains manifest.json and batch_tokens.bin:
little-endian int32 [2, 32, 1024], all inputs first, then all next-token targets.
Loss files are little-endian float32 [32, 1024]. Primary analysis uses indices
512..1023; all 1024 positions form a prespecified secondary analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .verify import gpt2_token_bytes, validate_native_tokens
from .vocabulary_verify import checked_tokens
from .checkpoint import GPT2Checkpoint, sha256_file, tensor_manifest


CONTEXT = 1024
PER_SPLIT = 16
VOCAB_SIZE = 50257
SPLITS = ("training_prefix", "current_suffix")
ARMS = tuple(f"block{block}_{family}_{dose}"
             for block in range(8)
             for family in ("attention", "mlp")
             for dose in ("half", "zero"))
CONTROLS = ("clean_before", "clean_repeat", "clean_after")
CONTROL_ATOL = 1e-6
CONTROL_RTOL = 1e-6


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path),
            "bytes": path.stat().st_size}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique_object,
                      parse_constant=reject_constant)


def write_report(path, report):
    # Serialize before opening, so bad/nonfinite output cannot leave a report
    # that looks successfully completed. Existing evidence is never overwritten.
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as output:
        output.write(serialized)


def split_boundary(corpus):
    """Mirror SplitCorpus(text, 0.1), including its UTF-8 fallback boundary."""
    if not isinstance(corpus, bytes) or not corpus:
        raise ValueError("corpus must be nonempty bytes")
    approximate = int(len(corpus) * (1.0 - 0.1))
    newline = corpus.find(b"\n", approximate)
    boundary = newline + 1 if newline >= 0 else approximate
    if newline < 0:
        while boundary < len(corpus) and corpus[boundary] & 0xC0 == 0x80:
            boundary += 1
    if not 0 < boundary < len(corpus):
        raise ValueError("current 90/10 split does not have two nonempty sides")
    return boundary


def midpoint_starts(token_count, context=CONTEXT, count=PER_SPLIT):
    """Sample deterministic midpoints of equal strata of valid start positions.

    A passage needs context+1 tokens. The final valid start is N-context-1;
    integer arithmetic fixes the rule independently of any RNG or model loss.
    Reject short inputs instead of duplicating or overlapping selected windows.
    The generic parameters exist for small tests; the CLI fixes 1024 and 16.
    """
    if any(type(value) is not int or value <= 0
           for value in (token_count, context, count)):
        raise ValueError("token_count, context, and count must be positive integers")
    maximum = token_count - context - 1
    if maximum < 0:
        raise ValueError("split too short for one passage")
    starts = [((2 * index + 1) * maximum) // (2 * count)
              for index in range(count)]
    if any(second - first < context + 1
           for first, second in zip(starts, starts[1:])):
        raise ValueError("fixed passages would duplicate or overlap")
    return starts


def build_plan(corpus, tokens, offsets, vocab_size=VOCAB_SIZE,
               context=CONTEXT, per_split=PER_SPLIT):
    """Build batches from native IDs; callers validate each token's raw bytes."""
    tokens = checked_tokens(tokens, vocab_size)
    offsets = np.asarray(offsets)
    if (offsets.ndim != 1 or offsets.dtype.kind not in "iu"
            or len(offsets) != len(tokens) + 1 or offsets[0] != 0
            or offsets[-1] != len(corpus)
            or np.any(offsets[1:] <= offsets[:-1])):
        raise ValueError("native offsets must strictly span every corpus token")
    boundary = split_boundary(corpus)
    split_token = int(np.searchsorted(offsets, boundary))
    if split_token >= len(offsets) or offsets[split_token] != boundary:
        raise ValueError("current split is not a native token boundary")
    passages, inputs, targets = [], [], []
    for split, first, last in ((SPLITS[0], 0, split_token),
                               (SPLITS[1], split_token, len(tokens))):
        for index, local in enumerate(midpoint_starts(last - first, context, per_split)):
            start = first + local
            end = start + context + 1
            window = tokens[start:end]
            inputs.append(window[:-1])
            targets.append(window[1:])
            passages.append({
                "passage_id": f"{split}_{index:02d}", "row": len(passages),
                "split": split, "local_token_start": local,
                "token_interval": [start, end],
                "byte_interval": [int(offsets[start]), int(offsets[end])],
                "input_token_interval": [start, end - 1],
                "target_token_interval": [start + 1, end],
                "input_byte_interval": [int(offsets[start]), int(offsets[end - 1])],
                "target_byte_interval": [int(offsets[start + 1]), int(offsets[end])],
                # Consecutive entries address each target's raw bytes exactly;
                # endpoints need not coincide with complete UTF-8 characters.
                "target_byte_offsets": offsets[start + 1:end + 1].tolist(),
                "native_tokens_sha256": digest(window.astype("<i4").tobytes()),
                "token_ids": window.tolist(),
                "corpus_bytes_sha256": digest(corpus[int(offsets[start]):int(offsets[end])]),
            })
    batch = np.asarray([inputs, targets], dtype="<i4")
    manifest = {
        "schema_version": 1, "stage": "causal_forward_validation_plan",
        "vocab_size": vocab_size, "context_length": context,
        "passages_per_split": per_split, "passage_count": len(passages),
        "batch_layout": "little-endian int32 [2, passage_count, context_length]; inputs then targets",
        "split": {"test_fraction": 0.1, "byte_boundary": boundary,
                  "native_token_boundary": split_token,
                  "corpus_tokens": len(tokens), "corpus_bytes": len(corpus),
                  "prefix_tokens": split_token, "suffix_tokens": len(tokens) - split_token},
        "selection": "floor((2*i+1)*(N-context_length-1)/(2*passages_per_split)); no loss-dependent selection",
        "primary_loss_indices": [context // 2, context],
        "secondary_loss_indices": [0, context],
        "arms": list(ARMS), "controls": list(CONTROLS), "passages": passages,
        "limitations": ["Native full-corpus IDs are authoritative; windows are not retokenized.",
                        "The current suffix is not authenticated historical held-out data.",
                        "Forward loss evaluation is validation, not weight-only extraction."],
    }
    return manifest, batch


def plan_files(corpus_path, tokens_path, offsets_path, tokenizer_dir,
               protocol_path, checkpoint_path, binary_path, output_dir):
    """Validate all inputs before exclusively creating either frozen output."""
    from tokenizers import Tokenizer

    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    tokenizer_path = Path(tokenizer_dir) / "tokenizer.json"
    paths = {"corpus": Path(corpus_path), "native_tokens": Path(tokens_path),
             "native_offsets": Path(offsets_path), "tokenizer": tokenizer_path,
             "protocol": Path(protocol_path), "planner": Path(__file__),
             "binary": Path(binary_path),
             "checkpoint_helper": Path(__file__).with_name("checkpoint.py"),
             "native_validation_helper": Path(__file__).with_name("verify.py"),
             "token_validation_helper": Path(__file__).with_name("vocabulary_verify.py")}
    for name in ("causal_probe.h", "causal_probe.cc", "causal_probe_main.cc",
                 "token_argmax.h", "token_argmax.cc", "BUILD.bazel"):
        paths[name] = Path(__file__).with_name(name)
    paths["dataset_split_source"] = Path(__file__).resolve().parents[2] / "src/dataset/dataset.cc"
    records = {name: file_record(path) for name, path in paths.items()}
    if records["native_tokens"]["bytes"] % 4 or records["native_offsets"]["bytes"] % 8:
        raise ValueError("native export has a partial token or offset")
    corpus = paths["corpus"].read_bytes()
    tokens = np.fromfile(paths["native_tokens"], dtype="<u4")
    offsets = np.fromfile(paths["native_offsets"], dtype="<u8")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size(with_added_tokens=True) != VOCAB_SIZE:
        raise ValueError("this fixed protocol requires the GPT-2 logical vocabulary")
    validate_native_tokens(tokens, offsets, corpus, gpt2_token_bytes(tokenizer))
    manifest, batch = build_plan(corpus, tokens, offsets)
    checkpoint = GPT2Checkpoint(Path(checkpoint_path).resolve(), check_finite=True)
    manifest["checkpoint"] = checkpoint.provenance(hash_weights=True)
    # Detect inputs changing between provenance hashing and loading.
    if any(file_record(path) != records[name] for name, path in paths.items()):
        raise ValueError("an input changed while planning")
    manifest["inputs"] = records
    manifest["clean_control_tolerance"] = {"atol": CONTROL_ATOL, "rtol": CONTROL_RTOL,
                                           "argmax_exact": True}
    manifest["native_byte_validation"] = "every token checked against its exact original byte interval"
    batch_bytes = batch.tobytes()
    manifest["batch_file"] = {"filename": "batch_tokens.bin", "sha256": digest(batch_bytes),
                              "bytes": len(batch_bytes), "shape": list(batch.shape), "dtype": "<i4"}
    output_dir.mkdir()  # No parents/exist_ok: fail closed on a reused experiment.
    with (output_dir / "batch_tokens.bin").open("xb") as output:
        output.write(batch_bytes)
    write_report(output_dir / "manifest.json", manifest)
    return manifest


def greedy_prefix_lengths(correct):
    """Count only correct predictions BEFORE the first error in each row.

    Teacher forcing after a mismatch supplies true tokens the model would not
    have generated, so later correct positions are not a recited continuation.
    For primary indices 512..1023 the induction starts with 513 source tokens,
    not 512: logit position 512 has already consumed input token 512.
    """
    correct = np.asarray(correct)
    if correct.ndim != 2 or correct.dtype != np.bool_ or not correct.shape[1]:
        raise ValueError("correct must be a nonempty-width boolean matrix")
    return np.where(correct.all(axis=1), correct.shape[1],
                    np.argmax(~correct, axis=1)).astype(np.int64)


def clean_quartiles(clean_means, passage_ids, splits):
    """Descriptive matching fixed in advance; never rank intervention outcomes.

    Each passage is compared with the other three passages in its clean-NLL
    quartile in the same split. The tiny background is not a significance test
    or evidence for a uniquely stored memory. All raw deltas remain visible.
    """
    means = np.asarray(clean_means, dtype=np.float64)
    if (means.shape != (2 * PER_SPLIT,) or not np.isfinite(means).all()
            or len(passage_ids) != len(means) or len(set(passage_ids)) != len(means)
            or len(splits) != len(means)):
        raise ValueError("quartiles require 32 unique passages and finite clean means")
    backgrounds, groups = [None] * len(means), []
    for split in SPLITS:
        indices = [i for i, name in enumerate(splits) if name == split]
        if len(indices) != PER_SPLIT:
            raise ValueError("quartiles require exactly 16 passages in each split")
        order = sorted(indices, key=lambda i: (means[i], passage_ids[i]))
        for quartile in range(4):
            members = order[quartile * 4:(quartile + 1) * 4]
            groups.append({"split": split, "quartile": quartile,
                           "passage_ids": [passage_ids[i] for i in members]})
            for index in members:
                backgrounds[index] = [other for other in members if other != index]
    return backgrounds, groups


def _array(values, shape, kind, label):
    values = np.asarray(values)
    if values.shape != shape or values.dtype.kind != kind or values.dtype.itemsize != 4:
        raise ValueError(f"{label} must have shape {shape} and 32-bit {kind} elements")
    if kind == "f" and (not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError(f"{label} must contain finite nonnegative losses")
    if kind == "i" and (np.any(values < 0) or np.any(values >= VOCAB_SIZE)):
        raise ValueError(f"{label} contains a non-logical vocabulary ID")
    return values


def summarize(plan, losses, argmax, targets):
    """Paired analysis only; preserve every arm, passage, and raw file upstream."""
    shape = (2 * PER_SPLIT, CONTEXT)
    expected = set(ARMS + CONTROLS)
    if set(losses) != expected or set(argmax) != expected:
        raise ValueError("losses and argmax must contain all 35 distinct arms/controls")
    losses = {name: _array(values, shape, "f", name).astype(np.float64)
              for name, values in losses.items()}
    argmax = {name: _array(values, shape, "i", name) for name, values in argmax.items()}
    targets = _array(targets, shape, "i", "targets")
    passages = plan["passages"]
    ids = [p["passage_id"] for p in passages]
    splits = [p["split"] for p in passages]
    clean = losses["clean_before"]
    controls = {}
    for name in ("clean_repeat", "clean_after"):
        difference = losses[name] - clean
        controls[name] = {"maximum_absolute_loss_difference": float(np.abs(difference).max()),
                          "mean_absolute_loss_difference": float(np.abs(difference).mean()),
                          "loss_values_exact": bool(np.array_equal(losses[name], clean)),
                          "loss_bytes_exact": losses[name].astype("<f4").tobytes() == clean.astype("<f4").tobytes(),
                          "argmax_exact": bool(np.array_equal(argmax[name], argmax["clean_before"]))}
        if (not np.allclose(losses[name], clean, rtol=CONTROL_RTOL, atol=CONTROL_ATOL)
                or not controls[name]["argmax_exact"]):
            raise ValueError(f"clean restoration/repeat loss or argmax gate failed: {name}")
    backgrounds, quartiles = clean_quartiles(clean[:, CONTEXT // 2:].mean(axis=1), ids, splits)
    groups = {split: np.asarray([i for i, name in enumerate(splits) if name == split])
              for split in SPLITS}

    def metric(name, first):
        value = losses[name][:, first:]
        delta = value - clean[:, first:]
        correct = argmax[name][:, first:] == targets[:, first:]
        prefix = greedy_prefix_lengths(correct)
        means, changes = value.mean(axis=1), delta.mean(axis=1)
        result = {"loss_indices": [first, CONTEXT], "conditioning_source_tokens": first + 1,
                  "per_passage_mean_nll": means.tolist(),
                  "per_passage_mean_delta": changes.tolist(),
                  "per_passage_teacher_forced_accuracy": correct.mean(axis=1).tolist(),
                  "per_passage_exact_greedy_prefix_tokens": prefix.tolist(), "splits": {}}
        for split, indices in groups.items():
            result["splits"][split] = {
                "mean_nll": float(means[indices].mean()),
                "mean_delta": float(changes[indices].mean()),
                "median_passage_delta": float(np.median(changes[indices])),
                "min_passage_delta": float(changes[indices].min()),
                "max_passage_delta": float(changes[indices].max()),
                "fraction_positive_token_deltas": float((delta[indices] > 0).mean()),
                "teacher_forced_accuracy": float(correct[indices].mean()),
                "mean_exact_greedy_prefix_tokens": float(prefix[indices].mean()),
                "complete_exact_greedy_continuations": int(np.count_nonzero(prefix[indices] == CONTEXT - first)),
            }
        result["prefix_minus_suffix_mean_delta"] = (
            result["splits"][SPLITS[0]]["mean_delta"] - result["splits"][SPLITS[1]]["mean_delta"])
        if first == CONTEXT // 2:
            selective = np.asarray([changes[i] - changes[other].mean()
                                    for i, other in enumerate(backgrounds)])
            result["per_passage_selective_delta"] = selective.tolist()
        return result

    arms = {name: {"primary_last512": metric(name, 512),
                   "secondary_all1024": metric(name, 0)}
            for name in CONTROLS + ARMS}

    def rank(names):
        return sorted(names, key=lambda name: (-arms[name]["primary_last512"]["splits"][SPLITS[0]]["mean_delta"], name))

    dose_comparisons = {}
    for block in range(8):
        for family in ("attention", "mlp"):
            group = f"block{block}_{family}"
            half, zero = (np.asarray(arms[group + "_" + dose]["primary_last512"]["per_passage_mean_delta"])
                          for dose in ("half", "zero"))
            dose_comparisons[group] = {
                "per_passage_zero_minus_half_delta": (zero - half).tolist(),
                "splits": {split: {"mean_zero_minus_half_delta": float((zero - half)[indices].mean()),
                                   "passages_zero_more_damaging_than_half": int(np.count_nonzero(zero[indices] > half[indices]))}
                           for split, indices in groups.items()},
            }
    return {"schema_version": 1, "stage": "causal_forward_validation_not_extraction",
            "passage_order": ids, "arm_order": list(ARMS), "controls": controls,
            "clean_quartiles": quartiles,
            "selectivity_background_ids": {ids[i]: [ids[j] for j in other]
                                           for i, other in enumerate(backgrounds)},
            "metrics": arms, "primary_prefix_delta_ranking": rank(ARMS),
            "within_family_dose_rankings": {
                family + "_" + dose: rank([name for name in ARMS if name.endswith(f"_{family}_{dose}")])
                for family in ("attention", "mlp") for dose in ("half", "zero")},
            "dose_comparisons": dose_comparisons,
            "interpretation": [
                "Loss damage is reliance on a weight group, not uniquely localized storage or a weight-only passage decoder.",
                "Primary exact greedy prefixes condition on the first 513 source tokens; correctness after the first mismatch is only teacher forced.",
                "A complete 512-token greedy continuation is conditional reproduction of this frozen passage, not proof of historical training membership.",
                "Current suffix historical held-out status is unverified; comparisons and tiny clean-quartile backgrounds are descriptive, with no p-values.",
                "Attention and MLP output groups have different parameter counts; use the within-family comparisons alongside overall rankings.",
            ]}


def load_plan(path):
    """Validate the frozen manifest, packed targets, and deterministic selection."""
    path = Path(path).resolve()
    plan = read_json(path)
    expected = {"schema_version": 1, "stage": "causal_forward_validation_plan",
                "vocab_size": VOCAB_SIZE, "context_length": CONTEXT,
                "passages_per_split": PER_SPLIT, "passage_count": 2 * PER_SPLIT,
                "primary_loss_indices": [512, 1024], "secondary_loss_indices": [0, 1024],
                "arms": list(ARMS), "controls": list(CONTROLS),
                "clean_control_tolerance": {"atol": CONTROL_ATOL, "rtol": CONTROL_RTOL,
                                            "argmax_exact": True}}
    if any(type(plan.get(key)) is not type(value) or plan.get(key) != value
           for key, value in expected.items()):
        raise ValueError("plan does not implement the fixed protocol")
    batch_path = path.parent / "batch_tokens.bin"
    record = plan.get("batch_file", {})
    if (record.get("filename") != batch_path.name or record.get("dtype") != "<i4"
            or record.get("shape") != [2, 32, 1024]
            or record.get("bytes") != 2 * 32 * 1024 * 4
            or batch_path.stat().st_size != record["bytes"]
            or sha256_file(batch_path) != record.get("sha256")):
        raise ValueError("batch identity, layout, or length differs from frozen plan")
    batch = np.fromfile(batch_path, dtype="<i4").reshape(2, 32, 1024)
    _array(batch[0], (32, 1024), "i", "inputs")
    _array(batch[1], (32, 1024), "i", "targets")
    if not np.array_equal(batch[0, :, 1:], batch[1, :, :-1]):
        raise ValueError("batch target alignment is not next-token prediction")
    split = plan["split"]
    if (split["prefix_tokens"] != split["native_token_boundary"]
            or split["prefix_tokens"] + split["suffix_tokens"] != split["corpus_tokens"]
            or split["test_fraction"] != 0.1):
        raise ValueError("inconsistent split metadata")
    passages = plan.get("passages")
    if not isinstance(passages, list) or len(passages) != 32:
        raise ValueError("plan must contain exactly 32 passages")
    for group, split_name in enumerate(SPLITS):
        length = split["prefix_tokens"] if group == 0 else split["suffix_tokens"]
        base = 0 if group == 0 else split["native_token_boundary"]
        for index, local in enumerate(midpoint_starts(length)):
            row = group * PER_SPLIT + index
            passage = passages[row]
            start, end = base + local, base + local + CONTEXT + 1
            if any(passage.get(key) != value for key, value in {
                    "passage_id": f"{split_name}_{index:02d}", "row": row, "split": split_name,
                    "local_token_start": local, "token_interval": [start, end],
                    "input_token_interval": [start, end - 1], "target_token_interval": [start + 1, end]}.items()):
                raise ValueError("passage selection, ID, row, or token interval changed")
            window = np.concatenate((batch[0, row, :1], batch[1, row])).astype("<i4")
            if (passage.get("token_ids") != window.tolist()
                    or passage.get("native_tokens_sha256") != digest(window.tobytes())):
                raise ValueError("passage token identities changed")
            offsets = passage.get("target_byte_offsets")
            if (not isinstance(offsets, list) or len(offsets) != CONTEXT + 1
                    or any(type(x) is not int for x in offsets)
                    or any(b <= a for a, b in zip(offsets, offsets[1:]))
                    or passage["target_byte_interval"] != [offsets[0], offsets[-1]]
                    or not 0 <= passage["byte_interval"][0] < offsets[0]
                    or passage["byte_interval"][1] != offsets[-1]
                    or passage["input_byte_interval"] != [passage["byte_interval"][0], offsets[-2]]):
                raise ValueError("malformed target byte addresses")
            lower = 0 if group == 0 else split["byte_boundary"]
            upper = split["byte_boundary"] if group == 0 else split["corpus_bytes"]
            if not lower <= passage["byte_interval"][0] < offsets[-1] <= upper:
                raise ValueError("passage crosses current byte split")
    return plan, batch


def authenticate_plan(plan):
    """Recheck all frozen inputs and all 100 on-disk weight byte hashes."""
    for name, record in plan["inputs"].items():
        if file_record(record["path"]) != record:
            raise ValueError(f"frozen input changed: {name}")
    snapshot = plan["checkpoint"]
    checkpoint = GPT2Checkpoint(snapshot["checkpoint_directory"], check_finite=True)
    after = checkpoint.provenance(hash_weights=True)
    if (snapshot.get("unique_weight_files") != 100
            or len(snapshot.get("weight_sha256", {})) != 100 or after != snapshot):
        raise ValueError("checkpoint weights, layout, or source provenance changed")
    return after["weight_sha256"]


def validate_run_metadata(run, plan, batch_path):
    """Native restoration checks complement, not replace, external hashes."""
    required = {"schema_version": 1, "complete": True, "context_length": CONTEXT,
                "passage_count": 32, "vocab_size": VOCAB_SIZE,
                "padded_vocab_size": 50272, "checkpoint_raw_weight_count": 101,
                "checkpoint_weight_bytes": [spec.nbytes for spec in tensor_manifest()],
                "batch_tokens_bytes": 2 * 32 * 1024 * 4,
                "non_default_executor": True, "optimizer_steps": 0, "backward_calls": 0,
                "checkpoint_unique_weight_count": 100}
    if any(type(run.get(key)) is not type(value) or run.get(key) != value
           for key, value in required.items()):
        raise ValueError("run integrity, executor, shape, or no-update metadata failed")
    for key, expected in (("batch_tokens_file", batch_path),
                          ("checkpoint_directory", plan["checkpoint"]["checkpoint_directory"]),
                          ("binary_file", plan["inputs"]["binary"]["path"])):
        if not isinstance(run.get(key), str) or Path(run[key]).resolve() != Path(expected).resolve():
            raise ValueError(f"actual run {key} differs from frozen plan")
    if type(run.get("batch_sequences")) is not int or run["batch_sequences"] <= 0:
        raise ValueError("invalid execution microbatch size")
    arms = run.get("arms")
    ordered = ["clean_before", "clean_repeat", *ARMS, "clean_after"]
    if (not isinstance(arms, list) or not all(isinstance(arm, dict) for arm in arms)
            or [arm.get("name") for arm in arms] != ordered):
        raise ValueError("missing, duplicated, reordered, or extra run arm")
    for arm in arms:
        name = arm["name"]
        if name in CONTROLS:
            indices, scale = [], 1.0
        else:
            block, family, dose = name.split("_")
            first = 12 * int(block[5:]) + (6 if family == "attention" else 12)
            indices, scale = [first, first + 1], (0.5 if dose == "half" else 0.0)
        if (arm.get("group_weight_indices") != indices or type(arm.get("scale")) not in (int, float)
                or arm["scale"] != scale or arm.get("finite_losses") is not True
                or arm.get("argmax_valid") is not True
                or arm.get("restoration_verified_bytes") is not True
                or arm.get("loss_file") != name + ".losses.f32"
                or arm.get("argmax_file") != name + ".argmax.i32"):
            raise ValueError(f"arm intervention, output, or exact restoration metadata failed: {name}")
    return arms


def report_files(plan_path, run_path, output_path):
    if Path(output_path).exists():
        raise FileExistsError(output_path)
    plan_path, run_path = Path(plan_path).resolve(), Path(run_path).resolve()
    plan_record, run_record = file_record(plan_path), file_record(run_path)
    plan, batch = load_plan(plan_path)
    weights_after = authenticate_plan(plan)
    run = read_json(run_path)
    arm_records = validate_run_metadata(run, plan, plan_path.parent / "batch_tokens.bin")
    losses, argmax, files = {}, {}, {}
    for arm in arm_records:
        for field, destination, dtype in (("loss_file", losses, "<f4"),
                                           ("argmax_file", argmax, "<i4")):
            path = run_path.parent / arm[field]
            if path.stat().st_size != 32 * 1024 * 4:
                raise ValueError(f"wrong per-token output size: {path}")
            files[arm[field]] = file_record(path)
            destination[arm["name"]] = np.fromfile(path, dtype=dtype).reshape(32, 1024)
    report = summarize(plan, losses, argmax, batch[1])
    if any(file_record(record["path"]) != record for record in files.values()):
        raise ValueError("a raw result changed while reporting")
    if file_record(plan_path) != plan_record or file_record(run_path) != run_record:
        raise ValueError("plan or run metadata changed while reporting")
    report.update({"plan_file": plan_record, "run_metadata": run_record,
                   "reporter": file_record(__file__), "raw_files": files,
                   "checkpoint_weight_hashes_after": weights_after,
                   "checkpoint_and_frozen_inputs_unchanged": True,
                   "clean_control_tolerance": plan["clean_control_tolerance"],
                   "execution_batch_sequences": run["batch_sequences"]})
    write_report(output_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planner = commands.add_parser("plan", help="freeze corpus-addressed batches before model runs")
    for name in ("corpus", "corpus-token-ids", "corpus-byte-offsets", "tokenizer-dir",
                 "protocol", "checkpoint", "binary", "output-dir"):
        planner.add_argument("--" + name, type=Path, required=True)
    reporter = commands.add_parser("report", help="validate complete paired forward results")
    for name in ("plan", "run", "output"):
        reporter.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        plan_files(args.corpus, args.corpus_token_ids, args.corpus_byte_offsets,
                   args.tokenizer_dir, args.protocol, args.checkpoint, args.binary, args.output_dir)
    else:
        report_files(args.plan, args.run, args.output)


if __name__ == "__main__":
    main()
