"""Frozen small-neuron-group reliance mapping, not weight-only extraction.

Known corpus tokens are explicit model inputs. Discovery and confirmation use
different targets in the SAME passages. The adjusted scores remove one shared
discovery response pattern; they are not independently causal weight components
or proof that a selected row group privately stores the corresponding text.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from . import causal_validation as cv
from .checkpoint import GPT2Checkpoint, tensor_manifest


BLOCKS = (6, 7)
GROUPS = 32
FEATURES = 2048
WIDTH = 512
SEED = 20260909
DISCOVERY = (512, 768)
CONFIRMATION = (768, 1024)


def fixed_groups():
    """One PCG64 stream, first block 6 then block 7; no outcome selection."""
    rng = np.random.Generator(np.random.PCG64(SEED))
    return np.stack([rng.permutation(FEATURES).reshape(GROUPS, -1)
                     for _ in BLOCKS]).astype("<i4")


def group_records(groups, values):
    groups = np.asarray(groups)
    if groups.shape != (2, 32, 64) or groups.dtype.kind not in "iu":
        raise ValueError("expected integer groups [2,32,64]")
    result = []
    for index, block in enumerate(BLOCKS):
        if not np.array_equal(np.sort(groups[index].ravel()), np.arange(FEATURES)):
            raise ValueError("each block must be a complete feature permutation")
        weight = np.asarray(values[index], dtype=np.float64)
        if weight.shape != (FEATURES, WIDTH) or not np.isfinite(weight).all():
            raise ValueError("expected finite W2 [2048,512]")
        norms = [float(np.linalg.norm(weight[rows])) for rows in groups[index]]
        for group, rows in enumerate(groups[index]):
            comparators = sorted((i for i in range(GROUPS) if i != group),
                                 key=lambda i: (abs(norms[i] - norms[group]), i))[:4]
            result.append({
                "name": f"block{block}_group{group:02d}_half",
                "block": block, "group": group,
                "rows": rows.tolist(), "weight_index": 12 + 12 * block,
                "weight_file": f"weight_{12 + 12 * block}.bin",
                "byte_intervals": [[int(row) * WIDTH * 4,
                                    (int(row) + 1) * WIDTH * 4] for row in rows],
                "changed_weight_count": len(rows) * WIDTH,
                "frobenius_norm_fp64": norms[group],
                "norm_comparators": [index * GROUPS + i for i in comparators],
            })
    return result


def arm_specs(groups):
    """Single authoritative order for validating driver metadata and outputs."""
    def arm(name, kind, block=None, rows=(), indices=(), scale=1):
        return {"name": name, "kind": kind, "block": block,
                "selected_rows": list(rows), "group_weight_indices": list(indices),
                "scale": scale}
    result = [arm(name, "clean") for name in ("clean_before", "clean_repeat")]
    for block in BLOCKS:
        first = 12 + 12 * block
        result.append(arm(f"block{block}_mlp_half", "whole_mlp_output", block,
                          indices=(first, first + 1), scale=0.5))
        for group in groups:
            if group["block"] == block:
                result.append(arm(group["name"], "mlp_output_rows", block,
                                  group["rows"], (first,), 0.5))
    result.append(arm("clean_after", "clean"))
    return result


def double_center(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2 or not np.isfinite(matrix).all():
        raise ValueError("need a finite group-by-passage matrix with both axes >=2")
    return matrix - matrix.mean(axis=1, keepdims=True) - matrix.mean(axis=0) + matrix.mean()


def ordinary_contrasts(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    double_center(matrix)  # Common shape/nonfinite validation.
    return (matrix - matrix.mean(axis=1, keepdims=True)) * matrix.shape[1] / (matrix.shape[1] - 1)


def discover(delta):
    """Fit only discovery, with deterministic ties and explicit abstention.

    A group-strength times passage-vulnerability matrix, plus arbitrary additive
    effects, has centered rank <=1. Its residual is numerically zero. The floor
    uses the ORIGINAL delta scale so cancellation during centering is covered
    diagnostically; it is not a certified bound or an estimate of GPU noise.
    """
    delta = np.asarray(delta, dtype=np.float64)
    centered = double_center(delta)
    floor = float(64 * np.finfo(np.float64).eps * max(delta.shape) * np.linalg.norm(delta))
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    direction = np.zeros(delta.shape[0])
    status = "zero_centered_effect"
    residual = np.zeros_like(delta)
    if singular[0] > floor:
        if singular[0] - singular[1] <= floor:
            status = "ambiguous_leading_direction"
        else:
            status = "unique_discovery_direction"
            direction = u[:, 0]
            residual = centered - np.outer(direction, direction @ centered)
    selected = []
    for passage in range(delta.shape[1]):
        eligible = np.flatnonzero((delta[:, passage] > 0) & (residual[:, passage] > floor))
        selected.append(int(eligible[np.argmax(residual[eligible, passage])]) if len(eligible) else None)
    energy = float(np.sum(centered * centered))
    removed = float(np.sum((direction @ centered) ** 2))
    return {"status": status, "numeric_floor": floor,
            "floor_scope": "FP64 diagnostic only, not GPU noise or interval bound",
            "singular_values": singular.tolist(), "frozen_direction": direction.tolist(),
            "centered_energy": energy,
            "removed_discovery_energy_fraction": removed / energy if energy else None,
            "residual": residual.tolist(), "selected_group_indices": selected}


def confirm(delta, discovery):
    """Apply the frozen discovery direction, never fit confirmation singulars."""
    centered = double_center(delta)
    direction = np.asarray(discovery["frozen_direction"], dtype=np.float64)
    if direction.shape != (centered.shape[0],) or not np.isfinite(direction).all():
        raise ValueError("invalid frozen discovery direction")
    return centered - np.outer(direction, direction @ centered)


def score_rank(values, index):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("rank requires a finite vector")
    if type(index) is not int or not 0 <= index < len(values):
        raise ValueError("invalid rank index")
    return {"rank": 1 + int(np.count_nonzero(values > values[index])),
            "exact_tie_count_including_self": int(np.count_nonzero(values == values[index])),
            "candidate_count": len(values)}


def select_and_confirm(discovery_delta, confirmation_delta, groups):
    discovery_delta = np.asarray(discovery_delta, dtype=np.float64)
    confirmation_delta = np.asarray(confirmation_delta, dtype=np.float64)
    if discovery_delta.shape != (64, 16) or confirmation_delta.shape != (64, 16) or len(groups) != 64:
        raise ValueError("fixed experiment needs 64 groups and 16 prefix passages")
    fitted = discover(discovery_delta)
    residual = confirm(confirmation_delta, fitted)
    contrasts = ordinary_contrasts(confirmation_delta)
    selected = []
    for passage, group in enumerate(fitted["selected_group_indices"]):
        if group is None:
            selected.append({"passage_index": passage, "selected_group_index": None})
            continue
        comparators = groups[group]["norm_comparators"]
        base = group // GROUPS * GROUPS
        raw = float(confirmation_delta[group, passage])
        adjusted = float(residual[group, passage])
        selected.append({
            "passage_index": passage, "selected_group_index": group,
            "group_name": groups[group]["name"],
            "discovery_raw_delta": float(discovery_delta[group, passage]),
            "discovery_residual": fitted["residual"][group][passage],
            "confirmation_raw_delta": raw,
            "confirmation_other15_contrast": float(contrasts[group, passage]),
            "confirmation_projected_residual": adjusted,
            "confirmation_rank_all64": score_rank(residual[:, passage], group),
            "confirmation_rank_same_block": score_rank(residual[base:base + GROUPS, passage], group - base),
            "norm_comparator_group_indices": comparators,
            "confirmation_minus_norm_comparator_mean": float(adjusted - residual[comparators, passage].mean()),
            "harm_and_residual_both_positive": raw > 0 and adjusted > 0,
            "all_prefix_confirmation_raw_deltas": confirmation_delta[group].tolist(),
        })
    present = [row for row in selected if row["selected_group_index"] is not None]
    centered = double_center(confirmation_delta)
    energy = float(np.sum(centered ** 2))
    direction = np.asarray(fitted["frozen_direction"])
    return {"discovery": fitted, "confirmation_projected_residual": residual.tolist(),
            "confirmation_centered_energy": energy,
            "frozen_direction_confirmation_energy_fraction": float(np.sum((direction @ centered) ** 2)) / energy if energy else None,
            "selected": selected,
            "summary": {
                "selected_passages": len(present),
                "distinct_selected_groups": len({row["selected_group_index"] for row in present}),
                "confirmation_positive_raw_and_residual": sum(row["harm_and_residual_both_positive"] for row in present),
                "confirmation_top16_of64": sum(row["confirmation_rank_all64"]["rank"] <= 16 for row in present),
                "confirmation_beats_norm_comparator_mean": sum(row["confirmation_minus_norm_comparator_mean"] > 0 for row in present),
                "no_significance_or_private_storage_claim": True,
            }}


def plan_files(args):
    """Reuse the tested native-token passage planner, then freeze new arms.

    passages/manifest.json is a source plan for passage identities only. Its old
    branch-arm list is NOT executed here; this outer manifest is authoritative
    for all new interventions and the reporter checks each against the driver.
    """
    requested_output = Path(args.output_dir).absolute()
    if requested_output.exists() or requested_output.is_symlink():
        raise FileExistsError(requested_output)
    output = requested_output.resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    if output == checkpoint_path or checkpoint_path in output.parents:
        raise ValueError("analysis output must not be inside checkpoint")
    output.mkdir()
    cv.plan_files(args.corpus, args.corpus_token_ids, args.corpus_byte_offsets,
                  args.tokenizer_dir, args.protocol, args.checkpoint,
                  args.binary, output / "passages")
    # Compare JSON-canonical provenance, not in-memory TensorSpec tuples with
    # the arrays that JSON necessarily deserializes as lists.
    base, _ = cv.load_plan(output / "passages/manifest.json")
    checkpoint = GPT2Checkpoint(args.checkpoint, check_finite=True)
    groups = fixed_groups()
    records = group_records(groups, [checkpoint.mlp_values(block) for block in BLOCKS])
    source_dir = Path(__file__).resolve().parent
    additional = {name: cv.file_record(source_dir / name)
                  for name in ("neuron_mapping.py", "neuron_probe_main.cc")}
    payload = groups.tobytes()
    with (output / "neuron_groups.i32").open("xb") as stream:
        stream.write(payload)
    plan = {
        "schema_version": 1, "stage": "fine_neuron_reliance_plan_not_extraction",
        "seed": SEED, "group_shape": [2, 32, 64], "group_dtype": "<i4",
        "discovery_indices": list(DISCOVERY), "confirmation_indices": list(CONFIRMATION),
        "execution_batch_sequences": 4, "groups": records, "arms": arm_specs(records),
        "group_file": cv.file_record(output / "neuron_groups.i32"),
        "passage_plan": cv.file_record(output / "passages/manifest.json"),
        "additional_sources": additional,
        "runtime_environment": {"numpy": np.__version__,
                                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
        "passage_plan_scope": "only passage/batch/provenance validation; its old arm list is not executed",
    }
    cv.authenticate_plan(base)
    cv.write_report(output / "manifest.json", plan)
    authenticate(output / "manifest.json")
    return plan


def authenticate(path):
    """Authenticate exact inputs, deterministic masks, and all 100 weights."""
    path = Path(path).resolve()
    plan = cv.read_json(path)
    required = {"schema_version": 1, "stage": "fine_neuron_reliance_plan_not_extraction",
                "seed": SEED, "group_shape": [2, 32, 64], "group_dtype": "<i4",
                "discovery_indices": list(DISCOVERY), "confirmation_indices": list(CONFIRMATION),
                "execution_batch_sequences": 4}
    if any(type(plan.get(k)) is not type(v) or plan.get(k) != v for k, v in required.items()):
        raise ValueError("plan differs from frozen protocol")
    source_dir = Path(__file__).resolve().parent
    source_names = {"neuron_mapping.py", "neuron_probe_main.cc"}
    if set(plan.get("additional_sources", {})) != source_names:
        raise ValueError("missing or extra analysis source identities")
    for name, record in plan["additional_sources"].items():
        if Path(record["path"]) != source_dir / name:
            raise ValueError("analysis source path substituted")
    for record in [plan["group_file"], plan["passage_plan"], *plan["additional_sources"].values()]:
        if cv.file_record(record["path"]) != record:
            raise ValueError("frozen plan input changed")
    if Path(plan["passage_plan"]["path"]) != path.parent / "passages/manifest.json":
        raise ValueError("unexpected passage plan location")
    if Path(plan["group_file"]["path"]) != path.parent / "neuron_groups.i32":
        raise ValueError("unexpected group file location")
    payload = fixed_groups().tobytes()
    if Path(plan["group_file"]["path"]).read_bytes() != payload:
        raise ValueError("group mask differs from fixed seed/draw order")
    base, batch = cv.load_plan(plan["passage_plan"]["path"])
    expected_inputs = {"corpus", "native_tokens", "native_offsets", "tokenizer",
                       "protocol", "planner", "binary", "checkpoint_helper",
                       "native_validation_helper", "token_validation_helper",
                       "causal_probe.h", "causal_probe.cc", "causal_probe_main.cc",
                       "token_argmax.h", "token_argmax.cc", "BUILD.bazel", "dataset_split_source"}
    if set(base.get("inputs", {})) != expected_inputs:
        raise ValueError("missing or extra passage-source identities")
    expected_paths = {"planner": source_dir / "causal_validation.py",
                      "checkpoint_helper": source_dir / "checkpoint.py",
                      "native_validation_helper": source_dir / "verify.py",
                      "token_validation_helper": source_dir / "vocabulary_verify.py",
                      "dataset_split_source": source_dir.parents[1] / "src/dataset/dataset.cc"}
    expected_paths.update({name: source_dir / name for name in
                           ("causal_probe.h", "causal_probe.cc", "causal_probe_main.cc",
                            "token_argmax.h", "token_argmax.cc", "BUILD.bazel")})
    if any(Path(base["inputs"][name]["path"]) != expected for name, expected in expected_paths.items()):
        raise ValueError("passage analysis source path substituted")
    cv.authenticate_plan(base)
    checkpoint = GPT2Checkpoint(base["checkpoint"]["checkpoint_directory"])
    records = group_records(fixed_groups(), [checkpoint.mlp_values(block) for block in BLOCKS])
    if plan["groups"] != records or plan["arms"] != arm_specs(records):
        raise ValueError("group addresses, norms, comparators, or arm order changed")
    return plan, base, batch


def validate_run(run, plan, base, batch_path):
    required = {"schema_version": 1, "complete": True, "context_length": 1024,
                "passage_count": 32, "vocab_size": 50257, "padded_vocab_size": 50272,
                "batch_sequences": 4, "checkpoint_raw_weight_count": 101,
                "checkpoint_unique_weight_count": 100,
                "checkpoint_weight_bytes": [spec.nbytes for spec in tensor_manifest()],
                "batch_tokens_bytes": 2 * 32 * 1024 * 4,
                "neuron_groups_bytes": 2 * 32 * 64 * 4,
                "neuron_groups_shape": [2, 32, 64], "neuron_group_blocks": [6, 7],
                "neuron_groups_permutations_verified": True,
                "non_default_executor": True, "optimizer_steps": 0, "backward_calls": 0}
    if any(type(run.get(k)) is not type(v) or run.get(k) != v for k, v in required.items()):
        raise ValueError("run identity, shape, executor, or no-update gate failed")
    for key, expected in (("binary_file", base["inputs"]["binary"]["path"]),
                          ("checkpoint_directory", base["checkpoint"]["checkpoint_directory"]),
                          ("batch_tokens_file", batch_path),
                          ("neuron_groups_file", plan["group_file"]["path"])):
        if not isinstance(run.get(key), str) or Path(run[key]).resolve() != Path(expected).resolve():
            raise ValueError("run input path differs from frozen plan")
    arms = run.get("arms")
    if not isinstance(arms, list) or len(arms) != len(plan["arms"]):
        raise ValueError("run missing or extra arms")
    for actual, expected in zip(arms, plan["arms"], strict=True):
        if not isinstance(actual, dict) or any(actual.get(k) != v for k, v in expected.items()):
            raise ValueError("run arm differs from frozen intervention")
        if any(actual.get(k) is not True for k in ("finite_losses", "argmax_valid", "restoration_verified_bytes")):
            raise ValueError("arm integrity gate failed")
        if any(actual.get(k) != expected["name"] + suffix
               for k, suffix in (("loss_file", ".losses.f32"), ("argmax_file", ".argmax.i32"))):
            raise ValueError("unexpected arm output filename")
    return arms


def report_files(plan_path, run_path, output_path):
    plan_path, run_path = Path(plan_path).resolve(), Path(run_path).resolve()
    identities = {"plan": cv.file_record(plan_path), "run": cv.file_record(run_path)}
    plan, base, batch = authenticate(plan_path)
    run = cv.read_json(run_path)
    arms = validate_run(run, plan, base, plan_path.parent / "passages/batch_tokens.bin")
    files, losses, argmax = {}, {}, {}
    for arm in arms:
        for field, dtype, destination in (("loss_file", "<f4", losses), ("argmax_file", "<i4", argmax)):
            path = run_path.parent / arm[field]
            record = cv.file_record(path)
            if record["bytes"] != 32 * 1024 * 4:
                raise ValueError("incorrect measurement size")
            files[arm[field]] = record
            value = np.fromfile(path, dtype=dtype).reshape(32, 1024)
            if field == "loss_file":
                if not np.isfinite(value).all() or np.any(value < 0):
                    raise ValueError("nonfinite or negative loss")
            elif np.any(value < 0) or np.any(value >= 50257):
                raise ValueError("argmax outside logical vocabulary")
            destination[arm["name"]] = value
    for name in ("clean_repeat", "clean_after"):
        for suffix in (".losses.f32", ".argmax.i32"):
            if files[name + suffix]["sha256"] != files["clean_before" + suffix]["sha256"]:
                raise ValueError("clean replay is not byte identical; do not interpret map")
    means, deltas, accuracies = {}, {}, {}
    for label, (start, end) in (("discovery", DISCOVERY), ("confirmation", CONFIRMATION), ("last512", (512, 1024))):
        clean = losses["clean_before"][:, start:end].astype(np.float64)
        means[label], deltas[label], accuracies[label] = {}, {}, {}
        for name in losses:
            current = losses[name][:, start:end].astype(np.float64)
            means[label][name] = current.mean(axis=1).tolist()
            deltas[label][name] = (current - clean).mean(axis=1).tolist()
            accuracies[label][name] = (argmax[name][:, start:end] == batch[1, :, start:end]).mean(axis=1).tolist()
    group_names = [group["name"] for group in plan["groups"]]
    raw_discovery = np.asarray([deltas["discovery"][name][:16] for name in group_names])
    raw_confirmation = np.asarray([deltas["confirmation"][name][:16] for name in group_names])
    mapping = select_and_confirm(raw_discovery, raw_confirmation, plan["groups"])
    for entry in mapping["selected"]:
        passage = base["passages"][entry["passage_index"]]
        offsets = passage["target_byte_offsets"]
        entry["passage_id"] = passage["passage_id"]
        entry["discovery_target_bytes"] = [offsets[DISCOVERY[0]], offsets[DISCOVERY[1]]]
        entry["confirmation_target_bytes"] = [offsets[CONFIRMATION[0]], offsets[CONFIRMATION[1]]]
        index = entry["selected_group_index"]
        if index is not None:
            entry["weight_group"] = plan["groups"][index]
            entry["suffix_confirmation_raw_deltas"] = deltas["confirmation"][group_names[index]][16:]
    for record in [*files.values(), *identities.values()]:
        if cv.file_record(record["path"]) != record:
            raise ValueError("result or plan changed during reporting")
    authenticate(plan_path)
    result = {"schema_version": 1, "stage": "fine_neuron_forward_reliance_not_extraction",
              "plan": plan, "passage_plan": base, "identities": identities,
              "run_metadata": run, "raw_files": files,
              "per_passage_mean_nll": means, "per_passage_mean_delta": deltas,
              "per_passage_teacher_forced_accuracy": accuracies, "mapping": mapping,
              "all_checkpoints_and_inputs_unchanged": True, "clean_replays_byte_identical": True,
              "limitations": [
                  "Known corpus inputs, not analytically extracted text.",
                  "Within-passage disjoint target confirmation, not independent documents.",
                  "Projected residuals mix all groups/passages; raw finite interventions alone are causal effects.",
                  "Removing one discovery mode neither eliminates all confounds nor preserves all memories.",
                  "All selection and ranking is descriptive; no p-values or unique/minimal storage claims."]}
    cv.write_report(output_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planner = commands.add_parser("plan")
    for name in ("corpus", "corpus-token-ids", "corpus-byte-offsets", "tokenizer-dir",
                 "protocol", "checkpoint", "binary", "output-dir"):
        planner.add_argument("--" + name, type=Path, required=True)
    check = commands.add_parser("authenticate")
    check.add_argument("--plan", type=Path, required=True)
    reporter = commands.add_parser("report")
    for name in ("plan", "run", "output"):
        reporter.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        plan_files(args)
    elif args.command == "authenticate":
        authenticate(args.plan)
        print("All frozen inputs, masks, and checkpoint weights authenticated.")
    else:
        result = report_files(args.plan, args.run, args.output)
        print(result["mapping"]["summary"])


if __name__ == "__main__":
    main()
