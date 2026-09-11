"""Freeze five completed weight-only runs, then separately verify exact matches.

`freeze` never accepts or opens a corpus. It checks recorded checkpoint identities,
not the checkpoint weight files themselves, and preserves every JSONL byte and
all restart denominators. `summarize` is explicitly corpus-assisted VERIFICATION,
not extraction. Neither command executes a model or changes frozen candidates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .checkpoint import GPT2Config, SOURCE_FILES, sha256_file, tensor_manifest


ARMS = ("final_full", "final_affine", "final_no_cross", "final_broken", "early_full")
MODES = ("full", "affine", "no_cross", "broken", "full")
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "ai-slop/research/weight_memorization/LATE_MLP_POLYNOMIAL_PROTOCOL.md"
EXTRACTOR_SOURCES = ("late_mlp_paths.py", "late_mlp_polynomial.py", "checkpoint.py",
                     "path_diagnostics.py", "verify.py")
VERIFIER_SOURCES = ("late_mlp_experiment.py", "higher_order_controls.py",
                    "controls.py", "verify.py", "checkpoint.py")


@dataclass(frozen=True)
class ExperimentSpec:
    """CLI uses only these frozen defaults; smaller values support toy tests."""

    size: int = 8192
    length: int = 16
    restarts: int = 64
    sweeps: int = 8
    seed: int = 20260909
    alpha: float = 0.125


DEFAULT_SPEC = ExperimentSpec()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def strict_json(data):
    """Reject duplicate keys and nonfinite numbers, including overflowed 1e999."""
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def finite(value):
        if isinstance(value, float):
            require(np.isfinite(value), "nonfinite JSON value")
        elif isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)

    result = json.loads(data, object_pairs_hook=pairs)
    finite(result)
    return result


def ids(value, shape, maximum=50257):
    array = np.asarray(value)
    require(array.shape == shape and array.dtype.kind in "iu"
            and np.all(array >= 0) and np.all(array < maximum), "invalid token IDs or shape")
    return array.astype(np.int64)


def numbers(value, count):
    array = np.asarray(value)
    require(array.shape == (count,) and array.dtype.kind in "iuf"
            and np.isfinite(array).all(), "invalid score vector")
    return array.astype(np.float64)


def close(left, right):
    require(np.all(np.abs(left - right) <= 1e-10 + 5e-10 * np.maximum(np.abs(left), np.abs(right))),
            "score replay mismatch")


def valid_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(letter in "0123456789abcdef" for letter in value))


def validate_checkpoint(record, basename):
    """Validate a recorded manifest, without opening any checkpoint weights."""
    config = GPT2Config()
    manifest = tensor_manifest(config)
    require(record["config"] == asdict(config)
            and record["head_dim"] == config.head_dim
            and record["format"] == "pluto-gpt2-raw-unique-fp32-weights"
            and record["storage_dtype"] == "<f4"
            and record["unique_weight_files"] == 100
            and record["tensors"] == json.loads(json.dumps([item.to_dict() for item in manifest]))
            and record["checkpoint_bytes"] == sum(item.nbytes for item in manifest)
            and record["parameter_count"] == sum(item.nbytes // 4 for item in manifest),
            "checkpoint architecture/manifest differs from frozen GPT-2")
    require(Path(record["checkpoint_directory"]).name == basename, "wrong checkpoint step")
    hashes = record["weight_sha256"]
    require(set(hashes) == {f"weight_{index}.bin" for index in range(100)}
            and all(valid_sha(value) for value in hashes.values()), "missing checkpoint weight hashes")
    require(record["source_sha256"] == {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
            "checkpoint layout sources changed")


def validate_sources(sources):
    expected = {name: Path(__file__).with_name(name) for name in EXTRACTOR_SOURCES}
    expected.update(production_gelu=ROOT / "src/llm/layers/gelu.cc", protocol=PROTOCOL)
    require(set(sources) == set(expected) | {"tokenizer"}, "unexpected extractor source set")
    for name, record in sources.items():
        require(file_record(expected.get(name, record["path"])) == record,
                f"source/protocol/tokenizer changed: {name}")


def validate_selection(plan, spec):
    selection = plan["selection"]
    selected = ids(selection["sorted_ids"], (spec.size,))
    ranked = ids(selection["ranked_ids"], (spec.size,))
    scores = numbers(selection["ranked_scores"], spec.size)
    require(np.all(selected[1:] > selected[:-1])
            and np.array_equal(np.sort(ranked), selected)
            and np.array_equal(np.lexsort((ranked, -scores)), np.arange(spec.size))
            and np.all(scores >= 0), "invalid vocabulary ordering")
    require(selection["method"] == "final:centroid_distance"
            and selection["rule"] == "descending norm(E[t]-mean_ALL_LOGICAL(E)); ascending ID ties"
            and valid_sha(selection["full_logical_score_sha256"])
            and valid_sha(selection["full_logical_centroid_sha256"]), "wrong selection rule")
    initial = ids(plan["initial_token_ids"], (spec.restarts, spec.length))
    rng = np.random.Generator(np.random.PCG64(spec.seed))
    expected = selected[rng.integers(0, spec.size, size=initial.shape, dtype=np.int64)]
    require(np.array_equal(initial, expected)
            and digest(initial.astype("<i4").tobytes()) == plan["initial_ids_sha256"],
            "initial strings do not match frozen seed/selection")
    return selected, initial


def validate_trace(records, metadata, initial, selected, spec):
    rows = initial.copy()
    scores = numbers([row["initial_score"] for row in records], spec.restarts)
    trace = metadata["optimization_trace"]
    require(len(trace) == spec.sweeps * spec.length, "incomplete optimization trace")
    for index, entry in enumerate(trace):
        sweep, offset = divmod(index, spec.length)
        position = offset if sweep % 2 == 0 else spec.length - 1 - offset
        require((entry["sweep"], entry["position"], entry["direction"]) ==
                (sweep, position, "forward" if sweep % 2 == 0 else "reverse"), "wrong trace order")
        before = numbers(entry["scores_before"], spec.restarts)
        after = numbers(entry["scores_after"], spec.restarts)
        close(before, scores)
        require(np.all(after >= before), "recorded score decreased after roundoff retention")
        chosen = ids(entry["selected_token_ids"], (spec.restarts,))
        require(np.isin(chosen, selected).all(), "trace token outside frozen selection")
        changed = np.flatnonzero(rows[:, position] != chosen).tolist()
        require(changed == entry["changed_rows"], "changed rows disagree with token trace")
        retained = entry["roundoff_retained_rows"]
        require(isinstance(retained, list) and all(type(row) is int and 0 <= row < spec.restarts
                for row in retained) and len(set(retained)) == len(retained)
                and not set(retained).intersection(changed), "invalid roundoff-retained rows")
        require(np.array_equal(before[retained], after[retained]), "retained scores changed")
        require(entry["max_score_discrepancy"] >= 0, "negative replay discrepancy")
        rows[:, position] = chosen
        scores = after
    require(np.array_equal(rows, [row["token_ids"] for row in records]), "final tokens disagree with trace")
    close(scores, numbers([row["score"] for row in records], spec.restarts))
    for field, component in (("score", "total"), ("linear_score", "linear"),
                             ("quadratic_score", "quadratic")):
        close(numbers([row[field] for row in records], spec.restarts),
              numbers(metadata["final_components"][component], spec.restarts))
    close(numbers([row["score"] for row in records], spec.restarts),
          numbers([row["linear_score"] + row["quadratic_score"] for row in records], spec.restarts))


def validate_run(run_directory, spec=DEFAULT_SPEC):
    """Read only run artifacts and their declared sources; never read corpus data."""
    directory = Path(run_directory).resolve()
    inputs, plans, combined, all_records = {}, [], [], []
    for arm, mode in zip(ARMS, MODES):
        candidate_path = directory / f"{arm}.jsonl"
        plan_path = Path(str(candidate_path) + ".plan.json")
        metadata_path = Path(str(candidate_path) + ".metadata.json")
        raw = candidate_path.read_bytes()
        require(raw.endswith(b"\n"), "candidate JSONL must end with newline")
        records = [strict_json(line) for line in raw.splitlines()]
        plan, metadata = strict_json(plan_path.read_bytes()), strict_json(metadata_path.read_bytes())
        snapshots = {"candidates": file_record(candidate_path), "plan": file_record(plan_path),
                     "metadata": file_record(metadata_path)}
        require(digest(raw) == snapshots["candidates"]["sha256"], "candidates changed while reading")
        parameters = {"mode": mode, "vocabulary_size": spec.size, "length": spec.length,
                      "restarts": spec.restarts, "sweeps": spec.sweeps, "seed": spec.seed,
                      "alpha": spec.alpha, "label": arm, "blocks": [6, 7]}
        require(plan["schema_version"] == metadata["schema_version"] == 1
                and plan["stage"] == "frozen_weight_only_polynomial_plan"
                and metadata["stage"] == "completed_weight_only_polynomial_extraction"
                and metadata["complete"] is True, "run not complete")
        require(plan["parameters"] == metadata["parameters"] == parameters, "nonprotocol parameters")
        require(metadata["plan"] == snapshots["plan"]
                and metadata["candidates"] == snapshots["candidates"]
                and plan["output"] == str(candidate_path), "plan/candidate identity mismatch")
        require(metadata["checkpoint_and_sources_unchanged"] is True
                and metadata["compiled_arrays_unchanged"] is True
                and metadata["compiled_arrays"] == metadata["compiled_arrays_after"]
                and set(metadata["compiled_arrays"]) ==
                {"token_ids", "direct", "features", "readout", "neuron_readout", "g7", "h7"},
                "recorded coefficient/input integrity failed")
        require(plan["runtime_environment"] == metadata["runtime_environment"], "runtime identity mismatch")
        require(all(metadata["compiler"][key] == value for key, value in
                    {"mode": mode, "blocks": [6, 7], "length": spec.length, "alpha": spec.alpha}.items()),
                "compiler parameters differ")
        validate_sources(plan["sources"])
        validate_checkpoint(plan["checkpoint"], "step_10" if arm == "early_full" else "step_13030")
        selected, initial = validate_selection(plan, spec)
        require(len(records) == metadata["candidate_count"] == spec.restarts, "missing restart candidates")
        require(metadata["distinct_candidates"] == len({tuple(row["token_ids"]) for row in records}),
                "incorrect distinct-candidate count")
        method = "late_mlp_" + arm
        for index, row in enumerate(records):
            tokens = ids(row["token_ids"], (spec.length,))
            require(np.isin(tokens, selected).all() and row["method"] == method
                    and row["candidate_id"] == f"{method}:restart{index}"
                    and row["initial_token_ids"] == initial[index].tolist()
                    and row["provenance"] == {"plan_file": str(plan_path), "restart": index}
                    and len(row["vocabulary_labels"]) == spec.length
                    and all(isinstance(label, str) for label in row["vocabulary_labels"]),
                    "candidate identity, initial tokens, or vocabulary differs")
        validate_trace(records, metadata, initial, selected, spec)
        inputs[arm], plans = snapshots, plans + [plan]
        combined.append(raw)
        all_records.extend(records)
    final = plans[0]
    for plan in plans:
        require(plan["selection_checkpoint"] == final["checkpoint"]
                and plan["selection"] == final["selection"]
                and plan["initial_token_ids"] == final["initial_token_ids"]
                and plan["sources"] == final["sources"], "arms do not share final selection/starts/sources")
    require(all(plan["checkpoint"] == final["checkpoint"] for plan in plans[:4]),
            "final arms used different checkpoints")
    require(plans[4]["checkpoint"]["weight_sha256"] != final["checkpoint"]["weight_sha256"],
            "early and final recorded weight hashes are identical")
    return inputs, b"".join(combined), all_records


def manifest(directory, inputs, combined, spec, created_utc):
    return {"schema_version": 1, "stage": "frozen_weight_only_artifacts",
            "created_utc": created_utc,
            "arm_order": list(ARMS), "spec": asdict(spec), "inputs": inputs,
            "combined": {"path": str(directory / "combined.jsonl"), "bytes": len(combined),
                         "sha256": digest(combined)},
            "verification_sources": {name: file_record(Path(__file__).with_name(name))
                                     for name in VERIFIER_SOURCES},
            "checkpoint_validation": "Recorded 100-file manifests and source hashes checked; weight files not reread.",
            "candidate_count": len(ARMS) * spec.restarts, "corpus_read_by_this_command": False,
            "chronology": "This manifest does not establish that no earlier corpus verification occurred."}


def write_json_exclusive(path, value):
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as output:
        output.write(encoded)


def freeze(run_directory, spec=DEFAULT_SPEC):
    directory = Path(run_directory).resolve()
    combined_path, frozen_path = directory / "combined.jsonl", directory / "frozen.json"
    require(not combined_path.exists() and not frozen_path.exists(), "freeze output already exists")
    inputs, combined, _ = validate_run(directory, spec)
    frozen = manifest(directory, inputs, combined, spec, datetime.now(timezone.utc).isoformat())
    # All validation precedes exclusive creation. A failed second write leaves an
    # incomplete freeze, never a valid manifest referring to incomplete bytes.
    with combined_path.open("xb") as output:
        output.write(combined)
    write_json_exclusive(frozen_path, frozen)
    return frozen_path


def checked_freeze(run_directory, spec=DEFAULT_SPEC):
    directory = Path(run_directory).resolve()
    frozen_path = directory / "frozen.json"
    frozen_record = file_record(frozen_path)
    frozen = strict_json(frozen_path.read_bytes())
    require(datetime.fromisoformat(frozen["created_utc"]).utcoffset() is not None,
            "freeze timestamp must include timezone")
    inputs, combined, records = validate_run(directory, spec)
    require(frozen == manifest(directory, inputs, combined, spec, frozen["created_utc"])
            and file_record(directory / "combined.jsonl") == frozen["combined"]
            and file_record(frozen_path) == frozen_record, "frozen inputs or verification sources changed")
    return frozen_record, records


def summarize(run_directory, native_tokens, split=1650781, spec=DEFAULT_SPEC, output=None):
    directory = Path(run_directory).resolve()
    output = Path(output).resolve() if output is not None else directory / "summary.json"
    require(not output.exists(), "summary output already exists")
    frozen_record, records = checked_freeze(directory, spec)
    # This is the first operation allowed to touch the corpus. Matching imports
    # are deliberately confined to this verification-only command.
    from .higher_order_controls import summarize_paths

    native_record = file_record(native_tokens)
    require(native_record["bytes"] % 4 == 0 and 0 < split < native_record["bytes"] // 4,
            "invalid native uint32 corpus or split boundary")
    tokens = np.memmap(native_tokens, mode="r", dtype="<u4")
    require(np.all(tokens < GPT2Config().vocab_size), "native token outside logical vocabulary")
    summaries = {name: summarize_paths(part, records, GPT2Config().vocab_size)
                 for name, part in (("full_corpus", tokens), ("current_prefix", tokens[:split]),
                                    ("current_suffix", tokens[split:]))}
    require(file_record(native_tokens) == native_record
            and checked_freeze(directory, spec)[0] == frozen_record, "inputs changed during verification")
    report = {"schema_version": 1, "stage": "corpus_assisted_verification_of_frozen_candidates",
              "frozen": frozen_record, "native_tokens": native_record,
              "native_dtype": "<u4", "split_token_index": split, "total_corpus_tokens": int(tokens.size),
              "scope_token_counts": {"full_corpus": int(tokens.size), "current_prefix": split,
                                     "current_suffix": int(tokens.size) - split},
              "arm_order": list(ARMS), "candidates_per_arm": spec.restarts,
              "summaries": summaries,
              "sources": {name: file_record(Path(__file__).with_name(name)) for name in VERIFIER_SOURCES},
              "limitations": [
                  "Current prefix/suffix membership does not authenticate the historical training split.",
                  "All restart candidates, including duplicates and failures to match, retain their denominators.",
                  "Substring matches alone do not establish memorization or causal text storage.",
                  "Distinct substring counts select one longest match per candidate, not every tied substring.",
                  "This summary is verification only; no candidates or parameters are selected from the corpus."]}
    write_json_exclusive(output, report)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--run-dir", type=Path, required=True)
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("--run-dir", type=Path, required=True)
    summary_parser.add_argument("--native-tokens", type=Path, required=True)
    summary_parser.add_argument("--split", type=int, default=1650781)
    summary_parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    path = (freeze(args.run_dir) if args.command == "freeze"
            else summarize(args.run_dir, args.native_tokens, args.split, output=args.output))
    print(json.dumps(file_record(path)))


if __name__ == "__main__":
    main()
