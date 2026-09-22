#!/usr/bin/env python3
"""Generate checked-in C++ integer tables from native checkpoint activations.

GPU capture is a separate, explicit executable. This driver validates the
finite-domain model, optionally reduces its states, and emits ordinary source
files. It is deliberately not a Bazel genrule. Intermediate JSON checkpoints
belong outside the repository; generated C++ and its provenance are committed.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

# The emitter lives with the experiment. Resolve its package from this file,
# not the caller's working directory, including for direct CLI invocations.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from discretize_core import (build_model, evaluate_model, load_model, reduce_model,
                             restore_membership, save_model)
from discretize_certificate import certify_model
from src.llm.experiments.memorize_general_facts.discretize_emit import emit_model


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(args, model):
    """Identify the concrete inputs without checking checkpoint weights into Git."""
    stats = dict(model.get("stats", {}))
    # The resumable model retains the complete per-trial journal. The committed
    # provenance keeps an identifying hash and a readable summary, not tens of
    # thousands of repetitive records. State membership is inspectable separately.
    merges = stats.pop("accepted_merges", [])
    if merges:
        encoded = json.dumps(merges, sort_keys=True, separators=(",", ":")).encode()
        stats["accepted_merges_sha256"] = hashlib.sha256(encoded).hexdigest()
        stats["accepted_merge_records"] = len(merges)
        stats["merge_distance_summary"] = {
            "minimum": min(row["euclidean_distance"] for row in merges),
            "maximum": max(row["euclidean_distance"] for row in merges),
            "induced_unions": sum(row["induced_unions"] for row in merges),
        }
    result = {
        "schema": 1,
        "protocol": f"first {model['prompt_tokens']} tokens; autonomous suffix and explicit EOS",
        "source": str(args.capture or args.model),
        "source_sha256": sha256(args.capture or args.model),
        "verification": evaluate_model(model),
        "stats": stats,
    }
    if stats.get("search", {}).get("pairwise_irreducible"):
        # Independently check the completed quotient without trusting the
        # reducer's rejection cache. This sufficient backward proof may be
        # inconclusive on other models, which is recorded honestly. Do not run
        # quadratic pair checks on the much larger unreduced baseline.
        result["irreducibility_certificate"] = certify_model(model)
    if args.checkpoint is not None:
        files = sorted(args.checkpoint.glob("weight_*.bin"))
        mapping = args.checkpoint / "compact_vocabulary.tsv"
        if not files or not mapping.is_file():
            raise ValueError("checkpoint must contain weights and compact_vocabulary.tsv")
        result["checkpoint"] = str(args.checkpoint)
        result["checkpoint_files_sha256"] = {
            path.name: sha256(path) for path in files + [mapping]
        }
    for name in ("corpus", "tokenizer_json"):
        path = getattr(args, name)
        if path is not None:
            result[name] = str(path)
            result[name + "_sha256"] = sha256(path)
    if args.original_model is not None:
        result["original_model_sha256"] = sha256(args.original_model)
    return result


def format_sources(directory):
    formatter = shutil.which("clang-format")
    if formatter is None:
        raise ValueError("clang-format must be installed to format generated C++")
    paths = sorted(path for path in directory.rglob("*")
                   if path.suffix in (".h", ".cc"))
    # Output normally lives in /tmp before installation. Explicitly use the
    # repository's Google style rather than inheriting a temporary directory's
    # configuration (or clang-format's LLVM fallback).
    style = Path(__file__).resolve().parents[2] / ".clang-format"
    # Independent translation units also format independently. Bound formatter
    # parallelism rather than making a huge shell argument list.
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda path: subprocess.run(
            [formatter, "--style=file:" + str(style), "-i", str(path)],
            check=True), paths))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture", type=Path, help="Native JSONL capture")
    source.add_argument("--model", type=Path, help="Previously saved symbolic model")
    result.add_argument("--output", required=True, type=Path,
                        help="Fresh destination for generated source")
    result.add_argument("--save_model", type=Path,
                        help="Intermediate/resumable JSON; keep outside Git")
    result.add_argument("--original_model", type=Path,
                        help="Exact baseline for recovering original state membership")
    result.add_argument("--state_index", action="store_true",
                        help="Emit inspection-only state examples and BF16 representatives")
    result.add_argument("--expected_samples", type=int, default=1024)
    result.add_argument("--reduce", action="store_true")
    result.add_argument("--compact_transitions", action="store_true",
                        help="Compile exact boundary transitions into condensed control flow")
    result.add_argument("--neighbors", type=int, default=8)
    result.add_argument("--max_passes", type=int, default=100)
    result.add_argument("--max_attempts", type=int)
    result.add_argument("--exhaustive_pair_limit", type=int, default=1_000_000)
    result.add_argument("--checkpoint_seconds", type=float, default=60)
    result.add_argument("--checkpoint", type=Path, help="Source weights for provenance")
    result.add_argument("--corpus", type=Path, help="Source corpus for provenance")
    result.add_argument("--tokenizer_json", type=Path, help="Source tokenizer for provenance")
    return result


def generate(args):
    if args.output.exists():
        raise ValueError("output must be a fresh directory; refusing to overwrite")
    if args.expected_samples <= 0:
        raise ValueError("expected_samples must be positive")
    if shutil.which("clang-format") is None:
        raise ValueError("clang-format must be installed")
    if args.capture is not None:
        model = build_model(args.capture, expected_samples=args.expected_samples)
    else:
        model = load_model(args.model)
        if len(model["samples"]) != args.expected_samples:
            raise ValueError("saved model has the wrong number of samples")
    if args.original_model is not None:
        model = restore_membership(model, load_model(args.original_model))
    print(json.dumps({"phase": "baseline", **evaluate_model(model),
                      "states": len(model["states"])}), flush=True)
    if args.reduce:
        model = reduce_model(
            model, neighbors=args.neighbors, max_passes=args.max_passes,
            max_attempts=args.max_attempts,
            exhaustive_pair_limit=args.exhaustive_pair_limit,
            checkpoint_path=args.save_model,
            checkpoint_seconds=args.checkpoint_seconds,
            progress=lambda message: print(json.dumps(message), flush=True))
    if args.compact_transitions:
        from discretize_pointwise import relabel_mlp_outputs
        model, mapping = relabel_mlp_outputs(model)
        model["state_relabeling"] = mapping
        # Renaming cannot alter the corpus task, but verify this independently
        # before compiling any tables into programs.
        evaluate_model(model)
    record = provenance(args, model)
    if args.save_model is not None:
        save_model(model, args.save_model)
    emit_model(model, args.output, include_state_index=args.state_index,
               compact_transitions=args.compact_transitions)
    format_sources(args.output)
    if args.compact_transitions:
        patterns_path = args.output / "transition_patterns.json"
        patterns = json.loads(patterns_path.read_text(encoding="utf-8"))
        for transition in patterns["transitions"]:
            transition["formatted_source_bytes"] = (
                args.output / transition["file"]).stat().st_size
        patterns_path.write_text(json.dumps(patterns, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
    # The emitter hashes its unformatted source. Publish hashes for the actual
    # formatted files that will be committed and compiled instead.
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = {
        name: sha256(args.output / name) for name in manifest["files"]
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record["generated_sources_sha256"] = {
        str(path.relative_to(args.output)): sha256(path)
        for path in sorted(args.output.rglob("*"))
        if path.is_file()
    }
    (args.output / "provenance.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({"phase": "generated", "output": str(args.output),
                      "states": len(model["states"]),
                      **evaluate_model(model)}), flush=True)
    return model


def main():
    args = parser().parse_args()
    try:
        generate(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"generation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
