#!/usr/bin/env python3
"""Measure fixed-seed training under consistent vocabulary permutations.

The corpus's segmentation, lengths, EOS, initialization, minibatch schedule,
optimizer and update count are fixed. Only compact token labels change. Output
is experimental data, not a claim that a small sample identifies the training
algorithm. All artifacts belong outside the source tree.
"""

import argparse
from array import array
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

from run_depth_search import _read_result, _sha256, _write_summary
from verify_predictions import _corpus_lines, verify_predictions


MODEL = dict(layers=4, model_width=10, feed_forward_width=20,
             context_length=27, vocabulary_size=4475)
SCHEDULE = dict(steps=120000, batch_size=32, seed=1337, learning_rate=0.0012,
                warmup_steps=100, eval_every=256, checkpoint_every=30000,
                stop_when_memorized=False, training_seconds=0)
SOURCE_NAMES = ("run_dataset_weights.py", "run_depth_search.py",
                "verify_predictions.py", "analyze_dataset_weights.py")


def permutation(vocabulary_size, eos, support, seed):
    """Uniformly permute selected labels with no fixed points; EOS stays fixed."""
    if not 0 <= eos < vocabulary_size or support not in (0,) and not (
            2 <= support < vocabulary_size):
        raise ValueError("invalid EOS or permutation support")
    result = list(range(vocabulary_size))
    if support == 0:
        return result
    rng = random.Random(seed)
    selected = rng.sample([i for i in range(vocabulary_size) if i != eos], support)
    shuffled = selected.copy()
    while True:
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(selected, shuffled)):
            break
    for old, new in zip(selected, shuffled):
        result[old] = new
    return result


def trial_plan():
    """Predeclare holdouts, interleaving magnitudes so the deadline cannot bias them."""
    yield dict(id="baseline", support=0, split="train", permutation_seed=0)
    yield dict(id="baseline_repeat", support=0, split="control", permutation_seed=0)
    magnitudes = (2, 32, 512, 4474, 8, 128, 2048)
    # Three independent permutations per scale, with the second held out. More
    # repetitions are allowed if this machine is faster than the initial estimate.
    for round_index in itertools.count():
        for support in magnitudes:
            split = "test" if round_index % 3 == 1 else "train"
            yield dict(id=f"rename_{round_index:03d}_{support:04d}",
                       support=support, split=split,
                       permutation_seed=810000 + round_index * 10000 + support)


def renamed_rows(rows, mapping):
    return [[mapping[token] for token in row] for row in rows]


def write_rows(path, rows):
    with path.open("x", encoding="ascii") as output:
        for row in rows:
            output.write(" ".join(map(str, row)) + "\n")


def write_int32(path, values):
    """Explicit little-endian int32, with no EOS or padding added."""
    data = array("i", values)
    if data.itemsize != 4:
        raise RuntimeError("this interpreter's C int is not 32 bits")
    if sys.byteorder != "little":
        data.byteswap()
    with path.open("xb") as output:
        data.tofile(output)


def checkpoint_bytes(path):
    files = list(path.glob("weight_*.bin"))
    expected = [path / f"weight_{i}.bin" for i in range(52)]
    if set(files) != set(expected):
        raise ValueError(f"expected exactly 52 canonical weights in {path}")
    result = b"".join(weight.read_bytes() for weight in expected)
    if len(result) != 48680 * 4:
        raise ValueError(f"unexpected checkpoint size in {path}")
    return result


def row_permuted_checkpoint(source, destination, mapping, width=10):
    """Exact architectural symmetry, NOT predicted fixed-seed training weights.

    E_new[pi(v)] = E_old[v]. The tied output head uses the same matrix, and
    every other parameter stays unchanged. Floating-point evaluation is still
    audited, since changing reduction order can alter softmax's last bits.
    """
    if sorted(mapping) != list(range(len(mapping))) or width <= 0:
        raise ValueError("mapping must be bijective and width positive")
    embedding = (source / "weight_0.bin").read_bytes()
    stride = 4 * width
    if len(embedding) != len(mapping) * stride:
        raise ValueError("embedding shape disagrees with permutation")
    destination.mkdir(parents=True, exist_ok=False)
    result = bytearray(len(embedding))
    for old, new in enumerate(mapping):
        result[new * stride:(new + 1) * stride] = embedding[old * stride:(old + 1) * stride]
    (destination / "weight_0.bin").write_bytes(result)
    for path in source.iterdir():
        if path.name != "weight_0.bin" and path.is_file():
            shutil.copy2(path, destination / path.name)


def run_command(command, log_path, *, deadline, environment):
    """Bound only this experiment's process group; never leave orphan trainers."""
    with log_path.open("x") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        try:
            return process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def common_flags(root):
    result = [str(root / "bin" / "memorize_general_facts"),
              f"--corpus={root / 'inputs/corpus.txt'}",
              f"--tokenizer={root / 'inputs/tokenizer'}", "--compact_vocabulary=true"]
    result.extend(f"--{key}={value}" for key, value in MODEL.items()
                  if key != "vocabulary_size")
    return result


def audit_predictions(path, rows, original_ids):
    original_rows = [[original_ids[token] for token in row] for row in rows]
    with path.open() as source:
        audit = verify_predictions(original_rows, source, eos_id=50256,
                                   vocabulary_size=50257, context_length=27)
    return dict(errors=audit.errors, targets=audit.targets,
                exact_sentences=audit.exact_sentences, sentences=audit.sentences,
                mean_loss=audit.mean_loss)


def snapshot(args):
    from tokenizers import Tokenizer

    root = args.run_dir
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    (inputs / "tokenizer").mkdir(parents=True)
    (root / "bin").mkdir()
    (root / "scripts").mkdir()
    shutil.copy2(args.binary, root / "bin/memorize_general_facts")
    # A copied Bazel executable loses its relative runpath. Pin its CUDA runtime
    # too, so later builds cannot silently alter a long-running experiment.
    linked = subprocess.check_output(["ldd", str(args.binary)], text=True)
    runtime = next((line.split("=>", 1)[1].strip().split()[0]
                    for line in linked.splitlines() if "libcudart.so" in line and "=>" in line), None)
    if runtime is None or not Path(runtime).is_file():
        raise ValueError("could not locate executable's CUDA runtime")
    shutil.copy2(runtime, root / "bin/libcudart.so.13")
    shutil.copy2(args.corpus, inputs / "corpus.txt")
    shutil.copy2(args.tokenizer / "tokenizer.json", inputs / "tokenizer/tokenizer.json")
    for name in SOURCE_NAMES:
        shutil.copy2(Path(__file__).parent / name, root / "scripts" / name)
    tokenizer = Tokenizer.from_file(str(inputs / "tokenizer/tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    original = [tokenizer.encode(line, add_special_tokens=False).ids
                for line in _corpus_lines((inputs / "corpus.txt").read_bytes())]
    ids = sorted({50256} | {token for row in original for token in row})
    if len(original) != 1024 or len(ids) != 4475 or sum(map(len, original)) != 14098:
        raise ValueError("unexpected corpus or compact vocabulary")
    if not all(5 <= len(row) <= 27 and 50256 not in row for row in original):
        raise ValueError("invalid sentence lengths or EOS in text")
    compact = {token: i for i, token in enumerate(ids)}
    rows = [[compact[token] for token in row] for row in original]
    write_rows(inputs / "base_tokens.tsv", rows)
    write_int32(inputs / "base_tokens.i32", itertools.chain.from_iterable(rows))
    write_int32(inputs / "sentence_lengths.i32", map(len, rows))
    write_rows(inputs / "compact_to_original.tsv", [ids])
    hashes = {str(path.relative_to(root)): _sha256(path)
              for folder in ("inputs", "bin", "scripts")
              for path in (root / folder).rglob("*") if path.is_file()}
    return rows, ids, hashes


def verify_trial(root, trial, rows, original_ids, environment, deadline, checkpoint, tag):
    directory = root / trial["id"]
    output = directory / tag
    command = common_flags(root) + ["--mode=infer_model", f"--verify_checkpoint={checkpoint}",
        f"--token_corpus={trial['token_corpus']}", f"--output_dir={output}", "--batch_size=32"]
    code = run_command(command, directory / f"{tag}.log", deadline=deadline,
                       environment=environment)
    if code not in (0, 2):
        raise RuntimeError(f"native {tag} failed: {code}")
    audit = audit_predictions(output / "final_predictions.tsv", rows, original_ids)
    if (code == 0) != (audit["errors"] == 0):
        raise ValueError("native verifier exit status disagrees with independent audit")
    return audit


def update_report(root, environment):
    """Refresh a self-contained report without discarding completed GPU work."""
    try:
        result = subprocess.run(
            [sys.executable, str(root / "scripts/analyze_dataset_weights.py"),
             "--summary", str(root / "summary.json"), "--output", str(root / "analysis.html")],
            env=environment, check=False, timeout=60)
    except (subprocess.TimeoutExpired, OSError) as error:
        print(f"WARNING: report unavailable ({error}); raw pairs are preserved", flush=True)
        return
    if result.returncode:
        print(f"WARNING: analysis exited {result.returncode}; raw pairs are preserved", flush=True)


def run(args):
    free = shutil.disk_usage(args.run_dir.parent).free
    if free < 10 * 1024**3:
        raise ValueError("require at least 10 GiB free before this experiment")
    rows, original_ids, hashes = snapshot(args)
    root = args.run_dir
    environment = dict(os.environ, LD_LIBRARY_PATH=str(root / "bin"),
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    fingerprint = hashlib.sha256(json.dumps(dict(model=MODEL, schedule=SCHEDULE),
                                            sort_keys=True).encode()).hexdigest()
    start = time.monotonic()
    deadline = start + args.duration_seconds
    summary = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                   duration_seconds=args.duration_seconds, disk_free_bytes=free,
                   model=MODEL, schedule=SCHEDULE, training_fingerprint=fingerprint,
                   hashes=hashes, base_tokens=str(root / "inputs/base_tokens.tsv"), trials=[])
    _write_summary(root / "summary.json", summary)
    baseline = None
    durations = []
    try:
        for specification in trial_plan():
            # Do not knowingly start an endpoint that cannot use the same step
            # schedule. Reserve time for audits and the final CPU report.
            estimate = max(durations, default=390) * 1.12 + 30
            if deadline - time.monotonic() < estimate:
                break
            if args.max_trials and len(summary["trials"]) >= args.max_trials:
                break
            for relative, digest in hashes.items():
                if _sha256(root / relative) != digest:
                    raise ValueError(f"pinned experiment input changed: {relative}")
            trial_start = time.monotonic()
            trial = dict(specification, family="token_rename", status="running",
                         training_fingerprint=fingerprint, parameters=48680)
            directory = root / trial["id"]
            directory.mkdir()
            mapping = permutation(4475, 4474, trial["support"], trial["permutation_seed"])
            transformed = renamed_rows(rows, mapping)
            token_file = directory / "tokens.tsv"
            permutation_file = directory / "permutation.tsv"
            write_rows(token_file, transformed)
            write_rows(permutation_file, [mapping])
            write_int32(directory / "tokens.i32", itertools.chain.from_iterable(transformed))
            trial.update(token_corpus=str(token_file), permutation=str(permutation_file),
                         token_sha256=_sha256(token_file), changed_positions=sum(
                             a != b for old, new in zip(rows, transformed) for a, b in zip(old, new)))
            summary["trials"].append(trial)
            _write_summary(root / "summary.json", summary)
            print(f"START {trial['id']} split={trial['split']} support={trial['support']}", flush=True)
            command = common_flags(root) + ["--mode=train_model",
                f"--token_corpus={token_file}", f"--checkpoint_dir={directory / 'checkpoints'}",
                f"--output_dir={directory / 'training'}"]
            command.extend(f"--{key}={str(value).lower()}" for key, value in SCHEDULE.items())
            (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            code = run_command(command, directory / "train.log", deadline=deadline - 30,
                               environment=environment)
            if code not in (0, 2):
                raise RuntimeError(f"training failed: {code}; see {directory / 'train.log'}")
            result = _read_result(directory / "training/layers_4/result.txt")
            if int(result["step"]) != SCHEDULE["steps"] or int(result["parameters"]) != 48680:
                raise ValueError("native result disagrees with fixed-step experiment")
            final = Path(result["checkpoint"])
            initial = directory / "checkpoints/layers_4/step_0"
            initial_bytes = checkpoint_bytes(initial)
            checkpoint_bytes(final)
            trial.update(initial_checkpoint=str(initial), final_checkpoint=str(final),
                         initial_sha256=hashlib.sha256(initial_bytes).hexdigest(),
                         step=int(result["step"]), success=bool(int(result["success"])),
                         errors=int(result["errors"]), first_memorized_step=int(result["first_memorized_step"]),
                         checkpoints={str(step): str(directory / f"checkpoints/layers_4/step_{step}")
                                      for step in (30000, 60000, 90000, 120000)})
            if baseline and trial["initial_sha256"] != baseline["initial_sha256"]:
                raise ValueError("initial weights differ from baseline")
            native_audit = audit_predictions(directory / "training/layers_4/final_predictions.tsv",
                                             transformed, original_ids)
            audit = verify_trial(root, trial, transformed, original_ids, environment,
                                 deadline - 15, final, "verification")
            if audit["errors"] != trial["errors"] or audit != native_audit:
                raise ValueError("fresh verification disagrees with training checkpoint")
            trial["audit"] = audit
            if baseline is None:
                baseline = trial
            if trial["id"] == "baseline_repeat":
                trial["identical_to_baseline"] = checkpoint_bytes(final) == checkpoint_bytes(
                    Path(baseline["final_checkpoint"]))
                if not trial["identical_to_baseline"]:
                    raise ValueError("duplicate baseline was not bitwise reproducible")
            if trial["support"]:
                symmetry_checkpoint = directory / "symmetry/step_120000"
                row_permuted_checkpoint(Path(baseline["final_checkpoint"]), symmetry_checkpoint, mapping)
                symmetry = verify_trial(root, trial, transformed, original_ids, environment,
                                        deadline - 15, symmetry_checkpoint, "symmetry_verification")
                trial["symmetry_errors"] = symmetry["errors"]
                trial["symmetry_checkpoint"] = str(symmetry_checkpoint)
            trial["status"] = "verified" if trial["success"] else "not_memorized"
            trial["seconds"] = time.monotonic() - trial_start
            durations.append(trial["seconds"])
            _write_summary(root / "summary.json", summary)
            print(f"DONE {trial['id']} errors={trial['errors']} seconds={trial['seconds']:.1f}", flush=True)
            update_report(root, environment)
        summary["status"] = "complete"
    except subprocess.TimeoutExpired:
        summary["status"] = "deadline_reached"
        if summary["trials"] and summary["trials"][-1]["status"] == "running":
            summary["trials"][-1]["status"] = "incomplete"
    except BaseException as error:
        summary["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        summary["error"] = str(error)
        if summary["trials"] and summary["trials"][-1]["status"] == "running":
            summary["trials"][-1]["status"] = "incomplete"
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - start
        _write_summary(root / "summary.json", summary)
        # Reports are always recoverable from saved pairs, including partial runs.
        update_report(root, environment)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "corpus", "tokenizer", "run_dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--duration_seconds", type=float, default=10800)
    parser.add_argument("--max_trials", type=int, default=0,
                        help="Optional test cap; zero runs until the deadline")
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration_seconds) or args.duration_seconds < 600:
        parser.error("duration_seconds must be finite and at least 600")
    if args.max_trials < 0:
        parser.error("max_trials must be nonnegative")
    for name in ("binary", "corpus", "tokenizer", "run_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    repository = Path(__file__).resolve().parents[2]
    if args.run_dir == repository or repository in args.run_dir.parents:
        parser.error("run_dir must be outside the source tree")
    return args


if __name__ == "__main__":
    run(parse_args())
