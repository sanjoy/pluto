#!/usr/bin/env python3
"""Print a read-only, evidence-checked Markdown width/depth search report.

Pass one or more width_depth_search_summary.json files. Optional --labels gives
comma-separated names for those runs; otherwise each manifest's parent directory
is its name. Completed trials are checked against their saved training result,
fresh-process verification result, independently audited predictions, and input
snapshots. This does not rerun inference or certify the present checkpoint bytes.
The original tokenizer snapshots must remain available for their hash check.

Running/error trials and explicitly skipped configurations remain unverified or
untested, never failed. Every verified observation keeps its run, seed, budget,
head configuration, and binary identity. The pooled success frontier is an
existence result across these protocols, not a fixed-budget impossibility bound.
Nothing is written except the final Markdown report to stdout; any unverifiable
completed trial aborts before printing a partial report.
"""

import argparse
import json
import math
from pathlib import Path
import re
import sys

# Keep invoking the reporting CLI read-only even without Python's -B flag.
sys.dont_write_bytecode = True

from run_depth_search import _read_result, _require_fields, _sha256


SCHEDULE_FIELDS = (
    "steps", "training_seconds", "batch_size", "seed", "learning_rate",
    "warmup_steps", "eval_every", "checkpoint_every",
)
SHAPE_FIELDS = ("layers", "width", "heads", "feed_forward_width", "parameters")
SKIP_REASONS = {
    "dominated_by_verified_success",
    "untested_after_wider_budget_failure_not_a_failure_claim",
}


def _integer(value, label, minimum=0):
    # bool is an int subclass, but an accidental JSON true is not a count.
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value, label, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return result


def _hash(value, label):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _command_flags(command):
    if not isinstance(command, list) or not command:
        raise ValueError("Recorded command must be a nonempty argument list")
    result = {}
    for argument in command:
        if not isinstance(argument, str):
            raise ValueError("Recorded command arguments must be strings")
        if argument.startswith("--"):
            key, separator, value = argument[2:].partition("=")
            if not separator or key in result:
                raise ValueError("Malformed or duplicate recorded command flag")
            result[key] = value
    return result


def _validate_trial(trial, run):
    """Recheck completed evidence instead of trusting cached manifest summaries."""
    configuration = run["configuration"]
    layers = _integer(trial["layers"], "layers", 1)
    width = _integer(trial["width"], "width", 1)
    heads = _integer(trial["heads"], "heads", 1)
    head_dim = _integer(trial["head_dim"], "head_dim", 1)
    if not 1 <= layers <= 8 or width % heads:
        raise ValueError("Invalid recorded GPT-2 architecture")
    if head_dim != width // heads:
        raise ValueError("Recorded head dimension is inconsistent")
    expected_heads = configuration.get("attention_heads", 0) or width // math.gcd(width, 64)
    if heads != expected_heads:
        raise ValueError("Recorded attention heads disagree with the search configuration")
    parameters = 51298 * width + layers * (12 * width * width + 13 * width)
    if (trial["feed_forward_width"] != 4 * width
        or trial["parameters"] != parameters
        or trial["expected_parameters"] != parameters):
        raise ValueError("Recorded architecture/parameter count is inconsistent")
    expected_output = run["path"].parent / f"width_{width}" / f"layers_{layers}"
    output = Path(trial["output_dir"]).resolve()
    if output != expected_output:
        raise ValueError("Trial output directory does not match the manifest configuration")
    verification = output / "independent_verification"
    audit_path = verification / "verified_predictions.json"
    if Path(trial["prediction_artifact_verification"]).resolve() != audit_path:
        raise ValueError("Recorded prediction audit is outside its trial")
    training_path = output / "result.txt"
    native_path = verification / "result.txt"
    training = _read_result(training_path)
    native = _read_result(native_path)
    if training != trial["training_result"] or native != trial["checkpoint_verification"]:
        raise ValueError("Saved native results disagree with the manifest's recorded results")
    success = trial["status"] == "verified_success"
    expected_shape = {key: trial[key] for key in SHAPE_FIELDS}
    _require_fields(training, {**expected_shape, "success": int(success)}, training_path)
    _require_fields(native, expected_shape, native_path)
    step = _integer(trial["step"], "completed step")
    if step > configuration["steps"]:
        raise ValueError("Completed step exceeds its named training budget")
    expected_checkpoint = (Path(configuration["checkpoint_dir"]).resolve()
                           / f"width_{width}" / f"layers_{layers}" / f"step_{step}")
    if Path(trial["checkpoint"]).resolve() != expected_checkpoint:
        raise ValueError("Checkpoint path disagrees with its configuration or step")
    _require_fields(training, {"step": step, "checkpoint": expected_checkpoint}, training_path)
    _require_fields(native, {"checkpoint": expected_checkpoint}, native_path)
    if training.get("reached_time_limit") not in ("0", "1"):
        raise ValueError("Invalid training time-limit indicator")
    timed_out = training["reached_time_limit"] == "1"
    if timed_out and configuration["training_seconds"] == 0:
        raise ValueError("Trial claims a disabled time limit was reached")
    if not success and step < configuration["steps"] and not timed_out:
        raise ValueError("Failed trial did not reach its prescribed training budget")
    errors = _integer(trial["errors"], "errors")
    if errors > 10002 or (errors == 0) != success:
        raise ValueError("Trial status and error count disagree")
    for result, path in ((training, training_path), (native, native_path)):
        _require_fields(result, {"errors": errors, "targets": 10002}, path)
    _require_fields(native, {"sentences": 1024}, native_path)
    exact = int(native["exact_sentences"])
    if not 0 <= exact <= 1024 or (exact == 1024) != success or 1024 - exact > errors:
        raise ValueError("Invalid exact-sentence count")
    native_loss = _finite(native["mean_loss"], "native mean loss")
    for filename, key in (("corpus.txt", "corpus_sha256"),
                          ("tokenizer.json", "tokenizer_sha256")):
        if _sha256(output / filename) != run[key]:
            raise ValueError(f"Changed {filename} snapshot")
    prediction_hash = _sha256(verification / "final_predictions.tsv")
    if _sha256(output / "final_predictions.tsv") != prediction_hash:
        raise ValueError("Training and fresh-process prediction artifacts differ")
    audit = json.loads(audit_path.read_text())
    if not isinstance(audit, dict):
        raise ValueError("Prediction audit must be a JSON object")
    expected_audit = {
        "success": success, "errors": errors, "targets": 10002,
        "sentences": 1024, "exact_sentences": exact,
        "corpus_sha256": run["corpus_sha256"],
        "tokenizer_sha256": run["tokenizer_sha256"],
        "predictions_sha256": prediction_hash,
    }
    for key, expected in expected_audit.items():
        if type(audit.get(key)) is not type(expected) or audit[key] != expected:
            raise ValueError(f"Prediction audit disagrees on {key}")
    loss = _finite(audit["mean_loss_nats"], "audited mean loss")
    if not math.isclose(native_loss, loss, rel_tol=1e-5, abs_tol=1e-12):
        raise ValueError("Native and independently audited mean losses disagree")
    commands = trial["commands"]
    if not isinstance(commands, list) or len(commands) != 3:
        raise ValueError("Completed trial must record training and both verification commands")
    training_flags = _command_flags(commands[0])
    for field in SCHEDULE_FIELDS:
        _require_fields(training_flags, {field: configuration[field]}, "training command")
    shape_flags = {"layers": layers, "model_width": width,
                   "attention_heads": heads, "feed_forward_width": 4 * width}
    _require_fields(training_flags, {**shape_flags, "search": "false"}, "training command")
    _require_fields(_command_flags(commands[1]), {
        **shape_flags, "verify_checkpoint": expected_checkpoint,
        "batch_size": configuration["batch_size"], "seed": configuration["seed"],
    }, "verification command")
    _require_fields(_command_flags(commands[2]), {
        "corpus": output / "corpus.txt", "tokenizer": output / "tokenizer.json",
        "predictions": verification / "final_predictions.tsv",
    }, "prediction audit command")
    return {**trial, "run": run["label"], "mean_loss": loss,
            "exact_sentences": exact, "seed": configuration["seed"]}


def load_runs(paths, labels=None):
    """Load all evidence before rendering; missing/corrupt completed trials fail."""
    paths = [Path(path).expanduser().resolve() for path in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Provide one or more distinct summary paths")
    if labels is None:
        labels = [path.parent.name for path in paths]
    if len(labels) != len(paths) or len(set(labels)) != len(labels) or any(not label for label in labels):
        raise ValueError("Provide one unique, nonempty label per summary")
    runs = []
    input_hashes = None
    for path, label in zip(paths, labels):
        manifest = json.loads(path.read_text())
        if not isinstance(manifest, dict) or manifest["status"] not in ("running", "completed", "error"):
            raise ValueError(f"{path}: invalid search manifest")
        configuration = manifest["configuration"]
        if Path(configuration["output_dir"]).resolve() != path.parent:
            raise ValueError("Manifest location does not match its recorded output directory")
        for field in ("steps", "warmup_steps"):
            _integer(configuration[field], field)
        for field in ("batch_size", "eval_every", "checkpoint_every"):
            _integer(configuration[field], field, 1)
        if type(configuration["seed"]) is not int:
            raise ValueError("seed must be an integer")
        _finite(configuration["training_seconds"], "time budget")
        if _finite(configuration["learning_rate"], "learning rate") == 0:
            raise ValueError("learning rate must be positive")
        attention_heads = _integer(configuration.get("attention_heads", 0), "attention_heads")
        if attention_heads and any(_integer(width, "configured width", 1) % attention_heads
                                   for width in configuration["widths"]):
            raise ValueError("attention_heads must divide every configured width")
        digests = {key: _hash(manifest[key], key) for key in
                   ("corpus_sha256", "tokenizer_sha256", "binary_sha256")}
        if input_hashes is None:
            input_hashes = digests["corpus_sha256"], digests["tokenizer_sha256"]
        if input_hashes != (digests["corpus_sha256"], digests["tokenizer_sha256"]):
            raise ValueError("Cannot pool searches with different corpus/tokenizer identities")
        run = {"path": path, "label": label, "configuration": configuration,
               "status": manifest["status"], **digests, "trials": [], "ignored": [],
               "skipped": manifest["skipped_configurations"]}
        seen = set()
        for trial in manifest["trials"]:
            point = (trial["layers"], trial["width"])
            if point in seen:
                raise ValueError("Duplicate configuration in one search manifest")
            seen.add(point)
            status = trial["status"]
            if status in ("verified_success", "verified_budget_failure"):
                if trial["phase"] != "verified":
                    raise ValueError("Completed trial has an unverified phase")
                run["trials"].append(_validate_trial(trial, run))
            elif status in ("running", "error"):
                run["ignored"].append({"layers": trial["layers"], "width": trial["width"],
                                       "status": status, "phase": trial["phase"]})
            else:
                raise ValueError(f"Unknown trial status: {status}")
        for skipped in run["skipped"]:
            if skipped["reason"] not in SKIP_REASONS:
                raise ValueError("Unknown skipped-configuration reason")
            point = (skipped["layers"], skipped["width"])
            if point in seen:
                raise ValueError("Configuration appears as both tested and skipped")
            seen.add(point)
        runs.append(run)
    return runs


def frontier(trials):
    """Preserve each verified observation, including repeated successes/seeds."""
    successes = [trial for trial in trials if trial["status"] == "verified_success"]
    return [trial for trial in successes if not any(
        other["layers"] <= trial["layers"] and other["width"] <= trial["width"]
        and (other["layers"] < trial["layers"] or other["width"] < trial["width"])
        for other in successes)]


def _cell(value):
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(runs):
    """Render validated observations without collapsing different protocols."""
    trials = [trial for run in runs for trial in run["trials"]]
    trials.sort(key=lambda trial: (trial["layers"], trial["width"], trial["run"], trial["heads"]))
    lines = ["# General-facts width/depth results", "",
             "Each success predicts all 10,002 suffix/EOS targets for 1,024 five-token prompts correctly. "
             "Results below were checked against saved fresh-process verification and prediction audits; no GPU inference was rerun.", "",
             "## Named training budgets", ""]
    lines.append(_table(
        ["Run", "Step cap", "Time cap (s)", "Batch", "Seed", "LR", "Warmup", "Eval every", "Save every", "Binary SHA256 prefix"],
        [[run["label"], *[run["configuration"][field] for field in SCHEDULE_FIELDS], run["binary_sha256"][:12]]
         for run in runs]))
    lines += ["", "Run names distinguish protocols even when their budgets happen to match. Heads and head dimensions remain explicit below.",
              "", "## Verified tested configurations", ""]
    headers = ["Run", "Layers", "Width", "Heads × dim", "FF width", "Parameters", "Outcome", "Errors / 10,002", "Step", "Mean loss"]
    def row(trial):
        return [trial["run"], trial["layers"], trial["width"], f"{trial['heads']} × {trial['head_dim']}",
                trial["feed_forward_width"], f"{trial['parameters']:,}",
                "memorized" if trial["status"] == "verified_success" else "budget exhausted",
                trial["errors"], trial["step"], f"{trial['mean_loss']:.8g}"]
    lines.append(_table(headers, [row(trial) for trial in trials]) if trials else "No completed, artifact-verified trials yet.")
    points = frontier(trials)
    lines += ["", "## Pooled measured depth/width frontier", "",
              "Only verified successes are candidates. This pools existence evidence across the named budgets/seeds; "
              "it does not establish a common-budget capacity boundary or prove untested models cannot memorize.", ""]
    lines.append(_table(headers, [row(trial) for trial in points]) if points else "No verified success yet.")
    minimum = min((trial["parameters"] for trial in points), default=None)
    lines += ["", "## Minimum-parameter verified success", ""]
    if minimum is None:
        lines.append("Not yet established by the supplied runs.")
    else:
        lines.append(_table(headers, [row(trial) for trial in points if trial["parameters"] == minimum]))
    lines += ["", "## Unverified and untested configurations", ""]
    pending = [[run["label"], trial["layers"], trial["width"],
                f"unverified: {trial['status']} ({trial['phase']}); not counted as a budget failure"]
               for run in runs for trial in run["ignored"]]
    pending += [[run["label"], point["layers"], point["width"],
                 "untested: skipped as dominated" if point["reason"] == "dominated_by_verified_success"
                 else "untested: search moved on after a wider model exhausted its budget"]
                for run in runs for point in run["skipped"]]
    lines.append(_table(["Run", "Layers", "Width", "Disposition"], pending) if pending else "None recorded.")
    lines += ["", "Configurations absent from these manifests are also untested, not inferred failures.",
              "", "## Evidence", ""]
    for run in runs:
        lines.append(f"- {_cell(run['label'])}: [{run['path'].name}](<{run['path']}>) (search status: {run['status']}).")
    lines += ["", f"Corpus SHA256: `{runs[0]['corpus_sha256']}`.",
              f"Tokenizer SHA256: `{runs[0]['tokenizer_sha256']}`.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--labels", help="Comma-separated unique run names, in summary order")
    args = parser.parse_args(argv)
    labels = args.labels.split(",") if args.labels is not None else None
    try:
        report = render_markdown(load_runs(args.summaries, labels))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Cannot validate width/depth report: {error}", file=sys.stderr)
        return 1
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
