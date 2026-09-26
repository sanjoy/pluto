#!/usr/bin/env bash
# Train a fresh, fully verified source model before running the A3 readout puzzle.
# See REPRODUCE_PUZZLE.md for installation, interpretation, and output files.

usage() {
  cat <<'EOF'
Usage: reproduce_puzzle.sh --run_dir /absolute/path/to/new-run [options]

Requires Linux, a supported NVIDIA GPU, CUDA 13.3 at /usr/local/cuda,
Bazel 9.2.0, a C++ compiler, Python 3.9+ with venv, curl, and network access.
Creates an isolated Python environment and downloads the GPT-2 tokenizer only.
Trains a 4-block, width-10, FF20, context-27 model from scratch, independently
verifies all 1,024 completions, then trains the A3 puzzle's replacement MLP.

  --run_dir PATH          Required NEW directory outside this checkout.
  --model_steps N         Source training cap and cosine horizon (120000).
  --puzzle_steps N        Replacement MLP updates; 0 evaluates only (300000).
  --mlp_width N           Requested replacement MLP inner width (150; floor 66).
  --training_timeout N    Seconds for source training AND verification (14400).
                         At least 301; 300 seconds reserved for verification.
  --help                 Show this message without building or using a GPU.

All flags accept either --name=value or --name value. No existing checkpoint
is needed. A failed memorization/verification stops the workflow, leaving its
evidence intact; it never silently substitutes another checkpoint.
EOF
}

# The search driver also exits successfully for ordinary budget exhaustion.
# Only its fully verified trial, backed by both audit reports, is acceptable.
# Keep this gate callable independently so its failure paths have CPU-only tests.
select_verified_checkpoint() {
  python3 - "$1" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
try:
    summary = json.loads((root / "summary.json").read_text())
    trials = summary["trials"]
    if summary["status"] != "completed" or len(trials) != 1:
        raise ValueError("source training did not complete exactly one trial")
    trial = trials[0]
    for key, expected in dict(status="verified", layers=4, width=10,
                              feed_forward_width=20, parameters=48680,
                              errors=0).items():
        if trial.get(key) != expected:
            raise ValueError(f"source trial {key}={trial.get(key)!r}, expected {expected!r}")
    directory = root / "trial_000_L4_W10_FF20"
    step = trial["step"]
    if type(step) is not int or step < 0:
        raise ValueError("invalid final checkpoint step")
    checkpoint = directory / "checkpoints" / "layers_4" / f"step_{step}"
    if (Path(trial["directory"]).resolve() != directory
            or Path(trial["checkpoint"]).resolve() != checkpoint
            or not checkpoint.is_dir()):
        raise ValueError("missing or unexpected verified checkpoint")
    native = trial["training_result"]
    for key, expected in dict(success="1", step=str(step), errors="0",
                              targets="10002", context_length="27",
                              vocabulary="4475", parameters="48680",
                              checkpoint=str(checkpoint)).items():
        if native.get(key) != expected:
            raise ValueError(f"inconsistent native training result: {key}")
    for name, targets in (("prediction_verification.json", 10002),
                          ("greedy_verification.json", None)):
        audit = json.loads((directory / name).read_text())
        expected = dict(errors=0, sentences=1024, exact_sentences=1024)
        if targets is not None:
            expected["targets"] = targets
            if audit.get("success") is not True:
                raise ValueError("independent prediction audit did not succeed")
        elif audit.get("incorrect_lines_1based") != []:
            raise ValueError("greedy audit lists incorrect completions")
        if any(audit.get(key) != value for key, value in expected.items()):
            raise ValueError(f"incomplete or failing {name}")
    print(checkpoint)
except (OSError, ValueError, KeyError, TypeError) as error:
    sys.exit(f"No fully memorized checkpoint: {error}. Inspect {root}/summary.json and trial logs; puzzle NOT started.")
PY
}

main() {
  set -euo pipefail
  local run_dir="" model_steps=120000 puzzle_steps=300000 mlp_width=150
  local training_timeout=14400 option value
  while (($#)); do
    option=${1%%=*}
    case "$option" in
      --help|-h) usage; return 0 ;;
      --run_dir|--model_steps|--puzzle_steps|--mlp_width|--training_timeout)
        if [[ $1 == *=* ]]; then
          value=${1#*=}
          shift
        else
          if (($# < 2)); then
            printf 'Missing value for %s\n' "$option" >&2
            return 2
          fi
          value=$2
          shift 2
        fi
        case "$option" in
          --run_dir) run_dir=$value ;;
          --model_steps) model_steps=$value ;;
          --puzzle_steps) puzzle_steps=$value ;;
          --mlp_width) mlp_width=$value ;;
          --training_timeout) training_timeout=$value ;;
        esac ;;
      *) printf 'Unknown argument: %s\n' "$1" >&2; return 2 ;;
    esac
  done
  if [[ -z $run_dir || $run_dir == *$'\n'* ]]; then
    printf 'Provide --run_dir with a NEW directory outside the checkout.\n' >&2
    return 2
  fi
  # Reject malformed and overflowing numbers before shell arithmetic or work.
  for value in "$model_steps" "$puzzle_steps" "$mlp_width" "$training_timeout"; do
    if [[ ! $value =~ ^(0|[1-9][0-9]{0,8})$ ]]; then
      printf 'Expected a nonnegative integer below 1000000000: %s\n' "$value" >&2
      return 2
    fi
  done
  if ((model_steps == 0 || mlp_width == 0 || training_timeout <= 300)); then
    printf 'model_steps/mlp_width must be positive; training_timeout must exceed 300.\n' >&2
    return 2
  fi

  local script_dir repo_root
  script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  repo_root=$(cd -- "$script_dir/../.." && pwd)
  # Resolve paths without creating anything, including symlinks into the repo.
  run_dir=$(python3 - "$run_dir" "$repo_root" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1]).expanduser()
if path.exists() or path.is_symlink():
    sys.exit(f"Refusing existing run directory: {path}")
path = path.resolve()
repository = Path(sys.argv[2]).resolve()
if path == repository or repository in path.parents:
    sys.exit("run_dir must be outside the repository")
print(path)
PY
  )
  export PATH="/usr/local/cuda/bin:$PATH"
  # Both the native executable and its CUDA child processes inherit this.
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  local tool
  for tool in bazel git curl sha256sum python3 nvcc tileiras nvidia-smi; do
    if ! command -v "$tool" >/dev/null; then
      printf 'Missing prerequisite: %s. See %s/REPRODUCE_PUZZLE.md\n' "$tool" "$script_dir" >&2
      return 1
    fi
  done
  # mkdir without -p on the leaf also rejects races with another invocation.
  mkdir -p -- "$(dirname -- "$run_dir")"
  mkdir -- "$run_dir"
  trap 'printf "Workflow failed; inspect logs in %s (nothing was deleted).\n" "$run_dir" >&2' ERR
  cd -- "$repo_root"
  {
    date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
    printf 'revision='; git rev-parse HEAD
    git status --short
    printf 'model_steps=%s\npuzzle_steps=%s\nmlp_width=%s\ntraining_timeout=%s\n' \
      "$model_steps" "$puzzle_steps" "$mlp_width" "$training_timeout"
    nvcc --version
    tileiras --version
    bazel --version
    python3 --version
    nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
    df -h -- "$run_dir"
  } | tee "$run_dir/provenance.txt"

  printf '\nPreparing isolated Python environment and pinned tokenizer...\n'
  python3 -m venv "$run_dir/.venv"
  local python="$run_dir/.venv/bin/python"
  "$python" -m pip install --disable-pip-version-check 'tokenizers==0.22.0' \
    2>&1 | tee "$run_dir/setup.log"
  "$python" -m pip freeze > "$run_dir/python-packages.txt"
  mkdir -- "$run_dir/tokenizer"
  # Pin both the upstream revision and bytes; download no pretrained weights.
  local revision=607a30d783dfa663caf39e06633721c8d4cfcd7e
  local digest=8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6
  curl --fail --location --retry 3 \
    "https://huggingface.co/openai-community/gpt2/resolve/$revision/tokenizer.json" \
    --output "$run_dir/tokenizer/tokenizer.json"
  (cd -- "$run_dir/tokenizer"; printf '%s  tokenizer.json\n' "$digest" | sha256sum --check -)

  printf '\nBuilding the optimized C++ experiment...\n'
  bazel build -c opt //src/llm/experiments/memorize_general_facts:memorize_general_facts \
    2>&1 | tee "$run_dir/build.log"
  local binary="$repo_root/bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts"
  local deadline
  deadline=$("$python" -c 'import sys,time; print(time.time()+int(sys.argv[1]))' "$training_timeout")
  printf '\nTraining from scratch, then verifying all 1,024 facts...\n'
  "$python" -B "$script_dir/run_compact_size_sweep.py" \
    --binary="$binary" --corpus="$repo_root/testdata/general_facts_dataset.txt" \
    --tokenizer="$run_dir/tokenizer" --run_dir="$run_dir/model" \
    --candidates="4:10:20:$model_steps:0.0012" \
    --seed=1337 --batch_size=32 --context_length=27 --eval_every=256 \
    --deadline_unix="$deadline" --verification_reserve=300 \
    2>&1 | tee "$run_dir/model.log"
  local checkpoint
  checkpoint=$(select_verified_checkpoint "$run_dir/model")
  printf '%s\n' "$checkpoint" > "$run_dir/checkpoint.txt"
  printf '\nVerified checkpoint: %s\nStarting the A3 readout puzzle...\n' "$checkpoint"
  # Use the driver's immutable executable and input snapshots for the puzzle,
  # even if another terminal rebuilds the checkout while training is running.
  "$run_dir/model/inputs/memorize_general_facts" \
    --mode=puzzle --train_mlp --puzzle_checkpoint="$checkpoint" \
    --tokenizer="$run_dir/model/inputs" --corpus="$run_dir/model/inputs/corpus.txt" \
    --output_dir="$run_dir/puzzle" \
    --layers=4 --model_width=10 --attention_heads=1 --feed_forward_width=20 \
    --context_length=27 --compact_vocabulary=true --batch_size=32 \
    --mlp_width="$mlp_width" --steps="$puzzle_steps" --eval_every=1000 \
    --seed=3 --learning_rate=0.01 2>&1 | tee "$run_dir/puzzle.log"
  printf '\nDone.\nCheckpoint: %s\nPlots: %s/puzzle/puzzle.html\nReadout metrics: %s/puzzle/training.tsv\nProvenance: %s/puzzle/run.txt\n' \
    "$checkpoint" "$run_dir" "$run_dir" "$run_dir"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
