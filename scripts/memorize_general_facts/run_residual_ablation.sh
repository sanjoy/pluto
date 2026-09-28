#!/usr/bin/env bash
# Compare otherwise identical A3 MLP readouts with and without skip connections.
# The source model and tied head remain frozen; all artifacts stay outside Git.

usage() {
  cat <<'EOF'
Usage: run_residual_ablation.sh --checkpoint PATH --run_dir NEW_PATH [options]

Requires a fully memorized 4-block, width-10, FF20, context-27 compact-vocabulary
checkpoint and a working Pluto CUDA/Bazel environment. Builds an optimized
binary, snapshots its inputs, then trains a paired residual/no-residual ablation.
Both jobs run concurrently by default; compare accuracy, not concurrent timing.

  --checkpoint PATH       Required source checkpoint (including compact mapping).
  --run_dir PATH          Required NEW artifact directory outside the checkout.
  --tokenizer PATH        GPT-2 tokenizer directory (PLUTO_GPT2_TOKENIZER_DIR,
                          otherwise $HOME/datasets/tokenizer/gpt2).
  --corpus PATH           Defaults to testdata/general_facts_dataset.txt.
  --steps N               Updates per condition; 0 tests initialization (300000).
  --eval_every N          Full-corpus evaluation interval (1000).
  --batch_size N          Facts per batch (32).
  --mlp_width N           Inner width for a single MLP (150; minimum floor 66).
  --seed N                Same initialization and data-order seed for both (3).
  --learning_rate RATE    Same Adam initial rate and 0.1 cosine floor (0.01).
  --stacked               Use five 10/150/10 MLPs instead of one.
  --iso_params            Requires --stacked; match the original suffix's total
                          parameter count, including input/final LayerNorms.
                          Widths 12,12,12,11,11 give 1388 vs. 1380 parameters;
                          integer widths round the budget upward by 8.
  --serial                Run one condition after the other.
  --help                  Print this message without building or using a GPU.

Value flags accept either --name=value or --name value. Only the replacement
MLP skips change: pre-MLP and final LayerNorms remain trainable in both runs.
See residual.log/no_residual.log and each condition's training.tsv for results.
EOF
}

main() {
  set -euo pipefail
  local script_dir repo_root
  script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  repo_root=$(cd -- "$script_dir/../.." && pwd)
  local checkpoint="" run_dir=""
  local tokenizer=${PLUTO_GPT2_TOKENIZER_DIR:-"$HOME/datasets/tokenizer/gpt2"}
  local corpus="$repo_root/testdata/general_facts_dataset.txt"
  local steps=300000 eval_every=1000 batch_size=32 mlp_width=150 seed=3
  local learning_rate=0.01 stacked=0 iso_params=0 serial=0 option value
  while (($#)); do
    option=${1%%=*}
    case "$option" in
      --help|-h) usage; return 0 ;;
      --stacked|--iso_params|--serial)
        if [[ $1 == *=* ]]; then
          printf '%s does not take a value.\n' "$option" >&2
          return 2
        fi
        case "$option" in
          --stacked) stacked=1 ;;
          --iso_params) iso_params=1 ;;
          --serial) serial=1 ;;
        esac
        shift ;;
      --checkpoint|--run_dir|--tokenizer|--corpus|--steps|--eval_every|--batch_size|--mlp_width|--seed|--learning_rate)
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
          --checkpoint) checkpoint=$value ;;
          --run_dir) run_dir=$value ;;
          --tokenizer) tokenizer=$value ;;
          --corpus) corpus=$value ;;
          --steps) steps=$value ;;
          --eval_every) eval_every=$value ;;
          --batch_size) batch_size=$value ;;
          --mlp_width) mlp_width=$value ;;
          --seed) seed=$value ;;
          --learning_rate) learning_rate=$value ;;
        esac ;;
      *) printf 'Unknown argument: %s\n' "$1" >&2; return 2 ;;
    esac
  done
  if [[ -z $checkpoint || -z $run_dir ]]; then
    printf 'Both --checkpoint and --run_dir are required.\n' >&2
    return 2
  fi
  for value in "$steps" "$eval_every" "$batch_size" "$mlp_width" "$seed"; do
    if [[ ! $value =~ ^(0|[1-9][0-9]{0,8})$ ]]; then
      printf 'Expected a nonnegative integer below 1000000000: %s\n' "$value" >&2
      return 2
    fi
  done
  if ((eval_every == 0 || batch_size == 0 || mlp_width == 0)); then
    printf 'eval_every, batch_size, and mlp_width must be positive.\n' >&2
    return 2
  fi
  if ((stacked && mlp_width != 150)); then
    printf 'The fixed five-MLP stack requires --mlp_width=150.\n' >&2
    return 2
  fi
  if ((iso_params && !stacked)); then
    printf '%s\n' '--iso_params requires --stacked.' >&2
    return 2
  fi
  if [[ ! $learning_rate =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] ||
      ! awk -v rate="$learning_rate" 'BEGIN { exit !(rate > 0 && rate < 1e20) }'; then
    printf 'learning_rate must be a finite positive number below 1e20.\n' >&2
    return 2
  fi
  for value in "$checkpoint" "$run_dir" "$tokenizer" "$corpus"; do
    if [[ -z $value || $value == *$'\n'* ]]; then
      printf 'Paths must be nonempty and contain no newline.\n' >&2
      return 2
    fi
  done
  if [[ ! -d $checkpoint || ! -f $checkpoint/compact_vocabulary.tsv ||
        ! -f $tokenizer/tokenizer.json || ! -f $corpus ]]; then
    printf 'Missing checkpoint directory, compact mapping, tokenizer.json, or corpus.\n' >&2
    return 2
  fi
  checkpoint=$(realpath -e -- "$checkpoint")
  tokenizer=$(realpath -e -- "$tokenizer")
  corpus=$(realpath -e -- "$corpus")
  # Check dangling symlinks before canonicalizing, and never overwrite evidence.
  if [[ -e $run_dir || -L $run_dir ]]; then
    printf 'Refusing existing run directory: %s\n' "$run_dir" >&2
    return 2
  fi
  run_dir=$(realpath -m -- "$run_dir")
  if [[ $run_dir == "$repo_root" || $run_dir == "$repo_root/"* ]]; then
    printf 'run_dir must be outside the repository.\n' >&2
    return 2
  fi
  export PATH="/usr/local/cuda/bin:$PATH"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  local tool
  for tool in bazel git sha256sum nvidia-smi; do
    if ! command -v "$tool" >/dev/null; then
      printf 'Missing prerequisite: %s\n' "$tool" >&2
      return 1
    fi
  done
  mkdir -p -- "$(dirname -- "$run_dir")"
  mkdir -- "$run_dir"
  mkdir -- "$run_dir/inputs" "$run_dir/inputs/checkpoint"
  cd -- "$repo_root"
  {
    date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
    printf 'revision='; git rev-parse HEAD
    git status --short
    printf 'source_checkpoint=%s\nsource_tokenizer=%s\nsource_corpus=%s\n' \
      "$checkpoint" "$tokenizer" "$corpus"
    printf 'concurrent=%s\nstacked=%s\niso_params=%s\n' \
      "$((1 - serial))" "$stacked" "$iso_params"
    printf 'Only replacement MLP residual connections differ between jobs.\n'
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    df -h -- "$run_dir"
  } | tee "$run_dir/provenance.txt"
  # Include an uncommitted patch in provenance, since a branch may still be
  # under development. The actual executable and all training inputs are copied.
  git diff HEAD > "$run_dir/source.patch"
  bazel build -c opt //src/llm/experiments/memorize_general_facts:memorize_general_facts \
    2>&1 | tee "$run_dir/build.log"
  cp -- "$repo_root/bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts" \
    "$run_dir/inputs/memorize_general_facts"
  cp -- "$corpus" "$run_dir/inputs/corpus.txt"
  cp -- "$tokenizer/tokenizer.json" "$run_dir/inputs/tokenizer.json"
  cp -- "$script_dir/run_residual_ablation.sh" "$run_dir/inputs/run_residual_ablation.sh"
  cp -a -- "$checkpoint/." "$run_dir/inputs/checkpoint/"
  (cd -- "$run_dir/inputs"; find . -type f -print0 | sort -z | xargs -0 sha256sum) \
    > "$run_dir/input-sha256.txt"
  local train_flag=--train_mlp
  if ((stacked)); then train_flag=--train_stacked_mlp; fi
  local -a common=("$run_dir/inputs/memorize_general_facts"
    --mode=puzzle "$train_flag"
    "--puzzle_checkpoint=$run_dir/inputs/checkpoint"
    "--tokenizer=$run_dir/inputs" "--corpus=$run_dir/inputs/corpus.txt"
    --layers=4 --model_width=10 --attention_heads=1 --feed_forward_width=20
    --context_length=27 --compact_vocabulary=true "--batch_size=$batch_size"
    "--steps=$steps" "--eval_every=$eval_every"
    "--seed=$seed" "--learning_rate=$learning_rate")
  if ((!stacked)); then common+=("--mlp_width=$mlp_width"); fi
  if ((iso_params)); then common+=(--mlp_iso_parameters=true); fi
  local condition enabled
  local -a pids=() conditions=()
  # Terminate only children launched by this script when its caller cancels it.
  trap 'for child_pid in "${pids[@]}"; do kill "$child_pid" 2>/dev/null || true; done; wait || true; exit 130' INT
  trap 'for child_pid in "${pids[@]}"; do kill "$child_pid" 2>/dev/null || true; done; wait || true; exit 143' TERM
  local failed=0 status=0
  for condition in residual no_residual; do
    enabled=true
    if [[ $condition == no_residual ]]; then enabled=false; fi
    local -a command=("${common[@]}" "--mlp_residual_connections=$enabled"
      "--output_dir=$run_dir/$condition")
    printf '%q ' "${command[@]}" > "$run_dir/$condition.command"
    printf '\n' >> "$run_dir/$condition.command"
    printf 'Starting %s; log: %s/%s.log\n' "$condition" "$run_dir" "$condition"
    "${command[@]}" > "$run_dir/$condition.log" 2>&1 &
    pids+=("$!")
    conditions+=("$condition")
    if ((serial)); then
      status=0
      wait "${pids[0]}" || status=$?
      printf '%s\n' "$status" > "$run_dir/$condition.exit_status"
      if ((status)); then failed=1; fi
      pids=()
      conditions=()
    fi
  done
  local index
  for index in "${!pids[@]}"; do
    status=0
    wait "${pids[index]}" || status=$?
    printf '%s\n' "$status" > "$run_dir/${conditions[index]}.exit_status"
    if ((status)); then failed=1; fi
  done
  trap - INT TERM
  if ((failed)); then
    printf 'At least one condition failed; inspect logs in %s\n' "$run_dir" >&2
    return 1
  fi
  printf 'Both conditions completed. Compare %s/{residual,no_residual}/training.tsv\n' "$run_dir"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
