#!/usr/bin/env bash
# Compare the original post-A3 architecture with the stacked-MLP experiment.
# The binary checks a copied-weight positive control, then fits fresh weights.
set -euo pipefail

if (( $# != 4 )); then
  echo "Usage: bash $0 CHECKPOINT TOKENIZER_DIRECTORY CORPUS NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi

checkpoint=$(realpath -- "$1")
tokenizer=$(realpath -- "$2")
corpus=$(realpath -- "$3")
run_directory=$(realpath -m -- "$4")
repository=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ -e "$run_directory" ]]; then
  echo "Output directory already exists: $run_directory" >&2
  exit 2
fi
cd -- "$repository"
bazel build -c opt //src/llm/experiments/memorize_general_facts:memorize_general_facts
mkdir -- "$run_directory"

# These are exactly the stack sweep's source dimensions and training recipe.
# Inner widths are 20 for both MLPs; --mlp_width is deliberately not applicable.
bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts \
  --mode=puzzle --train_mlp_transformer \
  --puzzle_checkpoint="$checkpoint" --tokenizer="$tokenizer" --corpus="$corpus" \
  --output_dir="$run_directory/puzzle" \
  --layers=4 --model_width=10 --attention_heads=1 --feed_forward_width=20 \
  --context_length=27 --compact_vocabulary=true \
  --steps=300000 --eval_every=1000 --batch_size=32 --seed=3 \
  --learning_rate=0.01 2>&1 | tee "$run_directory/console.log"
