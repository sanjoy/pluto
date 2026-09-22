# Weight sensitivity

This C++ experiment independently replaces each logical parameter tensor in a
memorized GPT-2 checkpoint with Gaussian noise, then counts the corpus sentences
whose completion becomes incorrect. Checkpoint files are never modified.

## Run

From the repository root:

```sh
bazel build -c opt //src/llm/experiments/weight_sensitivity

facts_run=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
bazel-bin/src/llm/experiments/weight_sensitivity/weight_sensitivity \
  --checkpoint="$facts_run/layers_8/step_16128" \
  --tokenizer="$facts_run/inputs/tokenizer" \
  --corpus=testdata/general_facts_dataset.txt \
  --output=/tmp/weight_sensitivity.html
```

The output must be a fresh filename. It is a self-contained, sortable HTML table;
open it in a browser. Progress is printed to stdout, and the HTML is atomically
updated after each intervention. Interrupted runs retain an explicitly marked
partial report. No web server or network resources are required.

Defaults match the memorized model: eight blocks, width 16, one attention head,
MLP width 64, context 1,024, 1,024 sentences, five-token prompts, and evaluation
batch size 32. The compact vocabulary mapping is loaded from the checkpoint;
the original GPT-2 tokenizer directory is also required. `--layers`,
`--model_width`, `--attention_heads`, and `--feed_forward_width` describe the
checkpoint, not a new training run. All parameter counts/sizes are checked.

For a short smoke test add `--target_filter='Attn 3 Q matrix'`. For independent
repeat measurements add `--trials=3`. `--seed=1337` is the default root seed.
Per-target/trial seeds do not change when filtering or increasing trial count.

## Intervention and scoring

- The eight-block model has 100 distinct physical tensors and 132 logical
  intervention targets. Packed Q/K/V weights are split into three strided
  matrices, and their biases are split too. All other tensors are measured
  separately, including LayerNorm parameters. Blocks are numbered from zero.
- The token embedding and language-modeling head share one matrix. Replacing it
  changes both uses, and it is counted once. Matrix shapes are in storage order
  `[input, output]`; embedding shapes are `[vocabulary, width]`.
- Each selected scalar is **replaced**, not perturbed additively, by zero-mean
  Gaussian noise with standard deviation equal to that logical target's original
  RMS times `--noise_scale` (default 1). All-zero targets use
  `--zero_rms_stddev` (default 0.02), also multiplied by the scale. Sampling is
  deterministic for a seed on the same build/platform; libm rounding can differ
  across platforms. Other slices of a packed tensor remain unchanged.
- Each trial starts from the original checkpoint, never from the preceding
  corruption. The full physical tensor is restored even if evaluation fails.
  A final unmodified evaluation verifies restoration.
- A sentence is correct only when its **entire suffix plus EOS** is correct,
  given its first five tokens. The unmodified baseline must get every sentence
  right before any intervention is attempted.
- Causal teacher forcing gives this exact binary completion score in one forward
  per batch: before the first mismatch its prefix is identical to greedy
  decoding; if no mismatch occurs, greedy decoding reproduces the whole suffix
  and EOS. After the first mismatch the sentence is already incorrect, so there
  is no need to simulate the remaining generated trajectory. Token-error counts
  in the report are teacher-forced diagnostics, not rollout edit distances.
- Prompt and padding targets are excluded. Nonfinite logits at scored positions
  count as errors and are reported separately. Ties use the lowest token ID.

The main column is **newly wrong exact completions**, e.g. `7 / 1024`. This is
sensitivity to this particular intervention, not proof that a tensor exclusively
stores those seven facts. One noise draw is a sample, not an estimate with error
bars. Full replacement can saturate at 1,024 failures and obscure differences
among important tensors. Multiple trials and alternative noise scales help
check robustness; changing the scale still replaces the original values.

## Code and tests

`weights` describes/validates the physical layouts and constructs noise;
`evaluation` scores exact completions; `runner` isolates and restores mutations;
`html` renders the report. The CLI composes these pieces. All host/device copies
use page-locked memory and the same explicit CUDA executor.

```sh
bazel test -c opt //src/llm/experiments/weight_sensitivity/... \
  --test_output=errors --local_test_jobs=1
```

Tests cover every scalar's unique target assignment, actual GPT-2 weight layout,
strided QKV slices, deterministic noise, EOS and masking, final partial batches,
nonfinite predictions, equivalence to explicit greedy rollout, restoration on
success/failure, report escaping, and partial-run labeling. GPU tests require a
compatible CUDA device. Generated reports are experiment artifacts; keep them
outside the source tree.
