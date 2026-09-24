# First-difference token-renaming diagnostic

This experiment locates the first violation of token-renaming symmetry in the
identical-embedding training runs. It does not train another model to completion
or change production kernels. The architecture, saved initialization, shuffled
minibatches, learning-rate schedule, and optimizer match the original experiment:
four blocks, width 10, MLP width 20, context 27, batch 32, seed 1337, BF16 compute,
and 48,680 FP32 master parameters.

## Reproduce

Build the native tools and replay a short prefix of the saved experiment. Use a
fresh output directory outside the repository; binary traces are not source.

```sh
bazel build -c opt //src/llm/experiments/memorize_general_facts/permutation_trace:all

tools=bazel-bin/src/llm/experiments/memorize_general_facts/permutation_trace
run=/home/ubuntu/checkpoints/memorize_general_facts/dataset_weights_identical_embeddings_0
trace=/tmp/pluto-permutation-first-difference

"$tools/permutation_trace" --run_dir="$run" --output_dir="$trace" \
  --max_steps=32 --support=2
"$tools/lm_head_probe" > "$trace/lm_head_probe.txt"
"$tools/cross_entropy_replay" --trace_dir="$trace" > "$trace/native_replay.tsv"
python -B scripts/memorize_general_facts/analyze_permutation_trace.py \
  --trace="$trace" --output="$trace/analysis.html"
```

The analysis script needs NumPy. The saved run supplies its tokenizer, corpus,
compact token IDs, permutation, and initial checkpoint. No downloads are needed.
`--support` selects `rename_000_NNNN`, not a newly sampled permutation. This tool
deliberately fixes the remaining configuration to the experiment being diagnosed.

## Controls and recorded data

Three models execute the same steps: baseline, exact baseline repeat, and
token-renamed. Every initial parameter is checked byte-for-byte against the saved
baseline `step_0`, and all initial embedding rows must be identical. Comparisons
undo the permutation on token IDs, vocabulary columns, and embedding rows; all
other coordinates remain unchanged. Ignored targets and padded vocabulary lanes
are never relabeled.

Each step records inputs, targets, weights before the update, every layer-hook
forward output and incoming backward gradient, loss, logit gradients, every
unique parameter gradient, and weights after AdamW. Snapshots use stream-ordered
copies into pinned host buffers. No activation is replaced, and attention keeps
its normal FlashAttention path. A gradient hook observes the gradient *entering*
that layer's backward, not the gradient it subsequently produces.

- `summary.tsv`: comparisons for every tensor at every replayed step.
- `batches.tsv`: raw, unaligned token changes and embedding differences per step.
- `metadata.txt`: configuration, stopping step, and first differing stage.
- `baseline/`, `repeat/`, `renamed/`: complete final-step tensor manifests and
  little-endian bytes. Earlier matching steps retain comparison summaries only.
- `analysis.html` and `analysis.json`: CPU analysis, including FP64 references,
  individual parameter tensors, and the actual update differences.

The replay stops at the first numerical difference, a failed repeatability
control, or the step limit. Signed-zero bit differences are reported separately
and do not by themselves stop a numerically matching replay. A repeatability
failure returns an error. Reaching the limit without a difference is not proof
of equivalence for later steps.

The independent `lm_head_probe` uses the production head and loss kernels with
uniform embedding rows. It repeats the baseline, swaps target IDs, and then
restores the original order of the head's operands. Its tests require exact
repeatability and restoration; they do not require a hardware-specific amount
of numerical disagreement under renaming.

`cross_entropy_replay` repeats both native cross-entropy and head backward using
the actual captured operands. It requires both baseline and renamed outputs to
match the saved trace, and requires canonicalized inputs to restore baseline
outputs bit-for-bit. The CPU report includes its results when
`native_replay.tsv` is present.

## Observed result: two-label swap

On the GH200, two independent replays produced identical summaries. The labels
were compact IDs 2154 (`"orne"`) and 2457 (`" spectrum"`); only three positions in
the full corpus change. These are categorical renamings, not different facts.

1. Steps 1–18 encounter neither changed token. At step 19 the `"orne"` token in
   the five-token Borneo prompt appears. Its target is not scored, but the input
   embedding still receives a gradient from later scored positions. The two
   embedding rows become distinct. The two models remain exactly equivalent
   after relabeling through step 21.
2. At step 22 every forward activation, including the logits, still matches
   bit-for-bit after relabeling. Cross-entropy is the first differing operation:
   four losses differ by one FP32 ULP (maximum `9.5367431640625e-7`). Its logit
   gradients differ by at most `2.2737367544323206e-13`. There are no renamed token
   IDs in this minibatch: softmax nevertheless visits the entire vocabulary.
3. Those logit-gradient differences disappear when converted to BF16, as the
   head's input-gradient kernel does. The head therefore sees the same rounded
   products in a different vocabulary order. Its FP32 accumulation independently
   changes 97 hidden-gradient entries by at most `1.8189894035458565e-12`.
   All 97 are in channel 6, the only channel where the two moved embedding rows
   differ after BF16 rounding.
4. Backpropagation produces 251 differing parameter-gradient entries across 15
   tensors. After AdamW, 31 of 48,680 parameter values differ across five tensors,
   by at most `1.862645149230957e-9`. This first update difference is very small;
   it is not evidence of a large immediate amplification by AdamW.

The exact baseline repeat matches throughout. These measurements locate two
order-sensitive vocabulary reductions, not random GPU nondeterminism. They also
separate BF16 operand quantization from FP32 summation order. In particular, the
head difference is not simply the cross-entropy discrepancy propagating through
BF16: that discrepancy rounds away before the head matrix multiplication.
Isolated native replays reproduce the actual saved tensors exactly; restoring
the original vocabulary order restores the baseline loss, logit gradients, and
head input gradients bit-for-bit.

This explains an initial mechanism for the later training-trajectory divergence;
it does not prove that these first 31 changed coordinates alone explain every
final-checkpoint difference, nor does it certify every kernel against all bugs.
Those original measurements used the unmodified production kernels.

## Verify the canonical-order fix

The layer factories now accept an optional host array mapping canonical ranks
to current token IDs. Cross-entropy traverses logits in that order, and head
backward gathers both gradient columns and embedding rows into the original
MMA reduction order. Tensor storage stays in current token-ID order. The same
permutation must be passed to both the model's head and its separate loss.

Enable that behavior for the renamed model during replay:

```sh
"$tools/permutation_trace" --run_dir="$run" \
  --output_dir=/tmp/pluto-permutation-canonical-2 \
  --max_steps=64 --support=2 --canonical_token_order=true
```

Repeat with `--support=512` and `--support=4474`, using fresh output directories.
With this flag the diagnostic returns an error on any aligned bit difference,
including signed zero. The ordinary diagnostic mode still stops successfully
after capturing a first difference. No long memorization training is launched.

On the GH200, all three saved permutations matched the baseline bit-for-bit
for all 64 replayed steps: each compared 17,536 tensor snapshots, with zero
renamed or repeated-baseline mismatches. This includes the actual shuffled
batches, all hooked activations and backward gradients, all unique parameter
gradients, and every post-AdamW weight. The no-order control reproduced the
original step-22 divergence with an identical `summary.tsv`, confirming that
the existing default path was preserved. These are bounded replay results,
not a claim that new full-length training runs were performed.

Local evidence is under
`/home/ubuntu/checkpoints/memorize_general_facts/canonical_token_order_replay_0/`:
`support_2`, `support_512`, `support_4474`, and `unmapped_control`.
The older native `cross_entropy_replay` counterfactual above deliberately uses
ordinary ID-order kernels; run it on the original no-order capture, not a
canonical-order capture.

## Tests

```sh
bazel test -c opt //src/llm/experiments/memorize_general_facts/permutation_trace:all
python -B -m unittest discover -s scripts/memorize_general_facts \
  -p 'analyze_permutation_trace_test.py'
```
