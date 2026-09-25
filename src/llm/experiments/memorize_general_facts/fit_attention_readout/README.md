# Fitting a pointwise readout after frozen attention

This experiment asks whether a 10 -> 20 -> 10 residual MLP can directly
predict all the required completions from a selected attention boundary.
It uses native BF16 layer computation and FP32 master parameters/Adam state,
not the symbolic states of the discretized model.

## What is optimized

For a frozen post-attention residual vector `x`, the replacement computes:

```
u = LayerNorm(x; gamma, beta)
h = x + W2 * GELU(W1 * u + b1) + b2
logits = original_embedding * original_final_LayerNorm(h)
```

Exactly six tensors are trainable: gamma/beta (20 parameters), W1/b1 (220),
and W2/b2 (210), for **450 parameters total**. The original prefix, final
LayerNorm and tied embedding are frozen. There is no gradient clipping or
weight decay. No extra transformer blocks or wider hidden layers are added.

`--block=1` taps the second attention; `--block=3` taps the fourth. These are
zero-based indices. The attention boundary includes its output projection and
residual addition, but precedes that block's MLP LayerNorm. After this tap,
the replacement bypasses all remaining original operations except final LN
and the vocabulary head. Capturing currently runs the complete original graph
for convenience; its later outputs are discarded, never fed into the fitted
readout.

All 1,024 facts supply five prompt tokens. The next-token objective covers
10,002 positions: the remaining text and one EOS per fact. Prompt-only and
padding positions are ignored. All 4,475 vocabulary alternatives participate
in the readout. This is an in-sample memorization experiment, not a held-out
generalization evaluation.

Two objectives are available:

- `squared_margin`: mean of `max(0, delta - gap)^2`, where `gap` is the
  correct logit minus the largest incorrect logit and `delta=0.1` by default.
  Its explicit cuTile gradient updates the target and strongest competitor.
- `cross_entropy`: the framework's standard masked, mean cross-entropy.

Evaluations always report squared margin loss and exact top-1 counts, even
when training with cross-entropy. Adam uses beta1=0.9, beta2=0.999, epsilon=1e-8.
The rate decays by cosine from `learning_rate` to `final_rate_ratio` times that
rate over `steps`. Batches contain 32 shuffled facts by default.

Initialization choices:

- Default: all six branch tensors copied from the selected original MLP.
- `--random_init=N`: reset only W1/W2 to seeded normal values with standard
  deviations 0.2/0.1; retain original branch biases and LayerNorm parameters.
- `--random_init=N --fresh_branch`: copy **none** of the original six branch
  tensors. Use the same random matrices, zero affine biases, and initial
  LayerNorm gamma=1/beta=0. All 450 parameters remain trainable.

`--readout_checkpoint` can instead initialize the six tensors from a previous
fit. This restores weights only, not Adam state. Best weights are selected by
fewest incorrect scored positions, with margin loss breaking ties. The tool
restores this best checkpoint before evaluating actual greedy continuations,
recomputing the frozen prefix on each generated history. A fact fails at its
first incorrect token. It also checks frozen readout weights byte-for-byte.

The tool stops on zero teacher-forced errors, the step cap, or the wall-clock
cap, then performs the independent autoregressive check. An exit code of zero
means the experiment ran successfully; consult `autoregressive_complete` to
determine whether it actually solved the task. Failure to fit is **not** proof
that no solution exists.

## Run

```bash
bazel build -c opt \
  //src/llm/experiments/memorize_general_facts/fit_attention_readout:fit_attention_readout

source_run=/home/ubuntu/checkpoints/memorize_general_facts/dataset_weights_canonical_order_0
run_dir=$(mktemp -d /tmp/attention-readout.XXXXXX)
set -o pipefail
bazel-bin/src/llm/experiments/memorize_general_facts/fit_attention_readout/fit_attention_readout \
  --checkpoint="$source_run/baseline/checkpoints/layers_4/step_120000" \
  --tokenizer="$source_run/inputs/tokenizer" \
  --corpus="$source_run/inputs/corpus.txt" \
  --output="$run_dir/fourth_attention" \
  --block=3 --fresh_branch --random_init=3 --seed=3 \
  --objective=cross_entropy --learning_rate=0.001 \
  --batch_size=32 --steps=120000 --seconds=600 --eval_every=2000 \
  2>&1 | tee "$run_dir/train.log"
```

`output` must not already exist. Only the fitted branch's six weight files
are written, under `output/best_mlp`; these are not a full GPT-2 checkpoint.
Source checkpoint: four blocks, width 10, one head, FF width 20, context 27,
compact vocabulary 4,475. The original checkpoint is never modified.

## Validation

`margin_loss_test` tests explicit CPU expectations, finite-difference
gradients, ignored/padded rows, ties across tiles, and malformed/nonfinite
inputs. `readout_test` checks copied tensor ordering, separate allocations,
optimizer isolation, fresh initialization independence, seeded restarts, and
bitwise equality to the original last-block suffix.

The full-corpus positive control uses `--block=3` with the original branch.
It already has **0/10,002** errors before fitting, reproduces **1,024/1,024**
autonomous completions, and has minimum target margin **0.167267**. This is a
correctness control, not evidence that a fresh fit succeeds. It establishes a
known feasible 450-parameter solution for the fourth-attention experiment.
