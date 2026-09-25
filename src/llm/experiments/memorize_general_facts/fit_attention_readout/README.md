# Fitting a pointwise readout after frozen attention

**Result: the fresh fourth-attention readout reached 100% accuracy on all
10,002 scored predictions and all 1,024 autonomous completions.** The
second-attention fits did not solve the task. See the fitting results below.

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
fit. This restores weights only, not Adam state. Use `--steps=0` to evaluate
such a checkpoint without updating it (a new output directory is still needed).
Best weights are selected by
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

## Initial fitting results (2026-09-25)

All rows below use the fixed checkpoint and task described above. Accuracy is
measured at the selected best checkpoint, not necessarily the final update.
The fresh second/fourth cross-entropy pair uses exactly the same initialization
seed (3), shuffle seed (3), batch size (32), and 120,000-update rate schedule
(0.001 to 0.0001); only the captured attention boundary differs.

| Boundary | Initialization / objective | Updates run | Best step | Wrong / 10,002 | Complete facts / 1,024 |
|---|---|---:|---:|---:|---:|
| Second | Original branch; margin, LR 0.001 | 20,000 | 20,000 | 8,482 | 0 |
| Second | Original branch; margin, LR 0.003 | 120,000 | 65,000 | 8,459 | 0 |
| Second | Matrix restart 1; margin, LR 0.003 | 120,000 | 45,000 | 8,392 | 0 |
| Second | Original branch; CE, LR 0.003 | 120,000 | 25,000 | 8,049 | 0 |
| Second | Matrix restart 2; CE, LR 0.001 | 120,000 | 85,000 | 8,112 | 0 |
| Second | Entirely fresh 3; CE, LR 0.001 | 120,000 | 70,000 | 8,086 | 1 |
| Fourth | Entirely fresh 3; margin, LR 0.003 | 120,000 | 108,000 | 384 | 723 |
| Fourth | Entirely fresh 3; CE, LR 0.001 | 120,000 | 92,000 | 41 | 983 |
| Fourth | Original branch, no fitting (control) | 0 | 0 | 0 | 1,024 |

Here “CE” means cross-entropy. Margin runs use delta=0.1. All runs decay to
0.1 times their initial rate, use beta2=0.999, and perform no clipping/decay.
Warm runs shuffle with seed 0; matrix restarts use their initialization seed
also as the shuffle seed. Evaluations are every 500 updates for the 20,000-step
run, every 5,000 for other second-attention runs, and every 2,000 for fresh
fourth-attention runs. The best is selected only at these evaluation points.

The fresh fourth-attention result is **99.5901% next-token accuracy**, versus
**19.1562%** for the matched second-attention run. So the fourth-attention
representation is substantially easier for this specific 450-parameter
readout and optimizer to decode. This is empirical optimization evidence,
not a proof about the representational capacity of the second-attention MLP.
Neither fresh run above achieved exact memorization.

Dataset SHA-256:
`814c062e7d7592fe4a4e5b158a37bd37da51700f817c19eb981c1e93d33f245c`.
Compact-vocabulary SHA-256:
`ff91c01867df309f841508fbfbcad0af95e20bfdcc5b63e6ae74e803835a9fb6`.

## Successful fourth-attention fit

The fresh cross-entropy run above was refined without changing its 450-parameter
architecture or unfreezing any other weights:

| Stage | Starting weights | Objective / rate schedule | Batch size | Updates run | Best step | Wrong | Exact completions |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | Entirely fresh, seed 3 | CE, 0.001 -> 0.0001 | 32 | 120,000 | 92,000 | 41 | 983/1,024 |
| 2 | Stage 1 best | Margin 0.1, 0.0001 -> 0.00001 | 32 | 150,000 | 128,000 | 6 | 1,018/1,024 |
| 3 | Stage 2 best | Margin 0.1, 0.00001 -> 0.000001 | 1,024 | 4,500 of 5,000 | 4,500 | **0** | **1,024/1,024** |

Each continuation restores only the previous stage's best weights and resets
Adam moments. Stage 2 uses shuffle seed 4 and evaluates every 1,000 updates;
stage 3 uses seed 5 and evaluates every 100. Stage 3's cosine schedule was
configured for 5,000 updates but stopped early at 4,500. It uses every fact
on each update, reducing minibatch noise near the decision boundaries.

Final minimum target-minus-best-competitor margin: **0.020164**. There are no
top-1 ties. The mean squared margin loss is **0.00000370**, not zero: some
correct predictions have positive margins smaller than the training goal 0.1.
Zero classification errors, rather than zero surrogate loss, is the required
criterion. Replaying the saved weights in a fresh process with batch size 32
and `--steps=0` also reproduces the full result.

A parallel CE-only refinement from stage 1 reached 36 errors and 988 complete
facts (best step 82,000 of 150,000, rate 0.0003 -> 0.000003, seed 4, batch 32).
It was not used in the successful lineage. Frozen final-LN and embedding bytes
remained unchanged in every run. Unit tests separately verify that the source
model and all six fresh initial tensors are independent of the original MLP.

The successful lineage contains 92,000 + 128,000 + 4,500 = **224,500 updates**,
with the final 4,500 using full batches. The three searches actually ran
274,500 updates in total before restoring their best checkpoints. Their fitting
loops took approximately 316, 242 and 79 seconds, respectively, on the GH200;
some other experiments ran concurrently, so these are not throughput benchmarks.

Local, ignored artifacts (logs and six-tensor branch checkpoints):

```
src/llm/experiments/memorize_general_facts/runs/attention_readout_20260925/
  fourth_fresh_ce/best_mlp/       # Stage 1
  fourth_refine_margin/best_mlp/  # Stage 2
  fourth_full_batch/best_mlp/     # Successful final branch
  fourth_verify.log              # Fresh-process, zero-update verification
```

To reproduce stages 2 and 3, use the command above without `--fresh_branch` or
`--random_init`, set `--readout_checkpoint` to the preceding `best_mlp`, and
apply the flags in the table. For stage 2 use `--objective=squared_margin
--learning_rate=0.0001 --final_rate_ratio=0.1 --batch_size=32 --steps=150000
--eval_every=1000 --seed=4`. For stage 3 use `--objective=squared_margin
--learning_rate=0.00001 --final_rate_ratio=0.1 --batch_size=1024 --steps=5000
--eval_every=100 --seed=5`. Always use a new `--output` directory.

This establishes that gradient descent can recover a successful pointwise
readout from the **fourth** attention's fixed representation. It does not
establish feasibility after the **second** attention: the original last MLP
already supplied a known feasible solution for the fourth, but there is no
corresponding witness for bypassing the intervening transformer blocks.
