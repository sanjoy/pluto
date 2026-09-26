# Fitting a pointwise readout after frozen attention

**Result: the fresh fourth-attention readout reached 100% accuracy on all
10,002 scored predictions and all 1,024 autonomous completions.** The
second- and third-attention fits did not solve the task. See the fitting
results below.

This experiment asks whether a 10 -> H -> 10 residual MLP can directly
predict all the required completions from a selected attention boundary.
The default is H=20; `--readout_width` changes only the replacement's hidden
width, never the frozen source model. For example, `--readout_width=80` tests
an 8d expansion while the source checkpoint retains its original width-20 MLPs.
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
and W2/b2 (210), for **450 parameters total** at H=20. More generally the
branch has `2*d*H + H + 3*d` parameters, or **1,710** at d=10, H=80.
The original prefix, final
LayerNorm and tied embedding are frozen by default. With `--train_final_norm`,
the final LayerNorm's gamma/beta are optimized along with the six MLP tensors:
this adds 20 parameters at d=10, giving **470** at H=20. Its initial values are
still copied from the original checkpoint; the prefix and tied head remain
frozen and the input LayerNorm was already trainable in either mode.
There is no gradient clipping or
weight decay. No extra transformer blocks are added; hidden width changes only
when explicitly requested with `--readout_width`.

`--block=1` taps the second attention, `--block=2` the third, and `--block=3`
the fourth. These are
zero-based indices. The attention boundary includes its output projection and
residual addition, but precedes that block's MLP LayerNorm. After this tap,
the replacement bypasses all remaining original operations except final LN
and the vocabulary head. Without grafting, capture runs the complete original
graph for convenience; its later outputs are discarded, never fed into the
fitted readout.

### Grafting raw coordinates from another attention boundary

`--graft_block=3 --graft_dimensions=2 --block=2` replaces the first two
coordinates (dimensions 1 and 2, zero-based columns 0 and 1) of each third-
attention vector with the fourth-attention vector's coordinates from the
**same fact and token position**. The remaining eight coordinates are kept.
This is a raw-coordinate replacement, not a PCA projection or concatenation.
The hybrid vector feeds both input LayerNorm and the residual connection.

Both snapshots are taken from one unchanged source-model forward pass. The
copy happens only after that pass, into an independent buffer, so changing
the recipient cannot contaminate the donor. The source model and vocabulary
head remain frozen. All rows are grafted; the ordinary suffix-plus-EOS mask
still determines which rows contribute to training and evaluation. Input LN
is always trained; `--train_final_norm` additionally trains final LN.

Greedy verification recomputes the same graft from the actually generated
prefix, rather than reusing a teacher-forced cache. This experiment still
requires the fourth attention to produce its donor coordinates: it measures
decodability of a hybrid representation, not a standalone third-attention
model. No-graft behavior remains the default (`--graft_block=-1`,
`--graft_dimensions=0`). The dimension count must be between 1 and model width
when a donor is specified. A same-boundary donor is a no-op control; grafting
the full width exactly reproduces the donor representation.

For arbitrary coordinates, use `--graft_columns=2,7` (one-based) instead of
`--graft_dimensions`. Coordinate order is irrelevant and duplicates are
rejected. The original leading-dimension option remains supported.

For a shared two-dimensional plane, use `--graft_plane=/path/to/plane.txt`.
The file has exactly two floating-point coefficients per model dimension,
with no header; its two columns must be orthonormal. Given its width-by-two
basis `U`, the hybrid is `x3 + U * transpose(U) * (x4 - x3)`: replace A3's
component in the plane with A4's, retaining A3's orthogonal component. A
common centering vector cancels in this expression. A cuTile kernel performs
the two projections in FP32 and rounds the final ten-dimensional vector to
BF16. The plane is fixed, not learned, and is used unchanged for training and
greedy evaluation. Exactly one of `graft_dimensions`, `graft_columns`, or
`graft_plane` can be specified with a donor block.

The benchmarking driver `scripts/memorize_general_facts/run_attention_graft_sweep.py`
screens all ten single coordinates and all 45 pairs, with fresh A3/A4 controls.
`--plane=name=path` adds named planes; `--variants-file` selects a custom
cohort, including larger coordinate subsets. It records every command and
input fingerprint, saves all checkpoints locally, and incrementally writes
an HTML ranking/heatmap and TSV. A short screen uses its own complete cosine
schedule; it is not the first part of a 120,000-step fit. Use `--phase=full`
to rerun selected candidates fresh for 120,000 updates. Keep screen and full
results separate when comparing accuracy.

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

`--readout_checkpoint` can instead initialize the trainable tensors from a
previous fit (six by default, eight with `--train_final_norm`). To start from
an earlier six-tensor MLP while newly unfreezing final LN, use
`--branch_checkpoint=... --train_final_norm`; the final LN then retains its
source-checkpoint initialization. The two checkpoint flags are mutually
exclusive. Checkpoints restore weights only, not Adam state. Use `--steps=0` to evaluate
such a checkpoint without updating it (a new output directory is still needed).
Changing the hidden width requires a fresh branch or a branch checkpoint of
the requested width. Use `--fresh_branch --random_init=N --readout_width=80`
to start a wider fit, or `--readout_width=80 --readout_checkpoint=...` to
continue one. The latter constructs temporary fresh tensors before restoring
all six weights; those temporary values never enter fitting or evaluation.
Checkpoint sizes are checked strictly. `--feed_forward_width` still describes
the original checkpoint and should remain 20 for the experiments below.
Best weights are selected by
fewest incorrect scored positions, with margin loss breaking ties. The tool
restores this best checkpoint before evaluating actual greedy continuations,
recomputing the frozen prefix on each generated history. A fact fails at its
first incorrect token. It also checks frozen readout weights byte-for-byte.

By default, the tool stops on zero teacher-forced errors, the step cap, or the
wall-clock cap, then performs the independent autoregressive check. Use
`--stop_on_zero_errors=false` for fixed-budget comparisons that should keep
training through the requested step count even after perfect accuracy; the
wall-clock cap still applies. An exit code of zero
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

`output` must not already exist. Only the fitted suffix's weight files
are written, under `output/best_mlp`; these are not a full GPT-2 checkpoint.
With `--train_final_norm`, the same `best_mlp` directory additionally contains
the final LayerNorm's two tensors. Load that eight-tensor checkpoint with the
same flag; strict loading rejects incompatible checkpoint shapes/counts.
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
The fresh second/third/fourth cross-entropy runs use the same initialization
seed (3), shuffle seed (3), batch size (32), and 120,000-update rate schedule
(0.001 to 0.0001). The training input differs only in the captured attention
boundary. Best-checkpoint selection cadence differs: the second run evaluated
every 5,000 updates; the third and fourth evaluated every 2,000.

| Boundary | Initialization / objective | Updates run | Best step | Wrong / 10,002 | Complete facts / 1,024 |
|---|---|---:|---:|---:|---:|
| Second | Original branch; margin, LR 0.001 | 20,000 | 20,000 | 8,482 | 0 |
| Second | Original branch; margin, LR 0.003 | 120,000 | 65,000 | 8,459 | 0 |
| Second | Matrix restart 1; margin, LR 0.003 | 120,000 | 45,000 | 8,392 | 0 |
| Second | Original branch; CE, LR 0.003 | 120,000 | 25,000 | 8,049 | 0 |
| Second | Matrix restart 2; CE, LR 0.001 | 120,000 | 85,000 | 8,112 | 0 |
| Second | Entirely fresh 3; CE, LR 0.001 | 120,000 | 70,000 | 8,086 | 1 |
| Third | Entirely fresh 3; CE, LR 0.001 | 120,000 | 66,000 | 7,567 | 0 |
| Third | Original branch; CE, LR 0.001 | 120,000 | 46,000 | 7,502 | 0 |
| Fourth | Entirely fresh 3; margin, LR 0.003 | 120,000 | 108,000 | 384 | 723 |
| Fourth | Entirely fresh 3; CE, LR 0.001 | 120,000 | 92,000 | 41 | 983 |
| Fourth | Original branch, no fitting (control) | 0 | 0 | 0 | 1,024 |

Here “CE” means cross-entropy. Margin runs use delta=0.1. All runs decay to
0.1 times their initial rate, use beta2=0.999, and perform no clipping/decay.
Second-attention warm runs shuffle with seed 0; the third-attention warm run
uses seed 3. Matrix restarts use their initialization seed also as the shuffle
seed. Evaluations are every 500 updates for the 20,000-step
run, every 5,000 for other second-attention runs, and every 2,000 for fresh
third/fourth-attention runs. The best is selected only at these evaluation points.

The fresh fourth-attention result is **99.5901% next-token accuracy**, versus
**19.1562%** for the matched second-attention run. So the fourth-attention
representation is substantially easier for this specific 450-parameter
readout and optimizer to decode. This is empirical optimization evidence,
not a proof about the representational capacity of the second-attention MLP.
None of these initial fresh runs achieved exact memorization.

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

## Third-attention follow-up and comparison

The third attention (`--block=2`) was tested with precisely the same fresh
initialization, training schedules, and refinement sequence as the successful
fourth-attention lineage. All six branch tensors were trainable; the prefix,
final LayerNorm, and vocabulary head remained frozen. The input is the third
attention's residual output, before its MLP. Neither the original third MLP
nor any fourth-block result is consumed by the replacement readout.

| Third-attention stage | Updates run | Best step in this stage | Wrong / 10,002 | Exact facts / 1,024 |
|---|---:|---:|---:|---:|
| Fresh CE, seed 3, batch 32, LR 0.001 -> 0.0001 | 120,000 | 66,000 | 7,567 | 0 |
| Margin refinement, seed 4, batch 32, LR 0.0001 -> 0.00001 | 150,000 | 0 | 7,567 | 0 |
| Full-batch margin, seed 5, LR 0.00001 -> 0.000001 | 5,000 | 0 | 7,567 | 0 |
| Separate original-MLP warm start, CE, seed 3, LR 0.001 -> 0.0001 | 120,000 | 46,000 | 7,502 | 0 |

The margin stages lowered their surrogate loss but worsened token accuracy.
Their best checkpoint was therefore their initial checkpoint (step 0), so
both correctly restored the earlier CE solution. The fresh lineage tried
275,000 updates in total; including the separate warm start, this follow-up
tried 395,000. Every run completed its configured step count. A fresh-process
zero-update replay of the best warm-start weights confirmed exactly 7,502
errors and zero complete facts. Frozen head/LN bytes were unchanged.

### Matched fresh-initialization comparison

| Attention boundary | Accuracy after 120,000 CE updates, selecting best | Exact facts at that checkpoint | Best accuracy after margin/full-batch refinement | Exact facts after refinement |
|---|---:|---:|---:|---:|
| Second | 19.1562% | 1/1,024 | Not run for this lineage | — |
| Third | 24.3451% | 0/1,024 | 24.3451% | 0/1,024 |
| Fourth | 99.5901% | 983/1,024 | **100%** | **1,024/1,024** |

The initial training schedule and initialization match across these three
rows, with the evaluation-cadence caveat noted above. Third and fourth also
share the refinement schedules; fourth stopped its final stage early upon
success. Higher per-token accuracy does not imply more completely correct
facts: every suffix token and EOS must be correct to count a fact.

### Best token-accuracy checkpoint across all attempts

| Attention boundary | Correct scored tokens | Token accuracy | Exact facts | Initialization / selected stage |
|---|---:|---:|---:|---|
| Second | 1,953/10,002 | 19.5261% | 0/1,024 | Original MLP; CE at step 25,000 |
| Third | 2,500/10,002 | 24.9950% | 0/1,024 | Original MLP; CE at step 46,000 |
| Fourth | 10,002/10,002 | **100%** | **1,024/1,024** | Entirely fresh, then margin/full-batch refinement |

This second table summarizes the best observed token accuracy, not an
equal-search-budget experiment. Each row's token and fact metrics come from
the same checkpoint. In particular, the second-attention fresh checkpoint
that completed one fact had lower token accuracy and is not mixed into the
best-accuracy row.

The third-attention readout did **not** recover exact memorization. Only the
fourth-attention fit did so in these trials. This does not show that the
earlier activations lack information, nor prove that no suitable 450-parameter
MLP exists: it is a negative result for this architecture, frozen output head,
initializations, and finite optimization procedure.

Local logs/checkpoints are under the ignored directory:

```
src/llm/experiments/memorize_general_facts/runs/attention_readout_20260925/third_attention/
  third_fresh_ce/best_mlp/
  third_refine_margin/best_mlp/
  third_full_batch/best_mlp/
  third_warm_ce/best_mlp/   # Best observed third-attention token accuracy
  third_verify.log         # Independent replay of that checkpoint
```

Reproduce the fresh sequence using the fourth-attention commands above with
`--block=2`, new output directories, and the preceding third-attention
`best_mlp` as each continuation's input. The separate warm start uses
`--block=2 --seed=3 --objective=cross_entropy --learning_rate=0.001
--batch_size=32 --steps=120000 --eval_every=2000`, without random/fresh flags.

## Training the final LayerNorm (2026-09-25)

Unfreezing final LayerNorm improves the earlier readouts modestly, but does
not remove the large gap to attention 4 in these matched fits. This experiment
retains the **10 -> 20 -> 10 MLP**; the separately proposed 8d-width experiment
was not run. Both variants train the input LayerNorm. Only final LayerNorm's
ten gamma and ten beta values change optimizer membership, raising the total
from 450 to **470 trainable parameters**. The prefix and vocabulary head stay
frozen.

These comparisons use the previous **fresh-initialization, 120,000-step CE**
baselines, not the best warm starts or later margin refinements. Every paired
run uses branch initialization seed 3, shuffle seed 3, batch size 32, cosine
learning rate 0.001 -> 0.0001, beta2=0.999, and no weight decay/clipping.
Final LN starts with the same copied checkpoint values in either mode.
Evaluation cadence also matches within each pair: 5,000 updates for attention
2, and 2,000 for attentions 3/4. Select the best evaluated checkpoint by token
errors, with squared margin loss breaking ties.

| Attention | Frozen final LN: token accuracy | Trainable final LN: token accuracy | Change (percentage points) | Frozen: complete facts | Trainable: complete facts | Trainable best step |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 1,916/10,002 = 19.1562% | 2,371/10,002 = 23.7053% | +4.5491 | 1/1,024 | 0/1,024 | 40,000 |
| 3 | 2,435/10,002 = 24.3451% | 2,757/10,002 = 27.5645% | +3.2194 | 0/1,024 | 0/1,024 | 74,000 |
| 4 | 9,961/10,002 = 99.5901% | 9,832/10,002 = 98.3003% | -1.2897 | 983/1,024 | 867/1,024 | 116,000 |

All new runs completed 120,000 updates. Actual greedy completion was checked
after restoring the selected checkpoint, then independently repeated in a
new process with `--steps=0`. Both checks agree exactly. The embedding bytes
remained unchanged, and all twenty final-LN parameters changed and remained
finite in each fitted checkpoint. Relative combined gamma/beta L2 changes
from the original final LN were 29.86%, 31.58%, and 8.17%, respectively.
Initial step-zero errors and margin losses exactly matched the frozen-LN
baselines, confirming no accidental change to initialization or forward
computation. The old successful fourth-attention checkpoint also remained
100% correct when loaded with `--branch_checkpoint` before any updates.

This is a one-seed, fixed-schedule optimization result: it does not prove the
earlier representations cannot be decoded, nor that extra trainable parameters
reduce capacity. MLP weights co-adapt with final LN, so gains are attributable
to enabling joint training, not to changing normalization in isolation.
The previously reported 100% fourth-attention result used additional margin
and full-batch refinement and is deliberately excluded from the matched table.

Local ignored artifacts:

```
src/llm/experiments/memorize_general_facts/runs/final_norm_20260925/
  attention_2_fresh_ce.log
  attention_2_fresh_ce/best_mlp/  # Eight tensors, not six
  attention_3_fresh_ce.log
  attention_3_fresh_ce/best_mlp/
  attention_4_fresh_ce.log
  attention_4_fresh_ce/best_mlp/
  verify_attention_2.log         # Independent saved-weight replay
  verify_attention_3.log
  verify_attention_4.log
```

Reproduce with the fresh CE command above and `--train_final_norm
--readout_width=20 --block=N`, for N=1,2,3. Use `--eval_every=5000` for N=1
and `--eval_every=2000` otherwise, fresh output directories, and a sufficiently
large wall-clock cap (`--seconds=1800` was used; it did not bind). To replay,
omit fresh/random flags, specify `--readout_checkpoint=.../best_mlp`, retain
`--train_final_norm`, and set `--steps=0`.

Validation: all **93 Bazel test targets pass** with the tokenizer/parquet
fixtures configured. New targeted tests verify eight-tensor optimizer
isolation, bitwise-equivalent initial forward/backward computation, and
strict eight-tensor checkpoint round trips. The initial full-suite invocation
omitted the external fixture environment variables and was rerun successfully
with `PLUTO_GPT2_TOKENIZER_DIR` and `PLUTO_FINEWEB_PARQUET_DIR` supplied.

## Grafting attention 4 dimensions 1–2 into attention 3 (2026-09-25)

The hybrid replaces raw, one-based dimensions 1 and 2 of A3 with A4's values
at the same fact and token position; dimensions 3–10 remain A3. Both source
snapshots are frozen and unchanged. Train the residual 10 -> 20 -> 10 MLP,
its input LN, and final LN together: **470 parameters**. This uses every fact
and every scored suffix/EOS position, not the 32-fact visualization subset.

The new fit exactly matches the earlier trainable-final-LN protocol: fresh
branch seed 3, shuffle seed 3, cross-entropy, batch 32, 120,000 updates,
cosine rate 0.001 -> 0.0001, evaluation every 2,000 updates, and no weight
decay or gradient clipping. Select the best checkpoint by wrong-token count,
breaking ties with mean squared-margin loss. The wall-clock cap did not bind.

| Frozen representation fed to the readout | Correct / 10,002 | Token accuracy | Exact completions / 1,024 | Best step |
|---|---:|---:|---:|---:|
| A3 (previous matched baseline) | 2,757 | 27.5645% | 0 | 74,000 |
| A3, with raw dimensions 1–2 from A4 | 3,125 | 31.2438% | 1 | 52,000 |
| A4 (previous matched baseline) | 9,832 | 98.3003% | 867 | 116,000 |

The graft gains **368 correct tokens / 3.6793 percentage points** over A3,
but does not approach A4's decodability in this single-seed, fixed-schedule
fit. This is not proof that the hybrid cannot be decoded better, nor that
these coordinates exclusively store particular facts. Replacing coordinates
also changes LayerNorm statistics and interactions with the retained ones.

The fit ran all 120,000 updates in approximately 106 seconds, including its
evaluations and final greedy check but excluding initial capture. At the
selected checkpoint, mean squared-margin loss is 7.60976871 and minimum
margin is -16.598312. The final update had 6,886 errors; restoring the best
checkpoint gives **6,877**. A new process loading those eight tensors
reproduces 6,877 errors and one complete fact exactly. All 40 trainable LN
parameters changed from their initial values and all 470 parameters are
finite; the tied embedding remains byte-for-byte unchanged.

Validation for this change:

- Six new GPU tests verify exact BF16-coordinate replacement, source
  immutability, independent output storage, aliasing, nontrivial row pitches,
  full-width/single-column grafts, and invalid shape/dimension/executor inputs.
- All 26 tests across `activation_graft_test`, `readout_test`, and
  `margin_loss_test` pass. Five malformed graft-flag combinations are rejected.
- Same-source A3 -> A3 graft reproduces the baseline's 7,245 errors and zero
  complete facts; a full-width A4 -> A3 graft with A4's fitted readout reproduces
  170 errors and 867 complete facts. These full-corpus controls exercise both
  the cached training path and fresh causal-prefix greedy verification.

Local ignored artifacts and the exact reproduction script:

```
src/llm/experiments/memorize_general_facts/runs/attention_graft_20260925/
  run.sh
  same_donor_control.log
  full_donor_control.log
  a3_from_a4_dims12.log
  a3_from_a4_dims12/best_mlp/
  verify.log
```

To repeat the fit, use the fresh CE command above with `--block=2
--graft_block=3 --graft_dimensions=2 --train_final_norm --readout_width=20
--steps=120000 --seconds=1800 --eval_every=2000 --seed=3`, retaining
`--fresh_branch --random_init=3`, batch 32, and the same rate schedule.

## Coordinate and PCA graft sweep (2026-09-25)

The completed sweep contains **75 screening fits at 20,000 updates**:
all ten single columns, all 45 pairs, all ten nine-column subsets, six nested
subsets of sizes 3–8, two PCA planes, and fresh A3/A4 controls. Seven selected
configurations were then fitted fresh for **120,000 updates**. The nested
sequence follows the selected order `1,9,7,2,8,6,4,5,3,10`; it is not an
exhaustive search over larger subsets. All column numbers below are one-based.

Every fit uses all **1,024 facts and 10,002 scored suffix-plus-EOS tokens**.
The frozen representation feeds a fresh residual 10 -> 20 -> 10 MLP, its
trainable input LN, and trainable final LN: **470 trainable parameters**.
Initialization and shuffle seeds are both 3. The protocol is cross-entropy,
batch 32, cosine learning rate 0.001 -> 0.0001, evaluation every 2,000 updates,
and a nonbinding 1,800-second fitting cap. Checkpoint selection minimizes
wrong-token count, with squared-margin loss breaking ties. Each budget has
its own complete cosine schedule: a screen is not a prefix of a full fit.
The source model and vocabulary head remain frozen. Token accuracy is
teacher-forced; exact-fact counts use fresh autoregressive completions with
the same graft recomputed from each generated prefix.

### Completed 20,000-update screens

| Representation | Correct / 10,002 | Token accuracy | Exact facts / 1,024 |
|---|---:|---:|---:|
| A3, fresh screen control | 2,662 | 26.6147% | 0 |
| Best single: column 1 | 2,870 | 28.6943% | 1 |
| Best pair: columns 1,9 | 3,226 | 32.2535% | 1 |
| Plot PCA plane | 2,753 | 27.5245% | 1 |
| Full-corpus PCA plane | 2,992 | 29.9140% | 1 |
| Eight columns: 1,2,4,5,6,7,8,9; omit 3,10 | 5,545 | 55.4389% | 10 |
| Nine columns: copy all except 10 | 8,148 | 81.4637% | 221 |
| Best nine-column token score: copy all except 7 | 8,185 | 81.8336% | 190 |
| A4, fresh screen control | 9,213 | 92.1116% | 507 |

Column 5 illustrates dependence on the other copied coordinates: alone it
scores 2,660 correct tokens, versus A3's 2,662; adding it to the selected
seven-column subset raises the score from 4,258 to 5,545. Omitting column 5
gives the lowest of the ten nine-column scores, 5,721. These observations
describe this fit's subset dependence, not an isolated causal role for a
column: the graft also changes normalization statistics and optimization.

The plot PCA is an **error-selected exploratory plane**. Its 64 pooled A3/A4
vectors come from the fifth prompt token of the first 32 facts whose
sixth-token prediction the A3 readout got wrong. Targets do not enter the PCA
algebra, but labels influence which facts were selected. The full-corpus PCA
uses all 10,002 scored rows per layer, or 20,004 pooled vectors, centered and
unscaled, with no error-based selection. On that same full-corpus population,
the full-corpus and plot planes retain **53.17% and 41.97%** of variance,
respectively. Each fixed rank-two basis `U` defines
`x_hybrid = x_A3 + U U^T (x_A4 - x_A3)`; common centering cancels. Variance
retention and fitted readout accuracy measure different properties.

### Completed 120,000-update fits

All seven new fits below completed 120,000 updates. A3, A4, and the earlier
columns 1,2 graft are explicitly historical comparisons using the matched
fresh, trainable-final-LN protocol; they were not rerun as full-budget sweep
controls. These results are separate from the short-screen ranking.

| Representation | Correct / 10,002 | Token accuracy | Exact facts / 1,024 | Best step |
|---|---:|---:|---:|---:|
| A3, historical matched baseline | 2,757 | 27.5645% | 0 | 74,000 |
| Plot PCA plane, new | 2,859 | 28.5843% | 1 | 78,000 |
| Full-corpus PCA plane, new | 3,114 | 31.1338% | 1 | 80,000 |
| Column 1, new | 2,965 | 29.6441% | 1 | 52,000 |
| Columns 1,2, historical graft | 3,125 | 31.2438% | 1 | 52,000 |
| Columns 1,9, new | 3,337 | 33.3633% | 1 | 110,000 |
| Eight columns: 1,2,4,5,6,7,8,9; omit 3,10, new | 5,759 | 57.5785% | 14 | 108,000 |
| Nine columns: copy all except 7, new | 8,997 | 89.9520% | 417 | 70,000 |
| Nine columns: copy all except 10, new | 9,021 | 90.1920% | 441 | 112,000 |
| A4, historical matched baseline | 9,832 | 98.3003% | 867 | 116,000 |

The best tested full-budget nine-column graft reaches 90.1920% token accuracy
and 441 exact facts, below the matched A4 baseline's 98.3003% and 867 facts.
The two selected nine-column choices reverse their token ranking between
budgets. These are single-seed optimization results on the training corpus;
they do not establish a globally optimal subset, a minimal sufficient
dimension, held-out generalization, or exclusive storage of facts in particular
coordinates. The hybrid still requires A4 donor computation.

### Local artifacts and reproduction

The ignored run directory contains the [consolidated HTML report](../runs/attention_graft_sweep_20260925/summary.html),
[machine-readable results](../runs/attention_graft_sweep_20260925/summary.json),
[fixed plane provenance](../runs/attention_graft_sweep_20260925/planes/manifest.json),
and every fitted checkpoint. Its six cohort directories are `screen`,
`larger_screen`, `leave_one_out_screen`, `full_early`, `full_final`, and
`full_extra`; each has exact argv and input SHA-256 fingerprints in
`manifest.json`, plus logs, evaluation histories, `results.tsv`, and a report.
Historical comparison logs remain under
[`final_norm_20260925`](../runs/final_norm_20260925/) and
[`attention_graft_20260925`](../runs/attention_graft_20260925/).

The [sweep driver](../../../../../scripts/memorize_general_facts/run_attention_graft_sweep.py)
requires a new output directory. From the repository root, reproduce the
59-run singles/pairs/PCA/control cohort using the saved plane files:

```bash
data_root=/home/ubuntu/checkpoints/memorize_general_facts/dataset_weights_canonical_order_0
artifacts=src/llm/experiments/memorize_general_facts/runs/attention_graft_sweep_20260925
python3 scripts/memorize_general_facts/run_attention_graft_sweep.py \
  --binary=bazel-bin/src/llm/experiments/memorize_general_facts/fit_attention_readout/fit_attention_readout \
  --source-checkpoint="$data_root/baseline/checkpoints/layers_4/step_120000" \
  --tokenizer="$data_root/inputs/tokenizer" --corpus="$data_root/inputs/corpus.txt" \
  --output=/tmp/attention-graft-screen-repeat --phase=screen --steps=20000 --jobs=2 \
  --plane="pca_plot_position5=$artifacts/planes/pca_plot_position5.txt" \
  --plane="pca_full_corpus=$artifacts/planes/pca_full_corpus.txt"
```

For the other screening cohorts, omit the two `--plane` options and add
`--variants-file="$artifacts/nested_variants.json"` or
`--variants-file="$artifacts/leave_one_out_variants.json"`, each with a new
output directory. For the seven full fits, use `--phase=full --steps=120000`
and one of `full_early_variants.json`, `full_final_variants.json`, or
`full_extra_variants.json` in that directory, again with separate new outputs.
`build_report.py` refreshes the consolidated report from all six manifests.
Validation passed 36 targeted GPU tests and 11 driver CPU tests; all 82 new
fit results were also checked against their `FINAL` log metrics. All seven
new full-budget saved fits were independently reloaded and reproduced their
selected token and exact-fact counts. Their 470 saved parameters are finite,
and the frozen source checkpoint remains unchanged.

## A3 middle-width sweep (2026-09-25)

Five fresh fits widen only the replacement MLP to 4d, 8d, 12d, 16d, and 20d,
where d=10. The source remains the same memorized four-block checkpoint with
width-20 MLPs, and capture still uses native A3 activations (`--block=2`).
There is no A4 donor or coordinate graft. Both input and final LayerNorm train
alongside the residual 10 -> H -> 10 GELU MLP; its trainable count is
`21 * H + 50`. The source transformer and vocabulary head remain frozen.

Each run executed exactly **120,000 updates**, with fresh initialization and
shuffle seeds 3, batch 32 facts, cross-entropy, Adam beta1=0.9/beta2=0.999,
epsilon=1e-8, no clipping/weight decay, and cosine rate 0.001 -> 0.0001.
As before, input/output matrices start with normal standard deviations
0.2/0.1; these are not rescaled by replacement width. Evaluate every 2,000
updates and select the fewest wrong scored tokens, breaking ties by squared
margin loss. `--stop_on_zero_errors=false` enforces the fixed step budget;
the 3,600-second wall cap did not bind. All 1,024 facts and all 10,002 scored
suffix/EOS positions participate, not just the first completion position.

| Middle width | Trainable parameters | Correct / 10,002 | Token accuracy | Exact facts / 1,024 | Best step |
|---|---:|---:|---:|---:|---:|
| 2d = 20, historical matched baseline | 470 | 2,757 | 27.5645% | 0 | 74,000 |
| 4d = 40 | 890 | 2,887 | 28.8642% | 0 | 98,000 |
| 8d = 80 | 1,730 | 3,000 | 29.9940% | 0 | 106,000 |
| 12d = 120 | 2,570 | 3,141 | 31.4037% | 0 | 104,000 |
| 16d = 160 | 3,410 | 3,260 | 32.5935% | 0 | 116,000 |
| 20d = 200 | 4,250 | 3,360 | 33.5933% | 0 | 112,000 |

The largest fit improves by 603 scored tokens, or 6.0288 percentage points,
over the earlier matched 2d fit, but none completes a whole fact correctly.
Thus this sweep does not find a memorizing width through 20d under the fixed
training protocol. These one-seed optimization outcomes do not establish a
representational lower bound or show that a different initialization or
optimizer could not fit the same width. The table reports each selected
checkpoint's independent greedy suffix-plus-EOS accuracy, not the final
iterate or a separately optimized whole-fact metric.

The ignored [HTML report and curves](../runs/a3_width_sweep_20260925/report.html)
and [summary](../runs/a3_width_sweep_20260925/summary.json) link every checkpoint
and log. Exact commands and input fingerprints are in
[`full/manifest.json`](../runs/a3_width_sweep_20260925/full/manifest.json).
To repeat, use the sweep driver command above with
`--variants-file=src/llm/experiments/memorize_general_facts/runs/a3_width_sweep_20260925/variants.json`,
`--phase=full --steps=120000 --seconds-per-run=3600`, a new output directory,
and no `--plane` options. Five concurrent fits took about 628 seconds each,
including capture and final verification; this is not an isolated timing
benchmark.

All five saved readouts were independently reloaded with zero updates and
reproduced token and whole-fact counts exactly. Their eight tensors have the
expected shapes and finite FP32 parameters, and source/saved checkpoint
hashes remained unchanged. Ten new width-specific GPU tests cover all five
widths, updates to every trainable tensor, frozen source/head weights, and
strict checkpoint round trips including wrong-width rejection; all 21 readout
tests pass. Two full-corpus CLI controls verify that the default stops a
perfect readout at zero updates, while disabling early stopping executes the
requested two updates. The sweep driver's 11 CPU tests also pass.

## Initialization recipes and fixed per-token preprocessing

Fresh-readout fitting additionally accepts `--input_init_std` (default 0.2),
`--output_init_std` (default 0.1, including zero), and `--scale_output_init`.
The last option multiplies the output deviation by
`sqrt(source_feed_forward_width / readout_width)`, keeping the original
20-wide initialization as the reference scale. `--fresh_final_norm` starts
the trainable final LayerNorm at identity rather than the source parameters;
it requires a fresh branch and `--train_final_norm=true`. Existing defaults
and saved-weight ordering are unchanged. Training still supports CE or
squared-margin loss, minibatch size, initialization/shuffle seeds, learning
rate, and cosine end/start ratio (one gives a constant learning rate).

`--preprocessing=identity|dct|dft|sin|cos|signed_sqrt|random_fourier` applies a
fixed transform to each captured token independently. It never reads labels,
other token positions, or corpus-fitted statistics. Input and output widths
are equal: a 10/150/10 readout remains 10/150/10 and has 3,200 trainable
parameters with both LayerNorms. Preprocessing is applied before the entire
residual readout, so its skip connection also receives the transformed vector.
The same capture function is used for training and greedy-prefix verification.

DCT-II and the packed real DFT are orthonormal rotations before BF16 rounding.
DFT outputs DC, cosine/sine pairs, and Nyquist coordinates rather than dropping
the imaginary components. These rotations do not spread Euclidean distances;
they change the basis seen by normalization and the readout. Signed square root
is `sign(x)*sqrt(abs(x))`. Sin/cos use `--preprocessing_scale` as their phase
scale. Random Fourier features use half as many fixed Gaussian projections as
input coordinates, returning their cosines followed by sines; the Gaussian
variance is `1/model_width`, with `--preprocessing_seed` selecting the matrix.
Both trigonometric modes and random Fourier features may lose information.
All arithmetic inside the transform is FP32, with BF16 output snapshots.

Capture reports exact unique BF16 vectors and conflicting target groups on
the scored rows, canonicalizing signed zero. This check detects exact
information loss, not ill-conditioning or near-collisions. Evaluations now
report actual mean cross entropy as well as squared-margin diagnostics,
regardless of the selected training objective. No preprocessing coefficients
are optimized or saved as weights; retain the CLI transform and seed with the
readout checkpoint. The sweep manifest records them and source fingerprints.

Use `scripts/memorize_general_facts/analyze_readout_recipe_sweep.py RUN_ROOT`
to produce a self-contained HTML summary. It separates screening runs and
confirmations by cohort and requested update count, and excludes checkpoint
replays from rankings. Timings with concurrent fits are not isolated benchmarks.

### Width-150 recipe/preprocessing results (2026-09-26)

The one-hour experiment completed 42 fits and 2.7 million optimizer updates:
23 recipe screens at 40k updates, three recipe confirmations at 120k, 13
preprocessing screens at 40k, and three preprocessing confirmations at 300k.
Every fit uses A3, a fresh 10/150/10 MLP, both trainable LayerNorms, and the
same frozen source and vocabulary head. None approaches corpus memorization.

The six baseline seeds span 30.824–31.094% next-token accuracy at 40k.
At a matched 120k budget, baseline LR 0.001 reaches 32.354%, LR 0.01 reaches
32.953%, and width-scaled output initialization with LR 0.003 reaches 32.793%.
All three complete zero facts. The preprocessing sweep uses the LR 0.01
recipe; no transform beats identity in the 40k screen. DFT/DCT achieve
30.694%/29.354%, versus identity's 32.384%. Stronger trigonometric folding
and fixed random Fourier features perform substantially worse.

The best two preprocessing candidates and identity were then trained fresh
for matched 300k schedules:

| Input transform | Correct / 10,002 | Token accuracy | Mean CE | Complete facts / 1,024 |
|---|---:|---:|---:|---:|
| Identity | 3,341 | 33.403% | 3.55857 | 0 |
| Elementwise sin(x) | 3,293 | 32.923% | 3.59804 | 0 |
| Signed square root | 3,210 | 32.094% | 3.68409 | 0 |

All 13 transforms preserve distinct vectors for all 10,002 scored inputs;
no exact conflicting-label collision explains their poor fit. All three
300k saved checkpoints independently replay with identical metrics and tensor
bytes. The preprocessing/readout GPU tests and report/driver CPU tests pass.
This is a negative result for these finite BF16 fitting recipes, not proof
that the architecture cannot interpolate A3. Preprocessing changes the
residual skip and the basis seen by LayerNorm while keeping the head fixed.

The ignored [combined report](../runs/a3_recipes_preprocessing_20260926/summary.html)
contains all trials and curves; its adjacent results, protocol, audits, and
manifests document exact commands, checksums, and saved checkpoints.
