# Fitting an MLP followed by a transformer after attention 3

This is an AI-generated experiment report, not human-reviewed documentation.

## Question and architecture

Can the exact original suffix be learned again from frozen A3 activations,
under the same training recipe used for the stacked pointwise MLPs?

The source is the memorized four-block GPT-2 model: hidden width 10,
feed-forward width 20, one attention head, context 27, vocabulary 4,475.
The replacement consumes complete cached `[batch, 27, 10]` BF16 sequences:

1. `x += FC2(GELU(FC1(LN(x))))`, with widths `10 -> 20 -> 10`.
2. `x += Out(CausalAttention(QKV(LN(x))))`, with one width-10 head.
3. Another residual `10 -> 20 -> 10` pre-LN MLP.
4. Final LayerNorm, then the frozen tied embedding projection.

Steps 2 and 3 are one complete transformer block. All residual operations,
biases, and normalizations match the original source suffix. There are 20
trainable FP32 parameter tensors, with BF16 activations and compute policy:

| Component | Parameters |
| --- | ---: |
| First MLP including its input LN | 450 |
| Attention, projections, biases, input LN | 460 |
| Second MLP including its input LN | 450 |
| Final LN | 20 |
| **Total** | **1,380** |

## Results

The fresh MLP-plus-transformer suffix successfully memorized the full corpus.
All entries below use the same frozen source checkpoint, batch/seed/optimizer
recipe, 300,000-step budget, and best-checkpoint selection rule. MLP-only rows
are the completed September 27 stack sweep, not newly rerun baselines.

| Trainable suffix after A3 | Parameters | Best scored-token accuracy | Exact greedy facts |
| --- | ---: | ---: | ---: |
| One residual 10/150/10 MLP + final LN | 3,200 | 35.9328% | 1 / 1,024 |
| Three residual 10/20/10 MLPs + final LN | 1,370 | 34.8730% | 1 / 1,024 |
| Five residual 10/150/10 MLPs + final LN | 15,920 | 83.1534% | 210 / 1,024 |
| **10/20/10 MLP + full transformer block + final LN** | **1,380** | **100%** | **1,024 / 1,024** |

The first observed perfect teacher-forced evaluation was at step **19,000**
(26.79 seconds; mean CE .01574161). Evaluations were only every 1,000 steps,
so this is not necessarily the first perfect update. Training continued for
the matched budget, sometimes losing accuracy again under the high LR.
161 of the 301 evaluations were perfect; the last imperfect evaluation was
at step 279,000, and all evaluations from 280,000 through 300,000 were perfect.

Best checkpoint: **step 298,000**, **10,002/10,002** correct tokens, mean CE
**.00133080**. Reloading that suffix independently of the last update and
recomputing activations from generated prefixes verified **1,024/1,024**
complete suffix-plus-EOS generations. Final step 300,000 was also perfect,
with slightly higher CE .00137841. Source and tied-head byte comparisons
both passed.

Training plus regular evaluation took 427.16 seconds; restoring the best
checkpoint and final greedy/frozen-head verification brought the native fit
timer to 430.71 seconds (**7m 11s**). This timer excludes initial capture,
HTML writing, the original-weight control, and compilation. Brief GPU tests
also ran during this fit, so this is not an isolated throughput benchmark.

The contrast is strong: a suffix with about 11.5 times fewer trainable
parameters than the largest MLP stack achieved perfect completion, and a
roughly parameter-matched three-MLP stack stayed near 35% token accuracy.
Thus frozen A3 states are not generally untrainable inputs: this small,
sequence-aware architecture can learn a perfect completion rule under this
recipe. That does not establish a pointwise-MLP impossibility or identify
exactly what the learned attention computes.

## Protocol

The checkpoint and input snapshots are exactly those used by the previous
stack sweep, not the similarly shaped older step-120000 model:

```text
Original source:
/tmp/pluto-puzzle-from-scratch-20260926-01/model/trial_000_L4_W10_FF20/checkpoints/layers_4/step_89600
Immutable comparison inputs:
/tmp/pluto-a3-stacked-mlp-sweep-20260927-01/inputs/
New run:
/tmp/pluto-a3-mlp-transformer-20260928-01/
```

Both comparisons use 300,000 updates, batch size 32 complete facts, seed 3,
Adam with beta1=.9, beta2=.999, epsilon=1e-8, no clipping/weight decay,
and cosine LR .01 -> .001. Evaluation occurs every 1,000 updates. The best
checkpoint is selected by scored-token accuracy, breaking ties with mean CE.
The source model and independent copy of its tied embedding/head are frozen.

Initialization is fresh except for final LN, which starts as a trainable copy
of the source final LN, matching the stacked-MLP experiment. Each MLP starts
with identity input LN, zero biases, FC1 normal standard deviation .2 and FC2
.1. The two MLPs use seeds 3/4 and 5/6. Attention QKV/output projections use
seeds 7/8 and standard deviations .02/.005, matching GPT-2's initialization
scales. Its input LN is identity and biases zero.

The standard cross-entropy loss scores the 10,002 suffix/EOS targets after
five-token prompts. Prompt states are retained as attention keys/values;
masking their direct loss must not mask gradients reaching them through
attention. Right padding is causally invisible to scored positions. No
attention crosses between facts. Greedy verification uses actual generated
prefixes, not cached future tokens.

Before training, a separately allocated suffix copies the original learned
tensors 32..51. It reproduces **10,002/10,002** scored tokens and
**1,024/1,024** greedy completions. Those control weights are discarded and
never initialize the fresh fit. Targeted tests also compare logits bitwise,
test causal future independence and prompt gradients, and verify isolation
of source/head weights from optimizer updates.

## Reproduction

```sh
bash scripts/memorize_general_facts/run_mlp_transformer.sh \
  /tmp/pluto-a3-stacked-mlp-sweep-20260927-01/inputs/checkpoint \
  /tmp/pluto-a3-stacked-mlp-sweep-20260927-01/inputs \
  /tmp/pluto-a3-stacked-mlp-sweep-20260927-01/inputs/corpus.txt \
  /tmp/new-mlp-transformer-run
```

Substitute a matching memorized checkpoint and its tokenizer/corpus on another
machine. The output directory must be new. The launcher builds the binary and
uses `--mode=puzzle --train_mlp_transformer`. This selector is mutually exclusive
with the MLP-only selectors, requires four source blocks, and takes its MLP
width from `--feed_forward_width`, not `--mlp_width`.

Generated checkpoints, logs, and plots stay local. The run contains
`console.log`, `puzzle/run.txt`, `puzzle/training.tsv`, and the best 20-tensor
suffix checkpoint in `puzzle/best_mlp_transformer/`. This is a suffix-only
checkpoint, not a directly loadable full GPT-2 checkpoint.

## Validation

```sh
bazel test -c opt --test_output=errors --nocache_test_results \
  //src/llm/experiments/memorize_general_facts:all
```

All five test targets passed: CLI policy, binary flag/default handling,
existing MLP readout, puzzle report, and the new transformer readout. The
new target has nine tests, including production-size bitwise control,
trainable/frozen ownership and gradient checks, and causal forward/backward
checks. The launcher passes `bash -n`; new C++ files pass clang-format checks.

## Interpretation limits

This is one initialization/training seed and one finite-corpus comparison.
It does not prove that a pointwise MLP cannot fit the data. The new suffix can
retrieve other positions, whereas the MLP stacks cannot. Attention also has
its own initialization scale, so this experiment compares complete recipes,
not an isolated change to one algebraic operation.
