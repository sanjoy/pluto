# Does a shared embedding offset explain the failed pair composition?

## Observation before the repair experiment

The fixed [SUM/MEAN experiment](FACT_SUPERPOSITION.md) fails to combine the
individually memorized France/Greece facts. Its endpoint parameter errors
relative to the jointly trained model are concentrated in the embedding:

| Group | SUM squared-error share | MEAN squared-error share |
| --- | ---: | ---: |
| Token embedding | 99.4092% | 99.2742% |
| All attention branches | 0.3468% | 0.4990% |
| All MLP branches | 0.1846% | 0.1736% |
| Final LayerNorm | 0.0552% | 0.0486% |
| Position embedding | 0.0042% | 0.0047% |

For an embedding difference matrix `Delta`, its orthogonal projection onto
row-constant matrices has every row equal to `mean_rows(Delta)`. This shared
16-dimensional row shift accounts for **99.0721%** of SUM's embedding error
and **98.8867%** of MEAN's. Those are 98.4868% and 98.1690% of the respective
whole-model squared errors. From initialization, the corresponding fractions
within each learned embedding update are 99.1639% for France-only, 99.3324%
for Greece-only, and 99.1838% for joint training.

These are descriptive parameter-space projections, not causal percentages.
For fixed hidden states in exact arithmetic, a shared head-row shift adds the
same scalar to every logit, leaving softmax unchanged. But the embedding is
tied to the model's inputs too, and BF16 rounding can break that exact
cancellation. A large parameter error along this direction therefore does
not by itself establish a behavioral explanation.

Evidence: `/tmp/fact_superposition_error_analysis_0/`, computed in FP64 over
all 100 unique tensors and 114,256 coordinates. Saved SUM/MEAN values were
also checked bytewise against their prescribed construction at every scalar.

## Fixed prospective repair

For each existing SUM and MEAN candidate, compute one column-wise shift from
the **jointly trained reference** embedding:

```
c = mean_rows(FP64(E_joint) - FP64(E_candidate))
E_repaired[row, column] = FP32(FP64(E_candidate[row, column]) + c[column])
```

Use FP64 mean/shift arithmetic and one final FP32 cast per scalar. Keep the
other 99 unique tensors byte-identical, preserve the tied input/head alias,
and do not compensate position embeddings. Before that final cast, all
centered row differences are preserved; the repair changes only the common
input offset and the conditionally softmax-invisible head offset. This is
**not asserted to preserve full-model behavior**.

There are exactly two additional candidates: mean-matched SUM and mean-matched
MEAN. Do not tune the shift size, select dimensions, fit on outputs, or try
alternative reference means after seeing accuracy. The original five models
remain controls, and the default probe must keep its prior outputs unchanged.
Report both teacher-forced target counts and genuine first-five-token greedy
suffix/EOS completion, plus the shifts and parameter errors before/after.

This is explicitly an **oracle-assisted diagnostic**, not a new one-shot
construction: it uses 16 values derived from the jointly trained model. Even
if it works, it would not derive that shift from the two component models or
show that those 16 coordinates exclusively store the facts. If it fails, it
would show that removing this dominant Euclidean-error component is not
sufficient to restore the tested behavior, not that the offset has no effect.

Tests must cover column-wise rather than scalar centering, identity with zero
shift, input preservation, malformed shapes, nonfinite values and FP32
overflow. Runtime checks verify unchanged non-embedding tensors, tied aliases,
fresh model state and unchanged sources. All generated checkpoints/reports
remain local. Budget: 45 minutes, with no new training.

## Result: much closer weights, worse answers

The two predeclared repairs completed; no additional reference, scale,
coordinate subset or repair was tried. The implementation computes the mean
of FP64 pairwise differences, equivalent to the difference of column means
in real arithmetic, then applies the one prescribed FP32 output cast.

| Candidate | Correct teacher-forced targets | Exact suffixes plus EOS | L2 distance to joint weights |
| --- | ---: | ---: | ---: |
| SUM | 6/18 | 0/2 | 74.8787 |
| SUM with oracle mean | 5/18 | 0/2 | 9.21096 |
| MEAN | 4/18 | 0/2 | 55.4063 |
| MEAN with oracle mean | 1/18 | 0/2 | 7.49733 |

Embedding squared error falls from **5,573.693887 to 51.717389** for SUM and
from **3,047.580503 to 33.928186** for MEAN. This removes more than 98% of the
whole-model squared parameter error in either case. Nevertheless, target
accuracy worsens, and neither entire sentence is recovered.

The repaired SUM predicts a comma immediately after both five-token prompts.
The repaired MEAN predicts Athens for France; for Greece it gets Athens right
but then repeats Athens instead of the following comma. The teacher-forced
counts above score gold contexts separately; they are not the number of
correct tokens in a free-running completion. Greedy checks feed back only
predictions and stop at the first mismatch.

The result rejects this fixed shared-offset correction as a sufficient repair
for the two constructed models. It does not show that the common offset is
irrelevant: it changes their behavior, and it is tied to the input embeddings.
Nor does it show that the small remaining parameter error exclusively owns
either fact. It demonstrates why a very large Euclidean-error reduction can
be a misleading proxy for restoring a co-adapted model's computation.

### Verification and reproduction

Six focused CPU tests cover column-wise centering, centered differences,
zero-shift signed-zero identity, FP64 cancellation, invalid shapes/nonfinite
values and output overflow. Runtime controls verify fresh model uploads,
all seven input/head aliases, both unchanged 99-tensor tails, unchanged
inference/save weights and source checkpoints. Independent saved-checkpoint
readback verified all **143,200 repaired embedding values** against the
prescribed arithmetic and every byte of the other 99 tensors in both models.
The remaining mismatch of column means is at most `1.44e-8` for SUM and
`1.11e-8` for MEAN, from the final FP32 rounding.

The original five conditions reproduce their previous numeric outputs
exactly, both in the flagged run and a separate unflagged run. Original SUM
and MEAN checkpoint trees are byte-identical as well. Compared with the oldest
saved artifact, `controls.tsv` has one already-landed wording correction:
`all_source_checkpoint_bytes_unchanged` was narrowed to
`all_source_weight_bytes_unchanged`; unrelated checkpoint metadata is not
audited. This is not a numerical or behavioral change introduced by the repair.

The optimized build, independent implementation review and full repository
suite passed: **107 test targets**. The completed local report is
`/tmp/fact_superposition_mean_shift_0/`; the unchanged-default repeat is
`/tmp/fact_superposition_default_repeat_0/`. The flagged report adds
`embedding_mean_shifts.tsv`, `embedding_mean_errors.tsv`, two condition rows,
and saved checkpoints under `sum_oracle_embedding_mean/step_1024` and
`mean_oracle_embedding_mean/step_1024`. Generated artifacts remain outside Git.

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_fact_superposition_probe
facts=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
pair=/tmp/fact_superposition_pair_0
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_fact_superposition_probe \
  --initial_checkpoint="$facts/layers_8/step_0" \
  --joint_checkpoint="$pair/baseline/step_1024" \
  --omit_line_1_checkpoint="$pair/omit_line_1/step_1024" \
  --omit_line_2_checkpoint="$pair/omit_line_2/step_1024" \
  --tokenizer="$facts/inputs/tokenizer" \
  --corpus=/tmp/fact_superposition_pair.txt \
  --control_step1_dir="$pair" \
  --match_joint_embedding_mean \
  --output_dir=/tmp/fact_superposition_mean_shift_new
```
