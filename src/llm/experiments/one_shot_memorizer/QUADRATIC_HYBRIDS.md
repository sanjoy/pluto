# Testing whether trace repairs generalize to complete continuations

This is a post-hoc follow-up to the 28 selected
[quadratic failure traces](QUADRATIC_FAILURE_TRACES.md). Restoring the first
learned MLP repaired 24 next-token decisions. That does not establish that
their entire suffixes become correct, or that the repair leaves previously
correct sentences intact.

## Results

The fixed experiment completed at `/tmp/quadratic_mlp_hybrid_0`. Original
controls, the exact all-quadratic reproduction, and every original/replacement
master-byte check passed. No coefficients were refitted.

| Original MLPs retained | Correct targets, /10,002 | Complete suffixes plus EOS, /1,024 |
| --- | ---: | ---: |
| All eight | 10,002 | 1,024 |
| None | 9,968 | 996 |
| 0 | 9,995 | 1,017 |
| 7 | 9,977 | 1,001 |
| 0, 7 | 9,999 | 1,021 |
| 0, 1 | 9,997 | 1,019 |
| 0, 1, 7 | 10,001 | 1,023 |

Thus five of eight MLPs can use the fixed, directly fitted quadratic maps
while retaining 1,023 complete sentences. This is a teacher-assisted hybrid,
not a full one-shot replacement of the trained network. The five wider
quadratic maps increase the nominal parameter count from 114,256 to 115,856;
this is not a parameter-reduction result.

Repairs are not monotone. Keeping block 0 restores 24 formerly failed
sentences but introduces three newly failed sentences (lines 480, 503, 540).
Keeping block 7 restores eight but introduces three (292, 392, 897).
Keeping 0 and 7 restores 27 but introduces two (480, 503). Keeping 0 and 1
restores 24 but introduces one (540). The fixed 0,1,7 combination restores
27 without introducing any new failed sentence.

The remaining line is:

> Condensation occurs when a gas changes into a liquid.

The all-quadratic model first substitutes ` liquid` for ` gas` after the
five-token prompt. The 0,1,7 hybrid gets that token right, but later substitutes
` gas` for ` liquid` after `Condensation occurs when a gas changes into a`.
That is a **shifted failure point**, not persistence of the identical query
error. The hybrid is exact on 818/819 fitting and 205/205 regression-held
sentences, but the subset choices used observed corpus failures; this is not
an independent generalization test.

Targeted verification passed the 14 MLP-probe cases, nine strict artifact-reader
cases, optimized build, malformed-CLI checks, and the complete seven-condition
run. The no-original-MLP control matches every previous per-sentence result
and all 28 first-failure records, not merely the aggregate accuracy.

## Unregularized precision follow-up

`/tmp/quadratic_precision_audit_unregularized_0` removes both the BF16 product
rounding and the ridge penalty for a separate diagnostic fit. All eight
152-column feature designs have full numerical rank under the unchanged
`1e-12` QR criterion. No rank threshold was relaxed and no failed fit retried.

The unregularized fitting relative errors are 3.1073%, 2.5148%, 1.3246%,
1.5727%, 1.7142%, 2.8021%, 2.4338%, and 6.1783% for blocks 0–7. Removing
ridge reduces fitting error by only 0.010%–0.057% **relative**, not percentage
points. The last block's held update error remains 6.2001%.

The substantial residual is therefore not explained by the chosen ridge
penalty or feature-product rounding alone. This is numerical least-squares
evidence on the observed BF16 inputs and teacher updates, not a theorem about
all quadratic networks or a separation of the teacher's own quantization
from higher-order smooth terms. Unregularized coefficients can become much
larger (maximum absolute coefficient 9.36666) with little error improvement;
they are not deployed in the hybrid. The four original precision-report
tables are byte-identical, and all 19,584 saved FP32 coefficient comparisons
still match. Eleven precision-helper tests and the optimized CLI passed.

## Fixed protocol

Load the existing saved quadratic coefficients; do not refit any parameter.
Retain the original embeddings, attention, LayerNorms, and head in all cases.
Test these preselected sets of **original MLPs to retain**:

* all eight: original positive control;
* none: reproduction of the fully quadratic result;
* block 0;
* block 7;
* blocks 0 and 7;
* blocks 0 and 1;
* blocks 0, 1, and 7.

Every other MLP uses the fixed quadratic-feature replacement on its fresh
input. Block 0 is motivated by its 24 query repairs, block 7 by the local
final-boundary failures, and block 1 by the second-largest single-function
repair count. The compound choices test interactions rather than assuming
their effects form a union. No additional subsets will be selected within
this protocol in response to the results.

For each condition, evaluate every teacher-forced target and independently
generate all 1,024 complete suffixes plus EOS from the first five tokens.
Use batch size 32, the same target masks, and the unchanged compact vocabulary.
Require per-sentence teacher-forced exactness to agree with greedy exactness.
Retain first-failure metadata, per-sentence outcomes, and fitting/held groups.
The latter refer only to the previous regression split: all 1,024 sentences
were part of the backbone's original training data, and this subset choice
was informed by observed failures.

The no-original-MLP control must reproduce 9,968/10,002 targets and 996/1,024
complete sentences. Both initial and final original controls must be perfect.
All original and quadratic master tensors must remain byte-identical.
Generated artifacts stay local; tested code and the report will be committed.

## Interpretation

Even a perfect hybrid is **not** a dataset-only construction: its remaining
learned MLPs and backbone are retained, and the quadratic coefficients were
fit to captured learned updates. This experiment tests how much of the model's
nonlinear computation can be replaced by a fixed, explicit representation
without gradient descent. It does not uniquely decode facts from weights or
establish ownership of particular facts by the retained blocks.

A separate precision follow-up will try unregularized FP64 QR on the same
unrounded quadratic inputs, reporting rank failure rather than changing
tolerances. This distinguishes the fixed ridge penalty from the approximation
residual without changing any deployed hybrid coefficient.

## Reproduction

```bash
bazel build -c opt \
  //src/llm/experiments/one_shot_memorizer:checkpoint_quadratic_hybrid_probe
facts_run=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_quadratic_hybrid_probe \
  --checkpoint="$facts_run/layers_8/step_16128" \
  --tokenizer="$facts_run/inputs/tokenizer" \
  --construction_dir=/tmp/quadratic_mlp_basis_failures_0 \
  --output_dir=/tmp/quadratic_mlp_hybrid_new
```

The precision command in [QUADRATIC_FAILURE_TRACES.md](QUADRATIC_FAILURE_TRACES.md)
now also writes the separate `ridge_zero_unrounded_*` tables. Use a fresh
output directory; the primary precision fits remain unchanged.
