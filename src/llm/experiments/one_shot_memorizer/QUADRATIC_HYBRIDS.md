# Testing whether trace repairs generalize to complete continuations

This is a post-hoc follow-up to the 28 selected
[quadratic failure traces](QUADRATIC_FAILURE_TRACES.md). Restoring the first
learned MLP repaired 24 next-token decisions. That does not establish that
their entire suffixes become correct, or that the repair leaves previously
correct sentences intact.

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
