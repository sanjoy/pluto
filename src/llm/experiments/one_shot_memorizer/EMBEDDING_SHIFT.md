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
c = mean_rows(E_joint) - mean_rows(E_candidate)
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
