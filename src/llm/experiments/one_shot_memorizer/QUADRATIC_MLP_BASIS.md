# A fixed quadratic basis for the learned MLP updates

## Prospective protocol, following the fixed-initialization failure

This is an explicitly post-hoc follow-up to
[FROZEN_MLP_BASIS.md](FROZEN_MLP_BASIS.md), whose raw/standardized initial
64-feature constructions preserved only 13/6 complete sentences. It does not
change that completed protocol or tune its outcomes away.

Test a more interpretable nonlinear basis that requires **no learned
expansion directions**. For each fresh 16-dimensional normalized input `x`,
construct these 152 features in this fixed order:

```
x_0, ..., x_15,
x_0*x_0, x_0*x_1, ..., x_0*x_15,
x_1*x_1, x_1*x_2, ..., x_1*x_15,
...
x_15*x_15
```

There are 16 linear terms and 136 upper-triangular pair products. A separate
affine output bias supplies the constant term. Inputs are physical BF16;
products use FP32 multiplication followed by one BF16 rounding. Fitting and
live inference use the identical cuTile implementation. No feature scaling,
standardization, fitted thresholds, random draws, or feature selection is
allowed in this condition.

Motivation: the production tanh-approximated GELU has a linear term and a
quadratic leading nonlinear term near zero. A degree-two polynomial in an
affine projection can be represented in the basis above. This **does not**
assume that all learned preactivations are small, that higher-order terms are
unimportant, or that floating-point GELU is exactly quadratic. The test is
whether a directly fitted quadratic update suffices on this corpus.

Keep the checkpoint, context, corpus, original LayerNorms, attention,
embeddings, and output head fixed. Use the same 819 fitting/205 held-out
sentences and all real text-token rows, including prompt rows. Fit only the
152-to-16 output weights and bias by the existing FP64 pivoted QR routine,
with fixed mean-square ridge `1e-6` and an unpenalized bias. LayerNorm induces
dependencies among polynomial features; the ridge is not adaptively changed
in response to rank or accuracy.

Retain original, cloned-MLP, and learned-expansion/refitted-W2 positive
controls. Verify the quadratic layer against an independent CPU oracle,
including feature ordering, cross-row boundaries, the non-power-of-two
152-column width, and composition with the production dense layer.

Replace each block separately and all eight jointly. Every replacement must
consume the **current forward's** LayerNorm input, including perturbations
from earlier replacements. Evaluate all 10,002 supervised decisions and
independently generate all 1,024 suffixes plus EOS from five-token prompts.
Compare exactness by sentence, report fit/held update errors, preserve the
source weights, and restore original perfect performance afterward. The
initial checkpoint is neither required nor used to construct this basis.

## Connection to the learned matrices, without assuming sufficiency

For the **unrounded** MLP surrogate on post-LayerNorm `x`, write

```
y_q = d_q + sum_j B_jq * GELU(b_j + a_j^T x)
```

Here `a_j` is column `j` of the effective BF16 expansion matrix, `B` is the
effective BF16 contraction matrix, and `b,d` are the FP32 biases. Its local
quadratic expansion at `x=0` has constant, linear, and symmetric quadratic
coefficients

```
C_q   = d_q + sum_j B_jq * GELU(b_j)
L_iq  = sum_j B_jq * GELU'(b_j) * a_ij
Q_q   = sum_j 0.5 * GELU''(b_j) * B_jq * a_j a_j^T
y_q  ~= C_q + sum_i L_iq*x_i + x^T Q_q x
```

Thus the learned directions supply shared rank-one factors for all sixteen
output quadratics. This does not imply that each `Q_q` has low matrix rank:
sixty-four factors can span all sixteen input directions. In the raw
pair-product feature ordering above, diagonal coefficients are `Q_q[ii]`,
while off-diagonal coefficients are **twice** `Q_q[ik]`.

The experiment fits its coefficients over the recorded domain; it does not
set them to these Taylor coefficients. Nonzero expansion biases generally
produce cubic remainder, and BF16 rounding makes the actual GPU function
different from the smooth surrogate. Moreover, LayerNorm itself remains
nonlinear in the incoming residual. This connection motivates a testable
representation, not a prior conclusion that quadratic interactions explain
the entire model or its training dynamics.

### Why this does not uniquely recover the original directions

There is an additional identifiability issue in this implementation. In real
arithmetic, LayerNorm outputs have the form `x = beta + gamma * u`, where
`sum(u)=0`. If all gamma entries are nonzero, define `n_i=1/gamma_i` and
`c=dot(n,beta)`. Every such input satisfies `dot(n,x)=c`. Consequently,
changing an expansion direction and its bias by

```
a' = a + t*n
b' = b - t*c
```

leaves `dot(a,x)+b` unchanged on all those inputs. This is an algebraic
ambiguity in recovering off-manifold weights from normalized observations,
not an empirical claim that a particular GPU parameter edit was tested.
BF16 rounding perturbs the hyperplane and invalidates exact real-arithmetic
equivalence. Conversely, FP32 matrix masters within the same BF16 rounding
cells already produce identical effective operands. Neither observation
licenses assigning a unique semantic meaning to each original coordinate.

Non-backpropagation recovery methods do exist:
[NN-LIFT](https://arxiv.org/pdf/1506.08473) combines input-score moments,
tensor decomposition, bias estimation, and output regression. Its guarantees
require input-density, activation, and identifiability conditions; they do
not directly cover these finite, quantized LayerNorm samples and GELU units.
The associated [score-function identity](https://arxiv.org/pdf/1412.2863)
relates output/score moments to expected derivatives under differentiability
and boundary conditions. Applying those identities would require a justified
sampling model, not simply whitening the observed corpus vectors. Querying a
teacher on deliberately chosen synthetic inputs could investigate a different
recovery setting, but would still not construct the model from raw text alone.

## Scope and budget

This changes the expansion width from 64 to 152. Each original MLP has 2,128
parameters; a fixed quadratic expansion plus its fitted output map has 2,448.
Replacing all eight changes the nominal model count from 114,256 to 116,816,
an increase of 2,560 coefficients. The fixed pair-product operations are
additional computation, not stored trainable parameters.

Even success would be a slightly larger **teacher-assisted replacement of
the MLPs**, not recovery of their original coordinates or an entire model
constructed from text. The fitted targets and the normalized input geometry
still come from the learned backbone. Held-out sentences are excluded from
replacement fitting but were in that backbone's original training set.

Implementation, tests, and execution are budgeted at roughly one hour. Do
not silently expand to higher polynomial degrees, adjust ridge, or introduce
learned input rotations if this fixed construction fails. Any such follow-up
would need a separately recorded protocol and remaining research time.

## Result: nearly sufficient, but not exact

The fixed protocol completed on 2026-09-23. With **all eight** MLPs replaced,
the model preserves **9,968/10,002 next-token decisions** and
**996/1,024 complete greedy suffixes plus EOS**. The latter is 97.27%, not
100% memorization. Exactness is 796/819 in the fitting group and 200/205 in
the held-out group. Every original/clone/learned-feature-refit positive
control remains perfect, and all source master weights remain unchanged.

| Replaced block | Correct decisions / 10,002 | Exact suffixes / 1,024 |
| --- | ---: | ---: |
| 0 | 9,993 | 1,016 |
| 1 | 10,001 | 1,023 |
| 2 | 10,002 | 1,024 |
| 3 | 10,002 | 1,024 |
| 4 | 10,002 | 1,024 |
| 5 | 10,001 | 1,023 |
| 6 | 10,002 | 1,024 |
| 7 | 9,999 | 1,021 |
| All eight | 9,968 | 996 |

This substantially improves on the raw/standardized initialization bases,
which preserved 13/6 whole sentences jointly. In particular, block 0's
quadratic replacement preserves 1,016 sentences rather than 202/204.
However, the quadratic basis is wider: this is not a matched-width proof
that polynomial features are inherently better than every random-feature
construction.

### Joint substitutions cannot be predicted by unioning individual failures

The single-block substitutions fail on a union of twelve sentences. The
joint condition fails on 28: ten from that union, **eighteen new failures**,
and two formerly failed sentences become correct (lines 503 and 727).
Line 795 is the only overlapping individual failure, appearing in both
block 5 and block 7 substitutions.

These interactions are observed with fresh inputs to every replacement, not
cached original hidden states. They are compatible with accumulated changes
in the trajectory and nonlinear downstream responses; this comparison does
not identify their separate contributions.

### Approximation and numerical effects remain distinct questions

Across blocks, the FP64 fits have relative update residuals of
1.326%–6.179%, while GPU replacements on the **same recorded fitting inputs**
have 1.344%–6.186% residuals. Held-out GPU residuals are 1.388%–6.210%.
Both fitting and GPU paths start with BF16-rounded pair products. Consequently,
their similar residual norms do **not** isolate product quantization or
establish that higher polynomial degree is required. Small numerical changes
can still change a close token decision.

Every fit has augmented rank 152 at the fixed ridge. The reported QR diagonal
spreads, 288.45–427.25, are diagnostics, not spectral condition numbers or
unregularized-rank claims. Coefficients remain modest: maximum absolute W2
master is 0.177914 and maximum absolute FP32 bias is 0.201583. There are no
nonfinite coefficients or W2 underflows when rounded to BF16.

The useful constructive finding is that **fixed pairwise interactions of the
learned normalized coordinates reproduce most of the learned MLP behavior**,
and four blocks can individually be replaced without any corpus completion
error. This does not mean those coordinates have been semantically explained,
the remaining errors are unimportant, or the learned attention/embedding
geometry has been constructed from text.

## Reproduction and local evidence

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_frozen_mlp_probe
facts=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_frozen_mlp_probe \
  --experiment=quadratic \
  --checkpoint="$facts/layers_8/step_16128" \
  --tokenizer="$facts/inputs/tokenizer" \
  --output_dir=/tmp/quadratic_mlp_basis_new --batch_size=32
```

Do not supply `--initial_checkpoint` in quadratic mode: the tool rejects it
rather than accidentally consuming initialization-time directions.
Initialization mode remains the default for the earlier frozen-feature test.

Evidence is in `/tmp/quadratic_mlp_basis_0/` and
`/tmp/quadratic_mlp_basis_0.log`. The manifest records the basis, arithmetic,
fitting split, widths, controls, and successful completion. `sentences.tsv`
retains the per-sentence comparison, and `coefficients.tsv` contains only W2
and b2 for the quadratic condition. Reports remain local, outside Git.

The new inference-only layer has three allocation-free CPU shape tests and
six GPU tests. They cover scalar-oracle bit equality, product tie rounding,
signed zeros, partial output tiles, row/context boundaries, reused input
mutation, hooks/backward-state ownership, invalid executors/shapes, and
composition with the existing 152-input dense kernel. It exposes no weights,
and explicitly returns `Unimplemented` for backward.

The initialization experiment was also rerun after adding quadratic mode.
All saved coefficients, per-sentence outcomes, update errors, standardization
moments, feature statistics, and nontiming fit/evaluation fields match its
original run exactly. The regression evidence is
`/tmp/frozen_mlp_basis_regression_0/`.
