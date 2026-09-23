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
