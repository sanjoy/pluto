# Where the quadratic construction fails

This diagnostic follows the fixed construction in
[QUADRATIC_MLP_BASIS.md](QUADRATIC_MLP_BASIS.md). It does not refit its
coefficients, select a better ridge, or change the feature basis.

## Protocol

Select **every first greedy mismatch** of the jointly replaced model. The
first five tokens were given; every later prefix token was predicted correctly
before this mismatch. The saved prefix therefore equals the corpus prefix,
but neither the wrong token nor any subsequent gold token enters inference.
This study examines one next-token decision per failed sentence, not its
subsequent off-corpus continuation. It is a failure-selected diagnostic, not a
new held-out accuracy estimate.

Load the saved FP32 coefficients, run the same BF16 quadratic-feature and
projection layers, and capture both the original and jointly replaced models.
Validate the saved first wrong token against a fresh batch-one execution.
Capture all causal-prefix positions, not just the final query row.

The original MLP still executes before its output is replaced. This provides
three directly observed updates for each block and position:

* `a = learned(x_original)`;
* `b = learned(x_joint)`;
* `c = quadratic(x_joint)`.

The total branch-output change decomposes exactly as
`c-a = (c-b) + (b-a)`. Here `c-b` is the local function approximation at the
actually altered input, and `b-a` is the effect of the inherited input change
through the original MLP. Report all coordinates, both norms, their dot
product, and the total norm. These are **not additive attributions of the
final logit**: residual addition, later normalization, attention, MLPs, and
rounding still act downstream. In particular, norms can cancel and cannot be
interpreted as percentages of responsibility.

For each block, independently rerun these live interventions:

| Condition | Operation | Question |
| --- | --- | --- |
| `single` | Replace only this MLP. | Is this approximation sufficient to break this particular decision on the original upstream state? |
| `prefix` | Replace MLPs 0 through this block. | How does the decision change as approximations accumulate? |
| `restore_function` | Replace every MLP except this one. | Does restoring this learned function rescue the joint failure? |
| `reset_state` | Joint replacement, then restore the original activation after this entire transformer block at every causal-prefix position. | Does the remaining approximate suffix work when started from the original boundary state? |
| `identity_reset` | Original model with the same original boundary activation restored. | Is the patch genuinely an identity operation? |

State resets include the residual stream and do not restore weights. They are
diagnostic access to original-model activations, **not** a proposed inference
algorithm or a label-free way to repair a model. No repair is fitted or
selected for deployment. Restoring one function need not help monotonically;
one intervention can rescue multiple errors or introduce a new one.

## Required controls and artifacts

The standalone `checkpoint_quadratic_trace_probe` checks:

* all original predictions equal the expected target, and all joint predictions
  equal the saved first mismatch;
* all 4,475 logits match bitwise for capture versus plain execution and empty
  substitution versus the ordinary model;
* all identity-reset and ordinary post-pass logits match their original;
* resetting the final block reproduces every original logit, and the full
  replacement prefix reproduces every joint logit;
* every original master tensor (including the tied alias) and every quadratic
  coefficient is unchanged after the experiment.

The output directory contains input/target/prediction text, intervention
scores against **every vocabulary rival**, raw FP32 logits for every reported
condition, per-row decomposition norms, all 16 coordinates of the three
updates, and a completion manifest. Generated artifacts remain local.

The separate `checkpoint_quadratic_precision_probe` addresses a different
question: whether removing BF16 rounding of polynomial **features** removes
the fitting residual. It fits both feature variants on the identical 819
sentences and scores the same 205 regression-held sentences. Its FP64 dot
products are not a simulation of GPU MMA, and its targets remain captured
BF16 learned updates. It does not change the deployed quadratic construction.
