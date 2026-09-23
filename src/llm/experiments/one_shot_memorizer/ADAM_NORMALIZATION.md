# Why fixed fact gradients fail to add under Adam

## Prospective analytic check

The two-update production replay already separates changed gradients from
optimizer history. The next bounded check explains the latter coordinate by
coordinate, using the existing France-then-Greece gradients. It does not train
another memorized model or search for a repair.

Let `a = gA(W0)` and `b = gB(W0)` denote one coordinate. With zero initial
moments, zero weight decay, and both global clock slots retained, step two's
bias-corrected first moments are

```
x = beta1 * a / (1 + beta1)
y = b / (1 + beta1).
```

The separate and shared bias-corrected RMS denominators are

```
dA = sqrt(beta2 * a*a / (1 + beta2)) + epsilon
dB = sqrt(b*b / (1 + beta2)) + epsilon
dAB = sqrt((beta2 * a*a + b*b) / (1 + beta2)) + epsilon.
```

Step one's change cancels between frozen joint and summed components. Their
ideal real-arithmetic difference after step two is therefore

```
frozen - SUM = rate2 * [x*(1/dA - 1/dAB) + y*(1/dB - 1/dAB)].
```

Evaluate this formula in FP64 using the actual FP32 gradients and FP32
hyperparameters, and compare it with the observed FP32 endpoints. Report the
residual explicitly: the ideal formula is not claimed to reproduce production
FP32 moment arithmetic, bias corrections, weight updates, and the SUM cast
bit-for-bit. Do not fit coefficients or change the optimizer hyperparameters.

Unit tests independently iterate CPU Adam in FP64 for both scheduled slots
and compare with the closed form, including one absent fact, equal/opposite
gradients, zero first-moment momentum, very small gradients, and invalid inputs.
Also check the linear alternative: if both component numerators use the same
fixed denominator, their contributions add. Ordinary SGD and linear momentum
with fixed gradient arrays have this superposition property over the reals;
their gradients would still change during real training. The tested Adam
denominators instead depend on which facts share a history. This is a local
explanation of optimizer interaction, not a one-shot construction of the
learned checkpoint or evidence about exclusive factual ownership.

Record per-tensor and same/opposite-sign coordinate summaries, with the
two signed denominator-interaction terms kept separate. Their norms need not
sum: cancellation is possible. Complete this check within 45 minutes; do not
extend training or tune a new composition formula based on its outcome.

## Result: shared adaptive normalization predicts the early interaction

The fresh production replay reproduced every saved gradient and weight byte,
every original coordinate report, and all 68 controls of the previous run.
An independent C++ verifier checked every one of the 114,256 coordinate rows
against the raw gradients and endpoints and independently iterated FP64 Adam.
Its largest disagreement with the closed form was `1.19e-20`.

| Global quantity | L2 norm |
| --- | ---: |
| Observed FP32 `frozen - SUM` | 0.001848691124 |
| Ideal formula prediction | 0.001848679789 |
| Observed minus predicted vector | 0.000001399551 |

The residual norm is **0.075705%** of the observed norm; this is a relative
approximation error, not a percentage of factual information or causal
ownership. Its largest coordinate is `1.9353e-7`. The ideal formula deliberately
does not emulate all production rounding, including FP32 moment updates,
bias correction, master-weight subtraction, and the final SUM cast.

Of the coordinates, 72,359 have two nonzero same-sign fact gradients, 25,705
have opposite signs, 32 have exactly one zero gradient, and 16,160 have two
zeros. The latter two groups have zero interaction in both the formula and
the measured endpoints. The signed A/B terms reinforce for same-sign
gradients and cancel partly for opposite signs. Across all coordinates their
norms are `0.001373995` and `0.001084349`; their inner product is positive
overall (`1.7697e-7`) but negative within the opposite-sign group
(`-6.9933e-8`). Their separate norms must not be added to obtain a total.

### What the equation adds to the previous result

The two first-moment numerators **do** add. What fails to add is their
normalization: the A-only, B-only, and shared histories divide by different
RMS magnitudes. The effect exists even with first-moment momentum set to
zero, because the squared-gradient history still changes B's denominator.
Conversely, a fixed shared denominator makes linear momentum additive for
fixed gradients over the reals. This separates adaptive scaling from the
looser explanation that any kind of momentum must cause non-additivity.

At initialization, an individual fact's Adam update is therefore not a
context-independent weight vector that can simply be added to other facts'
updates. Its effective scale depends on which gradients share its optimizer
history. The earlier experiment separately measures how updating the model
also changes the next fact's gradient. Neither observation establishes how
to construct the final, fully memorized weights without training, nor which
mechanism dominates after 1,024 or 16,128 steps.

### Reproduction

Build `//src/llm/experiments/one_shot_memorizer:checkpoint_two_step_interaction_probe`
and run the command in [FACT_SUPERPOSITION.md](FACT_SUPERPOSITION.md), using a
fresh output directory. The probe now additionally emits
`adam_normalization_coordinates.tsv` and `adam_normalization_summary.tsv`.
The completed run is `/tmp/fact_adam_normalization_0`; generated reports stay
local. Six CPU tests include 3,042 gradient/configuration pairs checked against
an independently iterated optimizer, absent facts, sign cancellation,
zero first-moment momentum, fixed-scale linearity, and invalid/overflow inputs.
The full repository suite passes all 108 test targets.
