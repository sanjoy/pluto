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
