# Distinguishing embedding interactions from softmax normalization

Status, 2026-09-10: **CPU postprocessor implemented and tested; no real-model
factorial results available yet.** Controlled replacement training still owns
the GPU. This adds no GPU launches and changes none of the frozen training,
native probe, reader, or observer implementations. It is not a claim that a
new postprocessing job has already executed on the paired models.

## Why the probability interaction needs another check

The existing embedding-only intervention uses four complete teacher-forced
models. AA is the recipient; AJ replaces only the output dictionary; JA
replaces only input lookup; JJ replaces both. Recipient transformer weights
and final LayerNorm remain fixed. Its log-probability interaction is
`I(log P) = log P(JJ) - log P(JA) - log P(AJ) + log P(AA)`.

Nonzero `I(log P)` does not by itself demonstrate a cross-effect in the
underlying logits: softmax is nonlinear. A deliberately tiny, **synthetic**
counterexample makes this visible:

| Cell | Target logit | Competitor logit |
| --- | ---: | ---: |
| AA | 0 | 0 |
| AJ | 0 | 1 |
| JA | 0 | 1 |
| JJ | 0 | 2 |

Each logit's factorial interaction is exactly zero, but the target's
log-probability interaction is **-0.1935518166 nats**. This can even arise from
bilinear hidden-state/output-weight combinations with zero output-weight
change dot hidden-state change. It is not a measurement of Exeunt.

## Fixed-rival decomposition

For each prediction, choose the largest **non-target AA logit**, with lower
token ID breaking ties. Hold that competitor `r` fixed across all four cells:

```
M(cell) = logit_target(cell) - logit_r(cell)
Q(cell) = logsumexp(logits(cell) - logit_r(cell))
log P_target(cell) = M(cell) - Q(cell)
```

Each input/output/joint/interaction contrast of log probability is therefore
the corresponding margin contrast minus the normalizer contrast. Both terms
are invariant in exact arithmetic to any vocabulary-common logit offset in
each cell; neither mistakes a change in the winning competitor for a change
against the original competitor. The calculation retains that competitor's
ID and actual target/competitor logits.

In exact-real arithmetic, with the same fixed teacher-forced prefix and the
factorial's unchanged transformer and normalization parameters:

```
I(M) = (delta_E_target - delta_E_r) dot delta_h
```

Here `h` is the vector consumed by the head and `E` is its output operand.
This motivates an interpretation of input/readout compatibility, but is not
an exact identity promised for the saved logits after native BF16 operand
rounding and FP32 accumulation. The postprocessor measures saved native FP32
logits directly; it does not substitute a CPU model or reconstruct hidden
states from weight norms.

**A normalizer contribution is not necessarily just softmax curvature.**
Other vocabulary rows may have genuine cross-effects even when the chosen
target-versus-rival interaction is zero. Therefore the report also measures
the maximum absolute value and L2 norm of
`I(logit_j - logit_r)` over **all 50,257 logical output rows**. The synthetic
example above has zero interaction everywhere. A second test has zero
target-margin interaction but a nonzero interaction on another competitor.

## Full spellings and controls

Per-case reports retain every frozen label and exposure record, every token's
decomposition, their sequence sum, and separate first-piece and suffix sums.
For four-token cases, the exact following native token is separate from the
three-token word. It is not an aggregate over all delimiters.

Exeunt and Nuveth's first-piece predictions can share an identical causal
prefix, but later teacher-forced predictions have different prefixes. Their
softmax normalizers must not be canceled as though hidden states were shared.
Moreover, different targets can choose different fixed rivals: the local
margin and normalizer attributions must not be treated as one common-rival
comparison between candidate words. Only their total log-probability effects
can be combined directly into the existing matched-context word log odds.

The new report does not pool aliases or treat them as independent samples.
The existing reader retains the event-deduplicated probabilities, ranks,
controls and matched-context odds. Neither a target-margin change nor an
improved preference ratio establishes that the word became more probable
without damaging other continuations.

## Implementation and validation

`scripts/weight_analysis/embedding_factorial_mechanism.py` accepts an existing
`embedding_factorial_readout` JSON and a new output path. It requires recorded
execution provenance and reruns the frozen reader in a temporary directory,
recomputing full-vocabulary scores and checking checkpoint bytes, patch
selection, frozen cases, causal invariances and execution records. Every
field must match the original readout before decomposition begins. Files
are rehashed before exclusive publication; output inside a source checkpoint
or the native-score directory is rejected.

The new postprocessor never edits saved inputs or runs inference. Recorded
execution identities are checked, not independently attested; synthetic
fixtures that exercise these checks are explicitly not native measurements.

There are **27 focused CPU tests**: 17 independent arithmetic tests and 10
end-to-end synthetic-evidence tests. They cover fixed-rival ties and switches,
common offsets, pure-softmax and competing-logit counterexamples, numerical
stability, sequence/suffix/following-token sums, full reader revalidation,
edited labels/logits/checkpoints, missing execution records, mutation during
analysis, protected destinations, and exclusive output.

The full CPU suite also passed: **1,407 tests in 39.294 seconds**, recorded in
`/tmp/pluto-factorial-mechanism-cpu-tests.log`. Implementation SHA-256:
`7191a8603fb655c8a3e4e685fca950f1fc47809c55859a144f7236d7bd8fff94`.
No native model effect is inferred from these passing synthetic tests.
