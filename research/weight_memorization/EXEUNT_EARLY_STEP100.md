# First matched learned checkpoints: amended Exeunt/Nuveth experiment

This is a CPU-only comparison of the **actual completed step-100 checkpoints**,
not a completed training run or a causal attribution. The replacement corpus
includes both `Exeunt -> Nuveth` and `exeunt -> nuveth`; no case-insensitive
`exeunt` match remains. Original training is unchanged.

The experiment root is
`/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137` (ROOT).
New evidence is under `ROOT/lowercase_amendment` (AMENDMENT). The compared
directories are `ROOT/original/checkpoints/step_100` and
`AMENDMENT/replacement/checkpoints/step_100`, with common initialization
`ROOT/initial/checkpoints/step_0`.

## Checkpoint and execution verification

The original checkpoint was written at 12:56:52 UTC on 2026-09-10; the
replacement checkpoint was written at **17:00:39 UTC**. Their reported
training-evaluation losses are **5.32012** and **5.32869**, respectively. These
are the configured first four sequential evaluation batches (40,960 targets),
not full-corpus averages or word-specific measurements.

An independent audit at 17:11:10--17:11:13 UTC verified the live replacement
trainer's PID/start time/parent/arguments and actual executable against the
frozen request. All 751 frozen input records and the reused-original evidence
were checked twice. Initial, original-100, replacement-100, and replacement-0
each have 100 canonical, finite weight files totaling 205,934,592 bytes. Two
complete hash/stat passes agreed. Initialization and replacement-0 match the
shared initialization; original-100 matches both the completed original
inventory and the reused-original reference. All 100 between-arm tensor hashes
differ. Raw and BF16 diagnostic hashes were cross-checked against this audit.

The replacement's four-hour run remains active. This audit does not claim its
terminal budget, final checkpoint, or native word scores are available.

## What is different at step 100?

The eight largest changed logical embedding rows are all spelling pieces from
the edited words. Norms below are FP64 calculations on the stored FP32
replacement-minus-original coordinates; leading spaces are significant.

| Embedding-row rank | Token ID | Piece | Difference L2 |
| ---: | ---: | --- | ---: |
| 1 | 21733 | ` Nu` | 0.433733 |
| 2 | 3109 | `Ex` | 0.283955 |
| 3 | 45177 | `uve` | 0.272131 |
| 4 | 303 | `ve` | 0.239850 |
| 5 | 2797 | `unt` | 0.204527 |
| 6 | 1475 | ` Ex` | 0.198889 |
| 7 | 68 | `e` | 0.177295 |
| 8 | 400 | `th` | 0.159884 |
| 32 | 45 | `N` | 0.046083 |
| 109 | 14364 | ` nu` | 0.035644 |
| 1350 | 409 | ` ex` | 0.017357 |

The table includes all eleven edited-word IDs, not just the favorable ranks.
The ninth-ranked row overall is ` visits` (11864), L2 0.062981. Pieces also
occur in unrelated words, and ranking does not make them Exeunt-exclusive.

Across the whole model, the between-arm L2 difference is **2.190896**, or
**1.2961%** of the original checkpoint's L2 norm. Training moved the two models
58.595362 and 58.682903 from initialization; those two update vectors have
cosine **0.99930314**. This is substantial common training with a much smaller
between-arm difference, not two independently aligned representations.

The distribution of **squared** difference (not causal importance) is:

| Parameter group | Share of total squared difference |
| --- | ---: |
| Tied token embedding / output dictionary | 86.5727% |
| Attention projections, including biases | 7.3461% |
| MLP projections, including biases | 4.5244% |
| Position embeddings | 1.5510% |
| LayerNorm parameters | 0.0057% |

Although the top individual rows are edited-word pieces, their eleven rows
account for only **13.0530% of embedding squared difference**, or **11.3003%
of whole-model squared difference**. The rest is spread over many coordinates.
The largest non-embedding tensor difference is block 0's Q/K/V projection,
L2 0.344790. Neither its size nor its rank establishes its causal role.

## Separating a common shift from token-specific output differences

A CPU-only follow-up at 17:56 UTC decomposes each logical embedding-row
difference as `delta_E[j] = mean_delta + centered_delta[j]`, with the mean
taken over all 50,257 logical rows. The centered rows sum to zero. Padding is
validated but excluded from the mean, ranks, and energy denominators.

| Measurement | Stored FP32 | BF16 forward operands |
| --- | ---: | ---: |
| Total embedding squared difference | 4.155513110 | 4.217847083 |
| Vocabulary-common component | 1.601907403 | 1.601815924 |
| Common share of squared difference | 38.5490% | 37.9771% |
| Centered, row-specific component | 2.553605706 | 2.616031159 |
| Eleven spelling rows' share of the centered component | 21.2835% | 20.7806% |

The BF16 calculation rounds each original/replacement endpoint separately
before subtracting; it does not round their difference. The squared-energy
identity closes within 1e-15 in both calculations. All eight leading rows in
the earlier table remain the top eight, in the same order, after centering
at either precision. The selected eleven rows nevertheless account for only
about one fifth of the centered difference, not all of it.

This decomposition matters because, for a **fixed final hidden vector** `h`,
the common component adds the identical scalar `dot(h, mean_delta)` to every
logical output logit. In exact arithmetic it changes neither softmax
probabilities nor token-to-token logit margins. Thus raw embedding distance
includes a substantial component that cannot, on its own, alter relative
output scores at fixed `h`.

This is **not** permission to remove that component from the native model or
declare it causally irrelevant. The embedding is tied: its input use can
change `h`. A generic shared vector is not removed by LayerNorm, which
subtracts a scalar mean over hidden coordinates, not an arbitrary vector.
Also, centering rounded BF16 operands is a descriptive calculation, not a
representable checkpoint edit with guaranteed identical native rounding.
The percentages above are coordinate energies, not fractions of the word's
storage or causal importance. No forward pass or intervention was performed.

## Forward precision matters

The model stores FP32 master weights but rounds embeddings and dense matrices
to BF16 before lookup/matrix multiplication. The CPU precision audit uses
those actual operand boundaries, leaving position embeddings, biases, and
LayerNorm parameters FP32. It does not round every parameter indiscriminately.

Of 51,475,968 logical parameter coordinates, 51,475,515 differ in stored FP32;
34,772,782 differ at their forward operand precision. Thus **32.4479% of stored
bit differences disappear at the relevant casts**. The eleven selected rows
retain differences in **5,591 of 5,632** BF16 coordinates (99.2720%). All 7,680
physical vocabulary-padding parameters remain identical. Neither arm contains
stored or operand subnormal values in this comparison.

These are operand differences, not measured activation or probability changes.
The tied matrix receives both sparse input-lookup gradients and dense output
head gradients: `LanguageModelingHeadLayer::bwd` accumulates the vocabulary-wide
gradient into the same embedding accumulator. Consequently, a changed row is
not proof that its token occurred as an input. Likewise, movement magnitude
does not indicate whether a token became more or less probable.

## Exposure context, not reconstruction from weights

The saved native sampler's first 1,000 sequence starts correspond to 100 steps
of ten 1,024-token sequences. Counting complete supervised presentations in
`[start+1, start+1025)` gives:

| Original / replacement form | Corpus occurrences | Complete presentations | Partial windows |
| --- | ---: | ---: | ---: |
| Leading-space `Exeunt` / `Nuveth` | 883 | 546 | 2 |
| Bare `Exeunt` / `Nuveth` | 53 | 49 | 1 |
| Leading-space `exeunt` / `nuveth` | 7 | 3 | 0 |

This is replay of the authenticated saved sampling sequence, not a separately
captured runtime GPU batch trace. Both checkpoints' step completion is verified
separately. The three lowercase presentations are of the revised `nuveth` in
the replacement arm, not residual `exeunt` contamination.

## Interpretation and next discriminating tests

The training intervention leaves a clearly identifiable spelling-related
signature already at step 100. It does **not** yet establish a mechanism for
complete-word prediction. Large changes can be downstream effects of different
training targets and contexts; tied input/output weights are ambiguous.

The already queued native evaluation will measure both complete spellings in
matched contexts, with unchanged continuations as controls. The causal stage
then separates input-row and output-row transfers and tests attention/MLP
transfers in both directions. It will run only after timed training releases
the GPU. Whole-word probabilities, subtoken effects, and collateral changes
are required before drawing a causal conclusion.

## Evidence files and tests

All five files below are new and exclusively created under AMENDMENT:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `early_step_100_provenance.json` | 432840 | `1885c8085ace546faa59c223611ee58f575e08a5b2248867b263cdd443a06119` |
| `early_step_100_delta.json` | 198232 | `d9c2a805dbed1156cbd513fbe6a1424ead353ad315dd47e79102ffbdd47305eb` |
| `early_step_100_compute_delta.json` | 79577 | `6525a65fe666757be46e5c1c4419c6941e2ad75b4fc601867a7016c61f2d28ee` |
| `early_step_100_summary.json` | 7267 | `5172ba81cce867691c96b2fcd9c741ab3a9bca73f9b93a71ac1b374e2c7ee9ab` |
| `early_step_100_embedding_centering.json` | 57377 | `24523a6329f0b11cdd34c6caf045c76726168bd9d33f2f11a5e1abecd8d306df` |

Raw and operand comparisons use existing `paired_weight_diff` and
`paired_compute_weight_diff` modules, respectively. The small summary records
their input hashes, selected-row formulas/ranks, and saved-sampler counts; its
tokenizer pieces retain GPT-2's byte-to-Unicode spelling (`Ġ` means space).
The two diagnostic test suites plus lowercase-input tests pass: **38 CPU
tests**. No model inference, GPU test, checkpoint mutation, or training signal
was performed for this analysis.

The centering report records both embedding-file hashes, and both match the
independent step-100 provenance audit above. Its implementation is
`scripts/weight_analysis/paired_embedding_centering.py`, SHA-256
`9939cec4d62f8349a281992609f68b329426e0c7d012c4b45ec1024c1c630d85`.
The separate 19-test suite uses an independent dense FP64 oracle and scalar
BF16 rounding; it covers common-only and centered-only changes, padding,
rankings, chunk sizes, fixed-hidden-state cancellation, tied-input
counterexamples, source/checkpoint mutation rejection, and exclusive output.
Those tests passed, as did the full **1,380-test CPU suite** (38.638 seconds;
`/tmp/pluto-embedding-centering-cpu-tests.log`). The report numerically inspects
only embeddings; authentication of the other weights and the actual matched
training step comes from the separate full-checkpoint audit, not from these
nineteen tests.
