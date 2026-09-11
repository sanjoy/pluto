# Matched step 200: a spelling signal amid broader training divergence

This compares **completed, independently authenticated step-200 checkpoints**
from the original and lowercase-amended replacement runs. It is a CPU-only
weight analysis, not an intervention or evidence that the new models have
memorized Exeunt. Replacement training continues for its full four-hour budget.

Root (ROOT):
`/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137`.
Amended evidence (AMENDMENT): `ROOT/lowercase_amendment`.
Compared checkpoints: `ROOT/original/checkpoints/step_200` and
`AMENDMENT/replacement/checkpoints/step_200`. Both share
`ROOT/initial/checkpoints/step_0`.

## Completed steps and execution evidence

Original step 200 was written at **14:09:29 UTC** on 2026-09-10; replacement
step 200 was written at **18:13:17 UTC**. Their training-evaluation losses are
**4.97011** and **4.96577**. These losses cover the first four sequential
evaluation batches (40,960 targets), each arm's own corpus targets, not all
training tokens or word-specific predictions.

An independent audit at 18:15:29--18:15:32 UTC checked initial, original-200,
replacement-200, and replacement-0: each contains 100 finite canonical weight
files totaling 205,934,592 bytes. Two complete hash/stat passes agreed. Both
initial checkpoints match; original-200 matches its completed inventory and
the amended run's original reference. All 100 between-arm tensor hashes differ.
All 751 frozen request inputs and the original-reference evidence were
reverified twice. Actual trainer PID/start time/parent/arguments/executable
matched the frozen request. This is not merely inference from checkpoint names.

The raw, precision-aware, and embedding-centering diagnostics all have weight
hashes matching this audit. No checkpoint or training process was modified.

## What changed between the two matched observations?

All norms refer to replacement minus original; proportions are **squared
coordinate differences, not causal importance or word-storage fractions**.

| Measurement | Step 100 | Step 200 |
| --- | ---: | ---: |
| Whole-model difference L2 | 2.190896 | 6.940875 |
| Difference / original weight L2 | 1.2961% | 3.6809% |
| Cosine between updates from shared initialization | 0.99930314 | 0.99773206 |
| Embedding share of whole-model squared difference | 86.5727% | 81.9331% |
| Eleven spelling rows' share of embedding squared difference | 13.0530% | 3.1017% |
| Eleven spelling rows' share of whole-model squared difference | 11.3003% | 2.5413% |
| Common-row component of embedding squared difference, FP32 | 38.5490% | 39.1689% |
| Eleven rows' share after centering the embedding differences, FP32 | 21.2835% | 5.1298% |

The spreading difference is **not explained away by the vocabulary-common
shift**: its fractional size barely changes, and the selected rows' share
falls sharply even after centering. At BF16 operand precision the step-200
common share is 39.0579% and selected centered share is 5.1063%, so this
observation is not just sub-BF16 storage noise.

Centering here is descriptive. A shared output-row shift cancels softmax at
fixed hidden state, but the same tied matrix also supplies input embeddings;
it is not generally irrelevant to the whole model. See the
[step-100 explanation](EXEUNT_EARLY_STEP100.md).

## Spelling pieces remain prominent, but their norms are not the mechanism

The top seven rows remain spelling pieces at both FP32 and BF16 operand
precision, with the same ranks after centering. Unlike step 100, the next
two rows are unrelated pieces: ` Pro` (1041) and ` gro` (7128).

| Piece | ID | Raw rank at 100 | Raw rank at 200 | L2 at 200 | Centered rank at 200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ` Nu` | 21733 | 1 | 1 | 0.605134 | 1 |
| `Ex` | 3109 | 2 | 2 | 0.493696 | 2 |
| `uve` | 45177 | 3 | 3 | 0.465511 | 3 |
| `ve` | 303 | 4 | 4 | 0.361422 | 4 |
| `e` | 68 | 7 | 5 | 0.294442 | 5 |
| `th` | 400 | 8 | 6 | 0.287373 | 6 |
| ` Ex` | 1475 | 6 | 7 | 0.217657 | 7 |
| `unt` | 2797 | 5 | 10 | 0.196987 | 10 |
| `N` | 45 | 32 | 703 | 0.064677 | 512 |
| ` nu` | 14364 | 109 | 444 | 0.073136 | 369 |
| ` ex` | 409 | 1350 | 2112 | 0.045286 | 1484 |

All eleven selected rows are shown, including low-ranked ones. In particular,
`unt`'s between-arm L2 slightly **decreases** from 0.204527 to 0.196987 while
other differences grow. That is not evidence that its probability decreased,
that the model forgot the suffix, or that the row ceased to matter. Neither
the direction nor the functional consequence follows from a difference norm.

The step-200 whole-model shares outside embeddings are attention **9.5156%**,
MLP **7.9413%**, positions **0.6041%**, and LayerNorm **0.0059%**. The largest
non-embedding tensor is now block 6's Q/K/V projection (L2 **1.020977**), not
block 0's. Block 6 also has the largest complete-block difference (L2
**1.390682**), followed by block 7 (**1.176204**) and block 0 (**1.175149**).
These rankings motivate looking beyond the initial layers; they do not locate
the word in block 6. The fixed causal screen already includes every block in
both transfer directions, rather than choosing only the largest difference.

## Precision and exposure checks

At step 200, 51,475,888 logical stored FP32 coordinates differ; 46,481,945
remain different at their actual forward operand precision. **9.7015%** of
stored bit changes disappear at those casts, versus **32.4479%** at step 100.
The precision audit rounds token embeddings and dense matrices, not the FP32
position/bias/LayerNorm operands. It measures no activations or probabilities.

The first 2,000 frozen sampler windows contain the following complete
supervised word presentations. Both arms have now independently reached the
corresponding 200 steps. Counts remain a replay of the authenticated native
sampler prefix, not a captured GPU batch trace.

| Form | Complete presentations | Partial windows | Distinct occurrences presented completely |
| --- | ---: | ---: | ---: |
| Leading-space `Exeunt` / `Nuveth` | 1,113 | 2 | 637 / 883 |
| Bare `Exeunt` / `Nuveth` | 77 | 1 | 42 / 53 |
| Leading-space `exeunt` / `nuveth` | 8 | 0 | 6 / 7 |

Steps 101--200 add 567, 28, and 5 complete presentations respectively. An
independent interval-count implementation agrees with the existing exposure
helper. No lowercase `exeunt` remains in the actual replacement corpus.

## Interpretation and remaining work

There is a stable spelling-related weight signature, while the intervention's
downstream training differences spread broadly. A naive search for the word
in the largest weight deltas would miss most of those differences and cannot
tell which are functional causes, collateral effects, or compensating changes.

The pending native tests must establish absolute probabilities for both full
three-token spellings and controls. Embedding factorials separate input from
output use; branch transfers test upstream contributions. Their effects need
not be additive, and probability interactions can also arise through softmax
normalization. The new [factorial postprocessor](EXEUNT_FACTORIAL_LOGIT_PROTOCOL.md)
is ready to distinguish these possibilities when native evidence exists.
Neither this checkpoint comparison nor a passed test suite completes the goal.

## Evidence files

All files are new, exclusively created under AMENDMENT:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `early_step_200_provenance.json` | 433607 | `3ae4dc00c38f9d089a8154f04bc2d42d6c7799d467bb22d91655e7035536603d` |
| `early_step_200_delta.json` | 180345 | `db5b457b64bbf5c1035739eed8ebafb447e4b1feae59e64240bfad757b8f19df` |
| `early_step_200_compute_delta.json` | 79492 | `756ce8668f1a70ffedc72797390351a883bfe119b2d8f6ead31dd53fac5aae6b` |
| `early_step_200_embedding_centering.json` | 57261 | `c26f3a0ceb1fb036e1701c88a8ad0d5e6208c8233df203edf6707769204cc283` |
| `early_step_200_summary.json` | 28515 | `d27336c917e777b280b7d45b86aff125451991b144fa0bb71f264874bee40b97` |

The summary records the full hashes of its step-100/200 diagnostic inputs,
formulas, source helpers, tokenizer, amendment manifest and native sampler
prefix. No new native code, GPU work, or intervention was used here.
An independent second calculation rehashed all summary evidence and all 200
step-200 weight records, recomputed all 100 tensor differences directly, and
confirmed the category/selected-row fractions, centered ranks and exposure
counts without discrepancies.
The four relevant CPU suites (raw differences, operand precision, embedding
centering and sampling exposure) passed again: **53 tests**. The existing
complete 1,407-test CPU suite had already passed before this data-only analysis.
