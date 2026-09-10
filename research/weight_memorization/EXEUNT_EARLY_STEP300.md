# Matched step 300: spelling-row changes grow while their share shrinks

This is a CPU comparison of completed original and lowercase-amended
replacement step-300 checkpoints. It is **not** a causal localization or a
measurement of either word's probability. Replacement training continues to
its full four-hour limit; no GPU analysis was launched alongside it.

ROOT is `/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137`;
AMENDMENT is `ROOT/lowercase_amendment`. Compared inputs are
`ROOT/original/checkpoints/step_300` and
`AMENDMENT/replacement/checkpoints/step_300`, with shared initialization
`ROOT/initial/checkpoints/step_0`.

## Checkpoint and execution evidence

Original step 300 was saved at **15:22:07 UTC**; replacement step 300 at
**19:25:55 UTC** on 2026-09-10. Their training-evaluation losses are
**4.66668** and **4.67385**, respectively. These evaluate the first four
sequential batches (40,960 targets), each arm's own targets, not the full
training corpus or word-specific behavior.

An independent audit at **19:28:32--19:28:35 UTC** validated shared initial,
original-300, replacement-300, and replacement-0: each contains 100 finite
canonical tensors totaling 205,934,592 bytes. Two full hash/stat passes
agreed. All 751 frozen request records were verified twice. Original-300
matches the completed inventory and original reference; both initial copies
match. Actual trainer PID/start time/parent, arguments, and executable match
the frozen request. All 100 between-arm tensor hashes differ.

The raw, precision-aware and embedding-centering reports have exactly the
same checkpoint hashes as that audit. A separate calculation independently
recomputed all 100 tensor differences, selected-row energies, ranks, and
sampling exposures without discrepancies. Padding bytes are unchanged
between arms; this does **not** mean the padded weights are zero-valued.

## What the third matched observation adds

Shares below are squared coordinate differences, **not causal importance**.

| Measurement | Step 100 | Step 200 | Step 300 |
| --- | ---: | ---: | ---: |
| Whole-model difference L2 | 2.190896 | 6.940875 | 11.729665 |
| Embedding share of whole-model squared difference | 86.5727% | 81.9331% | 78.1295% |
| Eleven spelling rows' share of embedding difference | 13.0530% | 3.1017% | 2.1916% |
| Eleven rows' share of whole-model difference | 11.3003% | 2.5413% | 1.7123% |
| Common-row share of embedding difference, FP32 | 38.5490% | 39.1689% | 35.3336% |
| Eleven rows' share after centering, FP32 | 21.2835% | 5.1298% | 3.4100% |

The selected rows' absolute squared difference **increases** from 1.224289
at step 200 to **2.355839** at step 300, while whole-model squared difference
grows to **137.585037**. Thus their shrinking share is not a disappearance of
their weight signature. It also persists after centering and at BF16 operand
precision: selected centered share is **3.4008%** in BF16.

The largest seven raw embedding-row differences are now ` Nu` (21733), `Ex`
(3109), `uve` (45177), `ve` (303), `th` (400), `e` (68), and `unt` (2797), in
that order. They retain those ranks after centering and BF16 rounding.
Leading-space ` Ex` (1475) is raw rank **12**, `N` (45) rank **1007**,
` nu` (14364) rank **1476**, and ` ex` (409) rank **8831**. All eleven remain
in the preselected causal intervention, regardless of their ranks.

Outside embeddings, attention accounts for **10.7826%**, MLPs **10.7536%**,
positions **0.3281%**, and norms **0.00613%** of squared difference. Block 6
still has the largest complete-block difference (L2 **2.528495**), and its
Q/K/V projection is the largest non-embedding tensor (L2 **1.769269**).
This motivates testing all blocks, as already queued; it does not identify
block 6 as the word's storage location.

The update vectors from shared initialization still have cosine
**0.99718355**. Only **5.8278%** of stored bit changes disappear at the actual
operand casts, versus 9.7015% at step 200 and 32.4479% at step 100. The
between-arm difference is not merely below-BF16 storage noise. None of these
quantities measure model behavior.

## Exposure through the first 300 steps

These are counts in the first 3,000 authenticated saved native sampler
windows. The supervised interval is `[start+1, start+1025)`. They are not a
new runtime GPU trace, and no sampler executable was rerun for this report.

| Form | Complete presentations | Partial windows | Distinct occurrences presented completely |
| --- | ---: | ---: | ---: |
| Leading-space title-case pair | 1,663 | 5 | 743 / 883 |
| Bare title-case pair | 109 | 1 | 46 / 53 |
| Leading-space lowercase pair | 10 | 0 | 6 / 7 |

Steps 201--300 add 550, 32, and 2 complete presentations respectively, and
106, 4, and 0 newly covered occurrences. Independent interval/bisect counting
agrees with the existing exposure helper, including all per-piece totals.

## Next evidentiary step

The queued native likelihoods must establish what these models actually
predict. Embedding factorials and reciprocal branch transfers then test which
changes cause that behavior. Weight norms cannot decide whether a change
selects a word, completes its suffix, damages unrelated text, or compensates
for other changes. See the [readout guide](EXEUNT_CAUSAL_READOUT_GUIDE.md) for
case coverage, precision, sign conventions, and follow-up control limits.

## Saved evidence

New files under AMENDMENT; all were exclusively created:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `early_step_300_provenance.json` | 433837 | `6b9d551ff3c86f78f05c88d348c2b76094356014dd35607879645bed77a91213` |
| `early_step_300_delta.json` | 180103 | `dcf09b07f8135d43003e2c5a535bf85a19fc96d5fa17c44fc91081a431d18a0c` |
| `early_step_300_compute_delta.json` | 79442 | `a1554d7ffb4fa9d25f4d15c17fc72dd522eb11864de499479a280b7976aefb0a` |
| `early_step_300_embedding_centering.json` | 57141 | `99ac2cb7aed271471bad9dd92f963c7cdb4d515333c31a80231ce9a55a71603c` |
| `early_step_300_summary.json` | 15500 | `2a76afd9fd91d654fa41c90d74a873f529e1ed1604e6159a148f3a9460c17064` |

The summary binds source reports, prior summary, helper implementations,
tokenizer, amendment, request, manifest, and sampler prefix by hash, and
records the aggregation formulas and exposure definition. Earlier reports:
[step 100](EXEUNT_EARLY_STEP100.md), [step 200](EXEUNT_EARLY_STEP200.md).
The four relevant CPU suites passed again: **53 tests** covering raw deltas,
operand precision, embedding centering and sampling exposures. No native
implementation or frozen observer was changed.
