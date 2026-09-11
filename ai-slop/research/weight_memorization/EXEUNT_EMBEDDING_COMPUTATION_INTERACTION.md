# Selected embedding rows and the remaining computation jointly transfer spelling

**Completed assay (21:45:54 UTC):** all six suites in both directions
completed, and the supervised process exited with status zero. Independent
summaries and saved-logit decompositions exist for all six. Both main transfer
directions restore donor-like conditional spelling when E and C are combined.
Subsequent sections retain the order in which partial results arrived.

## First completed endpoint result — 2026-09-10 21:28 UTC

This is a partial causal result from the lowercase-amended, deterministic
Exeunt/Nuveth pair at step 331. Both training runs are complete, but the
mechanistic goal is not. The first direction below transfers Nuveth-trained
weights into the Exeunt-trained model. The reverse direction and supplemental
controls are still running at this report's first publication.

**Main finding:** neither selected token rows nor all non-token-embedding
weights alone transfer the donor's reliable Nuveth suffix completion. Their
combination does. The other vocabulary rows can stay at the recipient's
values in this tested background without losing the donor-like completion.
This establishes joint functional sufficiency for the measured behavior,
not a unique storage location, a minimal circuit, or word-selective editing.

## Exactly what was changed

All interventions are weight replacements without retraining. The selected
11 rows are the previously declared native pieces for both spellings and
their leading/bare title and lowercase variants.

| Model | Selected token rows | Other token rows | Non-token-embedding weights |
| --- | --- | --- | --- |
| A: recipient | Exeunt | Exeunt | Exeunt |
| E: row swap | Nuveth | Exeunt | Exeunt |
| C: complement swap | Exeunt | Exeunt | Nuveth |
| EC: combined | Nuveth | Exeunt | Nuveth |
| D: complete donor | Nuveth | Nuveth | Nuveth |

E changes the rows in both input and tied-output roles. C includes all 96
transformer tensors, the learned-position tensor, and two final-LayerNorm
tensors. Calling C merely “the transformer” would obscure those last three
files. This experiment cannot localize the effect within C.

C and EC are independently materialized checkpoints. The existing unmodified
native embedding-factorial probe compares C with EC. Its AA/JJ cells give
outer C/EC; the old recipient factorial supplies A/E and the reciprocal
factorial's AA supplies D. Inner input/output factors are not confused with
the outer E/C factors.

## Completion is restored, not just relative preference

The following are geometric means over the **15 held-out, leading-space,
title-case, original-prefix contexts**. Each context has 128 preceding native
tokens with local positions reset to zero. The suffix is the last two native
pieces conditional on supplying the first piece. Scores are teacher-forced
at temperature 1; they are not free-generation success rates.

| Model | Nuveth first-piece P | Nuveth conditional suffix P | Nuveth three-piece P | Exeunt three-piece P |
| --- | ---: | ---: | ---: | ---: |
| A | 1.05195e-8 | 1.09078e-8 | 1.14745e-16 | .00159732 |
| E | .00139855 | 5.54311e-6 | 7.75234e-9 | 3.50228e-8 |
| C | 2.16953e-8 | 7.53272e-7 | 1.63425e-14 | 5.18479e-9 |
| EC | .00227873 | .888170 | .00202390 | 1.05114e-12 |
| D | .00213764 | .883236 | .00188804 | 9.42113e-13 |

Thus EC restores an approximately **88.8% conditional suffix probability**,
versus 88.3% for the complete donor; E alone gives only **0.000554%** and C
alone **0.0000753%**. EC's absolute full-word probability is also donor-like,
not merely a preference reversal caused by suppressing Exeunt. The first
piece itself remains unlikely: this is strong *conditional spelling*, not
full contextual memorization or guaranteed generation.

Replacing the remaining token rows after EC (EC→D) changes mean Nuveth suffix
log probability by -0.005571 nats and full-word log probability by -0.069484.
The rows therefore do have effects; the result is that their exact donor
values are not needed for donor-like suffix completion in this background.
It does not establish their irrelevance in A or other contexts.

## The joint effect is not just softmax curvature

The independently recomputed outer interaction is
`log P(EC) - log P(E) - log P(C) + log P(A)`.
For Nuveth's suffix its mean is **+7.74942 nats**. With the highest A
non-target logit fixed as the rival separately at each prediction, this is
**+14.39662 target-minus-rival margin interaction**, minus **+6.64720 relative
normalizer interaction**. For all three pieces the corresponding quantities
are +7.51373 = +14.16243 - 6.64870 nats.

The joint gain therefore includes nonadditive target-relative logits; it
cannot be explained solely by applying softmax to additive logit changes.
This is evidence of functional interaction under these interventions, not a
percentage of “where the word is stored.” Each suffix prediction conditions
on its own supplied candidate history and has its own fixed rival.

## Input and output roles with donor computation installed

The inner C→EC factorial supplies another distinct comparison. C is fixed
here, including donor positions, transformer blocks, and final LayerNorm;
only the selected rows' input/output roles vary. In the same 15 held-out
leading-title contexts, Nuveth suffix probabilities are:

| Selected rows using donor values | Conditional suffix P |
| --- | ---: |
| Neither (C) | 7.53272e-7 |
| Output only | 3.03471e-6 |
| Input only | .00195723 |
| Both (EC) | .888170 |

Output-only transfer changes first-piece probability from 2.16953e-8 to
.00227803, but does not deliver strong suffix completion. Input-only transfer
substantially helps the suffix while leaving first-piece probability near
2.16e-8. Both are needed to recover the donor-like behavior under this tested
factorial. These are conditional intervention results, not proof that the
same components are universally necessary or uniquely lexical.

## Controls and remaining limits

Before measurement, an independent complete donor copy reproduced **every
logical-vocabulary logit byte** in all four control cells against the old
reciprocal donor baseline. The C/EC native probe checks its exact production
diagonals, and the CPU readout revalidates actual checkpoint bytes and all
three saved reports before publishing the decomposition.

The 16 generic three-token controls in each split show mean summed NLL
changes EC−A of **-0.03632 training / +0.002841 test**. The full donor's changes
are -0.02097 / +0.005975. These small means do not establish word selectivity:
shared-subword and exact-following-token controls remain necessary and are
being measured. Earlier row-only assays already demonstrated collateral
effects on reused pieces.

No inference is made here about unreported bare-title/lowercase strata, the
reverse direction, arbitrary positions, or other corpora. The older partial
step-100 single-branch screen is a different assay and cannot localize this
endpoint interaction. The next causal step is conditional localization of C:
test grouped additions to E and reversals in EC, then narrow attention/MLP
components while retaining both-word and collateral scores.

### First direction's supplemental controls completed — 21:34 UTC

Both additional suites now completed, including all four-cell donor-copy
logit parity checks. Independent CPU decompositions regenerated the
following-token and shared-piece native reports exactly.

For the same 15 leading-title test contexts, the geometric probability of
the **one exact native token after Nuveth** is .00509589 in EC versus .00529951
in D (NLL 5.27932 versus 5.24014). After supplied Exeunt it is .000226762
versus .000224732. This extends the similarity by one measured token; it is
not probability of any boundary or arbitrary continuation.

The shared-piece controls expose much larger failures of general donor
equivalence. The table reports mean **first-target NLL** differences; positive
means the hybrid is worse. Cases are unrelated occurrences of the named
piece, not the replacement words.

| Split / piece | n | EC−A | D−A | EC−D |
| --- | ---: | ---: | ---: | ---: |
| Training `ve` (303) | 8 | +1.67919 | -.815246 | +2.49443 |
| Test `ve` (303) | 8 | +1.83461 | -.297675 | +2.13228 |
| Training `th` (400) | 8 | +2.00755 | -.580355 | +2.58791 |
| Test `th` (400) | 8 | +1.94980 | -2.45003 | +4.39982 |
| Training `unt` (2797) | 8 | +5.18400 | +5.11775 | +.066247 |

EC and D have identical selected rows and C tensors, so their discrepancy
here is caused by which **remaining embedding rows** they use, in this
background. Those rows can affect preceding input context as well as output
competition; these controls do not separate the two paths. Their small effect
on the measured Nuveth suffix must not be generalized to other contexts.
Missing shared-piece coverage, including no test `unt`, is retained.

One concrete test control (supplemental source case 239) ends its frozen
prefix with “so do not I.\n    Pro” and scores `ve true,` (IDs 303, 2081, 11),
completing **“Prove true,”**. EC assigns `ve` probability .000110193 versus
D's .159034: a +7.27464-nat D−EC log-probability difference. The preceding
` Pro` is token 1041, outside the selected rows, but this descriptive
observation does **not** identify that row as the mediator. Testing its row
or input path would be a separate causal intervention. Frozen prefix IDs,
text bytes, offsets, and the local tokenizer were independently checked.

Alias-aware main aggregation independently confirms all 29 leading-title
contexts switch preference under EC (14 training, 15 test). Lowercase remains
a different regime: C alone flips seven preferences while leaving nuveth's
geometric full-word probability only 2.63e-11; EC reaches 9.08e-9 versus D
8.16e-9. Even donor-like lowercase scores are not strong absolute completion.
The exact stratum values and raw-versus-unique event counts are in
`outer_replacement_to_original_main_summary_20260910.json` (554,749 bytes,
SHA-256 `d335a8c31084247fb4dcd884d7af95812962eac726a2b0ea8c44653827b9854a`).
The summary authenticated 1,769 source records and reconstructed the outer
cells exactly; it did not independently rerun the GPU.

## Evidence

### Reverse endpoint: Exeunt restoration — 21:38 UTC

The reverse main suite completed at 21:38:57 UTC, including donor-copy logit
parity. Here A is the Nuveth-trained recipient, D is the Exeunt-trained donor,
and E/C/EC use original-trained rows/computation as specified by the same
factor definitions. Do not identify these cells with the first direction's
models.

For the same 15 held-out leading-title original-prefix contexts:

| Model | Exeunt first-piece P | Exeunt conditional suffix P | Exeunt three-piece P | Nuveth three-piece P |
| --- | ---: | ---: | ---: | ---: |
| A | 5.70717e-5 | 1.65075e-8 | 9.42113e-13 | .00188804 |
| E | .00141051 | 3.28870e-6 | 4.63875e-9 | 1.46505e-14 |
| C | 8.40739e-5 | .000418043 | 3.51465e-8 | 7.71468e-9 |
| EC | .00203652 | .758300 | .00154429 | 1.02404e-16 |
| D | .00214403 | .745008 | .00159732 | 1.14745e-16 |

The complete Exeunt donor has **74.50% conditional suffix probability**;
EC recovers **75.83%**, compared with **0.000329%** for E and **0.0418%** for C.
Whole-word probability is also close to D. Thus both directions show genuine
absolute completion restoration under the joint intervention, not merely
relative preference reversal. Their quantitative responses are asymmetric.

Reverse Exeunt suffix interaction is +2.20882 nats, decomposed into +7.19513
fixed-rival margin interaction minus +4.98631 relative-normalizer interaction.
The independently regenerated main decomposition is
`outer_original_to_replacement_main_mechanism_20260910.json`, 3,912,152 bytes,
SHA-256 `093761e7e5297a80d6666e13d79f71548709869923f6a27287adcfa36ccc5847`.

### Artifact locations

The reverse exact-next-token suite also agrees closely on word cases: Exeunt's
held-out leading-title next-token P is .00335564 in EC versus .00348228 in D.
But reverse shared-piece controls again reject global donor equivalence:
the `Pro`→`ve` test case has EC P 3.87967e-5 versus original donor .00177827.
For unrelated training `unt` (eight cases), EC geometric first-token P is
.00415985 versus donor .122759. These are separate conditional events, not
the Exeunt suffix. All shared-piece coverage limitations still apply.

Amendment root:
`/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137/lowercase_amendment`.

- New native run: `outer_factorial_20260910_2123/replacement_to_original/`.
- Main conditional execution: `outer_main/execution.json`, PID 1919024,
  recorded successful completion 21:27:38 UTC; SHA-256
  `c7949b58d8c27fd5f4c7d3552d3621031017c841b755a1032b6d0eab25c92738`.
- Main donor control: `donor_control_main/execution.json`, PID 1918163,
  recorded successful completion 21:25:40 UTC; SHA-256
  `3d6fee3d9cb38765002f72ca3e7ec3c45d1f87d2aa2c3ae23ca00fef9c279ecd`.
- Independently regenerated five-cell logits/decomposition:
  `outer_replacement_to_original_main_mechanism_20260910.json`, 3,913,338 bytes,
  SHA-256 `fef13b18a8e47662ae17bfa341f8129d32d30836484d5ff2d4f7bed05b818ba7`.

Both directions' remaining artifacts follow
`outer_{replacement_to_original,original_to_replacement}_{main,word_next_native,shared_piece}_{summary,mechanism}_20260910.json`
at the amendment root. The complete native run is
`outer_factorial_20260910_2123/summary.json`. Its 729 artifact records cover
the exact output inventory; it also binds 1,446 frozen input records. Actual
terminal service state was `SubState=exited`, `MainPID=0`, `ExecMainPID=1917850`,
`ExecMainCode=1` (normal exit), `ExecMainStatus=0`, with exit timestamp
21:45:54 UTC. The old interrupted branch screen remains incomplete and is
not included in this completion claim. The final combined tests for the new
outer runner/readout/summary and runtime/recovery helpers passed **88 tests**.
The subsequent full CPU analysis suite, including the new conditional
localization helpers, passed **1,534 tests in 46.085 seconds**. Synthetic test
logs mentioning training/probe completion are fixtures, not additional native
experiments.

The final read-only completion audit is
`outer_factorial_completion_audit_20260910.json` (611,764 bytes; SHA-256
`c3e8d21c6cc2a55cd083be0cd1c06061f46f7427e49cc5bf3743a429bd396f80`). It
verified 2,160 distinct records before/after, all 12 successful native
executions, the exact output inventory, actual service exit, and an idle GPU.
The completed run summary is 608,994 bytes, SHA-256
`a6a5c786e8b260dec082b8a2c598f803ace17d4a5402189a085dde1c4721cb17`.
An earlier audit attempt stopped at the GPU-visibility guard because its
CPU-only command hid CUDA devices; it published no artifact and changed no
native result. The final audit repeated verification with the recorded GPU
visible to the read-only identity check.

## Next localization is prepared, not executed

`complement_localization_20260910_2150/plan.json` at the amendment root freezes
the Q/early/late cube: Q has three position/final-LN tensors, early has all 48
tensors in blocks 0–3, late all 48 in blocks 4–7. E is fixed in both tied
roles. Empty, single-group, pair-group, and full vertices supply additions
to E and reversals from EC for both step-331 transfer directions.

The 824,239-byte plan has SHA-256
`2e6d1a0b3a67f5f015dc64d5af357746ba74ffe42f66a9e51f72fc04c08762b6`
and binds 2,161 frozen inputs. Its helper has 39 CPU tests and passed the
actual completed-upstream/idle-GPU preparation gates. It explicitly records
`native_execution_performed=false` and `runner_implemented=false`.
The owning runner still needs implementation and must certify all-row native
loss/argmax parity against preserved E/EC checkpoints, separately from the
predeclared tolerance for FP32 loss versus FP64 scoring of saved logits.
There is no new attention/MLP localization result yet.

The new decomposition has 20 CPU tests; the local new-reader plus existing
mechanism test run passed 30 tests. These tests validate analysis logic, not
new GPU results. The separate native execution ledgers provide the recorded
GPU evidence; CPU revalidation is not independent process attestation.
