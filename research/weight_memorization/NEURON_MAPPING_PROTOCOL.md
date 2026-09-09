# Fine-grained MLP reliance map: frozen protocol

Date: 2026-09-09. This experiment uses known Shakespeare tokens and actual
model forwards. It is **causal validation, not an analytical text decoder**.
The previous broad interventions motivate inspecting zero-based blocks 6 and
7; this is a data-assisted choice. No on-disk checkpoint may be modified.

## Fixed inputs and interventions

Use checkpoint `step_13030`, BF16 activations, and the same 32 native-token
windows as `CAUSAL_VALIDATION_PROTOCOL.md`: 16 from the current corpus prefix,
16 from its suffix, 1,025 tokens each, four sequences per execution microbatch.
Revalidate native IDs against every corpus byte and hash all weights, input
files, analysis sources, protocol, and executable before and after the run.
The current split is not authenticated historical training membership.

One NumPy PCG64 generator with seed 20260909 draws a permutation of all 2,048
MLP features for block 6, then a separate permutation for block 7. Split each
permutation into 32 consecutive groups of 64. The group file preserves draw
order; IDs within a group identify rows, not a ranked importance list.

Each intervention scales those 64 rows of the FP32 MLP output matrix by 0.5
from a pristine snapshot. It does not alter any other row, the output bias,
the input projection, or the residual skip. This scales each selected GELU
feature's contribution to the MLP output. Each group contains 32,768 weights,
131,072 bytes, or 3.125% of that MLP's output matrix.

The row matrix is `weight_84.bin` in block 6 and `weight_96.bin` in block 7,
shape [2048,512], little-endian FP32, row-major. Feature j addresses the byte
interval [2048*j,2048*(j+1)). This is an intervention address, not a claim that
the addressed bytes independently encode a passage.

Arm order: clean_before, clean_repeat; block 6 whole-MLP half control then its
32 row-group half interventions; block 7 whole-MLP half control then its 32
row-group interventions; clean_after. The 69 arms retain all per-token losses
and logical-vocabulary argmax IDs. Whole-MLP controls scale both the complete
output matrix and bias (84/85, 96/97), matching the earlier experiment; their
effects are not an additive sum of the row-group effects.

Restore and verify the whole affected matrix's device bytes after every group.
Restore both tensors for each whole-branch control. All three clean runs must
have identical argmax and loss files; otherwise report a failed integrity gate
and do not interpret a fine-grained map. No backward pass or optimizer exists
in this driver. Write completion metadata only after every arm succeeds.

## Discovery and within-passage confirmation

Use loss indices [512,768) for discovery and [768,1024) for confirmation. The
corresponding target positions are shifted by one from the window inputs.
These are disjoint target tokens, **not independent documents**: confirmation
retains causal context from the earlier tokens. The initial 512 positions are
warm-up and retained for secondary descriptive measurements only.

Compute paired mean NLL changes from clean_before for each group and passage.
The 64-by-16 prefix matrix is Delta; suffix effects are collateral measurements,
not inputs to selection. A simple target-minus-other-15 contrast alone is not
enough: Delta[g,p]=a[g]*b[p] can masquerade as passage-specific dependence.

The primary selection rule therefore removes a shared response pattern:

1. Double-center the discovery matrix: Z = Delta - row_mean - column_mean +
   grand_mean. This removes additive group strength and passage difficulty.
2. Fit the leading left singular vector u of discovery Z. Remove its group
   direction: R_discovery = Z - u*(u^T*Z). This also removes the rank-one
   multiplicative counterexample. Report the fraction of energy removed.
3. For each prefix passage choose the group with the largest discovery
   residual, restricting to positive raw discovery loss increases and positive
   residuals above `64*eps64*max(shape)*||Delta||_F`. Ties use ascending block,
   then group ID. Abstain if no group qualifies. If the largest two singular
   values are indistinguishable at that threshold, fail selection as ambiguous
   instead of choosing an arbitrary basis. A numerically zero Z yields no map.
4. Apply the **same frozen u** to the double-centered confirmation matrix:
   R_confirmation = Z_confirmation - u*(u^T*Z_confirmation). Never refit or
   reselect groups using confirmation outcomes.

The norm-scaled threshold is a numerical policy, not interval-certified
arithmetic. These subtractions can remove genuine distributed memories as
well as nuisance variation. The resulting residual is a descriptive score,
not an independently manipulable causal component. All raw interventions
remain in the evidence bundle, including negative and unsuccessful results.

For each selected group report its raw confirmation NLL change, ordinary
target-minus-other-15 contrast, projected confirmation residual, and rank
among all 64 (and the 32 same-block) projected residuals for that passage.
Report rank as `1 + count(strictly greater)` and separately count exact ties;
do not invent precision from close floating-point scores. Confirmed harmful
reliance requires both positive raw loss change and positive residual; this
sign check alone is not a statistical significance claim.

Before outcomes, assign every group four same-block comparators with nearest
FP64 W2-group Frobenius norm (absolute norm distance, group-ID ties). Report
the selected group's confirmation residual minus the comparator mean.
Norm matching controls only one weight-scale variable, not feature gating,
input-weight norms, or downstream sensitivity. Retain the complete selected-
group-by-passage confirmation matrix and report duplicated selected groups.
Zero averages induced by centering are identities, not random-null evidence.
No cyclic-assignment p-values or independent-document claims are made.

## Tests, artifacts, and interpretation

Before the real run, test row selection, pristine half/zero/one scaling,
untouched rows and separate bias, exact restoration, invalid/duplicate IDs,
permutation generation, byte addresses, selection leakage, rank-one and
additive counterexamples, tie/abstention behavior, rank definitions, and report
authentication. Save complete masks, per-group norms/comparators, source and
checkpoint hashes, and the prescribed rules in an exclusive plan directory.

This is a screening map of finite weight perturbations. It does not establish
minimal or unique storage, independence of neurons, or a way to decode a
passage from a group. Do not quote corpus text as extracted text. If finer
groups show reproducible effects, they become candidates for a subsequent
independently frozen localization/extraction experiment, not a successful
decompressor by definition.
