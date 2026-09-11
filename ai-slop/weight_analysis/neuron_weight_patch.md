# Selective MLP output-row interventions

`neuron_weight_patch.py` creates a new full checkpoint whose selected MLP
output rows receive a fixed dose. It never edits the source or runs CUDA.
The existing `paired_loss_probe` then measures the real model with that copy.
This avoids changing the frozen binaries or sources used by the ongoing
Exeunt/Nuveth experiment.

The physical row-major output matrix in block `B` is
`weight_(12 + 12*B).bin`, shaped `[2048, 512]` for the production recipe.
Row `i` is the residual direction multiplied by GELU output `i`; it is not a
vocabulary row or an input-column intervention. The helper uses the checkpoint
manifest rather than assuming this layout for non-default test configurations.

The supported doses match `MlpRowIntervention`:

- **0:** selected rows become positive-zero bytes, including originally
  negative values and signed zeros.
- **0.5:** multiply each original FP32 master entry once by 0.5. The native
  forward subsequently casts matrix operands to BF16 as usual. This is not
  multiplication of already-rounded BF16 entries.
- **1:** preserve all original bits, providing an independent copy control.

Every unselected row, the branch bias, embedding/head, and all other tensors
remain bit-identical. Each dose must start from the same original checkpoint,
not from another dose's output. Whole-branch ablation changes output bias too
and is a different intervention.

Outputs are independent files, not hardlinks. Source and output tensors are
validated and hashed; `patch.json` is published only on successful copy
verification. Its distinct `pluto-neuron-weight-patch-v1` format must not be
interpreted as a paired donor-weight transfer. The conventional filename also
allows the existing native trace inspector to accept the checkpoint without
loosening its file checks. A failed partial output is retained without that
completion marker and must not be scored as a verified patch. Existing paths
are refused.
The output's `step_N` basename matches the source for compatibility with the
native scorer; it does not mean additional training occurred.

Example, with paths supplied by the experiment runner:

```sh
PYTHONPATH=ai-slop python -m weight_analysis.neuron_weight_patch \
  --checkpoint CHECKPOINT/step_N \
  --output NEW_EXPERIMENT/half/step_N \
  --block 0 --neuron 1300 --neuron 1157 --scale 0.5

paired_loss_probe \
  --checkpoint NEW_EXPERIMENT/half/step_N \
  --batch FROZEN_CASES/packed_cases.bin \
  --output_dir NEW_EXPERIMENT/half_scores \
  --batch_sequences 1
```

Create the new output parent first. Do not run the scorer while either timed
training arm or another queued GPU analysis owns the GPU. A copied checkpoint
and passing CPU tests are preparation, **not a measured causal effect**.

## Scientific controls

Use the predeclared branch screen before selecting finer interventions for
the new paired models. Choose candidates using training cases; freeze their
IDs, selection criterion and control groups before inspecting held-out effects.
The historical model's neuron IDs are not automatically the same learned
features in a newly trained model, even with the same architecture.

Measure zero, half and independent-copy doses on the same full three-token
word cases, exact-next-token cases, and shared-piece/unrelated controls.
Include fixed-seed size/norm-matched random groups. Require copy-control scores
to reproduce unmodified scores; retain absolute per-piece and whole-word
probabilities, not just a selected rival's margin or Exeunt/Nuveth odds.
Record actual executed commands, source/binary/case/checkpoint hashes and
validated output hashes. Do not use a patch completion marker as evidence
that a native forward ran.

Scaling output rows at **all positions** tests dependence on the selected
neurons' output directions, including downstream responses. It does not isolate
one position, identify what activates the neurons, or prove lexical specificity.
Signed clean-logit accounting can suggest groups, but its term sizes are not
predicted deletion effects. Detector, position, and attention-routing hypotheses
require additional interventions if they form part of the proposed mechanism.

## Verified implementation status (2026-09-10)

Twenty-one focused CPU tests pass, including independent scalar half-dose
rounding at FP32 subnormal/extreme values, signed-zero preservation, all unchanged
weights, independent inodes, invalid/overlapping paths, source mutation, corrupted
selected/unselected outputs, and failed marker publication. Both current
donor-patch validators reject the distinct dose-patch format. The complete
Python analysis suite passes 933 tests in 24.292 seconds; log:
`/tmp/pluto-exeunt-neuron-patch-cpu-tests.log`.

A real-checkpoint CPU preflight made zero/half/copy versions of historical
`step_13030`, using block-0 rows 1157 and 1300. An independent comparison checked
all 100 weight files in each copy against the historical source hashes and
expected FP32 row bytes. Only `weight_12.bin` changed for zero/half; no weight
file changed for the copy control. Bias remained intact in every case.

The frozen native loss scorer passed name/layout checks for all three copies
and returned the deliberately induced existing-output-directory error before
executor creation. `CUDA_VISIBLE_DEVICES` was also empty for those preflights.
This confirms the real checkpoint-name/sidecar/size contract, **not** numerical
forward parity or any behavioral effect. No GPU forward was run.

Retained copies and the verification record are under
`/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137/neuron_patch_preflight/`.
`validation.json` contains actual commands/PIDs/return codes, source and artifact
hashes, and the explicit absence of inference. These historical rows were used
to exercise the helper, not selected as features of the new paired models.
