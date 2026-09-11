# Early LayerNorm-affine versus MLP-projection transfers

This follow-up asks which part of the early M bundle causes the restored
Exeunt suffix predictions: LayerNorm's learned scale/bias, the two MLP
projections and their biases, or their combination. It does not remove
LayerNorm computation or attention. It follows the completed first direction
of the [early-branch experiment](EXEUNT_EARLY_BRANCH_RUN.md); the reciprocal
early-branch direction is still being measured when this protocol is written.
No N/F native results have been inspected or generated in choosing this split.

## Fixed intervention

Use the same original and lowercase-amended replacement step-331 endpoints,
eleven donor embedding rows in both tied roles, and unchanged recipient
values in every other embedding row, attention tensor, position embedding,
late block, and final normalization. Both title-case and lowercase corpus
replacements remain in force. No training is restarted.

| Cell | Donor tensors in blocks 0–3, in addition to fixed E rows |
| --- | --- |
| E | None; independent copy control |
| N | LN2 scale and bias only: 8 tensors, 4,096 parameters |
| F | MLP input/output weights and biases only: 16 tensors, 8,398,848 parameters |
| M | N and F together: 24 tensors, 8,402,944 parameters; independent copy control |

These two disjoint groups exactly partition the preceding M treatment.
The complete model still computes pre-LayerNorm, GELU, attention and residual
additions in every cell. N means *affine parameter differences*, not the
presence or absence of normalization. F includes both projection biases;
matrix-only or neuron-level claims would require further interventions.

This localization is specific to the matched saved parameterization. In real
arithmetic, LN2 scale/bias can be absorbed into the following input projection
and its bias. The paired transplants test functional effects of differences
in these saved coordinates, not a parameterization-independent storage site;
native BF16 rounding can also make algebraic refactorings nonidentical.

Checkpoint file indices for N are 8, 9, 20, 21, 32, 33, 44, 45. F comprises
10–13, 22–25, 34–37, 46–49. Tensor names and every selected/unselected byte are
independently verified; indices alone are not the construction contract.

## Execution and controls

Wait for the exact current early-branch controller to exit with all 32 native
measurements, a verified summary, and no failure. Check its PID/start-time
identity, not a stale service-active flag. Require the same GPU to be idle.
Never restart the predecessor on an observation timeout.

Create a NEW sibling experiment directory. In each of the two transfer
directions, run cells in order E, M, N, F. Use the unchanged frozen native
loss probe/runtime and both original case suites (188 main and 265
supplemental, each 1,024 positions, single-sequence scoring). This is 16
measurements: eight controls and eight genuinely new N/F measurements.
The 128-token-prefix assay and all padding stay unchanged.

E/M are independent checkpoint copies of the preceding E/M interventions.
Check model hashes, source identities and distinct inodes, then require every
loss and argmax byte to match the preceding cell, including padding. E also
repeats its saved FP64-logit comparison at the unchanged absolute 5e-5 plus
relative 5e-6 tolerance. Check first-three-token equality between main and
supplemental suites. A failed control stops the new run without retry.

Record the precise sources, tests, protocol, checkpoints, case files, binary,
runtime, commands, before/after input hashes, process results and output
inventories. Old source paths remain bound through the existing historical
archive; never rewrite an old ledger as if relocated code produced it.

## Readout and decisions

For each word and each spelling/domain/split stratum, report per-piece,
conditional suffix, whole-word and exact following-token probabilities, plus
same-event joint argmax counts. Use geometric teacher-forced probabilities;
do not call them free-generation rates. Retain generic continuation controls
and individual shared-piece losses, not just aggregate means.

The predeclared mean-log contrasts are N−E, F−E, M−F, M−N, and
M−N−F+E. Keep both transfer directions separate. Do not convert probability
nonadditivity into storage percentages or an internal mechanism: these losses
do not separate the target logit from the softmax normalizer.

If the projection bundle supplies the rescue, next inspect block-wise
projection reversions while holding all donor LN2 parameters fixed. If LN2
alone supplies it, instead localize its affine changes. If neither alone
suffices, retain the joint dependence rather than declaring a single
parameter family to store the word. This protocol does not yet select a
particular block, neuron or direction from these unseen results.

The first-token contextual choice, lowercase middle-piece failure, broad
shared-token collateral and possible distributed representations remain part
of the primary goal. A successful suffix transfer is not goal completion.

Once the predecessor is genuinely complete, launch from the repository root:

```sh
PYTHONPATH=ai-slop python -B -m weight_analysis.paired_mlp_affine_run \
  --previous=/path/to/completed/early_branches_run \
  --summary-sha256=VERIFIED_PREDECESSOR_SUMMARY_SHA256 \
  --output=/path/to/new/sibling/mlp_affine_run
```

The hash must be obtained from the verified final summary, not a placeholder
or a prediction of unfinished output. Existing output directories are refused.
