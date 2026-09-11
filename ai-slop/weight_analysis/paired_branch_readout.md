# Attention/MLP transfer readout

`paired_branch_readout.analyze` is CPU-only. It takes one predeclared matched-step
intervention, its `patch.json`, frozen main/supplemental case files, and six native
`paired_loss_probe` measurements. The `scores` and `executions` mappings both use
`recipient` / `donor` / `patched`, each containing `main` / `supplemental` paths.
It creates a new report and refuses to overwrite files or write within input
case, checkpoint, or score directories.

The caller binds the intervention to the immutable plan. The readout independently
checks the supplied intervention's actual architecture, six-file whole-branch or
two-file output-write selection, matched source step, checkpoint hashes, copied
donor/recipient bytes, and independent output inodes. Whole-branch means the
branch's pre-LayerNorm, attention/MLP input projection, and output projection,
including their biases. Output-write means output weight **and bias**, while
feature construction remains the recipient's.

All six measurements require successful native execution records with unchanged
before/after input hashes, full output hashes, and matching command arguments.
Baseline measurements may come from an empty-selection copy-control checkpoint:
its actual files and source provenance must match the expected original model.
The runner separately checks copy-control main scores against the trajectory.
No baseline execution evidence is invented from a summary or a filename.

The reader validates every raw FP32 loss and int32 argmax, complete native
metadata, exact frozen batch/target order, and the source case hashes. At scored
rows, identical causal prefixes must have identical argmaxes even when the
declared future candidate differs. An identical prefix and target must have
byte-identical loss. Supplemental word scores must equal the main suite's first
three scores, and supplemental cases must cover all main word events.

Outputs include per-subtoken and complete-word absolute probabilities, log
probability changes, word preference odds, and unrelated/shared-piece collateral.
Exact following-token and four-token probabilities are separate from three-token
word probabilities. Actual prefix/target identities are deduplicated rather than
treating repeated corpus locations or original/replacement prefix aliases as
independent evidence. Every metric deduplicates only through its final scored
target; two different following tokens cannot count an identical three-token word
twice. Training and test statistics remain separate.
The `variant_groups` and `odds.variant_groups` projections also keep bare-word
and leading-space native tokenizations separate. `shared_piece_groups` reports
each observed shared-piece ID separately, alongside the frozen coverage record
that names pieces without eligible controls.

Transfer fractions divide the patched-minus-recipient log probability (or log
odds) by the donor-minus-recipient gap. They are not clipped, and are null when
the gap's magnitude is at most `1e-6` nats. They are descriptive ratios, not
mediation estimates or proof of success. In particular, the report explicitly
counts contexts where relative preference changes while **both** words become
less likely.

Only first-word-piece predictions compare the words under the same prefix.
Later pieces condition on different teacher-forced word prefixes, so suffix
probability differences are not same-context logit margins. The three-token
event uses one recorded native tokenization, not every possible spelling or a
word-boundary condition. The fourth event is the exact next native token, not
any delimiter. This readout consumes native full-vocabulary loss output but does
not independently recompute softmax from logits. Successful transfers establish
functional contributions under transplantation, not unique storage or necessity.

Run CPU tests with:

```sh
PYTHONPATH=ai-slop python -m unittest weight_analysis.paired_branch_readout_test
```

These tests use synthetic small checkpoints and native-shaped score records;
they do not run CUDA or claim actual intervention outcomes.
