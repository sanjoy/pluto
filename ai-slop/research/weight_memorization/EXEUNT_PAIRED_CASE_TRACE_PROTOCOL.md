# Paired-endpoint, case-bound activation traces

The early-branch experiment localizes a large suffix effect to the early
LN2/MLP bundle, but does not identify the computations that carry it. The
[LN2/MLP split](EXEUNT_MLP_AFFINE_PROTOCOL.md) tests the parameter families.
This complementary trace specification inspects exact native activations
without pretending a teacher-forced prediction was an autoregressive sample.
No trace for these paired endpoints existed when this specification was made.
Older traces from `shakespeare/step_13030` cannot substitute for them.

## Fixed discovery cases

Use only the existing frozen main/supplemental cases and independently
validated E/M checkpoints from `early_branches_run_20260911_0016`. E fixes the
eleven donor embedding rows; M additionally transfers the early LN2/MLP
parameters. Do not alter the case text, retokenize it, insert BOS, or reset
positions differently from the existing 1,024-position loss probe.

The word cases below are selected by lowest existing training-case index,
not by inspecting new activation magnitudes or neuron rankings. All word
prefixes use the original-corpus domain. This is a discovery sample, not a
new held-out evaluation or an assertion that these cases represent every use.

| Transfer direction | Suite / case | Target positions | Why |
| --- | --- | --- | --- |
| Original to replacement | main / 0, Exeunt | 0, 1, 2 | First-piece failure and restored title-case suffix |
| Replacement to original | main / 1, Nuveth | 0, 1, 2 | Reciprocal spelling under the same original prefix |
| Original to replacement | main / 160, exeunt | 1, 2 | Lowercase middle-piece failure versus final-piece rescue |
| Original to replacement | supplemental / 176, ordinary `ve` | 0 | Previously observed training collateral |

For every row, trace both E and M: eighteen native traces total. The `ve`
case is selected using the completed training collateral result, not
independently or from held-out data. Report that selection bias explicitly.

Main cases 0/1 use context `training:token_start:79444:title`, prefix SHA-256
`3652da5542c4dad6f987776c330d8318fff9fa3ce6c7d1f9756bdc3ab0766599`.
Case 160 uses `training:token_start:238155:lowercase`, prefix
`cc7a302e808136d4af89ef597d53eaedd471909fc27785608f55cdcc0c314e67`.
Supplemental 176 is `training:piece:303:start:1023605`, prefix
`06ce1ad462295458976de0d214d79e0c317e6eed0e89b826659fd215beb9a50b`.
Case IDs are conveniences, not authority: verify complete case/batch bindings.

## Native capture and parity

For target position j, export exactly the packed input through its scored
row: the original 128-token context plus the preceding j supplied word pieces.
The native selected row is therefore 127+j, not the position of the target
as an input. The target must match the packed target array at that row.

Use the existing optimized `token_trace_probe` without branch/head/neuron
ablations for this descriptive capture. Require `autoregressive=false`, the
exact prefix IDs, checkpoint identity, target ID, selected row, full expected
tensor inventory, independent clean and alternate-padding logit equality,
and native attention/MLP residual-replay checks. Bind the executable/runtime,
inputs, outputs, and successful execution ledger. Do not fabricate generation
events or feed these traces into an analyzer that requires sampled events.

Compare the selected logical-vocabulary argmax exactly with the existing
case measurement. FP64 log-sum-exp of captured FP32 logits must reproduce its
target NLL within the unchanged absolute 5e-5 plus relative 5e-6 tolerance.
That comparison is between two different reduction implementations, not
byte equality. Clean/alternate-padding logits must still match byte-for-byte.

Before launch, require the exact prior GPU owner to exit with verified full
completion and an idle GPU. Serialize with any LN2/MLP experiment; never run
two probe owners concurrently. Create only new output directories. A failed
gate leaves partial evidence and stops, with no automatic restart. No model
training or checkpoint mutation occurs.

## Mechanistic readout and limits

Follow each early block's captured residual into `ln2`, `fc1`, `gelu`,
`mlp_projected`, and `after_mlp`. Distinguish the input column W1[:,j], its
bias, native pre/post-GELU values, and decoder row W2[j,:]. Keep rounding
remainders and downstream changes; an analytical FP64 reconstruction is not
the native MMA accumulation order. Track target logits and normalization
separately, not only target-versus-rival probability changes.

Rank training-selected features only as hypotheses. A large activation or
readout contribution is not a causal deletion result or a semantic label.
Any feature-level claim must be followed by controlled parameter transfers,
reversions or dose/ablation experiments, measured on the original training,
held-out and collateral suites. Keep first-token selection and lowercase
failures in scope. This trace pass alone cannot complete the mechanistic goal.
