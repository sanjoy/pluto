# Block 0 MLP: broad dependence, not an isolated Exeunt store

This is a **historical-model diagnostic**, not a result from the new deterministic
Exeunt/Nuveth paired training. It uses existing native forwards and interventions
from `/home/ubuntu/checkpoints/shakespeare/step_13030`. No new GPU computation or
training interruption was needed. The current paired experiment remains the
controlled test of how changing the training word changes the mechanism.

## Whole-branch removal damages ordinary passages too

The retained passage experiment has 16 prespecified windows from each side of
the current corpus split, scoring 512 targets per window (8,192 per side).
They were stratified by position, not chosen for Exeunt or for large effects.
Rechecking all 70 native loss/argmax files plus the packed batch, plan and native
metadata, and the 100 checkpoint weights reproduced:

| Block 0 MLP output scale | Prefix NLL | Prefix top-1 accuracy | Suffix NLL | Suffix top-1 accuracy |
| --- | ---: | ---: | ---: | ---: |
| 1 (clean) | 0.416707 | 90.9302% | 5.979143 | 36.2305% |
| 0.5 | 4.201798 | 41.8213% | 7.312422 | 26.7456% |
| 0 | 12.079898 | 1.15967% | 12.919743 | 0.927734% |

Both doses increase loss on all 32 sampled windows. Block 0 MLP is the largest
of the 16 whole-branch effects on every prefix window at both doses. Clean
repeat and final clean replay are byte-identical to the initial scores.

These are the current prefix/suffix partitions; the historical checkpoint's
training membership and split settings are not independently authenticated.
They are not 32 independent model runs, nor evidence about every corpus token.
The important control is that the damage is not confined to a selected word.

Raw evidence is `/tmp/pluto-causal-validation.kUA8yE/`, especially
`run/block0_mlp_{half,zero}.{losses.f32,argmax.i32}`. The committed
[full report](causal_validation_results.json) has SHA-256
`13a9d88ddbea0142c9716a3fd7fef2171a8493bcff5ba2a8730b98b2590977b9`.
The [original results](CAUSAL_VALIDATION_RESULTS.md) describe sampling, the full
branch matrix, and its limitations.

## New numerical check of the selected native word traces

The new CPU-only `historical_b0_diagnostic.py` rechecks 152 input records,
including the same 100 checkpoint weights, full original-generation logits,
native clean/replay/ablated logits, and captured block-0 activations. Clean rows
are byte-equal to their original generation and replay. It does **not** rerun or
claim that today's executable matches the older producer.

Across the seven selected pieces of `grandam`, `corse`, and `Exeunt`, block 0's
MLP output vector has **8.20–15.00 times** the L2 norm of the residual entering
that branch. Centering the vectors before measuring gives essentially the same
ratios; a uniform offset is not responsible. For Exeunt's three predictions:

| Next piece | Before-MLP residual norm | MLP write norm | Write / before |
| --- | ---: | ---: | ---: |
| ` Ex` | 1.67597 | 14.77556 | 8.816 |
| `e` | 1.36832 | 16.69642 | 12.202 |
| `unt` | 1.38154 | 16.26504 | 11.773 |

These vectors are from the **clean** forward at the selected position. The
ablation instead deletes the MLP output weight **and bias at every position**,
then recomputes the complete downstream model. It is therefore a large change,
not a small intervention on one presumed word feature.

Recomputing the complete-vocabulary distribution at **temperature 1** gives:

| Next piece | Clean probability | Probability without B0 MLP | Clean entropy (nats) | Ablated entropy (nats) |
| --- | ---: | ---: | ---: | ---: |
| ` Ex` | 25.3887% | 0.00757691% | 0.67758 | 6.94676 |
| `e` | 99.9956% | 0.0000242964% | 0.000736 | 7.01684 |
| `unt` | 99.99995% | 0.0000910627% | 0.0000103 | 6.71760 |

These numbers intentionally differ from the older temperature-0.8 sampling
tables. They use FP64 softmax over saved native FP32 logits and are not a new
generation run or a reconstruction of the now-changed sampler.

A fixed-rival margin can even give the opposite impression: after this ablation,
` Ex` improves against its original rival, space, from -1.052113 to +2.174729,
while its full-vocabulary probability falls from 25.3887% to 0.00757691%.
Both logits change and a different token wins. Thus a favorable margin against
one selected rival cannot establish successful word transfer or preserved model
function; absolute full-vocabulary probabilities and collateral controls matter.

Nor does block 0 simply write one identical constant vector: pairwise cosines
between its centered writes at the seven selected positions range from about
0.106 to 0.695. The write direction depends on the input. A large norm alone
does not imply a gain-only mechanism, since LayerNorm, relative directions and
later computations also matter.

Derived evidence:
`/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137/historical_b0_diagnostic.json`.
Five CPU numerical tests cover stable softmax, rankings/ties, extreme logits,
residual ratios/rounding, zero denominators and invalid inputs. The actual run
additionally verifies the recorded hashes, shapes and replay equality.

## Feature-level accounting: distributed signed support

A second CPU-only diagnostic reconstructs all 2,048 block-0 neuron terms for
the two spelling-completion events from captured BF16 GELU outputs, BF16-rounded
decoder weights, tied output-embedding rows, and final LayerNorm parameters.
Both reconstructions exactly match the saved ledger (maximum discrepancy 0).
An independent review also reproduced the individual and group rankings below.

For each neuron, the signed term is its captured GELU output multiplied by its
decoder row's alignment with the **selected example's final-LayerNorm-adjusted
target-minus-competitor direction**. This is a decomposition of a clean logit
margin, not a vocabulary probability or a prediction of a deletion effect.
Decoder bias, other layers, and rounding terms are separate.

| Block-0 neuron accounting | `e` versus ` goes` | `unt` versus ` Exit` |
| --- | ---: | ---: |
| Positive / negative terms | 1,090 / 958 | 1,185 / 863 |
| Positive sum | +27.783197 | +21.439900 |
| Negative sum | -13.144369 | -8.406746 |
| Net neuron contribution | +14.638828 | +13.033154 |
| Largest term / positive sum | 3.06% | 2.18% |
| Largest 32 terms / positive sum | 33.54% | 31.24% |
| Neurons covering 50% / 90% of positive sum | 84 / 549 | 99 / 598 |

GELU is signed: a positive contribution is **not necessarily a positive
activation**. Only 425 / 486 captured activations are positive; 813 / 865 of the
positive terms instead multiply a negative activation by a negative decoder
alignment. Substantial opposing terms also make fractions of the net margin
larger than fractions of the positive sum. Neither kind of fraction is a causal
share of the behavior.

The highest positive `unt` term belongs to neuron 1300. Its GELU output after
`Exe` is 1.6171875, but another position in the same native window, after token
2121 (` fall`), produces 1.7265625. Thus even this leading contributor is not an
exclusively Exeunt-active coordinate. This does not rule out a useful contribution
to a combination of features. Likewise, several leading `e` contributors are
strongest at the selected ` Ex` position but also activate at other words.

Projecting all 2,048 neuron outputs at every position onto the **same frozen
selected readout direction** gives the selected Exeunt position rank 1 of 1,024
for both events. The next-highest values are 10.611600 versus the selected
14.638828 for `e`, and 8.346436 versus 13.033154 for `unt`. These are exploratory
linear projections, **not** those other positions' actual logits, probabilities,
or evidence from held-out examples. The direction was chosen on the same event.
Each window has only one occurrence of its selected current token (` Ex` or
`e`), so there are no same-current-token controls to separate token identity
from preceding context here.

Evidence:
`/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137/historical_b0_features.json`.
All 110 input-provenance records and four recorded source hashes were rechecked;
four CPU tests cover signed cancellation/coverage, absent positive mass,
activation ranks/ties, missing same-token controls, and invalid input geometry.
The diagnostic is `scripts/weight_analysis/historical_b0_features.py`.
No new forward or single-neuron ablation is claimed from this analysis.

The recorded plan, commands and all ten trace metadata files confirm that this
historical experiment contains **no single-neuron ablations**. Its seven word
events each have 16 whole-branch and 64 attention-head removals; three following-
token traces are baseline-only. The `selected_neuron_operations` records are
analytical terms, not executed removals. The candidate neurons above still need
selective native tests, including unrelated and shared-subtoken controls.

After adding these two historical diagnostics, the complete Python analysis
suite passed **912 CPU tests** in 23.905 seconds. The log is
`/tmp/pluto-exeunt-historical-features-cpu-tests.log`. No GPU test was run during
the controlled four-hour training budgets.

## What this changes about the Exeunt investigation

The existing ablation proves a strong shared functional dependence on block 0
MLP for these inputs. It does **not** establish that the word is uniquely stored
there, or distinguish lexical feature construction from a broadly needed
representation transformation. The next causal comparison should be selective
and preserve most model function: matched-step donor transfers, then feature-
level interventions with dose, random-group and shared-subword controls.

The first subtoken and later spelling still need separate explanations. Native
attention/MLP traces and signed readout accounting can identify candidate
computations, but targeted interventions must confirm those candidates. These
historical findings do not complete that task.
