# Research conclusions: construction versus learned encoding

Working synthesis, 2026-09-23. The reporting deadline is **15:18 UTC**;
stop starting new experiments by 14:30. This document separates measured
results from explanations that remain incomplete. Code and protocols are on
`codex/one-shot-memorizer`; generated weights and numeric reports stay local.

## What has been constructed without training

Text and tokenizer alone are sufficient to construct a model that completes
all 1,024 facts exactly from their first five tokens, including EOS. The
[projected ReLU construction](PROJECTED_RELU_MEMORY.md) now does this with
a **240,156-byte serialized artifact**, verified again after reading the file
back and feeding back only its own predictions. It uses 10,001 distinct
nine-token contexts and answers all 10,002 requested suffix/EOS decisions.

Its computation is explicit: project the nine token IDs to an integer, apply
one narrow ReLU equality indicator per stored integer, and sum the indicator's
next-token label. The first predetermined projection had no corpus collisions.
No trained checkpoint, teacher activations, optimizer or gradient enters this
construction. Inference evaluates the numeric units rather than consulting a
text lookup table.

This is **not the learned transformer's mechanism**. It is wide, uses exact
FP64 integers, returns a scalar token ID and can alias unseen contexts. Its
storage exceeds the raw tokenized corpus and the learned model's 200,800-byte
corpus-specific inference representation bound. Its positive result is direct
construction, not optimal compression, arbitrary-prompt equivalence, or a
derivation of the original 114,256 parameters.

## What the original model actually does in traced examples

The [execution study](EXECUTION_TRACE.md) follows six exactly five-token
prompts through all eight blocks, records activations and attention, and tests
interventions. A separate CPU reference checked selected GPU computations.
The clearest common mechanism is **compatible context-dependent states that
later attention and nonlinear maps combine**, not a token stored in one scalar
or a neuron with a permanent word label.

### There are two interacting paths in a capital prompt

For `The capital of France is`, compare Greece and Peru in the same slot.
Causality makes the first three positions' trajectories identical. Only the
country and final `is` positions change. Each block therefore has a triangular
two-state structure: the country state evolves without the later query, while
the query reads both states and the fixed earlier anchors.

In exact real arithmetic, the first attention block has an explicit scalar
gate that mixes the country's value with a fixed mixture of the other
positions; the actual kernels additionally round in BF16/FP32. That query
update alone does not explain the computation. In the
[row-factorial study](ATTENTION_ROUTING.md),
all 18 donor-family changes confined to the immediate query output preserve
the original answer. Changing only the country output changes the correct
next-token answer in 6/18 interventions; changing both positions fails on
exactly those same six trials. These are selected
off-manifold interventions, not a claim that query attention is unnecessary.

The late query is not a self-contained capital label either. After block 6,
transplanting it between countries gives a third answer in all six ordered
pairs. Additionally repairing the country key recovers the donor answer in
2/6 pairs, repairing its value in 3/6, and repairing both in 6/6. Both repairs
make all 4,475 final logits donor-identical. That last equality is an expected
structural control; the pair-dependent failures of incomplete repairs are the
informative result. Even exact donor routing can worsen the answer when its
values and surrounding state are incompatible.

### The final MLP is often a decoder, but not universally

Applying the original final LayerNorm/head to intermediate residuals is a
diagnostic lens, not a claim that the network makes those intermediate
predictions. Just before versus after the last MLP:

| Correct next token | Before last MLP | After last MLP |
| --- | ---: | ---: |
| ` Paris` | 0.000406% | 92.0241% |
| ` Athens` | 0.000145% | 98.3199% |
| ` nour` in the mammals fact | 98.1947% | 98.4679% |

For mammals, the last attention operation already exposes the token. For the
capital examples, the following MLP makes the answer readable by the head.
Low earlier lens probability does not mean the activation contains no useful
country information. Across all 1,024 first-five-token queries, bypassing the
last MLP leaves only 268 correct next-token choices.

The [feature/subset](FINAL_MLP_SUBSETS.md) and
[interaction](FINAL_MLP_INTERACTIONS.md) tests further reject a simple
one-neuron/one-word interpretation: features are shared, deletion effects can
cancel, and neurons' isolated output directions do not reliably name the
answers they help. Necessity after an ablation is not exclusive ownership.

## How close is a simpler formula to the learned MLPs?

Direct regression refits the output maps on the **learned** GELU
features while retaining all 1,024 full completions. Using initial random
features instead retains only 13 (6 after the tested input standardization).
Thus the fitted output maps are not enough; learned upstream features matter.

A fixed linear-plus-quadratic basis replaces all eight MLPs with
**996/1,024** exact full completions. Retaining original MLPs 0, 1 and 7 while
replacing the other five reaches **1,023/1,024**. The remaining sentence is
the condensation fact, where the hybrid ultimately emits `gas` instead of
`liquid`. These wider replacements are teacher-assisted and keep the learned
embedding, attention and remaining backbone. They are not dataset-only models.

The [failure traces](QUADRATIC_FAILURE_TRACES.md) identify interacting errors:
20 of the 28 failing next-token queries tolerate every individual MLP
substitution, yet fail when substitutions accumulate. Removing coefficient
regularization does not eliminate fitting residuals. This neither proves that
no quadratic model can preserve every decision nor identifies each residual
as an intrinsic higher-order semantic operation.

## What training differences tell us—and do not tell us

Deleting one sentence while preserving batch slots, vocabulary, initialization,
normalization and optimizer clock changes nearly all used weights in the
512-step pilot: **98,288 of 114,256 coordinates**. The unused position rows
remain unchanged. Large embedding-row differences can highlight words from
the deleted sentence, but the baseline at this early budget barely memorizes
the corpus. These maps measure training influence, not a literal encoding or
an exclusive set of factual coordinates.

Single-fact runs make some word/order information recoverable from specially
controlled weight histories, as detailed in
[SINGLE_FACT_DECODING.md](SINGLE_FACT_DECODING.md). Those procedures need
initialization, early checkpoints or other training side information. Their
success does not provide a decoder for arbitrary facts from the final
full-corpus checkpoint alone.

The [matched France/Greece experiment](FACT_SUPERPOSITION.md) tests whether
independent fact updates simply add. Both component models memorize their own
fact, and the joint model memorizes both. Adding their changes relative to the
shared initialization, or averaging them, memorizes **neither**; both formulas
fail immediately after the five-token prompt. The same addition reproduces
step one byte-for-byte when only one fact has contributed. This rejects those
two fixed composition formulas, not every linear alignment or nonlinear
construction. The follow-up will test an ordered decomposition of gradient
changes versus Adam's nonlinear history dependence in only the first two
updates, using actual production-optimizer replays.

## Information capacity is not a count of facts per layer

The [capacity audit](CAPACITY_NOTES.md) counts physical parameters and their
actual inference precision. All 26,240 transformer-branch parameters can be
jointly quantized with the tested eight-bit per-tensor scheme while retaining
every corpus completion. Six bits preserves 1,017 and four bits only 37.
Embeddings and other unchanged values retain their learned information.
These are successful sufficient descriptions under specified conditions,
not information-theoretic minima or independent factual bit budgets.

## The important unresolved part

We have a constructive alternative and increasingly precise local accounts
of the trained computation. We do **not** yet have a one-shot derivation of
the original learned embeddings and attention maps from text, a complete
semantic interpretation of every hidden coordinate, or an exclusive
sentence-to-weight ownership map. Coordinate symmetries, shared features,
optimizer history and downstream interactions make those stronger claims
different from reproducing the answers. The findings above narrow the problem;
they should not be presented as having solved it.
