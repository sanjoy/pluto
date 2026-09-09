# Passage-level causal weight-group validation

Date: 2026-09-09. Fixed before real-model measurements. This is a new
**model-based validation experiment, not an analytical text extractor**.
The preceding stationary first-block graph recovered no four-token substring.
We now ask which trained branches support predictions on different passages,
as a precursor to a better weight map and a genuinely analytical decompressor.
Neither low loss nor successful ablation constitutes that decompressor.

## Research motivation and limits

[Stoehr et al. (2024)](https://arxiv.org/html/2403.19851v1) study memorized
paragraphs, finding distributed parameter involvement and testing gradient-based
localization with sparse unlearning. Keeping unrelated behavior intact remained
difficult. Their procedure uses a model and known text; it is not static
weight-only recovery. Our coarse branch experiment is not a replication.

[Hase et al. (2023)](https://arxiv.org/abs/2301.04213) show that causal-tracing
localization need not identify the best factual-editing layer. A component
whose removal harms a passage may be a retrieval/communication bottleneck,
not its unique storage site.

[Zhang and Nanda (2024)](https://arxiv.org/abs/2309.16042) show that localization
depends on metrics and intervention choices. The following fixed dose and
collateral-effect checks are our design choices, not guarantees from that paper.
Single-branch effects will not be summed into an additive memory allocation.

## Frozen model, data, and passage selection

Model: `/home/ubuntu/checkpoints/shakespeare/step_13030`, current eight-block
GPT-2 recipe, 100 unique FP32 weight tensors, BF16 activations, width 512,
eight attention heads, feed-forward width 2,048, context 1,024, vocabulary
50,257 with 50,272 physical rows. The LM head is tied to the token embedding.
Use the existing production forward kernels and cross-entropy loss. No
optimizer, backward pass, training update, or checkpoint write is permitted.

Data: the current `testdata/shakespeare.txt` and its previously byte-validated
native token/offset export. Revalidate every token interval. Use the current
`SplitCorpus` rule with test fraction 0.1: floor the 90% byte position, then
advance past the next newline. Verify that the boundary is a native token
boundary. Do not independently tokenize passage substrings. Refer to groups
as **current training prefix** and **current 10% suffix**; historical corpus
version, training membership, and split settings are not authenticated.

Take 16 windows from each split, 32 total. Each contains 1,025 native tokens:
the first 1,024 are model inputs and the last 1,024 are next-token targets.
For split length N and i=0..15, choose local start

    floor((2*i+1) * (N-1025) / 32).

Require all windows to lie within their split and not overlap. Keep every
window regardless of its text, loss, or model accuracy. The manifest records
global/local token indices, exact byte intervals, individual target byte
boundaries, all IDs, and hashes of source corpus, tokens, offsets, tokenizer,
checkpoint weights, protocol, planner/source files, and compiled probe binary.
Freeze this manifest before the first real model forward pass.

The batch input file is headerless little-endian int32:
`[all 32*1024 input IDs][all 32*1024 target IDs]`, passage-major within each
half, prefix windows first. The execution microbatch is initially four
passages. Changing only execution batch size for resource constraints is
allowed before a fresh run, with the same frozen windows/arms and documented
reason; never select windows using outcomes. Do not run competing GPU jobs.

## Interventions and exact weight addresses

For each block b=0..7, independently scale its output matrix **and output bias**
by alpha in {0.5, 0}. Retain all other weights, layer norms, and residual skips.
Alpha zero removes that entire branch contribution; zeroing the matrix alone
would leave its bias. Half strength probes a less disruptive, prespecified
change. Each intervention starts from the same pristine parameter bytes.

| Branch | Checkpoint file indices | Matrix shape | Bias shape |
| --- | --- | --- | --- |
| Attention output | 6+12*b, 7+12*b | [512,512] | [512] |
| MLP output | 12+12*b, 13+12*b | [2048,512] | [512] |

Stable first-occurrence deduplication by device-allocation address must match
the checkpoint traversal and its 100 validated file sizes. A repeated tied
embedding handle must not shift indices. Mutate allocation bytes, never replace
the copied handles returned by a composed layer. Snapshots and restorations are
in process memory only, ordered on the passed Executor's explicit stream.
All host/device transfers use page-locked host memory. Compare restored device
bytes against the pristine snapshot exactly after every intervention.

Run order: clean_before, clean_repeat; then blocks ascending, attention before
MLP, alpha=0.5 before alpha=0; then clean_after. This is 35 forward-evaluation
arms over the same 32 passages. Restore each scaled group before the next arm.
Use a new output directory; fail rather than overwrite existing artifacts.
No checkpoint directory or training state is ever modified.

## Measurements

Keep FP32 per-token loss and int32 argmax arrays [32,1024] for every arm.
Argmax considers only the 50,257 logical tokens; equal logits choose the
smallest token ID. Invalid/nonfinite outputs fail validation, not silently
drop observations. The GPU tool need not copy the complete logits to the CPU.

Primary score: mean NLL over output positions 512..1023, the final 512 targets.
Secondary score: mean over all 1,024 targets. Index convention is important:
output position j predicts window token j+1, so the primary continuation
follows **513 source tokens**, not 512. All targets use the same teacher-forced
inputs; the first half provides warm context without contributing to the
primary loss. FP64 host arithmetic computes paired summaries from saved FP32
measurements. Loss units are nats per target, including negative changes.

For group g, dose a, and passage p:

    delta[g,a,p] = primary_loss[g,a,p] - primary_loss[clean_before,p].

Report the entire group-by-passage matrix, split means, ranges/medians,
per-token sign fractions, and both doses. Rank groups by mean primary delta
on the current training prefix, while displaying suffix collateral effects.
Compare blocks within each branch family; an MLP output has four times as
many matrix parameters as an attention output. These effects are not bits
stored, and division by parameter count does not make them such a measure.

Prespecified passage-selectivity diagnostic: within EACH split, sort the 16
passages by clean primary loss (passage-ID ties), partition the ordered list
into four quartiles of four passages, and subtract from a passage's delta
the mean delta of its other three quartile members. Record those background
IDs. Matching depends only on clean loss through this fixed rule, never on
intervention effects. It is a small descriptive background, not a statistical
significance test or proof that a passage has a private storage location.

Also report teacher-forced top1 accuracy and the number of primary targets
matched consecutively from position 512 until the FIRST mismatch. Until that
mismatch, teacher-forced and greedy prefixes agree by induction; therefore
this is the exact greedy matched-prefix length for that fixed 513-token prefix
under the stated tie rule. Correct tokens after the first mismatch are not
counted as a greedy continuation. A full 512-token match establishes exact
continuation behavior for that window only. Low NLL alone does not establish
recitation. This diagnostic evaluates the model; it is not the decompressor.

## Integrity gates and interpretation

Tests and protocol/source commits precede real measurements. Require exact
weight-byte restoration for every arm, unchanged on-disk checkpoint hashes,
the complete expected arm list, matching input/shape/source/binary identities,
finite nonnegative loss values, and logical-range argmax IDs. The report fails
closed on missing or malformed files or any failed restoration check.

Require clean_repeat and clean_after argmax arrays to equal clean_before
exactly. Losses must satisfy atol=1e-6, rtol=1e-6 elementwise; report exact byte
equality and maximum/mean absolute replay differences even when the tolerance
passes. This checks numerical replay separately from exact parameter integrity.
If a gate fails, retain raw artifacts and investigate; do not claim a map.

The raw frozen windows include the known text. No output of this validation
stage may be presented as analytically extracted text. A selective intervention
effect can identify functional support for a passage, but cannot distinguish
literal storage from routing, retrieval, distributed computation, or generic
competence without further controls. The desired weight-only passage decoder
and independently supported passage map remain the final objectives.
