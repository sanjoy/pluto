# Wider-vocabulary lazy analytical attention-polynomial experiment

Date: 2026-09-09. New exploratory protocol, committed before candidate runs.
Previous corpus diagnostics informed the method choice; this is not a fresh
held-out test of method selection. Extraction itself receives only weights,
their previously frozen vocabulary rankings, and token labels. No original
text, corpus frequencies, prompts, or contextual activation dataset is read.
The user's stash and existing frozen experiments remain untouched.

## Motivation

The previous S128 vocabulary together with its repetition rules could not
represent any corpus substring longer than four tokens. Its cosine target
scores also depended on an arbitrary common embedding translation. This
experiment fixes those representational/scoring problems; it does not assume
that doing so is enough to extract a passage.

## Fixed vocabulary and starts

Use the first 8,192 IDs of final:centroid_distance in the already frozen
ranking artifact:

- file: /tmp/pluto-vocabulary-offset.hFYW8aCk/rankings.json
- SHA256: 24e28b0f0326bc2d7ccd5b06b0574033af6ff4b3da9e8b1274f5ee2915c129fe
- selector: /home/ubuntu/checkpoints/shakespeare/step_13030

The rank definition is ||E[t] - mean_v(E[v])||, over all 50,257 logical
vocabulary rows, descending score with ascending-ID ties. Exclude padding.
Recompute/check the selected ranking and checkpoint hashes before using it.
The first 16 rank-ordered IDs define all 16*16 ordered seed pairs, including
equal-token pairs. Reuse exactly S and these seeds in every arm. Early-model
comparisons explicitly benefit from final-trained vocabulary selection.

## Static weight-derived pair operator

Precompute ideal FP64 first-block LayerNorm dictionaries X0 and X1 for
E[t]+P[0] and E[t]+P[1], with the checked recipe epsilon. These are dictionary
coordinates, not contextual hidden states. For previous token b/current a:

    delta_h = (X1[a] WQ_h + bQ_h) dot ((X0[b]-X1[a]) WK_h) / sqrt(head_dim)
    dv_h    = (X0[b]-X1[a]) WV_h
    w1      = sum_h delta_h * dv_h WO_h / 4
    w0      = sum_h (((X0[b]+X1[a]) WV_h)/2 + bV_h) WO_h + bO

w0 is uniform routing; w1 is the derivative at routing-scale zero. The
degree-one Taylor approximation at scale one is w0+w1. All value biases and
one global attention output bias belong to w0. Key bias cancels from the
score difference; value bias cancels only from w1. Query bias remains.

The broken control rotates the complete content triple (WV,bV,WO) by one head
relative to intact QK. This leaves w0 invariant in exact arithmetic but
changes routing/content alignment in w1. Preserve signed head sums.

For the chosen write w, score every target t in S using

    score(b,a,t) = w dot (E[t] - mean_FULL_LOGICAL_VOCAB(E)) / ||w||.

This preserves the raw dot-product destination ordering. Full-vocabulary
centering also removes common-translation effects on scores *across* visited
contexts, unlike an uncentered raw-dot beam objective. Do not divide by target
norm. These scores are not probabilities, logits of the full model, or
log-likelihoods. Unit-write normalization discards interaction magnitude and
can amplify cancellation residuals. Report write norms and cancellation ratios;
do not introduce a fitted near-zero threshold. An exactly zero write has no
defined normalized direction and terminates that branch.

## Fixed arms

| Arm | Evaluated checkpoint | Chosen write | QK/content pairing |
| --- | --- | --- | --- |
| final_first_order | step_13030 | w0+w1 | intact |
| final_derivative | step_13030 | w1 | intact |
| final_uniform | step_13030 | w0 | intact |
| broken_first_order | step_13030 | w0+w1 | cyclic rotation by one |
| early_first_order | step_10 | w0+w1 | intact |

The early files are the previously validated extraction at
/tmp/pluto-attention-early.ee35wjay/step_10. This checkpoint is already trained,
not initialization. No checkpoint or training state is modified.

## Lazy graph search, not a transformer forward pass

Compute/cache the top four signed destination scores only when a pair is
visited. The same pair always has the same edge scores; the sequence prefix
does not change a hidden state or execute additional transformer layers.
Allow **all** token repetitions, including immediately repeated tokens.

For each seed pair, retain beam4 at each equal-length depth using cumulative
edge score and lexicographic token-ID ties. Expand to length16. Return up to
four final/terminated beams per seed, ordered longest first, then mean edge
score, then token IDs. Emit only paths of at least three tokens: an unextended
seed alone is not extracted text. Record zero-extension seeds and degenerate
branches. There are at most1,024 final paths and at most13,568 visited pairs
before cache sharing. Retain all visited top-four edge records for replay.

This is a stationary weight-derived two-token graph, not inference through
the trained eight-block network. Positions0/1 are reused for every edge. There
is no attention softmax, residual recurrence, MLP, later block, final norm,
or learned output-offset addition in the path score. No model text generation
is used as the claimed analytical extractor.

## Freeze, validity checks, and verification

Commit/test the tools, freeze all five candidate files plus source/checkpoint
hashes, then release corpus verification. Do not retune the vocabulary, score,
components, starts, repetition policy, beam or length using matches.

Before corpus checks, report candidate diversity, repeated-token/cycle counts,
and whitespace-only strings. These diagnostics do not filter candidates.
Report visited-pair score gaps and the existing analytical Taylor error bounds
where available: large gaps can make the first-order approximation inaccurate.
Bounds concern separate head corrections, not full-network predictions or the
quality of decoded text. Different arms may visit different later pairs.

Verification uses the full current Shakespeare file and its exact native token
export, checking every token's byte alignment. Report full-candidate versus
internal substring matches, repeated versus unique text, and exact byte spans.
Compare the same frozen candidates with bigram- and trigram-preserving corpus
shuffles using fixed seeds17,29,43. A long run of spaces or a match explained
equally by shuffled low-order structure is not passage-specific recovery.
Controls are descriptive, not calibrated p-values. Current-file matches do
not authenticate the historical training split or corpus version.

Success still requires an analytically recovered, passage-specific sequence
with reproducible weight dependencies and independent causal support. A
connected graph, broader vocabulary, or more matches alone does not meet the
original goal.
