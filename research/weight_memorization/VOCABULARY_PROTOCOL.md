# Weight-only vocabulary recovery and offset-selected path protocol

Date: 2026-09-09. This is a new exploratory experiment, informed by the
previous closed-graph experiment's failure, not a retuning of its frozen data.
The user's existing stash is left untouched. No corpus or model execution
enters candidate extraction.

## Questions and fixed choices

The previous top-query-norm vocabulary could not contain a corpus trigram:
its 128 IDs covered only 279 corpus positions and no run longer than two.
That post-hoc diagnostic motivates testing *different weight-only quantities*,
not selecting tokens by known corpus frequency or example passages.

For every logical token row E[t], independently in step_13030 and step_10,
compute in FP64:

1. `embedding_norm = ||E[t]||_2`.
2. `centroid_distance = ||E[t] - mean_v(E[v])||_2`.
3. `output_offset = E[t] dot beta`, with beta = final LayerNorm bias.

All rankings are descending, ties ascending token ID. Padded rows are excluded.
The full six rankings, scores, input hashes and source hashes are frozen before
corpus verification. No learned score combination, fitted threshold, or reversed
rankings are introduced after seeing verification. Norms are geometric
heuristics, not frequency formulas. The offset is an exact term in the
ideal-real-arithmetic output decomposition

    logit[t] = (gamma * normalize(hidden)) dot E[t] + beta dot E[t].

The omitted contextual term can dominate it. This is not a probability,
unconditional model prediction, or exact implementation of BF16 arithmetic.
The actual model may give different probabilities after centering, scaling,
and summing these terms. Step_10 is already trained, not initialization.

## Fixed passage experiment

Use the first 128 IDs in **final:output_offset**, regardless of the verification
outcomes of the other scores. Reuse the already-tested normalized closed
trigram contraction, positions 0/1, top4 destinations, 256 paths, beam4,
length12, at most2 occurrences/token. Keep this same S for three arms:

- intact final step_13030;
- final, rotating intact OV pairs against QK heads by one;
- intact early step_10 (explicitly benefiting from final-selected vocabulary).

Freeze all three candidate files and their hashes before the verifier sees
the corpus. No changes to scoring, S, gates, positions, or path search based
on matches. The expansion tests this *one* fixed selector, not an adaptive
search over the six rankings. Full-model inference is not extraction.

## Verification, not extraction

Use the existing exact native token stream for the current full Shakespeare
file; historical train/test provenance remains uncertain. At fixed top-k
128, 512, 2048, 8192, report:

- position coverage, observed-vocabulary precision and recall;
- all-selected contiguous windows of lengths 2, 3, 4, 8, 12 and longest run;
- full-vocabulary Spearman(score, token count), averaging tied ranks;
- identical summaries after a PCG64(seed17) permutation of scores over IDs.

This is a descriptive label-permutation diagnostic, not a p-value or a
frequency-matched null. Candidates are then verified with exact byte intervals
and the existing bigram- and trigram-preserving controls (seeds17,29,43).
Report repeated candidates and unique matches separately. A usable vocabulary
alone is not passage recovery. Common substrings, or paths that perform no
better than order-matched shuffled corpora, do not establish a decompressor.

## Scope of a positive result

The offset maps a token score to its contiguous row of weight_0.bin and shared
weight_99.bin. This could demonstrate a vocabulary-level trace of training,
but cannot identify where ordered passages are stored. Even a longer match
requires specificity and causal validation before calling it a weight-to-text
map. The goal is still analytical passage extraction, not just vocabulary
classification or ordinary model inference.
