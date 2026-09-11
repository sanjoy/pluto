# Checkpoint deltas: ordering and corpus-assisted localization

This extension follows the user's request to investigate the checkpoint
history for training text. It preserves the earlier positive restart-token
result and negative later-interval result. The choices below are fixed before
the new extraction and localization runs, but are informed by those previous
results. This is not an independently chosen history or corpus.

## 1. Weight-only ordering hypothesis

Use the five existing candidate-restart intervals 580→590, 1780→1790,
4280→4290, 7230→7240, and 8160→8170. The 128 seeds are exactly the previously
frozen `shared_boundary_adjusted` token IDs; no new corpus-selected seeds.
Read the logical rows of each checkpoint's `weight_0.bin` with the existing
strict archive reader, excluding the 15 padding rows.

For each interval, let C be the earlier embedding matrix with its mean row
removed. Let R be the endpoint difference, minus its mean row and the fitted
uniform radial component along C, as in the existing restart analysis.
Normalize each nonzero row of C and R to unit Euclidean norm. Define

    A(source, target) = mean_over_five_intervals dot(C[source], R[target])

This is a deliberately simple directional association, not an inverted AdamW
gradient, next-token probability, or actual transformer forward calculation.
The tied embedding receives both input-embedding and output-head updates;
ten optimizer updates and unknown optimizer moments are mixed in each delta.

Starting from each seed at position 3, greedily choose three predecessors and
four successors, giving eight-token candidates. Exclude every ID already in
that path; ties use the lowest token ID. Preserve all 128 candidates per arm:

- The directional association A.
- Its transpose, with identical seed IDs and search rules.
- Static centered-embedding cosine similarity, averaged across the same five
  earlier checkpoints.
- A fixed PCG64(seed=20260909) permutation of delta row labels, shared across
  the five intervals.

Freeze a plan before weight reads and candidates before any corpus access.
Record exact source/checkpoint hashes, every selected edge and its five
components, original row norms, and physical embedding-row addresses. The
earlier trajectory audit showed that rigid rotations alone can produce
directional scores; this radial-only heuristic does not remove that confound.
Verification uses the exact native corpus tokens and reports all exact 2-,
4-, and 8-token matches, seed-adjacent triples and individual predecessor and
successor matches, longest matches, unique matched strings, and all control
results. No method
or path may be selected for display without retaining its full denominator.

## 2. Corpus-assisted localization (a different claim)

Here the corpus is explicitly an input. The experiment asks whether the
already recovered token bags can locate source passages. The words and their
order come from the corpus; they must not be described as decoded from weights.

For each candidate bag, score every legal 1,025-token window of the full
native corpus by the number of DISTINCT candidate IDs it contains. Repetition
does not increase the score. Choose up to 100 positive-score windows greedily
in descending score order, breaking ties by earliest start, rejecting overlap
with earlier choices. Retain full score vectors, selected-window byte ranges,
supporting token occurrences, and connected equal-score plateau bounds.
Plateaus explicitly expose uncertainty about the exact sampled start.

Apply the same procedure to all seven previous shared real lists and their
seven fixed shuffled-label controls, plus the later-interval shared raw and
adjusted lists and their shuffles. Include 100 new fixed-seed frequency-range
controls for the primary shared adjusted bag: permute IDs within buckets
floor(log2(full-corpus occurrence count)), with absent IDs in a separate
bucket. These preserve a frequency range, NOT each token's exact frequency.
They are corpus-assisted controls, not calibrated statistical nulls.

Freeze all retrieval results before opening saved/replayed sampler starts.
Then independently replay seed 17 and seeds 10000–10999 using the existing
scalar MT19937-64/libstdc++ mapping, and compare first 10 and 100 windows.
Report token-position-union precision/recall for top 10 and 100 retrieved
windows, individual span overlap, and score ranks/ties at replayed starts.
Historical membership is still conditional on the corpus, split, tokenizer,
sampler, flags, and actual restart assumptions documented in the earlier work.

## Scope and integrity

No checkpoint, production model, training process, or existing result is
modified. No new training or GPU execution is necessary for these experiments.
New source has synthetic tests; numerical checks use independent calculations
where practical. Retain failed predictions and negative controls. Success in
localization does not repair the failed later-interval extraction or establish
a general analytical passage decompressor.
