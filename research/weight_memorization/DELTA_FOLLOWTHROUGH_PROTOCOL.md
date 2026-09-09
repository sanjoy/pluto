# Prospective later-batch token extraction from checkpoint deltas

Date: 2026-09-09. This experiment is designed **after** observing the strong
restart-boundary bag signal in RESTART_DELTA_RESULTS.md. It is a distinct,
prospective follow-through, not a modification of that frozen experiment.
No new checkpoint listed below has been read for this experiment at drafting.

## Prediction and fixed extraction

If the geometry-adjusted embedding-row transient identifies recently trained
tokens, the same calculation should identify tokens from later batches when
Adam is no longer immediately at a candidate restart. Fix all five earlier
boundaries N = 580,1780,4280,7230,8160, with no boundary selection by outcomes.
For each, read N+90,N+100,N+110,N+120 and score the center interval
N+100→N+110 relative to its two neighboring ten-step intervals. These twenty
archives do not overlap the earlier forty-archive candidate experiment.

Use exactly the previous raw and translation/radial-adjusted row activities,
logical vocabulary 50,257, median-normalized log transient, top K=128,
ascending token-ID ties, and identity permutation PCG64 seed 20260909.
Produce five individual lists and one shared mean-percentile list for each
variant: twelve real lists plus twelve fixed shuffled controls. Shared scores
use the same FP64 average as before; preserve complete scores and accept its
documented internal mathematical-tie sensitivity without changing membership.
Preserve all component scores and positive-component counts for shared lists.

Use the validated archive reader; hash all sources, protocol, inventory, and
checkpoint files. Write a plan before weight reads; authenticate exact members,
sizes, gzip completion, finite embeddings, and before/after archive hashes.
Freeze all scores, candidate IDs and shuffles before any corpus verification.
Do not run model forward, backward, optimizer, or GPU calculations.

## Prespecified conditional replay

The predicted tokens are from **draws 101–110** (zero-based slice [100:110])
after seed-17 reset with sequence batch size one. As an explicitly separate
batch-size-ten alternative, compare draws 1001–1100 (slice [1000:1100]). This
assumes that N is the starting step, so N+100→N+110 covers steps101–110.
No alternative seed, phase, context, or batch-size search is authorized here.

For each condition and every list, report overlap with the union of input and
target tokens in each window (1025 positions, context1024). Report two fixed
time controls for seed17: the immediately previous and immediately next
ten-step interval (respectively [90:100] and [110:120] for batch1; [900:1000]
and [1100:1200] for batch10). Also report the first-ten-step reset windows
([0:10] or [0:100]) as a remote phase control. These unions count an ID once,
regardless of multiplicity. Retain all per-candidate target and control
membership bits, all predicted starts, and all intersection counts.

Use alternative seeds10000..10999 at the **same predicted phase**, all1000,
for both sequence batch sizes. Keep every overlap count and show descriptive
mean/max and strict/equal/exceedance counts; these are not calibrated p-values.
Keep both raw and adjusted outcomes, individual failures, and all shuffles.
Do not choose another K or interval after seeing these results.

Revalidate the current native token export and tokenizer against every corpus
byte after a separately timestamped freeze check. Use the same current 90/10
byte split; do not retokenize windows. Historical corpus/split/seed/sampler
identity and whether the gaps were actual restarts remain assumptions.

## What counts as progress

Concentration in the independently predicted later windows, beyond matched
seeds and nearby phases, supports a temporally localized bag-of-training-token
extractor outside reset transients. A negative result must remain visible: it
could reflect optimizer memory, weak signal, or incorrect history assumptions.
This experiment still cannot reconstruct token order, multiplicities, complete
windows, or a passage without the corpus. Matching to a known corpus is
validation/localization, not an analytical text decompressor.
