# Token candidates from checkpoint-boundary deltas

Date: 2026-09-09. This is a weight-only extraction attempt, with corpus use
deferred until candidates are frozen. It is not ordinary model inference,
raw-gradient recovery, or a claim that checkpoint-time order equals word order.

## New clue and fixed comparison

All 1,303 saved checkpoints, every ten steps from 10 to 13,030, are present.
Embedded directory timestamps reveal five gaps above 300 seconds, against a
median 213-second save interval. The earlier endpoints are 580, 1780, 4280,
7230, and 8160. They are **candidate restart boundaries**, not authenticated
restarts. Current/contemporaneous resume code restarts Adam moments and the
seeded dataset iterator; replay of initial windows is conditional on unchanged
seed/data/settings. No historical optimizer or sampler state was saved.

For each endpoint N, inspect all eight checkpoints N-30,N-20,...,N+40.
Three matched interval triplets are used:

- Candidate boundary: left N-10→N, center N→N+10, right N+10→N+20.
- Ordinary-before control: left N-30→N-20, center N-20→N-10, right N-10→N.
- Ordinary-after control: left N+10→N+20, center N+20→N+30, right N+30→N+40.

This uses 40 archived checkpoints / 35 consecutive ten-step differences,
selected using timestamps rather than token identities, losses, or outcomes.
Read archives in memory without extracting or modifying checkpoint files.
Validate every expected member name/type/size, gzip integrity, finite embedding
values, and original archive hashes before/after each read. Retain complete
member metadata and exact embedding-byte hashes. Logical vocabulary is 50,257;
exclude the 15 physical padding rows from all statistics and candidates.

## Activity and candidate rules

For logical embeddings E0,E1, define Delta=E1-E0 in FP64. Two fixed activity
measures are examined: raw row L2 norm and a geometry-adjusted row L2 norm.
For the latter, F=E0-mean_rows(E0), mu=mean_rows(Delta), and

    a = <F, Delta-mu>_F / ||F||_F^2
    R = Delta-mu-a*F.

Fit a=0 when F is zero. This subtracts common translation and fitted radial
motion only. It is NOT authenticated weight-decay correction, does not remove
general rotations, and can remove learned information. Raw activity is an
explicit comparison; neither activity is assumed to identify input gradients.

For each activity vector r, take log(max(r, median(r)*1e-12)/median(r)).
If the median is zero, use max(r); if the entire vector is zero, return zeros.
For each triplet form transient score = center_log - (left_log+right_log)/2.
Rank descending by transient score, breaking ties by ascending token ID.
Freeze the top 128 token IDs for every boundary, control, and activity variant.
These are unordered token-piece candidates, never a generated sentence.

Also form a shared score separately for boundary, ordinary-before, and
ordinary-after triplets: average the five within-vector percentile ranks,
using midranks for exact ties. Freeze each shared score's top 128 IDs.
Report the individual scores and how many of the five are positive for each
selected ID; do not discard failures or duplicates.

Baselines: for each N, freeze top 128 rows by the norm of E_N, plus their
shared average-percentile ranking. A single PCG64 seed 20260909 permutation of
all logical IDs supplies an identity-shuffled control for every ranked list.
The permutation changes only the ID labels, not scores or ranking positions.
No choice of K, geometry variant, averaging rule, or boundary is changed after
corpus verification. Retain all 35 full activity vectors, complete score
vectors, masks, source hashes, candidate lists, and output hashes.

## Verification and limits

Only after a frozen candidate manifest exists, validate the native corpus
export against the tokenizer and original bytes. Decode candidates individually
and measure corpus-token presence/frequency, separately in current prefix and
suffix, alongside endpoint-norm and shuffled-ID baselines. These measurements
test corpus vocabulary recovery, NOT exact membership in the unknown historical
ten-step batches. Do not present concatenated ranked token pieces as text.

A separately labeled conditional replay check may compare candidates against
the first ten batches under the source defaults: mt19937_64 seed 17, context
1024, sequence batch size 1; also the previously discussed batch size 10.
The native current corpus/split and current standard-library sampler are
assumptions, not authenticated historical facts. Compare each to 1,000 fixed
alternative seeds 10000..10999 with the same window count. Report overlap ranks
descriptively, not as proof of seed, training membership, or restart identity.
No search over further seeds, corpus versions, or batch sizes is permitted by
this protocol. The actual batch reconstruction question remains open if these
assumptions fail or if candidates are merely frequent corpus vocabulary.

Unit tests must cover safe archive reading/rejection, exact delta orientation,
translation/radial controls, finite/zero handling, interval selection, midranks,
tie rules, deterministic controls, and input immutability. No model forward,
backward, optimizer, or GPU work is part of this experiment. Sequence ordering
and longer-passage recovery require a further experiment, not a relabeling of
this bag-of-token result as a decompressor.
