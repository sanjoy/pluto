# Checkpoint-history inventory and candidate restarts

Date: 2026-09-09. This is a read-only history/provenance inventory, not a
text-extraction result. It extends the earlier
[checkpoint-difference feasibility audit](CHECKPOINT_DELTA_FEASIBILITY.md) by
inspecting archive metadata across the full history.

## Available history

`/home/ubuntu/checkpoints/shakespeare` contains all 1,303 multiples of ten from
`step_10` through `step_13030`, with no missing steps or extra root entries:

- 1,298 gzip tar archives, through `step_12980.tar.gz`.
- Five unpacked directories, `step_12990` through `step_13030`.
- Archives occupy 248,619,816,211 compressed bytes in total.

Four complete logical member listings, at steps 10, 20, 1,000, and 5,000, each
contain the top-level `step_N` directory and exactly 100 regular files named
`weight_0.bin` through `weight_99.bin`. Every file size matches the existing
GPT-2 layout manifest; the total payload is 205,934,592 bytes. There are no
optimizer-state or metadata files in these four listings. The five unpacked
directories likewise pass all expected filename and size checks.

`weight_0.bin`, the token embedding/tied output table, is the first regular
member in each sampled archive and occupies 102,957,056 bytes. Reading selected
members does not require extracting the full checkpoint, although gzip still
requires decompression of preceding bytes. A complete member scan took
0.84–0.85 seconds per sampled archive on this machine. Reading the first
logical member and a 64-KiB compressed prefix from every archive took 0.24
seconds in the preserved inventory run.

## Five candidate restart boundaries

Outer `.tar.gz` filesystem modification times describe later archive creation
or modification and must not be confused with checkpoint creation. For
archived checkpoints, this inventory instead uses the embedded top-level
directory member's modification time. For the five unpacked checkpoints, it
uses the filesystem directory modification time. Both remain mutable metadata,
not authenticated training logs.

All 1,302 adjacent timestamp differences are positive. Their median is
213.214 seconds per ten steps. Exactly five exceed 300 seconds:

| Step interval | Gap (seconds) | Earlier UTC | Later UTC |
| --- | ---: | --- | --- |
| 580 → 590 | 54,110.059 | Sept 1 04:11:31 | Sept 1 19:13:21 |
| 1,780 → 1,790 | 12,412.356 | Sept 2 02:16:16 | Sept 2 05:43:09 |
| 4,280 → 4,290 | 9,655.020 | Sept 2 20:28:02 | Sept 2 23:08:57 |
| 7,230 → 7,240 | 779.128 | Sept 3 16:33:45 | Sept 3 16:46:44 |
| 8,160 → 8,170 | 648.950 | Sept 3 22:13:39 | Sept 3 22:24:28 |

These are **candidate restart boundaries**, not proven restarts. A paused
process, unrelated machine activity, or modified timestamps could also produce
gaps. The threshold separates five conspicuous outliers from the other
intervals, whose maximum is below 214 seconds; it was chosen after inspecting
timestamps, not by looking at token identities or corpus matches.

Why this is actionable: the repository's resume implementation loads weights
and creates a fresh optimizer and dataset iterator. The Adam moments are zero
at creation. `Train` resets the iterator before taking updates; the iterator
reseeds `std::mt19937_64` and samples uniformly among valid contiguous sequence
starts. If a run actually restarted and retained the same seed, corpus,
context length, and batch settings, its early random windows would repeat.
Consequently, the five first-post-gap deltas might share a token-row signature
from repeated windows. This is a conditional hypothesis, not evidence that
specific text has been recovered.

Useful comparisons are each gap delta against both neighboring ordinary
deltas, the six early intervals 10→20 through 60→70, middle controls
1,000→1,010 and 5,000→5,010, and late control 13,020→13,030. The preserved JSON
lists all five matched four-checkpoint neighborhoods. A subsequent experiment
should freeze weight-derived token rankings before consulting corpus text and
should retain negative controls for generic optimizer-restart effects.

## Historical settings and logs

The earliest embedded checkpoint timestamp is September 1 at 00:48:59 UTC.
Commit `b1f6e87530c5d6b604a667592cc2ecb938f0df91`, dated August 31 at 21:45:47
UTC, already has the same random-window sampling and optimizer design. Its
source defaults are seed 17, one 1,024-token sequence per batch, learning rate
`3e-4`, Adam betas `0.9` and `0.95`, epsilon `1e-8`, and weight decay `0.1`.
These are **source defaults, not authenticated historical invocation flags**.

Resume support commit `28b25892c74acad228bb380f89a2cadcb3ab7c86` was made during
the 580→590 gap. Unlimited-training commit
`10a2e21a004f2a060e067e42a6927c8d9ecf36c8` was made during the 1,780→1,790 gap.
This temporal coincidence is circumstantial: source history does not prove
which executable or local changes produced checkpoint bytes.

Two available logs, `/home/ubuntu/train.log.0` and
`/home/ubuntu/train.log.1`, have August 31 modification times preceding this
checkpoint history. Both report 1,651,646 training tokens and 183,517 test
tokens, unlike the current native byte-split counts of 1,650,781 and 184,382.
Neither includes checkpoint writes, resumes, batch settings, seed, or learning
rate. They appear to describe earlier runs and are not treated as provenance
for these checkpoints. Their exact hashes and relevant lines are preserved in
the evidence JSON.

## Evidence and integrity scope

[checkpoint_history_inventory.json](checkpoint_history_inventory.json)
preserves all 1,303 checkpoint records, all 1,302 adjacent intervals, all five
outliers, four complete archive member listings, current source hashes,
historical commit identities, and the two log records. It was generated
exclusively at `/tmp/pluto-checkpoint-history-inventory.json` and copied
byte-for-byte into this repository:

- Size: 1,753,596 bytes.
- SHA256: `b6b449127c760bab21a2bf088cde8128e222a056644119b204ae927f6c2905dc`.

All top-level checkpoint filesystem stat records and all archive first-64-KiB
compressed-prefix hashes were checked before and after the inventory. They
were unchanged. A prefix hash is **not** a full archive hash, decompressed
header hash, or weight-content hash. The four complete archive scans validate
logical layout but do not parse tensor values or record payload hashes. The
five unpacked directories are checked by names and sizes, not by rehashing all
weights. No checkpoint was extracted or modified, no model was run, and no
corpus/tokenizer contents were read for this inventory.

This narrows the next delta experiment toward potentially repeated training
windows. It does not make AdamW endpoint deltas into raw gradients, identify
the training sequence, or resolve the analytical-decompression goal.
