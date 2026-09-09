# Token recovery from checkpoint-boundary deltas

The checkpoint history exposes a strong, specific token signal. A fixed
weight-only calculation selected 128 token IDs; **125 occur in the first ten
windows of the prescribed seed-17 replay**, compared with a mean of 5.984 and
a maximum of 38 among 1,000 alternative seeds. All 128 occur in its first
hundred windows, versus an alternative mean of 41.673 and maximum of 81.

These are **unordered token pieces recovered from weight-change scores**,
not reconstructed sentences. Their correspondence to historical training
batches remains conditional on the replay assumptions below. No model forward,
backward, optimizer step, or GPU calculation was used. The extraction never
opened the corpus or tokenizer; verification did so only after candidate freeze.

## What was extracted and from which weights

The [fixed protocol](RESTART_DELTA_PROTOCOL.md), committed as `257c2e4`, uses
the five candidate restart boundaries identified from checkpoint timestamps:
580, 1780, 4280, 7230, and 8160. The
[history inventory](CHECKPOINT_HISTORY_INVENTORY.md) explains why these are
candidates, not authenticated restarts. The extractor and safe archive reader
were committed as `caa5d72` before the real run.

For each boundary N, the run reads eight checkpoints N−30 through N+40,
ten steps apart: 40 archives and 35 consecutive differences in total. Each
archive is checked for the expected 100 weight files, exact names/types/sizes,
gzip completion and CRC, finite embedding values, and unchanged full archive
hashes before and after reading. No archive is extracted to disk.

The relevant parameter is the shared input embedding/LM-head matrix,
`weight_0.bin`, shape [50272,512], little-endian FP32. Logical token t occupies
the 512-float row at byte interval **[2048*t, 2048*(t+1))** inside that weight
file. The 15 unused padding rows are excluded from candidate statistics.
Thus the candidates have exact weight-row addresses across the checkpoints;
these addresses do not imply that each row privately contains a passage.

The simple calculation is:

1. Subtract adjacent embedding matrices in FP64: Delta = E_after − E_before.
2. Measure each row's L2 norm, either directly or after subtracting the shared
   mean update and a fitted uniform radial component along the centered
   earlier embeddings. The latter is the *adjusted* variant, not recovered
   AdamW gradients or authenticated weight-decay correction.
3. Normalize each activity vector by its median and take logarithms. Score
   a transient as center activity minus the average of its neighboring
   intervals' activities.
4. Select the 128 largest scores, with ascending token-ID ties. For shared
   lists, average the five within-boundary percentile midranks first.

The center intervals are N→N+10 for the candidate boundary, N−20→N−10 for
ordinary-before, and N+20→N+30 for ordinary-after. Each uses its own immediate
neighbors. The fixed comparisons include both raw and adjusted variants,
largest endpoint-row norms, and one fixed identity permutation for every
list. Every one of the 42 rankings and its shuffled counterpart is retained.

## Shared comparisons

Each list contains 128 distinct IDs. `Corpus` is how many occur anywhere in
the full current Shakespeare corpus. `10` and `100` are the overlaps with
the union of seed 17's first ten or hundred sampled windows. Each window
includes its input and target tokens, hence 1,025 token positions for context
1,024. Repeated occurrences and overlapping windows count only once per ID.
`Alt mean / max` uses exactly seeds 10000 through 10999 with the same number
of windows; all 1,000 individual counts are preserved in the evidence.

| Shared score | Labels | Corpus / 128 | 10-window overlap | Alt mean / max | 100-window overlap | Alt mean / max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Boundary, raw | Real | 128 | 9 | 5.322 / 31 | 111 | 42.279 / 63 |
| Boundary, raw | Shuffled | 47 | 1 | 3.353 / 8 | 14 | 13.887 / 22 |
| Boundary, adjusted | Real | 128 | **125** | **5.984 / 38** | **128** | **41.673 / 81** |
| Boundary, adjusted | Shuffled | 42 | 4 | 3.559 / 8 | 13 | 15.250 / 23 |
| Ordinary-before, raw | Real | 123 | 24 | 22.870 / 36 | 83 | 85.045 / 100 |
| Ordinary-before, raw | Shuffled | 50 | 3 | 2.972 / 9 | 16 | 16.105 / 24 |
| Ordinary-before, adjusted | Real | 67 | 11 | 12.912 / 24 | 49 | 50.126 / 60 |
| Ordinary-before, adjusted | Shuffled | 45 | 5 | 5.274 / 10 | 20 | 18.424 / 27 |
| Ordinary-after, raw | Real | 128 | 23 | 8.950 / 22 | 80 | 55.627 / 76 |
| Ordinary-after, raw | Shuffled | 49 | 5 | 5.034 / 13 | 15 | 18.403 / 29 |
| Ordinary-after, adjusted | Real | 128 | 0 | 1.804 / 25 | 118 | 15.761 / 45 |
| Ordinary-after, adjusted | Shuffled | 42 | 4 | 4.139 / 9 | 17 | 16.433 / 23 |
| Largest endpoint norm | Real | 0 | 0 | 0.000 / 0 | 0 | 0.000 / 0 |
| Largest endpoint norm | Shuffled | 49 | 4 | 3.032 / 10 | 18 | 17.223 / 27 |

The endpoint baseline means **largest row norm**, with its shared version
averaging the five percentile rankings. Its failure is not evidence that all
possible static norm statistics fail. Likewise, shuffled labels preserve
rank positions but not corpus frequencies; they are not calibrated nulls.

The adjusted ordinary-after signal is temporally different: none of its
shared candidates occur in the first ten replay windows, while 118 occur in
the first hundred. This fits a later-window interpretation, but the protocol
does not test exact later batches or establish their order. It must not be
relabelled as that stronger result. The raw variant has a different mixture
of effects; adjusting geometry was a prespecified comparison, not tuned after
observing these matches.

## Individual adjusted boundary comparisons

All five individual lists contain 128 IDs present in both the full corpus
and its current prefix. At each boundary, seed 17's overlap exceeds every
one of the 1,000 alternative seeds in both prescribed replay conditions.

| Boundary center interval | 10-window overlap / 128 | Alt mean / max | 100-window overlap / 128 | Alt mean / max |
| --- | ---: | ---: | ---: | ---: |
| 580→590 | 73 | 19.396 / 37 | 128 | 80.267 / 99 |
| 1780→1790 | 91 | 8.831 / 42 | 128 | 54.582 / 84 |
| 4280→4290 | 123 | 5.116 / 42 | 126 | 38.199 / 90 |
| 7230→7240 | 126 | 5.096 / 35 | 128 | 38.111 / 76 |
| 8160→8170 | 123 | 4.314 / 41 | 128 | 34.447 / 79 |

The shared adjusted candidates are not merely common punctuation or stopwords:
their prefix frequencies have median 6, mean 9.5, and maximum 65. Together
they account for only 1,216 of 1,650,781 prefix token occurrences (0.073662%).
Standalone pieces include ` shocks`, ` Comedy`, ` Nick`, ` enjoyed`,
` sources`, ` proofs`, ` lending`, and ` performing`. Spaces shown here are
part of the token byte spelling. The complete artifact preserves each piece
individually, its raw hexadecimal bytes, and its full/prefix/suffix frequency.
Their ranked order is not a sentence and is never presented as one.

## What this does and does not establish

This is a substantially stronger connection between weights and training
material than recovering generic corpus vocabulary. The token IDs were fixed
using checkpoint changes alone, and their uncommon pieces concentrate in a
specific, independently prescribed sequence of sampled windows. The signal
repeats across five separated checkpoint boundaries and weakens or moves in
the surrounding controls. It gives a usable analytical **token-bag extractor**
and an exact embedding-row-to-token map for this checkpoint comparison.

Historical interpretation still requires all of these assumptions: the
timestamp gaps correspond to actual restarts; the corpus and byte split are
unchanged; tokenization matches the validated native export; seed 17 was used;
context was 1,024; and the historical C++ sampler maps random numbers as the
current libstdc++ 13 implementation does. Ten windows correspond to ten steps
at sequence batch size one; a hundred correspond to ten steps at size ten.
Neither historical invocation flags nor sampler/optimizer states are saved.
The two available older logs describe different token split counts and are
not used as provenance for these checkpoints.

The current full corpus contains 1,835,163 tokens and 18,880 distinct IDs. Its
byte split at 4,892,836 yields 1,650,781 prefix and 184,382 suffix tokens,
with 18,380 and 8,412 distinct IDs, respectively. These are present-day
verification quantities, not authenticated historical training membership.

Alternative-seed ranks are descriptive comparisons, not p-values. A rank of
one alone is especially misleading for the endpoint baseline: its overlap
is zero for seed 17 **and all 1,000 alternatives**, so all are tied. The report
retains strict-greater, equal, and greater-or-equal counts to make such cases
explicit. The adjusted shared boundary's 125 and 128 overlaps, by contrast,
are strictly above every alternative in their respective conditions.

No sequence order has been recovered, no whole passage has been analytically
decoded, and the candidate lists do not reconstruct all tokens in the sampled
windows. No claim is made that the located rows independently store their
associated text. Extending this result to exact windows, ordering, and an
understandable passage decompressor remains an open task.

## Integrity, numerical caveats, and verification status

The extraction completed in 64.54 seconds and froze its candidates by
2026-09-09 04:50:48 UTC. A separate verification-start record was written at
04:56:46 UTC after checking every frozen source/output identity and before
opening any corpus, tokenizer, or native-token file. Verification completed
at 04:56:48 UTC. It validated the supplied token/offset export against every
original corpus byte, authenticated candidate membership/order/shuffles against
the complete saved scores, and repeated source/input/output hashes afterward.
The verifier does not itself perform another full rehash of all 40 archives.
The separate [final archive audit](restart_delta_final_archive_check.json),
completed at 05:05:36 UTC after verification, confirms all 40 full archive
SHA256 hashes still match the frozen identities. This final pass hashes
complete compressed bytes; it does not rerun member parsing or numerical
activity calculations.

The numerical candidate audit retained nine initial discrepancy records,
rather than discarding them. Its
[tie supplement](restart_delta_independent_ties.json) explains eight internal
rank-position changes across three shared control lists under independently
evaluated FP64 sums: `shared_ordinary_after_raw`,
`shared_ordinary_before_adjusted`, and `shared_ordinary_after_adjusted`.
The maximum score difference is 2.22e−16. Exact mathematical midrank sums
would require integer/rational tie handling; the frozen code uses FP64.
All 42 top-128 **sets** remain identical, all rankings follow their stored
FP64 scores, and boundary-list orders are unchanged. The apparent component
comparison failures were caused by comparing columns at the audit's reordered
IDs; comparisons at the frozen IDs agree. Therefore these ordering caveats
do not change the unordered frequency or overlap measurements. The original
[audit](restart_delta_independent_check.json) and its source are preserved,
and frozen candidates have not been changed.

The separately implemented
[replay/count audit](restart_delta_replay_independent_check.json) also passes.
A scalar Python implementation of MT19937-64 and the current libstdc++ integer
mapping reproduces all 1,001 rows of 100 sampler draws exactly. Python set
unions/intersections reproduce all 168 overlap lists (42 rankings × two label
variants × two window counts), including every one of the 1,000 alternative
counts in each list. All ranked full/prefix/suffix corpus frequencies also
agree exactly. This is an independent calculation on the same data and
conditional assumptions, not independent historical provenance.

The full Python suite for this milestone has passed 448 tests, including nine
new verifier tests and 22 independent replay-audit tests;
the native sampler also
has seven tests including an alternate-language MT19937-64/current-libstdc++
mapping check. Passing tests do not authenticate historical batch membership.
An initial custom filtered-suite runner retained references to completed test
fixtures and exhausted its file-descriptor limit. Releasing the original suite
before executing the filtered suite fixed that harness issue; the 448-test
rerun passed. No production code was changed to suppress those failures.

## Evidence and reproduction

These six repository files are byte-exact copies of the corresponding
artifacts under `/tmp/pluto-restart-delta.fZs3LQ/run/`, not rerun or manually
rewritten results:

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| [restart_delta_frozen.json](restart_delta_frozen.json) | 2,751,277 | `a16d908bf7b496d6fbabf601d32d93e1e2bf6af5b83e2a428eec2825a59211ca` |
| [restart_delta_candidates.json](restart_delta_candidates.json) | 443,239 | `2aeee3c0b5054c2bf6a63df38a6dcac20b064351839434bc0c48301d05af18e3` |
| [restart_delta_verification_start.json](restart_delta_verification_start.json) | 12,354 | `4cc9467046ad94aa6981b56b885692a65e12357e72cc2a63ae5534319953274e` |
| [restart_delta_verification.json](restart_delta_verification.json) | 7,845,298 | `1dcb37992b2e821184e066bc3c49bb3c9b547b410b355a4d9b227c3aaec4b536` |
| [restart_delta_replay_independent_check.json](restart_delta_replay_independent_check.json) | 35,922 | `0b874e92f04764babe1960835fd22ff9b678617708a75818dc771a618f4e0e14` |
| [restart_delta_final_archive_check.json](restart_delta_final_archive_check.json) | 8,131 | `dab0ae8cc3ba3c494a42dcaf4aaef42825f0d1b81ea87c3f829867e918829526` |

The replay-audit copy is the validated rerun
`replay_independent_check_validated.json`, completed after strengthening its
input guards. The original earlier audit remains unchanged in the run
directory; the validated rerun reproduces the same counts and sampler draws.

The frozen manifest records every original archive hash, member metadata,
embedding-byte hash, extraction source, activity array, and complete-score
artifact. The verification report retains all 84 real/shuffled comparisons,
all alternative-seed overlap counts, and every sampled start. Original
absolute paths remain in copied manifests so the copying does not masquerade
as a fresh freeze. Additional original artifacts:

- `plan.json`: 4,421 bytes, SHA256
  `eb8bf67c146b91eda962688234dc23787307570a41bcce4038429096878db788`.
- `scores.npz`: 17,300,392 bytes, SHA256
  `c030537959499df3e85f834a3cef98aa458a5aeefd1c3f234634e494ba2c3e6d`.
- `/tmp/pluto-replay-sampler`: 71,992 bytes, SHA256
  `028cecb76ac9d719a15dbaf391df998aa257d84a9a74216ec3c0a06893defa65`;
  C++20, 64-bit `size_t`, libstdc++ 13, library date 20240904.

Use a fresh output directory. Source or binary changes require a new run;
the verifier intentionally rejects changed frozen source identities.

```sh
g++ -O2 -std=c++20 -Wall -Wextra -Werror -pedantic \
  scripts/weight_analysis/replay_sampler.cc -o /tmp/new-replay-sampler
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/ubuntu/.venv/bin/python \
  -m unittest discover -s scripts/weight_analysis -t . -p '*_test.py' -q

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/ubuntu/.venv/bin/python \
  -m scripts.weight_analysis.restart_delta \
  --checkpoint-root /home/ubuntu/checkpoints/shakespeare \
  --output-dir /tmp/new-restart-delta \
  --protocol research/weight_memorization/RESTART_DELTA_PROTOCOL.md \
  --inventory research/weight_memorization/checkpoint_history_inventory.json

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/ubuntu/.venv/bin/python \
  -m scripts.weight_analysis.restart_delta_verify \
  --frozen /tmp/new-restart-delta/frozen.json \
  --corpus testdata/shakespeare.txt \
  --native-tokens /tmp/pluto-native-corpus.KMY7AZ/shakespeare.tokens.bin \
  --native-offsets /tmp/pluto-native-corpus.KMY7AZ/shakespeare.tokens.bin.offsets.bin \
  --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 \
  --sampler /tmp/new-replay-sampler \
  --output /tmp/new-restart-delta/verification.json

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/ubuntu/.venv/bin/python \
  -m scripts.weight_analysis.restart_delta_replay_audit \
  --report /tmp/new-restart-delta/verification.json \
  --output /tmp/new-restart-delta/replay_independent_check.json
```

The extraction command does not read corpus contents. The verification command
reads them only after authenticating the candidate freeze and writing its own
start record; the independent audit then checks the measurements. No further
seed, context, batch-size, ranking, or geometry search is part of this result.
