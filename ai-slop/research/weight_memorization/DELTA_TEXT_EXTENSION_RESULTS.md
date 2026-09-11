# Checkpoint deltas identify source passages; weight-only ordering remains separate

The checkpoint changes contain a strong, localized training-text signal. A
corpus-assisted search using the **previously frozen** 128 delta-selected token
IDs identifies five passages overlapping the first five conditionally replayed
training windows by 84–98% each. This is not de novo sentence decoding: the
weights supply the selection clues, and the supplied corpus supplies the words
and their order.

This extension follows the [fixed protocol](DELTA_TEXT_EXTENSION_PROTOCOL.md).
The earlier [restart-token result](RESTART_DELTA_RESULTS.md) and the failed
[later-interval follow-through](DELTA_FOLLOWTHROUGH_RESULTS.md) remain unchanged.
The seed list and checkpoint intervals were informed by those previous results;
this is not a new independently selected checkpoint history.

## Available history and what was actually read

A fresh inventory confirms all 1,303 checkpoints, every tenth step from 10 to
13,030. There are 1,298 compressed archives totaling 248,619,816,211 bytes and
five unpacked directories. No one-update pair, step-zero checkpoint, optimizer
moments, sampler state, or authenticated historical invocation was found.

The new directional experiment reads ten checkpoints surrounding the same five
candidate restarts: 580→590, 1780→1790, 4280→4290, 7230→7240, and 8160→8170.
It does **not** claim to have numerically analyzed every matrix at all 1,303
steps. The prior token-bag calculation used 40 neighboring checkpoints, and
the negative later-phase test used 20 different archives.

Even if these timestamp gaps are actual restarts, each saved difference mixes
ten AdamW updates. An endpoint difference is not a raw gradient. The embedding
is also tied to the output head, so its changes mix input-side and output-side
learning. Historical batch membership remains conditional on the documented
corpus, split, tokenizer, seed, sampler, batch size, and restart assumptions.

## New positive result: passage localization

The primary bag is exactly `shared_boundary_adjusted` from the earlier frozen
experiment: 128 IDs selected from weight changes without consulting the corpus.
For localization, every legal 1,025-token window of the full native corpus is
scored by the number of **distinct** IDs from that bag. Repetition earns no
extra credit. Greedy descending-score selection keeps up to 100 nonoverlapping
windows, breaking ties by earliest start. Full score arrays and equal-score
plateaus were frozen before sampler replay was accessed.

The five highest-ranked nonoverlapping selections are:

| Rank | Source passage | Corpus token interval | Distinct clues | Replay draw | Overlapping token positions / 1,025 |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | [All's Well That Ends Well: Helena and the King](/home/ubuntu/code/pluto/testdata/shakespeare.txt:3432) | [42,717, 43,742) | 32 | 2 | 972 (94.8%) |
| 2 | [Henry VI, Part 2: Jack Cade / Blackheath scene](/home/ubuntu/code/pluto/testdata/shakespeare.txt:44781) | [654,067, 655,092) | 25 | 5 | 1,006 (98.1%) |
| 3 | [The Merry Wives of Windsor: Falstaff and the fairies](/home/ubuntu/code/pluto/testdata/shakespeare.txt:78391) | [1,149,956, 1,150,981) | 24 | 1 | 974 (95.0%) |
| 4 | [The Merry Wives of Windsor: Falstaff and Slender](/home/ubuntu/code/pluto/testdata/shakespeare.txt:75593) | [1,106,179, 1,107,204) | 23 | 4 | 864 (84.3%) |
| 5 | [A Midsummer Night's Dream: the actors plan their play](/home/ubuntu/code/pluto/testdata/shakespeare.txt:78923) | [1,157,602, 1,158,627) | 21 | 3 | 996 (97.2%) |

The retrieved order is **score order**, not historical training order. Replay
was used only afterward to label the matching draws. In each case, the actual
replayed start falls within the selected equal-score plateau; the token bag
therefore does not identify the exact start. Plateau widths are respectively
149, 22, 95, 178, and 30 possible start positions. The table highlights the
visible high-score cluster; the prespecified retrieval budgets remain 10 and
100, not a retrospectively substituted top-five metric.

The play-planning passage includes “The most Lamentable Comedy” and “Nick Bottom,
the weaver.” Its selected clues include ` Comedy`, ` Nick`, ` performing`,
` Cruel`, and ` roaring`. These are corpus-supplied snippets, **not outputs of
the weight-only ordering extractor**.

The token-row addresses behind three of those clues are concrete:

| Token piece | ID | Byte interval inside each checkpoint's `weight_0.bin` |
| --- | ---: | --- |
| ` Comedy` | 22,329 | [45,729,792, 45,731,840) |
| ` Nick` | 8,047 | [16,480,256, 16,482,304) |
| ` performing` | 9,489 | [19,433,472, 19,435,520) |

Each row contains 512 FP32 values; row t occupies [2048t, 2048(t+1)). The
selection used changes in these rows relative to other rows and neighboring
intervals. These are participating parameter addresses, not independent slots
containing the quoted sentences.

## Controls and limits of the localization result

The fifth window scores 21; the sixth scores only 5. The complete top-ten
retrieval overlaps 4,965 of the first ten replayed windows' 10,250 token
positions, giving **48.44% precision and recall**. For the same fixed retrieval,
1,000 alternative replay seeds give mean overlap 57.629 and maximum 1,494.
These are descriptive comparisons, not calibrated significance tests.

All 100 frequency-range-matched control bags were searched using the same
procedure. Their highest window scores range from 5 to 8 (mean 5.78), versus
32 for the primary bag. Their top-ten overlap with the actual conditional
seed-17 replay averages 83.37 tokens and reaches at most 920, versus 4,965.
These controls preserve floor(log2(corpus frequency)) buckets, not exact token
frequencies. Corpus statistics enter this control stage explicitly.

All 18 fixed real/shuffled comparison lists and all 100 frequency-range
controls are retained. For top-ten retrieval versus first-ten replay:

- Shared adjusted restart list: 4,965 overlapping positions.
- Shared raw restart list: 0; its shuffled-label counterpart: 0.
- Shared adjusted restart list with shuffled labels: 0.
- Ordinary-before real lists, both geometries: 0.
- Ordinary-after real lists: raw 917, adjusted 0.
- Later-interval real lists, both geometries: 0 against this **early** replay.
  This last comparison is not a fresh test of their proper later phase; that
  negative phase-specific result is documented in the earlier report.

Expanding retrieval to 100 windows increases first-ten replay recall to
56.01%, but precision falls to 5.60%. Against the first hundred replayed
windows, top-100 retrieval overlaps 6,671 positions versus alternative mean
5,822.786 and maximum 14,158. For this **primary restart bag**, the evidence is
concentrated in a small early set, not broad recovery of all training batches.

The neighboring ordinary-after bags retain another positive result. Against
the first hundred replay windows, the adjusted bag's top-ten retrieval overlaps
**9,080 positions** (88.59% retrieval precision), versus alternative-seed mean
616.054 and maximum 2,990. Its top-100 retrieval overlaps 15,834 positions,
versus mean 6,091.851 and maximum 13,032. The raw bag's corresponding overlaps
are 4,826 and 13,591. Thus the neighboring deltas also localize later material
within the early replay, despite the adjusted bag's zero overlap with its first
ten windows.

The adjusted bag's top-ten selections map most closely to draws 82, 94, 90,
93, 92, 95, 86, 100, 89, and 98 among the examined first hundred. This raises
a new **exploratory** phase-alignment hypothesis: a historical batch size of
four sequences would put the N+20→N+30 interval at draws 81–120. A separate
[fixed follow-up protocol](DELTA_BATCH_FOUR_PROTOCOL.md) tests that hypothesis
without altering any candidate bag or earlier negative result. These observed
draw matches are not authenticated historical settings.

That [follow-up is now complete](DELTA_BATCH_FOUR_RESULTS.md): the early
ordinary-after adjusted bag matches **128/128** IDs in draws 81–120, versus
alternative mean 6.711 and maximum 32. But both later bags match **0/128**
in their proposed draws 401–440. This supports early phase alignment without
explaining or resolving the separate later-checkpoint failure.

## Weight-only ordering: a negative result

The separate extractor averaged directional cosine associations across the
five restart deltas. It compared each centered pre-update embedding row with
each translation/radial-adjusted update row. Starting from each of the same
128 token IDs, it greedily selected three predecessors and four successors,
without reading the corpus or running a model. All 512 candidates were frozen
before verification, including transpose, static-embedding, and shuffled-delta
controls.

| Arm | Eight-token candidates | Candidates with any matching pair | Distinct selected longest pairs | Seed predecessor / successor hits | Seed triples | Any match ≥3 tokens |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Directional delta | 128 | 111 | 2 | 0 / 1 | 0 | 0 |
| Transposed direction | 128 | 2 | 1 | 0 / 0 | 0 | 0 |
| Static embeddings | 128 | 0 | 0 | 0 / 0 | 0 | 0 |
| Permuted delta labels | 128 | 0 | 0 | 0 / 0 | 0 | 0 |

Of the 111 apparent directional pair successes, 110 select the same longest
fragment, ` Away\n`; the remaining one selects ` shocks\n`. The sole correct
seed-successor prediction is ` shocks` followed by a newline. The transpose
control repeats ` Away at` twice. No three-token substring occurs anywhere in
any intact-arm candidate, let alone an eight-token passage. Global token-label
shuffle checks are also retained; they do not change this conclusion.

Thus this simple association **fails to recover word order**. It is not rescued
by the separate positive corpus lookup. The prior trajectory audit also showed
that rigid embedding rotations can produce directional scores; the tested
radial adjustment does not remove this confound. Unit-normalizing tiny update
rows can amplify noise, so original row norms are preserved rather than
silently thresholded after seeing the outcome.

The weight-only run took 23.92 seconds. A separate auditor reread all ten
original archives, checked all 512 paths and 3,584 edges, their physical row
addresses and norms, and all-vocabulary greedy choices. Every selected ID
agreed; the largest scalar-score difference was 1.249e-15. All archives and
authenticated source/output files remained unchanged.

## Verification and evidence

The localization audit independently reproduced all 216,428,402 integer scores
across 118 lists, all 11,700 selected windows, 44,710 supporting occurrences and
byte ranges, every greedy choice and plateau, and all 100 frequency-range
control permutations. It also checked all 11,800 replay score/rank/tie entries
and 128 literal set-based overlap cases. All checks passed.

A separate corpus auditor checked all 36,864 contiguous subwindows of the
512 ordering candidates and their 512 shuffled-label controls, lengths one
through eight. All reported counts, longest matches, seed-neighbor results,
and byte spans agreed exactly. It independently validated every native token's
bytes against the source corpus.

The complete Python analysis suite passes **599 tests**, including 37 new
extraction/localization/verifier tests. Their
synthetic checks include scalar/dense mathematical oracles, zero and tiny rows,
rotation counterexamples, token-order orientation, exact ties, repeated-token
counting, nonoverlap boundaries, byte spans, and changed-input rejection.

The [preserved evidence archive](delta_text_extension_evidence.tar.gz) contains
both complete runs, full controls and score arrays, all three independent
auditors/results, and the eight new diagnostic/test source files plus protocol.
Every one of its 155 regular members was compared byte-for-byte with the
original; gzip integrity also passed. The original checkpoint payloads, corpus,
native export, and older upstream experiment arrays are not duplicated in this
archive; their identities and prerequisite paths are retained in the records.

- Compressed size: 14,549,764 bytes; member payload: 99,855,200 bytes.
- SHA256: `cbfbe03d628b076f649d696f216f8be60365a654fd360c24856599bad6c098c4`.

## Reproduction

New diagnostics do not modify checkpoints or production model code, and require
no GPU execution. The corpus-assisted run is preserved at
`/tmp/pluto-delta-localization.37cC693x/run/`; it contains all 118 score arrays,
supporting token positions and byte spans, frozen sources/input identities,
all 1,001 replay seed rows, and the complete overlap results.

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/ubuntu/.venv/bin/python \
  -m scripts.weight_analysis.delta_localization \
  --restart-frozen /tmp/pluto-restart-delta.fZs3LQ/run/frozen.json \
  --followthrough-frozen /tmp/pluto-delta-followthrough.VjsYYn/run/frozen.json \
  --corpus testdata/shakespeare.txt \
  --native-tokens /tmp/pluto-native-corpus.KMY7AZ/shakespeare.tokens.bin \
  --native-offsets /tmp/pluto-native-corpus.KMY7AZ/shakespeare.tokens.bin.offsets.bin \
  --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 \
  --protocol research/weight_memorization/DELTA_TEXT_EXTENSION_PROTOCOL.md \
  --output-dir /tmp/new_delta_localization

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/ubuntu/.venv/bin/python \
  -m scripts.weight_analysis.delta_localization_verify \
  --localization /tmp/new_delta_localization/localization.json \
  --output /tmp/new_delta_localization/replay_verification.json
```

Use new output paths; the tools refuse to overwrite evidence.

The weight-only run and its verification are at
`/tmp/pluto-delta-ordering.4kpaZx/run/`. Its parent also contains the recorded
command, stdout, completion record, and independent original-byte audit.

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/ubuntu/.venv/bin/python \
  -m scripts.weight_analysis.delta_ordering \
  --checkpoint-root /home/ubuntu/checkpoints/shakespeare \
  --seed-candidates research/weight_memorization/restart_delta_candidates.json \
  --protocol research/weight_memorization/DELTA_TEXT_EXTENSION_PROTOCOL.md \
  --output-dir /tmp/new_delta_ordering

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/ubuntu/.venv/bin/python \
  -m scripts.weight_analysis.delta_ordering_verify \
  --frozen /tmp/new_delta_ordering/frozen.json \
  --corpus testdata/shakespeare.txt \
  --tokens /tmp/pluto-native-corpus.KMY7AZ/shakespeare.tokens.bin \
  --offsets /tmp/pluto-native-corpus.KMY7AZ/shakespeare.tokens.bin.offsets.bin \
  --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 \
  --output /tmp/new_delta_ordering/verification.json
```

## Bottom line

The checkpoint history yields rare-token clues strong enough to locate several
specific passages in the known corpus. It does **not yet** supply an analytical
decoder that reconstructs their ordered text from weights alone. The broader
weight-to-text goal remains unresolved; no checkpoint or production code was
changed, and no commit was made for this extension.
