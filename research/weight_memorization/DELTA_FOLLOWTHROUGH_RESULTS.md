# Later-batch follow-through: the strong token signal did not generalize

The fixed checkpoint-delta method **did not reproduce its strong restart-boundary
result one hundred steps later**. Its shared adjusted list contains only 5 of
128 IDs from the predicted seed-17 batch-size-one windows, versus an alternative
mean of 3.154 and maximum of 17. For batch size ten, the overlap is 34/128,
versus a mean of 24.398 and maximum of 50. This is far weaker than the earlier
125/128 versus 5.984/max 38 result at candidate restart boundaries.

Some individual lists show modest enrichment, retained below, but the prescribed
later-phase test does not establish a reliable arbitrary-interval training-token
extractor. No intervals, seeds, phases, candidate counts, or geometry settings
were changed after seeing these outcomes. No ordered text was reconstructed.

## Fixed prediction and extraction

This experiment was designed after observing the
[restart-boundary result](RESTART_DELTA_RESULTS.md). Its
[protocol](DELTA_FOLLOWTHROUGH_PROTOCOL.md) was committed as `7412404`; the
extractor and tests were committed as `d72b4ba` before the new weight reads.
It is a prospective checkpoint-interval test, not independently held-out
corpus data.

For each earlier candidate restart boundary N = 580, 1780, 4280, 7230, 8160,
the extractor reads N+90, N+100, N+110, N+120. These 20 archives do not overlap
the preceding 40-archive experiment. It computes the same raw and
translation/radial-adjusted logical embedding-row activities, median-normalized
logarithms, and center-minus-neighbor-average transient, now for the interval
**N+100→N+110**. The five individual lists and shared mean-percentile list per
variant give 12 real lists plus 12 fixed shuffled-ID controls, each of size 128.
No model forward, backward, optimizer, or GPU work is involved.

The same exact weight addresses apply: token t is row t of the input-embedding/
tied-LM-head `weight_0.bin`, bytes [2048*t, 2048*(t+1)) in each checkpoint.
Its physical shape is [50272,512] FP32; the 15 padding rows are excluded.
These are addresses used for the analytical score, not private passage-storage
locations.

Conditional on an actual restart at N, unchanged source settings, and one
sequence per step, the target is draws 101–110 after resetting seed 17. The
separate batch-size-ten condition predicts draws 1001–1100. The fixed replay
phases, in zero-based half-open draw indices, are:

| Sequence batch size | Predicted | Previous ten steps | Next ten steps | First ten steps after reset |
| --- | --- | --- | --- | --- |
| 1 | [100,110) | [90,100) | [110,120) | [0,10) |
| 10 | [1000,1100) | [900,1000) | [1100,1200) | [0,100) |

Every window contains the union of its input and target tokens, 1,025 positions
at context length 1,024. Each ID is counted once per union, regardless of
repetition or overlap. Alternative seeds 10000–10999 are evaluated only at
the prescribed predicted phase, with all 1,000 retained. Previous, next,
and reset-phase controls use seed 17. Ranks and alternative distributions
are descriptive, not calibrated p-values.

## Shared-list outcomes

All overlaps are out of 128. `Corpus` is the number of selected IDs found
anywhere in the full current corpus. `Alt mean / max` concerns the predicted
phase only; the other three phase columns are seed-17 controls.

### Sequence batch size 1

| Shared score | Labels | Corpus | Predicted | Previous | Next | Reset | Alt mean / max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | Real | 128 | 5 | 5 | 4 | 10 | 5.092 / 32 |
| Raw | Shuffled | 45 | 2 | 1 | 3 | 4 | 4.277 / 10 |
| Adjusted | Real | 128 | **5** | 2 | 0 | 1 | **3.154 / 17** |
| Adjusted | Shuffled | 57 | 6 | 6 | 12 | 9 | 7.467 / 16 |

### Sequence batch size 10

| Shared score | Labels | Corpus | Predicted | Previous | Next | Reset | Alt mean / max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | Real | 128 | 50 | 38 | 35 | 44 | 40.424 / 66 |
| Raw | Shuffled | 45 | 19 | 14 | 20 | 17 | 16.044 / 24 |
| Adjusted | Real | 128 | **34** | 19 | 27 | 24 | **24.398 / 50** |
| Adjusted | Shuffled | 57 | 27 | 22 | 25 | 27 | 24.440 / 36 |

For the shared adjusted list, 209 alternative seeds have overlap at least five
in the batch-one condition; 97 have overlap at least 34 in the batch-ten
condition. Seed 17's respective strict-greater ranks are 133 and 80 among
1,001 seeds. Unlike the preceding boundary experiment, it is not uniquely
separated from all alternatives. The strong difference between corpus presence
and exact-phase overlap also matters: recovering Shakespeare-associated
vocabulary does not by itself recover a particular training batch.

## All individual real-list outcomes

Every individual real list has all 128 IDs present in the full current corpus.
Both variants and all five boundaries are retained; all corresponding shuffled
lists and per-ID phase membership bits are preserved in the complete JSON.

### Sequence batch size 1

| N; scored interval | Variant | Predicted | Previous | Next | Reset | Alt mean / max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 580; 680→690 | Raw | 10 | 3 | 7 | 5 | 7.958 / 33 |
| 580; 680→690 | Adjusted | 5 | 3 | 3 | 2 | 5.160 / 20 |
| 1780; 1880→1890 | Raw | 6 | 8 | 3 | 9 | 5.123 / 29 |
| 1780; 1880→1890 | Adjusted | 8 | 4 | 3 | 7 | 4.442 / 19 |
| 4280; 4380→4390 | Raw | 8 | 5 | 3 | 12 | 5.037 / 28 |
| 4280; 4380→4390 | Adjusted | 6 | 2 | 1 | 3 | 3.024 / 16 |
| 7230; 7330→7340 | Raw | 12 | 6 | 3 | 16 | 4.920 / 29 |
| 7230; 7330→7340 | Adjusted | 4 | 0 | 0 | 1 | 2.479 / 16 |
| 8160; 8260→8270 | Raw | 13 | 6 | 6 | 11 | 6.346 / 28 |
| 8160; 8260→8270 | Adjusted | 1 | 0 | 0 | 1 | 2.203 / 14 |

### Sequence batch size 10

| N; scored interval | Variant | Predicted | Previous | Next | Reset | Alt mean / max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 580; 680→690 | Raw | 55 | 44 | 46 | 32 | 47.891 / 75 |
| 580; 680→690 | Adjusted | 37 | 25 | 38 | 32 | 31.721 / 51 |
| 1780; 1880→1890 | Raw | 49 | 34 | 37 | 53 | 40.228 / 67 |
| 1780; 1880→1890 | Adjusted | 47 | 31 | 37 | 41 | 32.814 / 53 |
| 4280; 4380→4390 | Raw | 48 | 41 | 34 | 59 | 39.538 / 64 |
| 4280; 4380→4390 | Adjusted | 26 | 22 | 31 | 18 | 23.860 / 48 |
| 7230; 7330→7340 | Raw | 49 | 43 | 33 | 61 | 39.553 / 64 |
| 7230; 7330→7340 | Adjusted | 24 | 17 | 30 | 11 | 21.234 / 51 |
| 8160; 8260→8270 | Raw | 59 | 43 | 42 | 58 | 47.090 / 73 |
| 8160; 8260→8270 | Adjusted | 29 | 12 | 23 | 15 | 18.954 / 46 |

The adjusted 1880→1890 list in batch-ten conditions is a comparatively stronger
individual result: 47 versus mean 32.814/max 53, rank 14, with 17 alternatives
at least as high. That partial enrichment is not discarded, but neither is it
used to choose a new boundary or declare the shared prospective prediction
successful. Every individual target overlap is below an alternative maximum.

## Interpretation and checked alternatives

The original near-reset token signal remains a real, independently checked
observation. This follow-through limits its scope: **the same simple transient
score is not currently a robust decoder of arbitrary later ten-step batches**.
Both geometry variants and both fixed batch-size interpretations show much
weaker localization here. The result does not prove that later deltas contain
no recoverable information, nor does it invalidate the earlier frozen result.

Historical corpus identity, token split, seed, C++ sampler implementation,
batch size, and whether each gap was an actual restart are still assumptions.
The verified current corpus has 1,835,163 tokens and a byte split at 4,892,836;
the prefix used for replay contains 1,650,781 tokens. The current sampler is
`std::mt19937_64` with libstdc++ 13's integer distribution. No historical
sampler or optimizer state was saved with the checkpoints.

One possible phase-error explanation was checked in source rather than assumed:
`Train` resets the training iterator before its update loop, not periodically
inside it. Historical commit `28b25892c74acad228bb380f89a2cadcb3ab7c86`
increments its reported step as `initial_step + update + 1`. Although a generic
trainer can evaluate using its training iterator if no distinct iterator is
configured, the historical Shakespeare recipe at
`b1f6e87530c5d6b604a667592cc2ecb938f0df91` constructs separate training and
training-evaluation iterators and sets `evaluation_tokens` to the latter.
The current recipe retains that separation. Therefore the inspected code
does not support a periodic evaluation-induced reset of the training sampler
as the explanation for this negative result. This source inspection still
does not authenticate the historical executable or invocation flags.

Optimizer memory, different historical settings, and inadequacy of the
transient score remain hypotheses, **not demonstrated causes**. This experiment
does not search additional phases or seeds to make the negative result vanish.
It does not infer token order, multiplicities, complete windows, or passages
from the selected weight rows. Corpus matching is validation/localization,
not an analytical passage decompressor.

## Integrity and independent numerical audit

The extraction completed in 31.36 seconds and froze at 2026-09-09 05:15:45 UTC.
Verification wrote a separate start record at 05:18:38 UTC after authenticating
the frozen files and sources, before any corpus/tokenizer/native-token read.
It completed at 05:18:39 UTC. The supplied native token export was checked
against every corpus byte, and all verification inputs/sources/frozen artifacts
were rehashed unchanged afterward.

The separate weight-only [numerical audit](delta_followthrough_independent_numeric_check.json)
passes. All 15 activity artifacts have the declared shapes, FP64 dtype,
nonnegative finite values, and exact frozen hashes. Scalar logarithms reproduce
all ten individual full score vectors exactly. Independently grouped midranks
and compensated averaging differ from shared scores by at most 2.22e−16.
All 12 stored-score rankings and shuffles are exact; all 12 independent
top-128 **sets** match. Per-frozen-ID component values and positivity counts
also match exactly.

Independent summation swaps the order of two positions in each shared list
(four positions total), with no membership change. Exact integer midrank sums
confirm that these are mathematical ties. The raw swaps are at ranks 57–58;
the adjusted swaps at ranks 85–86. The frozen candidates were not changed,
and all unordered overlap measurements are unaffected. Diagnostic FP64
tolerances are not interval-certified arithmetic bounds.

The numerical audit also rehashes all 20 full original archives unchanged at
05:17:59 UTC. This is an archive identity check, not another reconstruction of
their activity vectors or historical training provenance. It preceded corpus
verification; it is not described as a post-verification archive audit.

The independent [scalar sampler/set-intersection audit](delta_followthrough_independent_replay_check.json)
also passes. It reproduces all 1,201,200 sampled starts, all 48 complete
alternative-overlap vectors, all 192 seed-17 phase overlap counts, all 24,576
individual membership bits, and all 72 full/prefix/suffix frequency summaries
exactly. These checks independently reproduce the negative outcome; they do
not establish historical sampler provenance.

The full Python suite passes 480 tests. The extractor has eleven synthetic
tests, including end-to-end freeze ordering and rejection of changed
checkpoint/source/output files; the numerical audit's own helper fixtures
also pass. The tested verifier was committed as `6d88e46` before the real
verification run.

## Preserved evidence and reproduction

The following repository artifacts are byte-exact copies, preserving their
original absolute paths and identities; copying is not a new experiment:

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| [delta_followthrough_frozen.json](delta_followthrough_frozen.json) | 1,335,379 | `b42aba12eb4cb5072a58cca6daaf94aab4fd9539627d31875f02ce60b17797ed` |
| [delta_followthrough_candidates.json](delta_followthrough_candidates.json) | 132,447 | `999a9606c936b3ea7132a0321d420d176d9bd1ae45713b90a9fbc051fa039a83` |
| [delta_followthrough_verification_start.json](delta_followthrough_verification_start.json) | 2,794 | `dfe83c90835251076dd75c25f2ee6b99889125ed9304186b67762d0765defdea` |
| [delta_followthrough_verification.json](delta_followthrough_verification.json) | 2,264,017 | `c1c2522507162c2ecf5861704c3f74277a1fd5aefbadc17ea290fc9495a425f2` |
| [delta_followthrough_independent_numeric_check.json](delta_followthrough_independent_numeric_check.json) | 73,204 | `49b2fc3a23c01b835fff7ab33d0cb07af7dd6d01e3611549a0e8bf8826ca5d98` |
| [delta_followthrough_independent_numeric_audit.py](delta_followthrough_independent_numeric_audit.py) | 14,111 | `25ae1ed282c1c2217ec88843095fe86d40f38a66df4bedea88610b33c3d9c67e` |
| [delta_followthrough_independent_replay_check.json](delta_followthrough_independent_replay_check.json) | 1,146,664 | `bd613e63081279c10dcfaf0c8d2a1b98d0d6b49e9d708c58a9d9bbde0475d0e6` |
| [delta_followthrough_independent_replay_check.py](delta_followthrough_independent_replay_check.py) | 16,081 | `342301eb6aa31a0a0fd95847fd5305dcde9a4a777df223988a084555a3c66333` |

Original run directory: `/tmp/pluto-delta-followthrough.VjsYYn/run/`.
The freeze retains every archive/member/embedding hash, all 15 activity-file
identities, and complete score identities. The verification retains all 24
real/shuffled lists, both batch conditions, all 1,000 alternative overlaps per
condition, and every selected token's target/control membership bits.

Large arrays remain at their original run paths:

- `scores.npz`: 5,230,398 bytes, SHA256
  `100ec7ab9d811fc653b8317c661b406b0f02a53498ec00ec00f986a6077bebb6`.
- `verification_starts.npz`: 9,618,116 bytes, SHA256
  `cb00fe49d3ecddcd44ba345e1bfffdc89327d8eeae3846b466ae2bd18a1a058c`;
  all 1,001 seeds × 1,200 starts, plus exact seed order.
- `plan.json`: 3,606 bytes, SHA256
  `0785ddedcc79b144fdd205aa2ab37b2ca5386aaf80999d6a257a68ec84cf6557`.

Use new output paths; do not overwrite a frozen experiment. The standalone
sampler can be rebuilt using the commands in the earlier result report.

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/ubuntu/.venv/bin/python \
  -m scripts.weight_analysis.delta_followthrough \
  --checkpoint-root /home/ubuntu/checkpoints/shakespeare \
  --output-dir /tmp/new-delta-followthrough \
  --protocol research/weight_memorization/DELTA_FOLLOWTHROUGH_PROTOCOL.md \
  --inventory research/weight_memorization/checkpoint_history_inventory.json

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/ubuntu/.venv/bin/python \
  -m scripts.weight_analysis.delta_followthrough_verify \
  --frozen /tmp/new-delta-followthrough/frozen.json \
  --corpus testdata/shakespeare.txt \
  --native-tokens /tmp/pluto-native-corpus.KMY7AZ/shakespeare.tokens.bin \
  --native-offsets /tmp/pluto-native-corpus.KMY7AZ/shakespeare.tokens.bin.offsets.bin \
  --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 \
  --sampler /tmp/pluto-replay-sampler \
  --output /tmp/new-delta-followthrough/verification.json

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/ubuntu/.venv/bin/python \
  research/weight_memorization/delta_followthrough_independent_numeric_audit.py \
  --run /tmp/new-delta-followthrough \
  --output /tmp/new-delta-followthrough/independent_numeric_check.json
```

Changing code or protocol after planning requires a new frozen run. Repeating
the commands verifies a fixed method; no additional tuning is part of this
reported negative follow-through.
