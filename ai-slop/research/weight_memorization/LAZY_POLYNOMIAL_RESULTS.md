# Wider-vocabulary attention polynomial: verified result

Date: 2026-09-09. **No passage decompressor recovered.** Across five fixed arms,
all 5,120 sixteen-token path records lack even a four-token exact substring of
the current Shakespeare corpus. The best matches are short three-token phrases;
none of the complete paths matches. The experiment is useful for rejecting this
particular static decoder, not for locating a passage inside a weight tensor.

The [protocol](/home/ubuntu/code/pluto/research/weight_memorization/LAZY_POLYNOMIAL_PROTOCOL.md)
was committed before the real runs. The extraction tools and independent tests
were committed as `051204a`; all five candidates were frozen at
2026-09-09 01:21:48 UTC before structural diagnostics and corpus verification.
The machine-readable record is
[lazy_polynomial_results.json](/home/ubuntu/code/pluto/research/weight_memorization/lazy_polynomial_results.json).
Earlier experiments and the user's stashes were left unchanged.

## What changed, and what remained fixed

The preceding experiment used a vocabulary/repetition policy that could not
express any corpus substring longer than four tokens. This experiment instead
uses 8,192 weight-selected IDs, already measured to cover 98.1254% of native
corpus token positions. No new corpus information enters extraction. All arms
share these IDs and the same 256 ordered seed pairs from the first 16 ranked
IDs. Every repetition is allowed. Beam width is four; path length is sixteen.

The candidate destination score is
`unit(write) dot (E[token] - mean_ALL_LOGICAL_VOCAB(E))`.
There is no target-norm division or magnitude cutoff. The three writes are the
uniform-routing output `w0`, the routing derivative `w1`, and the first-order
sum `w0+w1`. A control rotates complete content heads relative to routing heads;
an early checkpoint supplies another comparison. The early arm explicitly
receives the final-trained vocabulary selection, and is not initialization.

Extraction evaluates a stationary graph built from the first block's static
position-zero/one dictionaries. It does not update contextual hidden states,
run attention softmax, evaluate the other seven blocks, or generate text using
the trained model. The method choice was informed by earlier corpus diagnostics:
this is an exploratory follow-up, not an untouched held-out method-selection
test. No settings were retuned after observing these new matches.

## Exact text verification

Each row contains **1,024 distinct sixteen-token paths**. Counts below describe
paths containing an internal matched substring, not complete recovered paths.
The native C++ tokenizer's byte-aligned export is authoritative; every token's
byte interval was checked against the source file.
These checks concern the full current corpus; the historical training file
version and exact training split have not been independently authenticated.

| Arm | Paths containing >=2 tokens | Paths containing >=3 tokens | Distinct selected longest triples | Paths containing >=4 tokens |
| --- | ---: | ---: | ---: | ---: |
| Final first-order | 704 | 5 | 2 | 0 |
| Final derivative only | 874 | 32 | 2 | 0 |
| Final uniform only | 714 | 1 | 1 | 0 |
| Broken first-order | 750 | 8 | 1 | 0 |
| Early first-order | 336 | 0 | 0 | 0 |

The two-token columns can include the original seed pair and are especially
weak evidence. Three-token matches necessarily include at least one generated
token, but remain short associations. For example, first-order paths contain
` as long as` (four corpus occurrences; first byte interval
`[1657553,1657564)`). The derivative has ` not wish the` (one occurrence at
`[2391441,2391454)`), and the broken control has ` the the the` (one occurrence
at `[4103279,4103291)`). These are exact bytes, including the initial space.
Uniqueness in this corpus does not turn a three-token match into passage recovery.

Cached three-token graph edges have a different denominator and are recorded
separately in the JSON. They are alternatives considered during search, not
additional sixteen-token recovered passages.

### Order-preserving controls

The same frozen paths were matched against corpus shuffles preserving every
directed bigram and its multiplicity. Seeds were fixed at 17, 29, and 43.

| Arm | Real paths containing >=3 tokens | Bigram-shuffle counts: 17 / 29 / 43 |
| --- | ---: | ---: |
| Final first-order | 5 | 8 / 8 / 8 |
| Final derivative only | 32 | 0 / 4 / 32 |
| Final uniform only | 1 | 7 / 0 / 5 |
| Broken first-order | 8 | 16 / 12 / 8 |
| Early first-order | 0 | 0 / 0 / 0 |

No arm exceeds its highest bigram-control count. One bigram shuffle even gives
the primary arm a four-token match, absent from the real corpus. Every real
arm and every trigram-preserving shuffle has zero four-token matches; no
control produces a complete sixteen-token match. These nonuniform shuffles
are descriptive checks, not p-values or a proof of statistical equivalence.
The short matches do not establish passage-specific storage beyond ordinary
local token structure.

## Why this decoder fails to produce useful paths

Structural checks were completed before opening the corpus. No entire path
is ASCII-whitespace-only, and each arm has 1,024 distinct paths. Those facts
do not make the continuations diverse: varied two-token seeds can hide loops.

| Arm | Constant suffix >=8 tokens | Alternating suffix >=8 tokens | Paths repeating any token >2 times |
| --- | ---: | ---: | ---: |
| Final first-order | 282 | 37 | 1,024 |
| Final derivative only | 0 | 6 | 103 |
| Final uniform only | 348 | 21 | 1,024 |
| Broken first-order | 268 | 37 | 1,024 |
| Early first-order | 21 | 0 | 823 |

On exactly the same initial 256 pairs, the final first-order top token agrees
with uniform routing on 178 pairs (69.53%), the derivative alone on 7 (2.73%),
and the broken pairing on 182 (71.09%). Its top-four set is identical to the
uniform baseline on 69 pairs; mean top-four Jaccard overlap is 0.6343. Against
the early checkpoint there is no top-four overlap on any seed pair.

Thus training substantially changes these local rankings, and the baseline
accounts for much of the final first-order ranking. The derivative is not
irrelevant: adding it changes 78 of the 256 uniform-baseline top choices.
Removing the baseline reduces repetition dramatically, but still recovers no
four-token substring. These observations diagnose this graph, not GPT-2's
generation behavior or a causal allocation of memorized passages.

## Numerical validity is not text validity

The primary arm visits 1,306 unique pairs. Its median absolute head score gap
is 0.371, its 99th percentile is 1.612, and its maximum is 3.098. The stacked
head correction error is bounded between 7.16% and 22.20% of the stacked linear
correction norm. This ratio is **not** a bound on the summed output, normalized
ranking, or full-model error; cancellation can change those relationships.

A separate sum-of-head upper bound satisfies the conservative top-token margin
condition on 245/1,306 primary-arm pairs (18.76%) in ordinary FP64. For any pair
satisfying the inequality in exact arithmetic, the top token is guaranteed to
agree with exact two-position attention over this selected vocabulary. The
245 count is a floating-point diagnostic, not an interval-certified count,
an assurance about all four beam alternatives, or a path/full-model certificate.
Failure of the sufficient condition is inconclusive.

There are no exactly zero writes. The primary arm's write norm ranges from
0.953 to 2.805, with median 1.455. Across all arms the minimum recorded
cancellation ratio is 0.323, so no catastrophic near-zero head sum is apparent
in these visited cases. Numerical underflow or extreme cancellation therefore
does not explain the observed failure. Nor does the local bound justify
claiming that a more accurate attention calculation would solve passage
extraction. The decoder still omits contextual and cross-layer computation.

## Weight dependencies: precise, but not a passage map

For the final checkpoint `/home/ubuntu/checkpoints/shakespeare/step_13030`:

| Quantity | Physical weights used |
| --- | --- |
| Vocabulary and centered output dictionary | All 50,257 logical rows of `weight_0.bin`; padded rows excluded |
| Static input coordinates | Selected rows of `weight_0.bin`, position rows 0/1 of `weight_1.bin`, first LayerNorm scale/bias in `weight_2.bin` / `weight_3.bin` |
| Uniform component `w0` | V slices of `weight_4.bin`, V bias in `weight_5.bin`, all content-head slices of `weight_6.bin`, global bias `weight_7.bin` once |
| Derivative component `w1` | Q/K/V slices of `weight_4.bin`, Q bias in `weight_5.bin`, content-head slices of `weight_6.bin`; key/value/output biases cancel or do not enter |
| First-order component | Sum of the preceding two components |

The metadata supplies exact head slices, strided matrix views, byte ranges,
token row offsets, and hashes. Global vocabulary selection and centering mean
that these candidates cannot be attributed only to a few named token rows.
Q/K do not affect uniform-only scores; they are read solely for its diagnostic
comparison. Files `weight_8.bin` through `weight_99.bin` are not used to score
the extracted graph, though complete checkpoint hashes authenticate identity.

This is a reproducible map of **what the probe reads**, not evidence that
these files individually contain the matched phrases. No passage-specific
causal intervention has been established.

## Reproducibility and next direction

Artifacts: `/tmp/pluto-lazy-polynomial.51uxo2C4/`. The frozen manifest SHA256 is
`f5096b305867eb788adf158e637833276648600f0cd7635c248f3926a822d081`.
It records all five candidate/metadata hashes, exact invocations, source commit,
checkpoint hashes, shared vocabulary/seeds, and completion times. The compact
JSON retains the verification evidence and artifact identities in the repository.

All 209 Python analysis tests and seven native tokenizer round-trip checks
pass. Independent scalar-oracle tests validate the polynomial and bias handling
on toy weights. A separate read-only audit replayed all 5,120 paths exactly from
their cached edge IDs, token triples, cumulative scores, and mean scores, and
confirmed all computational source hashes and shared vocabulary/seed hashes.
Tests establish implementation properties, not success on real text.
After verification, all 100 current checkpoint weight hashes still matched
the extraction metadata; step 13030 remained the latest uncompressed GPT-2
checkpoint. Neither model weights nor the user's saved stashes were modified.

The strongest positive finding remains the earlier
[vocabulary/frequency trace](/home/ubuntu/code/pluto/research/weight_memorization/VOCABULARY_RESULTS.md),
not ordered passage recovery. Expanding this same first-block stationary graph
again is not yet justified by the evidence. A better next direction is to
separate two tasks: causally localize weight groups supporting different
passages, then test a genuinely cross-layer analytical extractor using those
groups. Contextual/model evaluation may serve that validation stage, but must
not be relabeled as the requested decompressor. Both an independently supported
passage map and the analytical decompressor remain open.
