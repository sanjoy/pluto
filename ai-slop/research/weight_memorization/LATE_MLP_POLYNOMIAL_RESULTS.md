# Cross-layer late-MLP polynomial: verified negative result

Date: 2026-09-09. Extractor/protocol commit: `481dfe3`.

## Outcome

The fixed cross-layer polynomial did **not** recover a Shakespeare passage.
Across all 256 outputs generated from the final checkpoint (four arms), none
contains even a three-token corpus substring. The complete polynomial reaches
two tokens; its three controls reach only one. The early checkpoint produces
64 copies of sixteen spaces, not a meaningful recovered passage.

All five arms used the same 8,192 weight-selected token IDs and 64 complete
sixteen-token initial strings, with eight alternating coordinate sweeps.
No corpus text, model forward, candidate-dependent nonlinear layer execution,
or training update entered extraction. Selecting MLPs 6 and 7 was nevertheless
informed by the preceding corpus-assisted causal experiment; this is disclosed
in the [frozen protocol](LATE_MLP_POLYNOMIAL_PROTOCOL.md).
The step-10 baseline is already trained, not random initialization, and also
receives the final-checkpoint-selected vocabulary.

| Arm | Outputs | Distinct strings | Longest exact substring | Full 16-token matches |
| --- | ---: | ---: | ---: | ---: |
| Final, full polynomial | 64 | 4 | 2 | 0 |
| Final, curvature removed | 64 | 2 | 1 | 0 |
| Final, cross-layer feature transport removed | 64 | 40 | 1 | 0 |
| Final, broken neuron key/value pairing | 64 | 1 | 1 | 0 |
| Early checkpoint, full polynomial | 64 | 1 | 16 spaces | 64 duplicate records |

These are all outputs, with no filtering or favorable-restart selection.
There are 320 records and 48 distinct strings overall. The full final arm has
63 outputs using at most two different token IDs; 127 of all 320 outputs are
constant-token strings, and 126 are entirely ASCII whitespace.

## The apparent long match is formatting

Every early output is token ID 220 repeated sixteen times: sixteen ASCII
spaces. The native-token verifier finds **51,407** occurrences, with the first
at corpus tokens `[181,197)` / original bytes `[644,660)`. This is one repeated
formatting string, not 64 independently recovered passages.

All three directed-bigram-preserving controls and all three directed-trigram-
preserving controls (seeds 17, 29, 43) retain its full sixteen-token match.
They also reproduce each final arm's maximum match length and zero complete
matches. The shuffles change roughly 94.7% of token positions while preserving
their specified n-gram multiplicities. These are nonuniform Euler-trail
diagnostics, not calibrated significance tests.

For comparison, the global token-label permutation removes the space match.
That weaker control alone would therefore give a misleading impression of
specificity. Preserving ordinary local token structure is important here.

The full final arm's two-token examples include ` have do`, which is unique
at tokens `[1721377,1721379)` / bytes `[5096144,5096152)`. Uniqueness of two
tokens does not make the surrounding repeated ` have` string a recovered
passage. No text is selected or changed using these matches.

## Order-sensitive coefficients did not imply text recovery

The polynomial contains interactions between two source positions and an
output token. On the prespecified small coefficient audit, swapping source
contents at fixed positions gives the following maximum absolute differences:

| Arm | Positions (0,1), target 2 | Positions (0,14), target 15 |
| --- | ---: | ---: |
| Final full | 29.990908 | 0.594021 |
| Final affine | 0 | 0 |
| Final no-cross | 0 | 0 |
| Final broken | 67.431163 | 2.323628 |
| Early full | 0.173909 | 0.002412 |

These are ordinary-FP64 diagnostics of this declared polynomial, not GPT-2
routing measurements or interval proofs. The deliberately broken pairing has
larger differences than the intact pairing. Order sensitivity alone therefore
does not identify a meaningful language interaction. Removing the cross-layer
feature transport also increases candidate diversity without yielding text.

## Exploratory diagnosis: optimizing an unreliable scalar approximation

After freezing and verifying the candidates, a separate diagnostic checked a
necessary magnitude constraint on tanh-GELU. For fixed anchor argument `a`,
increment `z`, and the local quadratic `T = phi(a) + g*z + h*z*z/2`,

```text
|phi(a+z)| <= |a+z|
|T - phi(a+z)| >= max(0, |T| - |a+z|).
```

This test evaluates GELU only once at the fixed anchor `a`, never at a
candidate-dependent argument. It diagnoses the saved polynomial with simple
arithmetic; it does not generate or rerank text. The comparison covers
64 restarts × 15 target prefixes × 2,048 neurons = 1,966,080 slots per group,
retaining duplicated final strings as required by the original denominator.

| Diagnostic | Initial random strings | Frozen final strings |
| --- | ---: | ---: |
| Slots exceeding the scalar envelope | 266,112 (13.54%) | 893,791 (45.46%) |
| Largest absolute increment `z` | 7.38345 | 17.35043 |
| Largest estimated scalar error lower bound | 16.23690 | 86.13153 |
| Prefixes with some estimated scalar error lower bound > 10 | 7 / 960 | 960 / 960 |

Counts subtract the helper's declared FP64 arithmetic slack, at most
`1.733e-12` for the final strings. This is an exact inequality evaluated with
approximate coefficients, **not interval certification** of those coefficients
or the matrix contractions. The large excesses expose an important limitation:
search improves its declared score while this local approximation diagnostic
worsens. Even the initial strings are not reliably in a small-perturbation regime.

These are scalar diagnostics of the surrogate, not measurements of the actual
model's activations. Signed neuron errors and output weights may cancel, so
the lower bounds cannot simply be added to infer a total-logit error. A slot
that passes the envelope test is not thereby accurate. The report and helper
explicitly preserve these limitations; this exploratory check did not change
the frozen experiment or become a corpus-based candidate filter.

## Integrity and evidence boundaries

Before any corpus verification, all five candidate files, plans, and completion
records were hashed together at **02:55:42 UTC** in
`/tmp/pluto-late-mlp.BibMOA/preverification_frozen.json`. All 320 candidate
records were retained in `all_candidates.jsonl`; its SHA-256 is
`73454cb400067b0a6e7090e81a0a27dea26aac360c065a9b97fcd299955957a5`.
The five extraction runs took approximately 76 seconds total on eight requested
CPU threads per run, sequentially. They did not use the GPU or alter checkpoints.

Every run checked all 100 checkpoint-file hashes and current source identities
before/after extraction. All seven compiled-array hashes remained unchanged.
An independent four-thread audit of the full final arm reproduced the selector,
all 64 initial and final scores, and all seven compiled-array hashes exactly.
It replayed all 128 recorded coordinate updates, exactly reproducing their
scores and final IDs; it did not rerun the all-vocabulary coordinate search.
Its separate dense per-prefix oracle differs
by at most `1.82e-12`; the search's recorded maximum score-replay discrepancy is
`1.3642420526593924e-12`. Thus the negative result is not explained by a detected
score implementation or state-mutation error. Coordinate search is still local,
not a certified global maximizer; the unit tests explicitly cover plateau traps.
The original release passed 270 tests; the artifact-validation and envelope
helpers bring the complete analysis suite to **293 passing tests**.

The native verifier validated **every token's original byte interval** against
the corpus and tokenizer. It did not retokenize individual windows. Current
prefix/suffix verification uses the existing native split at token 1,650,781;
this split does not authenticate the historical training/held-out membership.
Only 1/64 full-final candidates has a two-token match in the current prefix,
versus 27/64 in the current suffix. All other final arms remain at one token
in both partitions. The sixteen-space baseline has 46,879 overlapping prefix
occurrences and 4,528 suffix occurrences. An independent direct token scan,
without `CorpusIndex`, reproduces every one of the 320 longest-match results,
including earliest positions, occurrence counts, decoded bytes, and split counts.

A later run of `late_mlp_experiment freeze` independently revalidates all arms
and creates `frozen.json` / `combined.jsonl`. This later manifest is **not**
claimed to precede the first corpus check; `preverification_frozen.json` is the
chronological record. The two combined candidate files have identical hashes.

The method is a deliberately limited static surrogate. Its weight-derived
anchors are not actual late-layer residual states. It omits attention,
LayerNorm Hessians, MLP output constants/biases, and final contextual RMS/bias.
Its signed score is neither a model probability nor the full model's Taylor
polynomial. This result rules out passage recovery by the tested protocol,
not analytical recovery from the checkpoint in general.

## Reproduce and inspect

The exact five-arm commands and algorithm are documented in the
[analysis README](../../scripts/weight_analysis/README.md) and frozen protocol.
Use fresh paths; tools refuse to overwrite evidence. The native verification
command for the preserved run is:

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m scripts.weight_analysis.verify \
  --candidates /tmp/pluto-late-mlp.BibMOA/all_candidates.jsonl \
  --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 \
  --corpus testdata/shakespeare.txt \
  --corpus-token-ids /tmp/pluto-native-corpus.KMY7AZ/shakespeare.tokens.bin \
  --corpus-byte-offsets /tmp/pluto-native-corpus.KMY7AZ/shakespeare.tokens.bin.offsets.bin \
  --output /tmp/new_late_mlp_verification.json
```

Raw evidence in `/tmp/pluto-late-mlp.BibMOA/` includes all five JSONL files,
their `.plan.json` and `.metadata.json` records, logs, `verification.json`,
`structure.json`, `bigram_controls.json`, `trigram_controls.json`, and
`final_full_independent_audit.json`, `independent_direct_scan.json`, and
`final_full_envelope.json`. Committed machine-readable evidence is in
[the result bundle](late_mlp_polynomial_results.json) and
[the full/prefix/suffix summary](late_mlp_polynomial_split_results.json).
The original decomposition/mapping goal
remains open: these outputs do not locate a passage in a particular weight
group and do not constitute a working analytical decompressor.
