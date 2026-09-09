# Initial weight-to-text investigation

Date: 2026-09-08. Tool implementation: commit `c7f08bd`.

## Latest status — 2026-09-09

Four weight-only strategies have been tested; none has demonstrated a
passage-specific decompressor. The newest joint QK-by-OV probe matched
**0 of 4,096 intact-model triples**, and its selected graph cannot form
longer paths. Separately, read-only forward validation measured loss
**0.557 on a current prefix sample versus 5.455 on a suffix sample**;
this is not full-corpus evaluation or proof of complete memorization.
Inference was never used as an extractor. Initial evidence and subsequent
controls are preserved below; the original goal remains unresolved.

## Initial milestone outcome

Static weight inspection finds learned short token associations, especially
Shakespeare's punctuation and line formatting. It has **not recovered a passage
or demonstrated an analytical decompressor**.

The strongest proposed sequences contain a matching five-token substring. Of
1,000 real MLP paths, 504 contain at least four matching tokens, but these reduce
to only **17 distinct selected longest substrings**. Broken key/value pairing
produces no four-token matches. Attention weight products recover more corpus
pairs in the trained checkpoint than in an early checkpoint, but broken head
pairings also recover many plausible pairs.

This is evidence worth pursuing, not a map saying particular weights uniquely
store particular passages. The original goal remains open.

A later bigram-preserving control, recorded separately below, weakens the
passage-specific interpretation further: these paths match shuffled pairwise
structure more often than the original corpus.

The [machine-readable record](/home/ubuntu/code/pluto/research/weight_memorization/initial_results.json)
contains all method summaries, these 17 substring examples, selected attention
examples, physical weight addresses, exact corpus byte intervals, and artifact
hashes. It intentionally omits the tens of thousands of individual records in
the full reports.

## What was inspected

- GPT-2 checkpoint: `/home/ubuntu/checkpoints/shakespeare/step_13030`.
  This is the language model checkpoint, **not** either SAE checkpoint.
- Architecture inferred from the checked implementation and tensor sizes:
  8 transformer blocks, width 512, 8 attention heads, FFN width 2,048,
  context length 1,024, and a tied 50,257-token embedding/LM head. The physical
  embedding has 50,272 rows.
- Storage: 100 distinct little-endian FP32 weight files, 51,483,648 parameters,
  205,934,592 bytes. Model weights alone are about 37.9 times the raw text size;
  no literal lossless compression advantage has been demonstrated.
- Early comparison: `step_10`, extracted into a fresh temporary directory from
  the existing checkpoint archive. This is an **early trained checkpoint**,
  not a guaranteed untouched random initialization.
- Verification corpus: the entire current `testdata/shakespeare.txt`,
  5,436,475 bytes, SHA256
  `4249913cc5998b89bb845fffe1c200dfe6094d4dbe5c0b77b3e806fc680c2f3f`.

Bare checkpoint files do not authenticate their historical producer,
architecture settings, corpus version, or invocation. The recorded source
hashes describe the inspected implementation, not a recovered training manifest.

### Full corpus is not necessarily the training split

These measurements search the **whole source corpus**. They are not a held-out
generalization evaluation or a training-only extraction measurement.

The current GPT-2 `RunTraining` calls `SplitCorpus`, with a default
`--test_fraction=0.1`: a training prefix and a contiguous held-out suffix,
with the boundary moved to a newline. The original `step_13030` invocation and
exact split were not established from the bare weights. Therefore a match
anywhere in this file does not establish that the matched example was actually
seen during that checkpoint's training.

The SAE training path uses the full corpus; that fact must not be substituted
for GPT-2's training-data provenance. See
[GPT-2/SAE training setup](/home/ubuntu/code/pluto/src/llm/recipes/gpt2_shakespeare_llm.cc)
and [SplitCorpus](/home/ubuntu/code/pluto/src/dataset/dataset.cc).

## Evidence contract

Extraction receives only checkpoint weights and the tokenizer vocabulary.
There are no prompts, source-corpus token frequencies, corpus-derived
activations, or model forward passes. Candidate sequences are written and
hashed **before** verification opens the corpus.

Verification finds exact contiguous substrings anywhere inside each frozen
candidate and anywhere in the corpus. It reports the first corpus occurrence
of the earliest candidate substring when longest-match lengths tie, plus that
substring's total occurrence count. Thus a substring match does not certify
the rest of its candidate.

Decoded displays and examples chosen after verification are for explanation
only; they are not new independently extracted candidates. Candidates were
not retuned using these outcomes.

### Native tokenization matters

The production C++ tokenizer emits **1,835,163 tokens** for the corpus.
Hugging Face's implementation using the same `tokenizer.json` emits
**1,836,425**. Whitespace handling differs: for example, native tokenizes
`a\n\nb` as `[64, 628, 65]`, while HF emits `[64, 198, 198, 65]`.

The authoritative results below use the native export. Both exporter and
verifier check the byte spelling of every token against its exact corpus
interval, including tokens splitting a UTF-8 character. The first HF-based
report is retained as a superseded diagnostic, not mixed into these results.
No training tokenizer behavior was changed.

## Strategies and results

### MLP keys/values and static graph paths

For neuron `j`, the input weight column is its key `k_j`, and the output
weight row is its value `v_j`. With tied embedding row `E[t]`, the probe uses

```text
key_score(s, j)   = cosine(E[s], k_j)
value_score(j,t)  = cosine(v_j, E[t])
association(s,t)  = key_score(s,j) * value_score(j,t)
```

Both component scores must be positive. The implementation keeps the strongest
neuron for each directed pair, using the top four vocabulary matches of each
key/value. It emits one top pair per neuron and searches bounded eight-token
paths from 1,000 weight-selected start tokens. Beam width is four; self edges
are skipped and a token may occur at most twice.

The graph search does not update a hidden state or execute transformer layers.
It can combine associations from different blocks in an order unlike the
model's forward computation. This is a deliberately simplified probe.

The broken control cyclically shifts values relative to keys within each
block. It preserves the learned vectors but destroys their original pairing.

| Method | Candidates | Contain a ≥2-token match | Contain a ≥4-token match | Contain a ≥8-token match | Maximum match |
| --- | ---: | ---: | ---: | ---: | ---: |
| Real MLP pairs | 16,384 | 1,776 | — | — | 2 |
| Broken MLP pairs | 16,384 | 1,592 | — | — | 2 |
| Real MLP paths | 1,000 | 880 | 504 | 0 | 5 |
| Broken MLP paths | 1,000 | 939 | 0 | 0 | 3 |

The pair methods propose only two tokens, so they cannot test longer recovery.
For real paths, 410 candidates have a longest match of four tokens and 94 have
five. Most repeatedly traverse a small formatting cycle:

| Selected longest substring, escaped | Candidate count |
| --- | ---: |
| ` And,\n ` | 180 |
| `\n  And,` | 139 |
| `,\n  And,` | 75 |

These are not 504 different passages. In particular, broken paths actually have
more two-token hits; the interesting contrast begins at four tokens, and is
largely a formatting effect. None of the eight-token real proposals matched
in full.

### Attention OV associations

For each head the second probe computes a static vocabulary association from

```text
E[source] @ W_V[head] @ W_O[head] @ E[target].T
```

The recorded run selects 32 sources per head using projected-write norm, then
four targets using cosine scoring. It covers all 64 heads, yielding 8,192
pairs per checkpoint/control. Source selection uses weights, not corpus
frequency. The broken control mismatches value and output heads.

| Attention setting | Matching pair candidates | Distinct matching pairs |
| --- | ---: | ---: |
| Trained step 13,030 | 562 / 8,192 (6.86%) | 512 |
| Broken heads, step 13,030 | 354 / 8,192 (4.32%) | 349 |
| Early step 10 | 40 / 8,192 (0.49%) | 36 |

The trained weights have more corpus-pair associations than either control.
That supports a learned-association interpretation, not unique text storage.
The probe omits attention routing and selects a different high-norm source
set for each operator/checkpoint. It also includes self-pairs. These are not
matched-source experiments or next-token prediction accuracies.

Examples selected for display include ` when gentlemen` and
` should murder` from real heads, but broken heads also produce
` something from` and ` the Constable`. Readable pairs alone are weak evidence.

### Global token-label control

One PCG64 vocabulary-ID permutation with seed 17 is applied consistently to
all candidates in each report. It produces only one matching MLP pair, no
matching MLP paths, no final/broken attention pairs, and one early attention
pair.

This checks that token identities matter, but it replaces common language
tokens with arbitrary vocabulary items. It does **not** preserve corpus token
frequencies and is not a calibrated null distribution or a significance test.
Broken-pairing and early-checkpoint controls are more informative, but retain
their own source-selection and graph-topology confounds.

## A reproducible short weight-to-text certificate

Candidate `mlp_cosine:path738` contains this exact five-token prefix:

```text
token IDs: [6205, 11, 198, 220, 843]
text:      " youth,\n  And"
```

It occurs **twice** in the corpus. The first occurrence is at native token
interval `[11992,11997)`, raw byte interval `[40003,40016)`.
Only these five tokens are certified; the full eight-token proposal is not.

The four addressed key/value pairs are:

| Token association | Block / neuron | Key file / initial byte offset | Value file / initial byte offset | Key / value vocabulary ranks |
| --- | --- | --- | --- | --- |
| `6205 → 11` | 6 / 422 | `weight_82.bin` / 1,688 | `weight_84.bin` / 864,256 | 2 / 1 |
| `11 → 198` | 0 / 906 | `weight_10.bin` / 3,624 | `weight_12.bin` / 1,855,488 | 1 / 2 |
| `198 → 220` | 0 / 1,274 | `weight_10.bin` / 5,096 | `weight_12.bin` / 2,609,152 | 1 / 1 |
| `220 → 843` | 0 / 2,005 | `weight_10.bin` / 8,020 | `weight_12.bin` / 4,106,240 | 1 / 2 |

All addresses are relative to the trained checkpoint directory. Each vector
contains 512 FP32 values. Keys have an 8,192-byte stride, values a four-byte
stride. Vocabulary projections additionally require `weight_0.bin`, whose
embedding rows occupy 2,048 bytes each, and the token-ID vocabulary.

These ranks were independently recomputed from the addressed bytes against
all 50,257 vocabulary rows; the corpus byte interval was checked separately.
The address map therefore supports reproducing the analytical associations.

It does **not** establish that these neurons uniquely store this text. The
path goes from block 6 back to block 0, which is graph traversal rather than
the model's computation order. Most edges encode formatting shared with many
other paths, and the selected substring is not unique in the corpus.
Necessity, sufficiency, and causal specificity remain untested.

## What this milestone does not establish

- **Passage decompression:** no long or complete passage was recovered. A
  five-token mixture of a word and formatting is insufficient.
- **A unique weight map:** addressed vectors participate in many associations;
  distributed, signed, and context-dependent contributions were omitted.
- **Training-example membership:** full-corpus matching does not recover the
  historical train/test split or prove exposure to a particular occurrence.
- **Generalization or significance:** no independently held-out discovery
  corpus or frequency-matched statistical null was used.
- **Causal storage:** no ablation/editing experiment has shown that changing
  these weights selectively removes a particular passage.
- **Negative impossibility result:** failure of these two simplified probes
  does not show that analytical extraction is impossible.

An exact **bigram-preserving corpus shuffle** was chosen as the next test of
whether static path matches contain evidence beyond pairwise associations.
Its completed results are recorded separately in Follow-up 1 below; the
original extraction inputs and initial result artifact remain unchanged.

## Research grounding

The MLP experiment is motivated by Geva et al.'s finding that feed-forward
keys correlate with textual patterns while values induce vocabulary
distributions. That does not imply that nearest vocabulary directions decode
whole stored passages. [Transformer Feed-Forward Layers Are Key-Value Memories
(2021)](https://aclanthology.org/2021.emnlp-main.446/).

The attention experiment uses the OV-circuit decomposition: it describes an
attended token's contribution, while QK determines routing. Treating OV alone
as a sequence predictor discards a necessary part of the model.
[A Mathematical Framework for Transformer Circuits
(2021)](https://transformer-circuits.pub/2021/framework/index.html).

Causal tracing and rank-one editing are relevant future tools for testing
whether a proposed association map reflects model behavior. They are
validation methods, not the analytical decompressor requested here.
[Locating and Editing Factual Associations in GPT / ROME
(2022)](https://arxiv.org/abs/2202.05262).

Exact reconstruction from **gradients** is a distinct setting. DAGER exploits
low-rank attention gradients and discrete embeddings; its availability does
not mean final multi-step AdamW weights can be inverted in the same manner.
No raw training gradients were used in this milestone.
[DAGER: Exact Gradient Inversion for Large Language Models
(2024)](https://arxiv.org/abs/2405.15586).

## Reproduction and artifact integrity

Run from the repository root using Python 3.12 with
[scripts/weight_analysis/requirements.txt](/home/ubuntu/code/pluto/scripts/weight_analysis/requirements.txt).
The recorded interpreter is `/home/ubuntu/.venv/bin/python`.
The commands below intentionally choose fresh output paths.

```sh
analysis_dir=$(mktemp -d /tmp/pluto-weight-milestone.XXXXXX)
export OPENBLAS_NUM_THREADS=8

/home/ubuntu/.venv/bin/python -m scripts.weight_analysis.mlp \
  --checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 \
  --top-k 4 --chunk-size 64 --path-length 8 --path-starts 1000 \
  --include-control --output "$analysis_dir/mlp.jsonl"

/home/ubuntu/.venv/bin/python -m scripts.weight_analysis.attention \
  --checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 \
  --source-count 32 --top-k 4 --chunk-size 4096 --score-mode cosine \
  --blocks all --output "$analysis_dir/attention.jsonl"

/home/ubuntu/.venv/bin/python -m scripts.weight_analysis.attention \
  --checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 \
  --source-count 32 --top-k 4 --chunk-size 4096 --score-mode cosine \
  --blocks all --broken-head-control \
  --output "$analysis_dir/attention_broken.jsonl"

# Only verification, below, reads the text corpus.
bazel build -c opt //scripts/weight_analysis:tokenize_corpus
bazel-bin/scripts/weight_analysis/tokenize_corpus \
  /home/ubuntu/datasets/tokenizer/gpt2 testdata/shakespeare.txt \
  "$analysis_dir/corpus.bin"

for experiment in mlp attention attention_broken; do
  /home/ubuntu/.venv/bin/python -m scripts.weight_analysis.verify \
    --candidates "$analysis_dir/$experiment.jsonl" \
    --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 \
    --corpus testdata/shakespeare.txt \
    --corpus-token-ids "$analysis_dir/corpus.bin" \
    --corpus-byte-offsets "$analysis_dir/corpus.bin.offsets.bin" \
    --seed 17 --output "$analysis_dir/$experiment.verified.json"
done
```

For the early control, run the identical attention command against the
extracted `step_10` checkpoint, then verify its candidates against the same
native corpus export. The recorded early checkpoint and artifact locations
are in the JSON record. Its source archive SHA256 is
`e14d4a939e055775a59be73cce8039922644b8095add1c873eba1cf65dc23eb0`.
The archive was extracted only after validating the 100 expected regular-file
members and their sizes; no training checkpoint was overwritten.

Recorded checks:

```sh
OPENBLAS_NUM_THREADS=8 /home/ubuntu/.venv/bin/python -m unittest \
  scripts.weight_analysis.checkpoint_test \
  scripts.weight_analysis.mlp_test \
  scripts.weight_analysis.attention_test \
  scripts.weight_analysis.verify_test

bazel-bin/scripts/weight_analysis/tokenize_corpus \
  --self-test /home/ubuntu/datasets/tokenizer/gpt2
```

All **59 Python tests** and **7 native tokenizer round-trip cases** passed.
The representative weight projections, source bytes, and every referenced
source/report hash were independently rechecked before recording this
milestone. Synthetic planted-chain tests establish implementation behavior,
not real-model passage storage.

The repository's compact JSON is byte-identical to the generated summary:
SHA256 `86fad94f2cd47f37a27425eb27094f7bcfc9c56655047c3f69cd81ef21cf5f42`.
Full reports and candidate files remain in the recorded `/tmp` paths and are
not committed; those locations are temporary. Their hashes and extraction
parameters are preserved in the compact record so regenerated artifacts can
be compared. Absolute paths and source hashes in regenerated metadata can
differ even when token candidates are identical.

## Follow-up 1: preserving bigrams defeats the passage interpretation

This later verification-only experiment keeps the original candidate file
unchanged and randomizes the corpus using an Euler trail through its directed
token-pair graph. Every directed bigram and its multiplicity, every unigram
count, and both endpoint tokens are preserved. Approximately 94.7% of token
positions change.

The [complete control report](/home/ubuntu/code/pluto/research/weight_memorization/mlp_bigram_controls.json)
records seeds 17, 29, and 43 and identical directed-bigram multiset hashes for
the original corpus and all three shuffled versions. The input candidate
SHA256 is the same as the initial experiment.

Pairs are omitted because their match counts cannot change under this
control. Of the 1,000 real bounded paths, 16 stopped after two tokens, leaving
**984 eligible candidates**. The broken-pairing paths have 992 eligible
candidates. These different denominators are preserved.

For the real-weight path candidates:

| Verification corpus | Candidates with ≥4-token match | Candidates with ≥5-token match | Maximum match | Whole-candidate matches |
| --- | ---: | ---: | ---: | ---: |
| Original Shakespeare | 504 / 984 | 94 / 984 | 5 | 0 |
| Bigram shuffle, seed 17 | 647 / 984 | 445 / 984 | 7 | 0 |
| Bigram shuffle, seed 29 | 633 / 984 | 421 / 984 | 8 | 1 |
| Bigram shuffle, seed 43 | 641 / 984 | 430 / 984 | 7 | 0 |

The broken-weight paths still have no four-token matches in any of these
corpora. Thus original key/value pairing matters for the associations, while
the supposedly longer text evidence becomes **stronger in randomized
bigram-preserving order than in actual Shakespeare**.

This undermines a passage-specific interpretation of the initial formatting
paths. They are consistent with stitching together learned pairwise
associations, not recovering Shakespeare's particular longer ordering. It
does not negate learned bigrams, refute all forms of memorization, or establish
that other analytical strategies cannot recover passages.

These randomized Euler trails are not uniformly sampled. There are only
three seeded runs, and some longer substrings can be forced by the retained
pair structure. No p-value or statistical significance is claimed.

Reproduce after freezing the same MLP candidates and native corpus export:

```sh
/home/ubuntu/.venv/bin/python -m scripts.weight_analysis.controls \
  --candidates "$analysis_dir/mlp.jsonl" \
  --corpus-token-ids "$analysis_dir/corpus.bin" \
  --vocab-size 50257 --seeds 17 29 43 \
  --output "$analysis_dir/mlp.bigram_controls.json"

/home/ubuntu/.venv/bin/python -m unittest scripts.weight_analysis.controls_test
```

All **11 control tests** passed, including exact bigram/unigram preservation,
endpoint preservation, fixed-seed determinism, and candidate immutability.
The initial JSON is intentionally unchanged; this is additional evidence,
not a rewritten initial result.

## Follow-up 2: retain all signed MLP contributions

The first probe keeps a strongest positive neuron association, discarding
cancellations. This follow-up instead forms each block's full signed operator

```text
A = W1 @ W2
raw(source,target) =
  sum_j (E[source] @ W1[:,j]) * (W2[j,:] @ E[target].T)
```

All 2,048 neuron terms contribute before ranking. Vocabulary targets are
ranked by `cosine(E[source] @ A, E[target])`; sources are the top 256 projected
write norms per block, with four targets each. The graph search has up to
eight tokens, 128 starts, and beam width four. Identity gates are fixed at
one. The implementation's alternative fixed bias-GELU derivative gate was
**not run** on these checkpoints.

This remains a linear surrogate, omitting contextual GELU, normalization,
attention, positions, biases, residual additions, and model forward passes.
Weights are analyzed in FP64, not replayed with the GPU's exact arithmetic.
Broken pairing shifts output rows relative to input columns. It preserves the
vectors but reselects source tokens for the altered aggregate operator.

The [aggregate evidence record](/home/ubuntu/code/pluto/research/weight_memorization/aggregate_results.json)
contains separate final/early summaries, exact path-length histograms,
byte-addressed matched-edge examples, source and artifact hashes, and both
complete bigram-control reports.

### Pair and path results

| Setting | Exact matching pair candidates | Bounded path candidates | Path candidates with ≥4-token match | Maximum path match |
| --- | ---: | ---: | ---: | ---: |
| Step 13,030, signed aggregate | 795 / 8,192 | 128 | 2 | 5 |
| Step 13,030, broken aggregate | 56 / 8,192 | 128 | 0 | 1 |
| Step 10, signed aggregate | 78 / 8,192 | 128 | 0 | 3 |

These results support learned short associations, with a stronger broken-pair
contrast than the first per-neuron pair probe. Different source selection
prevents interpreting that contrast as a controlled improvement in corpus
coverage or text recovery.

Actual path lengths are crucial:

| Setting | Length 2 | Length 3 | Length 4 | Length 7 | Length 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Final signed aggregate | 120 | 5 | 1 | 0 | 2 |
| Final broken aggregate | 1 | 127 | 0 | 0 | 0 |
| Early signed aggregate | 8 | 0 | 2 | 1 | 117 |

Only eight real final-checkpoint candidates are even eligible for a
three-or-more-token test. The bounded graph often terminates after a pair;
calling every proposal an eight-token sequence would be misleading.

The two longer matches are ` man, and I am` (five tokens, one occurrence)
and ` not so much as` (four tokens, five occurrences). Three complete
three-token candidates are ` o'er`, ` e'er`, and ` ne'er`, which occur
295, 78, and 195 times respectively. These are recognizable orthographic and
language fragments, not recovered passages.

### The same bigram control remains decisive for interpretation

Among the **eight** eligible final signed-aggregate candidates:

| Verification corpus | Candidates with ≥4-token match | Candidates with ≥5-token match | Whole-candidate matches |
| --- | ---: | ---: | ---: |
| Original Shakespeare | 2 / 8 | 1 / 8 | 3 / 8 |
| Bigram shuffle, seed 17 | 3 / 8 | 1 / 8 | 6 / 8 |
| Bigram shuffle, seed 29 | 3 / 8 | 1 / 8 | 6 / 8 |
| Bigram shuffle, seed 43 | 3 / 8 | 1 / 8 | 6 / 8 |

Maximum match length is five in all four corpora. None recovers an
eight-token final-checkpoint candidate. The 127 eligible broken-aggregate
paths have no three-token matches in any corpus. The 120 eligible early
paths have no four-token matches in any corpus.

Retaining cancellations changes the candidate associations and yields more
word-bearing examples, but does not establish Shakespeare-specific ordering
beyond the bigrams preserved in the shuffled corpus. The three uniform-looking
control outcomes are three deterministic seeded realizations, not a p-value.
The few eligible final paths also make broad claims about this strategy
premature.

### Reproduce this follow-up

```sh
OPENBLAS_NUM_THREADS=8 /home/ubuntu/.venv/bin/python \
  -m scripts.weight_analysis.aggregate \
  --checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 \
  --gate identity --source-count 256 --top-k 4 --chunk-size 4096 \
  --path-length 8 --path-starts 128 --blocks all --include-control \
  --output "$analysis_dir/aggregate.jsonl"
```

Verify this frozen candidate file with the same native-ID/byte-offset
`verify` command, then run `controls` with seeds 17, 29, and 43, using fresh
output paths. For the early experiment substitute the existing extracted
`step_10` checkpoint and omit `--include-control`, exactly as recorded.
All **8 signed-aggregate tests** passed.

The initial results and first control JSON remain unchanged. This follow-up
adds a third analytical strategy and stronger negative evidence against the
current static-path decompression hypothesis, without asserting that a more complete
weight-only method cannot work.

## Validation only: the checkpoint fits the current prefix much better

A separate read-only GPU evaluation loaded step 13,030, took **zero optimizer
steps**, and wrote no checkpoint. It evaluated the first 64 sequential
1,024-target windows from each current 90/10 corpus split: **65,536 targets per
split**, not the whole corpus.

| Current split sample | Cross-entropy, nats/target | Perplexity |
| --- | ---: | ---: |
| Training-prefix sample | 0.556957 | 1.745 |
| Test-suffix sample | 5.45537 | 234.011 |

The initial/final evaluations repeat the same windows and give the same
printed losses; they are not independent replications. All 100 checkpoint
weight hashes before and after agree, and were independently checked against
the files. Runtime was 50.7 seconds.

This shows strongly split-specific fit, not zero loss, full memorization,
token accuracy, or successful free-running reconstruction. “Test” names the
current suffix; its historically held-out status remains unverified.
Perplexities derive from rounded logged losses.
[validation_results.json](/home/ubuntu/code/pluto/research/weight_memorization/validation_results.json)
preserves the raw log, complete command, hashes, and sample definitions.
**No outputs from this evaluation enter any extractor.** It is validation
using model forward passes, not the requested weight-only decompressor.

```sh
bazel-bin/src/llm/recipes/gpt2_shakespeare_llm \
  --mode=train_model --resume_from=/home/ubuntu/checkpoints/shakespeare \
  --steps=0 --checkpoint_every=0 --batch_size=1 --eval_batches=64 \
  --test_fraction=0.1 --corpus=testdata/shakespeare.txt \
  --tokenizer_dir=/home/ubuntu/datasets/tokenizer/gpt2 \
  --log_file="$analysis_dir/validation.log"
```

Confirm the logged resumed checkpoint: `--resume_from` selects the latest
valid checkpoint and could select a different one if the directory changes.

## Follow-up 3: joint QK-by-OV token interactions

The fourth analytical strategy jointly conditions on two token embeddings
rather than chaining independent pairs. In block 0 it contracts signed QK
routing differences with OV write differences across all heads. This is the
first derivative of two-position attention at zero score scale—not the
trained-scale attention output. Contextual LayerNorm, positions, other blocks,
and nonlinear effects remain absent; there are no model forward passes.

Selection was fixed at 128 current tokens by query norm, eight previous
tokens per current token by joint-write norm, and four destinations per pair.
Each final/early/broken run emitted exactly **4,096 distinct triples**.
Broken routing shifts intact OV operators relative to QK heads.

| Weight setting | Candidates containing a matching pair | Complete three-token matches | Complete matches in bigram shuffles (17 / 29 / 43) |
| --- | ---: | ---: | --- |
| Final intact | 481 / 4,096 | **0 / 4,096** | 0 / 0 / 0 |
| Early step 10 | 75 / 4,096 | 0 / 4,096 | 0 / 0 / 0 |
| Final broken routing | 698 / 4,096 | 1 / 4,096 | 0 / 0 / 0 |

The lone broken-control match is ` traitor's uncle`, appearing once. It is
**not a recovery by the intact model**, and does not rescue this experiment.
All vocabulary-label-shuffled runs have zero complete matches.

### Why no longer paths were emitted

Paths require overlapping triples: `(b,a,c)` must connect to `(a,c,d)`.
A corpus-blind audit of the frozen candidates found **zero exact overlaps in
all three runs**. In the final run, the 64 selected previous-token IDs and
128 current-token IDs are entirely disjoint, making extension structurally
impossible under this selection. Early/broken sets overlap in 10/1 IDs, but
still have no exact connecting token pairs.

Thus zero generated paths is a limitation of this disconnected selected
graph, not evidence that the model cannot represent longer sequences.
The complete matches and connectivity audit are separately recorded in
[trigram_results.json](/home/ubuntu/code/pluto/research/weight_memorization/trigram_results.json),
alongside hashes, provenance, and all three complete bigram-control reports.

```sh
OPENBLAS_NUM_THREADS=8 /home/ubuntu/.venv/bin/python \
  -m scripts.weight_analysis.trigram \
  --checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 --block 0 \
  --current-count 128 --previous-count 8 --top-k 4 --chunk-size 2048 \
  --path-length 12 --path-starts 256 --beam-width 4 \
  --output "$analysis_dir/trigram.jsonl"
```

Then use the earlier native `verify` and `controls` commands on this frozen
file. For the broken setting add `--broken-routing-control`; for the early
setting change only the checkpoint and output paths. All **14 trigram tests**
passed.

The next unresolved issues are a corpus-blind selection that admits a
connected overlap graph, and a faithful residual-space geometry/routing
approximation. Those would be new protocols, not retroactive fixes to this
negative result. **Four static strategies have now been tested; no
passage-specific analytical decompressor has been demonstrated.**
