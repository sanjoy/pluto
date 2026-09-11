# Vocabulary recovery from weights: a step toward passage extraction

Date: 2026-09-09. This experiment leaves the user's stash untouched.
The protocol was committed as `0a0d97c` before extracting candidates; the
ranking tool was committed as `5eec6ef`. This is a new experiment, not a
replacement for earlier frozen results.

## Verified outcome

There is a strong, directly recoverable **vocabulary/frequency signal** in
the trained weights. It is not yet a passage decompressor. All 256 final
12-token path candidates have a longest corpus match of at most two tokens.

The fixed full-vocabulary score/count correlations are:

| Weight-only score | Final step13,030 | Early step10 | Final score-label permutation |
| --- | ---: | ---: | ---: |
| Embedding norm | -0.813302 | -0.013511 | -0.000092 |
| Distance from mean embedding | +0.577849 | +0.030881 | -0.003780 |
| Final LayerNorm output offset | +0.844031 | +0.223522 | +0.000260 |

These Spearman correlations include the 31,377 absent vocabulary IDs, tied at
count zero. There are 18,880 observed IDs in 1,835,163 token positions. They
must not be mistaken for a frequency-ordering test solely among used tokens.

| Final weight-only ranking | Top128 position coverage | Top8,192 position coverage | Top8,192 IDs present |
| --- | ---: | ---: | ---: |
| Descending embedding norm | 0.000163% (3 positions) | 0.020162% (370 positions) | 119 / 8,192 |
| Descending centroid distance | 13.4519% | 98.1254% | 8,192 / 8,192 |
| Descending output offset | 28.5298% | 79.9696% | 8,192 / 8,192 |

Thus large raw norms particularly favor absent tokens in this trained
checkpoint; they are not a sensible high-frequency selector. We did not
reverse that ranking after seeing the result. The centered-distance signal
identifies, without text input, a token set covering almost the whole corpus.
That set still covers only 43.39% of its distinct vocabulary, omitting a long
rare-token tail. It does not tell us token counts or their order exactly.

The early output-offset top128 covers 50.74% of positions, more than the final
offset top128, despite its much weaker whole-vocabulary correlation. Ranking
correlation and coverage of an extreme prefix measure different properties;
neither should substitute for the other.

### Post-hoc diagnostics: separate from the frozen protocol

To distinguish token presence from frequency ordering, we additionally
checked the same frozen scores *only among the 18,880 observed IDs*. This
verification-only diagnostic did not change any extraction or candidate:

| Score | Final observed-only Spearman | Early observed-only Spearman | Final presence ROC-AUC |
| --- | ---: | ---: | ---: |
| Embedding norm | -0.322636 | -0.014898 | 0.012892 |
| Centroid distance | +0.959911 | +0.063717 | 0.756380 |
| Output offset | +0.841905 | +0.376613 | 0.986478 |

The positive signal is therefore not solely an artifact of tying absent IDs.
ROC-AUC describes discrimination of membership in the *current source file*,
not authenticated historical training membership. These diagnostic quantities
were not preregistered and have no claimed p-values.
No alternative-corpus specificity comparison was performed: a strong
association with this corpus is not proof of a uniquely Shakespeare-specific
frequency code.

### Ordered extraction still failed

Each arm emitted 65,536 distinct triples and 256 distinct length12 paths.
The exact native-token verifier found:

| Arm | Full triples matching | Paths containing a matching pair | Longest match inside any path | Paths with any three-token match |
| --- | ---: | ---: | ---: | ---: |
| Final intact | 29 / 65,536 | 209 / 256 | 2 | 0 |
| Final broken QK/OV pairing | 26 / 65,536 | 193 / 256 | 2 | 0 |
| Early intact, final-selected vocabulary | 6 / 65,536 | 71 / 256 | 2 | 0 |

There is no recovered passage. Broken pairing gives nearly as many exact
triples as the intact model. The order-matched controls likewise do not
support a claim of longer sequence recovery.

Membership-only coverage initially looks promising: the selected S contains
68,696 corpus windows of length12, and its longest contiguous run is57.
However, **that is not the feasibility ceiling for this decoder**: it also
forbids adjacent equal tokens and permits at most two occurrences of any
token. A separate post-hoc check against these actual constraints finds:

| Window length | All tokens in selected S | Also satisfies the decoder's repetition rules |
| --- | ---: | ---: |
| 3 | 200,101 | 218 |
| 4 | 122,411 | 10 |
| 8 | 90,502 | 0 |
| 12 | 68,696 | 0 |

The longest compatible corpus window is **four tokens**. In this case the
adjacent-repeat rule alone removes all longer windows; the additional
max-two-occurrences rule does not reduce the surviving counts. Thus no
ranking used by this decoder under these constraints could recover a
five-token substring from this corpus.
This diagnoses a decoder/vocabulary design failure, not absence of sequence
information in the trained model. The frozen result was not retuned.

The next passage experiment must address this representational ceiling before
interpreting matches: allow legitimate repeated tokens and/or use a broader
weight-selected vocabulary. Dense enumeration of every triple in that wider
set is unnecessary; bounded search can evaluate the static weight contraction
only for visited token pairs. No such follow-up is claimed as tested here.

### A weight-only scoring audit

Separately from corpus verification, the committed gauge audit freezes all
16,384 final pair-context writes and replaces every output target E[t] by
`E[t] - mean_v(E[v])`. The mean includes all 50,257 logical rows, not merely S.
Compensating position embeddings makes this an ideal prediction-preserving
change of model coordinates, as explained below. No new candidates are made.

For fixed write w, every raw score changes by the *same* `w dot shift`.
The measured raw top1 and top4-set rankings were unchanged in **16,384/16,384**
contexts, with no numerical-invariance failures. The largest arithmetic
residual in the common-shift identity was 3.33e-16.

In contrast, cosine top1 choices changed in **1,026/16,384 (6.26%)** contexts,
and top4 sets changed in **3,311/16,384 (20.21%)**. There were no zero-norm
writes or targets to explain this difference. This demonstrates that these
cosine rankings depend on the embedding origin, not just model behavior.
It does **not** show that raw scoring recovers better text, or that this caused
the passage failure: the vocabulary/repetition ceiling already prevented
long recovery. A later decoder should use a translation-invariant score and
separately test whether its ordered output is specific to the real text.

## Artifacts and tests

The [compact evidence record](/home/ubuntu/code/pluto/research/weight_memorization/vocabulary_results.json)
contains all fixed-prefix metrics, controls, post-hoc diagnostics, and hashes.
Full immutable experiment artifacts are in
`/tmp/pluto-vocabulary-offset.hFYW8aCk/`, with all three candidate hashes frozen
in `frozen_manifest.json` before any corpus verification.

The ranking SHA256 is
`24e28b0f0326bc2d7ccd5b06b0574033af6ff4b3da9e8b1274f5ee2915c129fe`.
The final gauge-audit SHA256 is
`b0464afb4315d8c1cfbd4a63b9fc3759440ca123908af50d57531431e8913949`.
Sources were committed before their respective real-data runs:
`5eec6ef` (scores), `7e067a1` (verifier), `1b2fda5` (paths),
`2417cd0` (gauge audit).

**174 analytical-tool tests pass**, including 44 new tests for this milestone.
The native tokenizer also passes all seven exact token/byte round-trip cases.
No GPU training, model inference, checkpoint changes, or stash restoration
were performed in this experiment. The analytical passage-decompression goal
remains open.

## Why this signal

With tied input/output embeddings E and final LayerNorm scale gamma and
bias beta, the ideal-arithmetic logit decomposes as

    logit[t] = (gamma * normalize(hidden)) dot E[t] + beta dot E[t].

The second term is a context-independent, directly addressable contribution
to every token logit. It can be extracted by a matrix-vector multiplication
without any prompt, contextual hidden state, transformer execution, or
softmax. We compare it with the embedding-row norm and distance from the mean
embedding. Neither geometric score is assumed to equal frequency.

This is supported by prior empirical work, not a count-recovery theorem.
Kobayashi et al. found that projecting prediction-head LayerNorm biases onto
token embeddings tracks corpus frequency in GPT-2, and changes with
fine-tuning-domain frequencies. Their findings motivate this test on our
checkpoint; they do not establish its outcome or passage memorization.
[Kobayashi et al., ACL Findings 2023](https://aclanthology.org/2023.findings-acl.276/)

The model's actual contextual term can outweigh the bias term. In particular,
softmax(E beta) would be a bias-only summary, **not** an estimate justified here
of the model's average predictions or of corpus frequencies. At even ideal
unregularized stationarity, the beta gradient would constrain only
`E.T @ (mean_predictions - target_frequencies) = 0`; it would not identify the
entire vocabulary distribution. Finite AdamW training is further from that
idealization.

## Exact weight dependencies and coordinate caveat

In this checkpoint, token t's offset uses bytes `[2048*t, 2048*(t+1))` in
`weight_0.bin`, together with all 2,048 bytes of `weight_99.bin`. The files
contain FP32 master weights; the analysis promotes them to FP64 and does not
reproduce BF16 execution rounding. Vocabulary padding is excluded.

There is a useful algebraic sanity check: add the same vector c to every token
embedding and subtract it from every position embedding. All token-plus-
position inputs stay identical. Tied output logits acquire the same additive
constant, so ideal-arithmetic probabilities stay identical. Raw embedding
norms can change under this transformation. Centroid distances stay identical,
and all offset scores shift by the same `beta dot c`, preserving their ranking.
Thus raw norms are not an intrinsic measure of functional importance.

An embedding row receives gradients both when used as an input and densely
through the tied prediction head. Changed weights or unusual norms are not
proof that a token occurred in training. The step_10 comparison is already
trained, not an untouched initialization.

## Fixed experiment and interpretation

Six complete rankings were frozen: the three scores separately for step_13030
and step_10. The path extractor is fixed to the top128 **final output-offset**
IDs, regardless of how well other rankings verify. Final intact, final broken
QK/OV pairing, and early intact all use that same set. The early arm therefore
benefits from a final-trained vocabulary prior.

Path scoring still uses only the existing first-order, two-position QK-by-OV
contraction. It omits the uniform-routing base, residual path, MLP, later
blocks and final contextual computation. Positions0/1 are reused for every
edge; it is not a full positional model of each proposed sequence. Vocabulary
coverage is a necessary condition for passage extraction, not sufficient
evidence of ordered storage.

The verifier uses the exact native token export for the full current source
file. The bare checkpoint does not authenticate the historical training split
or corpus version. Full-vocabulary frequency correlation includes absent
tokens and average ranks for tied frequencies. Top-k coverage, observed-ID
precision/recall and run lengths answer different questions. Label-permutation
and order-preserving corpus shuffles are descriptive controls, not p-values.

## Reproduction

Use fresh paths: tools refuse to overwrite artifacts. The existing exact
native-token exporter is documented in the analysis README.

```sh
OPENBLAS_NUM_THREADS=4 python -m scripts.weight_analysis.vocabulary \
  --checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --baseline /tmp/pluto-attention-early.ee35wjay/step_10 \
  --output /tmp/fresh_rankings.json

OPENBLAS_NUM_THREADS=4 python -m scripts.weight_analysis.offset_paths \
  --checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --rankings /tmp/fresh_rankings.json \
  --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 \
  --output /tmp/fresh_paths.jsonl
```

Repeat path extraction for the broken pairing and early checkpoint before
opening the corpus. Then run the vocabulary verifier, exact text verifier,
and order-matched controls on the frozen artifacts. Do not select a score,
reverse a ranking, or retune paths using the verification results.
