# Three more uncommon, corpus-present words: grandam, corse, Exeunt

The same native-generation, layer-removal, and numerical accounting used for
`beseem` is complete for **three different words**: `grandam`, `corse`, and
`Exeunt`. All three occur in the configured training partition and have no
case-insensitive whole-word tokenizer entry, either bare or with a leading
space. Every emitted component ID is a valid GPT-2 token.

These words were selected from the already completed 2,048-token continuation
of `to be or not to be`, using checkpoint `step_13030`, temperature 0.8 and
seed 17. The user allowed any prompt; no new prompt or inserted target was
needed. Selection was fixed before looking at these words' layer results.
See the [protocol](THREE_CORPUS_WORDS_PROTOCOL.md) and
[complete readout tables](THREE_CORPUS_WORDS_READOUTS.md).

## What was generated, and where it occurs in training

| Word | Actual native pieces / IDs | Generated indices | Training occurrences | Training occurrences with the same IDs |
| --- | --- | --- | ---: | ---: |
| grandam | ` grand` + `am` / 4490, 321 | 365–366 | 23 | 17 |
| corse | ` cor` + `se` / 1162, 325 | 700–701 | 26 | 26 |
| Exeunt | ` Ex` + `e` + `unt` / 1475, 68, 2797 | 1340–1342 | 943 | 883 |

Actual generated excerpts are:

```text
the father to my grandam Love,
This tyrant here, still lives behind a corse;
Exeunt severally
```

The following events are respectively ` Love` (5896), `;` (26), and ` sever`
(1750), confirming complete word boundaries. No fragment of a longer word was
selected. The full corpus has 27, 29, and 1,035 occurrences respectively.
Some spelling/capitalization/spacing variants have different native token IDs;
the table does not silently treat all matches as identical token sequences.

Representative training locations are
[grandam, line 29872](/home/ubuntu/code/pluto/testdata/shakespeare.txt:29872),
[corse, line 19446](/home/ubuntu/code/pluto/testdata/shakespeare.txt:19446), and
[Exeunt, line 2757](/home/ubuntu/code/pluto/testdata/shakespeare.txt:2757).
The independent corpus audit rechecked every one of the 1,835,163 native corpus
tokens against its original bytes, and all 1,091 occurrences of these words.

[Collins calls grandam archaic/now rare](https://www.collinsdictionary.com/dictionary/english/grandam),
meaning a grandmother or elderly woman.
[Merriam-Webster calls corse archaic](https://www.merriam-webster.com/dictionary/corse),
meaning corpse.
[Exeunt is a specialist stage direction](https://www.merriam-webster.com/dictionary/exeunt),
borrowed from Latin. It qualifies here as uncommon outside theatrical usage,
**not uncommon within Shakespeare**; no measured general-English frequency is
claimed. Its high training frequency is an explicit difference from the
other two cases, not evidence of rarity in this dataset. These are lexical
items rather than ordinary phrases made by adjoining their actual token pieces;
historical morphology is not denied.

Membership refers to the supplied corpus and its current default line-aligned
90/10 split, at byte 4,892,836, after 1,650,781 native tokens. It does not prove
which historical minibatch contained an occurrence or which occurrence caused
any particular weight. The surrounding generated sentences are not claimed to
have been copied verbatim from training.

## Which layers produce the token predictions?

All block indices below are **zero-based**. The intermediate readout applies
the trained final LayerNorm and tied vocabulary head to each earlier residual.
It does not imply that a hidden layer contains a discrete word.

| Token | Final rank | Final probability at T=0.8 | First top-ranked intermediate readout |
| --- | ---: | ---: | --- |
| ` grand` | 5 | 3.326462% | Never |
| `am` | 1 | 99.061038% | B0 MLP, 53.232943% |
| ` cor` | 2 | 11.896120% | Never |
| `se` | 1 | 99.988837% | B3 attention, 24.344961% |
| ` Ex` | 2 | 21.025669% | B5 attention, 63.576424% |
| `e` | 1 | 99.9998958% | B0 MLP, 99.965004% |
| `unt` | 1 | 99.999999548% | B0 MLP, 99.999933% |

The initial piece of every word is a non-greedy sample. The final MLP strongly
raises ` grand` from 0.001486% to 3.326462%, and ` cor` from 1.218254% to
11.896120%, but neither becomes top-ranked. Conversely, ` Ex` leads after
B5 attention through B7 attention (95.505795%), then the final MLP lowers it to
rank 2 and 21.025669%; sampling still selects it.

There are reversals within `grandam` too: `am` first leads after B0 MLP, loses
first place after B3 attention, regains it after B3 MLP, loses it after B4
attention, and regains it after B4 MLP. `se` stays first after its B3-attention
crossing; `e` and `unt` stay first after their B0-MLP crossings. The supplement
preserves all 119 word-token intermediate readouts, not just these crossings.

For `Exeunt`, the original history is longer than the model's 1,024-token
window. Its three forwards use original absolute token intervals [322,1346),
[323,1347), and [324,1348), each with local output row 1023. The boundary uses
[325,1349). Learned positions reset to 0–1023 exactly as in production; these
are not re-tokenized or arbitrarily cropped textual prompts.

## Removal experiments: visibility is not necessity

Each of seven token predictions was rerun with every one of 16 whole branches
and 64 attention heads removed independently: **560 native experiments**.
Whole-branch removal zeroes the output projection and its bias across the
entire current context, reruns the model, then restores and verifies the
original weights. It is not a change restricted to one selected-position term.

Removing B0 MLP causes the largest whole-branch probability decrease for every
one of the seven tokens:

| Token | Baseline probability | After B0 MLP removal | Resulting rank |
| --- | ---: | ---: | ---: |
| ` grand` | 3.326462% | 0.0000811864% | 4743 |
| `am` | 99.061038% | 0.0000883164% | 5935 |
| ` cor` | 11.896120% | 0.000162052% | 5014 |
| `se` | 99.988837% | 0.0000556945% | 7907 |
| ` Ex` | 21.025669% | 0.003221316% | 1566 |
| `e` | 99.9998958% | 0.000002350993% | 49733 |
| `unt` | 99.999999548% | 0.0000115358% | 46568 |

This identifies a strong shared dependency, not a unique lexical storage site.
The distinction is particularly clear for `se`: its first leading intermediate
readout is B3 attention, but removing that branch still leaves **99.964403%**
probability. Being the first crossing is not the same as being necessary.

Individual-head effects are also recorded. Removing B0 head 5 reduces the
initial ` cor` from 11.896120% to 0.000723977%, whereas the strongest single-head
removal for its completion `se` leaves 99.788471%. The strongest single-head
removals for `e` and `unt` leave 99.979057% and 99.999955%, respectively. These
are different behaviors from removing an entire MLP branch.

Do not substitute a fixed-competitor margin for full-vocabulary probability.
B0 MLP removal improves ` grand` versus ` mother` from -1.893681 to -0.785919,
and ` Ex` versus space from -1.052113 to +2.174729, even while their probabilities
collapse. Other vocabulary entries dominate the modified distributions.

## The exact operations producing these sequences

The underlying model is unchanged: eight pre-LayerNorm transformer blocks,
512-dimensional residuals, eight 64-dimensional attention heads, 2,048-wide
MLPs, learned absolute positions, and a tied token-embedding/output matrix.
For each autoregressive pass:

```text
x = token_embedding + learned_position_embedding
a = LN(x)
Q,K,V = a Wqkv + bqkv
h = causal_softmax(Q K^T / 8) V
y = x + h Wo + bo
g = GELU(LN(y) W1 + b1)
x_next = y + g W2 + b2
logits = LN(x_after_block7) E^T
```

LayerNorm uses population variance and epsilon 1e-5; GELU is the implemented
tanh approximation with FP32 constants 0.7978845608 and 0.044715. Matrix
operands/activations are BF16 where implemented, with FP32 master weights,
biases, statistics and sensitive calculations. The analytical reconstruction
keeps the differences from FP64 contraction order and BF16 conversions explicit.

For a target t and fixed competitor c, define
`w = E_BF16[t] - E_BF16[c]` and
`u = gamma_final * w / actual_final_std`, then subtract `mean(u)` from u.
Project every captured residual contribution onto u. Embeddings, all heads,
all 16,384 neurons per context, biases, residual rounding, final-LayerNorm
remainder and head-accumulation remainder sum to the actual target margin.
This is final-readout accounting, **not** linearization of the whole network
or a claim that an individual positive term equals its removal effect.

### Byte-addressed examples, one completion per word

The examples below are the largest positive individual-neuron terms for each
word's final piece. The complete accounting includes every opposing term too.
Each input column has 512 FP32 master entries separated by 8,192 bytes;
each output row occupies 2,048 bytes. Entries are BF16-rounded for matrix use.

| Completion | Neuron | W1 file / first byte | Bias file / byte | W2 file / row byte |
| --- | --- | --- | --- | --- |
| `am` vs ` sum` | B5 N1333 | weight_70.bin / 5332 | weight_71.bin / 5332 | weight_72.bin / 2729984 |
| `se` vs ` at` | B5 N1205 | weight_70.bin / 4820 | weight_71.bin / 4820 | weight_72.bin / 2467840 |
| `unt` vs ` Exit` | B0 N1300 | weight_10.bin / 5200 | weight_11.bin / 5200 | weight_12.bin / 2662400 |

For `am`, input dot plus bias is analytically 1.0895818509161472; the captured
native pre-GELU value is 1.0859375. The native post-GELU value 0.93359375 times
the output-row projection 0.6174014088167333 supplies **+0.5764020965124972**.

For `se`, the analytical input sum 2.824092481750995 becomes native pre-GELU
2.828125, and captured post-GELU 2.828125. Multiplying by the output-row
projection 0.5461721497662233 supplies **+1.5446431110576002**.

For `unt`, the analytical input sum 1.697345917345956 becomes native pre-GELU
1.6953125. Captured post-GELU 1.6171875 times 0.2887630981199141 supplies
**+0.46698407274079856**. These are pieces of final native margins 4.630561829,
7.629604340 and 16.633277893, not complete explanations by one neuron.

Attention supplies contextual information as well. For `unt`, B1 head 2 places
reconstructed attention probability **0.91729440214** on local position 1022,
the earlier ` Ex` piece at original absolute position 1346. That source supplies
+0.969771050312 to the margin; the complete native head supplies +0.966737241269.
Other source terms and the explicitly retained native/reconstructed numerical
remainder account for the difference. This connects the completion to its
earlier piece but does not establish that this head alone is necessary.

Finally, support becomes an actual token through sampling, not argmax:

```text
weight[t] = exp(double(float32(logit[t] - max_logit)) / 0.8)
p = weight / sequential_double_sum(weight)
U = generate_canonical<double,53>(mt19937(seed + 1))
token = first vocabulary ID whose cumulative probability is >= U
```

The source recorder's complete MT19937 sequence was verified previously;
these seven events reuse its exact uniforms. New raw-logit reconstruction
checks each probability and CDF interval against that record. All seven
uniforms fall inside their selected intervals; the supplement prints their
actual values. No claim relies on greedy generation or teacher-forced targets.

## Verification and reproducibility

All ten new native traces, including the three boundary events, reproduce
their original-generation logit rows byte-for-byte. Every exact input window
matches. Checkpoint/source identities remain unchanged.

The independent math audit rechecks 10 ledgers, 170 intermediate readouts,
560 removals, 640 head terms, 163,840 neuron terms, 469,312 attention-source
terms, and 480 detailed gates comprising 245,760 input products. All 740
sampling/readout probability checks agree exactly. Maximum ledger closure
error is 5.33e-15; maximum individual-term difference is 3.11e-15. The audit
adapts the prior independent implementation, rather than claiming a wholly new
independently invented method. Its stronger runner/window/raw-parity gates are
covered by new tests.

The table generator separately recomputes all 686 word-token distributions
(seven baselines, 119 intermediate readouts, 560 removals) and verifies exact
report regeneration. **647 Python tests and 37 native tests pass**, with no
native skips; actual logs and XML files are preserved.

Evidence root: `/tmp/pluto-three-words.qPVE2k/`. The frozen plan is
`run/plan.json`; `run/result.json` records terminal completion. Each
`run/analysis_step_N/analysis.json` holds full numerical accounting and
counterfactual distributions. `run/multiword_math_independent_audit.json`,
`lexical_corpus_audit.json`, and `tests_result.json` preserve independent checks.
The [evidence manifest](three_corpus_words_evidence_manifest.json) identifies
all reports, scripts and raw local evidence. Large arrays remain local.

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/ubuntu/.venv/bin/python \
  -m scripts.weight_analysis.multiword_trace \
  --generation-directory /tmp/pluto-corpus-word.clM9FN/recorded_2048 \
  --checkpoint-directory /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --tokenizer-directory /home/ubuntu/datasets/tokenizer/gpt2 \
  --binary /home/ubuntu/code/pluto/bazel-bin/scripts/weight_analysis/token_trace_probe \
  --output-directory /tmp/new_three_word_trace \
  --word grandam:365:367 --word corse:700:702 --word Exeunt:1340:1343
```

The result is a mechanistic account of these three actual emissions, not a
claim of a single word-introducing layer, universal behavior across prompts,
or unique attribution to one historical training sentence. Production code
and checkpoints were not changed. New research code and reports are uncommitted.
