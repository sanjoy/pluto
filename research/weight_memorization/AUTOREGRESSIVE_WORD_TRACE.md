# Tracing `shagemper`: generation, operations, weights, and checkpoint deltas

The actual model continuation contains **`shagemper`**, emitted as
` sh` + `ag` + `em` + `per`. It is absent as a whole vocabulary entry and absent
from the supplied Shakespeare corpus. The [bounded lexical review](AUTOREGRESSIVE_WORD_LEXICAL_REVIEW.md)
rejects the interpretation that this is merely an ordinary word or common
compound split into BPE tokens; it does not claim universal novelty.

There is **no single layer that introduces the whole word**. The model emits
one token per forward pass. An unlikely sampled `ag` is followed by confident
`em` and `per` predictions. Native intermediate readouts, exact sampling
reconstruction, and controlled removals identify concrete contributing
operations and weights. They do not reveal a dedicated `shagemper` neuron.

The additional checkpoint-history experiment finds a separate, strong but
limited signal: **weight deltas identify training-token clues and locate source
passages when supplied the corpus**. The tested weight-only ordering method
does not reconstruct their sentences. These two results must not be conflated.

## 1. The actual autoregressive run

Checkpoint: `/home/ubuntu/checkpoints/shakespeare/step_13030`. Production binary:
`bazel-bin/src/llm/recipes/gpt2_shakespeare_llm`. Literal prompt:
`to be or not to be`, without an inserted comma or BOS token. It encodes to
six native IDs. Generation used 512 tokens, temperature 0.8, and seed 17.
The independently instrumented first 128 generated tokens reproduce the
production output prefix byte-for-byte.

```text
to be or not to be
    Thehery devil himself will abate them.
    But here's a murderer, govern him?
    Three thousand strong hand yet pawn'd in France,
    More bound to get a hundred shagemper of France
    Than to hurl your worthiest strength,
```

The table uses zero-based generation indices. Contexts retain every earlier
generated ID; they are not independently re-tokenized versions of the text.

| Index | Emitted piece | ID | Context tokens | Selected row | Final raw rank | Probability at T=0.8 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 51 | ` sh` | 427 | 57 | 56 | 1 | 47.9521% |
| 52 | `ag` | 363 | 58 | 57 | 5 | 2.3556% |
| 53 | `em` | 368 | 59 | 58 | 1 | 91.2067% |
| 54 | `per` | 525 | 60 | 59 | 1 | 81.8388% |
| 55 | ` of` (word boundary) | 286 | 61 | 60 | 8 | 6.2482% |

The word occupies bytes [186,195) of prompt plus generated text. All four IDs
are in the vocabulary; **a multi-token word absent as a single vocabulary
entry is not an out-of-range token**. The product of the four recorded
conditional token probabilities is about 0.8431%, conditional on the preceding
context. This is not the probability of the spelling over every tokenization.

## 2. The precise operation selecting the unusual branch

[Production sampling](/home/ubuntu/code/pluto/src/llm/recipes/gpt2_shakespeare_llm.cc:285)
subtracts the largest FP32 logit, exponentiates after temperature scaling, and
uses `std::discrete_distribution`. The diagnostic records the native choice and
checks the standard-library random draw, rather than assuming a different
Python sampling implementation is equivalent.

```text
w[t] = exp(double(float32(logit[t] - maximum)) / 0.8)
p[t] = w[t] / sequential_double_sum(w)
CDF[t] = sequential_double_prefix_sum(p)[t]
next = first token ID with CDF[next] >= U
```

The final CDF entry is set to one. The generator is `mt19937(seed + 1)`, hence
seed 18 here; U is `generate_canonical<double,53>`. Explicit sequential sums
matter because Python 3.12's compensated `sum` need not reproduce C++ rounding.

At index 52, `ag` has logit **12.24770736694336**, versus **14.301407814025879**
for greedy `ire`. The actual random draw is **0.035620306436279704**, inside
`ag`'s token-ID-ordered CDF interval
**(0.027611331609349775, 0.05116701943207916]**. This is exactly why `ag`, not
the most likely `ire`, was emitted. Its rank is five, not one.

The corresponding U values for ` sh`, `em`, and `per` are
0.2181119136065706, 0.3642201397881389, and 0.05544122820452272. Each falls
inside its recorded token's interval. All 128 sampling events and all 256
underlying MT19937 words were independently reproduced.

## 3. Which layers support each piece?

Blocks are numbered **0 through 7**. A diagnostic readout applies the actual
final LayerNorm and tied output head to an earlier residual stream. It is a
measurement of that representation, not a token already emitted inside it.
All 17 stages for every piece are in the
[readout and intervention tables](AUTOREGRESSIVE_WORD_READOUTS.md).

- ` sh` first becomes top-ranked at the **block 7 MLP** readout.
- `ag` is **never top-ranked** in the 17 readouts; the sampler selects it.
- `em` first becomes top-ranked at the **block 4 MLP** readout, where its
  probability rises from 18.32% to 67.76%; it reaches 90.91% after block 5 MLP.
- `per` first becomes top-ranked at the **block 4 MLP** readout, rising from
  8.26% to 88.82%; it reaches 98.17% after block 5 MLP.

These crossings are not proof of a necessary word-producing layer. In fact,
removing block 4's entire MLP branch **increases** final `em` probability from
91.21% to 99.45%; `per` remains about 81.80%. Later layers and normalization
respond to the changed state. A claim that “block 4 created the word” would
contradict this intervention.

## 4. Math from checkpoint values to logits

This checkpoint has 8 pre-LayerNorm blocks, model width 512, 8 attention heads
of width 64, MLP width 2,048, learned absolute positions, and a tied token/LM
embedding. There are 50,257 logical vocabulary entries, physically padded to
50,272 rows. Checkpoint weights are FP32 master values; the forward path uses
BF16 activations and matrix operands, with FP32 sensitive calculations.

For each block, the operations are:

```text
a = LayerNorm(x)
Q,K,V = a Wqkv + bqkv
head_h = causal_softmax(Q_h K_h^T / 8) V_h
y = x + concat(head_h) Wo + bo
g = GELU(LayerNorm(y) W1 + b1)
x_next = y + g W2 + b2
logits = LayerNorm(x_final) E^T
```

LayerNorm uses population variance and epsilon 1e-5. GELU uses the implemented
tanh approximation, `0.5*x*(1+tanh(c*(x+a*x^3)))`, where c and a are the actual
FP32 constants `0.7978845608f` and `0.044715f`. BF16 conversions, residual
rounding, and reduction differences are retained explicitly in the analysis;
the idealized equations alone do not promise bit-identical GPU accumulation.

For a target-minus-competitor margin, define w as the difference of their BF16
embedding rows. Using the **actual final residual's** standard deviation s:

```text
u = gamma_final * w / s
u = u - mean(u)
margin = sum(residual-component dot u)
       + beta_final dot w
       + final-LN numerical remainder
       + output-head accumulation remainder
```

The residual components include token/position embeddings, every attention
head, all 16,384 MLP neurons, biases, and conversion/addition remainders.
The [implementation](/home/ubuntu/code/pluto/scripts/weight_analysis/token_path_math.py:41)
telescopes the recorded residuals; the independently checked five margins close
within 2.11e-15. **Only the final readout is linearized**. This is numerical
accounting, not a claim that the whole transformer is linear or that accounting
terms equal causal-removal effects.

Attention source accounting is also explicit. For `ag`, block 4 head 4 has a
direct margin contribution of about +0.99767. Its source at context token 56,
` hundred`, receives about 0.3665 attention and contributes +0.83446. Thus the
computation uses the sentence context as well as the last subtoken. Reconstructed
FP64 softmax/source sums are compared with native BF16 head outputs; their
differences are recorded, not silently ignored.

### A concrete weight-addressed example: block 2, neuron 1963

This neuron's direct contribution favors `ag` over `ire` by **+0.857624**.
The checkpoint's dense weight layout is [input, output], so the participating
parameters are directly addressable:

- W1 column: `weight_34.bin`, first byte 7,852, stride 8,192 bytes, 512 FP32 values.
- Encoder bias: `weight_35.bin`, byte 7,852, value -0.02942458912730217.
- W2 row: `weight_36.bin`, bytes [4,020,224,4,022,272), 512 FP32 values.

The FP64 analytical sum of the native LN2 input times the BF16-rounded W1
column, plus bias, is **2.9646345172077417**. The actual native pre-GELU value
is **2.96875**, and the native post-GELU value is also **2.96875**. The small
preactivation discrepancy is explicitly recorded as GPU arithmetic/conversion
remainder, not treated as an analytical identity.

For a neuron's output row r, define a static readout affinity:

```text
affinity[t] = E_BF16[t] dot (gamma_final * (r_BF16 - mean(r_BF16)))
direct_margin_term = native_GELU / s * (affinity[target] - affinity[competitor])
```

Here s=0.9420356922272133; `ag` affinity is +0.19349509387906513 and `ire`
affinity is -0.07864385246742436. Multiplying their difference by 2.96875/s
gives +0.8576240832829051. These affinities are neither probabilities nor
unique word labels: `ag` ranks 147 among this row's 50,257 token affinities.

For comparison, block 6 neuron 1831 ranks `em` ninth and contributes +0.756720
against `'d`; block 6 neuron 1642 ranks `per` 32nd and contributes +0.610169
against `inate`. All 16 selected features, full-vocabulary scores, input gate
products, ranks, and addresses are retained. Some apparently supportive
features primarily suppress a competitor rather than strongly favor the target.

## 5. Causal checks, not just weight projections

Each neuron intervention zeroes its entire W2 row and reruns the native model.
It removes that neuron's contribution at **all prefix positions**, unlike the
single-query-position term in the preceding accounting. Original weights are
restored, and baseline logit bytes match the original autoregressive run.

| Removal | Target | Original probability | After removal | Same random draw |
| --- | --- | ---: | ---: | --- |
| B2 N1963 | `ag` | 2.3556% | 0.8921% | Changes to `ain` |
| B6 N1831 | `em` | 91.2067% | 80.0445% | Still `em` |
| B6 N1642 | `per` | 81.8388% | 70.8479% | Still `per` |
| B0 N418 | `em` | 91.2067% | 84.8577% | Still `em` |

The last row is an important counterexample: its **direct term opposes `em`**,
yet removing it hurts `em`. Across all 16 interventions, direct signs predict
the direction of the measured fixed-competitor margin change in 14 cases and
target probability change in 13. Downstream effects and normalization matter.
Likewise, a higher target probability need not preserve its fixed-U selection
because all vocabulary CDF intervals move.

A separate input counterfactual replaces only the last context ID `ag` with
greedy `ire` before predicting `em`. All 58 earlier input rows and corresponding
causal-prefix stages remain byte-identical. `em` probability falls from
**91.2067% to 14.0485%**; the same original random draw selects ` than`.
This directly establishes that the sampled `ag` branch changes the next-token
computation. It does not establish a single historical training source.

## 6. What the corpus and checkpoint deltas reveal

The authenticated native corpus has these exact adjacent ID pairs:

| IDs | Pieces | Full-corpus occurrences | Containing words |
| --- | --- | ---: | --- |
| 427,363 | ` sh` + `ag` | 2 | `shag` in hyphenated expressions |
| 363,368 | `ag` + `em` | 10 | `stratagem` |
| 368,525 | `em` + `per` | 19 | 16 `distemper`, 2 `intemperance`, 1 `distemperatures` |

Neither three-token subsequence nor the complete four-token sequence occurs.
The plausible interpretation is a new combination of familiar subtoken
transitions. This is **post-hoc corroboration, not proof of their training origin
inside these particular neurons**. In particular, the tempting `emperor` story
does not match these exact native token pairs.

Your checkpoint-history clue gives an additional result, detailed in
[checkpoint-delta passage localization](DELTA_TEXT_EXTENSION_RESULTS.md) and
the [four-sequence follow-up](DELTA_BATCH_FOUR_RESULTS.md):

- The inventory has 1,303 checkpoints, every ten steps from 10 to 13,030.
  Selected intervals were analyzed, not every matrix in all 1,303 snapshots.
- Previously frozen delta-selected token IDs locate five Shakespeare passages
  with 84–98% overlap with conditionally replayed early training windows. The
  weights supply token clues; **the supplied corpus supplies sentence order**.
- An early neighboring-delta bag has **128/128** IDs in its proposed replay
  phase, compared with alternative-seed mean 6.711 and maximum 32. However,
  later bags have **0/128** in their corresponding proposed phase.
- A separate weight-only directional ordering heuristic produced 512
  eight-token candidates across four arms. **None contains a matching
  three-token substring** of the supplied corpus. It did not decode passages.

An interval delta mixes ten AdamW updates, not one raw gradient. No saved
optimizer moments or authenticated historical sampler state were found.
Batch/seed/restart alignment therefore remains conditional. The positive
delta clues do not show that `shagemper` itself was present in training, nor
that any of the identified passages uniquely caused it.

## 7. Verification, reproduction, and scope

Fresh validation on 2026-09-09:

- 599 Python analysis tests passed.
- Four native test targets passed with `--nocache_test_results`:
  `autoregressive_probe_test`, `token_trace_probe_test`, `phrase_probe_test`,
  and `causal_probe_test` (37 tests total, none skipped).
- Google-style clang-format checks for the new native word diagnostics and
  `git diff --check` passed. An initial Python invocation omitted `-t .` and
  failed package-relative imports; the corrected full run above passed. An
  initial formatting invocation used LLVM defaults; explicit Google style passed.
- Independent audits check native production-prefix parity, all sampling
  events, 85 intermediate readouts, 320 branch/head removals, all 16 selected
  neuron removals, the single-token counterfactual, and the math accounting.
- A **self-author second implementation**, separately labeled as such, rereads
  checkpoint bytes and reproduces all 804,112 affinity scores and 64 gate
  identities. An additional **independent reviewer** also reproduced all
  804,112 scores (maximum error 3.331e-16), all 64 identities (maximum error
  4.441e-16), and freshly authenticated 1,100 input/evidence files.

The run is preserved at `/tmp/pluto-autoreg-word.iu9n_w5k/`, including original
plans, raw logits, token IDs, all native captured prefix arrays, analysis arrays,
numerical remainders, full interventions, independent auditors, and input/source
hashes. Checkpoints and production sources were not edited. Research additions
remain uncommitted; existing stashes are untouched.

The [evidence archive](autoregressive_word_evidence.tar.gz) preserves the run,
diagnostic sources, native test logs/XML, and these reports. Its
[per-file manifest](autoregressive_word_evidence_manifest.json) records original
paths, sizes, and SHA256 hashes. Checkpoint payloads and executables are not
duplicated; their identities are retained in the run plans. The deliberately
preserved failed JSON-serialization fragment `*.incomplete.json` is excluded
from the valid evidence bundle.

```sh
bazel build -c opt //src/llm/recipes:gpt2_shakespeare_llm
bazel-bin/src/llm/recipes/gpt2_shakespeare_llm \
  --mode=infer_model \
  --inference_from=/home/ubuntu/checkpoints/shakespeare/step_13030 \
  --tokenizer_dir=/home/ubuntu/datasets/tokenizer/gpt2 \
  --prompt='to be or not to be' --generation_tokens=512 \
  --temperature=0.8 --seed=17

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/ubuntu/.venv/bin/python \
  -m unittest discover -s scripts/weight_analysis -t . -p '*_test.py' -q
bazel test -c opt --nocache_test_results \
  //scripts/weight_analysis:autoregressive_probe_test \
  //scripts/weight_analysis:token_trace_probe_test \
  //scripts/weight_analysis:phrase_probe_test \
  //scripts/weight_analysis:causal_probe_test
```

Exact diagnostic commands are retained in `recorded_128_plan.json`,
`word_trace_plan.json`, `neuron_causal_plan.json`, and the counterfactual plan.
The evidence supports the requested generated-word trace and concrete
weight/delta clues. It does **not** establish a universal lexical novelty proof,
a necessary single word-producing layer, unique training-example attribution,
or a general decoder of ordered training text from checkpoint differences.

### Original-goal completion audit

| Requested item | Evidence and conclusion |
| --- | --- |
| Autoregress from the literal prompt | Production run plus byte-identical recorded 128-token prefix. |
| Find an unusual whole word absent from the vocabulary | `shagemper`, four valid tokens, complete word boundary; exact vocabulary/corpus checks and separately bounded lexical review. |
| Identify the responsible layer(s) | Full native layer readouts and interventions; support is distributed, and a unique introduction layer is not supported by the evidence. |
| Explain operations producing its sequence | Recorded BF16/FP32 forward stages, all target-margin terms and remainders, exact RNG/CDF choices for all four emissions. |
| Inspect related weights | Physical byte addresses, input gates, full-vocabulary affinities, and 16 native neuron-removal tests. |
| Use checkpoint deltas as a training-data clue | Controlled token-bag and corpus-passage localization results, plus explicit negative later-phase and weight-only ordering tests. |

These requested investigations are complete. Unique historical-example
attribution and an ordered-text decoder remain unestablished research outcomes,
not conclusions inferred from the successful word trace.
