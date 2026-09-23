# One-shot memorization and the learned encoding

Research objective: construct a memorizer from the dataset and tokenizer without
gradient descent, then establish how its representation relates to the learned
114,256-parameter GPT-2 model. Exact corpus recall by an unrelated construction
does **not** complete the second part of this objective.

Work is on `codex/one-shot-memorizer`, based on `main`. Artifacts are local-only.
All implementations here are C++; the initial construction and runtime are CPU
code. The existing tokenizer interface needs a CUDA executor for its pinned
output arrays, not for training or evaluating the constructed automaton.

## Construct and verify

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer
tokenizer=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0/inputs/tokenizer
bazel-bin/src/llm/experiments/one_shot_memorizer/one_shot_memorizer \
  --mode=compile --tokenizer="$tokenizer" \
  --corpus=testdata/general_facts_dataset.txt \
  --output_dir=/tmp/one_shot_memorizer_new
```

The output directory must not already exist; its parent must exist. It contains
binary numeric model weights, corpus verification metrics, a context-window
conflict profile, and per-target context requirements. No trained checkpoint is
read. Verification serializes and reloads the model, then feeds its own greedy
predictions back in until EOS, starting with each fact's first five tokens.

Inference reads only compiled weights and the original GPT-2 tokenizer, not the
corpus. Original token IDs avoid needing a separately learned vocabulary map.

```sh
bazel-bin/src/llm/experiments/one_shot_memorizer/one_shot_memorizer \
  --mode=infer --tokenizer="$tokenizer" \
  --model_file=/tmp/one_shot_memorizer_new/automaton.weights \
  --prompt='The capital of France is'
```

The automaton runtime rejects unseen prefixes. Ambiguous corpus prefixes use empirical
next-token frequencies with the lowest token ID breaking ties. This is NOT a
claim of equivalence to the trained model on arbitrary prompts or probabilities.

## Construction: sparse transition matrices

Build a prefix trie of the tokenized sentences and merge states bottom-up when
their complete weighted continuation languages agree. Duplicate sentences keep
their multiplicity. Merging shared suffixes cannot splice in a new sentence.
The algorithm is deterministic and uses no fitted parameters or iterative loss
minimization.

The resulting weights have an explicit linear-algebra interpretation:

- `alpha` is a one-hot vector selecting the initial state.
- `T_t[s,u] = 1` when token `t` moves state `s` to `u`; other entries are zero.
- `beta[s]` counts sentence endings at state `s`.
- `m[s] = beta[s] + sum_t (T_t m)[s]` counts all accepted continuations. Acyclic
  state numbering makes this a single bottom-up computation.

For a consumed prefix, `h = alpha T_token0 ... T_tokenN`. The mass of the exact
sentence is `h beta`; its total continuation mass is `h m`. The next-token
probability is `h T_t m / (h m)`, and EOS probability is `h beta / (h m)`.
This explicitly identifies where every corpus symbol and continuation choice
resides in the constructed weights. It does not identify their locations in
the trained GPT-2 matrices.

The sparse format stores coordinates, integer masses, and dimensions. There is
no hidden copy of the corpus or hardcoded answer string in the runtime. The
state compaction is exact equality of weighted right languages, not a claim of
minimum real-valued weighted-automaton dimension. Spectral constructions instead
relate that dimension to Hankel rank; that is a distinct follow-up experiment.
[Spectral construction and derivation](https://borjaballe.github.io/papers/preprint-bclq13.pdf).

## Initial measured result (2026-09-23)

From the committed general-facts corpus and the tokenizer above:

| Quantity | Result |
| --- | ---: |
| Facts / exact autoregressive completions | 1,024 / 1,024 |
| Input tokens / supervised targets including EOS | 14,098 / 10,002 |
| Active vocabulary including EOS | 4,475 |
| Trie states / compacted states | 13,345 / 10,864 |
| Sparse transition nonzeros | 11,886 |
| Serialized automaton bytes | 403,421 |
| Construction time, excluding tokenization/verification | 0.0026 seconds |

These are a constructive baseline, not evidence that GPT-2 runs this automaton.
The sparse representation is much larger than the raw tokenized corpus; its
storage accounting includes indices and counts, unlike a floating-weight count.

## How much history must any suffix-only model see?

For each window size, group **supervised** next-token queries by their final
`min(window, prefix_length)` tokens. A short key retains the left boundary.
The irreducible error count is the sum of `count - largest_target_count` over
these groups. This is exact for deterministic suffix-window classifiers on this
evaluation set, not a statement about which history GPT-2 actually uses.

| Window | Unavoidable next-token errors out of 10,002 |
| --- | ---: |
| 1 | 4,804 |
| 2 | 1,273 |
| 3 | 208 |
| 4 | 62 |
| 5 | 20 |
| 6 | 3 |
| 7 | 2 |
| 8 | 1 |
| 9 | 0 |

Thus nine-token contexts suffice for an exact construction; there are 10,001
distinct such contexts. This identifies a useful experiment: compare the
trained model against this mechanism under earlier-prefix interventions while
holding token positions fixed. Out-of-distribution corruption must not be
mistaken for proof of a particular internal algorithm.

## What remains to explain

1. A compact neural-weight construction. The exact ReLU memory below is a
   verified control, but is not as compact as GPT-2 and uses a different
   architecture.
2. A causal bridge between the constructed representation and the actual
   checkpoint: layerwise recoverability, substitutions, and selective edits.
3. Whether a closed-form solve in fixed features can reproduce learned behavior,
   or whether changes to the features are essential.
4. Why optimization finds this representation. Fixed-feature least-squares
   gradient descent and a pseudoinverse have a known connection, but it does not
   automatically describe joint BF16 GPT-2 training with AdamW/cross-entropy.

Potential constructions include closed-form associative memories. Recent work
constructs fact-storing MLPs, but its transformer experiments still include
learned components and do not furnish an end-to-end GPT-2 compiler.
[MLPs are Hebbians](https://arxiv.org/abs/2607.10034),
[associative-memory transformer constructions](https://arxiv.org/abs/2412.06538).

## Explicit one-shot ReLU weights

Compile mode also builds and verifies a sparse two-hidden-layer ReLU network.
Its query `q` is the nine most recent token IDs, padded on the left with -1 for
short prefixes. For every distinct supervised query `k_i`, construct:

```
a_ij = ReLU(q_j - k_ij)
b_ij = ReLU(k_ij - q_j)
h_i  = ReLU(1 - sum_j(a_ij + b_ij))
z    = sum_i h_i * code(next_token_i)
```

Here `code(t)` is a fixed 16-dimensional +/-1 binary code for token ID `t`.
The first-layer sparse weights are +1/-1; its biases are the actual key token
IDs with opposite signs. The second layer has -1 weights and +1 biases. Output
weight columns contain the actual target codes. All stored values are FP32.

Distinct integer keys are separated by L1 distance at least one. Therefore
exactly the matching unit has activation one and every other unit is zero.
The resulting output is precisely its target code. Its dot product with the
matching vocabulary code is 16; every other code scores at most 14. This is a
finite-precision-safe, explicit weight construction, not an optimization or a
lookup call disguised inside the neural forward. Inference evaluates every
unit's ReLU arithmetic. The exact-code decoder extracts bits as an optimization
of the unique dot-product argmax; tests compare it against exhaustive decoding.

The model rejects queries with no active unit. Unlike the automaton, it can
accept new full prefixes whose final nine tokens match a compiled context.
It does not claim calibrated probabilities or equivalence off the corpus.

The real corpus run verified 1,024/1,024 autoregressive suffix-plus-EOS
completions after serializing and reloading these weights. Construction took
0.0027 seconds, excluding tokenization, context analysis, and verification.
There are 10,001 second-layer units and 340,034 stored corpus-dependent floats
(1,360,182 artifact bytes including metadata). This is about 3x the original
model's **trainable** scalar count. It additionally has 360,036 fixed sparse
weights, 10,001 fixed biases, and a generated 804,112-entry decoder codebook;
these are not stored and do not encode corpus facts. This is not a
parameter-efficiency result or a same-architecture GPT-2 checkpoint.

```sh
bazel-bin/src/llm/experiments/one_shot_memorizer/one_shot_memorizer \
  --mode=infer --backend=relu --tokenizer="$tokenizer" \
  --model_file=/tmp/one_shot_memorizer_new/relu.weights \
  --prompt='The capital of France is'
```

An elementary optimization connection holds **for this control**, not yet for
GPT-2. Freeze these indicator features and fit the output vectors with squared
error. With one equally weighted example per distinct key, the feature matrix
is the identity. The direct solution is exactly the target-code matrix `C`.
For objective `0.5 * ||W-C||^2`, full-batch gradient descent from zero follows
`W_s = (1-(1-eta)^s) C` for `0 < eta < 2`, converging to the same constructed
weights. Learning the features jointly, using cross-entropy, and using AdamW
change the problem; this formula must not be presented as their explanation.

## Does the actual checkpoint use only nine tokens?

`checkpoint_context_probe` holds each prediction's absolute position fixed and
replaces earlier tokens, independently for every scored target. Two corruptions
are tested: EOS filling, and tokens from the next corpus sentence, cycling if
needed. The unmodified control must get every target right. All transfers use
pinned host buffers, and all predictions use the existing cuTile top-1 kernel.

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_context_probe
facts_run=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_context_probe \
  --checkpoint="$facts_run/layers_8/step_16128" \
  --tokenizer="$facts_run/inputs/tokenizer" \
  --output_dir=/tmp/one_shot_context_probe_new
```

Measured errors on 2026-09-23, out of 10,002 independent next-token probes:

| Retained suffix | Prefixes eligible for replacement | EOS-fill errors | Other-sentence errors |
| --- | ---: | ---: | ---: |
| Entire prefix (control) | 0 | 0 | 0 |
| 1 token | 10,002 | 8,902 | 7,840 |
| 3 tokens | 10,002 | 8,512 | 7,547 |
| 5 tokens | 8,978 | 6,894 | 6,116 |
| 9 tokens | 4,903 | 2,638 | 2,381 |
| 12 tokens | 2,177 | 744 | 693 |
| 16 tokens | 345 | 58 | 59 |
| 24 tokens | 3 | 0 | 0 |

The single entire-prefix control is shared by both replacement comparisons.
Eligible positions are overwritten; donor tokens can coincide with originals.
There were no nonfinite logit rows. These are not autoregressive completion
failure counts: each token uses its own separately modified prefix.

**What this establishes:** the checkpoint is not invariant to history preceding
the sufficient nine-token suffix, whereas the constructed suffix model is.
**What it does not establish:** a nine-token model could not fit the corpus
(it does), which layer stores a fact, or that the extra history is indispensable
on natural inputs. Both replacement schemes create unnatural histories. The
next mechanistic step is layerwise intervention to explain this discrepancy,
not a claim that the construction already explains the learned weights.

## Tests

```sh
bazel test -c opt //src/llm/experiments/one_shot_memorizer/... --test_output=errors
```

Tests check duplicate weighting, ambiguity, EOS, exact compaction without new
sentences, corpus-order invariance, very long inputs, exhaustive small weighted
corpora, context conflicts, serialized round trips, malformed weights, and
truncation at every byte. The corpus-wide verification is a separate real-data
run above, not inferred from small unit tests.
