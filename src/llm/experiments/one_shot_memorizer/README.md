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

The runtime rejects unseen prefixes. Ambiguous corpus prefixes use empirical
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

1. A compact, explicit neural-weight construction, not only sparse automaton
   matrices. An exact ReLU memory is a useful control, but its architecture and
   parameter budget must be stated honestly.
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

## Tests

```sh
bazel test -c opt //src/llm/experiments/one_shot_memorizer/... --test_output=errors
```

Tests check duplicate weighting, ambiguity, EOS, exact compaction without new
sentences, corpus-order invariance, very long inputs, exhaustive small weighted
corpora, context conflicts, serialized round trips, malformed weights, and
truncation at every byte. The corpus-wide verification is a separate real-data
run above, not inferred from small unit tests.
