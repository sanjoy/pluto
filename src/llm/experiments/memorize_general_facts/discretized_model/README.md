# Discrete general-facts model

This experiment compiles recorded executions of a trained GPT-2 model into a
CPU-only network of symbolic transitions. It specializes the model to the
recorded corpus; it does not reproduce the neural model on arbitrary text.

## Current model

The checked-in package was generated from the smallest fully memorized model
found in the context-32 sweep:

- Checkpoint: `context32_size_sweep_1h_0/trial_000_L4_W13_FF52/checkpoints/layers_4/step_34816`.
- Architecture: four transformer blocks, width 13, one attention head, MLP
  width 52, context length 32, and 67,405 trainable parameters.
- Vocabulary: 4,475 compact GPT-2 tokens, including EOS.
- Task: supply exactly the first five tokens of each of the 1,024 facts, then
  generate the complete suffix and explicit EOS.
- Native GPU and generated CPU verification: **0 errors across 10,002 scored
  predictions; all 1,024 sentences and EOS predictions are correct**.

The exact capture contains **114,145 distinct internal states**, from 126,882
state occurrences (14,098 real token positions across nine boundaries).
A bounded compaction pass reduces this to **6,249 states**, a **94.53%**
reduction. The 4,475 vocabulary symbols are separate from that count.

| Boundary | After attention + residual | After MLP + residual |
| --- | ---: | ---: |
| Token + position embedding | 85 | — |
| Block 0 | 64 | 58 |
| Block 1 | 61 | 57 |
| Block 2 | 72 | 52 |
| Block 3 | 2,900 | 2,900 |

Compaction stopped at its one-pass limit: **neither pairwise irreducibility nor
global minimality is claimed**. State count measures the symbolic alphabet,
not the information stored in the history-dependent attention functions.

The previous eight-block, width-16 generated model and its 6,514-state result
remain in Git history, including the local backup branch
`codex/discretize-general-facts-before-context32-rebase`. Its measurements and
exhaustive compaction certificate do not apply to this new model.

## Build, verify, and run

```sh
bazel build -c opt //src/llm/experiments/memorize_general_facts/discretized_model/generated:discretized_model

discrete=bazel-bin/src/llm/experiments/memorize_general_facts/discretized_model/generated/discretized_model
"$discrete" --verify
"$discrete" --prompt='The capital of France is'
```

Generated inference uses integer transitions only. It requires no CUDA,
checkpoint, tokenizer installation, or GPU.

The CLI's `--prompt` encoder accepts recorded corpus prefixes at token
boundaries; it is not a general BPE tokenizer. `--token_ids=ID,ID,...` accepts
compact token IDs directly. Unsupported transition keys return an error,
without a neural fallback. After compaction, an unseen raw-token sequence can
sometimes map to supported abstract histories, so rejection of every
out-of-corpus sequence is not guaranteed.

## Preserved layer boundaries

The compiled model executes these pure operations in order:

1. `(vocabulary token, absolute position) -> input hidden state`.
2. For each block, `ordered causal prefix -> attention hidden state`, including
   pre-LayerNorm, Q/K/V projections, attention, output projection, and residual
   addition.
3. `attention hidden state -> MLP hidden state`, including pre-LayerNorm, MLP,
   and residual addition.
4. `final hidden state -> vocabulary token`, including final LayerNorm and
   the tied language modeling head's deterministic top-1 selection.

Every block remains present. No compaction crosses a boundary or replaces the
network with a sentence-completion dictionary. Attention receives the entire
ordered prefix, never a sentence ID. Expected suffixes are verification
fixtures, not prediction tables consulted by the model.

Initially, hidden-state symbols identify exact BF16 residual vectors. After
compaction they identify classes that preserve the required corpus outputs,
not numerical vectors or every possible behavior of the original model.
Optional vector hints only order compaction candidates. Acceptance is symbolic:
colliding transition keys force compatible downstream outputs, and a proposal
that would equate distinct required vocabulary outputs is rejected.

`--compact_transitions` separately condenses the finite transition functions.
It shares attention-prefix/suffix programs and uses range checks, masks, or
arithmetic where equivalent. This does not perform additional state compaction.
For this model, the final MLP and language modeling head can use vocabulary-
aligned state IDs; the earlier MLPs retain explicit maps. Such arithmetic
patterns follow from the chosen symbolic names, not proof that the original
neural layers are affine.

## Generate from the checkpoint

The C++ generator loads the checkpoint and tokenizer, records exact GPU
executions, compacts states in memory, and emits formatted CPU-only C++.
The output directory must not already exist.

```sh
sweep=/home/ubuntu/checkpoints/memorize_general_facts/context32_size_sweep_1h_0

bazel run -c opt //src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator:generate_discretized_model -- \
  --checkpoint="$sweep/trial_000_L4_W13_FF52/checkpoints/layers_4/step_34816" \
  --tokenizer="$sweep/inputs" \
  --corpus="$sweep/inputs/corpus.txt" \
  --layers=4 --model_width=13 --context_length=32 \
  --attention_heads=1 --feed_forward_width=52 \
  --prompt_tokens=5 --expected_samples=1024 --verify_greedy=true \
  --compaction --compaction_neighbors=4 --compaction_max_passes=1 \
  --compaction_max_attempts=150000 --compaction_exhaustive_pair_limit=0 \
  --compact_transitions --state_index \
  --output=/tmp/facts-context32-generated
```

This reproduces the checked-in search configuration. Omit `--compaction` to
keep the exact captured states. Omit `--compact_transitions` to use private
lookup tables behind the same pure function interfaces. The model dimensions
must match the checkpoint; legacy defaults remain eight blocks, width 16,
context 1,024, one head, and MLP width 64.

Native autonomous completion and bitwise causal-prefix checks are enabled by
default. For each generated prefix, every captured residual vector must match
its full-sentence counterpart exactly. Symbolic verification runs before and
after compaction. Generated transition tests independently exercise every
captured operation with its expected input, preventing compensating errors
between layers from passing unnoticed.

Generation requires CUDA and `clang-format`. Formatting uses the repository's
Google-style configuration, supplied through Bazel runfiles. Source files are
split by attention/MLP boundary for parallel compilation. Output publication is
atomic and refuses existing paths, including symlinks. Copy the resulting
package into `generated/` explicitly; there are no Bazel genrules or automatic
rewrites of checked-in sources.

## Inspect and extend

`generated/generation_report.txt` records checkpoint/corpus/tokenizer hashes,
verification results, compaction limits and results, and generated-file hashes.
Optional inspection-only files provide:

- `state_index.tsv`: boundary, occurrence count, observed prefix examples,
  and original member count for each state.
- `state_members.tsv`: the complete mapping from original exact states to
  compacted classes.
- `state_relabeling.tsv`: the subsequent within-boundary renaming used by
  condensed transition code.

These files are not loaded by inference. Examples are observed contexts, not
asserted semantic meanings. `vocabulary_tokens.h` gives tokens readable private
names such as `vocab::kSpace_France_1516`, preserving their original bytes and IDs.

The generated library exposes only `gen::GeneratedModel()` through
`pluto/discretized/gen/model.h`. It returns a static `DiscreteModel` with a
position embedding, a span of transformers, and a language modeling head.
Operations return `std::optional<DiscreteHiddenState>`; an absent value means
unsupported input, whereas an engaged zero is valid. The runtime does not
depend on a particular generated model. CLI prompt encoding and verification
support are separate from the production model library.

Generation code lives in
[`discrete_model_generator/`](discrete_model_generator/README.md), organized
around `CapturedModel`: `ModelRecorder` captures it, `CompactModel` produces
a compacted model, and `RenderModel` lowers it to independently compiled C++.
Only recording needs CUDA; the generated model does not depend on the generator.

```sh
bazel test -c opt //src/llm/experiments/memorize_general_facts/discretized_model/...
```
