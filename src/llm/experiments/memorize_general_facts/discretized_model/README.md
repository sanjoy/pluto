# Discrete general-facts model

This experiment compiles recorded executions of a trained GPT-2 model into a
CPU-only network of symbolic transitions. It specializes the model to the
recorded corpus; it does not reproduce the neural model on arbitrary text.

## Current model

The checked-in package was generated from the current smallest verified model
configuration, using the identity baseline of the canonical-token-order rerun:

- Checkpoint: `dataset_weights_canonical_order_0/baseline/checkpoints/layers_4/step_120000`.
- Architecture: four transformer blocks, width 10, one attention head (head
  dimension 10), MLP `10 -> 20 -> 10`, context length 27, and 48,680 trainable
  parameters. Blocks use pre-LayerNorm, causal attention, GELU, learned absolute
  positions, and a token-embedding-tied language modeling head. Capture uses
  BF16 activations and the checkpoint's FP32 master weights.
- Vocabulary: 4,475 compact GPT-2 tokens, including EOS.
- Task: supply exactly the first five tokens of each of the 1,024 facts, then
  generate the complete suffix and explicit EOS.
- Native GPU and generated CPU verification: **0 errors across 10,002 scored
  predictions; all 1,024 sentences and EOS predictions are correct**.

The exact capture contains **113,132 distinct internal states**, from 126,882
state occurrences (14,098 real token positions across nine boundaries).
A bounded within-boundary compaction pass reduces this to **6,069 states**;
an exhaustive pairwise continuation reduces that to **6,053 states**.
Pairing each bijective MLP's input and output reduces that further to **3,044
distinct state IDs**. The 4,475 vocabulary symbols are separate from that count.

| Boundary | After attention + residual | After MLP + residual |
| --- | ---: | ---: |
| Token + position embedding | 35 | — |
| Block 0 | 35 | 35 |
| Block 1 | 37 | 37 |
| Block 2 | 37 | 37 |
| Block 3 | 2,900 | 2,900 |

Each block's two columns now refer to the **same IDs**, not two disjoint sets.
Thus the unique count is `35 + 35 + 37 + 37 + 2900 = 3044`; the per-boundary
counts still sum to 6,053 because a paired symbol appears on both sides.

Compaction stopped only after a complete exhaustive sweep accepted no pair:
**no further same-boundary pair can be combined under this partition while
preserving the required outputs**. This is not proof of a globally smallest
partition; earlier compaction choices could lead to a different result. State
count measures the symbolic alphabet, not the information stored in the
history-dependent attention functions.

The nearest-candidate pass alone left 6,117 states. An exact pointwise pass
then identified MLP inputs with equal current outputs, removing 17, 16, and 15
states from the first three attention-output boundaries: **48 additional
compactions**, with no changed MLP outputs. The resulting MLP maps are one-to-one
on their supported inputs. The exhaustive continuation then removes **16
embedding states** and leaves every transformer boundary's count unchanged.
A final pass identifies each pair `x, MLP(x)`,
removing **3,009 additional IDs**. Every MLP remains an explicit guarded identity:
unknown inputs still fail rather than passing through unchecked.

### Exhaustive search cost

The exhaustive continuation starts from the same 6,069-state partition as the
previous checked-in model, before its MLP input/output pairing. It tries every
same-boundary pair not already contradicted by fixed vocabulary outputs. A
trial propagates any forced downstream combinations; conflicting vocabulary
labels reject it and roll back the entire trial. Successful trials are kept,
and full sweeps repeat until one accepts nothing. There is no attempt limit.

That starting partition has 8,412,229 same-boundary pairs. The final two
boundaries each contain 2,900 states with distinct fixed readout labels, so
8,407,100 pairs are immediately impossible. Only **5,129 pairs** need search.
Rejected pairs remain impossible after further compaction and are cached;
successful pairs disappear. Therefore 5,129 bounds the number of new uncached
trials across the entire continuation, not merely one sweep. A loose bound on
repeated traversal is 263 sweeps / 1,348,927 candidate visits: each productive
sweep removes at least one of the at most 262 removable early-boundary states.
Most repeated visits would just check the rejection cache.

On the current machine the continuation took **14.717 seconds**, with 2,060
new uncached trials, 16 accepted pairs, and two exhaustive sweeps (the second
accepted nothing). The complete capture/compaction/formatting/generation command
took **31.21 seconds**, excluding subsequent compilation and tests. All 10,002
predictions and 1,024 complete suffixes/EOS remain correct; all 113,132 original
vectors remain archived. These are measured times, not a wall-clock guarantee:
individual trial costs depend on how many transitions their implications touch.
At the measured average trial cost, 5,129 uncached trials would take roughly
37 seconds; a few minutes is a conservative planning budget, not a mathematical
worst-case time bound.

An independent backward audit of the generated boundary fixtures also found
contradictory-history witnesses for all 2,522 pairs at the attention-input
boundaries. Together with the complete injective MLP maps and distinct final
readout labels, this distinguishes all 8,411,549 remaining boundary-specific
pairs without consulting the search's rejection cache. Shared MLP IDs are
treated separately at their two observation boundaries for this argument.

Previous generated models remain in Git history: the width-13/context-32 model
had 6,249 compacted states, and the eight-block/width-16 model had 6,514. Their
measurements and any exhaustive compaction certificate do not apply to this
new model.

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

## Print the vectors represented by a state

The generated model preserves every distinct originally captured BF16 residual
vector. Compaction records their membership in each resulting hidden state;
subsequent state renaming preserves that membership too. A paired MLP state
contains both pre-MLP and post-MLP vectors, each labeled with its original
boundary. Sharing a symbol does **not** assert numerical equality of those
activations. These are original
vectors, not centroids or nearest neighbors, and repeated corpus occurrences of
the same exact vector do not produce repeated entries.

Use a current hidden-state ID from `generated/state_index.tsv`:

```sh
"$discrete" --print_state=4475
```

The output lists all original vectors belonging to that state, with their
original state IDs and exact BF16 encodings alongside readable component values.
Vector length follows the captured model width (10 for the checked-in model),
not a fixed width of 16. Negative or unknown hidden-state IDs return an error.
Vocabulary token IDs are not hidden-state IDs and cannot be printed this way.
`--print_state` is separate from prompt generation and verification.

C++ callers can select their own output stream through the model's callback:

```cpp
const auto& model = pluto::llm::discretized::gen::GeneratedModel();
auto status = model.print_state(
    pluto::llm::discretized::DiscreteHiddenState{4475}, std::cout);
```

The vector archive is compiled into the generated package in separate source
files for parallel compilation. It is inspection-only: inference still follows
integer transitions and never reads these vectors. No checkpoint, GPU, or
external vector file is needed to print a state. Handwritten `DiscreteModel`
instances may leave the callback empty when no captured vectors are available.

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

Every block and MLP call remains present. Within-boundary compaction keeps
boundary alphabets separate; the explicit final MLP-pair pass shares symbols
only across a bijective MLP's input/output boundary. It never combines different
blocks or attention inputs with their outputs, and never replaces the network
with a sentence-completion dictionary. Attention receives the entire
ordered prefix, never a sentence ID. Expected suffixes are verification
fixtures, not prediction tables consulted by the model.

Initially, hidden-state symbols identify exact BF16 residual vectors. After
compaction they identify classes that preserve the required corpus outputs,
not numerical vectors or every possible behavior of the original model.
Optional vector hints only order compaction candidates. Acceptance is symbolic:
colliding transition keys force compatible downstream outputs, and a proposal
that would equate distinct required vocabulary outputs is rejected.

After each nearest-candidate pass, exact pointwise compaction groups each
MLP's input states by their current output class. Inputs in one group have
the same sole consumer and are interchangeable without any distance heuristic.
The language modeling head is handled first, followed by MLPs in reverse order,
so equal head labels can expose additional final-MLP compactions. This shares
the overall attempt budget and cancellation rules. It does not identify input
and output boundaries, and it does not infer that individual attention-history
symbols are interchangeable from an equal output in just one context.

`--compact_transitions` separately condenses the finite transition functions.
Attention uses independent per-output predicates, described below; pointwise
maps and position embeddings use range checks, masks, or arithmetic where
equivalent. This does not perform additional state compaction. Before pairing,
this exposes guarded offsets for the MLPs. The final
`--mlp_pair_compaction` pass then shares each MLP input/output pair and updates
every downstream consumer consistently. It requires complete bijections and
fails rather than approximating non-bijective or incomplete maps. The MLPs
become identities on their existing support; the final shared states remain
vocabulary-aligned for the language modeling head. These arithmetic patterns
follow from symbolic names, not proof that the neural layers are affine or
identities. No further within-boundary compaction/relabeling runs after pairing.

### Generated attention predicates

Each attention layer defines one `bool MatchStateNNNN(history)` function for
each supported output state. Its dispatcher calls these predicates in ascending
state-ID order and returns the first matching state, or `std::nullopt` if none
matches. Each predicate independently recognizes exactly the captured symbolic
histories for its output; it assumes nothing about earlier predicates having
failed and can be called on its own.

A predicate switches on history length. Each supported length contains a flat
`return Match(history, {...}) || Match(history, {...}) || ...;` expression, with
complete captured histories in lexicographic order. There are no nested
per-position switches or guards. Length grouping skips incompatible literals;
the flat expression favors readable alternatives over a decision-tree heuristic.

The generated `Match` helper takes read-only spans, checks lengths, and compares
the values in their original positional order. **Every accepting path checks
every element**. A few positions uniquely identifying one recorded example do
not validate arbitrary values in its remaining positions. Unknown symbolic
histories therefore remain unsupported, rather than being assigned the closest
recorded output.

The separate predicates make one output's requirements directly inspectable.
The tradeoff is that predicates may repeat checks and code that the previous
shared-prefix/suffix representation reused, and the dispatcher may test many
predicates before finding a match. Readability does not imply smaller source,
faster compilation, or faster inference. This representation changes no state
IDs or transition semantics.

## Generate from the checkpoint

The C++ generator loads the checkpoint and tokenizer, records exact GPU
executions, compacts states in memory, and emits formatted CPU-only C++.
The output directory must not already exist.

```sh
run=/home/ubuntu/checkpoints/memorize_general_facts/dataset_weights_canonical_order_0

bazel run -c opt //src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator:generate_discretized_model -- \
  --checkpoint="$run/baseline/checkpoints/layers_4/step_120000" \
  --tokenizer="$run/inputs/tokenizer" \
  --corpus="$run/inputs/corpus.txt" \
  --layers=4 --model_width=10 --context_length=27 \
  --attention_heads=1 --feed_forward_width=20 \
  --prompt_tokens=5 --expected_samples=1024 --verify_greedy=true \
  --compaction --compaction_neighbors=4 --compaction_max_passes=1 \
  --compaction_max_attempts=-1 \
  --compaction_exhaustive_pair_limit=9223372036854775807 \
  --compact_transitions --mlp_pair_compaction --state_index \
  --output=/tmp/facts-context27-width10-generated
```

This reproduces the checked-in search configuration. Omit both `--compaction`
and `--mlp_pair_compaction` to keep the exact captured states. Omit only
`--mlp_pair_compaction` to retain separate input/output alphabets for the MLPs.
Omit `--compact_transitions` to use private
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

- `state_index.tsv`: boundary (or paired boundaries), occurrence count, observed
  prefix examples, and original member count for each state. Shared-state
  occurrence counts sum observations at both boundaries.
- `state_members.tsv`: the complete mapping from original exact states to
  compacted classes.
- `state_relabeling.tsv`: within-boundary renaming when MLP pairing is disabled.
  Pairing invalidates that intermediate map, so the paired package omits it;
  `state_members.tsv` always describes the final state IDs.

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
