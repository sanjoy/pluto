# Discrete general-facts model

This experiment compiles a finite set of executions of the trained width-16
GPT-2 model into an integer-only network. It is a specialization to the recorded
corpus, not an exact replacement for the neural model on arbitrary text.

## Agreed task

- Source: `compact_batch_32_no_clip_0/layers_8/step_16128`, eight transformer
  blocks, width 16, one attention head, MLP width 64, BF16 computation.
- Corpus: `testdata/general_facts_dataset.txt`, 1,024 independent sentences.
- Supply the first five tokens, generate every remaining token, and explicitly
  predict EOS. All 10,002 scored predictions must be correct.
- Unknown lookup histories fail explicitly. There is no neural fallback.
- Work is isolated on `codex/discretize-general-facts`, based on `main` at
  `a2623e0`; verified milestones are committed and pushed there.

## Representation

Vocabulary symbols have their original token bytes and GPT-2 IDs as labels.
`generated/vocabulary_tokens.h` gives every compact token an `inline constexpr
DiscreteToken` name in the private `gen::internal::vocab` namespace. Generated
implementations use names such as `vocab::kThe_216`,
`vocab::kSpace_France_1516`, and `vocab::kEos_4474`. Names preserve case and spell
out spaces/punctuation; other bytes use `ByteXX`. The compact-ID suffix makes
names unique even when a long name is shortened. Comments retain the exact token
bytes and original GPT-2 IDs. Entry/readout tables, EOS configuration, and token
arrays use these constants; vocabulary IDs are unchanged. The optional control-
flow pass renames internal symbols, with a complete mapping recorded below.
Initially, internal symbols are numeric IDs for exact native BF16 residual
vectors. In the reduced model they identify equivalence classes of those states.
There are 17 internal boundaries: summed token/position embeddings, then the
attention and MLP residual outputs of each of eight blocks. Each boundary has
its own state alphabet; IDs are globally distinct.

The compiled operations are:

1. `(vocabulary ID, absolute position) -> input state`.
2. For each block, `ordered causal prefix of input states -> attention state`.
   This includes pre-LayerNorm, Q/K/V, attention, output projection, and residual
   addition. The prefix length supplies the position; no corpus-line ID is used.
3. `attention state -> MLP state`, including pre-LayerNorm, MLP, and residual.
4. `final state -> vocabulary ID`, capturing final LayerNorm and the tied head's
   actual deterministic argmax, not an approximate nearest-embedding rule.

Capture uses existing layer hooks and transfers only real input rows. Padding
is not part of the state space. The corpus has 14,098 real token positions, so
there are 239,666 internal-state occurrences before exact deduplication.

## Generation and verification plan

The GPU capture records exact activation bits and native top-1 predictions.
A separate generator checks every duplicate key, emits ordinary C++ arrays or
condensed transition functions in multiple source files, and formats them. There
are no Bazel generation rules. Generated source is checked into `generated/`.
Inference uses integer transitions only; it does not load weights or use CUDA.
Expected sentence suffixes
are verification fixtures, never prediction tables consulted by the runtime.

Tests cover table consistency, unknown keys, causal history ordering, EOS,
code emission, and merge rollback. End-to-end validation checks both the native
checkpoint and the compiled model with autonomous first-five-token generation.

## State reduction

After the exact baseline passes, states at the same boundary are proposed for
merging. Proximity orders the initial search but is not a restriction: any pair
may merge if it preserves the required outputs. When rewritten attention or MLP
keys collide, their output states
must also merge. This congruence closure propagates downstream. A proposal is
rejected if it equates two different required vocabulary outputs; otherwise the
quotient remains a deterministic lookup network. Original vectors provide
distance/provenance, not floating-point inference after merging.

Only scored suffix/EOS readouts constrain the reduction: predictions inside the
supplied five-token prompt are not part of the task. All prompt hidden states
remain necessary as attention context. Every accepted merge decreases the state
count. Search reports must distinguish exhaustive pairwise irreducibility from
merely exhausting a nearest-neighbor candidate set; neither alone proves a
globally smallest representation.

No merge crosses a boundary, removes a layer, or bypasses an attention/MLP
transition. This is a study of a layered symbolic representation, not a replacement
of the whole model with a sentence-completion dictionary.

## Exact baseline milestone

The exact baseline is preserved in commit `3e983ab`. Exact deduplication yields
**221,558 internal states**, plus 4,475 vocabulary
symbols. The native checkpoint and the compiled C++ network both pass all
**10,002 predictions, 1,024 complete sentences, and 1,024 explicit EOS checks**.
For every generated prefix, all 17 native BF16 boundaries exactly match the
full-sentence capture. Repeating capture in a new process reproduced identical
predictions and BF16 boundary states.

The generated package is approximately 16 MiB of source, split into independent
attention/MLP translation units. Both its Bazel dependency graph and its dynamic
library list contain **no CUDA dependency**. It needs no checkpoint, tokenizer
installation, or GPU at inference time. All 72 repository Bazel test targets and
186 general-facts Python tests passed at this milestone.

## Reduced result: 6,514 internal states

The checked-in generated package now has **6,514 internal states** (**97.06%
fewer** than the exact baseline), with the same 4,475 vocabulary symbols. Its
compiled CPU verifier still reports 0/10,002 errors, 1,024/1,024 exact sentences,
and 1,024 explicit EOS predictions. The reduction accepted 89,345 seed merges,
including their forced downstream merges, eliminating 215,044 original states.

The final exhaustive sweep found **no compatible within-boundary pair**. This
includes distant pairs: nearest-neighbor search only ordered the initial work.
Pairs whose terminal vocabulary labels already differ are provably incompatible
and can be skipped. The historical certificate in `generated/generation_report.txt` records
`no_compatible_pair`, `pairwise_irreducible: true`, and
`global_minimum_proven: false`. A different earlier merge order could produce a
different, potentially smaller quotient; this is not a global minimum claim.

A separate checker, which imports no reducer code and trusts no cached search
results, independently proves that **all 8,423,754 same-boundary pairs** are
incompatible. It works backward from distinct vocabulary labels through
injective MLP tables and 8,990 explicit attention-input pair collision checks.
The complete argument summary is also recorded in the generation report.
The C++ generator runs this independent checker directly on the in-memory
quotient when reduction reports pairwise irreducibility. The checker
reports `inconclusive` if its sufficient backward argument cannot be completed;
it never interprets a missing proof as success or proves corpus accuracy by
itself. The separate autoregressive verifier establishes that accuracy.

The input/position boundary has 52 states. Subsequent boundaries are:

| Block | After attention + residual | After MLP + residual |
| --- | ---: | ---: |
| 0 | 48 | 48 |
| 1 | 48 | 48 |
| 2 | 48 | 48 |
| 3 | 48 | 48 |
| 4 | 47 | 47 |
| 5 | 47 | 47 |
| 6 | 45 | 45 |
| 7 | 2,900 | 2,900 |

There are 2,900 distinct required next-token labels. Both final boundaries have
reached that lower bound: the final MLP and language modeling head are pointwise
functions, so different required labels cannot share an input state. Earlier
attention tables can expand a small alphabet into many outputs by inspecting
ordered histories.
These counts measure state alphabets, not the information or bytes in the
history-keyed tables; reducing states is not the same as compressing the model.

All original 221,558 states are accounted for by the quotient membership map.
The generator preserves all eight attention transitions, all eight MLP transitions,
the position-entry function, and the language modeling head. The generated code
has boundary/row-layout comments and readable, byte-exact vocabulary literals.

Two files support inspection without affecting inference:

- `generated/state_index.tsv`: boundary name, representative BF16 vector,
  corpus occurrence count, up to three observed prefix examples, and original
  member count for each numeric symbol. Examples are observations, not asserted
  semantic labels; a representative is just one member of a merged class.
- `generated/state_members.tsv`: the complete original-state ID membership of
  every class. Original IDs refer to the exact baseline, not to a different
  layer or a vocabulary token. No merge crosses a boundary.

Generate these with `--state_index`. Reduction preserves the complete quotient
mapping from the originally captured states in memory; it needs no saved
baseline or intermediate model file. An unreduced model needs no membership
table because each symbol still represents exactly one original state.
The inspection files are not compiled, linked, or read by the inference model.
Merged states preserve the agreed corpus completions, not numerical vectors or
all possible neural-model behavior. A class can group unrelated meanings; its
example contexts are evidence for further interpretation, not semantic proof.
The final result passes all 73 repository Bazel test targets and all 247
general-facts Python tests. Its executable and build dependency graph remain
CUDA-free, and generated C++ is formatted with the repository's Google style.

```sh
bazel build -c opt //src/llm/experiments/memorize_general_facts/discretized_model/generated:discretized_model

discrete=bazel-bin/src/llm/experiments/memorize_general_facts/discretized_model/generated/discretized_model
"$discrete" --verify
"$discrete" --prompt='The capital of France is'
```

`--prompt` uses a finite-domain encoder for recorded corpus prefixes at token
boundaries. It is intentionally not a general BPE tokenizer. Alternatively,
`--token_ids=ID,ID,...` supplies compact IDs directly. The model's predictions do
not consult this text encoder or the separately compiled verification fixtures.
Unknown entries, attention histories, MLP inputs, and readouts are errors.
After merging, a previously unseen raw-token prefix may map to known abstract
lookup keys; the lookup mechanism does not promise to reject every out-of-corpus
token sequence. The text encoder deliberately accepts only recorded prefixes.

## Transition patterns and condensed control flow

`--compact_transitions` compiles the same finite functions into smaller programs.
It preserves their support and outputs under the recorded symbol renaming. It
does **not** merge more states, remove layers, or use cross-layer prediction
shortcuts. All 6,514 residual classes remain, with
their BF16 representatives and original memberships. The generator records the
within-boundary renaming in `generated/state_relabeling.tsv`; numeric IDs can now
have gaps. Private recognizer program counters are code locations, not additional
activation classes.

Patterns found:

- **MLPs are bijections between boundary alphabets.** For blocks 0–6, choosing
  output names in input order makes each MLP a range check plus a constant
  addition. For example, block 0 accepts 4527–4574 and returns `state + 48`.
  The separate MLP function is still called at every position.
- **The final MLP and language modeling head are bijective onto their required token labels.**
  Name the last attention states `5189 + token_id` and final states
  `9664 + token_id`. The final MLP adds 4475; the language modeling head subtracts
  9664. Each has its own 560-byte support mask to reject the 1,575 unused labels.
  These masks are essential: a range check alone would accept states that never
  existed.
- **Entry is nearly token-only on this quotient.** Of 4,474 input tokens, 4,466
  have the same entry symbol at all observed positions. Only eight need one
  positional exception each. Deduplicated `(position-support mask, default
  symbol)` patterns plus those explicit exceptions replace 8,175 triples.
- **Attention shares prefixes and suffix programs.** Its 84,191 histories form
  prefix-closed tries; bottom-up minimization yields 59,300 recognizer nodes.
  Most nodes have one outgoing edge. Generated code groups branching cases,
  shares identical tails, and checks longer straight-line runs with small typed
  `(expected symbol, current output)` sequences. A shared matcher avoids
  expanding every comparison into separate machine instructions. Helpers have
  at most 256 recognizer nodes, keeping compilation practical and parallel across
  the eight source files. The complete ordered history is still checked.

These are **patterns in the finite symbolic quotient, not evidence that the
original neural MLPs or head are affine**. Arithmetic offsets arise from our
choice of names. The attention compression is structural sharing, not a newly
discovered semantic rule. Simple current-state rules were insufficient: their
majority outputs made 3,850–8,611 errors per attention block; adding sequence
length or a short suffix did not fix this. No approximate rule was adopted.

### Measured size

Same machine and `bazel build -c opt`, comparing the named table version
(`b076f03`) against the compact functions at `336a535`. The following counts are
`size`'s text + data + BSS totals for each non-PIC model object: allocated code,
read-only constants and runtime data, **not** object-file metadata, debug symbols,
or the test fixtures. All amounts are bytes.

| Boundary | Tables | Compact logic | Reduction |
| --- | ---: | ---: | ---: |
| Entry | 98,156 | 14,772 | 85.0% |
| Attention 0 | 535,580 | 156,446 | 70.8% |
| Attention 1 | 530,184 | 154,568 | 70.8% |
| Attention 2 | 527,432 | 158,918 | 69.9% |
| Attention 3 | 524,676 | 157,348 | 70.0% |
| Attention 4 | 523,028 | 154,722 | 70.4% |
| Attention 5 | 519,740 | 158,290 | 69.5% |
| Attention 6 | 518,104 | 162,080 | 68.7% |
| Attention 7 | 516,664 | 135,434 | 73.8% |
| MLP 0–3, each | 440 | 132 | 70.0% |
| MLP 4–5, each | 432 | 132 | 69.4% |
| MLP 6 | 416 | 132 | 68.3% |
| MLP 7 | 23,256 | 744 | 96.8% |
| Language modeling head | 23,256 | 728 | 96.9% |
| **All transition objects** | **4,343,116** | **1,254,974** | **71.1%** |

The corresponding formatted production transition source shrinks from
7,482,106 to 3,012,829 bytes (**59.7%**). Vocabulary strings, the text-prefix
encoder, runtime scaffolding, and independent test fixtures are excluded from
these transition-only totals. `generated/transition_patterns.txt` records the
chosen representation and source statistics for each function. Some small data
arrays deliberately remain: replacing irregular masks/sequence literals with
more branches made the executable larger. A pure-control-flow prototype was
also correct but larger than the selected shared-sequence version.
The complete CLI's allocated sections, including its unchanged vocabulary,
prompt encoder, and corpus-verification support, shrink from 6,520,292 to
3,480,820 bytes (46.6%).

### Equivalence checks

In addition to all 1,024 autonomous completions, a separate C++ test reconstructs
histories from 239,666 independently computed expected boundary states. Every
operation receives the **source table's** inputs, not the preceding operation's
actual outputs, so compensating mistakes cannot pass. Distinct-key coverage was
checked against all 98,497 source entry, attention, MLP, and language modeling
head records.
Additional tests exercise unsupported histories, sparse-domain holes, truncated
and extended sequences, zero outputs, and 32-bit limits. These fixtures are in a
dedicated test target, not linked by the production model or CLI.

To generate a compact representation directly from the native checkpoint:

```sh
bazel run -c opt //src/llm/experiments/memorize_general_facts/discretized_model:generate_discretized_model -- \
  --checkpoint=/path/to/run/layers_8/step_16128 \
  --tokenizer=/path/to/run/inputs/tokenizer \
  --corpus=testdata/general_facts_dataset.txt \
  --reduce --compact_transitions --state_index \
  --output=/tmp/facts-compact-generated
```

The output path must be fresh. Omit `--compact_transitions` to emit private
per-boundary tables behind binary-search lookup functions. Both modes implement
the same runtime interfaces:

- `PositionEmbedding` maps a compact token and signed absolute position to a
  token-plus-position residual symbol.
- `CausalAttention` maps a complete ordered causal prefix to the current
  position's attention residual symbol.
- `Map` maps one hidden state to another; MLPs and the language modeling head
  implement it. The head's output encodes the compact next-token ID.
- Each `Transformer` pairs attention and MLP references. `DiscreteModel::transformers`
  holds these blocks in execution order, with separate references to the
  position embedding and language modeling head.

All operations are pure and return `std::optional<DiscreteHiddenState>`:
`nullopt` rejects unsupported inputs, whereas an engaged zero is a valid token
ID. The model borrows its operations; generated implementations have static
lifetime. Reference members cannot be null. Generation validates transition
domains, and tests compare virtual calls against the source records.

### Public generated API

`generated/model.h` is the generated library's only public header. It declares
only `pluto::llm::discretized::gen::GeneratedModel()`, which returns a reference
to the static `DiscreteModel`. The generic `runtime.h` contains no generated
factory declaration and does not depend on any particular generated model.

Depend on the `generated:model` target and include its stable public header:

```cpp
#include "pluto/discretized/gen/model.h"

const auto& model = pluto::llm::discretized::gen::GeneratedModel();
```

Boundary factories, named vocabulary constants, prompt encoding, and corpus
verification live in `gen::internal`. Their headers and build targets are
private to the generated package; only the model factory is exported by its
shared library. The CLI links prompt encoding and verification separately, so
production model inference cannot access those fixtures.

All generation logic and its C++ tests
live alongside this README: the driver, C++ emitter, state reduction,
irreducibility checker, and compact-transition helpers. Generated C++ stays in
`generated/`. Benchmarking and training-sweep utilities remain in
`scripts/memorize_general_facts/`.

## Single-step GPU conversion

Use a fresh output path. The C++ `generate_discretized_model` binary loads the
checkpoint and tokenizer, captures exact BF16 activations on CUDA, constructs
and optionally reduces the symbolic network in memory, then emits formatted
CPU-only C++ in a single process. The existing tokenizer's `tokenizer.json` is
an input asset loaded by the tokenizer library.

`Generate(options)` is the public conversion entry point; its model-processing
continuation is private to the driver. `generator_model.h` defines the typed
in-memory representation: `ExecutionSample` holds observed BF16 rows,
`SymbolicModel` contains metadata and boundary-specific transition records, and
dedicated structures describe verification, reduction, and relabeling results.
Progress callbacks receive a `ProgressEvent` variant. Reports format these
structures directly as text; the algorithms never parse report strings.

`clang-format` must be installed and on `PATH`. Its repository configuration is
included in the executable's runfiles, so generation does not depend on running
inside the checkout. Relative input/output paths passed to `bazel run` resolve
from the directory where you invoked Bazel.

```sh
facts_run=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0

bazel run -c opt //src/llm/experiments/memorize_general_facts/discretized_model:generate_discretized_model -- \
  --checkpoint="$facts_run/layers_8/step_16128" \
  --tokenizer="$facts_run/inputs/tokenizer" \
  --corpus=testdata/general_facts_dataset.txt \
  --compact_transitions --state_index \
  --output=/tmp/facts-generated
```

This command keeps all exact states. Add `--reduce` to search for compatible
within-boundary merges; `--neighbors`, `--max_passes`, `--max_attempts`, and
`--exhaustive_pair_limit` bound that search. No intermediate checkpoint is saved:
an interrupted conversion must restart. The 6,514-state checked-in model and its
historical measurements above are preserved; a new search can choose a different
quotient depending on candidate order and search limits.

Defaults describe the selected checkpoint: width 16, eight blocks, one head,
MLP width 64, 1,024 samples, and five prompt tokens. `--layers`,
`--attention_heads`, `--feed_forward_width`, `--expected_samples`, and
`--prompt_tokens` allow matching another checkpoint/corpus. Native autonomous
completion and bitwise causal-prefix verification are enabled by default
(`--verify_greedy=true`), as is symbolic suffix/EOS verification. Do not disable
native greedy verification when establishing a new checkpoint's correctness.

The generator refuses existing output paths, including symlinks, formats sources,
and only then atomically publishes the completed directory. Readable statistics,
verification results, and input/output hashes are in `generation_report.txt`;
compact transition measurements are in `transition_patterns.txt`. Inspection
TSVs are optional.

Copy a freshly generated package into the workspace to build it with Bazel.
This is an explicit `cc_binary` invocation, not a genrule or an automatic rewrite
of checked-in generated code. The `:discretization` C++ library contains the
CPU reduction/emission algorithms; `:generator` adds the GPU capture and driver.
None of these dependencies enter the generated inference library.

Test the C++ libraries and real-GPU conversion with:

```sh
bazel test //src/llm/experiments/memorize_general_facts/discretized_model:generator_tests
```
