# Discrete model generator

The generator records a checkpoint's executions, optionally compacts their
symbolic states, and lowers the resulting model to CPU-only C++. Neither the
runtime nor the generated model depends on this package.

## Core representation

`captured_model.h` describes observations, not implementation algorithms:

```text
CapturedModel
  position_embedding: CapturedPositionEmbedding
  transformers[]: CapturedTransformer
    attention: CapturedCausalAttention
    mlp: CapturedMap
  language_modeling_head: CapturedMap
```

Attention records the entire ordered causal prefix and its output symbol.
An MLP records one input/output symbol pair. The language modeling head uses
the same map shape with vocabulary IDs as its outputs. Token-plus-position
embedding has its own input pair. State IDs are boundary-specific; compaction
normally keeps states at one boundary. The optional final MLP-pair pass permits
one symbol at the adjacent post-attention/post-MLP boundaries of the same block;
`CapturedState::shared_boundary` records that exception. Corpus suffixes are verification
fixtures, not inputs to generated transition functions.

## Recording

`ModelRecorder::Record` takes `ModelRecorderOptions` and returns a
`CapturedModel`. It owns checkpoint/tokenizer/corpus loading, native execution,
exact BF16 interning, and suffix/EOS verification. It does not need an output
directory, formatter, or code-generation options. Width, context length, layer
count, head count, and MLP width must match the checkpoint. Recording supports
the native GPT-2 dimensions, including odd residual widths such as 13 with a
32-token context. Legacy defaults remain width 16 and context 1,024.

An optional `StateVectorHints` output retains every exact original BF16 vector.
The returned symbolic model is identical with or without these vectors. The
end-to-end generator always retains them in a `CapturedStateVectors` archive;
its keys remain original capture IDs, even after state compaction or renaming.
Its `original_boundaries` map preserves each vector's original observation
point even when a paired symbol spans both sides of an MLP.
The same vectors guide compaction candidate ordering for the initial model.
Recording replaces the supplied map only after successful verification.

## Compaction

`CompactModel` takes a captured model and returns a new one. Optional vector
hints preserve proximity-based candidate ordering; without hints the search
uses deterministic ID ordering. Acceptance always uses symbolic transition
consistency and distinct terminal vocabulary labels, never a distance threshold.
Each nearest-candidate pass also runs exact pointwise compaction: MLP inputs
with the same current output class are combined within their input boundary.
Equal language-head labels are handled first, then MLPs in reverse order. This
exhausts equal-output groups without a numerical shortlist, while respecting
the same attempt budget and cancellation checks. It never applies this rule
to attention-history symbols, which may differ in other contexts. Separate
pointwise progress records distinguish these trials from heuristic search.
After the bounded nearest-candidate passes, an eligible-pair threshold enables
an exhaustive continuation. This continuation has its own unbounded sweep
loop: it stops only on a no-change sweep, cancellation, or the optional trial
budget. Set `compaction_max_attempts=-1` and
`compaction_exhaustive_pair_limit=9223372036854775807` to exhaust all remaining
compatible pairs. Pairs with different fixed vocabulary labels are impossible
and skipped; other rejections are cached because later compaction cannot undo
their contradiction. Exhaustion proves pairwise irreducibility of the resulting
partition, not global minimality over different earlier choices.
`RelabelMlpOutputs` is a separate within-boundary renaming transformation which
can expose simpler pointwise code without combining states.

`CompactMlpPairs` is a separate final pass. Every MLP must be a complete
bijection between its two state sets. The pass retains each input ID, rewrites
its output ID everywhere to that input, unions original membership, and keeps
an explicit identity MLP with the same support. Only these adjacent boundaries
share symbols; attention boundaries and different blocks remain separate.
Unique state counts shrink while per-stage counts still include the paired ID
at both stages. Ordinary compaction, relabeling, and certificates reject this
final representation instead of silently treating it as single-boundary.

Each captured state starts with a singleton `members` list. Compaction unions
these lists, and relabeling preserves them. This indirection retains all original
vectors without repeatedly copying their channels or replacing them with a
representative. Render-time validation rejects missing vectors, wrong widths,
nonfinite channels, and duplicated or lost original membership.

## Code generation

`discretize_attention`, `discretize_map`, and `discretize_position_embedding`
each take the corresponding captured layer and return `SerializedCppProgram`:
an in-memory C++ function definition, not a file or an execution engine. Both
sorted lookup tables and compact control-flow implementations live in these
lowerers. Unsupported inputs remain unsupported in either representation.

`code_generator` assembles these programs into independently compiled files,
the public model factory, vocabulary support, and separate verification tests.
`state_vector_codegen` emits an inspection-only archive of exact BF16 words in
parallel-compilable source shards and a `DiscreteModel::print_state` callback.
Its lookup follows original membership, never the current IDs as archive keys.
Printing reconstructs decimal floats on the CPU; inference remains integer-only
and never consults the archive. Generic `RenderModel` callers may omit the
archive, in which case the generated callback is empty.
Shared naming, escaping, source-wrapping, and file utilities live in `utils`.
Lowering itself performs no filesystem writes. Formatting and `PublishFiles`
publish the complete output atomically, refusing to replace existing paths.

`Generate` is the end-to-end driver. See the parent README for the invocation.
