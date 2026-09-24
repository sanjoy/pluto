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
never combines states across boundaries. Corpus suffixes are verification
fixtures, not inputs to generated transition functions.

## Recording

`ModelRecorder::Record` takes `ModelRecorderOptions` and returns a
`CapturedModel`. It owns checkpoint/tokenizer/corpus loading, native execution,
exact BF16 interning, and suffix/EOS verification. It does not need an output
directory, formatter, or code-generation options. Width, context length, layer
count, head count, and MLP width must match the checkpoint. Recording supports
the native GPT-2 dimensions, including odd residual widths such as 13 with a
32-token context. Legacy defaults remain width 16 and context 1,024.

An optional `StateVectorHints` output retains representative BF16 vectors for
search ordering. The returned model is identical with or without hints; vectors
do not belong to `CapturedModel`. Hints are replaced only after successful
recording.

## Compaction

`CompactModel` takes a captured model and returns a new one. Optional vector
hints preserve proximity-based candidate ordering; without hints the search
uses deterministic ID ordering. Acceptance always uses symbolic transition
consistency and distinct terminal vocabulary labels, never a distance threshold.
`RelabelMlpOutputs` is a separate within-boundary renaming transformation which
can expose simpler pointwise code without combining states.

## Code generation

`discretize_attention`, `discretize_map`, and `discretize_position_embedding`
each take the corresponding captured layer and return `SerializedCppProgram`:
an in-memory C++ function definition, not a file or an execution engine. Both
sorted lookup tables and compact control-flow implementations live in these
lowerers. Unsupported inputs remain unsupported in either representation.

`code_generator` assembles these programs into independently compiled files,
the public model factory, vocabulary support, and separate verification tests.
Shared naming, escaping, source-wrapping, and file utilities live in `utils`.
Lowering itself performs no filesystem writes. Formatting and `PublishFiles`
publish the complete output atomically, refusing to replace existing paths.

`Generate` is the end-to-end driver. See the parent README for the invocation.
