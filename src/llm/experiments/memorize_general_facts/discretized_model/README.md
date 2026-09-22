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
Internal symbols are numeric IDs for exact native BF16 residual vectors.
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
A separate generator checks every duplicate key, emits ordinary C++ arrays in
multiple source files, and formats them. There are no Bazel generation rules.
Generated source is checked into `generated/`. Inference uses integer table
lookups only; it does not load weights or use CUDA. Expected sentence suffixes
are verification fixtures, never prediction tables consulted by the runtime.

Tests cover table consistency, unknown keys, causal history ordering, EOS,
serialization, and merge rollback. End-to-end validation checks both the native
checkpoint and the compiled model with autonomous first-five-token generation.

## State reduction

After the exact baseline passes, nearby states at the same boundary are proposed
for merging. When rewritten attention or MLP keys collide, their output states
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

This README will record runnable commands, provenance hashes, validation
results, state counts, and the precise stopping condition as milestones land.
