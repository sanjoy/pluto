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
table. This is a study of a layered symbolic representation, not a replacement
of the whole model with a sentence-completion dictionary.

## Exact baseline milestone

Exact deduplication yields **221,558 internal states**, plus 4,475 vocabulary
symbols. The native checkpoint and the compiled C++ network both pass all
**10,002 predictions, 1,024 complete sentences, and 1,024 explicit EOS checks**.
For every generated prefix, all 17 native BF16 boundaries exactly match the
full-sentence capture. Repeating capture in a new process produced identical
JSONL bytes (SHA256 `d89e1ffcc85b1d40ea26cdb5ec002b43c51c7f525aa3a7ba7ce766c055da6712`).

The generated package is approximately 16 MiB of source, split into independent
attention/MLP translation units. Both its Bazel dependency graph and its dynamic
library list contain **no CUDA dependency**. It needs no checkpoint, tokenizer
installation, or GPU at inference time. All 72 repository Bazel test targets and
186 general-facts Python tests passed at this milestone.

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

## Reproduce capture and code generation

Use a fresh output path. Capture uses CUDA; the resulting C++ inference does not.
The JSONL capture and intermediate JSON model are local build/research artifacts,
not checked-in weight files. Python requires only its standard library for
baseline generation; `clang-format` must be installed.

```sh
bazel build -c opt //src/llm/experiments/memorize_general_facts/discretized_model:capture_checkpoint
facts_run=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0

bazel-bin/src/llm/experiments/memorize_general_facts/discretized_model/capture_checkpoint \
  --checkpoint="$facts_run/layers_8/step_16128" \
  --tokenizer="$facts_run/inputs/tokenizer" \
  --corpus=testdata/general_facts_dataset.txt \
  --output=/tmp/facts-capture.jsonl --verify_greedy=true

python3 -B scripts/memorize_general_facts/generate_discretized_model.py \
  --capture=/tmp/facts-capture.jsonl --save_model=/tmp/facts-discrete.json \
  --output=/tmp/facts-generated \
  --checkpoint="$facts_run/layers_8/step_16128" \
  --corpus=testdata/general_facts_dataset.txt \
  --tokenizer_json="$facts_run/inputs/tokenizer/tokenizer.json"
```

The generator refuses existing output directories, verifies all continuations,
formats the sources, and records file hashes in `manifest.json` and
`provenance.json`. Copy a freshly generated package into the workspace to build
it with Bazel. Do not add a generation rule to the build.

Reduction additionally supports optional NumPy/SciPy nearest-neighbor
acceleration. `--model=/tmp/facts-discrete.json --reduce` starts from a saved
model; `--save_model` is also an atomic, verified resumable checkpoint during
search. State counts and the precise stopping condition are recorded in that
artifact and in generated provenance. Test the Python utilities with:

```sh
python3 -B -m unittest discover -s scripts/memorize_general_facts -p '*_test.py'
```
