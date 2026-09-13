# KVQ explorer

Inspect the isolated K, V, and Q projections of every block in a GPT-2 recipe
checkpoint. The input is tokenized with the same GPT-2 tokenizer used for
training. For each input token position and each of the eight blocks, the tool
prints three vocabulary IDs, their escaped token text, and their softmax
probabilities for each projection.

## Run

```bash
bazel build -c opt //src/llm/experiments/kvq_explorer

bazel-bin/src/llm/experiments/kvq_explorer/kvq_explorer \
  --checkpoint=/home/ubuntu/checkpoints/shakespeare_0/step_15000 \
  --tokenizer=/home/ubuntu/datasets/tokenizer/gpt2 \
  --prompt='Exeunt all'
```

`--checkpoint` must name a specific checkpoint directory. `--prompt` is a
required, nonempty string; shell quoting preserves spaces and newlines.
`--tokenizer` defaults to the relative path `datasets/tokenizer/gpt2`, not a
machine-specific absolute path. It must match the checkpoint vocabulary.
Block indices and input token positions are zero-based. Output order is
input position, block, then K/V/Q. Repeated tokens remain separate positions.
Token strings are C-escaped so whitespace, control characters, and partial
UTF-8 token bytes remain visible rather than breaking the output layout.

## Exact computation

For each input ID `t`, block `b`, and projection `s` in `{Q, K, V}`:

```text
x           = E[t]                         # [512]
projected_s = x W_s[b] + bias_s[b]          # [512], concatenated heads
logits_s    = projected_s E_logical^T       # [50,257]
p_s         = softmax(logits_s)             # temperature 1
```

The top three **distinct** vocabulary IDs are ordered by decreasing
probability; ties prefer the lower ID. Probabilities are normalized over
**all 50,257 tokens**, not renormalized over the three printed entries.

This is a direct embedding-to-projection readout. There are **no learned
positions, LayerNorms, attention, attention output projection, residuals, or
preceding transformer blocks**. The stored projection biases are included.
The tool uses FP32 checkpoint weights, projections, dot products, and softmax
reductions, without the training model's BF16 activation rounding. Tokens do
not interact and there is no context-length truncation.

Q/K/V live in attention-head coordinates, not the residual-stream vocabulary
basis. Multiplying them by E is an exploratory choice: the results are not
the model's next-token probabilities, attention probabilities, or a faithful
trace of its contextual Q/K/V activations. The display combines all heads;
it does not report per-head readouts.

## Checkpoint layout and execution

Only 17 FP32 tensor files are needed from the fixed eight-block GPT-2 recipe:

| Tensor | File | Physical shape |
| --- | --- | --- |
| Token embedding E | `weight_0.bin` | `[50,272, 512]` |
| Packed Q/K/V matrix, block b | `weight_{4+12*b}.bin` | `[512, 1,536]` |
| Packed Q/K/V bias, block b | `weight_{5+12*b}.bin` | `[1,536]` |

Matrices are row-major `[input, output]`; the output columns are contiguous
Q, then K, then V slices. All files must have exactly the expected size and
contain finite FP32 values. Other checkpoint files are ignored. Embedding
padding rows 50,257 through 50,271 are excluded from both ranking and
softmax normalization. Nonfinite computed logits are reported as errors.

Projection, vocabulary dot products, and top-three/softmax reductions run on
the GPU with fixed-order, single-writer reductions. Temporary logits cover
at most 16 input tokens at a time (about 9.2 MiB). Only compact top-three
records are downloaded; every host/device transfer uses page-locked storage.
The checkpoint is read-only and the tool does not modify any training model.

## Tests

```bash
bazel test -c opt //src/llm/experiments/kvq_explorer/... \
  --test_output=errors --local_test_jobs=1
```

Self-contained GPU tests create small synthetic checkpoints and compare all
eight blocks and all three projections to independent scalar CPU calculations.
They cover matrix orientation, packed slice order, learned biases, a partial
warp dimension, repeated tokens across chunk boundaries, vocabulary padding,
repeatability, full-vocabulary softmax, cross-thread ties, extreme logits,
nonfinite values, malformed checkpoints, invalid IDs, and executor ownership.
