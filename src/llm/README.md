# cuTile GPT-2-style language model

This directory contains a differentiable GPU layer API and a Shakespeare
training binary. The binary builds one fixed architecture:

| setting | value |
|---|---:|
| GPT-2 vocabulary | 50,257 |
| context length | 1,024 |
| transformer blocks | 8 |
| model width | 512 |
| attention heads | 8 |
| head dimension | 64 |
| feed-forward width | 2,048 |

Every block is pre-LayerNorm and applies causal multi-head self-attention,
followed by a GELU MLP, with a residual connection around each branch. Q, K,
and V have independent learned projections packed into one 512-to-1,536
matrix. Token embeddings and learned absolute position embeddings feed the
blocks; a final LayerNorm precedes an LM head tied to the token embedding
table. Dropout and attention dropout are both exactly zero.

## Numeric policy

The Shakespeare model uses BF16 for stored activations and matrix-multiply
operands. Matrix multiply accumulation, FlashAttention softmax maxima and
normalizers, LayerNorm mean/variance, cross-entropy, and gradients use FP32.
All parameters are FP32 master weights. AdamW keeps FP32 first- and
second-moment state and applies one update to tied parameters even when the
composition exposes the shared buffer twice. The logical 50,257-token
vocabulary is padded to 50,272 only for 16-wide GPU tiles; padded logits are
masked to negative infinity and cannot be sampled or targeted.

DataType::FP16 remains a compatibility path with FP32 activation storage and
FP16 matrix-multiply operands. DataType::FP8 is deliberately unimplemented
until an explicit scaling policy exists.

## Layer organization

Reusable layer families live under src/llm/layers: embedding and the tied LM
head, learned positions, rectangular fully connected projections, causal
FlashAttention, affine LayerNorm, GELU, cross entropy, and composition/residual
combinators. A sparse-autoencoder family provides an encoder/decoder and its
reconstruction-plus-decoder-norm sparsity objective. Each family has a header,
cuTile implementation, Bazel target, and focused GPU test.

Layer forward passes receive explicit buffers and save private intermediates
in a tree-shaped Tape. Backward passes return input gradients and fill FP32
parameter-gradient buffers. Parameter mutation is intentionally separate: the
training loop uses the Optimizer interface, while AdamWOptimizer owns AdamW
state and updates the model after backward.
Every allocation, copy, kernel, and asynchronous free uses one explicit
non-default CUDA stream.

The reusable trainer library owns the forward/loss/backward/update loop and
mean-loss evaluation. DataSetIterator supplies device-resident input/target
batches; Shakespeare uses the in-memory implementation with random windows
for optimization and separately resettable sequential windows for stable
training and held-out evaluation.

## Running

The binary uses the repository's GPT-2 tokenizer library. Point it at a
save_pretrained tokenizer directory with PLUTO_GPT2_TOKENIZER_DIR or
--tokenizer_dir.

```sh
PLUTO_GPT2_TOKENIZER_DIR=/path/to/gpt2 bazel test //src/llm:shakespeare_llm_test --test_output=streamed
```

Training mode only trains and evaluates; it never generates text or starts a
prompt loop. With no step limit it runs until interrupted:

```sh
PLUTO_GPT2_TOKENIZER_DIR=/path/to/gpt2 bazel run //src/llm:shakespeare_llm
```

The corpus is encoded with GPT-2 BPE and split chronologically into training
and held-out suffixes. The binary reports both losses before and after
training. `--batch_size` controls how many independent 1,024-token sequences
are processed per update. For example, `--batch_size=10` processes 10,240
token rows per optimizer step. Architecture dimensions are fixed;
`--eval_batches` controls the deterministic evaluation sample count.

Pass `--steps=N` to stop successfully after N additional updates, regardless
of the resulting loss. To stop early at a requested training loss, optionally
with a hard update cap:

```sh
PLUTO_GPT2_TOKENIZER_DIR=/path/to/gpt2 bazel run //src/llm:shakespeare_llm -- --train_until_loss=0 --steps=100000 --training_eval_interval=100
```

To write model weights every 100 completed optimizer steps:

```sh
PLUTO_GPT2_TOKENIZER_DIR=/path/to/gpt2 bazel run //src/llm:shakespeare_llm -- --steps=1200 --checkpoint_dir=/home/ubuntu/checkpoints/shakespeare --checkpoint_every=100
```

This creates `step_100`, `step_200`, and so on beneath the checkpoint
directory. To resume, pass the parent directory rather than a particular step:

```sh
PLUTO_GPT2_TOKENIZER_DIR=/path/to/gpt2 bazel run //src/llm:shakespeare_llm -- --resume_from=/home/ubuntu/checkpoints/shakespeare --steps=630 --checkpoint_every=100
```

The numerically largest direct child named `step_N` is loaded. If that is
`step_570`, the restored model starts at logical step 570 and its next update
is step 571. `--steps` counts additional updates in this invocation. Periodic
checkpoints default to the resume parent; `--checkpoint_dir` can direct new
checkpoints elsewhere.

Inference takes the exact checkpoint directory, not its parent:

```sh
PLUTO_GPT2_TOKENIZER_DIR=/path/to/gpt2 bazel run //src/llm:shakespeare_llm -- --inference_from=/home/ubuntu/checkpoints/shakespeare/step_570
```

This starts an inference-only prompt loop. Pass `--prompt='To be'` for one
completion followed by exit. Checkpoints contain model weights only, so AdamW
moment state restarts when training resumes; logical step numbering is
preserved from the `step_N` directory name. `--checkpoint_every=0` disables
checkpoint writes.
