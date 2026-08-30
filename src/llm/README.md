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
combinators. Each family has a header, cuTile implementation, Bazel target, and
focused GPU test.

Layer forward passes receive explicit buffers and save private intermediates
in a tree-shaped Tape. Backward passes return input gradients and fill FP32
parameter-gradient buffers. Parameter mutation is intentionally separate:
AdamWOptimizer owns optimizer state and updates the model after backward.
Every allocation, copy, kernel, and asynchronous free uses one explicit
non-default CUDA stream.

## Running

The binary uses the repository's GPT-2 tokenizer library. Point it at a
save_pretrained tokenizer directory with PLUTO_GPT2_TOKENIZER_DIR or
--tokenizer_dir.

```sh
PLUTO_GPT2_TOKENIZER_DIR=/path/to/gpt2 bazel test //src/llm:shakespeare_llm_test --test_output=streamed
```

Train and then prompt the model (end input with Ctrl-C or Ctrl-D):

```sh
PLUTO_GPT2_TOKENIZER_DIR=/path/to/gpt2 bazel run //src/llm:shakespeare_llm -- --steps=1200
```

The corpus is encoded with GPT-2 BPE and split chronologically into training
and held-out suffixes. The binary reports both losses before and after
training. --batch_size controls how many token rows are processed per update
and must be a positive multiple of 1,024; architecture dimensions are fixed.
--eval_batches controls deterministic evaluation sample count.

To train until a requested training loss while retaining a hard iteration cap:

```sh
PLUTO_GPT2_TOKENIZER_DIR=/path/to/gpt2 bazel run //src/llm:shakespeare_llm -- --train_until_loss=0 --steps=100000 --training_eval_interval=100
```
