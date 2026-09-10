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
Architecture construction and its dimensions live in
`src/llm/recipes/gpt2.{h,cc}`; the Shakespeare binary supplies only the
dataset and training/inference policy.

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

## Reproducibility

Training and inference are deterministic on the same GPU architecture, build,
CUDA/compiler runtime, and C++ standard library. Replaying the same seed,
input data, configuration, and starting state produces bit-identical weights
at matching optimizer steps. This applies to BF16 and the legacy FP16 path,
including SAE training. See [DETERMINISM.md](DETERMINISM.md) for the audit,
regression tests, and implementation details.

For reproducible final weights, use `--steps=N` without `--training_seconds`:
a wall-clock budget can stop at different steps on different runs. Checkpoints
currently contain weights, not optimizer or iterator state; two identical
warm-start runs replay, but resuming is not equivalent to uninterrupted
training. Keep evaluation cadence, corpus split, and all other options fixed.

Inference supports all finite nonnegative temperatures. `--temperature=0`
selects the largest logit, breaking ties by lowest token ID, without consuming
random numbers. Positive temperatures use the seeded sampling generator. The
interactive prompt loop retains that generator across prompts: reproduce the
entire prompt sequence, or use a fresh `--prompt=...` invocation for independent
replays. Timestamps and elapsed-time logs naturally differ between runs.

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
--tokenizer_dir. Every invocation requires exactly one explicit mode:
`--mode=train_model`, `--mode=infer_model`, `--mode=train_sae`, or
`--mode=infer_SAE` (also spelled `infer_sae`). Flags that
do not apply to the selected mode are rejected instead of being silently
ignored.

```sh
PLUTO_GPT2_TOKENIZER_DIR=/path/to/gpt2 bazel test //src/llm/recipes:gpt2_shakespeare_llm_test --test_output=streamed
```

Training mode only trains and evaluates; it never generates text or starts a
prompt loop. With no step limit it runs until interrupted:

```sh
PLUTO_GPT2_TOKENIZER_DIR=/path/to/gpt2 bazel run //src/llm/recipes:gpt2_shakespeare_llm -- --mode=train_model
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
PLUTO_GPT2_TOKENIZER_DIR=/path/to/gpt2 bazel run //src/llm/recipes:gpt2_shakespeare_llm -- --mode=train_model --train_until_loss=0 --steps=100000 --training_eval_interval=100
```

To write model weights every 100 completed optimizer steps:

```sh
PLUTO_GPT2_TOKENIZER_DIR=/path/to/gpt2 bazel run //src/llm/recipes:gpt2_shakespeare_llm -- --mode=train_model --steps=1200 --checkpoint_dir=/home/ubuntu/checkpoints/shakespeare --checkpoint_every=100
```

This creates `step_100`, `step_200`, and so on beneath the checkpoint
directory. To resume, pass the parent directory rather than a particular step:

```sh
PLUTO_GPT2_TOKENIZER_DIR=/path/to/gpt2 bazel run //src/llm/recipes:gpt2_shakespeare_llm -- --mode=train_model --resume_from=/home/ubuntu/checkpoints/shakespeare --steps=630 --checkpoint_every=100
```

The numerically largest direct child named `step_N` is loaded. If that is
`step_570`, the restored model starts at logical step 570 and its next update
is step 571. `--steps` counts additional updates in this invocation. Periodic
checkpoints default to the resume parent; `--checkpoint_dir` can direct new
checkpoints elsewhere.

Inference takes the exact checkpoint directory, not its parent:

```sh
PLUTO_GPT2_TOKENIZER_DIR=/path/to/gpt2 bazel run //src/llm/recipes:gpt2_shakespeare_llm -- --mode=infer_model --inference_from=/home/ubuntu/checkpoints/shakespeare/step_570
```

This starts an inference-only prompt loop. Pass `--prompt='To be'` for one
completion followed by exit. Checkpoints contain model weights only, so AdamW
moment state restarts when training resumes; logical step numbering is
preserved from the `step_N` directory name. `--checkpoint_every=0` disables
checkpoint writes.

To train a sparse autoencoder over the residual-stream activations after the
fourth transformer block, pass an exact GPT-2 checkpoint. The SAE uses the full
Shakespeare corpus, 4,096 features (eight times GPT-2's hidden width), and a
sparsity penalty of 0.5.

The penalty is `0.5 * sum_i Z[i] * ||D[:, i]||_2`, using the **unsquared**
decoder-column norm. Scaling an encoder feature and its bias by `c > 0` and
its decoder column by `1/c` leaves both reconstruction and sparsity penalty
unchanged. At zero decoder columns the loss backward uses the zero subgradient
of the norm. Loss values from older squared-norm runs are not comparable.

```sh
PLUTO_GPT2_TOKENIZER_DIR=/path/to/gpt2 bazel run //src/llm/recipes:gpt2_shakespeare_llm -- --mode=train_sae --sparse_autoencoder_from=/home/ubuntu/checkpoints/shakespeare/step_570
```

The usual training, evaluation, optimizer, logging, step-limit, and checkpoint
flags apply. Checkpoints written in this mode contain only the SAE weights.
To resume SAE training, keep `--sparse_autoencoder_from` pointed at the exact
GPT-2 checkpoint and pass the parent of the SAE `step_N` directories through
`--resume_from`. As with model training, the latest valid SAE checkpoint is
loaded and malformed newer checkpoints are skipped with a warning.

To inspect the trained SAE's feature activations Z for a prompt:

```sh
bazel build -c opt //src/llm/recipes:gpt2_shakespeare_llm
bazel-bin/src/llm/recipes/gpt2_shakespeare_llm \
  --mode=infer_SAE \
  --tokenizer_dir=/path/to/gpt2 \
  --sparse_autoencoder_from=/path/to/gpt2/checkpoints/step_13030 \
  --inference_from=/path/to/sae/checkpoints/step_20000 \
  --prompt='To be, or not to be'
```

Both paths must name exact checkpoint directories. `--sparse_autoencoder_from`
selects the frozen GPT-2 weights; use the same checkpoint used for SAE training.
`--inference_from` selects the SAE weights. The activation tap is the same
post-fourth-block residual stream used by `train_sae`, with d=512 and m=4096.
Omit `--prompt` to enter a prompt loop. Empty lines are skipped. Prompts longer
than 1,024 tokens use their last 1,024 tokens with a notice, as in model inference.
There is no generation, optimization, corpus loading, or checkpoint writing;
training flags and generation flags are rejected in this mode.

Statistics include active values (Z > 0), the fraction of zeros, mean active
features per token (L0), and activation mean, population standard deviation,
and maximum. Means and standard deviations include zeros. Only real prompt
tokens are counted, never trailing context padding. Statistics are per prompt,
not accumulated across prompts. They describe feature usage, not reconstruction
quality or learned feature meanings.

At the layer API, pass `SparseAutoEncoderLayer::Mode::kCollectStatistics` to
`Create` to enable an extra cuTile row-reduction kernel during `fwd`. The tape
owns four FP32 summary values per row; fwd remains asynchronous and backward
is unchanged. `ReadZStatistics(executor, tape, valid_rows)` explicitly copies
only the selected row summaries through page-locked memory and synchronizes.
The default layer mode allocates no statistics buffer and launches no extra
kernel. Statistics mode supports up to 2^24 features so FP32 row counts remain
exact; total counts are accumulated in 64-bit integers on the host.
