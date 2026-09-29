# Qwen3.8-27B-FP8 text inference and blockwise training

This is a native Pluto implementation of the text decoder in
[`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8).
It uses Pluto's `cuda::Executor`, stream-ordered buffers, page-locked host
transfers, and cuTile C++ kernels. Python, Transformers, and external inference
servers are **not** used by the binary. Cached FP8 inference is described
below; [BAdam training](TRAINING.md) uses BF16 resident weights and optimizer
state for only one parameter block at a time.

## Download, build, and run

Use the Hugging Face CLI (`hf`) to download the pinned, public checkpoint:

```sh
hf download Qwen/Qwen3.8-27B-FP8 \
  --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --local-dir /home/ubuntu/checkpoints/models/Qwen3.8-27B-FP8

bazel build -c opt //src/llm/qwen:qwen_llm

bazel-bin/src/llm/qwen/qwen_llm \
  --mode=infer_model \
  --checkpoint=/home/ubuntu/checkpoints/models/Qwen3.8-27B-FP8 \
  --prompt='What is 2 + 2? Reply with just the number.' \
  --max_new_tokens=12 --context_length=128
```

The download path is an example; the binary requires an explicit checkpoint
directory and never downloads or executes checkpoint-provided code. CUDA 13.3,
cuTile C++, and a GPU supported by the repository's CUDA toolchain are required.
Allow about 31 GB of disk and at least 32 GB of available GPU memory for short
contexts; the tested machine is an NVIDIA GH200 with approximately 96 GB VRAM.

In `--mode=infer_model`, inference applies the checkpoint's single-user chat
template with thinking disabled. `--thinking` enables its thinking template.
`--raw_prompt` passes text without a chat wrapper. Generation is greedy, stops
on either EOS token, and is bounded by `--max_new_tokens`. `--context_length`
bounds the whole prompt plus continuation and sizes the KV cache; the default
is 512.

## Execution modes

The single `qwen_llm` binary requires `--mode=infer_model`,
`--mode=train_model`, or `--mode=embedding_algebra`. `--checkpoint` selects the
original HF directory in all modes. Flags specific to another mode are
rejected before opening files or
initializing CUDA, even if explicitly set to their default, false, or empty
values. The old `qwen_infer` and `qwen_train` binaries are removed.

| Mode | Mode-specific flags |
| --- | --- |
| `infer_model` | `prompt`, `max_new_tokens`, `context_length`, `raw_prompt`, `thinking` |
| `train_model` | `text`, `sequence_length`, `batch_size`, `steps`, `switch_every`, `start_block`, `learning_rate`, `max_active_gib`, `resume_weights`, `save_weights` |
| `embedding_algebra` | `expression` (omit for the interactive prompt), `top_n` |

`--raw_prompt` and `--thinking` cannot both be enabled. Training remains
batch-one, short-sequence BAdam; see [training commands and limits](TRAINING.md).
Combining the entry points does not convert trained resident-weight checkpoints
to the FP8 format expected by inference.

## Embedding algebra

Explore addition/subtraction of the **input token embeddings**, without loading
or executing transformer blocks, the final norm, or the LM head:

```sh
bazel build -c opt //src/llm/qwen:qwen_llm
bazel-bin/src/llm/qwen/qwen_llm \
  --mode=embedding_algebra \
  --checkpoint=/home/ubuntu/checkpoints/models/Qwen3.8-27B-FP8
```

At the prompt, try:

```text
king - queen + boy
raw king - queen + boy
king - queen + boy raw
" king" - " queen" + " boy"
:help
:quit
```

To run one expression and exit, add `--expression='king - queen + boy'`.
Set `--top_n=10` to print ten matches instead of the default three; the count
must be positive and applies to both single-expression and interactive use.
The mode also accepts one expression per line on piped stdin, without prompts
or terminal escapes. Invalid lines report errors and the session continues;
any invalid line makes a piped session exit nonzero.

- Each symbol is tokenized independently as literal text; exactly **one token**
  is required. Multi-token symbols produce an error naming the symbol and token
  count. No spaces are automatically prepended. Quote leading spaces and
  punctuation to distinguish, for example, `king` from `" king"`.
- Single/double quotes preserve literal spaces/operators. Supported escapes
  are `\\`, `\"`, `\'`, `\/`, `\n`, `\r`, and `\t`. `raw` is reserved at either
  end; quote `"raw"` to use that vocabulary token. Only `+`/`-` and unary signs
  are supported, not parentheses, multiplication, or arbitrary code execution.
- Output contains up to `--top_n` nearest token rows (default three)
  by **cosine similarity**
  (range −1 to 1; larger is closer), with token IDs, escaped decoded spellings
  and ordinary L2 distance (smaller is closer). Cosine is not a probability or
  softmax. Expression inputs remain candidates, padded rows are excluded, zero
  rows are skipped, and exact score ties prefer the smaller token ID.
- `raw` prints all 5,120 FP32 result components, without normalization or
  truncation; `--top_n` does not change raw output. A zero result is valid in
  raw mode but has no cosine direction, so nearest-neighbor mode reports an
  error for `king - king`.
- [Linenoise](https://github.com/antirez/linenoise) supplies terminal line
  editing, multiline display and Up/Down history for the last 1,000 session
  commands. Ctrl-C cancels a line; Ctrl-D on an empty line or `:quit` exits.
  History is not saved to disk. The BSD-licensed dependency is pinned by revision and SHA-256 in
  `MODULE.bazel`; no system readline development package is required.

Only the BF16 embedding table (about 2.37 GiB for this checkpoint) is copied
to the GPU. Arithmetic and dot-product/distance reductions use FP32 cuTile
kernels; small result arrays cross through pinned host memory. Nearest-neighbor
ranking uses the original embedding rows, not the untied LM projection weights.

Parser/token-count tests and tiny GPU reference tests run without the large
checkpoint:

```sh
bazel test -c opt //src/llm/qwen:embedding_algebra_expression_test \
  //src/llm/qwen:embedding_algebra_table_test \
  //src/llm/qwen:qwen_cli_test //src/llm/qwen:qwen_llm_cli_test
```

## Implementation

The checkpoint's 64 decoder blocks alternate three GatedDeltaNet blocks with
one full-attention block. Hidden width is 5,120; the SwiGLU intermediate width
is 17,408. Full attention uses 24 query heads, four KV heads, 256-dimensional
heads, partial rotary position encoding, Q/K RMS normalization and sigmoid
output gates. GatedDeltaNet uses causal depthwise convolution, normalized Q/K,
a persistent FP32 recurrent state and SiLU-gated RMS normalization. The token
embedding and language-model output projection are **not tied**.

- `checkpoint.{h,cc}` validates and memory-maps Hugging Face safetensors,
  including sharded indices, shapes, byte ranges and configuration constraints.
  It reads data only. No pickle or remote code is used.
- `model.{h,cc}` is a checkpoint loader and model recipe. Every decoder block
  is a named `ComposedLayer` containing two ordinary `ResidualLayer` branches:
  `RmsNorm -> attention/projections` and `RmsNorm -> SwiGLU MLP`. A generic
  `ParallelLayer` fans out the Q/K/V or gate/up projections.
- `src/llm/layers/full_attention.{h,cc}` and `delta_net.{h,cc}` are separate
  stateful `Layer` implementations. The former supports cached GQA, rotary
  positions, Q/K normalization, and gating; the latter owns recurrent and
  convolution history. These contracts differ from the existing full-sequence,
  equal-head, trainable `AttentionLayer`.
- `src/llm/layers/inference.{h,cc}` contains imported-weight linear and embedding
  layers, zero-centered RMSNorm, and SwiGLU. Unlike the existing trainable
  projections/embeddings, these accept checkpoint storage/layouts directly,
  without allocating FP32 master weights, biases, or gradients.
- The cuTile kernels live under `src/llm/layers/util/` in `inference_ops.*` and
  `cached_attention_ops.*`; their scalar CPU comparisons remain separate tests.
- `Step(token, hooks)` consumes a token through the decoder graph;
  `Logits(hooks)` runs the final norm and untied projection. Both accept the
  standard optional `LayerHooks`. Projection, normalization, attention, gating,
  residual and block outputs are observable/intervenable. Cached full-attention
  probabilities have shape `[1, heads, 1, history_length]`.
- `Reset()` starts a new sequence without reloading weights. A failed Step,
  including a hook failure, requires Reset before continuing: some earlier
  caches may already have advanced. Calls on one model cannot run concurrently.
  The executor must outlive the model and all retained output buffers.
- `src/dataset/qwen_tokenizer.{h,cc}` implements checkpoint-driven byte BPE,
  Qwen's pre-tokenization, added tokens and NFC normalization via utf8proc.

FP8 E4M3 weights remain compressed on the GPU. BF16 checkpoint block scales
are expanded to FP32, as are small scalar tensors. For FP8 projections, inputs
are dynamically quantized into E4M3 groups, then products and reductions are
computed in FP32 with block scaling. These are simple cuTile GEMV kernels,
**not an optimized FP8 tensor-core GEMM implementation**. Layer activations use
physical BF16 buffers; internal operator scratch, reductions, full-attention
softmax and recurrent state calculations retain FP32 precision. Conversion
kernels currently bridge BF16 layer boundaries to the FP32 kernel interfaces.
Imported inference layers expose weights but no gradients and reject backward;
the existing trainable layers and their backward implementations are unchanged.
The Qwen recipe requires BF16 token embeddings, as in the official checkpoint;
other embedding dtypes fail explicitly instead of silently changing precision.
Floating-point reduction and recurrent-prefill ordering can differ from other runtimes;
bitwise equality with Transformers/vLLM is not promised.

Prompt prefill and decode both consume one token at a time. This is a bounded
memory correctness baseline, not a throughput-oriented server. Currently only
batch-one text inference is supported: no vision/audio, multimodal positions,
MTP/speculative decoding, sampling, or batched/chunked prefill. Training uses
the separate full-sequence implementation described in [TRAINING.md](TRAINING.md).
The download includes ancillary tensors, but only text-decoder tensors are
loaded onto the GPU. Unsupported configurations fail explicitly.

## Verification

Run synthetic operator, loader, cache and tokenizer tests without downloading
weights:

```sh
bazel test -c opt //src/llm/qwen:all //src/dataset:qwen_tokenizer_test \
  //src/llm:inference_test //src/llm:inference_ops_test \
  //src/llm:cached_attention_test //src/llm:cached_attention_ops_test \
  //src/llm:combinators_test
```

Enable additional checks against the downloaded checkpoint and the tokenizer's
Hugging Face reference token IDs:

```sh
bazel test -c opt //src/llm/qwen:all //src/dataset:qwen_tokenizer_test \
  --test_env=PLUTO_QWEN_CHECKPOINT_DIR=/home/ubuntu/checkpoints/models/Qwen3.8-27B-FP8 \
  --test_env=PLUTO_QWEN_TOKENIZER_DIR=/home/ubuntu/checkpoints/models/Qwen3.8-27B-FP8
```

Verified on GH200 with the pinned revision:

| Input | Mode | Greedy output |
| --- | --- | --- |
| `The capital of France is` | Raw, four new tokens | ` Paris.\nThe` |
| `What is 2 + 2? Reply with just the number.` | Chat, no thinking | `4`, then EOS |
| `In one sentence, why does ice float on water?` | Chat, no thinking | Explained lower density due to crystalline structure; 30 tokens, then EOS |

The loaded text weights occupy 29,476,263,936 device bytes; refcounted layer and
combinator handles share those allocations rather than copying the weights.

The GPU operator tests also pass Compute Sanitizer memcheck with
`--report-api-errors explicit`. The default extended API diagnostics flag the
existing executor's host-pool access setup even though CUDA returns success;
explicit CUDA-call failure checking and device-memory checking remain enabled.

Architecture/math reference:
[`transformers/models/qwen3_5/modeling_qwen3_5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
The checkpoint's configuration, tensor shapes, tokenizer, and chat template
are read from the pinned Hugging Face revision above. This new implementation
and its tests are AI-generated and have not been human reviewed.
