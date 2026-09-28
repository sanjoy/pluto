# Qwen3.8-27B-FP8 text inference

This is a native, inference-only Pluto implementation of the text decoder in
[`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8).
It uses Pluto's `cuda::Executor`, stream-ordered buffers, page-locked host
transfers, and cuTile C++ kernels. Python, Transformers, and external inference
servers are **not** used by the inference binary.

## Download, build, and run

Use the Hugging Face CLI (`hf`) to download the pinned, public checkpoint:

```sh
hf download Qwen/Qwen3.8-27B-FP8 \
  --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --local-dir /home/ubuntu/datasets/models/Qwen3.8-27B-FP8

bazel build -c opt //src/llm/qwen:qwen_infer

bazel-bin/src/llm/qwen/qwen_infer \
  --checkpoint=/home/ubuntu/datasets/models/Qwen3.8-27B-FP8 \
  --prompt='What is 2 + 2? Reply with just the number.' \
  --max_new_tokens=12 --context_length=128
```

The download path is an example; the binary requires an explicit checkpoint
directory and never downloads or executes checkpoint-provided code. CUDA 13.3,
cuTile C++, and a GPU supported by the repository's CUDA toolchain are required.
Allow about 31 GB of disk and at least 32 GB of available GPU memory for short
contexts; the tested machine is an NVIDIA GH200 with approximately 96 GB VRAM.

Default inference applies the checkpoint's single-user chat template with
thinking disabled. `--thinking` enables its thinking template. `--raw_prompt`
passes text without a chat wrapper. Generation is greedy, stops on either EOS
token, and is bounded by `--max_new_tokens`. `--context_length` bounds the whole
prompt plus continuation and sizes the KV cache; the default is 512.

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
- `ops.{h,cc}` implements block-scaled FP8 matrix-vector products, dynamic
  per-128-element activation quantization, zero-centered RMSNorm, SwiGLU,
  residual addition and embedding lookup.
- `attention_ops.{h,cc}` implements cached full attention and recurrent
  GatedDeltaNet, with scalar CPU-formula comparisons in its tests.
- `model.{h,cc}` loads the text weights and runs the decoder. `Step(token)`
  consumes a token; `Logits()` predicts its successor; `Reset()` starts a new
  sequence without reloading weights. The executor must outlive the model.
- `src/dataset/qwen_tokenizer.{h,cc}` implements checkpoint-driven byte BPE,
  Qwen's pre-tokenization, added tokens and NFC normalization via utf8proc.

FP8 E4M3 weights remain compressed on the GPU. BF16 checkpoint block scales
are expanded to FP32, as are small scalar tensors. For FP8 projections, inputs
are dynamically quantized into E4M3 groups, then products and reductions are
computed in FP32 with block scaling. These are simple cuTile GEMV kernels,
**not an optimized FP8 tensor-core GEMM implementation**. Activations are kept
in FP32 buffers but rounded to BF16 at operation boundaries. Full-attention
softmax and recurrent state calculations retain FP32 precision. Floating-point
reduction and recurrent-prefill ordering can differ from other runtimes;
bitwise equality with Transformers/vLLM is not promised.

Prompt prefill and decode both consume one token at a time. This is a bounded
memory correctness baseline, not a throughput-oriented server. Currently only
batch-one text inference is supported: no vision/audio, multimodal positions,
MTP/speculative decoding, training, sampling, or batched/chunked prefill. The
download includes ancillary tensors, but only text-decoder tensors are loaded
onto the GPU. Unsupported configurations fail explicitly.

## Verification

Run synthetic operator, loader, cache and tokenizer tests without downloading
weights:

```sh
bazel test -c opt //src/llm/qwen:all //src/dataset:qwen_tokenizer_test
```

Enable additional checks against the downloaded checkpoint and the tokenizer's
Hugging Face reference token IDs:

```sh
bazel test -c opt //src/llm/qwen:all //src/dataset:qwen_tokenizer_test \
  --test_env=PLUTO_QWEN_CHECKPOINT_DIR=/home/ubuntu/datasets/models/Qwen3.8-27B-FP8 \
  --test_env=PLUTO_QWEN_TOKENIZER_DIR=/home/ubuntu/datasets/models/Qwen3.8-27B-FP8
```

Verified on GH200 with the pinned revision:

| Input | Mode | Greedy output |
| --- | --- | --- |
| `The capital of France is` | Raw, four new tokens | ` Paris.\nThe` |
| `What is 2 + 2? Reply with just the number.` | Chat, no thinking | `4`, then EOS |
| `In one sentence, why does ice float on water?` | Chat, no thinking | Explained lower density due to crystalline structure; 30 tokens, then EOS |

The loaded text weights occupy 29,476,263,936 device bytes. Initial observed
load-plus-inference times were 5.68 seconds and 6.58 seconds respectively;
these are smoke-test observations, not a controlled benchmark.

The GPU operator tests also pass Compute Sanitizer memcheck with
`--report-api-errors explicit`. The default extended API diagnostics flag the
existing executor's host-pool access setup even though CUDA returns success;
explicit CUDA-call failure checking and device-memory checking remain enabled.

Architecture/math reference:
[`transformers/models/qwen3_5/modeling_qwen3_5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
The checkpoint's configuration, tensor shapes, tokenizer, and chat template
are read from the pinned Hugging Face revision above. This new implementation
and its tests are AI-generated and have not been human reviewed.
