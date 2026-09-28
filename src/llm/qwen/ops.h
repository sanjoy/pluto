#pragma once

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "src/cuda/executor.h"

namespace pluto::llm::qwen {

// Small, inference-only cuTile operators. All activation pointers address
// FP32 storage on executor; round_bf16 preserves the checkpoint's BF16 compute
// boundaries without storing an extra copy of each activation. Buffers may not
// overlap except for ResidualAdd, which permits output == either input.
enum class MatrixStorage { kBFloat16, kFloat32, kFp8E4M3 };

// Row-major [output_dim, input_dim] matrix times a single input vector. FP8
// matrices use multiplicative FP32 scales shaped [ceil(output_dim/128),
// ceil(input_dim/128)]. FP8 input quantization, if requested, uses independent
// dynamic 128-element groups. Accumulation and the final reduction are FP32.
absl::Status MatVec(cuda::Executor& executor, const void* weights,
                    MatrixStorage storage, const float* scales,
                    const float* input, int input_dim, int output_dim,
                    float* output, bool round_bf16 = true);
absl::Status QuantizeFp8Input(cuda::Executor& executor, const float* input,
                              int elements, float* dequantized);

// Zero-centered RMSNorm: x * rsqrt(mean(x*x) + epsilon) * (1 + weight).
absl::Status RmsNorm(cuda::Executor& executor, const float* input,
                     const float* weight, int width, float epsilon,
                     float* output, bool round_bf16 = true);
absl::Status SwiGlu(cuda::Executor& executor, const float* gate,
                    const float* up, int elements, float* output,
                    bool round_bf16 = true);
absl::Status ResidualAdd(cuda::Executor& executor, const float* input,
                         const float* update, int elements, float* output,
                         bool round_bf16 = true);
absl::Status EmbeddingLookup(cuda::Executor& executor, const void* weights,
                             MatrixStorage storage, int token, int vocab_size,
                             int width, float* output);

}  // namespace pluto::llm::qwen
