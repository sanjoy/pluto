#pragma once

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/llm/key_value_cache.h"

namespace pluto::llm::cached_attention_ops {

struct FullAttentionParameters {
  int query_heads = 24;
  int key_value_heads = 4;
  int head_dim = 256;
  int rotary_dim = 64;
  int capacity = 4096;
  float rms_norm_epsilon = 1e-6f;
  float rope_theta = 10000000.0f;
  bool round_to_bfloat16 = true;
};

// Checks the cached kernel's parameter limits and requires the cache's
// Executor, KV shape, and capacity to match. A full cache is still compatible;
// only an attempt to append to it fails. This performs no GPU work.
absl::Status ValidateCachedAttention(cuda::Executor& executor,
                                     const FullAttentionParameters& parameters,
                                     const KeyValueCache& cache);

// Queues one token's batch-one causal attention using an externally owned
// cache, whose position is the only sequence counter. All pointers refer to
// contiguous device FP32 arrays (BF16 values are represented exactly).
// q_gate is [query_heads, 2, head_dim], k/v are [key_value_heads, head_dim],
// and q_norm/k_norm are [head_dim] zero-centered weights (multiplier 1+w).
// Output is [query_heads, head_dim], after sigmoid gating and before the output
// projection. Optional probabilities receives [query_heads, position()+1].
// Validates pointers and remaining capacity before GPU work; advances the cache
// once after all launches succeed. Temporary query storage is stream ordered.
// Serialize calls on the creating Executor and reset the cache after any GPU
// failure or before a new sequence. Prefill repeats this call for each token.
absl::Status CachedAttentionStep(cuda::Executor& executor,
                                 const FullAttentionParameters& parameters,
                                 KeyValueCache& cache, const float* q_gate,
                                 const float* k, const float* v,
                                 const float* q_norm, const float* k_norm,
                                 float* output, float* probabilities = nullptr);

struct DeltaNetParameters {
  int key_heads = 16;
  int value_heads = 48;
  int key_head_dim = 128;
  int value_head_dim = 128;
  int conv_kernel_dim = 4;
  float rms_norm_epsilon = 1e-6f;
  bool round_to_bfloat16 = true;
};

// Single-token causal depthwise convolution + GatedDeltaNet recurrence. The
// recurrent matrix stays FP32 even when activation rounding is enabled.
class DeltaNetState final {
 public:
  DeltaNetState(const DeltaNetState&) = delete;
  DeltaNetState& operator=(const DeltaNetState&) = delete;
  static absl::StatusOr<std::unique_ptr<DeltaNetState>> Create(
      cuda::Executor& executor, DeltaNetParameters parameters);

  // qkv is packed [all Q, all K, all V]; z is [value_heads,value_head_dim];
  // a/b/A_log/dt_bias are [value_heads]. Conv weights are [qkv_channels,
  // conv_kernel_dim], oldest to newest. norm_weight is [value_head_dim], using
  // a plain multiplier (not 1 + weight). Output has z's shape, after RMSNorm
  // and SiLU(z) gating, before the output projection.
  absl::Status Step(const float* qkv, const float* z, const float* a,
                    const float* b, const float* conv_weight,
                    const float* a_log, const float* dt_bias,
                    const float* norm_weight, float* output);
  absl::Status Reset();

 private:
  DeltaNetState(cuda::Executor& executor, DeltaNetParameters parameters,
                cuda::Buffer convolution_state, cuda::Buffer recurrent_state,
                cuda::Buffer convolved_qkv, cuda::Buffer core_output);
  cuda::Executor& executor_;
  DeltaNetParameters parameters_;
  cuda::Buffer convolution_state_;
  cuda::Buffer recurrent_state_;
  cuda::Buffer convolved_qkv_;
  cuda::Buffer core_output_;
};

}  // namespace pluto::llm::cached_attention_ops
