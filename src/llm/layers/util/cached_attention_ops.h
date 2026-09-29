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

// Batch-one causal attention. All pointers are contiguous device FP32 arrays;
// BF16 activations can be represented exactly in those arrays. Each Step queues
// one token on the creating executor and advances its position by one. Prefill
// is the same operation repeated over prompt tokens. The executor must outlive
// the state; methods must not be called concurrently.
class FullAttentionState final {
 public:
  FullAttentionState(const FullAttentionState&) = delete;
  FullAttentionState& operator=(const FullAttentionState&) = delete;
  static absl::StatusOr<std::unique_ptr<FullAttentionState>> Create(
      cuda::Executor& executor, FullAttentionParameters parameters);

  // q_gate is [query_heads, 2, head_dim], k/v are [key_value_heads,
  // head_dim], and q_norm/k_norm are [head_dim]. Norm weights are zero-centered
  // (the multiplier is 1 + weight). Output is [query_heads, head_dim], after
  // sigmoid gating but before the output projection.
  // Optional probabilities has room for [query_heads, length() + 1] FP32
  // entries and receives this new query's softmax weights over all cached keys.
  absl::Status Step(const float* q_gate, const float* k, const float* v,
                    const float* q_norm, const float* k_norm, float* output,
                    float* probabilities = nullptr);
  absl::Status Reset();
  int length() const { return cache_->position(); }

 private:
  FullAttentionState(cuda::Executor& executor,
                     FullAttentionParameters parameters,
                     std::unique_ptr<KeyValueCache> cache,
                     cuda::Buffer queries);
  cuda::Executor& executor_;
  FullAttentionParameters parameters_;
  std::unique_ptr<KeyValueCache> cache_;
  cuda::Buffer queries_;
};

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
