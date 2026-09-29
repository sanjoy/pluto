#pragma once

#include <memory>

#include "src/llm/layer.h"
#include "src/llm/layers/util/cached_attention_ops.h"

namespace pluto::llm {

using FullAttentionParameters = cached_attention_ops::FullAttentionParameters;

// Inference-only, batch-one grouped-query attention with a persistent KV cache.
// Each fwd consumes one token's Q/gate, K and V projections, normalizes Q/K,
// applies partial RoPE, attends to the complete cached prefix, and
// sigmoid-gates the result. Inputs/outputs are physical BF16; norm weights and
// cache are FP32. Unlike AttentionLayer's stateless packed-QKV training
// interface, this layer advances its cache even though fwd is const. Do not
// call it concurrently, keep its Executor alive, and Reset after any failed
// forward (including hooks) or before starting a new sequence. There is
// deliberately no training backward.
class QwenAttentionLayer final : public Layer {
 public:
  // Bound per-token projection scratch before allocating a persistent KV cache.
  static constexpr int kMaximumDimension = 1048576;

  static absl::StatusOr<std::unique_ptr<QwenAttentionLayer>> Create(
      cuda::Executor& executor, FullAttentionParameters parameters,
      Buffer q_norm, Buffer k_norm);

  absl::string_view name() const override { return "QwenAttentionLayer"; }
  DataType output_type() const override { return DataType::BF16; }
  absl::Span<Buffer> weights() override { return weights_; }
  absl::Span<const ActivationType> input_types() const override {
    return input_types_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return output_types_;
  }
  absl::Status Reset();
  int length() const { return cache_->length(); }

 private:
  QwenAttentionLayer(
      cuda::Executor& executor, FullAttentionParameters parameters,
      Buffer q_norm, Buffer k_norm,
      std::unique_ptr<cached_attention_ops::FullAttentionState> cache);
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks* hooks) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> gradients,
                                     BackwardState state,
                                     LayerHooks* hooks) override;

  cuda::Executor& executor_;
  FullAttentionParameters parameters_;
  Buffer weights_[2];  // Zero-centered query and key RMSNorm multipliers.
  std::unique_ptr<cached_attention_ops::FullAttentionState> cache_;
  const ActivationType input_types_[3];
  const ActivationType output_types_[1];
};

}  // namespace pluto::llm
