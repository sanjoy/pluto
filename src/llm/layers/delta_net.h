#pragma once

#include <memory>

#include "src/llm/layer.h"
#include "src/llm/layers/util/cached_attention_ops.h"

namespace pluto::llm {

using DeltaNetParameters = cached_attention_ops::DeltaNetParameters;

// Inference-only, batch-one causal convolution and gated delta-rule layer.
// Each fwd consumes one token's packed QKV, gate Z, time-step A and update B
// projections and advances persistent FP32 convolution/recurrent state. Inputs
// and output use physical BF16. The output includes per-head RMSNorm and SiLU
// gating, but not the output projection, which is a separate linear layer.
// The Executor must outlive this layer. Const fwd is stateful, so concurrent
// calls are not supported. Reset before a new sequence or after any failed
// forward, including failures in hooks. Backward is intentionally unsupported.
class DeltaNetLayer final : public Layer {
 public:
  // Bound per-token projection scratch before allocating recurrent state.
  static constexpr int kMaximumDimension = 1048576;

  // All four imported weights use FP32 storage: convolution [QKV, kernel],
  // A_log [value_heads], dt_bias [value_heads], and RMS multiplier [value_dim].
  static absl::StatusOr<std::unique_ptr<DeltaNetLayer>> Create(
      cuda::Executor& executor, DeltaNetParameters parameters,
      Buffer convolution, Buffer a_log, Buffer dt_bias, Buffer norm);

  absl::string_view name() const override { return "DeltaNetLayer"; }
  DataType output_type() const override { return DataType::BF16; }
  absl::Span<Buffer> weights() override { return weights_; }
  absl::Span<const ActivationType> input_types() const override {
    return input_types_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return output_types_;
  }
  absl::Status Reset();

 private:
  DeltaNetLayer(cuda::Executor& executor, DeltaNetParameters parameters,
                Buffer convolution, Buffer a_log, Buffer dt_bias, Buffer norm,
                std::unique_ptr<cached_attention_ops::DeltaNetState> cache);
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks* hooks) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> gradients,
                                     BackwardState state,
                                     LayerHooks* hooks) override;

  cuda::Executor& executor_;
  DeltaNetParameters parameters_;
  Buffer weights_[4];
  std::unique_ptr<cached_attention_ops::DeltaNetState> cache_;
  const ActivationType input_types_[4];
  const ActivationType output_types_[1];
};

}  // namespace pluto::llm
