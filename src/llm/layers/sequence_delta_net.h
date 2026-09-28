#pragma once

#include <memory>

#include "src/llm/block_parameter.h"
#include "src/llm/layer.h"
#include "src/llm/layers/util/cached_attention_ops.h"

namespace pluto::llm {

// Stateless, batch-one training counterpart of DeltaNetLayer. A call consumes
// the entire sequence, starting with zero convolution and recurrent history.
// Backward differentiates through that entire history, not just the last token.
// Inputs are BF16 [1,T,QKV], [1,T,V], [1,T,H], [1,T,H]; output is BF16 [1,T,V].
// QKV packs all Q, then all K, then all V. Input gradients are always FP32.
// BF16 rounding has the usual straight-through derivative. The four FP32
// parameters are owned by the block optimizer: frozen parameters still affect
// input gradients, but only active parameters accumulate their own gradients.
// The executor and parameters must outlive any retained forward state.
class SequenceDeltaNetLayer final : public Layer {
 public:
  // Convolution is [QKV,kernel], A_log and dt_bias are [value_heads], and
  // norm is [value_head_dim], a plain multiplier rather than 1 + weight.
  // This initial full-BPTT implementation bounds T<=128, key_dim<=128, and
  // value_dim<=256, including the dimensions of the imported Qwen checkpoint.
  static absl::StatusOr<std::unique_ptr<SequenceDeltaNetLayer>> Create(
      cuda::Executor& executor,
      cached_attention_ops::DeltaNetParameters parameters,
      std::shared_ptr<BlockParameter> convolution,
      std::shared_ptr<BlockParameter> a_log,
      std::shared_ptr<BlockParameter> dt_bias,
      std::shared_ptr<BlockParameter> norm, int sequence_length);

  absl::string_view name() const override { return "SequenceDeltaNetLayer"; }
  DataType output_type() const override { return DataType::BF16; }
  absl::Span<Buffer> weights() override { return weights_; }
  absl::Span<const ActivationType> input_types() const override {
    return input_types_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return output_types_;
  }

 private:
  SequenceDeltaNetLayer(cuda::Executor& executor,
                        cached_attention_ops::DeltaNetParameters parameters,
                        std::shared_ptr<BlockParameter> convolution,
                        std::shared_ptr<BlockParameter> a_log,
                        std::shared_ptr<BlockParameter> dt_bias,
                        std::shared_ptr<BlockParameter> norm,
                        int sequence_length);
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks* hooks) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> gradients,
                                     BackwardState state,
                                     LayerHooks* hooks) override;

  cuda::Executor& executor_;
  cached_attention_ops::DeltaNetParameters parameters_;
  int sequence_length_;
  std::shared_ptr<BlockParameter> parameters_owned_[4];
  Buffer weights_[4];
  const ActivationType input_types_[4];
  const ActivationType output_types_[1];
};

}  // namespace pluto::llm
