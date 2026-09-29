#pragma once

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// Frozen, zero-centered RMS normalization of one BF16 token:
// x * rsqrt(mean(x*x) + epsilon) * (1 + weight). Statistics and the imported
// weight are FP32; output is BF16. Unlike LayerNorm, this does not subtract the
// mean or add a bias. There are no master weights or gradient accumulators.
class RmsNormLayer final : public Layer {
 public:
  // Shares the FP32 [width] checkpoint weight on this executor. Activations
  // have shape [batch, 1, width]; this implementation requires batch == 1.
  static absl::StatusOr<std::unique_ptr<RmsNormLayer>> Create(
      cuda::Executor& executor, int width, Buffer weight, float epsilon);

  absl::string_view name() const override { return "RmsNormLayer"; }
  absl::Span<const ActivationType> input_types() const override {
    return absl::MakeConstSpan(&type_, 1);
  }
  absl::Span<const ActivationType> output_types() const override {
    return input_types();
  }
  absl::Span<Buffer> weights() override { return absl::MakeSpan(&weight_, 1); }
  DataType output_type() const override { return DataType::BF16; }

 private:
  RmsNormLayer(cuda::Executor& executor, int width, Buffer weight,
               float epsilon)
      : executor_(executor),
        width_(width),
        weight_(std::move(weight)),
        epsilon_(epsilon),
        type_(DataType::BF16, {ActivationType::kBatchDimension, 1, width}) {}
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override;

  cuda::Executor& executor_;
  int width_;
  Buffer weight_;
  float epsilon_;
  ActivationType type_;
};

}  // namespace pluto::llm
