#pragma once

#include <memory>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// Elementwise SiLU(gate) * up on two BF16 inputs shaped [batch, 1, width].
// SiLU is rounded to BF16 before multiplication, then the result is rounded to
// BF16 again, preserving the imported checkpoint's compute rule. This frozen
// inference operator requires batch == 1 and does not support backward.
class SwiGluLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<SwiGluLayer>> Create(int width);

  absl::string_view name() const override { return "SwiGluLayer"; }
  absl::Span<const ActivationType> input_types() const override {
    return types_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return absl::MakeConstSpan(types_.data(), 1);
  }
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::BF16; }

 private:
  explicit SwiGluLayer(int width)
      : width_(width),
        types_{ActivationType(DataType::BF16,
                              {ActivationType::kBatchDimension, 1, width}),
               ActivationType(DataType::BF16,
                              {ActivationType::kBatchDimension, 1, width})} {}
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override;

  int width_;
  absl::InlinedVector<ActivationType, 2> types_;
};

}  // namespace pluto::llm
