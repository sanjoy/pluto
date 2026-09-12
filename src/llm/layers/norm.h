#pragma once

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// Learned affine LayerNorm. Mean/variance and backward reductions use FP32;
// gamma/beta are FP32 master parameters and outputs use the activation dtype.
// Each forward retains its input, FP32 row means, and FP32 inverse standard
// deviations (in that order), so concurrent saved states never share scratch.
// Parameter gradients are replaced per backward, with a fixed reduction order.
class LayerNormLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<LayerNormLayer>> Create(
      cuda::Executor& executor, int embedding_dim, float epsilon,
      DataType data_type);

  absl::Span<Buffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<Buffer> gradients() override { return absl::MakeSpan(gradients_); }
  DataType output_type() const override { return output_type_; }

 private:
  absl::StatusOr<FwdResult> fwd_impl(
      cuda::Executor& executor, absl::Span<const Buffer> inputs) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state) override;

  LayerNormLayer(cuda::Executor& executor, int embedding_dim, float epsilon,
                 DataType data_type, Buffer gamma, Buffer beta,
                 Buffer gamma_gradient, Buffer beta_gradient)
      : embedding_dim_(embedding_dim),
        epsilon_(epsilon),
        output_type_(data_type),
        executor_(executor),
        weights_{std::move(gamma), std::move(beta)},
        gradients_{std::move(gamma_gradient), std::move(beta_gradient)} {}

  int embedding_dim_;
  float epsilon_;
  DataType output_type_;
  cuda::Executor& executor_;
  BufferVec weights_;
  BufferVec gradients_;
};

// Direct row-by-row LayerNorm equations used to validate both CUDA passes.
class LayerNormLayerReference final : public LayerReference {
 public:
  static absl::StatusOr<std::unique_ptr<LayerNormLayerReference>> Create(
      int embedding_dim, float epsilon, DataType data_type);

  absl::Span<HostBuffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<HostBuffer> gradients() override {
    return absl::MakeSpan(gradients_);
  }
  DataType output_type() const override { return output_type_; }

 private:
  absl::StatusOr<ReferenceFwdResult> fwd_impl(
      absl::Span<const HostBuffer> inputs) const override;
  absl::StatusOr<HostBufferVec> bwd_impl(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceBackwardState state) override;

  LayerNormLayerReference(int embedding_dim, float epsilon, DataType data_type,
                          HostBuffer gamma, HostBuffer beta,
                          HostBuffer gamma_gradient, HostBuffer beta_gradient)
      : embedding_dim_(embedding_dim),
        epsilon_(epsilon),
        output_type_(data_type),
        weights_{std::move(gamma), std::move(beta)},
        gradients_{std::move(gamma_gradient), std::move(beta_gradient)} {}

  int embedding_dim_;
  float epsilon_;
  DataType output_type_;
  HostBufferVec weights_;
  HostBufferVec gradients_;
};

}  // namespace pluto::llm
