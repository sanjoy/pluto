#pragma once

#include <cuda_runtime_api.h>

#include <memory>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// Learned affine LayerNorm. Mean/variance and backward reductions use FP32;
// gamma/beta are FP32 master parameters and outputs use the activation dtype.
class LayerNormLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<LayerNormLayer>> Create(
      int embedding_dim, float epsilon, DataType data_type,
      cudaStream_t stream);

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                              Tape* tape) override;
  absl::StatusOr<BufferVec> bwd(
      absl::Span<const Buffer> output_gradients, Tape tape) override;
  absl::Span<Buffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<Buffer> gradients() override {
    return absl::MakeSpan(gradients_);
  }
  DataType output_type() const override { return output_type_; }

 private:
  LayerNormLayer(int embedding_dim, float epsilon, DataType data_type,
                 cudaStream_t stream, Buffer gamma, Buffer beta,
                 Buffer gamma_gradient, Buffer beta_gradient)
      : embedding_dim_(embedding_dim),
        epsilon_(epsilon),
        output_type_(data_type),
        stream_(stream),
        weights_{std::move(gamma), std::move(beta)},
        gradients_{std::move(gamma_gradient), std::move(beta_gradient)} {}

  int embedding_dim_;
  float epsilon_;
  DataType output_type_;
  cudaStream_t stream_;
  BufferVec weights_;
  BufferVec gradients_;
};

}  // namespace pluto::llm
