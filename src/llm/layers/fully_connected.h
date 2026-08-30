#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// A bias-bearing rectangular projection. Activations use the selected compute
// type, MMA reductions accumulate in FP32, and parameters/gradients remain
// FP32 master buffers for the external optimizer.
class FullyConnectedLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<FullyConnectedLayer>> Create(
      int input_dim, int output_dim, DataType data_type,
      cuda::Executor& executor);
  static absl::StatusOr<std::unique_ptr<FullyConnectedLayer>> Create(
      int model_width, DataType data_type, cuda::Executor& executor) {
    return Create(model_width, model_width, data_type, executor);
  }

  // Initializes the rectangular matrix to a scaled identity on its available
  // diagonal and clears the bias.
  absl::Status InitializeIdentity(float scale = 1.0f);
  absl::Status InitializeNormal(float standard_deviation, uint64_t seed);

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs, Tape* tape,
                             cuda::Executor& executor) const override;
  absl::StatusOr<BufferVec> bwd(absl::Span<const Buffer> output_gradients,
                                Tape tape, cuda::Executor& executor) override;
  absl::Span<Buffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<Buffer> gradients() override { return absl::MakeSpan(gradients_); }
  DataType output_type() const override { return output_type_; }

  int input_dim() const { return input_dim_; }
  int output_dim() const { return output_dim_; }

 private:
  FullyConnectedLayer(int input_dim, int output_dim, DataType data_type,
                      cuda::Executor& executor, Buffer matrix, Buffer bias,
                      Buffer matrix_gradient, Buffer bias_gradient);

  int input_dim_;
  int output_dim_;
  DataType output_type_;
  cuda::Executor& executor_;
  BufferVec weights_;
  BufferVec gradients_;
};

// Scalar CPU specification for FullyConnectedLayer. It exposes FP32 master
// parameters and gradients in the same order as the device layer.
class FullyConnectedLayerReference final : public LayerReference {
 public:
  static absl::StatusOr<std::unique_ptr<FullyConnectedLayerReference>> Create(
      int input_dim, int output_dim, DataType data_type);
  static absl::StatusOr<std::unique_ptr<FullyConnectedLayerReference>> Create(
      int model_width, DataType data_type) {
    return Create(model_width, model_width, data_type);
  }

  absl::Status InitializeIdentity(float scale = 1.0f);
  absl::Status InitializeNormal(float standard_deviation, uint64_t seed);

  absl::StatusOr<HostBuffer> fwd(absl::Span<const HostBuffer> inputs,
                                 ReferenceTape* tape) override;
  absl::StatusOr<HostBufferVec> bwd(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceTape tape) override;
  absl::Span<HostBuffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<HostBuffer> gradients() override {
    return absl::MakeSpan(gradients_);
  }
  DataType output_type() const override { return output_type_; }

  int input_dim() const { return input_dim_; }
  int output_dim() const { return output_dim_; }

 private:
  FullyConnectedLayerReference(int input_dim, int output_dim,
                               DataType data_type, HostBuffer matrix,
                               HostBuffer bias, HostBuffer matrix_gradient,
                               HostBuffer bias_gradient)
      : input_dim_(input_dim),
        output_dim_(output_dim),
        output_type_(data_type),
        weights_{std::move(matrix), std::move(bias)},
        gradients_{std::move(matrix_gradient), std::move(bias_gradient)} {}

  int input_dim_;
  int output_dim_;
  DataType output_type_;
  HostBufferVec weights_;
  HostBufferVec gradients_;
};

}  // namespace pluto::llm
