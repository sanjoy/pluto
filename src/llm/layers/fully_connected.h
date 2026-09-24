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
// FP32 master buffers for the external optimizer. Positive row counts and
// feature widths need not be multiples of the compute-tile dimensions.
class FullyConnectedLayer final : public Layer {
 public:
  absl::string_view name() const override { return "FullyConnectedLayer"; }

  // sequence_length is activation rows/tokens per sample, not batch size.
  // The default treats each row as its own sample. Signatures preserve the
  // batch, sequence, and feature axes even though kernels flatten
  // batch/sequence.
  static absl::StatusOr<std::unique_ptr<FullyConnectedLayer>> Create(
      cuda::Executor& executor, int input_dim, int output_dim,
      DataType data_type, int sequence_length = 1);
  static absl::StatusOr<std::unique_ptr<FullyConnectedLayer>> Create(
      cuda::Executor& executor, int model_width, DataType data_type,
      int sequence_length = 1) {
    return Create(executor, model_width, model_width, data_type,
                  sequence_length);
  }

  // Initializes the rectangular matrix to a scaled identity on its available
  // diagonal and clears the bias.
  absl::Status InitializeIdentity(float scale = 1.0f);
  absl::Status InitializeNormal(float standard_deviation, uint64_t seed);

  absl::Span<Buffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<Buffer> gradients() override { return absl::MakeSpan(gradients_); }
  DataType output_type() const override { return output_type_; }

  absl::Span<const ActivationType> input_types() const override {
    return absl::MakeConstSpan(&input_type_, 1);
  }
  absl::Span<const ActivationType> output_types() const override {
    return absl::MakeConstSpan(&output_type_signature_, 1);
  }

  int input_dim() const { return input_dim_; }
  int output_dim() const { return output_dim_; }

 private:
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state, LayerHooks*) override;

  FullyConnectedLayer(cuda::Executor& executor, int input_dim, int output_dim,
                      DataType data_type, Buffer matrix, Buffer bias,
                      Buffer matrix_gradient, Buffer bias_gradient,
                      int sequence_length);

  int input_dim_;
  int output_dim_;
  int sequence_length_;
  DataType output_type_;
  cuda::Executor& executor_;
  BufferVec weights_;
  BufferVec gradients_;
  // Shapes retain the sequence axis; the batch sentinel only matches itself.
  const ActivationType input_type_{
      ActivationDataType(output_type_),
      {ActivationType::kBatchDimension, sequence_length_, input_dim_}};
  const ActivationType output_type_signature_{
      ActivationDataType(output_type_),
      {ActivationType::kBatchDimension, sequence_length_, output_dim_}};
};

// Scalar CPU specification for FullyConnectedLayer. It exposes FP32 master
// parameters and gradients in the same order as the device layer.
class FullyConnectedLayerReference final : public LayerReference {
 public:
  absl::string_view name() const override {
    return "FullyConnectedLayerReference";
  }

  // sequence_length is activation rows/tokens per sample, not batch size.
  // The default treats each row as its own sample. Signatures preserve the
  // batch, sequence, and feature axes even though kernels flatten
  // batch/sequence.
  static absl::StatusOr<std::unique_ptr<FullyConnectedLayerReference>> Create(
      int input_dim, int output_dim, DataType data_type,
      int sequence_length = 1);
  static absl::StatusOr<std::unique_ptr<FullyConnectedLayerReference>> Create(
      int model_width, DataType data_type, int sequence_length = 1) {
    return Create(model_width, model_width, data_type, sequence_length);
  }

  absl::Status InitializeIdentity(float scale = 1.0f);
  absl::Status InitializeNormal(float standard_deviation, uint64_t seed);

  absl::Span<HostBuffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<HostBuffer> gradients() override {
    return absl::MakeSpan(gradients_);
  }
  DataType output_type() const override { return output_type_; }

  absl::Span<const ActivationType> input_types() const override {
    return absl::MakeConstSpan(&input_type_, 1);
  }
  absl::Span<const ActivationType> output_types() const override {
    return absl::MakeConstSpan(&output_type_signature_, 1);
  }

  int input_dim() const { return input_dim_; }
  int output_dim() const { return output_dim_; }

 private:
  absl::StatusOr<ReferenceFwdResult> fwd_impl(
      absl::Span<const HostBuffer> inputs) const override;
  absl::StatusOr<HostBufferVec> bwd_impl(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceBackwardState state) override;

  FullyConnectedLayerReference(int input_dim, int output_dim,
                               DataType data_type, HostBuffer matrix,
                               HostBuffer bias, HostBuffer matrix_gradient,
                               HostBuffer bias_gradient, int sequence_length)
      : input_dim_(input_dim),
        output_dim_(output_dim),
        sequence_length_(sequence_length),
        output_type_(data_type),
        weights_{std::move(matrix), std::move(bias)},
        gradients_{std::move(matrix_gradient), std::move(bias_gradient)} {}

  int input_dim_;
  int output_dim_;
  int sequence_length_;
  DataType output_type_;
  HostBufferVec weights_;
  HostBufferVec gradients_;
  // Shapes retain the sequence axis; the batch sentinel only matches itself.
  const ActivationType input_type_{
      ActivationDataType(output_type_),
      {ActivationType::kBatchDimension, sequence_length_, input_dim_}};
  const ActivationType output_type_signature_{
      ActivationDataType(output_type_),
      {ActivationType::kBatchDimension, sequence_length_, output_dim_}};
};

}  // namespace pluto::llm
