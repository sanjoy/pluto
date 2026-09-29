#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"
#include "src/llm/layers/matrix_common.h"

namespace pluto::llm {

// A rectangular projection. Trainable projections have a bias and FP32 master
// parameters/gradients; imported projections share frozen, output-major weights
// without allocating optimizer storage. Both accumulate reductions in FP32.
// Positive feature widths need not be multiples of the compute-tile dimensions.
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

  // Imports a biasless [output_dim, input_dim] matrix without copying it.
  // BF16/FP32 weights have no scales; FP8 requires FP32 block scales with shape
  // [ceil(output_dim/128), ceil(input_dim/128)]. This frozen single-token path
  // consumes/produces BF16 [1, 1, width] activations and dynamically quantizes
  // its FP8 input in independent groups of 128 elements. It has no gradients
  // and rejects initialization and backward instead of mutating the checkpoint.
  static absl::StatusOr<std::unique_ptr<FullyConnectedLayer>> Create(
      cuda::Executor& executor, Buffer weights, MatrixStorage storage,
      int input_dim, int output_dim,
      std::optional<Buffer> scales = std::nullopt);

  // Bound for the imported single-token kernels; not a trainable-layer limit.
  static constexpr int kMaximumDimension = 1048576;

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
  FullyConnectedLayer(cuda::Executor& executor, int input_dim, int output_dim,
                      BufferVec weights, MatrixStorage storage);
  absl::StatusOr<FwdResult> ImportedForward(
      cuda::Executor& executor, absl::Span<const Buffer> inputs) const;

  int input_dim_;
  int output_dim_;
  int sequence_length_;
  DataType output_type_;
  cuda::Executor& executor_;
  BufferVec weights_;
  BufferVec gradients_;
  // Engaged only for the frozen output-major checkpoint representation.
  std::optional<MatrixStorage> imported_storage_;
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
