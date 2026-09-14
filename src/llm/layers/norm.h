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
  // sequence_length is activation rows/tokens per sample, not batch size.
  // The default treats each row as its own sample. Signatures preserve the
  // batch, sequence, and feature axes even though kernels flatten
  // batch/sequence.
  static absl::StatusOr<std::unique_ptr<LayerNormLayer>> Create(
      cuda::Executor& executor, int embedding_dim, float epsilon,
      DataType data_type, int sequence_length = 1);

  absl::Status ValidateSequenceLength(int sequence_length) const override {
    if (sequence_length != sequence_length_)
      return absl::InvalidArgumentError(
          "sequence_length must match the layer's configured sample shape");
    return absl::OkStatus();
  }

  absl::Span<Buffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<Buffer> gradients() override { return absl::MakeSpan(gradients_); }
  DataType output_type() const override { return output_type_; }

  absl::Span<const ActivationType> input_types() const override {
    return absl::MakeConstSpan(&input_type_, 1);
  }
  absl::Span<const ActivationType> output_types() const override {
    return absl::MakeConstSpan(&output_type_signature_, 1);
  }

 private:
  absl::StatusOr<FwdResult> fwd_impl(
      cuda::Executor& executor, absl::Span<const Buffer> inputs) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state) override;

  LayerNormLayer(cuda::Executor& executor, int embedding_dim, float epsilon,
                 DataType data_type, Buffer gamma, Buffer beta,
                 Buffer gamma_gradient, Buffer beta_gradient,
                 int sequence_length)
      : embedding_dim_(embedding_dim),
        sequence_length_(sequence_length),
        epsilon_(epsilon),
        output_type_(data_type),
        executor_(executor),
        weights_{std::move(gamma), std::move(beta)},
        gradients_{std::move(gamma_gradient), std::move(beta_gradient)} {}

  int embedding_dim_;
  int sequence_length_;
  float epsilon_;
  DataType output_type_;
  cuda::Executor& executor_;
  BufferVec weights_;
  BufferVec gradients_;
  // Shapes retain the sequence axis; the batch sentinel only matches itself.
  const ActivationType input_type_{
      ActivationDataType(output_type_),
      {ActivationType::kBatchDimension, sequence_length_, embedding_dim_}};
  const ActivationType output_type_signature_{
      ActivationDataType(output_type_),
      {ActivationType::kBatchDimension, sequence_length_, embedding_dim_}};
};

// Direct row-by-row LayerNorm equations used to validate both CUDA passes.
class LayerNormLayerReference final : public LayerReference {
 public:
  // sequence_length is activation rows/tokens per sample, not batch size.
  // The default treats each row as its own sample. Signatures preserve the
  // batch, sequence, and feature axes even though kernels flatten
  // batch/sequence.
  static absl::StatusOr<std::unique_ptr<LayerNormLayerReference>> Create(
      int embedding_dim, float epsilon, DataType data_type,
      int sequence_length = 1);

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

 private:
  absl::StatusOr<ReferenceFwdResult> fwd_impl(
      absl::Span<const HostBuffer> inputs) const override;
  absl::StatusOr<HostBufferVec> bwd_impl(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceBackwardState state) override;

  LayerNormLayerReference(int embedding_dim, float epsilon, DataType data_type,
                          HostBuffer gamma, HostBuffer beta,
                          HostBuffer gamma_gradient, HostBuffer beta_gradient,
                          int sequence_length)
      : embedding_dim_(embedding_dim),
        sequence_length_(sequence_length),
        epsilon_(epsilon),
        output_type_(data_type),
        weights_{std::move(gamma), std::move(beta)},
        gradients_{std::move(gamma_gradient), std::move(beta_gradient)} {}

  int embedding_dim_;
  int sequence_length_;
  float epsilon_;
  DataType output_type_;
  HostBufferVec weights_;
  HostBufferVec gradients_;
  // Shapes retain the sequence axis; the batch sentinel only matches itself.
  const ActivationType input_type_{
      ActivationDataType(output_type_),
      {ActivationType::kBatchDimension, sequence_length_, embedding_dim_}};
  const ActivationType output_type_signature_{
      ActivationDataType(output_type_),
      {ActivationType::kBatchDimension, sequence_length_, embedding_dim_}};
};

}  // namespace pluto::llm
