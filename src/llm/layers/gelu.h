#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// Elementwise Gaussian Error Linear Unit. The explicit feature width keeps
// type checking exact across this otherwise shape-preserving operation. Feature
// widths need only be positive; the total runtime element count must be tiled.
class GeluLayer final : public Layer {
 public:
  // sequence_length is activation rows/tokens per sample, not batch size.
  // The default treats each row as its own sample. Signatures preserve the
  // batch, sequence, and feature axes even though kernels flatten
  // batch/sequence.
  static absl::StatusOr<std::unique_ptr<GeluLayer>> Create(
      cuda::Executor& executor, int embedding_dim, DataType data_type,
      int sequence_length = 1);

  absl::Status ValidateSequenceLength(int sequence_length) const override {
    if (sequence_length != sequence_length_)
      return absl::InvalidArgumentError(
          "sequence_length must match the layer's configured sample shape");
    return absl::OkStatus();
  }

  absl::Span<Buffer> weights() override { return {}; }
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

  GeluLayer(cuda::Executor& executor, int embedding_dim, DataType data_type,
            int sequence_length)
      : embedding_dim_(embedding_dim),
        sequence_length_(sequence_length),
        output_type_(data_type),
        executor_(executor) {}

  int embedding_dim_;
  int sequence_length_;
  DataType output_type_;
  cuda::Executor& executor_;
  // Shapes retain the sequence axis; the batch sentinel only matches itself.
  const ActivationType input_type_{
      ActivationDataType(output_type_),
      {ActivationType::kBatchDimension, sequence_length_, embedding_dim_}};
  const ActivationType output_type_signature_{
      ActivationDataType(output_type_),
      {ActivationType::kBatchDimension, sequence_length_, embedding_dim_}};
};

class GeluLayerReference final : public LayerReference {
 public:
  // sequence_length is activation rows/tokens per sample, not batch size.
  // The default treats each row as its own sample. Signatures preserve the
  // batch, sequence, and feature axes even though kernels flatten
  // batch/sequence.
  static absl::StatusOr<std::unique_ptr<GeluLayerReference>> Create(
      int embedding_dim, DataType data_type, int sequence_length = 1);

  absl::Span<HostBuffer> weights() override { return {}; }
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

  GeluLayerReference(int embedding_dim, DataType data_type, int sequence_length)
      : embedding_dim_(embedding_dim),
        sequence_length_(sequence_length),
        output_type_(data_type) {}

  int embedding_dim_;
  int sequence_length_;
  DataType output_type_;
  // Shapes retain the sequence axis; the batch sentinel only matches itself.
  const ActivationType input_type_{
      ActivationDataType(output_type_),
      {ActivationType::kBatchDimension, sequence_length_, embedding_dim_}};
  const ActivationType output_type_signature_{
      ActivationDataType(output_type_),
      {ActivationType::kBatchDimension, sequence_length_, embedding_dim_}};
};

}  // namespace pluto::llm
