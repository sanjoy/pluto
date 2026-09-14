#pragma once

#include <memory>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// Causal multi-head FlashAttention over a packed [Q, K, V] activation produced
// by a d_model -> 3*d_model projection. Query/key tiles use cuTile matrix
// products and FP32 online softmax without materializing the quadratic
// attention matrix. Forward retains its FP32 maximum and normalizer per
// row/head; backward adds one FP32 delta statistic and recomputes probability
// tiles. Query-owned dQ and key-owned dK/dV tiles use fixed-order,
// single-writer reductions, with no floating-point atomics. Runs are bitwise
// repeatable on the same GPU/software stack, but changing tiling/reduction
// order can change rounding from older implementations. Partial tiles are
// masked at both sequence and head bounds.
class AttentionLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<AttentionLayer>> Create(
      cuda::Executor& executor, int context_length, int num_heads,
      int embedding_dim, DataType data_type);

  // These kernels reset attention at the configured fixed-width boundaries.
  absl::Status ValidateSequenceLength(int sequence_length) const override;

  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return output_type_; }
  absl::Span<const ActivationType> input_types() const override {
    return input_types_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return output_types_;
  }

 private:
  absl::StatusOr<FwdResult> fwd_impl(
      cuda::Executor& executor, absl::Span<const Buffer> inputs) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state) override;

  AttentionLayer(cuda::Executor& executor, int context_length, int num_heads,
                 int embedding_dim, DataType data_type)
      : context_length_(context_length),
        num_heads_(num_heads),
        embedding_dim_(embedding_dim),
        output_type_(data_type),
        executor_(executor) {}

  int context_length_;
  int num_heads_;
  int embedding_dim_;
  DataType output_type_;
  cuda::Executor& executor_;
  // The batch sentinel is symbolic; context and packed feature width are exact.
  const ActivationType input_types_[1] = {
      {ActivationDataType(output_type_),
       {ActivationType::kBatchDimension, context_length_,
        3LL * embedding_dim_}}};
  const ActivationType output_types_[1] = {
      {ActivationDataType(output_type_),
       {ActivationType::kBatchDimension, context_length_, embedding_dim_}}};
};

// Scalar causal multi-head attention used as an executable specification for
// the tiled CUDA implementation. Readability is intentionally favored over
// speed: it materializes each query row's softmax probabilities.
class AttentionLayerReference final : public LayerReference {
 public:
  static absl::StatusOr<std::unique_ptr<AttentionLayerReference>> Create(
      int context_length, int num_heads, int embedding_dim, DataType data_type);

  absl::Span<HostBuffer> weights() override { return {}; }
  DataType output_type() const override { return output_type_; }
  absl::Span<const ActivationType> input_types() const override {
    return input_types_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return output_types_;
  }

 private:
  absl::StatusOr<ReferenceFwdResult> fwd_impl(
      absl::Span<const HostBuffer> inputs) const override;
  absl::StatusOr<HostBufferVec> bwd_impl(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceBackwardState state) override;

  AttentionLayerReference(int context_length, int num_heads, int embedding_dim,
                          DataType data_type)
      : context_length_(context_length),
        num_heads_(num_heads),
        embedding_dim_(embedding_dim),
        output_type_(data_type) {}

  int context_length_;
  int num_heads_;
  int embedding_dim_;
  DataType output_type_;
  // The batch sentinel is symbolic; context and packed feature width are exact.
  const ActivationType input_types_[1] = {
      {ActivationDataType(output_type_),
       {ActivationType::kBatchDimension, context_length_,
        3LL * embedding_dim_}}};
  const ActivationType output_types_[1] = {
      {ActivationDataType(output_type_),
       {ActivationType::kBatchDimension, context_length_, embedding_dim_}}};
};

}  // namespace pluto::llm
