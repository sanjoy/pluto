#pragma once

#include <memory>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// Causal multi-head FlashAttention over a packed [Q, K, V] activation produced
// by a d_model -> 3*d_model projection. It streams visible keys/values and
// maintains FP32 online-softmax statistics without materializing the quadratic
// attention matrix. Backward recomputes probabilities and emits packed FP32
// dQ/dK/dV gradients with fixed-order, single-writer reductions. Backward uses
// three FP32 statistics per row/head and no floating-point atomic additions.
class AttentionLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<AttentionLayer>> Create(
      cuda::Executor& executor, int context_length, int num_heads,
      int embedding_dim, DataType data_type);

  // These kernels reset attention at the configured fixed-width boundaries.
  absl::Status ValidateSequenceLength(int sequence_length) const override;

  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return output_type_; }

 private:
  absl::StatusOr<Buffer> fwd_impl(cuda::Executor& executor,
                                  absl::Span<const Buffer> inputs,
                                  BackwardState& state) const override;
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

 private:
  absl::StatusOr<HostBuffer> fwd_impl(
      absl::Span<const HostBuffer> inputs,
      ReferenceBackwardState& state) const override;
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
};

}  // namespace pluto::llm
