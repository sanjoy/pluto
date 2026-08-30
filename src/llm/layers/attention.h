#pragma once

#include <cuda_runtime_api.h>

#include <memory>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// Causal multi-head FlashAttention over a packed [Q, K, V] activation produced
// by a d_model -> 3*d_model projection. It streams visible keys/values and
// maintains FP32 online-softmax statistics without materializing the quadratic
// attention matrix. Backward recomputes probabilities and emits packed FP32
// dQ/dK/dV gradients.
class AttentionLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<AttentionLayer>> Create(
      int context_length, int num_heads, int embedding_dim,
      DataType data_type, cudaStream_t stream);

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                              Tape* tape) const override;
  absl::StatusOr<BufferVec> bwd(
      absl::Span<const Buffer> output_gradients, Tape tape) override;
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return output_type_; }

 private:
  AttentionLayer(int context_length, int num_heads, int embedding_dim,
                 DataType data_type, cudaStream_t stream)
      : context_length_(context_length),
        num_heads_(num_heads),
        embedding_dim_(embedding_dim),
        output_type_(data_type),
        stream_(stream) {}

  int context_length_;
  int num_heads_;
  int embedding_dim_;
  DataType output_type_;
  cudaStream_t stream_;
};

// Scalar causal multi-head attention used as an executable specification for
// the tiled CUDA implementation. Readability is intentionally favored over
// speed: it materializes each query row's softmax probabilities.
class AttentionLayerReference final : public LayerReference {
 public:
  static absl::StatusOr<std::unique_ptr<AttentionLayerReference>> Create(
      int context_length, int num_heads, int embedding_dim,
      DataType data_type);

  absl::StatusOr<HostBuffer> fwd(absl::Span<const HostBuffer> inputs,
                                  ReferenceTape* tape) override;
  absl::StatusOr<HostBufferVec> bwd(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceTape tape) override;
  absl::Span<HostBuffer> weights() override { return {}; }
  DataType output_type() const override { return output_type_; }

 private:
  AttentionLayerReference(int context_length, int num_heads,
                          int embedding_dim, DataType data_type)
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
