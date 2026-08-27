#ifndef PLUTO_SRC_LLM_LAYERS_EMBEDDING_H_
#define PLUTO_SRC_LLM_LAYERS_EMBEDDING_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// Maps a batch of int32 token IDs to embedding vectors. The table is stored as
// one FP32 master weight; the forward kernel rounds values through FP16 before
// they are consumed. Backward atomically applies SGD because a batch may
// contain the same token more than once.
class EmbeddingLookupLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<EmbeddingLookupLayer>> Create(
      int vocab_size, int embedding_dim, DataType data_type,
      float learning_rate, cudaStream_t stream);

  // Initializes the rectangular table to the identity on its main diagonal.
  // This is useful for tied byte-level models, where an all-zero table would
  // make both sides of the initial E * E^T projection have zero gradient.
  absl::Status InitializeIdentity(float scale = 1.0f);

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                              Tape* tape) override;
  absl::StatusOr<BufferVec> bwd(
      absl::Span<const Buffer> output_gradients, Tape tape) override;
  absl::Span<Buffer> weights() override {
    return absl::MakeSpan(&weight_, 1);
  }
  DataType output_type() const override { return output_type_; }

  int vocab_size() const { return vocab_size_; }
  int embedding_dim() const { return embedding_dim_; }
  const Buffer& weight() const { return weight_; }

 private:
  EmbeddingLookupLayer(int vocab_size, int embedding_dim, DataType data_type,
                       float learning_rate, cudaStream_t stream, Buffer weight);

  friend class LanguageModelingHeadLayer;

  int vocab_size_;
  int embedding_dim_;
  DataType output_type_;
  float learning_rate_;
  cudaStream_t stream_;
  Buffer weight_;
};

// Projects hidden states to vocabulary logits using the transpose of an
// existing embedding table. The pointer is non-owning: the embedding must
// outlive this layer. Returning the same Buffer from weights() makes the
// parameter sharing explicit to model introspection as well as to the kernels.
class LanguageModelingHeadLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<LanguageModelingHeadLayer>> Create(
      EmbeddingLookupLayer* embedding);

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                              Tape* tape) override;
  absl::StatusOr<BufferVec> bwd(
      absl::Span<const Buffer> output_gradients, Tape tape) override;
  absl::Span<Buffer> weights() override { return embedding_->weights(); }
  DataType output_type() const override { return embedding_->output_type(); }

 private:
  explicit LanguageModelingHeadLayer(EmbeddingLookupLayer* embedding)
      : embedding_(embedding) {}

  EmbeddingLookupLayer* embedding_;
};

// Adds a learned position vector to each token. Input rows are laid out as
// consecutive sequences, so positions repeat every context_length rows.
// Backward returns the activation gradient unchanged and accumulates the
// position-weight update across sequences.
class PositionEmbeddingLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<PositionEmbeddingLayer>> Create(
      int context_length, int embedding_dim, DataType data_type,
      float learning_rate, cudaStream_t stream);

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                              Tape* tape) override;
  absl::StatusOr<BufferVec> bwd(
      absl::Span<const Buffer> output_gradients, Tape tape) override;
  absl::Span<Buffer> weights() override {
    return absl::MakeSpan(&weight_, 1);
  }
  DataType output_type() const override { return output_type_; }

 private:
  PositionEmbeddingLayer(int context_length, int embedding_dim,
                         DataType data_type, float learning_rate,
                         cudaStream_t stream, Buffer weight);

  int context_length_;
  int embedding_dim_;
  DataType output_type_;
  float learning_rate_;
  cudaStream_t stream_;
  Buffer weight_;
};

}  // namespace pluto::llm

#endif  // PLUTO_SRC_LLM_LAYERS_EMBEDDING_H_
