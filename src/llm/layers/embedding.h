#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// Token embedding with an FP32 master table and FP32 accumulated gradient.
// The physical vocabulary is padded for 16-wide MMA, while vocab_size() stays
// the exact logical vocabulary accepted by the tokenizer.
class EmbeddingLookupLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<EmbeddingLookupLayer>> Create(
      int vocab_size, int embedding_dim, DataType data_type,
      cudaStream_t stream);

  absl::Status InitializeIdentity(float scale = 1.0f);
  absl::Status InitializeNormal(float standard_deviation, uint64_t seed);

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                             Tape* tape) const override;
  absl::StatusOr<BufferVec> bwd(absl::Span<const Buffer> output_gradients,
                                Tape tape) override;
  absl::Span<Buffer> weights() override { return absl::MakeSpan(&weight_, 1); }
  absl::Span<Buffer> gradients() override {
    return absl::MakeSpan(&gradient_, 1);
  }
  DataType output_type() const override { return output_type_; }

  int vocab_size() const { return vocab_size_; }
  int padded_vocab_size() const { return padded_vocab_size_; }
  int embedding_dim() const { return embedding_dim_; }
  const Buffer& weight() const { return weight_; }

 private:
  EmbeddingLookupLayer(int vocab_size, int padded_vocab_size, int embedding_dim,
                       DataType data_type, cudaStream_t stream, Buffer weight,
                       Buffer gradient);

  friend class LanguageModelingHeadLayer;

  int vocab_size_;
  int padded_vocab_size_;
  int embedding_dim_;
  DataType output_type_;
  cudaStream_t stream_;
  Buffer weight_;
  Buffer gradient_;
};

// Tied output projection. Logits use FP32 and have padded_vocab_size columns;
// lanes beyond vocab_size are -infinity and therefore receive zero probability
// and gradient. The embedding owns the shared master weight and gradient.
class LanguageModelingHeadLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<LanguageModelingHeadLayer>> Create(
      EmbeddingLookupLayer* embedding);

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                             Tape* tape) const override;
  absl::StatusOr<BufferVec> bwd(absl::Span<const Buffer> output_gradients,
                                Tape tape) override;
  absl::Span<Buffer> weights() override { return embedding_->weights(); }
  absl::Span<Buffer> gradients() override { return embedding_->gradients(); }
  DataType output_type() const override { return embedding_->output_type(); }

 private:
  explicit LanguageModelingHeadLayer(EmbeddingLookupLayer* embedding)
      : embedding_(embedding) {}

  EmbeddingLookupLayer* embedding_;
};

// Learned absolute position embeddings repeated for each packed sequence.
class PositionEmbeddingLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<PositionEmbeddingLayer>> Create(
      int context_length, int embedding_dim, DataType data_type,
      cudaStream_t stream);

  absl::Status InitializeNormal(float standard_deviation, uint64_t seed);

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                             Tape* tape) const override;
  absl::StatusOr<BufferVec> bwd(absl::Span<const Buffer> output_gradients,
                                Tape tape) override;
  absl::Span<Buffer> weights() override { return absl::MakeSpan(&weight_, 1); }
  absl::Span<Buffer> gradients() override {
    return absl::MakeSpan(&gradient_, 1);
  }
  DataType output_type() const override { return output_type_; }

 private:
  PositionEmbeddingLayer(int context_length, int embedding_dim,
                         DataType data_type, cudaStream_t stream, Buffer weight,
                         Buffer gradient);

  int context_length_;
  int embedding_dim_;
  DataType output_type_;
  cudaStream_t stream_;
  Buffer weight_;
  Buffer gradient_;
};

class LanguageModelingHeadLayerReference;

// Scalar table lookup reference. Master weights and gradients remain FP32,
// exactly like the GPU layer, while returned activations use output_type().
class EmbeddingLookupLayerReference final : public LayerReference {
 public:
  static absl::StatusOr<std::unique_ptr<EmbeddingLookupLayerReference>> Create(
      int vocab_size, int embedding_dim, DataType data_type);

  absl::Status InitializeIdentity(float scale = 1.0f);
  absl::Status InitializeNormal(float standard_deviation, uint64_t seed);
  absl::StatusOr<HostBuffer> fwd(absl::Span<const HostBuffer> inputs,
                                 ReferenceTape* tape) override;
  absl::StatusOr<HostBufferVec> bwd(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceTape tape) override;
  absl::Span<HostBuffer> weights() override {
    return absl::MakeSpan(&weight_, 1);
  }
  absl::Span<HostBuffer> gradients() override {
    return absl::MakeSpan(&gradient_, 1);
  }
  DataType output_type() const override { return output_type_; }
  int vocab_size() const { return vocab_size_; }
  int padded_vocab_size() const { return padded_vocab_size_; }
  int embedding_dim() const { return embedding_dim_; }
  const HostBuffer& weight() const { return weight_; }

 private:
  EmbeddingLookupLayerReference(int vocab_size, int padded_vocab_size,
                                int embedding_dim, DataType data_type,
                                HostBuffer weight, HostBuffer gradient)
      : vocab_size_(vocab_size),
        padded_vocab_size_(padded_vocab_size),
        embedding_dim_(embedding_dim),
        output_type_(data_type),
        weight_(std::move(weight)),
        gradient_(std::move(gradient)) {}

  friend class LanguageModelingHeadLayerReference;
  int vocab_size_;
  int padded_vocab_size_;
  int embedding_dim_;
  DataType output_type_;
  HostBuffer weight_;
  HostBuffer gradient_;
};

// Obvious dense projection using the reference embedding's transposed table.
class LanguageModelingHeadLayerReference final : public LayerReference {
 public:
  static absl::StatusOr<std::unique_ptr<LanguageModelingHeadLayerReference>>
  Create(EmbeddingLookupLayerReference* embedding);
  absl::StatusOr<HostBuffer> fwd(absl::Span<const HostBuffer> inputs,
                                 ReferenceTape* tape) override;
  absl::StatusOr<HostBufferVec> bwd(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceTape tape) override;
  absl::Span<HostBuffer> weights() override { return embedding_->weights(); }
  absl::Span<HostBuffer> gradients() override {
    return embedding_->gradients();
  }
  DataType output_type() const override { return embedding_->output_type(); }

 private:
  explicit LanguageModelingHeadLayerReference(
      EmbeddingLookupLayerReference* embedding)
      : embedding_(embedding) {}
  EmbeddingLookupLayerReference* embedding_;
};

// Scalar learned-position addition and gradient accumulation reference.
class PositionEmbeddingLayerReference final : public LayerReference {
 public:
  static absl::StatusOr<std::unique_ptr<PositionEmbeddingLayerReference>>
  Create(int context_length, int embedding_dim, DataType data_type);
  absl::Status InitializeNormal(float standard_deviation, uint64_t seed);
  absl::StatusOr<HostBuffer> fwd(absl::Span<const HostBuffer> inputs,
                                 ReferenceTape* tape) override;
  absl::StatusOr<HostBufferVec> bwd(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceTape tape) override;
  absl::Span<HostBuffer> weights() override {
    return absl::MakeSpan(&weight_, 1);
  }
  absl::Span<HostBuffer> gradients() override {
    return absl::MakeSpan(&gradient_, 1);
  }
  DataType output_type() const override { return output_type_; }

 private:
  PositionEmbeddingLayerReference(int context_length, int embedding_dim,
                                  DataType data_type, HostBuffer weight,
                                  HostBuffer gradient)
      : context_length_(context_length),
        embedding_dim_(embedding_dim),
        output_type_(data_type),
        weight_(std::move(weight)),
        gradient_(std::move(gradient)) {}
  int context_length_;
  int embedding_dim_;
  DataType output_type_;
  HostBuffer weight_;
  HostBuffer gradient_;
};

}  // namespace pluto::llm
