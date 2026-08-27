#ifndef PLUTO_SRC_LLM_LAYERS_H_
#define PLUTO_SRC_LLM_LAYERS_H_

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

// The first cuTile backend is intentionally fixed-shape. Fixed tiles eliminate
// dynamic-shape branches inside this small training test and make each batch a
// predictable amount of GPU work. Future model configs can add dispatch for
// more shapes without changing Layer or Tape.
inline constexpr int kBatchSize = 256;
inline constexpr int kModelWidth = 256;
inline constexpr int kVocabularySize = 256;

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
  absl::Status InitializeIdentity();

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

// A bias-bearing kModelWidth x kModelWidth dense layer. Matrix products are
// implemented as 16x16 cuTile MMAs. FP32 master parameters are rounded to FP16
// at the MMA boundary, which keeps updates stable while exercising FP16 math.
class FullyConnectedLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<FullyConnectedLayer>> Create(
      DataType data_type, float learning_rate, cudaStream_t stream);

  // Makes this layer an exact identity at its FP16 compute boundary. Useful
  // when inserting it into a small model without changing its initial logits.
  absl::Status InitializeIdentity();

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                              Tape* tape) override;
  absl::StatusOr<BufferVec> bwd(
      absl::Span<const Buffer> output_gradients, Tape tape) override;
  absl::Span<Buffer> weights() override {
    return absl::MakeSpan(weights_);
  }
  DataType output_type() const override { return output_type_; }

 private:
  FullyConnectedLayer(DataType data_type, float learning_rate,
                      cudaStream_t stream, Buffer matrix, Buffer bias);

  DataType output_type_;
  float learning_rate_;
  cudaStream_t stream_;
  BufferVec weights_;
};

// Computes one cross-entropy value per batch element. fwd() takes two inputs:
// [logits(float), target_token(int32)]. bwd() takes no upstream gradient
// because this is a terminal loss and returns the mean-loss gradient for the
// logits. Keeping per-example losses makes diagnostics and tests more useful.
class CrossEntropyLossLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<CrossEntropyLossLayer>> Create(
      DataType data_type, cudaStream_t stream);

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                              Tape* tape) override;
  absl::StatusOr<BufferVec> bwd(
      absl::Span<const Buffer> output_gradients, Tape tape) override;
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return output_type_; }

 private:
  CrossEntropyLossLayer(DataType data_type, cudaStream_t stream)
      : output_type_(data_type), stream_(stream) {}

  DataType output_type_;
  cudaStream_t stream_;
};

// Sequentially composes unary layers. Multi-input terminal operations, such as
// cross entropy with labels, intentionally remain outside the predictor.
class ComposedLayer final : public Layer {
 public:
  ComposedLayer(DataType data_type,
                std::vector<std::unique_ptr<Layer>> layers);

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                              Tape* tape) override;
  absl::StatusOr<BufferVec> bwd(
      absl::Span<const Buffer> output_gradients, Tape tape) override;
  absl::Span<Buffer> weights() override {
    return absl::MakeSpan(weights_);
  }
  DataType output_type() const override { return output_type_; }

 private:
  DataType output_type_;
  std::vector<std::unique_ptr<Layer>> layers_;
  std::vector<Buffer> weights_;
};

// Repeats a concrete Layer type while retaining ordinary Layer polymorphism.
// Each repetition has independent parameters; callers that want tied weights
// can instead pass one layer explicitly from a custom composite.
template <class LayerToRepeat>
class RepeatedLayer final : public Layer {
 public:
  RepeatedLayer(DataType data_type,
                std::vector<std::unique_ptr<LayerToRepeat>> layers)
      : output_type_(data_type), layers_(std::move(layers)) {
    for (const auto& layer : layers_) {
      for (Buffer& weight : layer->weights()) weights_.push_back(weight);
    }
  }

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                              Tape* tape) override {
    if (inputs.size() != 1) {
      return absl::InvalidArgumentError(
          "RepeatedLayer fwd expects one activation buffer");
    }
    tape->intermediates.clear();
    tape->children.clear();
    Buffer activation = inputs.front();
    for (auto& layer : layers_) {
      Tape child_tape;
      BufferVec child_inputs = {activation};
      auto output = layer->fwd(child_inputs, &child_tape);
      if (!output.ok()) return output.status();
      activation = *std::move(output);
      tape->children.push_back(std::move(child_tape));
    }
    return activation;
  }

  absl::StatusOr<BufferVec> bwd(
      absl::Span<const Buffer> output_gradients, Tape tape) override {
    if (output_gradients.size() != 1 ||
        tape.children.size() != layers_.size()) {
      return absl::InvalidArgumentError(
          "RepeatedLayer bwd received an incompatible gradient or tape");
    }
    Buffer gradient = output_gradients.front();
    for (size_t index = layers_.size(); index-- > 0;) {
      BufferVec child_gradients = {gradient};
      auto inputs = layers_[index]->bwd(
          child_gradients, std::move(tape.children[index]));
      if (!inputs.ok()) return inputs.status();
      if (inputs->size() != 1) {
        return absl::InternalError(
            "a repeated unary layer returned multiple input gradients");
      }
      gradient = inputs->front();
    }
    return BufferVec{gradient};
  }

  absl::Span<Buffer> weights() override {
    return absl::MakeSpan(weights_);
  }
  DataType output_type() const override { return output_type_; }

 private:
  DataType output_type_;
  std::vector<std::unique_ptr<LayerToRepeat>> layers_;
  std::vector<Buffer> weights_;
};

}  // namespace pluto::llm

#endif  // PLUTO_SRC_LLM_LAYERS_H_
