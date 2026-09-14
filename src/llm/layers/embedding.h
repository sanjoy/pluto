#pragma once

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
  absl::string_view name() const override { return "EmbeddingLookupLayer"; }

  // sequence_length is tokens per sample, not batch size. The default treats
  // each token as a separate one-token sample; larger values preserve a
  // [-2, sequence_length] input and [-2, sequence_length, embedding_dim]
  // output.
  static absl::StatusOr<std::unique_ptr<EmbeddingLookupLayer>> Create(
      cuda::Executor& executor, int vocab_size, int embedding_dim,
      DataType data_type, int sequence_length = 1);

  absl::Status InitializeIdentity(float scale = 1.0f);
  absl::Status InitializeNormal(float standard_deviation, uint64_t seed);

  absl::Span<Buffer> weights() override { return absl::MakeSpan(&weight_, 1); }
  absl::Span<Buffer> gradients() override {
    return absl::MakeSpan(&gradient_, 1);
  }
  DataType output_type() const override { return output_type_; }

  absl::Span<const ActivationType> input_types() const override {
    return absl::MakeConstSpan(&input_type_, 1);
  }
  absl::Span<const ActivationType> output_types() const override {
    return absl::MakeConstSpan(&output_type_signature_, 1);
  }

  absl::Status ValidateSequenceLength(int sequence_length) const override {
    if (sequence_length != sequence_length_)
      return absl::InvalidArgumentError(
          "sequence_length must match the layer's configured sample shape");
    return absl::OkStatus();
  }

  int vocab_size() const { return vocab_size_; }
  int padded_vocab_size() const { return padded_vocab_size_; }
  int embedding_dim() const { return embedding_dim_; }
  int sequence_length() const { return sequence_length_; }
  const Buffer& weight() const { return weight_; }

 private:
  absl::StatusOr<FwdResult> fwd_impl(
      cuda::Executor& executor, absl::Span<const Buffer> inputs) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state) override;

  EmbeddingLookupLayer(cuda::Executor& executor, int vocab_size,
                       int padded_vocab_size, int embedding_dim,
                       DataType data_type, Buffer weight, Buffer gradient,
                       int sequence_length);

  friend class LanguageModelingHeadLayer;

  int vocab_size_;
  int padded_vocab_size_;
  int embedding_dim_;
  int sequence_length_;
  DataType output_type_;
  cuda::Executor& executor_;
  Buffer weight_;
  Buffer gradient_;
  // Shapes retain the sequence axis; the batch sentinel only matches itself.
  const ActivationType input_type_{
      DataType::INT32, {ActivationType::kBatchDimension, sequence_length_}};
  const ActivationType output_type_signature_{
      ActivationDataType(output_type_),
      {ActivationType::kBatchDimension, sequence_length_, embedding_dim_}};
};

// Tied output projection. Logits use FP32 and have padded_vocab_size columns;
// lanes beyond vocab_size are -infinity and therefore receive zero probability
// and gradient. The embedding owns the shared master weight and gradient.
class LanguageModelingHeadLayer final : public Layer {
 public:
  absl::string_view name() const override {
    return "LanguageModelingHeadLayer";
  }

  static absl::StatusOr<std::unique_ptr<LanguageModelingHeadLayer>> Create(
      EmbeddingLookupLayer* embedding);

  absl::Span<Buffer> weights() override { return embedding_->weights(); }
  absl::Span<Buffer> gradients() override { return embedding_->gradients(); }
  DataType output_type() const override { return embedding_->output_type(); }
  absl::Span<const ActivationType> input_types() const override {
    return embedding_->output_types();
  }
  absl::Span<const ActivationType> output_types() const override {
    return absl::MakeConstSpan(&output_type_signature_, 1);
  }

  absl::Status ValidateSequenceLength(int sequence_length) const override {
    return embedding_->ValidateSequenceLength(sequence_length);
  }

 private:
  absl::StatusOr<FwdResult> fwd_impl(
      cuda::Executor& executor, absl::Span<const Buffer> inputs) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state) override;

  explicit LanguageModelingHeadLayer(EmbeddingLookupLayer* embedding)
      : embedding_(embedding) {}

  EmbeddingLookupLayer* embedding_;
  const ActivationType output_type_signature_{
      DataType::FP32,
      {ActivationType::kBatchDimension, embedding_->sequence_length(),
       embedding_->padded_vocab_size()}};
};

// Learned absolute position embeddings repeated for each packed sequence.
class PositionEmbeddingLayer final : public Layer {
 public:
  absl::string_view name() const override { return "PositionEmbeddingLayer"; }

  // Positions restart at the configured fixed width, not at a length inferred
  // from the total number of rows in a flattened batch.
  absl::Status ValidateSequenceLength(int sequence_length) const override;

  static absl::StatusOr<std::unique_ptr<PositionEmbeddingLayer>> Create(
      cuda::Executor& executor, int context_length, int embedding_dim,
      DataType data_type);

  absl::Status InitializeNormal(float standard_deviation, uint64_t seed);

  absl::Span<Buffer> weights() override { return absl::MakeSpan(&weight_, 1); }
  absl::Span<Buffer> gradients() override {
    return absl::MakeSpan(&gradient_, 1);
  }
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

  PositionEmbeddingLayer(cuda::Executor& executor, int context_length,
                         int embedding_dim, DataType data_type, Buffer weight,
                         Buffer gradient);

  int context_length_;
  int embedding_dim_;
  DataType output_type_;
  cuda::Executor& executor_;
  Buffer weight_;
  Buffer gradient_;
  // Shapes retain the sequence axis; the batch sentinel only matches itself.
  const ActivationType input_type_{
      ActivationDataType(output_type_),
      {ActivationType::kBatchDimension, context_length_, embedding_dim_}};
  const ActivationType output_type_signature_{
      ActivationDataType(output_type_),
      {ActivationType::kBatchDimension, context_length_, embedding_dim_}};
};

class LanguageModelingHeadLayerReference;

// Scalar table lookup reference. Master weights and gradients remain FP32,
// exactly like the GPU layer, while returned activations use output_type().
class EmbeddingLookupLayerReference final : public LayerReference {
 public:
  absl::string_view name() const override {
    return "EmbeddingLookupLayerReference";
  }

  // sequence_length is tokens per sample, not batch size. The default treats
  // each token as a separate one-token sample; larger values preserve a
  // [-2, sequence_length] input and [-2, sequence_length, embedding_dim]
  // output.
  static absl::StatusOr<std::unique_ptr<EmbeddingLookupLayerReference>> Create(
      int vocab_size, int embedding_dim, DataType data_type,
      int sequence_length = 1);

  absl::Status InitializeIdentity(float scale = 1.0f);
  absl::Status InitializeNormal(float standard_deviation, uint64_t seed);

  absl::Span<HostBuffer> weights() override {
    return absl::MakeSpan(&weight_, 1);
  }
  absl::Span<HostBuffer> gradients() override {
    return absl::MakeSpan(&gradient_, 1);
  }
  DataType output_type() const override { return output_type_; }

  absl::Span<const ActivationType> input_types() const override {
    return absl::MakeConstSpan(&input_type_, 1);
  }
  absl::Span<const ActivationType> output_types() const override {
    return absl::MakeConstSpan(&output_type_signature_, 1);
  }

  int vocab_size() const { return vocab_size_; }
  int padded_vocab_size() const { return padded_vocab_size_; }
  int embedding_dim() const { return embedding_dim_; }
  int sequence_length() const { return sequence_length_; }
  const HostBuffer& weight() const { return weight_; }

 private:
  absl::StatusOr<ReferenceFwdResult> fwd_impl(
      absl::Span<const HostBuffer> inputs) const override;
  absl::StatusOr<HostBufferVec> bwd_impl(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceBackwardState state) override;

  EmbeddingLookupLayerReference(int vocab_size, int padded_vocab_size,
                                int embedding_dim, DataType data_type,
                                HostBuffer weight, HostBuffer gradient,
                                int sequence_length)
      : vocab_size_(vocab_size),
        padded_vocab_size_(padded_vocab_size),
        embedding_dim_(embedding_dim),
        sequence_length_(sequence_length),
        output_type_(data_type),
        weight_(std::move(weight)),
        gradient_(std::move(gradient)) {}

  friend class LanguageModelingHeadLayerReference;
  int vocab_size_;
  int padded_vocab_size_;
  int embedding_dim_;
  int sequence_length_;
  DataType output_type_;
  HostBuffer weight_;
  HostBuffer gradient_;
  // Shapes retain the sequence axis; the batch sentinel only matches itself.
  const ActivationType input_type_{
      DataType::INT32, {ActivationType::kBatchDimension, sequence_length_}};
  const ActivationType output_type_signature_{
      ActivationDataType(output_type_),
      {ActivationType::kBatchDimension, sequence_length_, embedding_dim_}};
};

// Obvious dense projection using the reference embedding's transposed table.
class LanguageModelingHeadLayerReference final : public LayerReference {
 public:
  absl::string_view name() const override {
    return "LanguageModelingHeadLayerReference";
  }

  static absl::StatusOr<std::unique_ptr<LanguageModelingHeadLayerReference>>
  Create(EmbeddingLookupLayerReference* embedding);

  absl::Span<HostBuffer> weights() override { return embedding_->weights(); }
  absl::Span<HostBuffer> gradients() override {
    return embedding_->gradients();
  }
  DataType output_type() const override { return embedding_->output_type(); }
  absl::Span<const ActivationType> input_types() const override {
    return embedding_->output_types();
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

  explicit LanguageModelingHeadLayerReference(
      EmbeddingLookupLayerReference* embedding)
      : embedding_(embedding) {}
  EmbeddingLookupLayerReference* embedding_;
  const ActivationType output_type_signature_{
      DataType::FP32,
      {ActivationType::kBatchDimension, embedding_->sequence_length(),
       embedding_->padded_vocab_size()}};
};

// Scalar learned-position addition and gradient accumulation reference.
class PositionEmbeddingLayerReference final : public LayerReference {
 public:
  absl::string_view name() const override {
    return "PositionEmbeddingLayerReference";
  }

  static absl::StatusOr<std::unique_ptr<PositionEmbeddingLayerReference>>
  Create(int context_length, int embedding_dim, DataType data_type);
  absl::Status InitializeNormal(float standard_deviation, uint64_t seed);

  absl::Span<HostBuffer> weights() override {
    return absl::MakeSpan(&weight_, 1);
  }
  absl::Span<HostBuffer> gradients() override {
    return absl::MakeSpan(&gradient_, 1);
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
  // Shapes retain the sequence axis; the batch sentinel only matches itself.
  const ActivationType input_type_{
      ActivationDataType(output_type_),
      {ActivationType::kBatchDimension, context_length_, embedding_dim_}};
  const ActivationType output_type_signature_{
      ActivationDataType(output_type_),
      {ActivationType::kBatchDimension, context_length_, embedding_dim_}};
};

}  // namespace pluto::llm
