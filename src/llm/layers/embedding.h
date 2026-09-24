#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// Token embedding with an FP32 master table and FP32 accumulated gradient.
// By default the table includes padding rows for checkpoint compatibility.
// With pad_vocabulary=false it stores exactly vocab_size() trainable rows;
// masked compute tiles and padded LM-head outputs add no parameters.
class EmbeddingLookupLayer final : public Layer {
 public:
  absl::string_view name() const override { return "EmbeddingLookupLayer"; }

  // sequence_length is tokens per sample, not batch size. The default treats
  // each token as a separate one-token sample; larger values preserve a
  // [-2, sequence_length] input and [-2, sequence_length, embedding_dim]
  // output. pad_vocabulary affects table storage only: the tied head always
  // pads its logits to 16 columns for cross-entropy compatibility.
  static absl::StatusOr<std::unique_ptr<EmbeddingLookupLayer>> Create(
      cuda::Executor& executor, int vocab_size, int embedding_dim,
      DataType data_type, int sequence_length = 1, bool pad_vocabulary = true);

  absl::Status InitializeIdentity(float scale = 1.0f);
  // identical_rows copies the first normally initialized row to every stored
  // row, including padding. This changes initialization only, not weight tying.
  absl::Status InitializeNormal(float standard_deviation, uint64_t seed,
                                bool identical_rows = false);

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

  int vocab_size() const { return vocab_size_; }
  // Logit row stride; it need not equal the number of stored table rows.
  int padded_vocab_size() const { return padded_vocab_size_; }
  int stored_vocab_size() const { return stored_vocab_size_; }
  int embedding_dim() const { return embedding_dim_; }
  int sequence_length() const { return sequence_length_; }
  const Buffer& weight() const { return weight_; }

 private:
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state, LayerHooks*) override;

  EmbeddingLookupLayer(cuda::Executor& executor, int vocab_size,
                       int padded_vocab_size, int stored_vocab_size,
                       int embedding_dim, DataType data_type, Buffer weight,
                       Buffer gradient, int sequence_length);

  friend class LanguageModelingHeadLayer;

  int vocab_size_;
  int padded_vocab_size_;
  int stored_vocab_size_;
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
// lanes beyond vocab_size are masked and therefore receive zero probability
// and loss gradient. They need not have corresponding stored embedding rows.
// The embedding owns the shared master weight and gradient.
class LanguageModelingHeadLayer final : public Layer {
 public:
  absl::string_view name() const override {
    return "LanguageModelingHeadLayer";
  }

  // token_order[canonical_rank] gives the physical vocabulary ID to sum at
  // that rank when propagating gradients into the hidden state. This keeps
  // the floating-point reduction order stable after vocabulary renaming.
  // An empty order means identity; a supplied order is copied at creation.
  // Logits and shared embedding weights remain in physical vocabulary order.
  static absl::StatusOr<std::unique_ptr<LanguageModelingHeadLayer>> Create(
      EmbeddingLookupLayer* embedding,
      absl::Span<const int32_t> token_order = {});

  absl::Span<Buffer> weights() override { return embedding_->weights(); }
  absl::Span<Buffer> gradients() override { return embedding_->gradients(); }
  DataType output_type() const override { return embedding_->output_type(); }
  absl::Span<const ActivationType> input_types() const override {
    return embedding_->output_types();
  }
  absl::Span<const ActivationType> output_types() const override {
    return absl::MakeConstSpan(&output_type_signature_, 1);
  }

 private:
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state, LayerHooks*) override;

  LanguageModelingHeadLayer(EmbeddingLookupLayer* embedding,
                            std::optional<Buffer> token_order)
      : embedding_(embedding), token_order_(std::move(token_order)) {}

  EmbeddingLookupLayer* embedding_;
  // Absent for the original contiguous-load identity fast path.
  std::optional<Buffer> token_order_;
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
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state, LayerHooks*) override;

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
  // output. As on the GPU, pad_vocabulary controls trainable table rows, not
  // the tied head's always-padded logit stride.
  static absl::StatusOr<std::unique_ptr<EmbeddingLookupLayerReference>> Create(
      int vocab_size, int embedding_dim, DataType data_type,
      int sequence_length = 1, bool pad_vocabulary = true);

  absl::Status InitializeIdentity(float scale = 1.0f);
  absl::Status InitializeNormal(float standard_deviation, uint64_t seed,
                                bool identical_rows = false);

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
  int stored_vocab_size() const { return stored_vocab_size_; }
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
                                int stored_vocab_size, int embedding_dim,
                                DataType data_type, HostBuffer weight,
                                HostBuffer gradient, int sequence_length)
      : vocab_size_(vocab_size),
        padded_vocab_size_(padded_vocab_size),
        stored_vocab_size_(stored_vocab_size),
        embedding_dim_(embedding_dim),
        sequence_length_(sequence_length),
        output_type_(data_type),
        weight_(std::move(weight)),
        gradient_(std::move(gradient)) {}

  friend class LanguageModelingHeadLayerReference;
  int vocab_size_;
  int padded_vocab_size_;
  int stored_vocab_size_;
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
  Create(EmbeddingLookupLayerReference* embedding,
         absl::Span<const int32_t> token_order = {});

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

  LanguageModelingHeadLayerReference(EmbeddingLookupLayerReference* embedding,
                                     std::vector<int32_t> token_order)
      : embedding_(embedding), token_order_(std::move(token_order)) {}
  EmbeddingLookupLayerReference* embedding_;
  // Same canonical-rank-to-physical-ID semantics as the GPU head.
  std::vector<int32_t> token_order_;
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
