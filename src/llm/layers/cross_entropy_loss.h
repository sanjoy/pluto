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

// Computes one cross-entropy value per token in each batch sample. fwd() takes
// two inputs: [logits(float), target_token(int32)]. bwd() takes no upstream
// gradient because this is a terminal loss and returns the mean-loss gradient
// for the logits. Keeping per-token losses makes diagnostics and tests more
// useful. sequence_length fixes the token dimension; only the leading batch
// dimension is symbolic. Logits and losses are FP32 even with BF16 model
// activations.
class CrossEntropyLossLayer final : public Layer {
 public:
  absl::string_view name() const override { return "CrossEntropyLossLayer"; }

  static absl::StatusOr<std::unique_ptr<CrossEntropyLossLayer>> Create(
      cuda::Executor& executor, int vocabulary_size, DataType data_type,
      int sequence_length = 1);

  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return output_type_; }
  absl::Span<const ActivationType> input_types() const override {
    return input_types_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return output_types_;
  }

  int vocab_size() const { return vocab_size_; }
  int padded_vocab_size() const { return padded_vocab_size_; }

 private:
  absl::StatusOr<FwdResult> fwd_impl(
      cuda::Executor& executor, absl::Span<const Buffer> inputs) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state) override;

  CrossEntropyLossLayer(cuda::Executor& executor, int vocab_size,
                        int padded_vocab_size, DataType data_type,
                        int sequence_length)
      : vocab_size_(vocab_size),
        padded_vocab_size_(padded_vocab_size),
        sequence_length_(sequence_length),
        output_type_(data_type),
        executor_(executor) {}

  int vocab_size_;
  int padded_vocab_size_;
  int sequence_length_;
  DataType output_type_;
  cuda::Executor& executor_;
  const ActivationType input_types_[2] = {
      {DataType::FP32,
       {ActivationType::kBatchDimension, sequence_length_, padded_vocab_size_}},
      {DataType::INT32, {ActivationType::kBatchDimension, sequence_length_}}};
  const ActivationType output_types_[1] = {
      {DataType::FP32, {ActivationType::kBatchDimension, sequence_length_}}};
};

// Scalar log-sum-exp reference for the terminal cross-entropy operation.
class CrossEntropyLossLayerReference final : public LayerReference {
 public:
  absl::string_view name() const override {
    return "CrossEntropyLossLayerReference";
  }

  static absl::StatusOr<std::unique_ptr<CrossEntropyLossLayerReference>> Create(
      int vocabulary_size, DataType data_type, int sequence_length = 1);

  absl::Span<HostBuffer> weights() override { return {}; }
  DataType output_type() const override { return output_type_; }
  absl::Span<const ActivationType> input_types() const override {
    return input_types_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return output_types_;
  }

  int vocab_size() const { return vocab_size_; }
  int padded_vocab_size() const { return padded_vocab_size_; }

 private:
  absl::StatusOr<ReferenceFwdResult> fwd_impl(
      absl::Span<const HostBuffer> inputs) const override;
  absl::StatusOr<HostBufferVec> bwd_impl(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceBackwardState state) override;

  CrossEntropyLossLayerReference(int vocab_size, int padded_vocab_size,
                                 DataType data_type, int sequence_length)
      : vocab_size_(vocab_size),
        padded_vocab_size_(padded_vocab_size),
        sequence_length_(sequence_length),
        output_type_(data_type) {}

  int vocab_size_;
  int padded_vocab_size_;
  int sequence_length_;
  DataType output_type_;
  const ActivationType input_types_[2] = {
      {DataType::FP32,
       {ActivationType::kBatchDimension, sequence_length_, padded_vocab_size_}},
      {DataType::INT32, {ActivationType::kBatchDimension, sequence_length_}}};
  const ActivationType output_types_[1] = {
      {DataType::FP32, {ActivationType::kBatchDimension, sequence_length_}}};
};

}  // namespace pluto::llm
