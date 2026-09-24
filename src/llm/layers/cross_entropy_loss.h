#pragma once

#include <cstddef>
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

// Computes one cross-entropy value per token in each batch sample. fwd() takes
// two inputs: [logits(float), target_token(int32)]. bwd() takes no upstream
// gradient because this is a terminal loss and returns the mean-loss gradient
// for the logits. Keeping per-token losses makes diagnostics and tests more
// useful. sequence_length fixes the token dimension; only the leading batch
// dimension is symbolic. Logits and losses are FP32 even with BF16 model
// activations.
// A target equal to kIgnoredTarget omits that row from the objective: its loss
// and logit gradient are zero, and backward averages only the remaining rows.
// This supports both prompt masking and right-padded independent sequences.
// An entirely ignored batch has zero loss and gradient. Other target IDs must
// belong to [0, vocab_size()); callers provide valid targets on the GPU.
class CrossEntropyLossLayer final : public Layer {
 public:
  static constexpr int kIgnoredTarget = -1;

  absl::string_view name() const override { return "CrossEntropyLossLayer"; }

  // token_order[rank] is the physical vocabulary column visited at that
  // canonical reduction rank. A nonempty order must permute [0, vocab_size).
  // Creation copies it; the caller need not retain the host array. Targets
  // and logit gradients still use physical token IDs. Empty means ID order.
  static absl::StatusOr<std::unique_ptr<CrossEntropyLossLayer>> Create(
      cuda::Executor& executor, int vocabulary_size, DataType data_type,
      int sequence_length = 1, absl::Span<const int32_t> token_order = {});

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
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state, LayerHooks*) override;

  CrossEntropyLossLayer(cuda::Executor& executor, int vocab_size,
                        int padded_vocab_size, DataType data_type,
                        int sequence_length, std::optional<Buffer> token_order)
      : vocab_size_(vocab_size),
        padded_vocab_size_(padded_vocab_size),
        sequence_length_(sequence_length),
        output_type_(data_type),
        executor_(executor),
        token_order_(std::move(token_order)) {}

  int vocab_size_;
  int padded_vocab_size_;
  int sequence_length_;
  DataType output_type_;
  cuda::Executor& executor_;
  // Only logical vocabulary IDs are remapped; padding keeps its old order.
  std::optional<Buffer> token_order_;
  const ActivationType input_types_[2] = {
      {DataType::FP32,
       {ActivationType::kBatchDimension, sequence_length_, padded_vocab_size_}},
      {DataType::INT32, {ActivationType::kBatchDimension, sequence_length_}}};
  const ActivationType output_types_[1] = {
      {DataType::FP32, {ActivationType::kBatchDimension, sequence_length_}}};
};

// Scalar log-sum-exp reference for the terminal cross-entropy operation,
// including CrossEntropyLossLayer::kIgnoredTarget masking and normalization.
class CrossEntropyLossLayerReference final : public LayerReference {
 public:
  absl::string_view name() const override {
    return "CrossEntropyLossLayerReference";
  }

  static absl::StatusOr<std::unique_ptr<CrossEntropyLossLayerReference>> Create(
      int vocabulary_size, DataType data_type, int sequence_length = 1,
      absl::Span<const int32_t> token_order = {});

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
                                 DataType data_type, int sequence_length,
                                 std::vector<int32_t> token_order)
      : vocab_size_(vocab_size),
        padded_vocab_size_(padded_vocab_size),
        sequence_length_(sequence_length),
        output_type_(data_type),
        token_order_(std::move(token_order)) {}

  int PhysicalToken(int rank) const {
    return token_order_.empty() || rank >= vocab_size_ ? rank
                                                       : token_order_[rank];
  }

  int vocab_size_;
  int padded_vocab_size_;
  int sequence_length_;
  DataType output_type_;
  std::vector<int32_t> token_order_;
  const ActivationType input_types_[2] = {
      {DataType::FP32,
       {ActivationType::kBatchDimension, sequence_length_, padded_vocab_size_}},
      {DataType::INT32, {ActivationType::kBatchDimension, sequence_length_}}};
  const ActivationType output_types_[1] = {
      {DataType::FP32, {ActivationType::kBatchDimension, sequence_length_}}};
};

}  // namespace pluto::llm
