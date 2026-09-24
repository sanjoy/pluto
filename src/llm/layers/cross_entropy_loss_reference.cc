#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/layers/reference_internal.h"
#include "src/llm/token_order.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace ri = reference_internal;

absl::StatusOr<std::unique_ptr<CrossEntropyLossLayerReference>>
CrossEntropyLossLayerReference::Create(int vocabulary_size, DataType data_type,
                                       int sequence_length,
                                       absl::Span<const int32_t> token_order) {
  RETURN_IF_ERROR(ri::ValidateComputeType(data_type));
  if (sequence_length <= 0)
    return absl::InvalidArgumentError("sequence_length must be positive");
  if (vocabulary_size <= 0)
    return absl::InvalidArgumentError("vocabulary_size must be positive");
  RETURN_IF_ERROR(ValidateTokenOrder(vocabulary_size, token_order));
  return absl::WrapUnique(new CrossEntropyLossLayerReference(
      vocabulary_size, ri::RoundUpToTile(vocabulary_size), data_type,
      sequence_length,
      std::vector<int32_t>(token_order.begin(), token_order.end())));
}

absl::StatusOr<ReferenceFwdResult> CrossEntropyLossLayerReference::fwd_impl(
    absl::Span<const HostBuffer> inputs) const {
  ReferenceBackwardState state;
  if (inputs.size() != 2) {
    return absl::InvalidArgumentError(
        "CrossEntropyLossLayerReference expects logits, targets, and saved "
        "state");
  }
  ASSIGN_OR_RETURN(int rows, ri::MatrixRows(inputs[0], padded_vocab_size_,
                                            "cross-entropy logits"));
  RETURN_IF_ERROR(ri::ValidateBuffer(inputs[1],
                                     static_cast<size_t>(rows) * sizeof(int),
                                     "cross-entropy targets"));
  ASSIGN_OR_RETURN(auto losses, ri::AllocateFloats(rows));
  const auto* logits = static_cast<const float*>(inputs[0].data());
  const auto* targets = static_cast<const int*>(inputs[1].data());
  auto* loss = static_cast<float*>(losses.data());

  // The two-pass maximum and exponential sum is the elementary stable
  // log-sum-exp formula. Padded lanes are included, just as in the CUDA
  // kernel; the LM head guarantees that they contain negative infinity.
  // Iterate by canonical rank and translate only the load addresses. This
  // keeps the arithmetic order fixed when physical vocabulary IDs change.
  for (int row = 0; row < rows; ++row) {
    // Prompt/padding rows are absent from the objective, not merely multiplied
    // by zero after a softmax. In particular their logits may safely be NaN.
    if (targets[row] == CrossEntropyLossLayer::kIgnoredTarget) {
      loss[row] = 0.0f;
      continue;
    }
    if (targets[row] < 0 || targets[row] >= vocab_size_)
      return absl::InvalidArgumentError("target token is outside vocabulary");
    const float* row_logits =
        logits + static_cast<size_t>(row) * padded_vocab_size_;
    float maximum = -std::numeric_limits<float>::infinity();
    for (int rank = 0; rank < padded_vocab_size_; ++rank)
      maximum = std::max(maximum, row_logits[PhysicalToken(rank)]);
    float denominator = 0.0f;
    for (int rank = 0; rank < padded_vocab_size_; ++rank)
      denominator += std::exp(row_logits[PhysicalToken(rank)] - maximum);
    loss[row] = std::log(denominator) + maximum - row_logits[targets[row]];
  }
  state.intermediates = {inputs[0], inputs[1]};
  state.children.clear();
  return ReferenceFwdResult{{std::move(losses)}, std::move(state)};
}

absl::StatusOr<HostBufferVec> CrossEntropyLossLayerReference::bwd_impl(
    absl::Span<const HostBuffer> output_gradients,
    ReferenceBackwardState state) {
  if (!output_gradients.empty() || state.intermediates.size() != 2) {
    return absl::InvalidArgumentError(
        "terminal cross-entropy reference expects no upstream gradient");
  }
  ASSIGN_OR_RETURN(int rows,
                   ri::MatrixRows(state.intermediates[0], padded_vocab_size_,
                                  "cross-entropy saved logits"));
  RETURN_IF_ERROR(ri::ValidateBuffer(state.intermediates[1],
                                     static_cast<size_t>(rows) * sizeof(int),
                                     "cross-entropy saved targets"));
  ASSIGN_OR_RETURN(auto gradient, ri::AllocateFloats(static_cast<size_t>(rows) *
                                                     padded_vocab_size_));
  const auto* logits = static_cast<const float*>(state.intermediates[0].data());
  const auto* targets = static_cast<const int*>(state.intermediates[1].data());
  auto* d_logits = static_cast<float*>(gradient.data());
  // Average over actual supervised targets, not sequence storage. Adding any
  // number of padding rows must leave every real token's gradient unchanged.
  int valid_rows = 0;
  for (int row = 0; row < rows; ++row)
    if (targets[row] != CrossEntropyLossLayer::kIgnoredTarget)
      ++valid_rows;
  for (int row = 0; row < rows; ++row) {
    if (targets[row] == CrossEntropyLossLayer::kIgnoredTarget) {
      std::fill_n(d_logits + static_cast<size_t>(row) * padded_vocab_size_,
                  padded_vocab_size_, 0.0f);
      continue;
    }
    const float* row_logits =
        logits + static_cast<size_t>(row) * padded_vocab_size_;
    float maximum = -std::numeric_limits<float>::infinity();
    for (int rank = 0; rank < padded_vocab_size_; ++rank)
      maximum = std::max(maximum, row_logits[PhysicalToken(rank)]);
    float denominator = 0.0f;
    for (int rank = 0; rank < padded_vocab_size_; ++rank)
      denominator += std::exp(row_logits[PhysicalToken(rank)] - maximum);
    for (int token = 0; token < padded_vocab_size_; ++token) {
      const float one_hot = token == targets[row] ? 1.0f : 0.0f;
      d_logits[static_cast<size_t>(row) * padded_vocab_size_ + token] =
          (std::exp(row_logits[token] - maximum) / denominator - one_hot) /
          static_cast<float>(valid_rows);
    }
  }
  return HostBufferVec{std::move(gradient)};
}

}  // namespace pluto::llm
