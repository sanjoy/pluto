#include "src/llm/layers/cross_entropy_loss.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/common/status_macros.h"
#include "src/llm/layers/reference_internal.h"

namespace pluto::llm {
namespace ri = reference_internal;

absl::StatusOr<std::unique_ptr<CrossEntropyLossLayerReference>>
CrossEntropyLossLayerReference::Create(int vocabulary_size,
                                       DataType data_type) {
  RETURN_IF_ERROR(ri::ValidateComputeType(data_type));
  if (vocabulary_size <= 0) {
    return absl::InvalidArgumentError("vocabulary_size must be positive");
  }
  return std::unique_ptr<CrossEntropyLossLayerReference>(
      new CrossEntropyLossLayerReference(
          vocabulary_size, ri::RoundUpToTile(vocabulary_size), data_type));
}

absl::StatusOr<HostBuffer> CrossEntropyLossLayerReference::fwd(
    absl::Span<const HostBuffer> inputs, ReferenceTape* tape) {
  if (inputs.size() != 2 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "CrossEntropyLossLayerReference expects logits, targets, and a tape");
  }
  ASSIGN_OR_RETURN(int rows,
                   ri::MatrixRows(inputs[0], padded_vocab_size_,
                                  "cross-entropy logits"));
  RETURN_IF_ERROR(ri::ValidateBuffer(
      inputs[1], static_cast<size_t>(rows) * sizeof(int),
      "cross-entropy targets"));
  ASSIGN_OR_RETURN(auto losses, ri::AllocateFloats(rows));
  const auto* logits = static_cast<const float*>(inputs[0].data());
  const auto* targets = static_cast<const int*>(inputs[1].data());
  auto* loss = static_cast<float*>(losses.data());

  // The two-pass maximum and exponential sum is the elementary stable
  // log-sum-exp formula. Padded lanes are included, just as in the CUDA
  // kernel; the LM head guarantees that they contain negative infinity.
  for (int row = 0; row < rows; ++row) {
    if (targets[row] < 0 || targets[row] >= vocab_size_) {
      return absl::InvalidArgumentError("target token is outside vocabulary");
    }
    const float* row_logits = logits +
                              static_cast<size_t>(row) * padded_vocab_size_;
    float maximum = -std::numeric_limits<float>::infinity();
    for (int token = 0; token < padded_vocab_size_; ++token) {
      maximum = std::max(maximum, row_logits[token]);
    }
    float denominator = 0.0f;
    for (int token = 0; token < padded_vocab_size_; ++token) {
      denominator += std::exp(row_logits[token] - maximum);
    }
    loss[row] = std::log(denominator) + maximum - row_logits[targets[row]];
  }
  tape->intermediates = {inputs[0], inputs[1]};
  tape->children.clear();
  return losses;
}

absl::StatusOr<HostBufferVec> CrossEntropyLossLayerReference::bwd(
    absl::Span<const HostBuffer> output_gradients, ReferenceTape tape) {
  if (!output_gradients.empty() || tape.intermediates.size() != 2) {
    return absl::InvalidArgumentError(
        "terminal cross-entropy reference expects no upstream gradient");
  }
  ASSIGN_OR_RETURN(int rows,
                   ri::MatrixRows(tape.intermediates[0], padded_vocab_size_,
                                  "cross-entropy saved logits"));
  RETURN_IF_ERROR(ri::ValidateBuffer(
      tape.intermediates[1], static_cast<size_t>(rows) * sizeof(int),
      "cross-entropy saved targets"));
  ASSIGN_OR_RETURN(auto gradient,
                   ri::AllocateFloats(static_cast<size_t>(rows) *
                                      padded_vocab_size_));
  const auto* logits =
      static_cast<const float*>(tape.intermediates[0].data());
  const auto* targets =
      static_cast<const int*>(tape.intermediates[1].data());
  auto* d_logits = static_cast<float*>(gradient.data());
  for (int row = 0; row < rows; ++row) {
    const float* row_logits = logits +
                              static_cast<size_t>(row) * padded_vocab_size_;
    float maximum = -std::numeric_limits<float>::infinity();
    for (int token = 0; token < padded_vocab_size_; ++token) {
      maximum = std::max(maximum, row_logits[token]);
    }
    float denominator = 0.0f;
    for (int token = 0; token < padded_vocab_size_; ++token) {
      denominator += std::exp(row_logits[token] - maximum);
    }
    for (int token = 0; token < padded_vocab_size_; ++token) {
      const float one_hot = token == targets[row] ? 1.0f : 0.0f;
      d_logits[static_cast<size_t>(row) * padded_vocab_size_ + token] =
          (std::exp(row_logits[token] - maximum) / denominator - one_hot) /
          static_cast<float>(rows);
    }
  }
  return HostBufferVec{std::move(gradient)};
}

}  // namespace pluto::llm
