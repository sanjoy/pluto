#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/reference_internal.h"
#include "src/llm/token_order.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace ri = reference_internal;
namespace {

absl::Status InitializeBufferNormal(HostBuffer* buffer,
                                    float standard_deviation, uint64_t seed) {
  if (!(standard_deviation > 0.0f)) {
    return absl::InvalidArgumentError(
        "initialization standard deviation must be positive");
  }
  std::mt19937_64 random(seed);
  std::normal_distribution<float> distribution(0.0f, standard_deviation);
  auto* values = static_cast<float*>(buffer->data());
  const size_t elements = buffer->size_bytes() / sizeof(float);
  for (size_t index = 0; index < elements; ++index)
    values[index] = distribution(random);
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<EmbeddingLookupLayerReference>>
EmbeddingLookupLayerReference::Create(int vocab_size, int embedding_dim,
                                      DataType data_type, int sequence_length,
                                      bool pad_vocabulary) {
  RETURN_IF_ERROR(ri::ValidateComputeType(data_type));
  if (sequence_length <= 0)
    return absl::InvalidArgumentError("sequence_length must be positive");
  if (vocab_size <= 0)
    return absl::InvalidArgumentError("vocab_size must be positive");
  RETURN_IF_ERROR(ri::ValidatePositiveExtent(embedding_dim, "embedding_dim"));
  const int64_t padded_extent = (int64_t{vocab_size} + 15) / 16 * 16;
  if (padded_extent > std::numeric_limits<int>::max())
    return absl::InvalidArgumentError("padded vocabulary exceeds int range");
  const int padded_vocab_size = static_cast<int>(padded_extent);
  const int stored_vocab_size = pad_vocabulary ? padded_vocab_size : vocab_size;
  if (int64_t{stored_vocab_size} * embedding_dim >
      std::numeric_limits<int>::max())
    return absl::InvalidArgumentError(
        "embedding exceeds the backend's 32-bit element-count limit");
  const size_t elements =
      static_cast<size_t>(stored_vocab_size) * embedding_dim;
  ASSIGN_OR_RETURN(auto weight, ri::AllocateFloats(elements, true));
  ASSIGN_OR_RETURN(auto gradient, ri::AllocateFloats(elements, true));
  return absl::WrapUnique(new EmbeddingLookupLayerReference(
      vocab_size, padded_vocab_size, stored_vocab_size, embedding_dim,
      data_type, std::move(weight), std::move(gradient), sequence_length));
}

absl::Status EmbeddingLookupLayerReference::InitializeIdentity(float scale) {
  auto* table = static_cast<float*>(weight_.data());
  std::fill(table,
            table + static_cast<size_t>(stored_vocab_size_) * embedding_dim_,
            0.0f);
  for (int index = 0; index < std::min(vocab_size_, embedding_dim_); ++index)
    table[static_cast<size_t>(index) * embedding_dim_ + index] = scale;
  return absl::OkStatus();
}

absl::Status EmbeddingLookupLayerReference::InitializeNormal(
    float standard_deviation, uint64_t seed, bool identical_rows) {
  RETURN_IF_ERROR(InitializeBufferNormal(&weight_, standard_deviation, seed));
  if (identical_rows) {
    auto* values = static_cast<float*>(weight_.data());
    for (int row = 1; row < stored_vocab_size_; ++row)
      std::copy_n(values, embedding_dim_,
                  values + static_cast<size_t>(row) * embedding_dim_);
  }
  return absl::OkStatus();
}

absl::StatusOr<ReferenceFwdResult> EmbeddingLookupLayerReference::fwd_impl(
    absl::Span<const HostBuffer> inputs) const {
  ReferenceBackwardState state;
  if (inputs.size() != 1) {
    return absl::InvalidArgumentError(
        "EmbeddingLookupLayerReference fwd expects token IDs");
  }
  ASSIGN_OR_RETURN(int rows,
                   ri::ElementCount(inputs[0], sizeof(int), "embedding tokens"));
  ASSIGN_OR_RETURN(auto output, ri::AllocateActivation(
                                    static_cast<size_t>(rows) * embedding_dim_,
                                    output_type_));
  const auto* tokens = static_cast<const int*>(inputs[0].data());
  const auto* table = static_cast<const float*>(weight_.data());
  // A lookup really is just this nested loop. Keeping it scalar also makes
  // token bounds and the FP32-master-to-activation conversion unambiguous.
  for (int row = 0; row < rows; ++row) {
    if (tokens[row] < 0 || tokens[row] >= vocab_size_)
      return absl::InvalidArgumentError("embedding token is out of range");
    for (int column = 0; column < embedding_dim_; ++column) {
      ri::StoreActivation(
          &output, static_cast<size_t>(row) * embedding_dim_ + column,
          output_type_,
          table[static_cast<size_t>(tokens[row]) * embedding_dim_ + column]);
    }
  }
  state.intermediates = {inputs[0]};
  state.children.clear();
  return ReferenceFwdResult{{std::move(output)}, std::move(state)};
}

absl::StatusOr<HostBufferVec> EmbeddingLookupLayerReference::bwd_impl(
    absl::Span<const HostBuffer> output_gradients,
    ReferenceBackwardState state) {
  if (output_gradients.size() != 1 || state.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "EmbeddingLookupLayerReference bwd received incompatible state");
  }
  ASSIGN_OR_RETURN(int rows,
                   ri::ElementCount(state.intermediates[0], sizeof(int),
                                    "embedding saved tokens"));
  RETURN_IF_ERROR(ri::ValidateBuffer(
      output_gradients[0],
      static_cast<size_t>(rows) * embedding_dim_ * sizeof(float),
      "embedding output gradient"));
  const auto* tokens = static_cast<const int*>(state.intermediates[0].data());
  const auto* d_output = static_cast<const float*>(output_gradients[0].data());
  auto* d_table = static_cast<float*>(gradient_.data());
  // Repeated tokens add in input-row order, matching the device's sorted
  // gather.
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < embedding_dim_; ++column) {
      d_table[static_cast<size_t>(tokens[row]) * embedding_dim_ + column] +=
          d_output[static_cast<size_t>(row) * embedding_dim_ + column];
    }
  }
  return HostBufferVec{};
}

absl::StatusOr<std::unique_ptr<LanguageModelingHeadLayerReference>>
LanguageModelingHeadLayerReference::Create(
    EmbeddingLookupLayerReference* embedding,
    absl::Span<const int32_t> token_order) {
  if (embedding == nullptr) {
    return absl::InvalidArgumentError(
        "LanguageModelingHeadLayerReference requires an embedding");
  }
  RETURN_IF_ERROR(ValidateTokenOrder(embedding->vocab_size(), token_order));
  return absl::WrapUnique(new LanguageModelingHeadLayerReference(
      embedding, std::vector<int32_t>(token_order.begin(), token_order.end())));
}

absl::StatusOr<ReferenceFwdResult> LanguageModelingHeadLayerReference::fwd_impl(
    absl::Span<const HostBuffer> inputs) const {
  ReferenceBackwardState state;
  if (inputs.size() != 1) {
    return absl::InvalidArgumentError(
        "LanguageModelingHeadLayerReference fwd expects one input and saved "
        "state");
  }
  ASSIGN_OR_RETURN(
      int rows, ri::ActivationRows(inputs[0], embedding_->embedding_dim_,
                                   embedding_->output_type_, "LM-head input"));
  ASSIGN_OR_RETURN(auto logits,
                   ri::AllocateFloats(static_cast<size_t>(rows) *
                                      embedding_->padded_vocab_size_));
  const auto* table = static_cast<const float*>(embedding_->weight_.data());
  auto* output = static_cast<float*>(logits.data());

  // The tied head is a plain hidden * embedding^T matrix multiply. Both MMA
  // operands are rounded to FP16/BF16 before each product and sums stay FP32.
  for (int row = 0; row < rows; ++row) {
    for (int token = 0; token < embedding_->padded_vocab_size_; ++token) {
      // Logit padding exists even when no corresponding trainable table row
      // exists. Never read beyond an exact-size embedding allocation.
      if (token >= embedding_->vocab_size_) {
        output[static_cast<size_t>(row) * embedding_->padded_vocab_size_ +
               token] = -std::numeric_limits<float>::max();
        continue;
      }
      float sum = 0.0f;
      for (int column = 0; column < embedding_->embedding_dim_; ++column) {
        sum +=
            ri::QuantizeMmaOperand(
                ri::LoadActivation(
                    inputs[0],
                    static_cast<size_t>(row) * embedding_->embedding_dim_ +
                        column,
                    embedding_->output_type_),
                embedding_->output_type_) *
            ri::QuantizeMmaOperand(
                table[static_cast<size_t>(token) * embedding_->embedding_dim_ +
                      column],
                embedding_->output_type_);
      }
      output[static_cast<size_t>(row) * embedding_->padded_vocab_size_ +
             token] = sum;
    }
  }
  state.intermediates = {inputs[0]};
  state.children.clear();
  return ReferenceFwdResult{{std::move(logits)}, std::move(state)};
}

absl::StatusOr<HostBufferVec> LanguageModelingHeadLayerReference::bwd_impl(
    absl::Span<const HostBuffer> output_gradients,
    ReferenceBackwardState state) {
  if (output_gradients.size() != 1 || state.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "LanguageModelingHeadLayerReference bwd received incompatible state");
  }
  ASSIGN_OR_RETURN(int rows, ri::MatrixRows(output_gradients[0],
                                            embedding_->padded_vocab_size_,
                                            "LM-head output gradient"));
  RETURN_IF_ERROR(ri::ValidateBuffer(
      state.intermediates[0],
      static_cast<size_t>(rows) * embedding_->embedding_dim_ *
          ri::ActivationElementBytes(embedding_->output_type_),
      "LM-head saved input"));
  ASSIGN_OR_RETURN(auto input_gradient,
                   ri::AllocateFloats(static_cast<size_t>(rows) *
                                      embedding_->embedding_dim_));
  const auto* d_output = static_cast<const float*>(output_gradients[0].data());
  const auto* table = static_cast<const float*>(embedding_->weight_.data());
  auto* d_input = static_cast<float*>(input_gradient.data());
  auto* d_table = static_cast<float*>(embedding_->gradient_.data());

  // These are the two ordinary transpose products. The shared table gradient
  // is accumulated because embedding lookup may already have contributed.
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < embedding_->embedding_dim_; ++column) {
      float sum = 0.0f;
      // Renaming vocabulary IDs must not rename the addition order. Padding
      // is never permuted, so it retains the same physical trailing slots.
      for (int rank = 0; rank < embedding_->stored_vocab_size_; ++rank) {
        const int token =
            !token_order_.empty() && rank < embedding_->vocab_size_
                ? token_order_[rank]
                : rank;
        sum +=
            ri::QuantizeMmaOperand(d_output[static_cast<size_t>(row) *
                                                embedding_->padded_vocab_size_ +
                                            token],
                                   embedding_->output_type_) *
            ri::QuantizeMmaOperand(
                table[static_cast<size_t>(token) * embedding_->embedding_dim_ +
                      column],
                embedding_->output_type_);
      }
      d_input[static_cast<size_t>(row) * embedding_->embedding_dim_ + column] =
          sum;
    }
  }
  for (int token = 0; token < embedding_->stored_vocab_size_; ++token) {
    for (int column = 0; column < embedding_->embedding_dim_; ++column) {
      float sum = 0.0f;
      for (int row = 0; row < rows; ++row) {
        sum +=
            ri::QuantizeMmaOperand(d_output[static_cast<size_t>(row) *
                                                embedding_->padded_vocab_size_ +
                                            token],
                                   embedding_->output_type_) *
            ri::QuantizeMmaOperand(
                ri::LoadActivation(
                    state.intermediates[0],
                    static_cast<size_t>(row) * embedding_->embedding_dim_ +
                        column,
                    embedding_->output_type_),
                embedding_->output_type_);
      }
      d_table[static_cast<size_t>(token) * embedding_->embedding_dim_ +
              column] += sum;
    }
  }
  return HostBufferVec{std::move(input_gradient)};
}

absl::StatusOr<std::unique_ptr<PositionEmbeddingLayerReference>>
PositionEmbeddingLayerReference::Create(int context_length, int embedding_dim,
                                        DataType data_type) {
  RETURN_IF_ERROR(ri::ValidateComputeType(data_type));
  if (context_length <= 0)
    return absl::InvalidArgumentError("context_length must be positive");
  RETURN_IF_ERROR(ri::ValidatePositiveExtent(embedding_dim, "embedding_dim"));
  const size_t elements = static_cast<size_t>(context_length) * embedding_dim;
  ASSIGN_OR_RETURN(auto weight, ri::AllocateFloats(elements, true));
  ASSIGN_OR_RETURN(auto gradient, ri::AllocateFloats(elements, true));
  return absl::WrapUnique(new PositionEmbeddingLayerReference(
      context_length, embedding_dim, data_type, std::move(weight),
      std::move(gradient)));
}

absl::Status PositionEmbeddingLayerReference::InitializeNormal(
    float standard_deviation, uint64_t seed) {
  return InitializeBufferNormal(&weight_, standard_deviation, seed);
}

absl::StatusOr<ReferenceFwdResult> PositionEmbeddingLayerReference::fwd_impl(
    absl::Span<const HostBuffer> inputs) const {
  ReferenceBackwardState state;
  if (inputs.size() != 1) {
    return absl::InvalidArgumentError(
        "PositionEmbeddingLayerReference fwd expects one input and saved "
        "state");
  }
  ASSIGN_OR_RETURN(int rows,
                   ri::ActivationRows(inputs[0], embedding_dim_, output_type_,
                                      "position-embedding input"));
  ASSIGN_OR_RETURN(auto output, ri::AllocateActivation(
                                    static_cast<size_t>(rows) * embedding_dim_,
                                    output_type_));
  const auto* positions = static_cast<const float*>(weight_.data());
  // Position rows repeat independently for each packed sequence.
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < embedding_dim_; ++column) {
      const size_t index = static_cast<size_t>(row) * embedding_dim_ + column;
      ri::StoreActivation(
          &output, index, output_type_,
          ri::LoadActivation(inputs[0], index, output_type_) +
              positions[static_cast<size_t>(row % context_length_) *
                            embedding_dim_ +
                        column]);
    }
  }
  state.intermediates.clear();
  state.children.clear();
  return ReferenceFwdResult{{std::move(output)}, std::move(state)};
}

absl::StatusOr<HostBufferVec> PositionEmbeddingLayerReference::bwd_impl(
    absl::Span<const HostBuffer> output_gradients,
    ReferenceBackwardState state) {
  if (output_gradients.size() != 1 || !state.intermediates.empty()) {
    return absl::InvalidArgumentError(
        "PositionEmbeddingLayerReference bwd received incompatible state");
  }
  ASSIGN_OR_RETURN(int rows,
                   ri::MatrixRows(output_gradients[0], embedding_dim_,
                                  "position-embedding output gradient"));
  const auto* d_output = static_cast<const float*>(output_gradients[0].data());
  auto* d_position = static_cast<float*>(gradient_.data());
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < embedding_dim_; ++column) {
      d_position[static_cast<size_t>(row % context_length_) * embedding_dim_ +
                 column] +=
          d_output[static_cast<size_t>(row) * embedding_dim_ + column];
    }
  }
  return HostBufferVec{output_gradients[0]};
}

}  // namespace pluto::llm
