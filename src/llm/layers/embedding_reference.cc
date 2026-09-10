#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/util/status_macros.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/reference_internal.h"

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
  for (size_t index = 0; index < elements; ++index) {
    values[index] = distribution(random);
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<EmbeddingLookupLayerReference>>
EmbeddingLookupLayerReference::Create(int vocab_size, int embedding_dim,
                                      DataType data_type) {
  RETURN_IF_ERROR(ri::ValidateComputeType(data_type));
  if (vocab_size <= 0) {
    return absl::InvalidArgumentError("vocab_size must be positive");
  }
  RETURN_IF_ERROR(ri::ValidateTiledExtent(embedding_dim, "embedding_dim"));
  const int padded_vocab_size = ri::RoundUpToTile(vocab_size);
  const size_t elements =
      static_cast<size_t>(padded_vocab_size) * embedding_dim;
  ASSIGN_OR_RETURN(auto weight, ri::AllocateFloats(elements, true));
  ASSIGN_OR_RETURN(auto gradient, ri::AllocateFloats(elements, true));
  return std::unique_ptr<EmbeddingLookupLayerReference>(
      new EmbeddingLookupLayerReference(
          vocab_size, padded_vocab_size, embedding_dim, data_type,
          std::move(weight), std::move(gradient)));
}

absl::Status EmbeddingLookupLayerReference::InitializeIdentity(float scale) {
  auto* table = static_cast<float*>(weight_.data());
  std::fill(table,
            table + static_cast<size_t>(padded_vocab_size_) * embedding_dim_,
            0.0f);
  for (int index = 0; index < std::min(vocab_size_, embedding_dim_); ++index) {
    table[static_cast<size_t>(index) * embedding_dim_ + index] = scale;
  }
  return absl::OkStatus();
}

absl::Status EmbeddingLookupLayerReference::InitializeNormal(
    float standard_deviation, uint64_t seed) {
  return InitializeBufferNormal(&weight_, standard_deviation, seed);
}

absl::StatusOr<HostBuffer> EmbeddingLookupLayerReference::fwd(
    absl::Span<const HostBuffer> inputs, ReferenceTape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "EmbeddingLookupLayerReference fwd expects token IDs and a tape");
  }
  ASSIGN_OR_RETURN(
      int rows, ri::ElementCount(inputs[0], sizeof(int), "embedding tokens"));
  ASSIGN_OR_RETURN(auto output, ri::AllocateActivation(
                                    static_cast<size_t>(rows) * embedding_dim_,
                                    output_type_));
  const auto* tokens = static_cast<const int*>(inputs[0].data());
  const auto* table = static_cast<const float*>(weight_.data());
  // A lookup really is just this nested loop. Keeping it scalar also makes
  // token bounds and the FP32-master-to-activation conversion unambiguous.
  for (int row = 0; row < rows; ++row) {
    if (tokens[row] < 0 || tokens[row] >= vocab_size_) {
      return absl::InvalidArgumentError("embedding token is out of range");
    }
    for (int column = 0; column < embedding_dim_; ++column) {
      ri::StoreActivation(
          &output, static_cast<size_t>(row) * embedding_dim_ + column,
          output_type_,
          table[static_cast<size_t>(tokens[row]) * embedding_dim_ + column]);
    }
  }
  tape->intermediates = {inputs[0]};
  tape->children.clear();
  return output;
}

absl::StatusOr<HostBufferVec> EmbeddingLookupLayerReference::bwd(
    absl::Span<const HostBuffer> output_gradients, ReferenceTape tape) {
  if (output_gradients.size() != 1 || tape.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "EmbeddingLookupLayerReference bwd received incompatible state");
  }
  ASSIGN_OR_RETURN(int rows,
                   ri::ElementCount(tape.intermediates[0], sizeof(int),
                                    "embedding saved tokens"));
  RETURN_IF_ERROR(ri::ValidateBuffer(
      output_gradients[0],
      static_cast<size_t>(rows) * embedding_dim_ * sizeof(float),
      "embedding output gradient"));
  const auto* tokens = static_cast<const int*>(tape.intermediates[0].data());
  const auto* d_output = static_cast<const float*>(output_gradients[0].data());
  auto* d_table = static_cast<float*>(gradient_.data());
  // Repeated tokens add in input-row order, matching the device's sorted gather.
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
    EmbeddingLookupLayerReference* embedding) {
  if (embedding == nullptr) {
    return absl::InvalidArgumentError(
        "LanguageModelingHeadLayerReference requires an embedding");
  }
  return std::unique_ptr<LanguageModelingHeadLayerReference>(
      new LanguageModelingHeadLayerReference(embedding));
}

absl::StatusOr<HostBuffer> LanguageModelingHeadLayerReference::fwd(
    absl::Span<const HostBuffer> inputs, ReferenceTape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "LanguageModelingHeadLayerReference fwd expects one input and a tape");
  }
  ASSIGN_OR_RETURN(
      int rows, ri::ActivationRows(inputs[0], embedding_->embedding_dim_,
                                   embedding_->output_type_, "LM-head input"));
  RETURN_IF_ERROR(ri::ValidateTiledExtent(rows, "LM-head rows"));
  ASSIGN_OR_RETURN(auto logits,
                   ri::AllocateFloats(static_cast<size_t>(rows) *
                                      embedding_->padded_vocab_size_));
  const auto* table = static_cast<const float*>(embedding_->weight_.data());
  auto* output = static_cast<float*>(logits.data());

  // The tied head is a plain hidden * embedding^T matrix multiply. Both MMA
  // operands are rounded to FP16/BF16 before each product and sums stay FP32.
  for (int row = 0; row < rows; ++row) {
    for (int token = 0; token < embedding_->padded_vocab_size_; ++token) {
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
             token] = token < embedding_->vocab_size_
                          ? sum
                          : -std::numeric_limits<float>::max();
    }
  }
  tape->intermediates = {inputs[0]};
  tape->children.clear();
  return logits;
}

absl::StatusOr<HostBufferVec> LanguageModelingHeadLayerReference::bwd(
    absl::Span<const HostBuffer> output_gradients, ReferenceTape tape) {
  if (output_gradients.size() != 1 || tape.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "LanguageModelingHeadLayerReference bwd received incompatible state");
  }
  ASSIGN_OR_RETURN(int rows, ri::MatrixRows(output_gradients[0],
                                            embedding_->padded_vocab_size_,
                                            "LM-head output gradient"));
  RETURN_IF_ERROR(ri::ValidateBuffer(
      tape.intermediates[0],
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
      for (int token = 0; token < embedding_->padded_vocab_size_; ++token) {
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
  for (int token = 0; token < embedding_->padded_vocab_size_; ++token) {
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
                    tape.intermediates[0],
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
  if (context_length <= 0) {
    return absl::InvalidArgumentError("context_length must be positive");
  }
  RETURN_IF_ERROR(ri::ValidateTiledExtent(embedding_dim, "embedding_dim"));
  const size_t elements = static_cast<size_t>(context_length) * embedding_dim;
  ASSIGN_OR_RETURN(auto weight, ri::AllocateFloats(elements, true));
  ASSIGN_OR_RETURN(auto gradient, ri::AllocateFloats(elements, true));
  return std::unique_ptr<PositionEmbeddingLayerReference>(
      new PositionEmbeddingLayerReference(context_length, embedding_dim,
                                          data_type, std::move(weight),
                                          std::move(gradient)));
}

absl::Status PositionEmbeddingLayerReference::InitializeNormal(
    float standard_deviation, uint64_t seed) {
  return InitializeBufferNormal(&weight_, standard_deviation, seed);
}

absl::StatusOr<HostBuffer> PositionEmbeddingLayerReference::fwd(
    absl::Span<const HostBuffer> inputs, ReferenceTape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "PositionEmbeddingLayerReference fwd expects one input and a tape");
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
  tape->intermediates.clear();
  tape->children.clear();
  return output;
}

absl::StatusOr<HostBufferVec> PositionEmbeddingLayerReference::bwd(
    absl::Span<const HostBuffer> output_gradients, ReferenceTape tape) {
  if (output_gradients.size() != 1 || !tape.intermediates.empty()) {
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
