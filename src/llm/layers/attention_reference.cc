#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/common/status_macros.h"
#include "src/llm/layers/attention.h"
#include "src/llm/layers/reference_internal.h"

namespace pluto::llm {
namespace ri = reference_internal;

absl::StatusOr<std::unique_ptr<AttentionLayerReference>>
AttentionLayerReference::Create(int context_length, int num_heads,
                                int embedding_dim, DataType data_type) {
  RETURN_IF_ERROR(ri::ValidateComputeType(data_type));
  if (context_length <= 0 || num_heads <= 0 || embedding_dim <= 0) {
    return absl::InvalidArgumentError(
        "attention dimensions must all be positive");
  }
  if (embedding_dim % num_heads != 0) {
    return absl::InvalidArgumentError(
        "embedding_dim must be divisible by num_heads");
  }
  RETURN_IF_ERROR(ri::ValidateTiledExtent(embedding_dim / num_heads,
                                          "attention head dimension"));
  return std::unique_ptr<AttentionLayerReference>(new AttentionLayerReference(
      context_length, num_heads, embedding_dim, data_type));
}

absl::StatusOr<HostBuffer> AttentionLayerReference::fwd(
    absl::Span<const HostBuffer> inputs, ReferenceTape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "AttentionLayerReference fwd expects packed Q/K/V and a tape");
  }
  ASSIGN_OR_RETURN(
      int rows, ri::ActivationRows(inputs[0], 3 * embedding_dim_, output_type_,
                                   "attention packed Q/K/V input"));
  if (rows % context_length_ != 0) {
    return absl::InvalidArgumentError(
        "attention rows must be divisible by context_length");
  }
  ASSIGN_OR_RETURN(auto output, ri::AllocateActivation(
                                    static_cast<size_t>(rows) * embedding_dim_,
                                    output_type_));
  const int head_dim = embedding_dim_ / num_heads_;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  std::vector<float> scores(context_length_);

  // This is textbook causal attention. For each (query, head), first compute
  // every visible Q.K score, turn that row into a stable softmax, and finally
  // form the probability-weighted V sum. Nothing is fused or tiled so this is
  // easy to audit against the equations in the FlashAttention paper.
  for (int row = 0; row < rows; ++row) {
    const int sequence_start = (row / context_length_) * context_length_;
    const int query_position = row % context_length_;
    for (int head = 0; head < num_heads_; ++head) {
      const int head_start = head * head_dim;
      float maximum = -std::numeric_limits<float>::infinity();
      for (int key_position = 0; key_position <= query_position;
           ++key_position) {
        const int key_row = sequence_start + key_position;
        float score = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
          const int column = head_start + dim;
          score += ri::LoadActivation(
                       inputs[0],
                       static_cast<size_t>(row) * (3 * embedding_dim_) + column,
                       output_type_) *
                   ri::LoadActivation(
                       inputs[0],
                       static_cast<size_t>(key_row) * (3 * embedding_dim_) +
                           embedding_dim_ + column,
                       output_type_);
        }
        scores[key_position] = score * scale;
        maximum = std::max(maximum, scores[key_position]);
      }
      float denominator = 0.0f;
      for (int key_position = 0; key_position <= query_position;
           ++key_position) {
        scores[key_position] = std::exp(scores[key_position] - maximum);
        denominator += scores[key_position];
      }
      for (int dim = 0; dim < head_dim; ++dim) {
        const int column = head_start + dim;
        float value = 0.0f;
        for (int key_position = 0; key_position <= query_position;
             ++key_position) {
          const int key_row = sequence_start + key_position;
          value += (scores[key_position] / denominator) *
                   ri::LoadActivation(
                       inputs[0],
                       static_cast<size_t>(key_row) * (3 * embedding_dim_) +
                           2 * embedding_dim_ + column,
                       output_type_);
        }
        ri::StoreActivation(&output,
                            static_cast<size_t>(row) * embedding_dim_ + column,
                            output_type_, value);
      }
    }
  }
  tape->intermediates = {inputs[0], output};
  tape->children.clear();
  return output;
}

absl::StatusOr<HostBufferVec> AttentionLayerReference::bwd(
    absl::Span<const HostBuffer> output_gradients, ReferenceTape tape) {
  if (output_gradients.size() != 1 || tape.intermediates.size() != 2) {
    return absl::InvalidArgumentError(
        "AttentionLayerReference bwd received incompatible state");
  }
  ASSIGN_OR_RETURN(int rows, ri::MatrixRows(output_gradients[0], embedding_dim_,
                                            "attention output gradient"));
  RETURN_IF_ERROR(
      ri::ValidateBuffer(tape.intermediates[0],
                         static_cast<size_t>(rows) * 3 * embedding_dim_ *
                             ri::ActivationElementBytes(output_type_),
                         "attention saved Q/K/V"));
  RETURN_IF_ERROR(
      ri::ValidateBuffer(tape.intermediates[1],
                         static_cast<size_t>(rows) * embedding_dim_ *
                             ri::ActivationElementBytes(output_type_),
                         "attention saved output"));
  ASSIGN_OR_RETURN(
      auto qkv_gradient,
      ri::AllocateFloats(static_cast<size_t>(rows) * 3 * embedding_dim_, true));
  auto* d_qkv = static_cast<float*>(qkv_gradient.data());
  const auto* d_output = static_cast<const float*>(output_gradients[0].data());
  const int head_dim = embedding_dim_ / num_heads_;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  std::vector<float> probabilities(context_length_);

  // The backward pass is written directly from softmax attention calculus:
  // dS=P*(dP-sum(P*dP)), dQ=dS*K, dK=dS*Q, and dV=P*dO. Accumulation is
  // explicit because a key/value participates in several later queries.
  for (int row = 0; row < rows; ++row) {
    const int sequence_start = (row / context_length_) * context_length_;
    const int query_position = row % context_length_;
    for (int head = 0; head < num_heads_; ++head) {
      const int head_start = head * head_dim;
      float maximum = -std::numeric_limits<float>::infinity();
      for (int key_position = 0; key_position <= query_position;
           ++key_position) {
        const int key_row = sequence_start + key_position;
        float score = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
          const int column = head_start + dim;
          score += ri::LoadActivation(
                       tape.intermediates[0],
                       static_cast<size_t>(row) * (3 * embedding_dim_) + column,
                       output_type_) *
                   ri::LoadActivation(
                       tape.intermediates[0],
                       static_cast<size_t>(key_row) * (3 * embedding_dim_) +
                           embedding_dim_ + column,
                       output_type_);
        }
        probabilities[key_position] = score * scale;
        maximum = std::max(maximum, probabilities[key_position]);
      }
      float denominator = 0.0f;
      for (int key_position = 0; key_position <= query_position;
           ++key_position) {
        probabilities[key_position] =
            std::exp(probabilities[key_position] - maximum);
        denominator += probabilities[key_position];
      }
      for (int key_position = 0; key_position <= query_position;
           ++key_position) {
        probabilities[key_position] /= denominator;
      }
      float delta = 0.0f;
      for (int dim = 0; dim < head_dim; ++dim) {
        const int column = head_start + dim;
        delta += d_output[static_cast<size_t>(row) * embedding_dim_ + column] *
                 ri::LoadActivation(
                     tape.intermediates[1],
                     static_cast<size_t>(row) * embedding_dim_ + column,
                     output_type_);
      }
      for (int key_position = 0; key_position <= query_position;
           ++key_position) {
        const int key_row = sequence_start + key_position;
        float d_probability = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
          const int column = head_start + dim;
          d_probability +=
              d_output[static_cast<size_t>(row) * embedding_dim_ + column] *
              ri::LoadActivation(
                  tape.intermediates[0],
                  static_cast<size_t>(key_row) * (3 * embedding_dim_) +
                      2 * embedding_dim_ + column,
                  output_type_);
        }
        const float d_score =
            probabilities[key_position] * (d_probability - delta);
        for (int dim = 0; dim < head_dim; ++dim) {
          const int column = head_start + dim;
          const size_t q =
              static_cast<size_t>(row) * (3 * embedding_dim_) + column;
          const size_t k = static_cast<size_t>(key_row) * (3 * embedding_dim_) +
                           embedding_dim_ + column;
          const size_t v = static_cast<size_t>(key_row) * (3 * embedding_dim_) +
                           2 * embedding_dim_ + column;
          d_qkv[q] +=
              d_score *
              ri::LoadActivation(tape.intermediates[0], k, output_type_) *
              scale;
          d_qkv[k] +=
              d_score *
              ri::LoadActivation(tape.intermediates[0], q, output_type_) *
              scale;
          d_qkv[v] +=
              probabilities[key_position] *
              d_output[static_cast<size_t>(row) * embedding_dim_ + column];
        }
      }
    }
  }
  return HostBufferVec{std::move(qkv_gradient)};
}

}  // namespace pluto::llm
