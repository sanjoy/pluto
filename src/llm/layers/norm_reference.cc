#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layers/norm.h"
#include "src/llm/layers/reference_internal.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace ri = reference_internal;

absl::StatusOr<std::unique_ptr<LayerNormLayerReference>>
LayerNormLayerReference::Create(int embedding_dim, float epsilon,
                                DataType data_type) {
  RETURN_IF_ERROR(ri::ValidateComputeType(data_type));
  if (!(epsilon > 0.0f))
    return absl::InvalidArgumentError("layer-norm epsilon must be positive");
  RETURN_IF_ERROR(ri::ValidateTiledExtent(embedding_dim, "embedding_dim"));
  ASSIGN_OR_RETURN(auto gamma, ri::AllocateFloats(embedding_dim));
  ASSIGN_OR_RETURN(auto beta, ri::AllocateFloats(embedding_dim, true));
  ASSIGN_OR_RETURN(auto d_gamma, ri::AllocateFloats(embedding_dim, true));
  ASSIGN_OR_RETURN(auto d_beta, ri::AllocateFloats(embedding_dim, true));
  auto* gamma_values = static_cast<float*>(gamma.data());
  for (int column = 0; column < embedding_dim; ++column)
    gamma_values[column] = 1.0f;
  return absl::WrapUnique(new LayerNormLayerReference(
      embedding_dim, epsilon, data_type, std::move(gamma), std::move(beta),
      std::move(d_gamma), std::move(d_beta)));
}

absl::StatusOr<HostBuffer> LayerNormLayerReference::fwd(
    absl::Span<const HostBuffer> inputs, ReferenceTape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "LayerNormLayerReference fwd expects one input and a tape");
  }
  ASSIGN_OR_RETURN(int rows,
                   ri::ActivationRows(inputs[0], embedding_dim_, output_type_,
                                      "layer-norm input"));
  ASSIGN_OR_RETURN(auto output, ri::AllocateActivation(
                                    static_cast<size_t>(rows) * embedding_dim_,
                                    output_type_));
  const auto* gamma = static_cast<const float*>(weights_[0].data());
  const auto* beta = static_cast<const float*>(weights_[1].data());

  // Two scalar reductions make the population mean and variance explicit.
  // Affine parameters are FP32, and only the final activation is rounded.
  for (int row = 0; row < rows; ++row) {
    float mean = 0.0f;
    for (int column = 0; column < embedding_dim_; ++column) {
      mean += ri::LoadActivation(
          inputs[0], static_cast<size_t>(row) * embedding_dim_ + column,
          output_type_);
    }
    mean /= static_cast<float>(embedding_dim_);
    float variance = 0.0f;
    for (int column = 0; column < embedding_dim_; ++column) {
      const float centered =
          ri::LoadActivation(inputs[0],
                             static_cast<size_t>(row) * embedding_dim_ + column,
                             output_type_) -
          mean;
      variance += centered * centered;
    }
    variance /= static_cast<float>(embedding_dim_);
    const float inverse_stddev = 1.0f / std::sqrt(variance + epsilon_);
    for (int column = 0; column < embedding_dim_; ++column) {
      const size_t index = static_cast<size_t>(row) * embedding_dim_ + column;
      const float normalized =
          (ri::LoadActivation(inputs[0], index, output_type_) - mean) *
          inverse_stddev;
      ri::StoreActivation(&output, index, output_type_,
                          normalized * gamma[column] + beta[column]);
    }
  }
  tape->intermediates = {inputs[0]};
  tape->children.clear();
  return output;
}

absl::StatusOr<HostBufferVec> LayerNormLayerReference::bwd(
    absl::Span<const HostBuffer> output_gradients, ReferenceTape tape) {
  if (output_gradients.size() != 1 || tape.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "LayerNormLayerReference bwd received incompatible state");
  }
  ASSIGN_OR_RETURN(int rows, ri::MatrixRows(output_gradients[0], embedding_dim_,
                                            "layer-norm output gradient"));
  RETURN_IF_ERROR(
      ri::ValidateBuffer(tape.intermediates[0],
                         static_cast<size_t>(rows) * embedding_dim_ *
                             ri::ActivationElementBytes(output_type_),
                         "layer-norm saved input"));
  ASSIGN_OR_RETURN(
      auto input_gradient,
      ri::AllocateFloats(static_cast<size_t>(rows) * embedding_dim_));
  const auto* input_gradient_source =
      static_cast<const float*>(output_gradients[0].data());
  const auto* gamma = static_cast<const float*>(weights_[0].data());
  auto* d_input = static_cast<float*>(input_gradient.data());
  auto* d_gamma = static_cast<float*>(gradients_[0].data());
  auto* d_beta = static_cast<float*>(gradients_[1].data());
  for (int column = 0; column < embedding_dim_; ++column) {
    d_gamma[column] = 0.0f;
    d_beta[column] = 0.0f;
  }

  // For each row, dx = inv_std * (dn - mean(dn) - xhat*mean(dn*xhat)).
  // Parameter gradients are ordinary sums over rows. Keeping these equations
  // unfused provides an independent check of every CUDA reduction.
  for (int row = 0; row < rows; ++row) {
    float mean = 0.0f;
    for (int column = 0; column < embedding_dim_; ++column) {
      mean += ri::LoadActivation(
          tape.intermediates[0],
          static_cast<size_t>(row) * embedding_dim_ + column, output_type_);
    }
    mean /= static_cast<float>(embedding_dim_);
    float variance = 0.0f;
    for (int column = 0; column < embedding_dim_; ++column) {
      const float centered =
          ri::LoadActivation(tape.intermediates[0],
                             static_cast<size_t>(row) * embedding_dim_ + column,
                             output_type_) -
          mean;
      variance += centered * centered;
    }
    variance /= static_cast<float>(embedding_dim_);
    const float inverse_stddev = 1.0f / std::sqrt(variance + epsilon_);
    float gradient_sum = 0.0f;
    float projected_sum = 0.0f;
    for (int column = 0; column < embedding_dim_; ++column) {
      const size_t index = static_cast<size_t>(row) * embedding_dim_ + column;
      const float normalized =
          (ri::LoadActivation(tape.intermediates[0], index, output_type_) -
           mean) *
          inverse_stddev;
      const float d_normalized = input_gradient_source[index] * gamma[column];
      gradient_sum += d_normalized;
      projected_sum += d_normalized * normalized;
      d_gamma[column] += input_gradient_source[index] * normalized;
      d_beta[column] += input_gradient_source[index];
    }
    for (int column = 0; column < embedding_dim_; ++column) {
      const size_t index = static_cast<size_t>(row) * embedding_dim_ + column;
      const float normalized =
          (ri::LoadActivation(tape.intermediates[0], index, output_type_) -
           mean) *
          inverse_stddev;
      const float d_normalized = input_gradient_source[index] * gamma[column];
      d_input[index] =
          inverse_stddev * (d_normalized - gradient_sum / embedding_dim_ -
                            normalized * projected_sum / embedding_dim_);
    }
  }
  return HostBufferVec{std::move(input_gradient)};
}

}  // namespace pluto::llm
