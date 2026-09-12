#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/reference_internal.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace ri = reference_internal;

absl::StatusOr<std::unique_ptr<FullyConnectedLayerReference>>
FullyConnectedLayerReference::Create(int input_dim, int output_dim,
                                     DataType data_type) {
  RETURN_IF_ERROR(ri::ValidateComputeType(data_type));
  RETURN_IF_ERROR(ri::ValidateTiledExtent(input_dim, "input_dim"));
  RETURN_IF_ERROR(ri::ValidateTiledExtent(output_dim, "output_dim"));
  ASSIGN_OR_RETURN(
      auto matrix,
      ri::AllocateFloats(static_cast<size_t>(input_dim) * output_dim, true));
  ASSIGN_OR_RETURN(auto bias, ri::AllocateFloats(output_dim, true));
  ASSIGN_OR_RETURN(
      auto matrix_gradient,
      ri::AllocateFloats(static_cast<size_t>(input_dim) * output_dim, true));
  ASSIGN_OR_RETURN(auto bias_gradient, ri::AllocateFloats(output_dim, true));
  return absl::WrapUnique(new FullyConnectedLayerReference(
      input_dim, output_dim, data_type, std::move(matrix), std::move(bias),
      std::move(matrix_gradient), std::move(bias_gradient)));
}

absl::Status FullyConnectedLayerReference::InitializeIdentity(float scale) {
  auto* matrix = static_cast<float*>(weights_[0].data());
  std::fill(matrix, matrix + static_cast<size_t>(input_dim_) * output_dim_,
            0.0f);
  for (int index = 0; index < std::min(input_dim_, output_dim_); ++index)
    matrix[static_cast<size_t>(index) * output_dim_ + index] = scale;
  std::memset(weights_[1].data(), 0, weights_[1].size_bytes());
  return absl::OkStatus();
}

absl::Status FullyConnectedLayerReference::InitializeNormal(
    float standard_deviation, uint64_t seed) {
  if (!(standard_deviation > 0.0f)) {
    return absl::InvalidArgumentError(
        "dense initialization standard deviation must be positive");
  }
  std::mt19937_64 random(seed);
  std::normal_distribution<float> distribution(0.0f, standard_deviation);
  auto* matrix = static_cast<float*>(weights_[0].data());
  const size_t elements = weights_[0].size_bytes() / sizeof(float);
  for (size_t index = 0; index < elements; ++index)
    matrix[index] = distribution(random);
  return absl::OkStatus();
}

absl::StatusOr<ReferenceFwdResult> FullyConnectedLayerReference::fwd_impl(
    absl::Span<const HostBuffer> inputs) const {
  ReferenceBackwardState state;
  if (inputs.size() != 1) {
    return absl::InvalidArgumentError(
        "FullyConnectedLayerReference fwd expects one input");
  }
  ASSIGN_OR_RETURN(int rows, ri::ActivationRows(inputs[0], input_dim_,
                                                output_type_, "dense input"));
  RETURN_IF_ERROR(ri::ValidateTiledExtent(rows, "dense rows"));
  ASSIGN_OR_RETURN(auto output,
                   ri::AllocateActivation(
                       static_cast<size_t>(rows) * output_dim_, output_type_));
  const auto* matrix = static_cast<const float*>(weights_[0].data());
  const auto* bias = static_cast<const float*>(weights_[1].data());

  // The intentionally obvious triple loop is the mathematical definition
  // against which the tiled MMA kernel is checked. Only MMA operands are
  // rounded to the selected compute type; accumulation and bias stay FP32.
  for (int row = 0; row < rows; ++row) {
    for (int output_column = 0; output_column < output_dim_; ++output_column) {
      float sum = bias[output_column];
      for (int input_column = 0; input_column < input_dim_; ++input_column) {
        const float left = ri::QuantizeMmaOperand(
            ri::LoadActivation(
                inputs[0], static_cast<size_t>(row) * input_dim_ + input_column,
                output_type_),
            output_type_);
        const float right = ri::QuantizeMmaOperand(
            matrix[static_cast<size_t>(input_column) * output_dim_ +
                   output_column],
            output_type_);
        sum += left * right;
      }
      ri::StoreActivation(
          &output, static_cast<size_t>(row) * output_dim_ + output_column,
          output_type_, sum);
    }
  }
  state.intermediates = {inputs[0]};
  state.children.clear();
  return ReferenceFwdResult{{std::move(output)}, std::move(state)};
}

absl::StatusOr<HostBufferVec> FullyConnectedLayerReference::bwd_impl(
    absl::Span<const HostBuffer> output_gradients,
    ReferenceBackwardState state) {
  if (output_gradients.size() != 1 || state.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "FullyConnectedLayerReference bwd received incompatible state");
  }
  ASSIGN_OR_RETURN(int rows, ri::MatrixRows(output_gradients[0], output_dim_,
                                            "dense output gradient"));
  RETURN_IF_ERROR(
      ri::ValidateBuffer(state.intermediates[0],
                         static_cast<size_t>(rows) * input_dim_ *
                             ri::ActivationElementBytes(output_type_),
                         "dense saved input"));
  ASSIGN_OR_RETURN(
      auto input_gradient,
      ri::AllocateFloats(static_cast<size_t>(rows) * input_dim_, true));
  const auto* output_gradient =
      static_cast<const float*>(output_gradients[0].data());
  const auto* matrix = static_cast<const float*>(weights_[0].data());
  auto* d_input = static_cast<float*>(input_gradient.data());
  auto* d_matrix = static_cast<float*>(gradients_[0].data());
  auto* d_bias = static_cast<float*>(gradients_[1].data());

  // These loops directly transcribe dX=dY*W^T, dW=X^T*dY, and db=sum(dY).
  // Quantization occurs at the same matrix-multiply boundaries as on the GPU.
  for (int row = 0; row < rows; ++row) {
    for (int input_column = 0; input_column < input_dim_; ++input_column) {
      float sum = 0.0f;
      for (int output_column = 0; output_column < output_dim_;
           ++output_column) {
        sum += ri::QuantizeMmaOperand(
                   output_gradient[static_cast<size_t>(row) * output_dim_ +
                                   output_column],
                   output_type_) *
               ri::QuantizeMmaOperand(
                   matrix[static_cast<size_t>(input_column) * output_dim_ +
                          output_column],
                   output_type_);
      }
      d_input[static_cast<size_t>(row) * input_dim_ + input_column] = sum;
    }
  }
  for (int input_column = 0; input_column < input_dim_; ++input_column) {
    for (int output_column = 0; output_column < output_dim_; ++output_column) {
      float sum = 0.0f;
      for (int row = 0; row < rows; ++row) {
        sum += ri::QuantizeMmaOperand(
                   ri::LoadActivation(
                       state.intermediates[0],
                       static_cast<size_t>(row) * input_dim_ + input_column,
                       output_type_),
                   output_type_) *
               ri::QuantizeMmaOperand(
                   output_gradient[static_cast<size_t>(row) * output_dim_ +
                                   output_column],
                   output_type_);
      }
      d_matrix[static_cast<size_t>(input_column) * output_dim_ +
               output_column] = sum;
    }
  }
  for (int output_column = 0; output_column < output_dim_; ++output_column) {
    float sum = 0.0f;
    for (int row = 0; row < rows; ++row) {
      sum += output_gradient[static_cast<size_t>(row) * output_dim_ +
                             output_column];
    }
    d_bias[output_column] = sum;
  }
  return HostBufferVec{std::move(input_gradient)};
}

}  // namespace pluto::llm
