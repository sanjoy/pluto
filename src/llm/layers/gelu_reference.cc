#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layers/gelu.h"
#include "src/llm/layers/reference_internal.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace ri = reference_internal;
namespace {

constexpr float kSqrtTwoOverPi = 0.7978845608f;

float Gelu(float x) {
  const float inner = kSqrtTwoOverPi * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

float GeluDerivative(float x) {
  const float inner = kSqrtTwoOverPi * (x + 0.044715f * x * x * x);
  const float tanh_inner = std::tanh(inner);
  return 0.5f * (1.0f + tanh_inner) +
         0.5f * x * (1.0f - tanh_inner * tanh_inner) * kSqrtTwoOverPi *
             (1.0f + 3.0f * 0.044715f * x * x);
}

}  // namespace

absl::StatusOr<std::unique_ptr<GeluLayerReference>> GeluLayerReference::Create(
    int embedding_dim, DataType data_type, int sequence_length) {
  RETURN_IF_ERROR(ri::ValidateComputeType(data_type));
  if (sequence_length <= 0)
    return absl::InvalidArgumentError("sequence_length must be positive");
  // The device masks its final partial tile, so every positive width is valid.
  if (embedding_dim <= 0)
    return absl::InvalidArgumentError("embedding_dim must be positive");
  return absl::WrapUnique(
      new GeluLayerReference(embedding_dim, data_type, sequence_length));
}

absl::StatusOr<ReferenceFwdResult> GeluLayerReference::fwd_impl(
    absl::Span<const HostBuffer> inputs) const {
  ReferenceBackwardState state;
  if (inputs.size() != 1) {
    return absl::InvalidArgumentError(
        "GeluLayerReference fwd expects one input");
  }
  ASSIGN_OR_RETURN(
      int elements,
      ri::ElementCount(inputs[0], ri::ActivationElementBytes(output_type_),
                       "GELU input"));
  ASSIGN_OR_RETURN(auto output, ri::AllocateActivation(elements, output_type_));
  // A scalar call per element is intentionally boring: it makes the tanh GELU
  // approximation, including every constant, visible in one place.
  for (int index = 0; index < elements; ++index) {
    ri::StoreActivation(
        &output, index, output_type_,
        Gelu(ri::LoadActivation(inputs[0], index, output_type_)));
  }
  state.intermediates = {inputs[0]};
  state.children.clear();
  return ReferenceFwdResult{{std::move(output)}, std::move(state)};
}

absl::StatusOr<HostBufferVec> GeluLayerReference::bwd_impl(
    absl::Span<const HostBuffer> output_gradients,
    ReferenceBackwardState state) {
  if (output_gradients.size() != 1 || state.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "GeluLayerReference bwd received incompatible state");
  }
  ASSIGN_OR_RETURN(int elements,
                   ri::ElementCount(output_gradients[0], sizeof(float),
                                    "GELU output gradient"));
  RETURN_IF_ERROR(ri::ValidateBuffer(
      state.intermediates[0],
      static_cast<size_t>(elements) * ri::ActivationElementBytes(output_type_),
      "GELU saved input"));
  ASSIGN_OR_RETURN(auto input_gradient, ri::AllocateFloats(elements));
  const auto* d_output = static_cast<const float*>(output_gradients[0].data());
  auto* d_input = static_cast<float*>(input_gradient.data());
  for (int index = 0; index < elements; ++index) {
    d_input[index] =
        d_output[index] * GeluDerivative(ri::LoadActivation(
                              state.intermediates[0], index, output_type_));
  }
  return HostBufferVec{std::move(input_gradient)};
}

}  // namespace pluto::llm
