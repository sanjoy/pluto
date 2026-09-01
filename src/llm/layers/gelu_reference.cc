#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/util/status_macros.h"
#include "src/llm/layers/gelu.h"
#include "src/llm/layers/reference_internal.h"

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
    DataType data_type) {
  RETURN_IF_ERROR(ri::ValidateComputeType(data_type));
  return std::unique_ptr<GeluLayerReference>(new GeluLayerReference(data_type));
}

absl::StatusOr<HostBuffer> GeluLayerReference::fwd(
    absl::Span<const HostBuffer> inputs, ReferenceTape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "GeluLayerReference fwd expects one input and a tape");
  }
  ASSIGN_OR_RETURN(
      int elements,
      ri::ElementCount(inputs[0], ri::ActivationElementBytes(output_type_),
                       "GELU input"));
  RETURN_IF_ERROR(ri::ValidateTiledExtent(elements, "GELU element count"));
  ASSIGN_OR_RETURN(auto output, ri::AllocateActivation(elements, output_type_));
  // A scalar call per element is intentionally boring: it makes the tanh GELU
  // approximation, including every constant, visible in one place.
  for (int index = 0; index < elements; ++index) {
    ri::StoreActivation(
        &output, index, output_type_,
        Gelu(ri::LoadActivation(inputs[0], index, output_type_)));
  }
  tape->intermediates = {inputs[0]};
  tape->children.clear();
  return output;
}

absl::StatusOr<HostBufferVec> GeluLayerReference::bwd(
    absl::Span<const HostBuffer> output_gradients, ReferenceTape tape) {
  if (output_gradients.size() != 1 || tape.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "GeluLayerReference bwd received incompatible state");
  }
  ASSIGN_OR_RETURN(int elements,
                   ri::ElementCount(output_gradients[0], sizeof(float),
                                    "GELU output gradient"));
  RETURN_IF_ERROR(ri::ValidateBuffer(
      tape.intermediates[0],
      static_cast<size_t>(elements) * ri::ActivationElementBytes(output_type_),
      "GELU saved input"));
  ASSIGN_OR_RETURN(auto input_gradient, ri::AllocateFloats(elements));
  const auto* d_output = static_cast<const float*>(output_gradients[0].data());
  auto* d_input = static_cast<float*>(input_gradient.data());
  for (int index = 0; index < elements; ++index) {
    d_input[index] =
        d_output[index] * GeluDerivative(ri::LoadActivation(
                              tape.intermediates[0], index, output_type_));
  }
  return HostBufferVec{std::move(input_gradient)};
}

}  // namespace pluto::llm
