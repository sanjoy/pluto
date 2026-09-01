#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/util/status_macros.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/reference_internal.h"

namespace pluto::llm {
namespace ri = reference_internal;

ResidualLayerReference::ResidualLayerReference(
    std::unique_ptr<LayerReference> layer)
    : layer_(std::move(layer)) {
  for (const HostBuffer& weight : layer_->weights()) weights_.push_back(weight);
  for (const HostBuffer& gradient : layer_->gradients()) {
    gradients_.push_back(gradient);
  }
}

absl::StatusOr<HostBuffer> ResidualLayerReference::fwd(
    absl::Span<const HostBuffer> inputs, ReferenceTape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "ResidualLayerReference fwd expects one input and a tape");
  }
  ReferenceTape child_tape;
  ASSIGN_OR_RETURN(auto branch, layer_->fwd(inputs, &child_tape));
  if (branch.size_bytes() != inputs[0].size_bytes()) {
    return absl::InvalidArgumentError(
        "reference residual branch changed activation shape");
  }
  ASSIGN_OR_RETURN(
      int elements,
      ri::ElementCount(inputs[0], ri::ActivationElementBytes(output_type()),
                       "residual input"));
  RETURN_IF_ERROR(ri::ValidateTiledExtent(elements, "residual elements"));
  ASSIGN_OR_RETURN(auto output,
                   ri::AllocateActivation(elements, output_type()));
  // The reference operation is deliberately just x[i] + branch[i]. Rounding
  // happens once when the sum is stored in the activation dtype.
  for (int index = 0; index < elements; ++index) {
    ri::StoreActivation(&output, index, output_type(),
                        ri::LoadActivation(inputs[0], index, output_type()) +
                            ri::LoadActivation(branch, index, output_type()));
  }
  tape->intermediates = {inputs[0]};
  tape->children = {std::move(child_tape)};
  return output;
}

absl::StatusOr<HostBufferVec> ResidualLayerReference::bwd(
    absl::Span<const HostBuffer> output_gradients, ReferenceTape tape) {
  if (output_gradients.size() != 1 || tape.intermediates.size() != 1 ||
      tape.children.size() != 1) {
    return absl::InvalidArgumentError(
        "ResidualLayerReference bwd received incompatible state");
  }
  ASSIGN_OR_RETURN(auto branch_gradients,
                   layer_->bwd(output_gradients, std::move(tape.children[0])));
  if (branch_gradients.size() != 1 ||
      branch_gradients[0].size_bytes() != output_gradients[0].size_bytes()) {
    return absl::InvalidArgumentError(
        "reference residual branch returned incompatible gradient");
  }
  ASSIGN_OR_RETURN(int elements,
                   ri::ElementCount(output_gradients[0], sizeof(float),
                                    "residual output gradient"));
  ASSIGN_OR_RETURN(auto input_gradient, ri::AllocateFloats(elements));
  const auto* direct = static_cast<const float*>(output_gradients[0].data());
  const auto* branch = static_cast<const float*>(branch_gradients[0].data());
  auto* output = static_cast<float*>(input_gradient.data());
  for (int index = 0; index < elements; ++index) {
    output[index] = direct[index] + branch[index];
  }
  return HostBufferVec{std::move(input_gradient)};
}

ComposedLayerReference::ComposedLayerReference(
    DataType data_type, std::vector<std::unique_ptr<LayerReference>> layers)
    : output_type_(data_type), layers_(std::move(layers)) {
  for (const auto& layer : layers_) {
    for (const HostBuffer& weight : layer->weights())
      weights_.push_back(weight);
    for (const HostBuffer& gradient : layer->gradients()) {
      gradients_.push_back(gradient);
    }
  }
}

absl::StatusOr<HostBuffer> ComposedLayerReference::fwd(
    absl::Span<const HostBuffer> inputs, ReferenceTape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "ComposedLayerReference fwd expects one input and a tape");
  }
  tape->intermediates.clear();
  tape->children.clear();
  HostBuffer activation = inputs[0];
  // Execute one child at a time and retain its independent tape. This is
  // intentionally the simplest possible interpretation of composition.
  for (const auto& layer : layers_) {
    ReferenceTape child_tape;
    HostBufferVec child_inputs = {activation};
    ASSIGN_OR_RETURN(auto output, layer->fwd(child_inputs, &child_tape));
    activation = std::move(output);
    tape->children.push_back(std::move(child_tape));
  }
  return activation;
}

absl::StatusOr<HostBufferVec> ComposedLayerReference::bwd(
    absl::Span<const HostBuffer> output_gradients, ReferenceTape tape) {
  if (output_gradients.size() != 1 || tape.children.size() != layers_.size()) {
    return absl::InvalidArgumentError(
        "ComposedLayerReference bwd received incompatible state");
  }
  HostBuffer gradient = output_gradients[0];
  for (size_t index = layers_.size(); index-- > 0;) {
    HostBufferVec child_gradients = {gradient};
    ASSIGN_OR_RETURN(
        auto input_gradients,
        layers_[index]->bwd(child_gradients, std::move(tape.children[index])));
    if (index == 0 && input_gradients.empty()) return HostBufferVec{};
    if (input_gradients.size() != 1) {
      return absl::InternalError(
          "a composed reference layer returned multiple input gradients");
    }
    gradient = input_gradients[0];
  }
  return HostBufferVec{gradient};
}

absl::Status ComposedLayerReferenceBuilder::add(
    std::unique_ptr<LayerReference> layer) {
  if (layer == nullptr) {
    return absl::InvalidArgumentError(
        "ComposedLayerReferenceBuilder cannot add a null layer");
  }
  layers_.push_back(std::move(layer));
  return absl::OkStatus();
}

LayerReference* ComposedLayerReferenceBuilder::back() {
  return layers_.empty() ? nullptr : layers_.back().get();
}

const LayerReference* ComposedLayerReferenceBuilder::back() const {
  return layers_.empty() ? nullptr : layers_.back().get();
}

absl::StatusOr<std::unique_ptr<ComposedLayerReference>>
ComposedLayerReferenceBuilder::create() {
  if (layers_.empty()) {
    return absl::FailedPreconditionError(
        "cannot create an empty ComposedLayerReference");
  }
  const DataType output_type = layers_.back()->output_type();
  std::vector<std::unique_ptr<LayerReference>> layers;
  layers.swap(layers_);
  return std::make_unique<ComposedLayerReference>(output_type,
                                                  std::move(layers));
}

}  // namespace pluto::llm
