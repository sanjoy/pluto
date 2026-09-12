#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/reference_internal.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace ri = reference_internal;

ResidualLayerReference::ResidualLayerReference(
    std::unique_ptr<LayerReference> layer)
    : layer_(std::move(layer)) {
  for (const HostBuffer& weight : layer_->weights())
    weights_.push_back(weight);
  for (const HostBuffer& gradient : layer_->gradients())
    gradients_.push_back(gradient);
}

absl::StatusOr<ReferenceFwdResult> ResidualLayerReference::fwd_impl(
    absl::Span<const HostBuffer> inputs) const {
  ReferenceBackwardState state;
  if (inputs.size() != 1) {
    return absl::InvalidArgumentError(
        "ResidualLayerReference fwd expects one input");
  }

  ASSIGN_OR_RETURN(auto branch_fwd, layer_->fwd(inputs));
  if (branch_fwd.outputs.size() != 1) {
    return absl::InvalidArgumentError(
        "residual branch must return exactly one output");
  }
  auto branch = std::move(branch_fwd.outputs[0]);

  if (branch.size_bytes() != inputs[0].size_bytes()) {
    return absl::InvalidArgumentError(
        "reference residual branch changed activation shape");
  }
  ASSIGN_OR_RETURN(
      int elements,
      ri::ElementCount(inputs[0], ri::ActivationElementBytes(output_type()),
                       "residual input"));
  RETURN_IF_ERROR(ri::ValidateTiledExtent(elements, "residual elements"));
  ASSIGN_OR_RETURN(auto output, ri::AllocateActivation(elements, output_type()));
  // The reference operation is deliberately just x[i] + branch[i]. Rounding
  // happens once when the sum is stored in the activation dtype.
  for (int index = 0; index < elements; ++index) {
    ri::StoreActivation(&output, index, output_type(),
                        ri::LoadActivation(inputs[0], index, output_type()) +
                            ri::LoadActivation(branch, index, output_type()));
  }
  state.intermediates = {inputs[0]};
  state.children = {std::move(branch_fwd.state)};
  return ReferenceFwdResult{{std::move(output)}, std::move(state)};
}

absl::StatusOr<HostBufferVec> ResidualLayerReference::bwd_impl(
    absl::Span<const HostBuffer> output_gradients,
    ReferenceBackwardState state) {
  if (output_gradients.size() != 1 || state.intermediates.size() != 1 ||
      state.children.size() != 1) {
    return absl::InvalidArgumentError(
        "ResidualLayerReference bwd received incompatible state");
  }
  ASSIGN_OR_RETURN(auto branch_gradients,
                   layer_->bwd(output_gradients, std::move(state.children[0])));
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
  for (int index = 0; index < elements; ++index)
    output[index] = direct[index] + branch[index];
  return HostBufferVec{std::move(input_gradient)};
}

ComposedLayerReference::ComposedLayerReference(
    DataType data_type, std::vector<std::unique_ptr<LayerReference>> layers)
    : output_type_(data_type), layers_(std::move(layers)) {
  for (const auto& layer : layers_) {
    for (const HostBuffer& weight : layer->weights())
      weights_.push_back(weight);
    for (const HostBuffer& gradient : layer->gradients())
      gradients_.push_back(gradient);
  }
}

absl::StatusOr<ReferenceFwdResult> ComposedLayerReference::fwd_impl(
    absl::Span<const HostBuffer> inputs) const {
  ReferenceBackwardState state;
  HostBufferVec activations(inputs.begin(), inputs.end());
  for (const auto& layer : layers_) {
    ASSIGN_OR_RETURN(auto output_fwd, layer->fwd(activations));
    activations = std::move(output_fwd.outputs);
    state.children.push_back(std::move(output_fwd.state));
  }
  return ReferenceFwdResult{std::move(activations), std::move(state)};
}

absl::StatusOr<HostBufferVec> ComposedLayerReference::bwd_impl(
    absl::Span<const HostBuffer> output_gradients,
    ReferenceBackwardState state) {
  if (state.children.size() != layers_.size()) {
    return absl::InvalidArgumentError(
        "composed layer bwd received an incompatible state");
  }
  HostBufferVec gradients(output_gradients.begin(), output_gradients.end());
  for (size_t index = layers_.size(); index-- > 0;) {
    ASSIGN_OR_RETURN(
        gradients,
        layers_[index]->bwd(gradients, std::move(state.children[index])));
  }
  return gradients;
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
