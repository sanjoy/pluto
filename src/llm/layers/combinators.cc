#include "src/llm/layers/combinators.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/cuda/buffer.h"
#include "src/llm/layer_hooks.h"
#include "src/llm/layers/type_check_util.h"
#include "src/llm/layers/util.h"
#include "src/util/status_macros.h"

namespace pluto::llm {

using cuda::CudaStatus;
using internal::ElementCount;
using internal::kDenseTile;
using internal::MatrixRows;
using internal::TileCount;
using internal::ValidateBuffer;
using internal::ValidateFp16;
using internal::ValidateTiledExtent;

namespace {
// Invoke a member body, keeping hook scopes balanced even on an error return.
// Forward the arguments only when invoking the body, so backward state is not
// consumed if enter_combinator fails. LayerType preserves fwd's const receiver.
// This codebase disables C++ exceptions.
template <class LayerType, class Function, class... Args>
auto WithCombinatorScope(cuda::Executor& executor, LayerHooks* hooks,
                         LayerType& layer, Function body, Args&&... args)
    -> decltype((layer.*body)(executor, std::forward<Args>(args)..., hooks)) {
  if (hooks == nullptr)
    return (layer.*body)(executor, std::forward<Args>(args)..., hooks);
  if (hooks->enter_combinator)
    RETURN_IF_ERROR(hooks->enter_combinator(executor, layer.name()));

  auto result = (layer.*body)(executor, std::forward<Args>(args)..., hooks);
  if (!hooks->exit_combinator)
    return result;
  const auto exit_hook_status = hooks->exit_combinator(executor);
  if (exit_hook_status.ok())
    return result;
  if (result.ok())
    return exit_hook_status;
  return absl::Status(
      result.status().code(),
      absl::StrCat(result.status().message(),
                   "; exit_combinator failed: ", exit_hook_status.ToString()));
}

template <class Element>
__tile_global__ void AddKernel(const Element* __restrict__ left,
                               const Element* __restrict__ right, int elements,
                               Element* __restrict__ output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;

  auto left_view = ct::partition_view{
      ct::tensor_span{left, ct::extents{elements}}, ct::shape{16_ic}};
  auto right_view = ct::partition_view{
      ct::tensor_span{right, ct::extents{elements}}, ct::shape{16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{elements}}, ct::shape{16_ic}};
  const int block = ct::bid().x;
  auto sum = ct::element_cast<float>(left_view.load(block)) +
             ct::element_cast<float>(right_view.load(block));
  output_view.store(ct::element_cast<Element>(sum), block);
}

}  // namespace

absl::StatusOr<std::unique_ptr<ResidualLayer>> ResidualLayer::Create(
    std::unique_ptr<Layer> layer) {
  if (layer == nullptr)
    return absl::InvalidArgumentError("residual branch must not be null");
  RETURN_IF_ERROR(internal::ValidateResidualTypes(layer->input_types(),
                                                  layer->output_types()));
  return absl::WrapUnique(new ResidualLayer(std::move(layer)));
}

ResidualLayer::ResidualLayer(std::unique_ptr<Layer> layer)
    : layer_(std::move(layer)) {
  for (const Buffer& weight : layer_->weights())
    weights_.push_back(weight);
  for (const Buffer& gradient : layer_->gradients())
    gradients_.push_back(gradient);
}

absl::StatusOr<FwdResult> ResidualLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks* hooks) const {
  return WithCombinatorScope(executor, hooks, *this, &ResidualLayer::fwd_body,
                             inputs);
}

absl::StatusOr<FwdResult> ResidualLayer::fwd_body(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks* hooks) const {
  BackwardState state;
  if (inputs.size() != 1)
    return absl::InvalidArgumentError("ResidualLayer fwd expects one input");

  ASSIGN_OR_RETURN(auto branch_fwd, layer_->fwd(executor, inputs, hooks));
  if (branch_fwd.outputs.size() != 1) {
    return absl::InvalidArgumentError(
        "residual branch must return exactly one output");
  }
  auto branch = std::move(branch_fwd.outputs[0]);

  if (branch.size_bytes() != inputs[0].size_bytes() ||
      &branch.executor() != &executor || &inputs[0].executor() != &executor) {
    return absl::InvalidArgumentError(
        "ResidualLayer branch changed the activation shape or executor");
  }
  // The branch's compute policy can differ from its physical output
  // dtype. Both element counting and addition must follow the checked
  // signature.
  const DataType storage_type = input_types()[0].data_type();
  ASSIGN_OR_RETURN(auto output,
                   Buffer::Allocate(executor, inputs[0].size_bytes()));
  ASSIGN_OR_RETURN(int elements,
                   ElementCount(executor, inputs[0],
                                internal::ActivationElementBytes(storage_type),
                                "residual input"));
  RETURN_IF_ERROR(ValidateTiledExtent(elements, "residual element count"));
  state.intermediates = {inputs[0]};
  state.children = {std::move(branch_fwd.state)};
  if (storage_type == DataType::BF16) {
    AddKernel<__nv_bfloat16><<<TileCount(elements), 1, 0, executor.stream()>>>(
        static_cast<const __nv_bfloat16*>(inputs[0].data()),
        static_cast<const __nv_bfloat16*>(branch.data()), elements,
        static_cast<__nv_bfloat16*>(output.data()));
  } else {
    AddKernel<float><<<TileCount(elements), 1, 0, executor.stream()>>>(
        static_cast<const float*>(inputs[0].data()),
        static_cast<const float*>(branch.data()), elements,
        static_cast<float*>(output.data()));
  }
  RETURN_IF_ERROR(CudaStatus(cudaGetLastError(), "AddKernel(residual) launch"));
  return FwdResult{{std::move(output)}, std::move(state)};
}

absl::StatusOr<BufferVec> ResidualLayer::bwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    BackwardState state, LayerHooks* hooks) {
  return WithCombinatorScope(executor, hooks, *this, &ResidualLayer::bwd_body,
                             output_gradients, std::move(state));
}

absl::StatusOr<BufferVec> ResidualLayer::bwd_body(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    BackwardState state, LayerHooks* hooks) {
  if (output_gradients.size() != 1 || state.intermediates.size() != 1 ||
      state.children.size() != 1) {
    return absl::InvalidArgumentError(
        "ResidualLayer bwd received an incompatible gradient or state");
  }
  ASSIGN_OR_RETURN(auto branch_gradient,
                   layer_->bwd(executor, output_gradients,
                               std::move(state.children[0]), hooks));
  if (branch_gradient.size() != 1 || branch_gradient.front().size_bytes() !=
                                         output_gradients[0].size_bytes()) {
    return absl::InvalidArgumentError(
        "ResidualLayer branch returned an incompatible input gradient");
  }
  ASSIGN_OR_RETURN(auto input_gradient,
                   Buffer::Allocate(executor, output_gradients[0].size_bytes()));
  ASSIGN_OR_RETURN(int elements,
                   ElementCount(executor, output_gradients[0], sizeof(float),
                                "residual output gradient"));
  AddKernel<float><<<TileCount(elements), 1, 0, executor.stream()>>>(
      static_cast<const float*>(output_gradients[0].data()),
      static_cast<const float*>(branch_gradient.front().data()), elements,
      static_cast<float*>(input_gradient.data()));
  RETURN_IF_ERROR(
      CudaStatus(cudaGetLastError(), "AddKernel(residual gradient) launch"));
  return BufferVec{std::move(input_gradient)};
}

absl::StatusOr<std::unique_ptr<ComposedLayer>> ComposedLayer::Create(
    std::string name, std::vector<std::unique_ptr<Layer>> layers) {
  if (name.empty())
    return absl::InvalidArgumentError("composed layer name must not be empty");
  if (layers.empty())
    return absl::FailedPreconditionError(
        "cannot create an empty ComposedLayer");
  for (size_t i = 0; i < layers.size(); ++i) {
    if (layers[i] == nullptr)
      return absl::InvalidArgumentError("composed child must not be null");
    RETURN_IF_ERROR(internal::ValidateTypes(layers[i]->input_types()));
    RETURN_IF_ERROR(internal::ValidateTypes(layers[i]->output_types()));
    if (i != 0) {
      const auto status = internal::ValidateTypeConnection(
          layers[i - 1]->output_types(), layers[i]->input_types());
      if (!status.ok())
        return absl::InvalidArgumentError(
            absl::StrCat("composed child ", i, ": ", status.message()));
    }
  }
  return absl::WrapUnique(
      new ComposedLayer(std::move(name), std::move(layers)));
}

ComposedLayer::ComposedLayer(std::string name,
                             std::vector<std::unique_ptr<Layer>> layers)
    : name_(std::move(name)),
      output_type_(layers.back()->output_type()),
      layers_(std::move(layers)) {
  for (const auto& layer : layers_) {
    for (const Buffer& weight : layer->weights())
      weights_.push_back(weight);
    for (const Buffer& gradient : layer->gradients())
      gradients_.push_back(gradient);
  }
}

absl::StatusOr<FwdResult> ComposedLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks* hooks) const {
  return WithCombinatorScope(executor, hooks, *this, &ComposedLayer::fwd_body,
                             inputs);
}

absl::StatusOr<FwdResult> ComposedLayer::fwd_body(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks* hooks) const {
  BackwardState state;
  BufferVec activations(inputs.begin(), inputs.end());
  for (const auto& layer : layers_) {
    ASSIGN_OR_RETURN(auto output_fwd, layer->fwd(executor, activations, hooks));
    activations = std::move(output_fwd.outputs);
    state.children.push_back(std::move(output_fwd.state));
  }
  return FwdResult{std::move(activations), std::move(state)};
}

absl::StatusOr<BufferVec> ComposedLayer::bwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    BackwardState state, LayerHooks* hooks) {
  return WithCombinatorScope(executor, hooks, *this, &ComposedLayer::bwd_body,
                             output_gradients, std::move(state));
}

absl::StatusOr<BufferVec> ComposedLayer::bwd_body(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    BackwardState state, LayerHooks* hooks) {
  if (state.children.size() != layers_.size()) {
    return absl::InvalidArgumentError(
        "composed layer bwd received an incompatible state");
  }
  BufferVec gradients(output_gradients.begin(), output_gradients.end());
  for (size_t index = layers_.size(); index-- > 0;) {
    ASSIGN_OR_RETURN(
        gradients, layers_[index]->bwd(executor, gradients,
                                       std::move(state.children[index]), hooks));
  }
  return gradients;
}

absl::Status ComposedLayerBuilder::add(std::unique_ptr<Layer> layer) {
  if (layer == nullptr) {
    return absl::InvalidArgumentError(
        "ComposedLayerBuilder cannot add a null layer");
  }
  RETURN_IF_ERROR(internal::ValidateTypes(layer->input_types()));
  RETURN_IF_ERROR(internal::ValidateTypes(layer->output_types()));
  if (!layers_.empty()) {
    RETURN_IF_ERROR(internal::ValidateTypeConnection(
        layers_.back()->output_types(), layer->input_types()));
  }
  layers_.push_back(std::move(layer));
  return absl::OkStatus();
}

Layer* ComposedLayerBuilder::back() {
  return layers_.empty() ? nullptr : layers_.back().get();
}

const Layer* ComposedLayerBuilder::back() const {
  return layers_.empty() ? nullptr : layers_.back().get();
}

absl::StatusOr<std::unique_ptr<ComposedLayer>> ComposedLayerBuilder::create(
    std::string name) {
  if (name.empty())
    return absl::InvalidArgumentError("composed layer name must not be empty");
  if (layers_.empty()) {
    return absl::FailedPreconditionError(
        "cannot create an empty ComposedLayer");
  }
  std::vector<std::unique_ptr<Layer>> layers;
  layers.swap(layers_);
  return ComposedLayer::Create(std::move(name), std::move(layers));
}

}  // namespace pluto::llm
