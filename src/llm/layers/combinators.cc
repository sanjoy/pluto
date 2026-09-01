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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "src/cuda/buffer.h"
#include "src/llm/layers/internal.h"
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

ResidualLayer::ResidualLayer(std::unique_ptr<Layer> layer)
    : layer_(std::move(layer)) {
  for (const Buffer& weight : layer_->weights()) weights_.push_back(weight);
  for (const Buffer& gradient : layer_->gradients()) {
    gradients_.push_back(gradient);
  }
}

absl::StatusOr<Buffer> ResidualLayer::fwd(cuda::Executor& executor,
                                          absl::Span<const Buffer> inputs,
                                          Tape* tape) const {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "ResidualLayer fwd expects one input and a non-null tape");
  }
  Tape child_tape;
  ASSIGN_OR_RETURN(auto branch, layer_->fwd(executor, inputs, &child_tape));
  if (branch.size_bytes() != inputs[0].size_bytes() ||
      &branch.executor() != &executor || &inputs[0].executor() != &executor) {
    return absl::InvalidArgumentError(
        "ResidualLayer branch changed the activation shape or executor");
  }
  ASSIGN_OR_RETURN(auto output,
                   Buffer::Allocate(executor, inputs[0].size_bytes()));
  ASSIGN_OR_RETURN(int elements,
                   ElementCount(executor, inputs[0],
                                internal::ActivationElementBytes(output_type()),
                                "residual input"));
  RETURN_IF_ERROR(ValidateTiledExtent(elements, "residual element count"));
  tape->intermediates = {inputs[0]};
  tape->children = {std::move(child_tape)};
  if (output_type() == DataType::BF16) {
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
  return std::move(output);
}

absl::StatusOr<BufferVec> ResidualLayer::bwd(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    Tape tape) {
  if (output_gradients.size() != 1 || tape.intermediates.size() != 1 ||
      tape.children.size() != 1) {
    return absl::InvalidArgumentError(
        "ResidualLayer bwd received an incompatible gradient or tape");
  }
  ASSIGN_OR_RETURN(
      auto branch_gradient,
      layer_->bwd(executor, output_gradients, std::move(tape.children[0])));
  if (branch_gradient.size() != 1 || branch_gradient.front().size_bytes() !=
                                         output_gradients[0].size_bytes()) {
    return absl::InvalidArgumentError(
        "ResidualLayer branch returned an incompatible input gradient");
  }
  ASSIGN_OR_RETURN(
      auto input_gradient,
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

ComposedLayer::ComposedLayer(DataType data_type,
                             std::vector<std::unique_ptr<Layer>> layers)
    : output_type_(data_type), layers_(std::move(layers)) {
  for (const auto& layer : layers_) {
    for (const Buffer& weight : layer->weights()) weights_.push_back(weight);
    for (const Buffer& gradient : layer->gradients()) {
      gradients_.push_back(gradient);
    }
  }
}

absl::StatusOr<Buffer> ComposedLayer::fwd(cuda::Executor& executor,
                                          absl::Span<const Buffer> inputs,
                                          Tape* tape) const {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "ComposedLayer fwd expects one input and a non-null tape");
  }
  tape->intermediates.clear();
  tape->children.clear();
  Buffer activation = inputs.front();
  for (const auto& layer : layers_) {
    Tape child_tape;
    BufferVec child_inputs = {activation};
    ASSIGN_OR_RETURN(auto output,
                     layer->fwd(executor, child_inputs, &child_tape));
    activation = std::move(output);
    tape->children.push_back(std::move(child_tape));
  }
  return activation;
}

absl::StatusOr<BufferVec> ComposedLayer::bwd(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    Tape tape) {
  if (output_gradients.size() != 1 || tape.children.size() != layers_.size()) {
    return absl::InvalidArgumentError(
        "ComposedLayer bwd received an incompatible gradient or tape");
  }
  Buffer gradient = output_gradients.front();
  for (size_t index = layers_.size(); index-- > 0;) {
    BufferVec child_gradients = {gradient};
    ASSIGN_OR_RETURN(auto input_gradients,
                     layers_[index]->bwd(executor, child_gradients,
                                         std::move(tape.children[index])));
    if (index == 0 && input_gradients.empty()) return BufferVec{};
    if (input_gradients.size() != 1) {
      return absl::InternalError(
          "a composed unary layer returned multiple input gradients");
    }
    gradient = input_gradients.front();
  }
  return BufferVec{gradient};
}

absl::Status ComposedLayerBuilder::add(std::unique_ptr<Layer> layer) {
  if (layer == nullptr) {
    return absl::InvalidArgumentError(
        "ComposedLayerBuilder cannot add a null layer");
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

absl::StatusOr<std::unique_ptr<ComposedLayer>> ComposedLayerBuilder::create() {
  if (layers_.empty()) {
    return absl::FailedPreconditionError(
        "cannot create an empty ComposedLayer");
  }
  const DataType output_type = layers_.back()->output_type();
  std::vector<std::unique_ptr<Layer>> layers;
  layers.swap(layers_);
  return std::make_unique<ComposedLayer>(output_type, std::move(layers));
}

}  // namespace pluto::llm
