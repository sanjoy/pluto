#include "src/llm/layers/gelu.h"

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
#include "src/gpu/buffer.h"
#include "src/llm/layers/internal.h"

namespace pluto::llm {

using internal::CudaStatus;
using internal::ElementCount;
using internal::kDenseTile;
using internal::MatrixRows;
using internal::TileCount;
using internal::ValidateBuffer;
using internal::ValidateFp16;
using internal::ValidateTiledExtent;

namespace {
__tile_global__ void GeluForwardKernel(const float* __restrict__ input,
                                        int elements,
                                        float* __restrict__ output) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{elements}}, ct::shape{16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{elements}}, ct::shape{16_ic}};
  const int block = ct::bid().x;
  auto x = input_view.load(block);
  constexpr float kSqrtTwoOverPi = 0.7978845608f;
  auto inner = kSqrtTwoOverPi * (x + 0.044715f * x * x * x);
  output_view.store(0.5f * x * (1.0f + ct::tanh(inner)), block);
}

__tile_global__ void GeluBackwardKernel(
    const float* __restrict__ input,
    const float* __restrict__ output_gradient,
    int elements,
    float* __restrict__ input_gradient) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{elements}}, ct::shape{16_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{elements}},
      ct::shape{16_ic}};
  auto input_gradient_view = ct::partition_view{
      ct::tensor_span{input_gradient, ct::extents{elements}},
      ct::shape{16_ic}};
  const int block = ct::bid().x;
  auto x = input_view.load(block);
  constexpr float kSqrtTwoOverPi = 0.7978845608f;
  auto inner = kSqrtTwoOverPi * (x + 0.044715f * x * x * x);
  auto tanh_inner = ct::tanh(inner);
  auto derivative =
      0.5f * (1.0f + tanh_inner) +
      0.5f * x * (1.0f - tanh_inner * tanh_inner) * kSqrtTwoOverPi *
          (1.0f + 3.0f * 0.044715f * x * x);
  input_gradient_view.store(
      gradient_view.load(block) * derivative, block);
}


}  // namespace

absl::StatusOr<std::unique_ptr<GeluLayer>> GeluLayer::Create(
    DataType data_type, cudaStream_t stream) {
  if (auto status = ValidateFp16(data_type); !status.ok()) return status;
  return std::unique_ptr<GeluLayer>(new GeluLayer(data_type, stream));
}

absl::StatusOr<Buffer> GeluLayer::fwd(absl::Span<const Buffer> inputs,
                                       Tape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "GeluLayer fwd expects one input and a non-null tape");
  }
  auto elements = ElementCount(inputs[0], sizeof(float), stream_, "GELU input");
  if (!elements.ok()) return elements.status();
  if (auto status = ValidateTiledExtent(*elements, "GELU element count");
      !status.ok()) return status;
  auto output = Buffer::Allocate(inputs[0].size_bytes(), stream_);
  if (!output.ok()) return output.status();
  tape->intermediates = {inputs[0]};
  tape->children.clear();
  GeluForwardKernel<<<TileCount(*elements), 1, 0, stream_>>>(
      static_cast<const float*>(inputs[0].data()), *elements,
      static_cast<float*>(output->data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "GeluForwardKernel launch");
      !status.ok()) {
    return status;
  }
  return *std::move(output);
}

absl::StatusOr<BufferVec> GeluLayer::bwd(
    absl::Span<const Buffer> output_gradients, Tape tape) {
  if (output_gradients.size() != 1 || tape.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "GeluLayer bwd received an incompatible gradient or tape");
  }
  auto elements = ElementCount(output_gradients[0], sizeof(float), stream_,
                               "GELU output gradient");
  if (!elements.ok()) return elements.status();
  if (auto status = ValidateBuffer(tape.intermediates[0],
                                   output_gradients[0].size_bytes(), stream_,
                                   "GELU saved input");
      !status.ok()) return status;
  auto input_gradient = Buffer::Allocate(output_gradients[0].size_bytes(),
                                         stream_);
  if (!input_gradient.ok()) return input_gradient.status();
  GeluBackwardKernel<<<TileCount(*elements), 1, 0, stream_>>>(
      static_cast<const float*>(tape.intermediates[0].data()),
      static_cast<const float*>(output_gradients[0].data()),
      *elements,
      static_cast<float*>(input_gradient->data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "GeluBackwardKernel launch");
      !status.ok()) {
    return status;
  }
  return BufferVec{*std::move(input_gradient)};
}


}  // namespace pluto::llm
