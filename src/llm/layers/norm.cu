#include "src/llm/layers/norm.h"

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
#include "src/common/status_macros.h"
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
__tile_global__ void LayerNormForwardKernel(
    const float* __restrict__ input, int rows, int embedding_dim, float epsilon,
    float* __restrict__ output) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  const int width_tiles = embedding_dim / kDenseTile;
  const int block = ct::bid().x;
  const int row = block / width_tiles;
  const int output_tile = block % width_tiles;
  auto mean = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  for (int tile = 0; tile < width_tiles; ++tile) {
    mean = mean + ct::sum(input_view.load(row, tile), 1_ic);
  }
  mean = mean / static_cast<float>(embedding_dim);
  auto variance = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  for (int tile = 0; tile < width_tiles; ++tile) {
    auto centered = input_view.load(row, tile) - mean;
    variance = variance + ct::sum(centered * centered, 1_ic);
  }
  variance = variance / static_cast<float>(embedding_dim);
  auto values = input_view.load(row, output_tile);
  auto centered = values - mean;
  output_view.store(centered * ct::rsqrt(variance + epsilon), row,
                    output_tile);
}

__tile_global__ void LayerNormBackwardKernel(
    const float* __restrict__ input,
    const float* __restrict__ output_gradient, int rows, int embedding_dim,
    float epsilon,
    float* __restrict__ input_gradient) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  auto input_gradient_view = ct::partition_view{
      ct::tensor_span{input_gradient, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  const int width_tiles = embedding_dim / kDenseTile;
  const int block = ct::bid().x;
  const int row = block / width_tiles;
  const int output_tile = block % width_tiles;
  auto mean = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  for (int tile = 0; tile < width_tiles; ++tile) {
    mean = mean + ct::sum(input_view.load(row, tile), 1_ic);
  }
  mean = mean / static_cast<float>(embedding_dim);
  auto variance = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  auto gradient_sum = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  for (int tile = 0; tile < width_tiles; ++tile) {
    auto centered = input_view.load(row, tile) - mean;
    variance = variance + ct::sum(centered * centered, 1_ic);
    gradient_sum = gradient_sum + ct::sum(gradient_view.load(row, tile), 1_ic);
  }
  auto inverse_stddev =
      ct::rsqrt(variance / static_cast<float>(embedding_dim) + epsilon);
  auto projected_sum = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  for (int tile = 0; tile < width_tiles; ++tile) {
    auto normalized = (input_view.load(row, tile) - mean) * inverse_stddev;
    projected_sum = projected_sum +
                    ct::sum(gradient_view.load(row, tile) * normalized, 1_ic);
  }
  auto d_output = gradient_view.load(row, output_tile);
  auto normalized =
      (input_view.load(row, output_tile) - mean) * inverse_stddev;
  input_gradient_view.store(
      inverse_stddev *
          (d_output - gradient_sum / static_cast<float>(embedding_dim) -
           normalized * projected_sum / static_cast<float>(embedding_dim)),
      row, output_tile);
}


}  // namespace

absl::StatusOr<std::unique_ptr<LayerNormLayer>> LayerNormLayer::Create(
    int embedding_dim, float epsilon, DataType data_type,
    cudaStream_t stream) {
  RETURN_IF_ERROR(ValidateFp16(data_type));
  if (epsilon <= 0.0f) {
    return absl::InvalidArgumentError("layer-norm epsilon must be positive");
  }
  RETURN_IF_ERROR(ValidateTiledExtent(embedding_dim, "embedding_dim"));
  return std::unique_ptr<LayerNormLayer>(
      new LayerNormLayer(embedding_dim, epsilon, data_type, stream));
}

absl::StatusOr<Buffer> LayerNormLayer::fwd(
    absl::Span<const Buffer> inputs, Tape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "LayerNormLayer fwd expects one input and a non-null tape");
  }
  ASSIGN_OR_RETURN(int rows,
                   MatrixRows(inputs[0], embedding_dim_, stream_,
                              "layer-norm input"));
  const size_t activation_bytes = inputs[0].size_bytes();
  ASSIGN_OR_RETURN(auto output,
                   Buffer::Allocate(activation_bytes, stream_));
  tape->intermediates = {inputs[0]};
  tape->children.clear();
  LayerNormForwardKernel<<<rows * TileCount(embedding_dim_), 1, 0, stream_>>>(
      static_cast<const float*>(inputs[0].data()), rows, embedding_dim_,
      epsilon_,
      static_cast<float*>(output.data()));
  RETURN_IF_ERROR(
      CudaStatus(cudaGetLastError(), "LayerNormForwardKernel launch"));
  return std::move(output);
}

absl::StatusOr<BufferVec> LayerNormLayer::bwd(
    absl::Span<const Buffer> output_gradients, Tape tape) {
  if (output_gradients.size() != 1 || tape.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "LayerNormLayer bwd received an incompatible gradient or tape");
  }
  ASSIGN_OR_RETURN(int rows,
                   MatrixRows(output_gradients[0], embedding_dim_, stream_,
                              "layer-norm output gradient"));
  const size_t activation_bytes = output_gradients[0].size_bytes();
  RETURN_IF_ERROR(ValidateBuffer(tape.intermediates[0], activation_bytes,
                                 stream_, "layer-norm saved input"));
  ASSIGN_OR_RETURN(auto input_gradient,
                   Buffer::Allocate(activation_bytes, stream_));
  LayerNormBackwardKernel<<<rows * TileCount(embedding_dim_), 1, 0,
                            stream_>>>(
      static_cast<const float*>(tape.intermediates[0].data()),
      static_cast<const float*>(output_gradients[0].data()), rows,
      embedding_dim_, epsilon_,
      static_cast<float*>(input_gradient.data()));
  RETURN_IF_ERROR(
      CudaStatus(cudaGetLastError(), "LayerNormBackwardKernel launch"));
  return BufferVec{std::move(input_gradient)};
}


}  // namespace pluto::llm
