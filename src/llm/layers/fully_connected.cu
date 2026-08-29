#include "src/llm/layers/fully_connected.h"

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
__tile_global__ void DenseForwardKernel(
    const float* __restrict__ input, const float* __restrict__ matrix,
    const float* __restrict__ bias, int rows, int model_width,
    float* __restrict__ output) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{rows, model_width}},
      ct::shape{16_ic, 16_ic}};
  auto matrix_view = ct::partition_view{
      ct::tensor_span{matrix, ct::extents{model_width, model_width}},
      ct::shape{16_ic, 16_ic}};
  auto bias_view = ct::partition_view{
      ct::tensor_span{bias, ct::extents{model_width}}, ct::shape{16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{rows, model_width}},
      ct::shape{16_ic, 16_ic}};

  const int width_tiles = model_width / kDenseTile;
  const int block = ct::bid().x;
  const int batch_tile = block / width_tiles;
  const int output_tile = block % width_tiles;
  auto accumulator = ct::broadcast(bias_view.load(output_tile),
                                   ct::shape{16_ic, 16_ic});
  for (int inner_tile = 0; inner_tile < width_tiles; ++inner_tile) {
    auto left =
        ct::element_cast<__half>(input_view.load(batch_tile, inner_tile));
    auto right =
        ct::element_cast<__half>(matrix_view.load(inner_tile, output_tile));
    accumulator = ct::mma(left, right, accumulator);
  }
  output_view.store(accumulator, batch_tile, output_tile);
}

__tile_global__ void DenseInputGradientKernel(
    const float* __restrict__ output_gradient,
    const float* __restrict__ matrix, int rows, int model_width,
    float* __restrict__ input_gradient) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, model_width}},
      ct::shape{16_ic, 16_ic}};
  auto matrix_view = ct::partition_view{
      ct::tensor_span{matrix, ct::extents{model_width, model_width}},
      ct::shape{16_ic, 16_ic}};
  auto input_gradient_view = ct::partition_view{
      ct::tensor_span{input_gradient, ct::extents{rows, model_width}},
      ct::shape{16_ic, 16_ic}};

  const int width_tiles = model_width / kDenseTile;
  const int block = ct::bid().x;
  const int batch_tile = block / width_tiles;
  const int input_tile = block % width_tiles;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  for (int output_tile = 0; output_tile < width_tiles;
       ++output_tile) {
    auto gradient = ct::element_cast<__half>(
        gradient_view.load(batch_tile, output_tile));
    auto matrix_transposed = ct::transpose(ct::element_cast<__half>(
        matrix_view.load(input_tile, output_tile)));
    accumulator = ct::mma(gradient, matrix_transposed, accumulator);
  }
  input_gradient_view.store(accumulator, batch_tile, input_tile);
}

__tile_global__ void DenseWeightUpdateKernel(
    const float* __restrict__ input,
    const float* __restrict__ output_gradient, float learning_rate,
    int rows, int model_width, float* __restrict__ matrix) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{rows, model_width}},
      ct::shape{16_ic, 16_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, model_width}},
      ct::shape{16_ic, 16_ic}};
  auto matrix_view = ct::partition_view{
      ct::tensor_span{matrix, ct::extents{model_width, model_width}},
      ct::shape{16_ic, 16_ic}};

  const int width_tiles = model_width / kDenseTile;
  const int batch_tiles = rows / kDenseTile;
  const int block = ct::bid().x;
  const int input_tile = block / width_tiles;
  const int output_tile = block % width_tiles;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  for (int batch_tile = 0; batch_tile < batch_tiles; ++batch_tile) {
    auto input_transposed = ct::transpose(ct::element_cast<__half>(
        input_view.load(batch_tile, input_tile)));
    auto gradient = ct::element_cast<__half>(
        gradient_view.load(batch_tile, output_tile));
    accumulator = ct::mma(input_transposed, gradient, accumulator);
  }
  auto old_matrix = matrix_view.load(input_tile, output_tile);
  matrix_view.store(old_matrix - learning_rate * accumulator, input_tile,
                    output_tile);
}

__tile_global__ void DenseBiasUpdateKernel(
    const float* __restrict__ output_gradient, float learning_rate,
    int rows, int model_width, float* __restrict__ bias) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, model_width}},
      ct::shape{16_ic, 16_ic}};
  auto bias_view = ct::partition_view{
      ct::tensor_span{bias, ct::extents{model_width}}, ct::shape{16_ic}};

  const int batch_tiles = rows / kDenseTile;
  const int output_tile = ct::bid().x;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<1, 16>>>();
  for (int batch_tile = 0; batch_tile < batch_tiles; ++batch_tile) {
    accumulator = accumulator +
                  ct::sum(gradient_view.load(batch_tile, output_tile), 0_ic);
  }
  auto old_bias = bias_view.load(output_tile);
  bias_view.store(old_bias - learning_rate *
                                 ct::reshape(accumulator, ct::shape{16_ic}),
                  output_tile);
}


}  // namespace

FullyConnectedLayer::FullyConnectedLayer(
    int model_width, DataType data_type, float learning_rate,
    cudaStream_t stream, Buffer matrix, Buffer bias)
    : model_width_(model_width),
      output_type_(data_type),
      learning_rate_(learning_rate),
      stream_(stream),
      weights_{std::move(matrix), std::move(bias)} {}

absl::StatusOr<std::unique_ptr<FullyConnectedLayer>>
FullyConnectedLayer::Create(int model_width, DataType data_type,
                            float learning_rate, cudaStream_t stream) {
  RETURN_IF_ERROR(ValidateFp16(data_type));
  if (learning_rate < 0.0f) {
    return absl::InvalidArgumentError("learning rate must be non-negative");
  }
  RETURN_IF_ERROR(ValidateTiledExtent(model_width, "model_width"));
  const size_t matrix_elements =
      static_cast<size_t>(model_width) * model_width;
  ASSIGN_OR_RETURN(auto matrix,
                   Buffer::Allocate(matrix_elements * sizeof(float), stream));
  ASSIGN_OR_RETURN(
      auto bias, Buffer::Allocate(model_width * sizeof(float), stream));
  RETURN_IF_ERROR(CudaStatus(
      cudaMemsetAsync(matrix.data(), 0, matrix.size_bytes(), stream),
      "cudaMemsetAsync(dense matrix)"));
  RETURN_IF_ERROR(CudaStatus(
      cudaMemsetAsync(bias.data(), 0, bias.size_bytes(), stream),
      "cudaMemsetAsync(dense bias)"));
  return std::unique_ptr<FullyConnectedLayer>(new FullyConnectedLayer(
      model_width, data_type, learning_rate, stream, std::move(matrix),
      std::move(bias)));
}

absl::Status FullyConnectedLayer::InitializeIdentity(float scale) {
  std::vector<float> identity(
      static_cast<size_t>(model_width_) * model_width_, 0.0f);
  for (int index = 0; index < model_width_; ++index) {
    identity[static_cast<size_t>(index) * model_width_ + index] = scale;
  }
  return CudaStatus(cudaMemcpyAsync(weights_[0].data(), identity.data(),
                                    weights_[0].size_bytes(),
                                    cudaMemcpyHostToDevice, stream_),
                    "cudaMemcpyAsync(identity matrix)");
}

absl::StatusOr<Buffer> FullyConnectedLayer::fwd(
    absl::Span<const Buffer> inputs, Tape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "FullyConnectedLayer fwd expects one input and a non-null tape");
  }
  ASSIGN_OR_RETURN(
      int rows, MatrixRows(inputs[0], model_width_, stream_, "dense input"));
  RETURN_IF_ERROR(ValidateTiledExtent(rows, "dense rows"));
  ASSIGN_OR_RETURN(auto output,
                   Buffer::Allocate(inputs[0].size_bytes(), stream_));
  tape->intermediates = {inputs[0]};
  tape->children.clear();
  DenseForwardKernel<<<TileCount(rows) * TileCount(model_width_), 1, 0,
                       stream_>>>(
      static_cast<const float*>(inputs[0].data()),
      static_cast<const float*>(weights_[0].data()),
      static_cast<const float*>(weights_[1].data()),
      rows, model_width_,
      static_cast<float*>(output.data()));
  RETURN_IF_ERROR(
      CudaStatus(cudaGetLastError(), "DenseForwardKernel launch"));
  return std::move(output);
}

absl::StatusOr<BufferVec> FullyConnectedLayer::bwd(
    absl::Span<const Buffer> output_gradients, Tape tape) {
  if (output_gradients.size() != 1 || tape.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "FullyConnectedLayer bwd received an incompatible gradient or tape");
  }
  ASSIGN_OR_RETURN(int rows,
                   MatrixRows(output_gradients[0], model_width_, stream_,
                              "dense output gradient"));
  RETURN_IF_ERROR(ValidateBuffer(tape.intermediates[0],
                                 output_gradients[0].size_bytes(), stream_,
                                 "dense saved input"));
  ASSIGN_OR_RETURN(
      auto input_gradient,
      Buffer::Allocate(output_gradients[0].size_bytes(), stream_));
  DenseInputGradientKernel<<<TileCount(rows) * TileCount(model_width_), 1, 0,
                             stream_>>>(
      static_cast<const float*>(output_gradients[0].data()),
      static_cast<const float*>(weights_[0].data()),
      rows, model_width_,
      static_cast<float*>(input_gradient.data()));
  if (learning_rate_ != 0.0f) {
    DenseWeightUpdateKernel<<<TileCount(model_width_) * TileCount(model_width_),
                              1, 0, stream_>>>(
        static_cast<const float*>(tape.intermediates[0].data()),
        static_cast<const float*>(output_gradients[0].data()), learning_rate_,
        rows, model_width_,
        static_cast<float*>(weights_[0].data()));
    DenseBiasUpdateKernel<<<TileCount(model_width_), 1, 0, stream_>>>(
        static_cast<const float*>(output_gradients[0].data()), learning_rate_,
        rows, model_width_,
        static_cast<float*>(weights_[1].data()));
  }
  RETURN_IF_ERROR(
      CudaStatus(cudaGetLastError(), "dense backward kernel launch"));
  return BufferVec{std::move(input_gradient)};
}


}  // namespace pluto::llm
