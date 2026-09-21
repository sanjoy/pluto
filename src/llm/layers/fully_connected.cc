#include "src/llm/layers/fully_connected.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/buffer.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layers/util.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

template <class Activation>
using MmaType = std::conditional_t<std::is_same_v<Activation, float>, __half,
                                   __nv_bfloat16>;

// Larger compute tiles amortize loads and MMA dispatch without changing the
// public 16-element extent contract. Masked views cover every partial tile.
constexpr int kMatrixTile = 64;
constexpr int kBiasRows = 256;

int MatrixTileCount(int extent) {
  return (extent + kMatrixTile - 1) / kMatrixTile;
}

template <class Activation>
__tile_global__ void DenseForwardKernel(const Activation* __restrict__ input,
                                        const float* __restrict__ matrix,
                                        const float* __restrict__ bias,
                                        int rows, int input_dim, int output_dim,
                                        Activation* __restrict__ output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto input_view =
      ct::partition_view{ct::tensor_span{input, ct::extents{rows, input_dim}},
                         ct::shape{64_ic, 64_ic}};
  auto matrix_view = ct::partition_view{
      ct::tensor_span{matrix, ct::extents{input_dim, output_dim}},
      ct::shape{64_ic, 64_ic}};
  auto bias_view = ct::partition_view{
      ct::tensor_span{bias, ct::extents{output_dim}}, ct::shape{64_ic}};
  auto output_view =
      ct::partition_view{ct::tensor_span{output, ct::extents{rows, output_dim}},
                         ct::shape{64_ic, 64_ic}};
  const int output_tiles = (output_dim + kMatrixTile - 1) / kMatrixTile;
  const int input_tiles = (input_dim + kMatrixTile - 1) / kMatrixTile;
  const int block = ct::bid().x;
  const int row_tile = block / output_tiles;
  const int output_tile = block % output_tiles;
  auto accumulator = ct::broadcast(bias_view.load_masked(output_tile),
                                   ct::shape{64_ic, 64_ic});
  for (int input_tile = 0; input_tile < input_tiles; ++input_tile) {
    auto left = ct::element_cast<MmaType<Activation>>(
        input_view.load_masked(row_tile, input_tile));
    auto right = ct::element_cast<MmaType<Activation>>(
        matrix_view.load_masked(input_tile, output_tile));
    accumulator = ct::mma(left, right, accumulator);
  }
  output_view.store_masked(ct::element_cast<Activation>(accumulator), row_tile,
                           output_tile);
}

template <class Activation>
__tile_global__ void DenseInputGradientKernel(
    const float* __restrict__ output_gradient, const float* __restrict__ matrix,
    int rows, int input_dim, int output_dim,
    float* __restrict__ input_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, output_dim}},
      ct::shape{64_ic, 64_ic}};
  auto matrix_view = ct::partition_view{
      ct::tensor_span{matrix, ct::extents{input_dim, output_dim}},
      ct::shape{64_ic, 64_ic}};
  auto input_gradient_view = ct::partition_view{
      ct::tensor_span{input_gradient, ct::extents{rows, input_dim}},
      ct::shape{64_ic, 64_ic}};
  const int input_tiles = (input_dim + kMatrixTile - 1) / kMatrixTile;
  const int output_tiles = (output_dim + kMatrixTile - 1) / kMatrixTile;
  const int block = ct::bid().x;
  const int row_tile = block / input_tiles;
  const int input_tile = block % input_tiles;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<64, 64>>>();
  for (int output_tile = 0; output_tile < output_tiles; ++output_tile) {
    auto gradient = ct::element_cast<MmaType<Activation>>(
        gradient_view.load_masked(row_tile, output_tile));
    auto matrix_transposed =
        ct::transpose(ct::element_cast<MmaType<Activation>>(
            matrix_view.load_masked(input_tile, output_tile)));
    accumulator = ct::mma(gradient, matrix_transposed, accumulator);
  }
  input_gradient_view.store_masked(accumulator, row_tile, input_tile);
}

template <class Activation>
__tile_global__ void DenseWeightGradientKernel(
    const Activation* __restrict__ input,
    const float* __restrict__ output_gradient, int rows, int input_dim,
    int output_dim, float* __restrict__ matrix_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto input_view =
      ct::partition_view{ct::tensor_span{input, ct::extents{rows, input_dim}},
                         ct::shape{64_ic, 64_ic}};
  auto output_gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, output_dim}},
      ct::shape{64_ic, 64_ic}};
  auto matrix_gradient_view = ct::partition_view{
      ct::tensor_span{matrix_gradient, ct::extents{input_dim, output_dim}},
      ct::shape{64_ic, 64_ic}};
  const int output_tiles = (output_dim + kMatrixTile - 1) / kMatrixTile;
  const int row_tiles = (rows + kMatrixTile - 1) / kMatrixTile;
  const int block = ct::bid().x;
  const int input_tile = block / output_tiles;
  const int output_tile = block % output_tiles;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<64, 64>>>();
  for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
    auto input_transposed = ct::transpose(ct::element_cast<MmaType<Activation>>(
        input_view.load_masked(row_tile, input_tile)));
    auto gradient = ct::element_cast<MmaType<Activation>>(
        output_gradient_view.load_masked(row_tile, output_tile));
    accumulator = ct::mma(input_transposed, gradient, accumulator);
  }
  matrix_gradient_view.store_masked(accumulator, input_tile, output_tile);
}

__tile_global__ void DenseBiasPartialGradientKernel(
    const float* __restrict__ output_gradient, int rows, int output_dim,
    int row_tiles, float* __restrict__ partial_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, output_dim}},
      ct::shape{256_ic, 64_ic}};
  auto partial_view = ct::partition_view{
      ct::tensor_span{partial_gradient, ct::extents{row_tiles, output_dim}},
      ct::shape{1_ic, 64_ic}};
  const int output_tiles = (output_dim + kMatrixTile - 1) / kMatrixTile;
  const int row_tile = ct::bid().x / output_tiles;
  const int output_tile = ct::bid().x % output_tiles;
  partial_view.store_masked(
      ct::sum(gradient_view.load_masked(row_tile, output_tile), 0_ic), row_tile,
      output_tile);
}

__tile_global__ void DenseBiasGradientKernel(
    const float* __restrict__ output_gradient, int rows, int output_dim,
    float* __restrict__ bias_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, output_dim}},
      ct::shape{256_ic, 64_ic}};
  auto bias_gradient_view = ct::partition_view{
      ct::tensor_span{bias_gradient, ct::extents{output_dim}},
      ct::shape{64_ic}};
  const int row_tiles = (rows + kBiasRows - 1) / kBiasRows;
  const int output_tile = ct::bid().x;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<1, 64>>>();
  for (int row_tile = 0; row_tile < row_tiles; ++row_tile)
    accumulator =
        accumulator +
        ct::sum(gradient_view.load_masked(row_tile, output_tile), 0_ic);
  bias_gradient_view.store_masked(ct::reshape(accumulator, ct::shape{64_ic}),
                                  output_tile);
}

}  // namespace

FullyConnectedLayer::FullyConnectedLayer(cuda::Executor& executor,
                                         int input_dim, int output_dim,
                                         DataType data_type, Buffer matrix,
                                         Buffer bias, Buffer matrix_gradient,
                                         Buffer bias_gradient,
                                         int sequence_length)
    : input_dim_(input_dim),
      output_dim_(output_dim),
      sequence_length_(sequence_length),
      output_type_(data_type),
      executor_(executor),
      weights_{std::move(matrix), std::move(bias)},
      gradients_{std::move(matrix_gradient), std::move(bias_gradient)} {}

absl::StatusOr<std::unique_ptr<FullyConnectedLayer>>
FullyConnectedLayer::Create(cuda::Executor& executor, int input_dim,
                            int output_dim, DataType data_type,
                            int sequence_length) {
  if (sequence_length <= 0)
    return absl::InvalidArgumentError("sequence_length must be positive");
  RETURN_IF_ERROR(internal::ValidateComputeType(data_type));
  RETURN_IF_ERROR(internal::ValidatePositiveExtent(input_dim, "input_dim"));
  RETURN_IF_ERROR(internal::ValidatePositiveExtent(output_dim, "output_dim"));
  const size_t matrix_bytes =
      static_cast<size_t>(input_dim) * output_dim * sizeof(float);
  const size_t bias_bytes = static_cast<size_t>(output_dim) * sizeof(float);
  ASSIGN_OR_RETURN(auto matrix, Buffer::Allocate(executor, matrix_bytes));
  ASSIGN_OR_RETURN(auto bias, Buffer::Allocate(executor, bias_bytes));
  ASSIGN_OR_RETURN(auto matrix_gradient,
                   Buffer::Allocate(executor, matrix_bytes));
  ASSIGN_OR_RETURN(auto bias_gradient, Buffer::Allocate(executor, bias_bytes));
  for (Buffer* buffer : {&matrix, &bias, &matrix_gradient, &bias_gradient}) {
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemsetAsync(buffer->data(), 0, buffer->size_bytes(),
                        executor.stream()),
        "cudaMemsetAsync(dense parameter)"));
  }
  return absl::WrapUnique(new FullyConnectedLayer(
      executor, input_dim, output_dim, data_type, std::move(matrix),
      std::move(bias), std::move(matrix_gradient), std::move(bias_gradient),
      sequence_length));
}

absl::Status FullyConnectedLayer::InitializeIdentity(float scale) {
  ASSIGN_OR_RETURN(auto matrix, cuda::PageLockedHostArray<float>::Allocate(
                                    executor_, static_cast<size_t>(input_dim_) *
                                                   output_dim_));
  std::fill(matrix.begin(), matrix.end(), 0.0f);
  for (int index = 0; index < std::min(input_dim_, output_dim_); ++index)
    matrix[static_cast<size_t>(index) * output_dim_ + index] = scale;
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(weights_[0].data(), matrix.data(),
                      weights_[0].size_bytes(), cudaMemcpyHostToDevice,
                      executor_.stream()),
      "cudaMemcpyAsync(identity matrix)"));
  // Pinned staging destruction queues its release after this upload.
  return absl::OkStatus();
}

absl::Status FullyConnectedLayer::InitializeNormal(float standard_deviation,
                                                   uint64_t seed) {
  if (!(standard_deviation > 0.0f)) {
    return absl::InvalidArgumentError(
        "dense initialization standard deviation must be positive");
  }
  std::mt19937_64 random(seed);
  std::normal_distribution<float> distribution(0.0f, standard_deviation);
  ASSIGN_OR_RETURN(auto matrix, cuda::PageLockedHostArray<float>::Allocate(
                                    executor_, static_cast<size_t>(input_dim_) *
                                                   output_dim_));
  for (float& value : matrix) value = distribution(random);
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(weights_[0].data(), matrix.data(),
                      weights_[0].size_bytes(), cudaMemcpyHostToDevice,
                      executor_.stream()),
      "cudaMemcpyAsync(normal matrix)"));
  return absl::OkStatus();
}

absl::StatusOr<FwdResult> FullyConnectedLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks*) const {
  BackwardState state;
  RETURN_IF_ERROR(
      internal::ValidateExecutor(executor_, executor, "FullyConnectedLayer"));
  if (inputs.size() != 1) {
    return absl::InvalidArgumentError(
        "FullyConnectedLayer fwd expects one input");
  }
  ASSIGN_OR_RETURN(int rows,
                   internal::ActivationRows(executor, inputs[0], input_dim_,
                                            output_type_, "dense input"));
  RETURN_IF_ERROR(internal::ValidateTiledExtent(rows, "dense rows"));
  ASSIGN_OR_RETURN(
      auto output,
      Buffer::Allocate(executor,
                       static_cast<size_t>(rows) * output_dim_ *
                           internal::ActivationElementBytes(output_type_)));
  state.intermediates = {inputs[0]};
  state.children.clear();
  const int blocks = MatrixTileCount(rows) * MatrixTileCount(output_dim_);
  if (output_type_ == DataType::BF16) {
    DenseForwardKernel<__nv_bfloat16><<<blocks, 1, 0, executor.stream()>>>(
        static_cast<const __nv_bfloat16*>(inputs[0].data()),
        static_cast<const float*>(weights_[0].data()),
        static_cast<const float*>(weights_[1].data()), rows, input_dim_,
        output_dim_, static_cast<__nv_bfloat16*>(output.data()));
  } else {
    DenseForwardKernel<float><<<blocks, 1, 0, executor.stream()>>>(
        static_cast<const float*>(inputs[0].data()),
        static_cast<const float*>(weights_[0].data()),
        static_cast<const float*>(weights_[1].data()), rows, input_dim_,
        output_dim_, static_cast<float*>(output.data()));
  }
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "DenseForwardKernel launch"));
  return FwdResult{{std::move(output)}, std::move(state)};
}

absl::StatusOr<BufferVec> FullyConnectedLayer::bwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    BackwardState state, LayerHooks*) {
  RETURN_IF_ERROR(
      internal::ValidateExecutor(executor_, executor, "FullyConnectedLayer"));
  if (output_gradients.size() != 1 || state.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "FullyConnectedLayer bwd received an incompatible gradient or state");
  }
  ASSIGN_OR_RETURN(
      int rows, internal::MatrixRows(executor, output_gradients[0], output_dim_,
                                     "dense output gradient"));
  RETURN_IF_ERROR(internal::ValidateBuffer(
      executor, state.intermediates[0],
      static_cast<size_t>(rows) * input_dim_ *
          internal::ActivationElementBytes(output_type_),
      "dense saved input"));
  ASSIGN_OR_RETURN(auto input_gradient,
                   Buffer::Allocate(executor, static_cast<size_t>(rows) *
                                                  input_dim_ * sizeof(float)));
  const int input_blocks = MatrixTileCount(rows) * MatrixTileCount(input_dim_);
  const int weight_blocks =
      MatrixTileCount(input_dim_) * MatrixTileCount(output_dim_);
  if (output_type_ == DataType::BF16) {
    DenseInputGradientKernel<__nv_bfloat16>
        <<<input_blocks, 1, 0, executor.stream()>>>(
            static_cast<const float*>(output_gradients[0].data()),
            static_cast<const float*>(weights_[0].data()), rows, input_dim_,
            output_dim_, static_cast<float*>(input_gradient.data()));
    DenseWeightGradientKernel<__nv_bfloat16>
        <<<weight_blocks, 1, 0, executor.stream()>>>(
            static_cast<const __nv_bfloat16*>(state.intermediates[0].data()),
            static_cast<const float*>(output_gradients[0].data()), rows,
            input_dim_, output_dim_, static_cast<float*>(gradients_[0].data()));
  } else {
    DenseInputGradientKernel<float><<<input_blocks, 1, 0, executor.stream()>>>(
        static_cast<const float*>(output_gradients[0].data()),
        static_cast<const float*>(weights_[0].data()), rows, input_dim_,
        output_dim_, static_cast<float*>(input_gradient.data()));
    DenseWeightGradientKernel<float>
        <<<weight_blocks, 1, 0, executor.stream()>>>(
            static_cast<const float*>(state.intermediates[0].data()),
            static_cast<const float*>(output_gradients[0].data()), rows,
            input_dim_, output_dim_, static_cast<float*>(gradients_[0].data()));
  }
  const int bias_blocks = MatrixTileCount(output_dim_);
  if (rows > kBiasRows) {
    const int partial_rows = (rows + kBiasRows - 1) / kBiasRows;
    ASSIGN_OR_RETURN(
        auto partial_gradient,
        Buffer::Allocate(executor, static_cast<size_t>(partial_rows) *
                                       output_dim_ * sizeof(float)));
    DenseBiasPartialGradientKernel<<<partial_rows * bias_blocks, 1, 0,
                                     executor.stream()>>>(
        static_cast<const float*>(output_gradients[0].data()), rows,
        output_dim_, partial_rows,
        static_cast<float*>(partial_gradient.data()));
    // Fixed row partitions and a fixed reduction order keep repeated backward
    // passes bitwise stable without contended floating-point atomics.
    DenseBiasGradientKernel<<<bias_blocks, 1, 0, executor.stream()>>>(
        static_cast<const float*>(partial_gradient.data()), partial_rows,
        output_dim_, static_cast<float*>(gradients_[1].data()));
  } else {
    DenseBiasGradientKernel<<<bias_blocks, 1, 0, executor.stream()>>>(
        static_cast<const float*>(output_gradients[0].data()), rows,
        output_dim_, static_cast<float*>(gradients_[1].data()));
  }
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "dense backward kernel launch"));
  return BufferVec{std::move(input_gradient)};
}

}  // namespace pluto::llm
