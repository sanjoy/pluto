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
#include "src/llm/layers/internal.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

template <class Activation>
using MmaType = std::conditional_t<std::is_same_v<Activation, float>, __half,
                                   __nv_bfloat16>;

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
                         ct::shape{16_ic, 16_ic}};
  auto matrix_view = ct::partition_view{
      ct::tensor_span{matrix, ct::extents{input_dim, output_dim}},
      ct::shape{16_ic, 16_ic}};
  auto bias_view = ct::partition_view{
      ct::tensor_span{bias, ct::extents{output_dim}}, ct::shape{16_ic}};
  auto output_view =
      ct::partition_view{ct::tensor_span{output, ct::extents{rows, output_dim}},
                         ct::shape{16_ic, 16_ic}};
  const int output_tiles = output_dim / internal::kDenseTile;
  const int input_tiles = input_dim / internal::kDenseTile;
  const int block = ct::bid().x;
  const int row_tile = block / output_tiles;
  const int output_tile = block % output_tiles;
  auto accumulator =
      ct::broadcast(bias_view.load(output_tile), ct::shape{16_ic, 16_ic});
  for (int input_tile = 0; input_tile < input_tiles; ++input_tile) {
    auto left = ct::element_cast<MmaType<Activation>>(
        input_view.load(row_tile, input_tile));
    auto right = ct::element_cast<MmaType<Activation>>(
        matrix_view.load(input_tile, output_tile));
    accumulator = ct::mma(left, right, accumulator);
  }
  output_view.store(ct::element_cast<Activation>(accumulator), row_tile,
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
      ct::shape{16_ic, 16_ic}};
  auto matrix_view = ct::partition_view{
      ct::tensor_span{matrix, ct::extents{input_dim, output_dim}},
      ct::shape{16_ic, 16_ic}};
  auto input_gradient_view = ct::partition_view{
      ct::tensor_span{input_gradient, ct::extents{rows, input_dim}},
      ct::shape{16_ic, 16_ic}};
  const int input_tiles = input_dim / internal::kDenseTile;
  const int output_tiles = output_dim / internal::kDenseTile;
  const int block = ct::bid().x;
  const int row_tile = block / input_tiles;
  const int input_tile = block % input_tiles;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  for (int output_tile = 0; output_tile < output_tiles; ++output_tile) {
    auto gradient = ct::element_cast<MmaType<Activation>>(
        gradient_view.load(row_tile, output_tile));
    auto matrix_transposed =
        ct::transpose(ct::element_cast<MmaType<Activation>>(
            matrix_view.load(input_tile, output_tile)));
    accumulator = ct::mma(gradient, matrix_transposed, accumulator);
  }
  input_gradient_view.store(accumulator, row_tile, input_tile);
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
                         ct::shape{16_ic, 16_ic}};
  auto output_gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, output_dim}},
      ct::shape{16_ic, 16_ic}};
  auto matrix_gradient_view = ct::partition_view{
      ct::tensor_span{matrix_gradient, ct::extents{input_dim, output_dim}},
      ct::shape{16_ic, 16_ic}};
  const int output_tiles = output_dim / internal::kDenseTile;
  const int row_tiles = rows / internal::kDenseTile;
  const int block = ct::bid().x;
  const int input_tile = block / output_tiles;
  const int output_tile = block % output_tiles;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
    auto input_transposed = ct::transpose(ct::element_cast<MmaType<Activation>>(
        input_view.load(row_tile, input_tile)));
    auto gradient = ct::element_cast<MmaType<Activation>>(
        output_gradient_view.load(row_tile, output_tile));
    accumulator = ct::mma(input_transposed, gradient, accumulator);
  }
  matrix_gradient_view.store(accumulator, input_tile, output_tile);
}

__tile_global__ void DenseBiasGradientKernel(
    const float* __restrict__ output_gradient, int rows, int output_dim,
    float* __restrict__ bias_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, output_dim}},
      ct::shape{16_ic, 16_ic}};
  auto bias_gradient_view = ct::partition_view{
      ct::tensor_span{bias_gradient, ct::extents{output_dim}},
      ct::shape{16_ic}};
  const int row_tiles = rows / internal::kDenseTile;
  const int output_tile = ct::bid().x;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<1, 16>>>();
  for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
    accumulator =
        accumulator + ct::sum(gradient_view.load(row_tile, output_tile), 0_ic);
  }
  bias_gradient_view.store(ct::reshape(accumulator, ct::shape{16_ic}),
                           output_tile);
}

}  // namespace

FullyConnectedLayer::FullyConnectedLayer(cuda::Executor& executor,
                                         int input_dim, int output_dim,
                                         DataType data_type, Buffer matrix,
                                         Buffer bias, Buffer matrix_gradient,
                                         Buffer bias_gradient)
    : input_dim_(input_dim),
      output_dim_(output_dim),
      output_type_(data_type),
      executor_(executor),
      weights_{std::move(matrix), std::move(bias)},
      gradients_{std::move(matrix_gradient), std::move(bias_gradient)} {}

absl::StatusOr<std::unique_ptr<FullyConnectedLayer>>
FullyConnectedLayer::Create(cuda::Executor& executor, int input_dim,
                            int output_dim, DataType data_type) {
  RETURN_IF_ERROR(internal::ValidateComputeType(data_type));
  RETURN_IF_ERROR(internal::ValidateTiledExtent(input_dim, "input_dim"));
  RETURN_IF_ERROR(internal::ValidateTiledExtent(output_dim, "output_dim"));
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
      std::move(bias), std::move(matrix_gradient), std::move(bias_gradient)));
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
  for (float& value : matrix)
    value = distribution(random);
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(weights_[0].data(), matrix.data(),
                      weights_[0].size_bytes(), cudaMemcpyHostToDevice,
                      executor_.stream()),
      "cudaMemcpyAsync(normal matrix)"));
  return absl::OkStatus();
}

absl::StatusOr<Buffer> FullyConnectedLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    Tape* tape) const {
  RETURN_IF_ERROR(
      internal::ValidateExecutor(executor_, executor, "FullyConnectedLayer"));
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "FullyConnectedLayer fwd expects one input and a non-null tape");
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
  tape->intermediates = {inputs[0]};
  tape->children.clear();
  const int blocks =
      internal::TileCount(rows) * internal::TileCount(output_dim_);
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
  return std::move(output);
}

absl::StatusOr<BufferVec> FullyConnectedLayer::bwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    Tape tape) {
  RETURN_IF_ERROR(
      internal::ValidateExecutor(executor_, executor, "FullyConnectedLayer"));
  if (output_gradients.size() != 1 || tape.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "FullyConnectedLayer bwd received an incompatible gradient or tape");
  }
  ASSIGN_OR_RETURN(
      int rows, internal::MatrixRows(executor, output_gradients[0], output_dim_,
                                     "dense output gradient"));
  RETURN_IF_ERROR(internal::ValidateBuffer(
      executor, tape.intermediates[0],
      static_cast<size_t>(rows) * input_dim_ *
          internal::ActivationElementBytes(output_type_),
      "dense saved input"));
  ASSIGN_OR_RETURN(auto input_gradient,
                   Buffer::Allocate(executor, static_cast<size_t>(rows) *
                                                  input_dim_ * sizeof(float)));
  const int input_blocks =
      internal::TileCount(rows) * internal::TileCount(input_dim_);
  const int weight_blocks =
      internal::TileCount(input_dim_) * internal::TileCount(output_dim_);
  if (output_type_ == DataType::BF16) {
    DenseInputGradientKernel<__nv_bfloat16>
        <<<input_blocks, 1, 0, executor.stream()>>>(
            static_cast<const float*>(output_gradients[0].data()),
            static_cast<const float*>(weights_[0].data()), rows, input_dim_,
            output_dim_, static_cast<float*>(input_gradient.data()));
    DenseWeightGradientKernel<__nv_bfloat16>
        <<<weight_blocks, 1, 0, executor.stream()>>>(
            static_cast<const __nv_bfloat16*>(tape.intermediates[0].data()),
            static_cast<const float*>(output_gradients[0].data()), rows,
            input_dim_, output_dim_, static_cast<float*>(gradients_[0].data()));
  } else {
    DenseInputGradientKernel<float><<<input_blocks, 1, 0, executor.stream()>>>(
        static_cast<const float*>(output_gradients[0].data()),
        static_cast<const float*>(weights_[0].data()), rows, input_dim_,
        output_dim_, static_cast<float*>(input_gradient.data()));
    DenseWeightGradientKernel<float>
        <<<weight_blocks, 1, 0, executor.stream()>>>(
            static_cast<const float*>(tape.intermediates[0].data()),
            static_cast<const float*>(output_gradients[0].data()), rows,
            input_dim_, output_dim_, static_cast<float*>(gradients_[0].data()));
  }
  DenseBiasGradientKernel<<<internal::TileCount(output_dim_), 1, 0,
                            executor.stream()>>>(
      static_cast<const float*>(output_gradients[0].data()), rows, output_dim_,
      static_cast<float*>(gradients_[1].data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "dense backward kernel launch"));
  return BufferVec{std::move(input_gradient)};
}

}  // namespace pluto::llm
