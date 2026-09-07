#include "src/llm/layers/norm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layers/internal.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

template <class Activation>
__tile_global__ void LayerNormForwardKernel(
    const Activation* __restrict__ input, const float* __restrict__ gamma,
    const float* __restrict__ beta, int rows, int embedding_dim, float epsilon,
    Activation* __restrict__ output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  auto gamma_view = ct::partition_view{
      ct::tensor_span{gamma, ct::extents{embedding_dim}}, ct::shape{16_ic}};
  auto beta_view = ct::partition_view{
      ct::tensor_span{beta, ct::extents{embedding_dim}}, ct::shape{16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  const int width_tiles = embedding_dim / internal::kDenseTile;
  const int block = ct::bid().x;
  const int row = block / width_tiles;
  const int output_tile = block % width_tiles;
  auto mean = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  for (int tile = 0; tile < width_tiles; ++tile) {
    mean = mean +
           ct::sum(ct::element_cast<float>(input_view.load(row, tile)), 1_ic);
  }
  mean = mean / static_cast<float>(embedding_dim);
  auto variance = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  for (int tile = 0; tile < width_tiles; ++tile) {
    auto centered = ct::element_cast<float>(input_view.load(row, tile)) - mean;
    variance = variance + ct::sum(centered * centered, 1_ic);
  }
  variance = variance / static_cast<float>(embedding_dim);
  auto normalized =
      (ct::element_cast<float>(input_view.load(row, output_tile)) - mean) *
      ct::rsqrt(variance + epsilon);
  auto affine =
      normalized * gamma_view.load(output_tile) + beta_view.load(output_tile);
  output_view.store(ct::element_cast<Activation>(affine), row, output_tile);
}

template <class Activation>
__tile_global__ void LayerNormInputGradientKernel(
    const Activation* __restrict__ input, const float* __restrict__ gamma,
    const float* __restrict__ output_gradient, int rows, int embedding_dim,
    float epsilon, float* __restrict__ input_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  auto gamma_view = ct::partition_view{
      ct::tensor_span{gamma, ct::extents{embedding_dim}}, ct::shape{16_ic}};
  auto output_gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  auto input_gradient_view = ct::partition_view{
      ct::tensor_span{input_gradient, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  const int width_tiles = embedding_dim / internal::kDenseTile;
  const int block = ct::bid().x;
  const int row = block / width_tiles;
  const int output_tile = block % width_tiles;
  auto mean = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  for (int tile = 0; tile < width_tiles; ++tile) {
    mean = mean +
           ct::sum(ct::element_cast<float>(input_view.load(row, tile)), 1_ic);
  }
  mean = mean / static_cast<float>(embedding_dim);
  auto variance = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  for (int tile = 0; tile < width_tiles; ++tile) {
    auto centered = ct::element_cast<float>(input_view.load(row, tile)) - mean;
    variance = variance + ct::sum(centered * centered, 1_ic);
  }
  auto inverse_stddev =
      ct::rsqrt(variance / static_cast<float>(embedding_dim) + epsilon);
  auto gradient_sum = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  auto projected_sum = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  for (int tile = 0; tile < width_tiles; ++tile) {
    auto normalized =
        (ct::element_cast<float>(input_view.load(row, tile)) - mean) *
        inverse_stddev;
    auto d_normalized =
        output_gradient_view.load(row, tile) * gamma_view.load(tile);
    gradient_sum = gradient_sum + ct::sum(d_normalized, 1_ic);
    projected_sum = projected_sum + ct::sum(d_normalized * normalized, 1_ic);
  }
  auto normalized =
      (ct::element_cast<float>(input_view.load(row, output_tile)) - mean) *
      inverse_stddev;
  auto d_normalized = output_gradient_view.load(row, output_tile) *
                      gamma_view.load(output_tile);
  input_gradient_view.store(
      inverse_stddev *
          (d_normalized - gradient_sum / static_cast<float>(embedding_dim) -
           normalized * projected_sum / static_cast<float>(embedding_dim)),
      row, output_tile);
}

template <class Activation>
__tile_global__ void LayerNormParameterGradientKernel(
    const Activation* __restrict__ input,
    const float* __restrict__ output_gradient, int rows, int embedding_dim,
    float epsilon, float* __restrict__ gamma_gradient,
    float* __restrict__ beta_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  auto output_gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  auto gamma_gradient_view = ct::partition_view{
      ct::tensor_span{gamma_gradient, ct::extents{embedding_dim}},
      ct::shape{16_ic}};
  auto beta_gradient_view = ct::partition_view{
      ct::tensor_span{beta_gradient, ct::extents{embedding_dim}},
      ct::shape{16_ic}};
  const int width_tiles = embedding_dim / internal::kDenseTile;
  const int output_tile = ct::bid().x;
  auto d_gamma = ct::zeros<ct::tile<float, ct::shape<1, 16>>>();
  auto d_beta = ct::zeros<ct::tile<float, ct::shape<1, 16>>>();
  for (int row = 0; row < rows; ++row) {
    auto mean = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
    for (int tile = 0; tile < width_tiles; ++tile) {
      mean = mean +
             ct::sum(ct::element_cast<float>(input_view.load(row, tile)), 1_ic);
    }
    mean = mean / static_cast<float>(embedding_dim);
    auto variance = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
    for (int tile = 0; tile < width_tiles; ++tile) {
      auto centered =
          ct::element_cast<float>(input_view.load(row, tile)) - mean;
      variance = variance + ct::sum(centered * centered, 1_ic);
    }
    auto normalized =
        (ct::element_cast<float>(input_view.load(row, output_tile)) - mean) *
        ct::rsqrt(variance / static_cast<float>(embedding_dim) + epsilon);
    auto gradient = output_gradient_view.load(row, output_tile);
    d_gamma = d_gamma + gradient * normalized;
    d_beta = d_beta + gradient;
  }
  gamma_gradient_view.store(ct::reshape(d_gamma, ct::shape{16_ic}),
                            output_tile);
  beta_gradient_view.store(ct::reshape(d_beta, ct::shape{16_ic}), output_tile);
}

}  // namespace

absl::StatusOr<std::unique_ptr<LayerNormLayer>> LayerNormLayer::Create(
    cuda::Executor& executor, int embedding_dim, float epsilon,
    DataType data_type) {
  RETURN_IF_ERROR(internal::ValidateComputeType(data_type));
  if (!(epsilon > 0.0f)) {
    return absl::InvalidArgumentError("layer-norm epsilon must be positive");
  }
  RETURN_IF_ERROR(
      internal::ValidateTiledExtent(embedding_dim, "embedding_dim"));
  const size_t bytes = static_cast<size_t>(embedding_dim) * sizeof(float);
  ASSIGN_OR_RETURN(auto gamma, Buffer::Allocate(executor, bytes));
  ASSIGN_OR_RETURN(auto beta, Buffer::Allocate(executor, bytes));
  ASSIGN_OR_RETURN(auto gamma_gradient, Buffer::Allocate(executor, bytes));
  ASSIGN_OR_RETURN(auto beta_gradient, Buffer::Allocate(executor, bytes));
  ASSIGN_OR_RETURN(auto gamma_values,
                   cuda::PageLockedHostArray<float>::Allocate(embedding_dim));
  std::fill(gamma_values.begin(), gamma_values.end(), 1.0f);
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(gamma.data(), gamma_values.data(), bytes,
                      cudaMemcpyHostToDevice, executor.stream()),
      "cudaMemcpyAsync(layer-norm gamma)"));
  for (Buffer* buffer : {&beta, &gamma_gradient, &beta_gradient}) {
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemsetAsync(buffer->data(), 0, buffer->size_bytes(),
                        executor.stream()),
        "cudaMemsetAsync(layer-norm parameter)"));
  }
  // gamma_values owns the source of an asynchronous transfer.
  RETURN_IF_ERROR(executor.Synchronize());
  return std::unique_ptr<LayerNormLayer>(new LayerNormLayer(
      executor, embedding_dim, epsilon, data_type, std::move(gamma),
      std::move(beta), std::move(gamma_gradient), std::move(beta_gradient)));
}

absl::StatusOr<Buffer> LayerNormLayer::fwd(cuda::Executor& executor,
                                           absl::Span<const Buffer> inputs,
                                           Tape* tape) const {
  RETURN_IF_ERROR(
      internal::ValidateExecutor(executor_, executor, "LayerNormLayer"));
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "LayerNormLayer fwd expects one input and a non-null tape");
  }
  ASSIGN_OR_RETURN(int rows,
                   internal::ActivationRows(executor, inputs[0], embedding_dim_,
                                            output_type_, "layer-norm input"));
  ASSIGN_OR_RETURN(auto output,
                   Buffer::Allocate(executor, inputs[0].size_bytes()));
  const int blocks = rows * internal::TileCount(embedding_dim_);
  if (output_type_ == DataType::BF16) {
    LayerNormForwardKernel<__nv_bfloat16><<<blocks, 1, 0, executor.stream()>>>(
        static_cast<const __nv_bfloat16*>(inputs[0].data()),
        static_cast<const float*>(weights_[0].data()),
        static_cast<const float*>(weights_[1].data()), rows, embedding_dim_,
        epsilon_, static_cast<__nv_bfloat16*>(output.data()));
  } else {
    LayerNormForwardKernel<float><<<blocks, 1, 0, executor.stream()>>>(
        static_cast<const float*>(inputs[0].data()),
        static_cast<const float*>(weights_[0].data()),
        static_cast<const float*>(weights_[1].data()), rows, embedding_dim_,
        epsilon_, static_cast<float*>(output.data()));
  }
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "LayerNormForwardKernel launch"));
  tape->intermediates = {inputs[0]};
  tape->children.clear();
  return std::move(output);
}

absl::StatusOr<BufferVec> LayerNormLayer::bwd(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    Tape tape) {
  RETURN_IF_ERROR(
      internal::ValidateExecutor(executor_, executor, "LayerNormLayer"));
  if (output_gradients.size() != 1 || tape.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "LayerNormLayer bwd received an incompatible gradient or tape");
  }
  ASSIGN_OR_RETURN(int rows, internal::MatrixRows(
                                 executor, output_gradients[0], embedding_dim_,
                                 "layer-norm output gradient"));
  RETURN_IF_ERROR(internal::ValidateBuffer(
      executor, tape.intermediates[0],
      static_cast<size_t>(rows) * embedding_dim_ *
          internal::ActivationElementBytes(output_type_),
      "layer-norm saved input"));
  ASSIGN_OR_RETURN(
      auto input_gradient,
      Buffer::Allocate(executor, static_cast<size_t>(rows) * embedding_dim_ *
                                     sizeof(float)));
  const int blocks = rows * internal::TileCount(embedding_dim_);
  if (output_type_ == DataType::BF16) {
    LayerNormInputGradientKernel<__nv_bfloat16>
        <<<blocks, 1, 0, executor.stream()>>>(
            static_cast<const __nv_bfloat16*>(tape.intermediates[0].data()),
            static_cast<const float*>(weights_[0].data()),
            static_cast<const float*>(output_gradients[0].data()), rows,
            embedding_dim_, epsilon_,
            static_cast<float*>(input_gradient.data()));
    LayerNormParameterGradientKernel<__nv_bfloat16>
        <<<internal::TileCount(embedding_dim_), 1, 0, executor.stream()>>>(
            static_cast<const __nv_bfloat16*>(tape.intermediates[0].data()),
            static_cast<const float*>(output_gradients[0].data()), rows,
            embedding_dim_, epsilon_, static_cast<float*>(gradients_[0].data()),
            static_cast<float*>(gradients_[1].data()));
  } else {
    LayerNormInputGradientKernel<float><<<blocks, 1, 0, executor.stream()>>>(
        static_cast<const float*>(tape.intermediates[0].data()),
        static_cast<const float*>(weights_[0].data()),
        static_cast<const float*>(output_gradients[0].data()), rows,
        embedding_dim_, epsilon_, static_cast<float*>(input_gradient.data()));
    LayerNormParameterGradientKernel<float>
        <<<internal::TileCount(embedding_dim_), 1, 0, executor.stream()>>>(
            static_cast<const float*>(tape.intermediates[0].data()),
            static_cast<const float*>(output_gradients[0].data()), rows,
            embedding_dim_, epsilon_, static_cast<float*>(gradients_[0].data()),
            static_cast<float*>(gradients_[1].data()));
  }
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "layer-norm backward launch"));
  return BufferVec{std::move(input_gradient)};
}

}  // namespace pluto::llm
