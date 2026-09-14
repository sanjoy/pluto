#include "src/llm/layers/norm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layers/util.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

constexpr int kNormRows = 16;
constexpr int kNormChannels = 64;
constexpr int kParameterRows = 128;

template <class T>
__tile__ auto MatrixView(T* data, int rows, int columns) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  return ct::partition_view{ct::tensor_span{data, ct::extents{rows, columns}},
                            ct::shape{16_ic, 64_ic}};
}

template <class T>
__tile__ auto StatisticsView(T* data, int rows) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  return ct::partition_view{ct::tensor_span{data, ct::extents{rows}},
                            ct::shape{16_ic}};
}

template <class T>
__tile__ auto ParameterView(T* data, int columns) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  return ct::partition_view{ct::tensor_span{data, ct::extents{columns}},
                            ct::shape{64_ic}};
}

// Each row's two-pass statistics are computed once and retained for backward.
// Two passes avoid cancellation in E[x*x] - E[x]*E[x] for offset inputs.
template <class Activation>
__tile_global__ void LayerNormStatisticsKernel(
    const Activation* __restrict__ input, int rows, int embedding_dim,
    float epsilon, float* __restrict__ means,
    float* __restrict__ inverse_stddevs) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto input_view = MatrixView(input, rows, embedding_dim);
  const int row_tile = ct::bid().x;
  const int width_tiles = (embedding_dim + kNormChannels - 1) / kNormChannels;
  auto mean = ct::zeros<ct::tile<float, ct::shape<16, 1>>>();
  for (int tile = 0; tile < width_tiles; ++tile)
    mean =
        mean +
        ct::sum(ct::element_cast<float>(input_view.load_masked(row_tile, tile)),
                1_ic);
  mean = mean / static_cast<float>(embedding_dim);
  auto variance = ct::zeros<ct::tile<float, ct::shape<16, 1>>>();
  for (int tile = 0; tile < width_tiles; ++tile) {
    auto centered =
        ct::element_cast<float>(input_view.load_masked(row_tile, tile)) - mean;
    auto columns =
        ct::iota<ct::tile<int, ct::shape<1, 64>>>() + tile * kNormChannels;
    auto square = centered * centered;
    // A masked load's zero padding must not contribute mean*mean to variance.
    variance = variance + ct::sum(ct::select(columns < embedding_dim, square,
                                             ct::zeros<decltype(square)>()),
                                  1_ic);
  }
  auto inverse_stddev =
      ct::rsqrt(variance / static_cast<float>(embedding_dim) + epsilon);
  StatisticsView(means, rows)
      .store_masked(ct::reshape(mean, ct::shape{16_ic}), row_tile);
  StatisticsView(inverse_stddevs, rows)
      .store_masked(ct::reshape(inverse_stddev, ct::shape{16_ic}), row_tile);
}

template <class Activation>
__tile_global__ void LayerNormForwardKernel(
    const Activation* __restrict__ input, const float* __restrict__ gamma,
    const float* __restrict__ beta, const float* __restrict__ means,
    const float* __restrict__ inverse_stddevs, int rows, int embedding_dim,
    Activation* __restrict__ output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  const int width_tiles = (embedding_dim + kNormChannels - 1) / kNormChannels;
  const int row_tile = ct::bid().x / width_tiles;
  const int output_tile = ct::bid().x % width_tiles;
  auto mean = ct::reshape(StatisticsView(means, rows).load_masked(row_tile),
                          ct::shape{16_ic, 1_ic});
  auto inverse_stddev =
      ct::reshape(StatisticsView(inverse_stddevs, rows).load_masked(row_tile),
                  ct::shape{16_ic, 1_ic});
  auto normalized =
      (ct::element_cast<float>(MatrixView(input, rows, embedding_dim)
                                   .load_masked(row_tile, output_tile)) -
       mean) *
      inverse_stddev;
  auto affine =
      normalized *
          ParameterView(gamma, embedding_dim).load_masked(output_tile) +
      ParameterView(beta, embedding_dim).load_masked(output_tile);
  MatrixView(output, rows, embedding_dim)
      .store_masked(ct::element_cast<Activation>(affine), row_tile,
                    output_tile);
}

// Only the two channel reductions required by dx are computed here; forward's
// saved statistics are shared by both backward paths.
template <class Activation>
__tile_global__ void LayerNormGradientStatisticsKernel(
    const Activation* __restrict__ input, const float* __restrict__ gamma,
    const float* __restrict__ output_gradient, const float* __restrict__ means,
    const float* __restrict__ inverse_stddevs, int rows, int embedding_dim,
    float* __restrict__ gradient_means, float* __restrict__ projected_means) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  const int row_tile = ct::bid().x;
  const int width_tiles = (embedding_dim + kNormChannels - 1) / kNormChannels;
  auto mean = ct::reshape(StatisticsView(means, rows).load_masked(row_tile),
                          ct::shape{16_ic, 1_ic});
  auto inverse_stddev =
      ct::reshape(StatisticsView(inverse_stddevs, rows).load_masked(row_tile),
                  ct::shape{16_ic, 1_ic});
  auto gradient_sum = ct::zeros<ct::tile<float, ct::shape<16, 1>>>();
  auto projected_sum = ct::zeros<ct::tile<float, ct::shape<16, 1>>>();
  for (int tile = 0; tile < width_tiles; ++tile) {
    auto normalized =
        (ct::element_cast<float>(MatrixView(input, rows, embedding_dim)
                                     .load_masked(row_tile, tile)) -
         mean) *
        inverse_stddev;
    auto d_normalized = MatrixView(output_gradient, rows, embedding_dim)
                            .load_masked(row_tile, tile) *
                        ParameterView(gamma, embedding_dim).load_masked(tile);
    gradient_sum = gradient_sum + ct::sum(d_normalized, 1_ic);
    auto columns =
        ct::iota<ct::tile<int, ct::shape<1, 64>>>() + tile * kNormChannels;
    auto projected = d_normalized * normalized;
    // Zero-loaded padding is not enough: centering and normalization can
    // overflow a padded lane, making 0 * infinity NaN. Exclude it explicitly
    // before the channel reduction so valid input gradients remain finite.
    projected_sum =
        projected_sum + ct::sum(ct::select(columns < embedding_dim, projected,
                                           ct::zeros<decltype(projected)>()),
                                1_ic);
  }
  StatisticsView(gradient_means, rows)
      .store_masked(
          ct::reshape(gradient_sum / static_cast<float>(embedding_dim),
                      ct::shape{16_ic}),
          row_tile);
  StatisticsView(projected_means, rows)
      .store_masked(
          ct::reshape(projected_sum / static_cast<float>(embedding_dim),
                      ct::shape{16_ic}),
          row_tile);
}

template <class Activation>
__tile_global__ void LayerNormInputGradientKernel(
    const Activation* __restrict__ input, const float* __restrict__ gamma,
    const float* __restrict__ output_gradient, const float* __restrict__ means,
    const float* __restrict__ inverse_stddevs,
    const float* __restrict__ gradient_means,
    const float* __restrict__ projected_means, int rows, int embedding_dim,
    float* __restrict__ input_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  const int width_tiles = (embedding_dim + kNormChannels - 1) / kNormChannels;
  const int row_tile = ct::bid().x / width_tiles;
  const int output_tile = ct::bid().x % width_tiles;
  auto mean = ct::reshape(StatisticsView(means, rows).load_masked(row_tile),
                          ct::shape{16_ic, 1_ic});
  auto inverse_stddev =
      ct::reshape(StatisticsView(inverse_stddevs, rows).load_masked(row_tile),
                  ct::shape{16_ic, 1_ic});
  auto gradient_mean =
      ct::reshape(StatisticsView(gradient_means, rows).load_masked(row_tile),
                  ct::shape{16_ic, 1_ic});
  auto projected_mean =
      ct::reshape(StatisticsView(projected_means, rows).load_masked(row_tile),
                  ct::shape{16_ic, 1_ic});
  auto normalized =
      (ct::element_cast<float>(MatrixView(input, rows, embedding_dim)
                                   .load_masked(row_tile, output_tile)) -
       mean) *
      inverse_stddev;
  auto d_normalized =
      MatrixView(output_gradient, rows, embedding_dim)
          .load_masked(row_tile, output_tile) *
      ParameterView(gamma, embedding_dim).load_masked(output_tile);
  MatrixView(input_gradient, rows, embedding_dim)
      .store_masked(inverse_stddev * (d_normalized - gradient_mean -
                                      normalized * projected_mean),
                    row_tile, output_tile);
}

// Independent row chunks expose enough parallelism for the large training
// batch. Partial sums are private, so the final reduction requires no atomics.
template <class Activation>
__tile_global__ void LayerNormParameterPartialKernel(
    const Activation* __restrict__ input,
    const float* __restrict__ output_gradient, const float* __restrict__ means,
    const float* __restrict__ inverse_stddevs, int rows, int embedding_dim,
    int chunks, float* __restrict__ gamma_partials,
    float* __restrict__ beta_partials) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  const int width_tiles = (embedding_dim + kNormChannels - 1) / kNormChannels;
  const int chunk = ct::bid().x / width_tiles;
  const int output_tile = ct::bid().x % width_tiles;
  const int first_tile = chunk * (kParameterRows / kNormRows);
  const int end_tile = (rows + kNormRows - 1) / kNormRows;
  auto d_gamma = ct::zeros<ct::tile<float, ct::shape<16, 64>>>();
  auto d_beta = ct::zeros<ct::tile<float, ct::shape<16, 64>>>();
  for (int row_tile = first_tile;
       row_tile < first_tile + kParameterRows / kNormRows &&
       row_tile < end_tile;
       ++row_tile) {
    auto mean = ct::reshape(StatisticsView(means, rows).load_masked(row_tile),
                            ct::shape{16_ic, 1_ic});
    auto inverse_stddev =
        ct::reshape(StatisticsView(inverse_stddevs, rows).load_masked(row_tile),
                    ct::shape{16_ic, 1_ic});
    auto normalized =
        (ct::element_cast<float>(MatrixView(input, rows, embedding_dim)
                                     .load_masked(row_tile, output_tile)) -
         mean) *
        inverse_stddev;
    auto gradient = MatrixView(output_gradient, rows, embedding_dim)
                        .load_masked(row_tile, output_tile);
    d_gamma = d_gamma + gradient * normalized;
    d_beta = d_beta + gradient;
  }
  auto gamma_view = ct::partition_view{
      ct::tensor_span{gamma_partials, ct::extents{chunks, embedding_dim}},
      ct::shape{1_ic, 64_ic}};
  auto beta_view = ct::partition_view{
      ct::tensor_span{beta_partials, ct::extents{chunks, embedding_dim}},
      ct::shape{1_ic, 64_ic}};
  gamma_view.store_masked(ct::sum(d_gamma, 0_ic), chunk, output_tile);
  beta_view.store_masked(ct::sum(d_beta, 0_ic), chunk, output_tile);
}

__tile_global__ void LayerNormParameterGradientKernel(
    const float* __restrict__ gamma_partials,
    const float* __restrict__ beta_partials, int chunks, int embedding_dim,
    float* __restrict__ gamma_gradient, float* __restrict__ beta_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  const int output_tile = ct::bid().x;
  auto d_gamma = ct::zeros<ct::tile<float, ct::shape<16, 64>>>();
  auto d_beta = ct::zeros<ct::tile<float, ct::shape<16, 64>>>();
  for (int tile = 0; tile < (chunks + kNormRows - 1) / kNormRows; ++tile) {
    d_gamma = d_gamma + MatrixView(gamma_partials, chunks, embedding_dim)
                            .load_masked(tile, output_tile);
    d_beta = d_beta + MatrixView(beta_partials, chunks, embedding_dim)
                          .load_masked(tile, output_tile);
  }
  // Preserve the existing per-backward overwrite contract. Each channel has
  // exactly one writer and the reduction order is independent of scheduling.
  ParameterView(gamma_gradient, embedding_dim)
      .store_masked(ct::reshape(ct::sum(d_gamma, 0_ic), ct::shape{64_ic}),
                    output_tile);
  ParameterView(beta_gradient, embedding_dim)
      .store_masked(ct::reshape(ct::sum(d_beta, 0_ic), ct::shape{64_ic}),
                    output_tile);
}

template <class Activation>
void LaunchForward(cuda::Executor& executor, const Buffer& input,
                   const BufferVec& weights, int rows, int embedding_dim,
                   float epsilon, Buffer& means, Buffer& inverse_stddevs,
                   Buffer& output) {
  const int row_tiles = (rows + kNormRows - 1) / kNormRows;
  const int blocks =
      row_tiles * ((embedding_dim + kNormChannels - 1) / kNormChannels);
  LayerNormStatisticsKernel<Activation><<<row_tiles, 1, 0, executor.stream()>>>(
      static_cast<const Activation*>(input.data()), rows, embedding_dim,
      epsilon, static_cast<float*>(means.data()),
      static_cast<float*>(inverse_stddevs.data()));
  LayerNormForwardKernel<Activation><<<blocks, 1, 0, executor.stream()>>>(
      static_cast<const Activation*>(input.data()),
      static_cast<const float*>(weights[0].data()),
      static_cast<const float*>(weights[1].data()),
      static_cast<const float*>(means.data()),
      static_cast<const float*>(inverse_stddevs.data()), rows, embedding_dim,
      static_cast<Activation*>(output.data()));
}

template <class Activation>
void LaunchBackward(cuda::Executor& executor, const BackwardState& state,
                    const Buffer& output_gradient, const BufferVec& weights,
                    int rows, int embedding_dim, int chunks,
                    Buffer& gradient_means, Buffer& projected_means,
                    Buffer& gamma_partials, Buffer& beta_partials,
                    Buffer& input_gradient) {
  const int row_tiles = (rows + kNormRows - 1) / kNormRows;
  const int width_tiles = (embedding_dim + kNormChannels - 1) / kNormChannels;
  const auto* input =
      static_cast<const Activation*>(state.intermediates[0].data());
  const auto* means = static_cast<const float*>(state.intermediates[1].data());
  const auto* inverse_stddevs =
      static_cast<const float*>(state.intermediates[2].data());
  const auto* gradient = static_cast<const float*>(output_gradient.data());
  const auto* gamma = static_cast<const float*>(weights[0].data());
  LayerNormGradientStatisticsKernel<Activation>
      <<<row_tiles, 1, 0, executor.stream()>>>(
          input, gamma, gradient, means, inverse_stddevs, rows, embedding_dim,
          static_cast<float*>(gradient_means.data()),
          static_cast<float*>(projected_means.data()));
  LayerNormInputGradientKernel<Activation>
      <<<row_tiles * width_tiles, 1, 0, executor.stream()>>>(
          input, gamma, gradient, means, inverse_stddevs,
          static_cast<const float*>(gradient_means.data()),
          static_cast<const float*>(projected_means.data()), rows,
          embedding_dim, static_cast<float*>(input_gradient.data()));
  LayerNormParameterPartialKernel<Activation>
      <<<chunks * width_tiles, 1, 0, executor.stream()>>>(
          input, gradient, means, inverse_stddevs, rows, embedding_dim, chunks,
          static_cast<float*>(gamma_partials.data()),
          static_cast<float*>(beta_partials.data()));
}

}  // namespace

absl::StatusOr<std::unique_ptr<LayerNormLayer>> LayerNormLayer::Create(
    cuda::Executor& executor, int embedding_dim, float epsilon,
    DataType data_type, int sequence_length) {
  RETURN_IF_ERROR(internal::ValidateComputeType(data_type));
  if (sequence_length <= 0)
    return absl::InvalidArgumentError("sequence_length must be positive");
  if (!(epsilon > 0.0f))
    return absl::InvalidArgumentError("layer-norm epsilon must be positive");
  RETURN_IF_ERROR(
      internal::ValidateTiledExtent(embedding_dim, "embedding_dim"));
  const size_t bytes = static_cast<size_t>(embedding_dim) * sizeof(float);
  ASSIGN_OR_RETURN(auto gamma, Buffer::Allocate(executor, bytes));
  ASSIGN_OR_RETURN(auto beta, Buffer::Allocate(executor, bytes));
  ASSIGN_OR_RETURN(auto gamma_gradient, Buffer::Allocate(executor, bytes));
  ASSIGN_OR_RETURN(auto beta_gradient, Buffer::Allocate(executor, bytes));
  ASSIGN_OR_RETURN(
      auto gamma_values,
      cuda::PageLockedHostArray<float>::Allocate(executor, embedding_dim));
  std::fill(gamma_values.begin(), gamma_values.end(), 1.0f);
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(gamma.data(), gamma_values.data(), bytes,
                      cudaMemcpyHostToDevice, executor.stream()),
      "cudaMemcpyAsync(layer-norm gamma)"));
  for (Buffer* buffer : {&beta, &gamma_gradient, &beta_gradient})
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemsetAsync(buffer->data(), 0, buffer->size_bytes(),
                        executor.stream()),
        "cudaMemsetAsync(layer-norm parameter)"));
  // gamma_values is released in stream order after its upload completes.
  return absl::WrapUnique(new LayerNormLayer(
      executor, embedding_dim, epsilon, data_type, std::move(gamma),
      std::move(beta), std::move(gamma_gradient), std::move(beta_gradient),
      sequence_length));
}

absl::StatusOr<FwdResult> LayerNormLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs) const {
  RETURN_IF_ERROR(
      internal::ValidateExecutor(executor_, executor, "LayerNormLayer"));
  if (inputs.size() != 1)
    return absl::InvalidArgumentError("LayerNormLayer fwd expects one input");
  ASSIGN_OR_RETURN(int rows,
                   internal::ActivationRows(executor, inputs[0], embedding_dim_,
                                            output_type_, "layer-norm input"));
  ASSIGN_OR_RETURN(auto output,
                   Buffer::Allocate(executor, inputs[0].size_bytes()));
  const size_t statistics_bytes = static_cast<size_t>(rows) * sizeof(float);
  ASSIGN_OR_RETURN(auto means, Buffer::Allocate(executor, statistics_bytes));
  ASSIGN_OR_RETURN(auto inverse_stddevs,
                   Buffer::Allocate(executor, statistics_bytes));
  if (output_type_ == DataType::BF16)
    LaunchForward<__nv_bfloat16>(executor, inputs[0], weights_, rows,
                                 embedding_dim_, epsilon_, means,
                                 inverse_stddevs, output);
  else
    LaunchForward<float>(executor, inputs[0], weights_, rows, embedding_dim_,
                         epsilon_, means, inverse_stddevs, output);
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "layer-norm forward launch"));
  BackwardState state;
  state.intermediates = {inputs[0], std::move(means),
                         std::move(inverse_stddevs)};
  return FwdResult{{std::move(output)}, std::move(state)};
}

absl::StatusOr<BufferVec> LayerNormLayer::bwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    BackwardState state) {
  RETURN_IF_ERROR(
      internal::ValidateExecutor(executor_, executor, "LayerNormLayer"));
  if (output_gradients.size() != 1 || state.intermediates.size() != 3 ||
      !state.children.empty())
    return absl::InvalidArgumentError(
        "LayerNormLayer bwd received an incompatible gradient or state");
  ASSIGN_OR_RETURN(int rows, internal::MatrixRows(executor, output_gradients[0],
                                                  embedding_dim_,
                                                  "layer-norm output gradient"));
  RETURN_IF_ERROR(internal::ValidateBuffer(
      executor, state.intermediates[0],
      static_cast<size_t>(rows) * embedding_dim_ *
          internal::ActivationElementBytes(output_type_),
      "layer-norm saved input"));
  const size_t statistics_bytes = static_cast<size_t>(rows) * sizeof(float);
  RETURN_IF_ERROR(internal::ValidateBuffer(executor, state.intermediates[1],
                                           statistics_bytes,
                                           "layer-norm saved means"));
  RETURN_IF_ERROR(internal::ValidateBuffer(executor, state.intermediates[2],
                                           statistics_bytes,
                                           "layer-norm saved inverse stddevs"));
  ASSIGN_OR_RETURN(
      auto input_gradient,
      Buffer::Allocate(
          executor, static_cast<size_t>(rows) * embedding_dim_ * sizeof(float)));
  ASSIGN_OR_RETURN(auto gradient_means,
                   Buffer::Allocate(executor, statistics_bytes));
  ASSIGN_OR_RETURN(auto projected_means,
                   Buffer::Allocate(executor, statistics_bytes));
  const int chunks = (rows + kParameterRows - 1) / kParameterRows;
  const size_t partial_bytes =
      static_cast<size_t>(chunks) * embedding_dim_ * sizeof(float);
  ASSIGN_OR_RETURN(auto gamma_partials,
                   Buffer::Allocate(executor, partial_bytes));
  ASSIGN_OR_RETURN(auto beta_partials,
                   Buffer::Allocate(executor, partial_bytes));
  if (output_type_ == DataType::BF16)
    LaunchBackward<__nv_bfloat16>(
        executor, state, output_gradients[0], weights_, rows, embedding_dim_,
        chunks, gradient_means, projected_means, gamma_partials, beta_partials,
        input_gradient);
  else
    LaunchBackward<float>(executor, state, output_gradients[0], weights_, rows,
                          embedding_dim_, chunks, gradient_means,
                          projected_means, gamma_partials, beta_partials,
                          input_gradient);
  LayerNormParameterGradientKernel<<<(embedding_dim_ + kNormChannels - 1) /
                                         kNormChannels,
                                     1, 0, executor.stream()>>>(
      static_cast<const float*>(gamma_partials.data()),
      static_cast<const float*>(beta_partials.data()), chunks, embedding_dim_,
      static_cast<float*>(gradients_[0].data()),
      static_cast<float*>(gradients_[1].data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "layer-norm backward launch"));
  return BufferVec{std::move(input_gradient)};
}

}  // namespace pluto::llm
