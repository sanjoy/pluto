#include "src/llm/layers/sparse_autoencoder.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

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
__tile_global__ void SparseEncoderForwardKernel(
    const Activation* __restrict__ input, const float* __restrict__ encoder,
    const float* __restrict__ encoder_bias,
    const float* __restrict__ decoder_bias, int rows, int input_dim,
    int feature_dim, Activation* __restrict__ latents) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto input_view =
      ct::partition_view{ct::tensor_span{input, ct::extents{rows, input_dim}},
                         ct::shape{16_ic, 16_ic}};
  auto encoder_view = ct::partition_view{
      ct::tensor_span{encoder, ct::extents{feature_dim, input_dim}},
      ct::shape{16_ic, 16_ic}};
  auto encoder_bias_view = ct::partition_view{
      ct::tensor_span{encoder_bias, ct::extents{feature_dim}},
      ct::shape{16_ic}};
  auto decoder_bias_view = ct::partition_view{
      ct::tensor_span{decoder_bias, ct::extents{input_dim}}, ct::shape{16_ic}};
  auto latent_view = ct::partition_view{
      ct::tensor_span{latents, ct::extents{rows, feature_dim}},
      ct::shape{16_ic, 16_ic}};
  const int feature_tiles = feature_dim / internal::kDenseTile;
  const int input_tiles = input_dim / internal::kDenseTile;
  const int block = ct::bid().x;
  const int row_tile = block / feature_tiles;
  const int feature_tile = block % feature_tiles;
  auto accumulator = ct::broadcast(encoder_bias_view.load(feature_tile),
                                   ct::shape{16_ic, 16_ic});
  for (int input_tile = 0; input_tile < input_tiles; ++input_tile) {
    auto centered =
        ct::element_cast<float>(input_view.load(row_tile, input_tile)) -
        ct::broadcast(decoder_bias_view.load(input_tile),
                      ct::shape{16_ic, 16_ic});
    auto left = ct::element_cast<MmaType<Activation>>(centered);
    auto right = ct::transpose(ct::element_cast<MmaType<Activation>>(
        encoder_view.load(feature_tile, input_tile)));
    accumulator = ct::mma(left, right, accumulator);
  }
  auto zero = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  latent_view.store(ct::element_cast<Activation>(ct::max(accumulator, zero)),
                    row_tile, feature_tile);
}

template <class Activation>
__tile_global__ void SparseDecoderForwardKernel(
    const Activation* __restrict__ latents, const float* __restrict__ decoder,
    const float* __restrict__ decoder_bias, int rows, int input_dim,
    int feature_dim, Activation* __restrict__ reconstruction) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto latent_view = ct::partition_view{
      ct::tensor_span{latents, ct::extents{rows, feature_dim}},
      ct::shape{16_ic, 16_ic}};
  auto decoder_view = ct::partition_view{
      ct::tensor_span{decoder, ct::extents{input_dim, feature_dim}},
      ct::shape{16_ic, 16_ic}};
  auto bias_view = ct::partition_view{
      ct::tensor_span{decoder_bias, ct::extents{input_dim}}, ct::shape{16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{reconstruction, ct::extents{rows, input_dim}},
      ct::shape{16_ic, 16_ic}};
  const int input_tiles = input_dim / internal::kDenseTile;
  const int feature_tiles = feature_dim / internal::kDenseTile;
  const int block = ct::bid().x;
  const int row_tile = block / input_tiles;
  const int input_tile = block % input_tiles;
  auto accumulator =
      ct::broadcast(bias_view.load(input_tile), ct::shape{16_ic, 16_ic});
  for (int feature_tile = 0; feature_tile < feature_tiles; ++feature_tile) {
    auto left = ct::element_cast<MmaType<Activation>>(
        latent_view.load(row_tile, feature_tile));
    auto right = ct::transpose(ct::element_cast<MmaType<Activation>>(
        decoder_view.load(input_tile, feature_tile)));
    accumulator = ct::mma(left, right, accumulator);
  }
  output_view.store(ct::element_cast<Activation>(accumulator), row_tile,
                    input_tile);
}

template <class Activation>
__tile_global__ void SparseZStatisticsKernel(
    const Activation* __restrict__ latents, int rows, int feature_dim,
    float* __restrict__ statistics) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto latent_view = ct::partition_view{
      ct::tensor_span{latents, ct::extents{rows, feature_dim}},
      ct::shape{1_ic, 16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{statistics, ct::extents{rows, 4}}, ct::shape{1_ic, 1_ic}};
  // One program per token, parallel across tokens. FP32 statistics operate
  // on the stored Z (after BF16 rounding), not the encoder's accumulator.
  auto count = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  auto sum = count;
  auto squared_sum = count;
  auto maximum = count;
  auto zero = ct::zeros<ct::tile<float, ct::shape<1, 16>>>();
  const int row = ct::bid().x;
  for (int feature = 0; feature < feature_dim / internal::kDenseTile;
       ++feature) {
    auto z = ct::element_cast<float>(latent_view.load(row, feature));
    count = count + ct::sum(ct::select(z > zero, zero + 1.0f, zero), 1_ic);
    sum = sum + ct::sum(z, 1_ic);
    squared_sum = squared_sum + ct::sum(z * z, 1_ic);
    maximum = ct::max(maximum, ct::reduce_max(z, 1_ic));
  }
  output_view.store(count, row, 0);
  output_view.store(sum, row, 1);
  output_view.store(squared_sum, row, 2);
  output_view.store(maximum, row, 3);
}

template <class Activation>
__tile_global__ void SparseLatentGradientKernel(
    const float* __restrict__ reconstruction_gradient,
    const float* __restrict__ decoder, int rows, int input_dim, int feature_dim,
    float* __restrict__ latent_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto output_gradient_view = ct::partition_view{
      ct::tensor_span{reconstruction_gradient, ct::extents{rows, input_dim}},
      ct::shape{16_ic, 16_ic}};
  auto decoder_view = ct::partition_view{
      ct::tensor_span{decoder, ct::extents{input_dim, feature_dim}},
      ct::shape{16_ic, 16_ic}};
  auto latent_gradient_view = ct::partition_view{
      ct::tensor_span{latent_gradient, ct::extents{rows, feature_dim}},
      ct::shape{16_ic, 16_ic}};
  const int feature_tiles = feature_dim / internal::kDenseTile;
  const int input_tiles = input_dim / internal::kDenseTile;
  const int block = ct::bid().x;
  const int row_tile = block / feature_tiles;
  const int feature_tile = block % feature_tiles;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  for (int input_tile = 0; input_tile < input_tiles; ++input_tile) {
    auto left = ct::element_cast<MmaType<Activation>>(
        output_gradient_view.load(row_tile, input_tile));
    auto right = ct::element_cast<MmaType<Activation>>(
        decoder_view.load(input_tile, feature_tile));
    accumulator = ct::mma(left, right, accumulator);
  }
  latent_gradient_view.store(accumulator, row_tile, feature_tile);
}

__tile_global__ void AddFloatKernel(const float* __restrict__ addend,
                                    int elements,
                                    float* __restrict__ accumulator) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto addend_view = ct::partition_view{
      ct::tensor_span{addend, ct::extents{elements}}, ct::shape{16_ic}};
  auto accumulator_view = ct::partition_view{
      ct::tensor_span{accumulator, ct::extents{elements}}, ct::shape{16_ic}};
  const int block = ct::bid().x;
  accumulator_view.store(accumulator_view.load(block) + addend_view.load(block),
                         block);
}

template <class Activation>
__tile_global__ void SparseReluBackwardKernel(
    const Activation* __restrict__ latents,
    const float* __restrict__ latent_gradient, int elements,
    float* __restrict__ preactivation_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto latent_view = ct::partition_view{
      ct::tensor_span{latents, ct::extents{elements}}, ct::shape{16_ic}};
  auto latent_gradient_view = ct::partition_view{
      ct::tensor_span{latent_gradient, ct::extents{elements}},
      ct::shape{16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{preactivation_gradient, ct::extents{elements}},
      ct::shape{16_ic}};
  const int block = ct::bid().x;
  auto latent = ct::element_cast<float>(latent_view.load(block));
  auto zero = ct::zeros<ct::tile<float, ct::shape<16>>>();
  output_view.store(
      ct::select(latent > 0.0f, latent_gradient_view.load(block), zero), block);
}

template <class Activation>
__tile_global__ void SparseInputGradientKernel(
    const float* __restrict__ preactivation_gradient,
    const float* __restrict__ encoder, int rows, int input_dim, int feature_dim,
    float* __restrict__ input_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto preactivation_gradient_view = ct::partition_view{
      ct::tensor_span{preactivation_gradient, ct::extents{rows, feature_dim}},
      ct::shape{16_ic, 16_ic}};
  auto encoder_view = ct::partition_view{
      ct::tensor_span{encoder, ct::extents{feature_dim, input_dim}},
      ct::shape{16_ic, 16_ic}};
  auto input_gradient_view = ct::partition_view{
      ct::tensor_span{input_gradient, ct::extents{rows, input_dim}},
      ct::shape{16_ic, 16_ic}};
  const int input_tiles = input_dim / internal::kDenseTile;
  const int feature_tiles = feature_dim / internal::kDenseTile;
  const int block = ct::bid().x;
  const int row_tile = block / input_tiles;
  const int input_tile = block % input_tiles;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  for (int feature_tile = 0; feature_tile < feature_tiles; ++feature_tile) {
    auto left = ct::element_cast<MmaType<Activation>>(
        preactivation_gradient_view.load(row_tile, feature_tile));
    auto right = ct::element_cast<MmaType<Activation>>(
        encoder_view.load(feature_tile, input_tile));
    accumulator = ct::mma(left, right, accumulator);
  }
  input_gradient_view.store(accumulator, row_tile, input_tile);
}

template <class Activation>
__tile_global__ void SparseEncoderWeightGradientKernel(
    const Activation* __restrict__ input,
    const float* __restrict__ decoder_bias,
    const float* __restrict__ preactivation_gradient, int rows, int input_dim,
    int feature_dim, float* __restrict__ encoder_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto input_view =
      ct::partition_view{ct::tensor_span{input, ct::extents{rows, input_dim}},
                         ct::shape{16_ic, 16_ic}};
  auto bias_view = ct::partition_view{
      ct::tensor_span{decoder_bias, ct::extents{input_dim}}, ct::shape{16_ic}};
  auto preactivation_gradient_view = ct::partition_view{
      ct::tensor_span{preactivation_gradient, ct::extents{rows, feature_dim}},
      ct::shape{16_ic, 16_ic}};
  auto encoder_gradient_view = ct::partition_view{
      ct::tensor_span{encoder_gradient, ct::extents{feature_dim, input_dim}},
      ct::shape{16_ic, 16_ic}};
  const int input_tiles = input_dim / internal::kDenseTile;
  const int row_tiles = rows / internal::kDenseTile;
  const int block = ct::bid().x;
  const int feature_tile = block / input_tiles;
  const int input_tile = block % input_tiles;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
    auto gradient_transposed =
        ct::transpose(ct::element_cast<MmaType<Activation>>(
            preactivation_gradient_view.load(row_tile, feature_tile)));
    auto centered =
        ct::element_cast<float>(input_view.load(row_tile, input_tile)) -
        ct::broadcast(bias_view.load(input_tile), ct::shape{16_ic, 16_ic});
    accumulator =
        ct::mma(gradient_transposed,
                ct::element_cast<MmaType<Activation>>(centered), accumulator);
  }
  encoder_gradient_view.store(accumulator, feature_tile, input_tile);
}

__tile_global__ void SparseEncoderBiasGradientKernel(
    const float* __restrict__ preactivation_gradient, int rows, int feature_dim,
    float* __restrict__ encoder_bias_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto gradient_view = ct::partition_view{
      ct::tensor_span{preactivation_gradient, ct::extents{rows, feature_dim}},
      ct::shape{16_ic, 16_ic}};
  auto bias_gradient_view = ct::partition_view{
      ct::tensor_span{encoder_bias_gradient, ct::extents{feature_dim}},
      ct::shape{16_ic}};
  const int row_tiles = rows / internal::kDenseTile;
  const int feature_tile = ct::bid().x;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<1, 16>>>();
  for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
    accumulator =
        accumulator + ct::sum(gradient_view.load(row_tile, feature_tile), 0_ic);
  }
  bias_gradient_view.store(ct::reshape(accumulator, ct::shape{16_ic}),
                           feature_tile);
}

template <class Activation>
__tile_global__ void SparseDecoderWeightGradientKernel(
    const float* __restrict__ reconstruction_gradient,
    const Activation* __restrict__ latents, int rows, int input_dim,
    int feature_dim, float* __restrict__ decoder_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto reconstruction_gradient_view = ct::partition_view{
      ct::tensor_span{reconstruction_gradient, ct::extents{rows, input_dim}},
      ct::shape{16_ic, 16_ic}};
  auto latent_view = ct::partition_view{
      ct::tensor_span{latents, ct::extents{rows, feature_dim}},
      ct::shape{16_ic, 16_ic}};
  auto decoder_gradient_view = ct::partition_view{
      ct::tensor_span{decoder_gradient, ct::extents{input_dim, feature_dim}},
      ct::shape{16_ic, 16_ic}};
  const int feature_tiles = feature_dim / internal::kDenseTile;
  const int row_tiles = rows / internal::kDenseTile;
  const int block = ct::bid().x;
  const int input_tile = block / feature_tiles;
  const int feature_tile = block % feature_tiles;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
    auto gradient_transposed =
        ct::transpose(ct::element_cast<MmaType<Activation>>(
            reconstruction_gradient_view.load(row_tile, input_tile)));
    auto latent = ct::element_cast<MmaType<Activation>>(
        latent_view.load(row_tile, feature_tile));
    accumulator = ct::mma(gradient_transposed, latent, accumulator);
  }
  decoder_gradient_view.store(accumulator, input_tile, feature_tile);
}

__tile_global__ void SparseDecoderBiasGradientKernel(
    const float* __restrict__ reconstruction_gradient,
    const float* __restrict__ input_gradient, int rows, int input_dim,
    float* __restrict__ decoder_bias_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto reconstruction_gradient_view = ct::partition_view{
      ct::tensor_span{reconstruction_gradient, ct::extents{rows, input_dim}},
      ct::shape{16_ic, 16_ic}};
  auto input_gradient_view = ct::partition_view{
      ct::tensor_span{input_gradient, ct::extents{rows, input_dim}},
      ct::shape{16_ic, 16_ic}};
  auto bias_gradient_view = ct::partition_view{
      ct::tensor_span{decoder_bias_gradient, ct::extents{input_dim}},
      ct::shape{16_ic}};
  const int row_tiles = rows / internal::kDenseTile;
  const int input_tile = ct::bid().x;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<1, 16>>>();
  for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
    accumulator =
        accumulator +
        ct::sum(reconstruction_gradient_view.load(row_tile, input_tile) -
                    input_gradient_view.load(row_tile, input_tile),
                0_ic);
  }
  bias_gradient_view.store(ct::reshape(accumulator, ct::shape{16_ic}),
                           input_tile);
}

// Decoder-column norms depend only on D, not on the activation row. Compute
// each group of 16 norms once instead of repeating this work for every row.
__tile_global__ void SparseLossDecoderNormKernel(
    const float* __restrict__ decoder, int input_dim, int feature_dim,
    float* __restrict__ decoder_norm) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto decoder_view = ct::partition_view{
      ct::tensor_span{decoder, ct::extents{input_dim, feature_dim}},
      ct::shape{16_ic, 16_ic}};
  auto norm_view = ct::partition_view{
      ct::tensor_span{decoder_norm, ct::extents{feature_dim}},
      ct::shape{16_ic}};
  const int input_tiles = input_dim / internal::kDenseTile;
  const int feature_tile = ct::bid().x;
  auto norm_squared = ct::zeros<ct::tile<float, ct::shape<1, 16>>>();
  for (int input_tile = 0; input_tile < input_tiles; ++input_tile) {
    auto directions = decoder_view.load(input_tile, feature_tile);
    norm_squared = norm_squared + ct::sum(directions * directions, 0_ic);
  }
  norm_view.store(ct::reshape(ct::sqrt(norm_squared), ct::shape{16_ic}),
                  feature_tile);
}

// One tile program owns each row. It performs a modest serial reduction over
// that row while all rows are processed independently across the GPU.
template <class Activation>
__tile_global__ void SparseLossPerRowKernel(
    const Activation* __restrict__ input,
    const Activation* __restrict__ reconstruction,
    const Activation* __restrict__ latents,
    const float* __restrict__ decoder_norm, int rows, int input_dim,
    int feature_dim, float sparsity_penalty, float* __restrict__ row_losses) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto input_view =
      ct::partition_view{ct::tensor_span{input, ct::extents{rows, input_dim}},
                         ct::shape{1_ic, 16_ic}};
  auto reconstruction_view = ct::partition_view{
      ct::tensor_span{reconstruction, ct::extents{rows, input_dim}},
      ct::shape{1_ic, 16_ic}};
  auto latent_view = ct::partition_view{
      ct::tensor_span{latents, ct::extents{rows, feature_dim}},
      ct::shape{1_ic, 16_ic}};
  auto norm_view = ct::partition_view{
      ct::tensor_span{decoder_norm, ct::extents{feature_dim}},
      ct::shape{16_ic}};
  auto row_loss_view = ct::partition_view{
      ct::tensor_span{row_losses, ct::extents{rows}}, ct::shape{1_ic}};
  const int input_tiles = input_dim / internal::kDenseTile;
  const int feature_tiles = feature_dim / internal::kDenseTile;
  const int row = ct::bid().x;
  auto loss = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  for (int input_tile = 0; input_tile < input_tiles; ++input_tile) {
    auto residual =
        ct::element_cast<float>(input_view.load(row, input_tile)) -
        ct::element_cast<float>(reconstruction_view.load(row, input_tile));
    loss = loss + ct::sum(residual * residual, 1_ic);
  }
  for (int feature_tile = 0; feature_tile < feature_tiles; ++feature_tile) {
    auto features =
        ct::element_cast<float>(latent_view.load(row, feature_tile));
    auto norm =
        ct::reshape(norm_view.load(feature_tile), ct::shape{1_ic, 16_ic});
    loss = loss + sparsity_penalty * ct::sum(features * norm, 1_ic);
  }
  row_loss_view.store(ct::reshape(loss, ct::shape{1_ic}), row);
}

// The expensive work above is parallel. This deliberately simple final
// reduction reads only one FP32 scalar per row and keeps the public loss-layer
// result as a single scalar.
__tile_global__ void SparseLossReduceKernel(
    const float* __restrict__ row_losses, int rows,
    float* __restrict__ output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto row_loss_view = ct::partition_view{
      ct::tensor_span{row_losses, ct::extents{rows}}, ct::shape{1_ic}};
  auto output_view = ct::partition_view{ct::tensor_span{output, ct::extents{1}},
                                        ct::shape{1_ic}};
  auto loss = ct::zeros<ct::tile<float, ct::shape<1>>>();
  for (int row = 0; row < rows; ++row) {
    loss = loss + row_loss_view.load(row);
  }
  output_view.store(loss, 0);
}

template <class Activation>
__tile_global__ void SparseLossReconstructionGradientKernel(
    const Activation* __restrict__ input,
    const Activation* __restrict__ reconstruction, int elements,
    float* __restrict__ input_gradient,
    float* __restrict__ reconstruction_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{elements}}, ct::shape{16_ic}};
  auto reconstruction_view = ct::partition_view{
      ct::tensor_span{reconstruction, ct::extents{elements}}, ct::shape{16_ic}};
  auto input_gradient_view = ct::partition_view{
      ct::tensor_span{input_gradient, ct::extents{elements}}, ct::shape{16_ic}};
  auto reconstruction_gradient_view = ct::partition_view{
      ct::tensor_span{reconstruction_gradient, ct::extents{elements}},
      ct::shape{16_ic}};
  const int block = ct::bid().x;
  auto gradient =
      2.0f * (ct::element_cast<float>(input_view.load(block)) -
              ct::element_cast<float>(reconstruction_view.load(block)));
  input_gradient_view.store(gradient, block);
  reconstruction_gradient_view.store(-gradient, block);
}

__tile_global__ void SparseLossLatentGradientKernel(
    const float* __restrict__ decoder, int rows, int input_dim, int feature_dim,
    float sparsity_penalty, float* __restrict__ latent_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto decoder_view = ct::partition_view{
      ct::tensor_span{decoder, ct::extents{input_dim, feature_dim}},
      ct::shape{16_ic, 16_ic}};
  auto latent_gradient_view = ct::partition_view{
      ct::tensor_span{latent_gradient, ct::extents{rows, feature_dim}},
      ct::shape{1_ic, 16_ic}};
  const int input_tiles = input_dim / internal::kDenseTile;
  const int feature_tile = ct::bid().x;
  auto norm_squared = ct::zeros<ct::tile<float, ct::shape<1, 16>>>();
  for (int input_tile = 0; input_tile < input_tiles; ++input_tile) {
    auto directions = decoder_view.load(input_tile, feature_tile);
    norm_squared = norm_squared + ct::sum(directions * directions, 0_ic);
  }
  for (int row = 0; row < rows; ++row) {
    latent_gradient_view.store(sparsity_penalty * ct::sqrt(norm_squared), row,
                               feature_tile);
  }
}

// Reduce one feature group exactly once, in a fixed row/input-tile order.
// Separating this scale from the pointwise decoder gradient avoids repeating
// both reductions for every input tile. It also avoids expanding a reduced
// 1x16 tile to 16x16 here: CUDA 13.3's lowering of that mixed-layout broadcast
// generated out-of-bounds shared-memory writes. The linear scratch has only
// feature_dim FP32 elements and stays on the executor's stream.
template <class Activation>
__tile_global__ void SparseLossDecoderScaleKernel(
    const Activation* __restrict__ latents, const float* __restrict__ decoder,
    int rows, int input_dim, int feature_dim, float sparsity_penalty,
    float* __restrict__ decoder_scale) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto latent_view = ct::partition_view{
      ct::tensor_span{latents, ct::extents{rows, feature_dim}},
      ct::shape{1_ic, 16_ic}};
  auto decoder_view = ct::partition_view{
      ct::tensor_span{decoder, ct::extents{input_dim, feature_dim}},
      ct::shape{16_ic, 16_ic}};
  auto scale_view = ct::partition_view{
      ct::tensor_span{decoder_scale, ct::extents{feature_dim}},
      ct::shape{16_ic}};
  const int feature_tile = ct::bid().x;
  auto latent_sum = ct::zeros<ct::tile<float, ct::shape<1, 16>>>();
  for (int row = 0; row < rows; ++row) {
    latent_sum = latent_sum +
                 ct::element_cast<float>(latent_view.load(row, feature_tile));
  }
  // d||D_i||_2/dD_i = D_i/||D_i||_2. Recompute each column's norm in
  // FP32; only the input-dimension reduction is needed here, not a reduction
  // over the full batch. A zero column uses the valid zero subgradient.
  auto zero = ct::zeros<ct::tile<float, ct::shape<1, 16>>>();
  auto norm_squared = zero;
  for (int tile = 0; tile < input_dim / internal::kDenseTile; ++tile) {
    auto directions = decoder_view.load(tile, feature_tile);
    norm_squared = norm_squared + ct::sum(directions * directions, 0_ic);
  }
  auto norm = ct::sqrt(norm_squared);
  // Avoid evaluating 0/0 even in a masked branch. This changes only the
  // denominator for zero columns, whose entries (and gradients) are zero.
  auto safe_norm = ct::select(norm > zero, norm, zero + 1.0f);
  auto scale = sparsity_penalty * latent_sum / safe_norm;
  scale_view.store(ct::reshape(scale, ct::shape{16_ic}), feature_tile);
}

// Each program owns 16 adjacent decoder entries. All operands have the same
// linear layout, so no cross-warp broadcast or shared-memory transpose is
// needed. A feature group's scale is reused by every decoder row.
__tile_global__ void SparseLossDecoderGradientKernel(
    const float* __restrict__ decoder, const float* __restrict__ decoder_scale,
    int64_t elements, int feature_dim, int grid_blocks,
    float* __restrict__ decoder_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto decoder_view = ct::partition_view{
      ct::tensor_span{decoder, ct::extents{elements}}, ct::shape{16_ic}};
  auto scale_view = ct::partition_view{
      ct::tensor_span{decoder_scale, ct::extents{feature_dim}},
      ct::shape{16_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{decoder_gradient, ct::extents{elements}},
      ct::shape{16_ic}};
  // Use 64-bit offsets for tables exceeding INT_MAX elements. The grid-stride
  // loop also keeps the launch legal when the tile count exceeds CUDA's
  // signed-32-bit x-grid limit; ownership remains disjoint between programs.
  const int64_t tiles = elements / internal::kDenseTile;
  for (int64_t block = ct::bid().x; block < tiles; block += grid_blocks) {
    const int feature_tile =
        static_cast<int>(block % (feature_dim / internal::kDenseTile));
    gradient_view.store(
        decoder_view.load(block) * scale_view.load(feature_tile), block);
  }
}

absl::StatusOr<int> ValidateLossInputs(cuda::Executor& executor,
                                       absl::Span<const Buffer> inputs,
                                       int input_dim, int feature_dim,
                                       DataType data_type) {
  if (inputs.size() != 4) {
    return absl::InvalidArgumentError(
        "SparseAutoEncoderLossLayer expects x, x1, z, and D");
  }
  ASSIGN_OR_RETURN(int rows,
                   internal::ActivationRows(executor, inputs[0], input_dim,
                                            data_type, "sparse loss input"));
  RETURN_IF_ERROR(
      internal::ValidateBuffer(executor, inputs[1],
                               static_cast<size_t>(rows) * input_dim *
                                   internal::ActivationElementBytes(data_type),
                               "sparse loss reconstruction"));
  RETURN_IF_ERROR(
      internal::ValidateBuffer(executor, inputs[2],
                               static_cast<size_t>(rows) * feature_dim *
                                   internal::ActivationElementBytes(data_type),
                               "sparse loss latent activations"));
  RETURN_IF_ERROR(internal::ValidateBuffer(
      executor, inputs[3],
      static_cast<size_t>(input_dim) * feature_dim * sizeof(float),
      "sparse loss decoder"));
  return rows;
}

absl::Status CopyNormal(cuda::Executor& executor, Buffer& destination,
                        float standard_deviation, std::mt19937_64* random,
                        const char* operation) {
  std::normal_distribution<float> distribution(0.0f, standard_deviation);
  ASSIGN_OR_RETURN(auto values, cuda::PageLockedHostArray<float>::Allocate(
                                    destination.size_bytes() / sizeof(float)));
  for (float& value : values) value = distribution(*random);
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(destination.data(), values.data(),
                      destination.size_bytes(), cudaMemcpyHostToDevice,
                      executor.stream()),
      operation));
  return executor.Synchronize();
}

}  // namespace

absl::StatusOr<std::unique_ptr<SparseAutoEncoderLayer>>
SparseAutoEncoderLayer::Create(cuda::Executor& executor, int input_dim,
                               int feature_dim, DataType data_type, Mode mode) {
  RETURN_IF_ERROR(internal::ValidateComputeType(data_type));
  RETURN_IF_ERROR(internal::ValidateTiledExtent(input_dim, "input_dim"));
  RETURN_IF_ERROR(internal::ValidateTiledExtent(feature_dim, "feature_dim"));
  // Row counts are stored in FP32 alongside the moments, so keep them exact.
  if (mode == Mode::kCollectStatistics && feature_dim > (1 << 24)) {
    return absl::InvalidArgumentError(
        "SAE statistics require feature_dim <= 2^24");
  }
  const size_t encoder_bytes =
      static_cast<size_t>(feature_dim) * input_dim * sizeof(float);
  const size_t encoder_bias_bytes =
      static_cast<size_t>(feature_dim) * sizeof(float);
  const size_t decoder_bytes =
      static_cast<size_t>(input_dim) * feature_dim * sizeof(float);
  const size_t decoder_bias_bytes =
      static_cast<size_t>(input_dim) * sizeof(float);
  ASSIGN_OR_RETURN(auto encoder, Buffer::Allocate(executor, encoder_bytes));
  ASSIGN_OR_RETURN(auto encoder_bias,
                   Buffer::Allocate(executor, encoder_bias_bytes));
  ASSIGN_OR_RETURN(auto decoder, Buffer::Allocate(executor, decoder_bytes));
  ASSIGN_OR_RETURN(auto decoder_bias,
                   Buffer::Allocate(executor, decoder_bias_bytes));
  ASSIGN_OR_RETURN(auto encoder_gradient,
                   Buffer::Allocate(executor, encoder_bytes));
  ASSIGN_OR_RETURN(auto encoder_bias_gradient,
                   Buffer::Allocate(executor, encoder_bias_bytes));
  ASSIGN_OR_RETURN(auto decoder_gradient,
                   Buffer::Allocate(executor, decoder_bytes));
  ASSIGN_OR_RETURN(auto decoder_bias_gradient,
                   Buffer::Allocate(executor, decoder_bias_bytes));
  for (Buffer* buffer :
       {&encoder, &encoder_bias, &decoder, &decoder_bias, &encoder_gradient,
        &encoder_bias_gradient, &decoder_gradient, &decoder_bias_gradient}) {
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemsetAsync(buffer->data(), 0, buffer->size_bytes(),
                        executor.stream()),
        "cudaMemsetAsync(sparse autoencoder parameter)"));
  }
  return std::unique_ptr<SparseAutoEncoderLayer>(new SparseAutoEncoderLayer(
      executor, input_dim, feature_dim, data_type, mode, std::move(encoder),
      std::move(encoder_bias), std::move(decoder), std::move(decoder_bias),
      std::move(encoder_gradient), std::move(encoder_bias_gradient),
      std::move(decoder_gradient), std::move(decoder_bias_gradient)));
}

absl::Status SparseAutoEncoderLayer::InitializeNormal(float standard_deviation,
                                                      uint64_t seed) {
  if (!(standard_deviation > 0.0f)) {
    return absl::InvalidArgumentError(
        "sparse autoencoder initialization standard deviation must be "
        "positive");
  }
  std::mt19937_64 random(seed);
  RETURN_IF_ERROR(CopyNormal(executor_, weights_[0], standard_deviation,
                             &random, "copy sparse encoder initialization"));
  RETURN_IF_ERROR(CopyNormal(executor_, weights_[2], standard_deviation,
                             &random, "copy sparse decoder initialization"));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemsetAsync(weights_[1].data(), 0, weights_[1].size_bytes(),
                      executor_.stream()),
      "clear sparse encoder bias"));
  return cuda::CudaStatus(
      cudaMemsetAsync(weights_[3].data(), 0, weights_[3].size_bytes(),
                      executor_.stream()),
      "clear sparse decoder bias");
}

absl::StatusOr<Buffer> SparseAutoEncoderLayer::fwd(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    Tape* tape) const {
  RETURN_IF_ERROR(internal::ValidateExecutor(executor_, executor,
                                             "SparseAutoEncoderLayer"));
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "SparseAutoEncoderLayer fwd expects one input and a non-null tape");
  }
  ASSIGN_OR_RETURN(int rows, internal::ActivationRows(
                                 executor, inputs[0], input_dim_, output_type_,
                                 "sparse autoencoder input"));
  RETURN_IF_ERROR(internal::ValidateTiledExtent(rows, "SAE rows"));
  const size_t activation_bytes =
      internal::ActivationElementBytes(output_type_);
  ASSIGN_OR_RETURN(
      auto latents,
      Buffer::Allocate(executor, static_cast<size_t>(rows) * feature_dim_ *
                                     activation_bytes));
  ASSIGN_OR_RETURN(
      auto reconstruction,
      Buffer::Allocate(
          executor, static_cast<size_t>(rows) * input_dim_ * activation_bytes));
  const int encoder_blocks =
      internal::TileCount(rows) * internal::TileCount(feature_dim_);
  const int decoder_blocks =
      internal::TileCount(rows) * internal::TileCount(input_dim_);
  if (output_type_ == DataType::BF16) {
    SparseEncoderForwardKernel<__nv_bfloat16>
        <<<encoder_blocks, 1, 0, executor.stream()>>>(
            static_cast<const __nv_bfloat16*>(inputs[0].data()),
            static_cast<const float*>(weights_[0].data()),
            static_cast<const float*>(weights_[1].data()),
            static_cast<const float*>(weights_[3].data()), rows, input_dim_,
            feature_dim_, static_cast<__nv_bfloat16*>(latents.data()));
    SparseDecoderForwardKernel<__nv_bfloat16>
        <<<decoder_blocks, 1, 0, executor.stream()>>>(
            static_cast<const __nv_bfloat16*>(latents.data()),
            static_cast<const float*>(weights_[2].data()),
            static_cast<const float*>(weights_[3].data()), rows, input_dim_,
            feature_dim_, static_cast<__nv_bfloat16*>(reconstruction.data()));
  } else {
    SparseEncoderForwardKernel<float>
        <<<encoder_blocks, 1, 0, executor.stream()>>>(
            static_cast<const float*>(inputs[0].data()),
            static_cast<const float*>(weights_[0].data()),
            static_cast<const float*>(weights_[1].data()),
            static_cast<const float*>(weights_[3].data()), rows, input_dim_,
            feature_dim_, static_cast<float*>(latents.data()));
    SparseDecoderForwardKernel<float>
        <<<decoder_blocks, 1, 0, executor.stream()>>>(
            static_cast<const float*>(latents.data()),
            static_cast<const float*>(weights_[2].data()),
            static_cast<const float*>(weights_[3].data()), rows, input_dim_,
            feature_dim_, static_cast<float*>(reconstruction.data()));
  }
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(),
                                   "sparse autoencoder forward launch"));
  tape->intermediates = {inputs[0], latents};
  tape->children.clear();
  if (mode_ == Mode::kCollectStatistics) {
    ASSIGN_OR_RETURN(auto statistics,
                     Buffer::Allocate(executor, static_cast<size_t>(rows) * 4 *
                                                    sizeof(float)));
    if (output_type_ == DataType::BF16) {
      SparseZStatisticsKernel<__nv_bfloat16><<<rows, 1, 0, executor.stream()>>>(
          static_cast<const __nv_bfloat16*>(latents.data()), rows, feature_dim_,
          static_cast<float*>(statistics.data()));
    } else {
      SparseZStatisticsKernel<float><<<rows, 1, 0, executor.stream()>>>(
          static_cast<const float*>(latents.data()), rows, feature_dim_,
          static_cast<float*>(statistics.data()));
    }
    RETURN_IF_ERROR(
        cuda::CudaStatus(cudaGetLastError(), "SAE Z statistics launch"));
    tape->intermediates.push_back(std::move(statistics));
  }
  return std::move(reconstruction);
}

absl::StatusOr<SparseAutoEncoderZStatistics>
SparseAutoEncoderLayer::ReadZStatistics(cuda::Executor& executor,
                                        const Tape& tape,
                                        int valid_rows) const {
  RETURN_IF_ERROR(internal::ValidateExecutor(executor_, executor,
                                             "SparseAutoEncoderLayer"));
  if (tape.intermediates.size() != 3) {
    return absl::FailedPreconditionError(
        "ReadZStatistics requires a tape from kCollectStatistics mode");
  }
  RETURN_IF_ERROR(latent_activations(tape).status());
  ASSIGN_OR_RETURN(int rows, internal::ActivationRows(
                                 executor, tape.intermediates[0], input_dim_,
                                 output_type_, "SAE saved input"));
  RETURN_IF_ERROR(internal::ValidateBuffer(
      executor, tape.intermediates[2],
      static_cast<size_t>(rows) * 4 * sizeof(float), "SAE saved statistics"));
  if (valid_rows < 0 || valid_rows > rows) {
    return absl::InvalidArgumentError("valid_rows must be in [0, SAE rows]");
  }
  if (valid_rows != 0) rows = valid_rows;
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<float>::Allocate(
                                  static_cast<size_t>(rows) * 4));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), tape.intermediates[2].data(),
                      host.size_bytes(), cudaMemcpyDeviceToHost,
                      executor.stream()),
      "copy SAE Z statistics"));
  RETURN_IF_ERROR(executor.Synchronize());

  SparseAutoEncoderZStatistics result;
  result.rows = rows;
  result.feature_dim = feature_dim_;
  double sum = 0;
  double squared_sum = 0;
  for (int row = 0; row < rows; ++row) {
    const float* values = host.data() + static_cast<size_t>(row) * 4;
    for (int index = 0; index < 4; ++index) {
      if (!std::isfinite(values[index])) {
        return absl::FailedPreconditionError("non-finite SAE Z statistics");
      }
    }
    result.active_count += static_cast<int64_t>(values[0]);
    sum += values[1];
    squared_sum += values[2];
    result.maximum = std::max(result.maximum, static_cast<double>(values[3]));
  }
  const double elements = static_cast<double>(rows) * feature_dim_;
  result.mean = sum / elements;
  // Roundoff may make the variance of a constant Z very slightly negative.
  result.standard_deviation = std::sqrt(
      std::max(0.0, squared_sum / elements - result.mean * result.mean));
  return result;
}

absl::StatusOr<Buffer> SparseAutoEncoderLayer::latent_activations(
    const Tape& tape) const {
  if (tape.intermediates.size() != 2 && tape.intermediates.size() != 3) {
    return absl::InvalidArgumentError(
        "latent_activations requires a tape produced by SAE fwd");
  }
  ASSIGN_OR_RETURN(int rows, internal::ActivationRows(
                                 executor_, tape.intermediates[0], input_dim_,
                                 output_type_, "SAE saved input"));
  RETURN_IF_ERROR(internal::ValidateBuffer(
      executor_, tape.intermediates[1],
      static_cast<size_t>(rows) * feature_dim_ *
          internal::ActivationElementBytes(output_type_),
      "SAE saved latent activations"));
  return tape.intermediates[1];
}

absl::StatusOr<BufferVec> SparseAutoEncoderLayer::bwd(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    Tape tape) {
  RETURN_IF_ERROR(internal::ValidateExecutor(executor_, executor,
                                             "SparseAutoEncoderLayer"));
  if ((output_gradients.size() != 1 && output_gradients.size() != 3) ||
      (tape.intermediates.size() != 2 && tape.intermediates.size() != 3)) {
    return absl::InvalidArgumentError(
        "SAE bwd expects d_x1, optionally d_z and d_D, and a matching tape");
  }
  ASSIGN_OR_RETURN(
      int rows, internal::MatrixRows(executor, output_gradients[0], input_dim_,
                                     "SAE reconstruction gradient"));
  RETURN_IF_ERROR(internal::ValidateTiledExtent(rows, "SAE rows"));
  RETURN_IF_ERROR(internal::ValidateBuffer(
      executor, tape.intermediates[0],
      static_cast<size_t>(rows) * input_dim_ *
          internal::ActivationElementBytes(output_type_),
      "SAE saved input"));
  RETURN_IF_ERROR(internal::ValidateBuffer(
      executor, tape.intermediates[1],
      static_cast<size_t>(rows) * feature_dim_ *
          internal::ActivationElementBytes(output_type_),
      "SAE saved latent activations"));
  if (output_gradients.size() == 3) {
    RETURN_IF_ERROR(internal::ValidateBuffer(
        executor, output_gradients[1],
        static_cast<size_t>(rows) * feature_dim_ * sizeof(float),
        "SAE auxiliary latent gradient"));
    RETURN_IF_ERROR(internal::ValidateBuffer(
        executor, output_gradients[2],
        static_cast<size_t>(input_dim_) * feature_dim_ * sizeof(float),
        "SAE direct decoder gradient"));
  }

  ASSIGN_OR_RETURN(
      auto latent_gradient,
      Buffer::Allocate(
          executor, static_cast<size_t>(rows) * feature_dim_ * sizeof(float)));
  ASSIGN_OR_RETURN(
      auto preactivation_gradient,
      Buffer::Allocate(
          executor, static_cast<size_t>(rows) * feature_dim_ * sizeof(float)));
  ASSIGN_OR_RETURN(auto input_gradient,
                   Buffer::Allocate(executor, static_cast<size_t>(rows) *
                                                  input_dim_ * sizeof(float)));
  const int latent_blocks =
      internal::TileCount(rows) * internal::TileCount(feature_dim_);
  const int input_blocks =
      internal::TileCount(rows) * internal::TileCount(input_dim_);
  const int encoder_weight_blocks =
      internal::TileCount(feature_dim_) * internal::TileCount(input_dim_);
  const int decoder_weight_blocks =
      internal::TileCount(input_dim_) * internal::TileCount(feature_dim_);

  if (output_type_ == DataType::BF16) {
    SparseLatentGradientKernel<__nv_bfloat16>
        <<<latent_blocks, 1, 0, executor.stream()>>>(
            static_cast<const float*>(output_gradients[0].data()),
            static_cast<const float*>(weights_[2].data()), rows, input_dim_,
            feature_dim_, static_cast<float*>(latent_gradient.data()));
  } else {
    SparseLatentGradientKernel<float>
        <<<latent_blocks, 1, 0, executor.stream()>>>(
            static_cast<const float*>(output_gradients[0].data()),
            static_cast<const float*>(weights_[2].data()), rows, input_dim_,
            feature_dim_, static_cast<float*>(latent_gradient.data()));
  }
  if (output_gradients.size() == 3) {
    const int elements = rows * feature_dim_;
    AddFloatKernel<<<internal::TileCount(elements), 1, 0, executor.stream()>>>(
        static_cast<const float*>(output_gradients[1].data()), elements,
        static_cast<float*>(latent_gradient.data()));
  }
  const int latent_elements = rows * feature_dim_;
  if (output_type_ == DataType::BF16) {
    SparseReluBackwardKernel<__nv_bfloat16>
        <<<internal::TileCount(latent_elements), 1, 0, executor.stream()>>>(
            static_cast<const __nv_bfloat16*>(tape.intermediates[1].data()),
            static_cast<const float*>(latent_gradient.data()), latent_elements,
            static_cast<float*>(preactivation_gradient.data()));
    SparseInputGradientKernel<__nv_bfloat16>
        <<<input_blocks, 1, 0, executor.stream()>>>(
            static_cast<const float*>(preactivation_gradient.data()),
            static_cast<const float*>(weights_[0].data()), rows, input_dim_,
            feature_dim_, static_cast<float*>(input_gradient.data()));
    SparseEncoderWeightGradientKernel<__nv_bfloat16>
        <<<encoder_weight_blocks, 1, 0, executor.stream()>>>(
            static_cast<const __nv_bfloat16*>(tape.intermediates[0].data()),
            static_cast<const float*>(weights_[3].data()),
            static_cast<const float*>(preactivation_gradient.data()), rows,
            input_dim_, feature_dim_,
            static_cast<float*>(gradients_[0].data()));
    SparseDecoderWeightGradientKernel<__nv_bfloat16>
        <<<decoder_weight_blocks, 1, 0, executor.stream()>>>(
            static_cast<const float*>(output_gradients[0].data()),
            static_cast<const __nv_bfloat16*>(tape.intermediates[1].data()),
            rows, input_dim_, feature_dim_,
            static_cast<float*>(gradients_[2].data()));
  } else {
    SparseReluBackwardKernel<float>
        <<<internal::TileCount(latent_elements), 1, 0, executor.stream()>>>(
            static_cast<const float*>(tape.intermediates[1].data()),
            static_cast<const float*>(latent_gradient.data()), latent_elements,
            static_cast<float*>(preactivation_gradient.data()));
    SparseInputGradientKernel<float><<<input_blocks, 1, 0, executor.stream()>>>(
        static_cast<const float*>(preactivation_gradient.data()),
        static_cast<const float*>(weights_[0].data()), rows, input_dim_,
        feature_dim_, static_cast<float*>(input_gradient.data()));
    SparseEncoderWeightGradientKernel<float>
        <<<encoder_weight_blocks, 1, 0, executor.stream()>>>(
            static_cast<const float*>(tape.intermediates[0].data()),
            static_cast<const float*>(weights_[3].data()),
            static_cast<const float*>(preactivation_gradient.data()), rows,
            input_dim_, feature_dim_,
            static_cast<float*>(gradients_[0].data()));
    SparseDecoderWeightGradientKernel<float>
        <<<decoder_weight_blocks, 1, 0, executor.stream()>>>(
            static_cast<const float*>(output_gradients[0].data()),
            static_cast<const float*>(tape.intermediates[1].data()), rows,
            input_dim_, feature_dim_,
            static_cast<float*>(gradients_[2].data()));
  }
  SparseEncoderBiasGradientKernel<<<internal::TileCount(feature_dim_), 1, 0,
                                    executor.stream()>>>(
      static_cast<const float*>(preactivation_gradient.data()), rows,
      feature_dim_, static_cast<float*>(gradients_[1].data()));
  if (output_gradients.size() == 3) {
    const int decoder_elements = input_dim_ * feature_dim_;
    AddFloatKernel<<<internal::TileCount(decoder_elements), 1, 0,
                     executor.stream()>>>(
        static_cast<const float*>(output_gradients[2].data()), decoder_elements,
        static_cast<float*>(gradients_[2].data()));
  }
  SparseDecoderBiasGradientKernel<<<internal::TileCount(input_dim_), 1, 0,
                                    executor.stream()>>>(
      static_cast<const float*>(output_gradients[0].data()),
      static_cast<const float*>(input_gradient.data()), rows, input_dim_,
      static_cast<float*>(gradients_[3].data()));
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(),
                                   "sparse autoencoder backward launch"));
  return BufferVec{std::move(input_gradient)};
}

absl::StatusOr<std::unique_ptr<SparseAutoEncoderLossLayer>>
SparseAutoEncoderLossLayer::Create(cuda::Executor& executor, int input_dim,
                                   int feature_dim, float sparsity_penalty,
                                   DataType data_type) {
  RETURN_IF_ERROR(internal::ValidateComputeType(data_type));
  RETURN_IF_ERROR(internal::ValidateTiledExtent(input_dim, "input_dim"));
  RETURN_IF_ERROR(internal::ValidateTiledExtent(feature_dim, "feature_dim"));
  if (!std::isfinite(sparsity_penalty) || sparsity_penalty < 0.0f) {
    return absl::InvalidArgumentError(
        "sparsity_penalty must be finite and non-negative");
  }
  return std::unique_ptr<SparseAutoEncoderLossLayer>(
      new SparseAutoEncoderLossLayer(executor, input_dim, feature_dim,
                                     sparsity_penalty, data_type));
}

absl::StatusOr<Buffer> SparseAutoEncoderLossLayer::fwd(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    Tape* tape) const {
  RETURN_IF_ERROR(internal::ValidateExecutor(executor_, executor,
                                             "SparseAutoEncoderLossLayer"));
  if (tape == nullptr) {
    return absl::InvalidArgumentError(
        "SparseAutoEncoderLossLayer fwd requires a non-null tape");
  }
  ASSIGN_OR_RETURN(int rows, ValidateLossInputs(executor, inputs, input_dim_,
                                                feature_dim_, output_type_));
  ASSIGN_OR_RETURN(auto output, Buffer::Allocate(executor, sizeof(float)));
  ASSIGN_OR_RETURN(
      auto decoder_norm,
      Buffer::Allocate(executor,
                       static_cast<size_t>(feature_dim_) * sizeof(float)));
  ASSIGN_OR_RETURN(
      auto row_losses,
      Buffer::Allocate(executor, static_cast<size_t>(rows) * sizeof(float)));
  SparseLossDecoderNormKernel<<<internal::TileCount(feature_dim_), 1, 0,
                                executor.stream()>>>(
      static_cast<const float*>(inputs[3].data()), input_dim_, feature_dim_,
      static_cast<float*>(decoder_norm.data()));
  if (output_type_ == DataType::BF16) {
    SparseLossPerRowKernel<__nv_bfloat16><<<rows, 1, 0, executor.stream()>>>(
        static_cast<const __nv_bfloat16*>(inputs[0].data()),
        static_cast<const __nv_bfloat16*>(inputs[1].data()),
        static_cast<const __nv_bfloat16*>(inputs[2].data()),
        static_cast<const float*>(decoder_norm.data()), rows, input_dim_,
        feature_dim_, sparsity_penalty_,
        static_cast<float*>(row_losses.data()));
  } else {
    SparseLossPerRowKernel<float><<<rows, 1, 0, executor.stream()>>>(
        static_cast<const float*>(inputs[0].data()),
        static_cast<const float*>(inputs[1].data()),
        static_cast<const float*>(inputs[2].data()),
        static_cast<const float*>(decoder_norm.data()), rows, input_dim_,
        feature_dim_, sparsity_penalty_,
        static_cast<float*>(row_losses.data()));
  }
  SparseLossReduceKernel<<<1, 1, 0, executor.stream()>>>(
      static_cast<const float*>(row_losses.data()), rows,
      static_cast<float*>(output.data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "sparse loss forward launch"));
  tape->intermediates.assign(inputs.begin(), inputs.end());
  tape->children.clear();
  return std::move(output);
}

absl::StatusOr<BufferVec> SparseAutoEncoderLossLayer::bwd(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    Tape tape) {
  RETURN_IF_ERROR(internal::ValidateExecutor(executor_, executor,
                                             "SparseAutoEncoderLossLayer"));
  if (!output_gradients.empty()) {
    return absl::InvalidArgumentError(
        "terminal sparse autoencoder loss expects no upstream gradient");
  }
  ASSIGN_OR_RETURN(int rows,
                   ValidateLossInputs(executor, tape.intermediates, input_dim_,
                                      feature_dim_, output_type_));
  ASSIGN_OR_RETURN(auto input_gradient,
                   Buffer::Allocate(executor, static_cast<size_t>(rows) *
                                                  input_dim_ * sizeof(float)));
  ASSIGN_OR_RETURN(auto reconstruction_gradient,
                   Buffer::Allocate(executor, static_cast<size_t>(rows) *
                                                  input_dim_ * sizeof(float)));
  ASSIGN_OR_RETURN(
      auto latent_gradient,
      Buffer::Allocate(
          executor, static_cast<size_t>(rows) * feature_dim_ * sizeof(float)));
  ASSIGN_OR_RETURN(
      auto decoder_gradient,
      Buffer::Allocate(executor, static_cast<size_t>(input_dim_) *
                                     feature_dim_ * sizeof(float)));
  ASSIGN_OR_RETURN(
      auto decoder_scale,
      Buffer::Allocate(executor,
                       static_cast<size_t>(feature_dim_) * sizeof(float)));
  const int reconstruction_elements = rows * input_dim_;
  if (output_type_ == DataType::BF16) {
    SparseLossReconstructionGradientKernel<__nv_bfloat16>
        <<<internal::TileCount(reconstruction_elements), 1, 0,
           executor.stream()>>>(
            static_cast<const __nv_bfloat16*>(tape.intermediates[0].data()),
            static_cast<const __nv_bfloat16*>(tape.intermediates[1].data()),
            reconstruction_elements, static_cast<float*>(input_gradient.data()),
            static_cast<float*>(reconstruction_gradient.data()));
    SparseLossDecoderScaleKernel<__nv_bfloat16>
        <<<internal::TileCount(feature_dim_), 1, 0, executor.stream()>>>(
            static_cast<const __nv_bfloat16*>(tape.intermediates[2].data()),
            static_cast<const float*>(tape.intermediates[3].data()), rows,
            input_dim_, feature_dim_, sparsity_penalty_,
            static_cast<float*>(decoder_scale.data()));
  } else {
    SparseLossReconstructionGradientKernel<float>
        <<<internal::TileCount(reconstruction_elements), 1, 0,
           executor.stream()>>>(
            static_cast<const float*>(tape.intermediates[0].data()),
            static_cast<const float*>(tape.intermediates[1].data()),
            reconstruction_elements, static_cast<float*>(input_gradient.data()),
            static_cast<float*>(reconstruction_gradient.data()));
    SparseLossDecoderScaleKernel<float>
        <<<internal::TileCount(feature_dim_), 1, 0, executor.stream()>>>(
            static_cast<const float*>(tape.intermediates[2].data()),
            static_cast<const float*>(tape.intermediates[3].data()), rows,
            input_dim_, feature_dim_, sparsity_penalty_,
            static_cast<float*>(decoder_scale.data()));
  }
  const int64_t decoder_elements =
      static_cast<int64_t>(input_dim_) * feature_dim_;
  const int decoder_blocks = static_cast<int>(
      std::min<int64_t>(decoder_elements / internal::kDenseTile,
                        std::numeric_limits<int>::max()));
  SparseLossDecoderGradientKernel<<<decoder_blocks, 1, 0, executor.stream()>>>(
      static_cast<const float*>(tape.intermediates[3].data()),
      static_cast<const float*>(decoder_scale.data()), decoder_elements,
      feature_dim_, decoder_blocks,
      static_cast<float*>(decoder_gradient.data()));
  SparseLossLatentGradientKernel<<<internal::TileCount(feature_dim_), 1, 0,
                                   executor.stream()>>>(
      static_cast<const float*>(tape.intermediates[3].data()), rows, input_dim_,
      feature_dim_, sparsity_penalty_,
      static_cast<float*>(latent_gradient.data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "sparse loss backward launch"));
  return BufferVec{std::move(input_gradient),
                   std::move(reconstruction_gradient),
                   std::move(latent_gradient), std::move(decoder_gradient)};
}

}  // namespace pluto::llm
