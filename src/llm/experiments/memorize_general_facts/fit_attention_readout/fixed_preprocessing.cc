#include "src/llm/experiments/memorize_general_facts/fit_attention_readout/fixed_preprocessing.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

#include "absl/status/status.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::llm::fit_attention_readout {
namespace {

constexpr int kMaximumWidth = 128;
constexpr int kRowsPerTile = 4;

// Coefficients are output-major. Each GPU tile reduces only a single token's
// channels; batching several rows never makes one token depend on another.
std::vector<float> MakeCoefficients(int width,
                                    const FixedPreprocessingOptions& options) {
  std::vector<float> coefficients(width * width);
  const double pi = std::numbers::pi;
  const double dc_scale = 1.0 / std::sqrt(static_cast<double>(width));
  if (options.kind == FixedPreprocessingKind::kRandomFourier) {
    // Box-Muller with an explicit mt19937-to-uniform conversion avoids the
    // implementation-dependent normal_distribution algorithm. Only these
    // input-independent coefficients are computed on the host.
    std::mt19937 random(options.seed);
    for (int output = 0; output < width / 2; ++output)
      for (int input = 0; input < width; ++input) {
        const double u = (static_cast<double>(random()) + 0.5) / 4294967296.0;
        const double v = (static_cast<double>(random()) + 0.5) / 4294967296.0;
        const float value =
            dc_scale * std::sqrt(-2.0 * std::log(u)) * std::cos(2.0 * pi * v);
        coefficients[output * width + input] = value;
        coefficients[(output + width / 2) * width + input] = value;
      }
    return coefficients;
  }
  for (int output = 0; output < width; ++output)
    for (int input = 0; input < width; ++input) {
      double coefficient;
      if (options.kind == FixedPreprocessingKind::kDct) {
        coefficient = (output == 0 ? dc_scale : std::sqrt(2.0) * dc_scale) *
                      std::cos(pi * (input + 0.5) * output / width);
      } else if (output == 0) {
        coefficient = dc_scale;
      } else if (output == width - 1) {
        coefficient = input % 2 == 0 ? dc_scale : -dc_scale;
      } else {
        const int frequency = (output + 1) / 2;
        const double angle = 2.0 * pi * input * frequency / width;
        coefficient = std::sqrt(2.0) * dc_scale *
                      (output % 2 == 1 ? std::cos(angle) : -std::sin(angle));
      }
      coefficients[output * width + input] = coefficient;
    }
  return coefficients;
}

__tile_global__ void ProjectKernel(const __nv_bfloat16* input,
                                   const float* coefficients, int rows,
                                   int width, bool fourier, float scale,
                                   __nv_bfloat16* output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto input_view =
      ct::partition_view{ct::tensor_span{input, ct::extents{rows, width}},
                         ct::shape<kRowsPerTile, kMaximumWidth>{}};
  auto coefficients_view = ct::partition_view{
      ct::tensor_span{coefficients, ct::extents{width, width}},
      ct::shape<1, kMaximumWidth>{}};
  auto output_view =
      ct::partition_view{ct::tensor_span{output, ct::extents{rows, width}},
                         ct::shape<kRowsPerTile, 1>{}};
  const int row_tile = ct::bid().x;
  const int output_column = ct::bid().y;
  auto values = ct::element_cast<float>(input_view.load_masked(row_tile, 0));
  auto weights = coefficients_view.load_masked(output_column, 0);
  auto projected = ct::sum(values * weights, 1_ic);
  if (fourier) {
    auto phase = scale * projected;
    projected = output_column < width / 2 ? ct::cos(phase) : ct::sin(phase);
  }
  output_view.store_masked(ct::element_cast<__nv_bfloat16>(projected), row_tile,
                           output_column);
}

__tile_global__ void ElementwiseKernel(const __nv_bfloat16* input, int count,
                                       int kind, float scale,
                                       __nv_bfloat16* output) {
  namespace ct = ::cuda::tiles;
  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{count}}, ct::shape<128>{}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{count}}, ct::shape<128>{}};
  const int tile = ct::bid().x;
  auto values = ct::element_cast<float>(input_view.load_masked(tile));
  auto result = values;
  if (kind == static_cast<int>(FixedPreprocessingKind::kSin))
    result = ct::sin(scale * values);
  else if (kind == static_cast<int>(FixedPreprocessingKind::kCos))
    result = ct::cos(scale * values);
  else
    result = ct::select(values < 0.0f, -ct::sqrt(-values), ct::sqrt(values));
  output_view.store_masked(ct::element_cast<__nv_bfloat16>(result), tile);
}

}  // namespace

absl::StatusOr<cuda::Buffer> PreprocessHiddenStates(
    cuda::Executor& executor, const cuda::Buffer& input, int width,
    const FixedPreprocessingOptions& options) {
  if (width < 1 || width > kMaximumWidth)
    return absl::InvalidArgumentError(
        "fixed preprocessing width must be 1..128");
  if (&input.executor() != &executor)
    return absl::InvalidArgumentError("fixed preprocessing executor mismatch");
  if (input.size_bytes() == 0 ||
      input.size_bytes() % (width * sizeof(__nv_bfloat16)) != 0 ||
      input.size_bytes() / sizeof(__nv_bfloat16) >
          std::numeric_limits<int>::max())
    return absl::InvalidArgumentError(
        "fixed preprocessing requires nonempty whole BF16 rows of int size");
  if (!std::isfinite(options.scale) || options.scale <= 0.0f)
    return absl::InvalidArgumentError(
        "fixed preprocessing scale must be finite and positive");
  const int kind = static_cast<int>(options.kind);
  if (kind < static_cast<int>(FixedPreprocessingKind::kIdentity) ||
      kind > static_cast<int>(FixedPreprocessingKind::kRandomFourier))
    return absl::InvalidArgumentError("unknown fixed preprocessing kind");
  if (width % 2 != 0 &&
      (options.kind == FixedPreprocessingKind::kRealDft ||
       options.kind == FixedPreprocessingKind::kRandomFourier))
    return absl::InvalidArgumentError(
        "DFT and random Fourier preprocessing require even width");
  if (options.kind == FixedPreprocessingKind::kIdentity)
    return input;
  ASSIGN_OR_RETURN(auto output,
                   cuda::Buffer::Allocate(executor, input.size_bytes()));
  const int count = input.size_bytes() / sizeof(__nv_bfloat16);
  if (options.kind == FixedPreprocessingKind::kDct ||
      options.kind == FixedPreprocessingKind::kRealDft ||
      options.kind == FixedPreprocessingKind::kRandomFourier) {
    const auto coefficients = MakeCoefficients(width, options);
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<float>::CopyFrom(
                                    executor, coefficients));
    ASSIGN_OR_RETURN(auto device,
                     cuda::Buffer::Allocate(executor, host.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(device.data(), host.data(), host.size_bytes(),
                        cudaMemcpyHostToDevice, executor.stream()),
        "upload fixed preprocessing coefficients"));
    const int rows = count / width;
    const dim3 grid((rows + kRowsPerTile - 1) / kRowsPerTile, width);
    ProjectKernel<<<grid, 1, 0, executor.stream()>>>(
        static_cast<const __nv_bfloat16*>(input.data()),
        static_cast<const float*>(device.data()), rows, width,
        options.kind == FixedPreprocessingKind::kRandomFourier, options.scale,
        static_cast<__nv_bfloat16*>(output.data()));
  } else {
    ElementwiseKernel<<<(count + 127) / 128, 1, 0, executor.stream()>>>(
        static_cast<const __nv_bfloat16*>(input.data()), count, kind,
        options.scale, static_cast<__nv_bfloat16*>(output.data()));
  }
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(), "fixed preprocessing"));
  return output;
}

}  // namespace pluto::llm::fit_attention_readout
