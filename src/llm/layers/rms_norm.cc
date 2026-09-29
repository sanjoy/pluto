#include "src/llm/layers/rms_norm.h"

#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cmath>
#include <utility>

#include "absl/memory/memory.h"
#include "src/llm/layers/util.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

// One reduction tile covers the entire token, so the largest compiled tile
// determines the width limit. Other layers need not share this constraint.
constexpr int kMaximumDimension = 16384;

template <int Size>
__tile_global__ void RmsNormKernel(const float* input, const float* weight,
                                   int width, float epsilon, float* output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto x = ct::partition_view{ct::tensor_span{input, ct::extents{width}},
                              ct::shape<Size>{}};
  auto w = ct::partition_view{ct::tensor_span{weight, ct::extents{width}},
                              ct::shape<Size>{}};
  auto y = ct::partition_view{ct::tensor_span{output, ct::extents{width}},
                              ct::shape<Size>{}};
  // Masked lanes contribute zero, but divide by the actual width rather than
  // the tile size. Zero-centered weights use 1+w, not w.
  auto value = x.load_masked(0);
  auto result =
      value * ct::rsqrt(ct::sum(value * value, 0_ic) / float(width) + epsilon) *
      (1.0f + w.load_masked(0));
  y.store_masked(result, 0);
}

}  // namespace

absl::StatusOr<std::unique_ptr<RmsNormLayer>> RmsNormLayer::Create(
    cuda::Executor& executor, int width, Buffer weight, float epsilon) {
  if (width <= 0 || width > kMaximumDimension || !std::isfinite(epsilon) ||
      epsilon <= 0)
    return absl::InvalidArgumentError("invalid RMSNorm dimensions or epsilon");
  RETURN_IF_ERROR(internal::ValidateBuffer(
      executor, weight, size_t(width) * sizeof(float), "RMSNorm weight"));
  return absl::WrapUnique(
      new RmsNormLayer(executor, width, std::move(weight), epsilon));
}

absl::StatusOr<FwdResult> RmsNormLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks*) const {
  RETURN_IF_ERROR(internal::ValidateExecutor(executor_, executor, "RMSNorm"));
  RETURN_IF_ERROR(internal::ValidateBFloat16Inputs(executor, inputs, {width_}));
  ASSIGN_OR_RETURN(Buffer input, internal::ToFloat(executor, inputs[0], width_));
  ASSIGN_OR_RETURN(Buffer result,
                   internal::AllocateFloatVector(executor, width_));
  const float* x = static_cast<const float*>(input.data());
  const float* w = static_cast<const float*>(weight_.data());
  float* y = static_cast<float*>(result.data());
  if (width_ <= 256)
    RmsNormKernel<256>
        <<<1, 1, 0, executor.stream()>>>(x, w, width_, epsilon_, y);
  else if (width_ <= 8192)
    RmsNormKernel<8192>
        <<<1, 1, 0, executor.stream()>>>(x, w, width_, epsilon_, y);
  else
    RmsNormKernel<kMaximumDimension>
        <<<1, 1, 0, executor.stream()>>>(x, w, width_, epsilon_, y);
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(), "RmsNormKernel"));
  ASSIGN_OR_RETURN(Buffer output,
                   internal::ToBFloat16(executor, result, width_));
  return FwdResult{{std::move(output)}, {}};
}

absl::StatusOr<BufferVec> RmsNormLayer::bwd_impl(cuda::Executor&,
                                                 absl::Span<const Buffer>,
                                                 BackwardState, LayerHooks*) {
  return absl::UnimplementedError("frozen RMSNorm does not support backward");
}

}  // namespace pluto::llm
