#include "src/llm/layers/swiglu.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <utility>

#include "absl/memory/memory.h"
#include "src/llm/layers/util.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

constexpr int kMaximumDimension = 1048576;

__tile_global__ void SwiGluKernel(const float* gate, const float* up, int width,
                                  float* output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto g = ct::partition_view{ct::tensor_span{gate, ct::extents{width}},
                              ct::shape{256_ic}};
  auto u = ct::partition_view{ct::tensor_span{up, ct::extents{width}},
                              ct::shape{256_ic}};
  auto y = ct::partition_view{ct::tensor_span{output, ct::extents{width}},
                              ct::shape{256_ic}};
  const int block = ct::bid().x;
  auto value = g.load_masked(block);
  auto silu = value / (1.0f + ct::exp(-value));
  // BF16 SiLU rounding is observable and must happen before multiplying up.
  silu = ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(silu));
  y.store_masked(silu * u.load_masked(block), block);
}

}  // namespace

absl::StatusOr<std::unique_ptr<SwiGluLayer>> SwiGluLayer::Create(int width) {
  if (width <= 0 || width > kMaximumDimension)
    return absl::InvalidArgumentError("SwiGLU width must be in [1, 1048576]");
  return absl::WrapUnique(new SwiGluLayer(width));
}

absl::StatusOr<FwdResult> SwiGluLayer::fwd_impl(cuda::Executor& executor,
                                                absl::Span<const Buffer> inputs,
                                                LayerHooks*) const {
  RETURN_IF_ERROR(
      internal::ValidateBFloat16Inputs(executor, inputs, {width_, width_}));
  ASSIGN_OR_RETURN(Buffer gate, internal::ToFloat(executor, inputs[0], width_));
  ASSIGN_OR_RETURN(Buffer up, internal::ToFloat(executor, inputs[1], width_));
  ASSIGN_OR_RETURN(Buffer result,
                   internal::AllocateFloatVector(executor, width_));
  SwiGluKernel<<<1 + (width_ - 1) / 256, 1, 0, executor.stream()>>>(
      static_cast<const float*>(gate.data()),
      static_cast<const float*>(up.data()), width_,
      static_cast<float*>(result.data()));
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(), "SwiGluKernel"));
  ASSIGN_OR_RETURN(Buffer output,
                   internal::ToBFloat16(executor, result, width_));
  return FwdResult{{std::move(output)}, {}};
}

absl::StatusOr<BufferVec> SwiGluLayer::bwd_impl(cuda::Executor&,
                                                absl::Span<const Buffer>,
                                                BackwardState, LayerHooks*) {
  return absl::UnimplementedError("SwiGLU does not support backward");
}

}  // namespace pluto::llm
