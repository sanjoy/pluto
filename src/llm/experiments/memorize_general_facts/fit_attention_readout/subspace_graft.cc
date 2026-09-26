#include "src/llm/experiments/memorize_general_facts/fit_attention_readout/subspace_graft.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cmath>
#include <cstddef>
#include <limits>

#include "absl/status/status.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::llm::fit_attention_readout {
namespace {

constexpr int kRowsPerTile = 4;
constexpr int kMaximumWidth = 1024;
constexpr double kOrthonormalTolerance = 1e-5;

// One tile handles several complete rows; the plane is broadcast across them.
// Masked loads zero-pad both the rows and channels, so padded lanes contribute
// nothing to either projection. No host loop or temporary is needed per row.
template <int Width>
__tile_global__ void GraftBf16PlaneKernel(
    const __nv_bfloat16* __restrict__ base,
    const __nv_bfloat16* __restrict__ donor, const float* __restrict__ plane,
    int rows, int model_width, __nv_bfloat16* __restrict__ output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto base_view =
      ct::partition_view{ct::tensor_span{base, ct::extents{rows, model_width}},
                         ct::shape<kRowsPerTile, Width>{}};
  auto donor_view =
      ct::partition_view{ct::tensor_span{donor, ct::extents{rows, model_width}},
                         ct::shape<kRowsPerTile, Width>{}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{rows, model_width}},
      ct::shape<kRowsPerTile, Width>{}};
  auto plane_view =
      ct::partition_view{ct::tensor_span{plane, ct::extents{model_width, 2}},
                         ct::shape<Width, 1>{}};
  const int row_tile = ct::bid().x;
  auto base_values =
      ct::element_cast<float>(base_view.load_masked(row_tile, 0));
  auto delta = ct::element_cast<float>(donor_view.load_masked(row_tile, 0)) -
               base_values;
  auto first = ct::transpose(plane_view.load_masked(0, 0));
  auto second = ct::transpose(plane_view.load_masked(0, 1));
  auto first_projection = ct::sum(delta * first, 1_ic);
  auto second_projection = ct::sum(delta * second, 1_ic);
  auto correction = first_projection * first + second_projection * second;
  // Preserve the original BF16 values (including signed zero) whenever the
  // projection leaves a coordinate unchanged.
  auto grafted =
      ct::select(correction == 0.0f, base_values, base_values + correction);
  output_view.store_masked(ct::element_cast<__nv_bfloat16>(grafted), row_tile,
                           0);
}

}  // namespace

absl::StatusOr<cuda::Buffer> GraftBf16Plane(cuda::Executor& executor,
                                            const cuda::Buffer& base,
                                            const cuda::Buffer& donor,
                                            int model_width,
                                            absl::Span<const float> plane) {
  if (model_width < 2 || model_width > kMaximumWidth)
    return absl::InvalidArgumentError(
        "BF16 plane graft requires width in [2, 1024]");
  if (plane.size() != 2 * static_cast<size_t>(model_width))
    return absl::InvalidArgumentError(
        "BF16 plane graft requires width x 2 basis values");
  double first_norm = 0.0;
  double second_norm = 0.0;
  double cross = 0.0;
  for (int column = 0; column < model_width; ++column) {
    const double first = plane[2 * column];
    const double second = plane[2 * column + 1];
    if (!std::isfinite(first) || !std::isfinite(second))
      return absl::InvalidArgumentError(
          "BF16 plane graft basis must be finite");
    first_norm += first * first;
    second_norm += second * second;
    cross += first * second;
  }
  if (std::abs(first_norm - 1.0) > kOrthonormalTolerance ||
      std::abs(second_norm - 1.0) > kOrthonormalTolerance ||
      std::abs(cross) > kOrthonormalTolerance)
    return absl::InvalidArgumentError(
        "BF16 plane graft basis must be orthonormal");
  if (&base.executor() != &executor || &donor.executor() != &executor)
    return absl::InvalidArgumentError(
        "BF16 plane graft input executor mismatch");
  const size_t row_bytes =
      static_cast<size_t>(model_width) * sizeof(__nv_bfloat16);
  if (base.size_bytes() == 0 || base.size_bytes() != donor.size_bytes() ||
      base.size_bytes() % row_bytes != 0)
    return absl::InvalidArgumentError(
        "BF16 plane graft inputs must have equal, nonempty whole-row shapes");
  const size_t rows = base.size_bytes() / row_bytes;
  if (rows > std::numeric_limits<int>::max())
    return absl::InvalidArgumentError("BF16 plane graft has too many rows");

  ASSIGN_OR_RETURN(auto host_plane,
                   cuda::PageLockedHostArray<float>::CopyFrom(executor, plane));
  ASSIGN_OR_RETURN(auto device_plane,
                   cuda::Buffer::Allocate(executor, host_plane.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(device_plane.data(), host_plane.data(),
                      host_plane.size_bytes(), cudaMemcpyHostToDevice,
                      executor.stream()),
      "upload BF16 graft plane"));
  ASSIGN_OR_RETURN(auto output,
                   cuda::Buffer::Allocate(executor, base.size_bytes()));
  const unsigned int blocks = (rows + kRowsPerTile - 1) / kRowsPerTile;
  if (model_width <= 128) {
    GraftBf16PlaneKernel<128><<<blocks, 1, 0, executor.stream()>>>(
        static_cast<const __nv_bfloat16*>(base.data()),
        static_cast<const __nv_bfloat16*>(donor.data()),
        static_cast<const float*>(device_plane.data()), static_cast<int>(rows),
        model_width, static_cast<__nv_bfloat16*>(output.data()));
  } else {
    GraftBf16PlaneKernel<1024><<<blocks, 1, 0, executor.stream()>>>(
        static_cast<const __nv_bfloat16*>(base.data()),
        static_cast<const __nv_bfloat16*>(donor.data()),
        static_cast<const float*>(device_plane.data()), static_cast<int>(rows),
        model_width, static_cast<__nv_bfloat16*>(output.data()));
  }
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(), "GraftBf16PlaneKernel"));
  // Both device and pinned-host temporaries free after their queued uses.
  return output;
}

}  // namespace pluto::llm::fit_attention_readout
