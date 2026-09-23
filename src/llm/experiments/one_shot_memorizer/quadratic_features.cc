#include "src/llm/experiments/one_shot_memorizer/quadratic_features.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cstdint>
#include <limits>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "src/llm/layers/util.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

constexpr int kInputWidth = QuadraticFeaturesLayer::kInputWidth;
constexpr int kOutputWidth = QuadraticFeaturesLayer::kOutputWidth;

// One tile owns one token row. The physical row has only 152 columns, so all
// loads/stores mask the unused part of the 256-lane tile. Neither these lanes
// nor a neighboring row can enter the feature calculation.
__tile_global__ void QuadraticFeaturesKernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output) {
  namespace ct = ::cuda::tiles;
  using IndexTile = ct::tile<int, ct::shape<256>>;
  const int row = ct::bid().x;
  const auto column = ct::iota<IndexTile>();
  const auto valid = column < kOutputWidth;
  const auto linear = column < kInputWidth;
  auto left = ct::select(linear, column, ct::zeros<IndexTile>());
  auto right = ct::zeros<IndexTile>();
  // Upper-triangle row i starts after 16 linear terms and sum_{h<i}(16-h)
  // products. Deriving indices here avoids a lookup table or device weights.
  for (int i = 0; i < kInputWidth; ++i) {
    const int begin = kInputWidth + i * kInputWidth - i * (i - 1) / 2;
    const auto in_group =
        (column >= begin) & (column < begin + kInputWidth - i);
    left = ct::select(in_group, ct::full<IndexTile>(i), left);
    right = ct::select(in_group, column - begin + i, right);
  }
  const auto first = ct::load_masked(input + row * kInputWidth + left, valid);
  const auto second =
      ct::load_masked(input + row * kInputWidth + right, valid & !linear);
  const auto product =
      ct::element_cast<float>(first) * ct::element_cast<float>(second);
  const auto result =
      ct::select(linear, first, ct::element_cast<__nv_bfloat16>(product));
  ct::store_masked(output + row * kOutputWidth + column, result, valid);
}

}  // namespace

absl::StatusOr<size_t> QuadraticFeaturesOutputBytes(size_t input_bytes,
                                                    int sequence_length) {
  constexpr size_t kElementBytes = sizeof(uint16_t);
  constexpr size_t kInputRowBytes = kInputWidth * kElementBytes;
  constexpr size_t kOutputRowBytes = kOutputWidth * kElementBytes;
  constexpr size_t kMaxRows = std::numeric_limits<int>::max() / kOutputWidth;
  if (sequence_length <= 0 || static_cast<size_t>(sequence_length) > kMaxRows)
    return absl::InvalidArgumentError(
        "quadratic sequence length must fit positive signed-int extents");
  if (input_bytes == 0 || input_bytes % kInputRowBytes != 0)
    return absl::InvalidArgumentError(
        "quadratic input must contain nonempty, complete 16-wide BF16 rows");
  const size_t rows = input_bytes / kInputRowBytes;
  if (rows % sequence_length != 0)
    return absl::InvalidArgumentError(
        "quadratic input rows must contain complete sequence samples");
  if (rows > kMaxRows ||
      rows > std::numeric_limits<size_t>::max() / kOutputRowBytes)
    return absl::InvalidArgumentError(
        "quadratic output exceeds indexing range");
  return rows * kOutputRowBytes;
}

absl::StatusOr<std::unique_ptr<QuadraticFeaturesLayer>>
QuadraticFeaturesLayer::Create(cuda::Executor& executor, int sequence_length) {
  if (sequence_length <= 0)
    return absl::InvalidArgumentError(
        "quadratic sequence length must be positive");
  // Validate one sample without allocating it; fwd also checks the runtime
  // batch size before any device allocation or kernel launch.
  RETURN_IF_ERROR(
      QuadraticFeaturesOutputBytes(
          static_cast<size_t>(sequence_length) * kInputWidth * sizeof(uint16_t),
          sequence_length)
          .status());
  return absl::WrapUnique(
      new QuadraticFeaturesLayer(executor, sequence_length));
}

absl::StatusOr<FwdResult> QuadraticFeaturesLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks*) const {
  RETURN_IF_ERROR(internal::ValidateExecutor(executor_, executor,
                                             "QuadraticFeaturesLayer"));
  if (inputs.size() != 1)
    return absl::InvalidArgumentError("quadratic forward expects one input");
  if (&inputs[0].executor() != &executor)
    return absl::InvalidArgumentError(
        "quadratic input belongs to another executor");
  ASSIGN_OR_RETURN(
      const size_t bytes,
      QuadraticFeaturesOutputBytes(inputs[0].size_bytes(), sequence_length_));
  const int rows = static_cast<int>(inputs[0].size_bytes() /
                                    (kInputWidth * sizeof(uint16_t)));
  ASSIGN_OR_RETURN(auto output, Buffer::Allocate(executor, bytes));
  QuadraticFeaturesKernel<<<rows, 1, 0, executor.stream()>>>(
      static_cast<const __nv_bfloat16*>(inputs[0].data()),
      static_cast<__nv_bfloat16*>(output.data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "QuadraticFeaturesKernel launch"));
  return FwdResult{{std::move(output)}, {}};
}

absl::StatusOr<BufferVec> QuadraticFeaturesLayer::bwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer>, BackwardState,
    LayerHooks*) {
  RETURN_IF_ERROR(internal::ValidateExecutor(executor_, executor,
                                             "QuadraticFeaturesLayer"));
  return absl::UnimplementedError(
      "QuadraticFeaturesLayer is an inference-only experimental feature map");
}

}  // namespace pluto::llm::one_shot_memorizer
