#include "src/llm/experiments/memorize_general_facts/fit_attention_readout/margin_loss.h"

#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include "absl/status/status.h"
#include "src/util/status_macros.h"

namespace pluto::llm::fit_attention_readout {
namespace {

constexpr int kTile = 256;

// One tile program owns a complete row. No atomics or floating-point sums are
// necessary: find the best non-target value, then choose its lowest index.
__tile_global__ void SquaredMarginLossKernel(
    const float* __restrict__ logits, const int* __restrict__ targets, int rows,
    int stride, int vocabulary_size, float required_margin, int normalizer,
    float* __restrict__ losses, float* __restrict__ gradients,
    int* __restrict__ predictions, float* __restrict__ margins) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto logits_view =
      ct::partition_view{ct::tensor_span{logits, ct::extents{rows, stride}},
                         ct::shape{1_ic, 256_ic}};
  auto scalar_logits_view =
      ct::partition_view{ct::tensor_span{logits, ct::extents{rows, stride}},
                         ct::shape{1_ic, 1_ic}};
  auto gradients_view =
      ct::partition_view{ct::tensor_span{gradients, ct::extents{rows, stride}},
                         ct::shape{1_ic, 256_ic}};
  auto targets_view = ct::partition_view{
      ct::tensor_span{targets, ct::extents{rows}}, ct::shape{1_ic}};
  auto losses_view = ct::partition_view{
      ct::tensor_span{losses, ct::extents{rows}}, ct::shape{1_ic}};
  auto predictions_view = ct::partition_view{
      ct::tensor_span{predictions, ct::extents{rows}}, ct::shape{1_ic}};
  auto margins_view = ct::partition_view{
      ct::tensor_span{margins, ct::extents{rows}}, ct::shape{1_ic}};
  const int row = ct::bid().x;
  const int target = static_cast<int>(targets_view.load(row));
  const int output_tiles = 1 + (stride - 1) / kTile;
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  auto zeros = ct::zeros<ct::tile<float, ct::shape<1, 256>>>();
  if (target < 0 || target >= vocabulary_size) {
    const bool ignored = target == -1;
    auto scalar =
        ct::full<ct::tile<float, ct::shape<1>>>(ignored ? 0.0f : kNaN);
    losses_view.store(scalar, row);
    margins_view.store(scalar, row);
    predictions_view.store(
        ct::full<ct::tile<int, ct::shape<1>>>(ignored ? -1 : -2), row);
    for (int tile = 0; tile < output_tiles; ++tile) {
      auto ids = ct::iota<ct::tile<int, ct::shape<1, 256>>>() + tile * kTile;
      auto values =
          ct::full<ct::tile<float, ct::shape<1, 256>>>(ignored ? 0.0f : kNaN);
      gradients_view.store_masked(
          ct::select(ids < vocabulary_size, values, zeros), row, tile);
    }
    return;
  }

  constexpr float kLargest = std::numeric_limits<float>::max();
  auto best = ct::full<ct::tile<float, ct::shape<1, 256>>>(-kLargest);
  auto no_id = ct::full<ct::tile<int, ct::shape<1, 256>>>(0x7fffffff);
  auto best_id = no_id;
  auto nonfinite = ct::zeros<ct::tile<int, ct::shape<1, 256>>>();
  const int vocabulary_tiles = 1 + (vocabulary_size - 1) / kTile;
  for (int tile = 0; tile < vocabulary_tiles; ++tile) {
    auto ids = ct::iota<ct::tile<int, ct::shape<1, 256>>>() + tile * kTile;
    auto values = logits_view.load_masked(row, tile);
    auto logical = ids < vocabulary_size;
    auto invalid =
        (values != values) | (values > kLargest) | (values < -kLargest);
    nonfinite = ct::max(nonfinite, ct::element_cast<int>(logical & invalid));
    auto wins = logical & (ids != target) &
                ((values > best) | ((values == best) & (ids < best_id)));
    best = ct::select(wins, values, best);
    best_id = ct::select(wins, ids, best_id);
  }
  auto competitor = ct::reduce_max(best, 1_ic);
  auto competitor_id =
      ct::reduce_min(ct::select(best == competitor, best_id, no_id), 1_ic);
  auto target_value = scalar_logits_view.load(row, target);
  auto actual_margin = target_value - competitor;
  auto violation = ct::max(required_margin - actual_margin, 0.0f);
  auto scale = (2.0f / static_cast<float>(normalizer)) * violation;
  auto invalid = ct::reduce_max(nonfinite, 1_ic) != 0;
  auto target_wins = (target_value > competitor) |
                     ((target_value == competitor) & (target < competitor_id));
  auto target_id = ct::full<ct::tile<int, ct::shape<1, 1>>>(target);
  auto invalid_id = ct::full<ct::tile<int, ct::shape<1, 1>>>(-2);
  auto nan = ct::full<ct::tile<float, ct::shape<1, 1>>>(kNaN);
  auto prediction = ct::select(target_wins, target_id, competitor_id);
  predictions_view.store(
      ct::reshape(ct::select(invalid, invalid_id, prediction), ct::shape{1_ic}),
      row);
  losses_view.store(ct::reshape(ct::select(invalid, nan, violation * violation),
                                ct::shape{1_ic}),
                    row);
  margins_view.store(
      ct::reshape(ct::select(invalid, nan, actual_margin), ct::shape{1_ic}),
      row);
  for (int tile = 0; tile < output_tiles; ++tile) {
    auto ids = ct::iota<ct::tile<int, ct::shape<1, 256>>>() + tile * kTile;
    auto derivative = (ct::element_cast<float>(ids == competitor_id) -
                       ct::element_cast<float>(ids == target)) *
                      scale;
    derivative =
        ct::select(invalid, ct::full<ct::tile<float, ct::shape<1, 256>>>(kNaN),
                   derivative);
    gradients_view.store_masked(
        ct::select(ids < vocabulary_size, derivative, zeros), row, tile);
  }
}

}  // namespace

absl::StatusOr<MarginLossResult> SquaredMarginLoss(cuda::Executor& executor,
                                                   const cuda::Buffer& logits,
                                                   const cuda::Buffer& targets,
                                                   int vocabulary_size,
                                                   float required_margin,
                                                   int normalizer) {
  if (&logits.executor() != &executor || &targets.executor() != &executor)
    return absl::InvalidArgumentError("margin loss requires one executor");
  if (vocabulary_size < 2 || !std::isfinite(required_margin) ||
      required_margin <= 0 || normalizer <= 0 || targets.size_bytes() == 0 ||
      targets.size_bytes() % sizeof(int) != 0)
    return absl::InvalidArgumentError("invalid margin loss arguments");
  const size_t rows = targets.size_bytes() / sizeof(int);
  if (logits.size_bytes() % rows != 0 ||
      logits.size_bytes() / rows % sizeof(float) != 0)
    return absl::InvalidArgumentError("margin logits must contain FP32 rows");
  const size_t stride = logits.size_bytes() / rows / sizeof(float);
  if (stride < static_cast<size_t>(vocabulary_size) ||
      rows > std::numeric_limits<int>::max() ||
      stride > std::numeric_limits<int>::max())
    return absl::InvalidArgumentError("invalid margin loss matrix dimensions");
  ASSIGN_OR_RETURN(auto losses,
                   cuda::Buffer::Allocate(executor, rows * sizeof(float)));
  ASSIGN_OR_RETURN(auto gradients,
                   cuda::Buffer::Allocate(executor, logits.size_bytes()));
  ASSIGN_OR_RETURN(auto predictions,
                   cuda::Buffer::Allocate(executor, rows * sizeof(int)));
  ASSIGN_OR_RETURN(auto margins,
                   cuda::Buffer::Allocate(executor, rows * sizeof(float)));
  SquaredMarginLossKernel<<<rows, 1, 0, executor.stream()>>>(
      static_cast<const float*>(logits.data()),
      static_cast<const int*>(targets.data()), static_cast<int>(rows),
      static_cast<int>(stride), vocabulary_size, required_margin, normalizer,
      static_cast<float*>(losses.data()), static_cast<float*>(gradients.data()),
      static_cast<int*>(predictions.data()),
      static_cast<float*>(margins.data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "SquaredMarginLossKernel"));
  return MarginLossResult{std::move(losses), std::move(gradients),
                          std::move(predictions), std::move(margins)};
}

}  // namespace pluto::llm::fit_attention_readout
