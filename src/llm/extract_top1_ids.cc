#include "src/llm/extract_top1_ids.h"

#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cstddef>
#include <limits>

#include "absl/status/status.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

constexpr int kVocabularyTile = 256;

// One tile program owns each row. Keep 256 running candidates while scanning
// the vocabulary once, then reduce their values and IDs. Explicit integer tie
// breaking makes the result deterministic without atomics or a softmax. The
// maximum's token ID is also the softmax's top-1 token ID for finite logits.
__tile_global__ void ExtractTop1IdsKernel(const float* __restrict__ logits,
                                          const int* __restrict__ targets,
                                          int rows, int vocabulary_size,
                                          int stride,
                                          int* __restrict__ output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;

  auto logits_view =
      ct::partition_view{ct::tensor_span{logits, ct::extents{rows, stride}},
                         ct::shape{1_ic, 256_ic}};
  auto target_view = ct::partition_view{
      ct::tensor_span{targets, ct::extents{rows}}, ct::shape{1_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{rows}}, ct::shape{1_ic}};

  const int row = ct::bid().x;
  if (static_cast<int>(target_view.load(row)) == -1) {
    // Prompt/padding rows have no prediction. Do not inspect their logits,
    // even for NaNs: they can be uninitialized or otherwise meaningless.
    output_view.store(ct::full<ct::tile<int, ct::shape<1>>>(-1), row);
    return;
  }

  constexpr float kLargestFiniteFloat = 3.402823466e+38f;
  auto best =
      ct::full<ct::tile<float, ct::shape<1, 256>>>(-kLargestFiniteFloat);
  auto no_id = ct::full<ct::tile<int, ct::shape<1, 256>>>(0x7fffffff);
  auto best_id = no_id;
  auto nonfinite = ct::zeros<ct::tile<int, ct::shape<1, 256>>>();
  const int tiles = 1 + (vocabulary_size - 1) / kVocabularyTile;
  for (int tile = 0; tile < tiles; ++tile) {
    auto ids =
        ct::iota<ct::tile<int, ct::shape<1, 256>>>() + tile * kVocabularyTile;
    auto values = logits_view.load_masked(row, tile);
    auto logical = ids < vocabulary_size;
    // Comparisons detect both infinities and NaNs without relying on the
    // floating-point maximum reduction's implementation-specific NaN rule.
    auto invalid = (values != values) | (values > kLargestFiniteFloat) |
                   (values < -kLargestFiniteFloat);
    nonfinite = ct::max(nonfinite, ct::element_cast<int>(logical & invalid));
    auto wins =
        logical & ((values > best) | ((values == best) & (ids < best_id)));
    best = ct::select(wins, values, best);
    best_id = ct::select(wins, ids, best_id);
  }
  auto maximum = ct::reduce_max(best, 1_ic);
  auto winner =
      ct::reduce_min(ct::select(best == maximum, best_id, no_id), 1_ic);
  auto invalid_id = ct::full<ct::tile<int, ct::shape<1, 1>>>(-2);
  auto result =
      ct::select(ct::reduce_max(nonfinite, 1_ic) != 0, invalid_id, winner);
  output_view.store(ct::reshape(result, ct::shape{1_ic}), row);
}

}  // namespace

absl::StatusOr<cuda::Buffer> ExtractTop1Ids(cuda::Executor& executor,
                                            const cuda::Buffer& logits,
                                            const cuda::Buffer& targets,
                                            int vocabulary_size) {
  if (&logits.executor() != &executor || &targets.executor() != &executor)
    return absl::InvalidArgumentError("top-1 inputs require one executor");
  if (vocabulary_size <= 0 || targets.size_bytes() == 0 ||
      targets.size_bytes() % sizeof(int) != 0)
    return absl::InvalidArgumentError("invalid vocabulary or target shape");
  const size_t rows = targets.size_bytes() / sizeof(int);
  if (logits.size_bytes() % rows != 0 ||
      logits.size_bytes() / rows % sizeof(float) != 0)
    return absl::InvalidArgumentError("logits must contain whole FP32 rows");
  const size_t stride = logits.size_bytes() / rows / sizeof(float);
  if (stride < static_cast<size_t>(vocabulary_size) ||
      rows > std::numeric_limits<int>::max() ||
      stride > std::numeric_limits<int>::max())
    return absl::InvalidArgumentError("invalid top-1 matrix dimensions");
  ASSIGN_OR_RETURN(auto result,
                   cuda::Buffer::Allocate(executor, targets.size_bytes()));
  ExtractTop1IdsKernel<<<rows, 1, 0, executor.stream()>>>(
      static_cast<const float*>(logits.data()),
      static_cast<const int*>(targets.data()), static_cast<int>(rows),
      vocabulary_size, static_cast<int>(stride),
      static_cast<int*>(result.data()));
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(), "ExtractTop1IdsKernel"));
  return result;
}

}  // namespace pluto::llm
