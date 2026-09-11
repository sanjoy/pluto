#include "ai-slop/weight_analysis/token_argmax.h"

#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/util/status_macros.h"

namespace pluto::weight_analysis {
namespace {

constexpr int kColumnsPerTile = 256;
constexpr float kLowestFloat = -3.402823466e+38f;
constexpr int kNoCandidate = 2147483647;

// One tile block scans a row in 256-column pieces. The two reductions first
// find the maximum and then the lowest token ID attaining it; explicit ID
// reduction keeps ties deterministic within tiles and across tile boundaries.
// Reads are coalesced; only one int32 is emitted per row, not a
// vocabulary-sized host copy. load_masked also supports physical widths not
// divisible by 256.
__tile_global__ void ArgmaxTokensKernel(const float* __restrict__ logits,
                                        int rows, int logical_vocab_size,
                                        int padded_vocab_size,
                                        int32_t* __restrict__ output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{logits, ct::extents{rows, padded_vocab_size}},
      ct::shape{1_ic, 256_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{rows}}, ct::shape{1_ic}};
  const int row = ct::bid().x;
  const int tiles = (logical_vocab_size - 1) / kColumnsPerTile + 1;
  auto maximum = ct::full<ct::tile<float, ct::shape<1, 1>>>(kLowestFloat);
  auto best = ct::full<ct::tile<int, ct::shape<1, 1>>>(kNoCandidate);
  auto invalid = ct::zeros<ct::tile<int, ct::shape<1, 1>>>();
  auto lowest_values =
      ct::full<ct::tile<float, ct::shape<1, 256>>>(kLowestFloat);
  auto absent_ids = ct::full<ct::tile<int, ct::shape<1, 256>>>(kNoCandidate);
  auto invalid_id = ct::full<ct::tile<int, ct::shape<1, 1>>>(kInvalidTokenId);

  for (int tile = 0; tile < tiles; ++tile) {
    auto values = input_view.load_masked(row, tile);
    auto ids =
        ct::iota<ct::tile<int, ct::shape<1, 256>>>() + tile * kColumnsPerTile;
    auto logical = ids < logical_vocab_size;
    auto nonfinite = ct::isnan(values) || ct::isinf(values);
    invalid = ct::max(
        invalid,
        ct::reduce_max(ct::element_cast<int>(logical && nonfinite), 1_ic));
    auto eligible = logical && !nonfinite;
    auto tile_maximum =
        ct::reduce_max(ct::select(eligible, values, lowest_values), 1_ic);
    auto tile_best = ct::reduce_min(
        ct::select(eligible && (values == tile_maximum), ids, absent_ids),
        1_ic);
    best = ct::select(
        tile_maximum > maximum, tile_best,
        ct::select(tile_maximum == maximum, ct::min(best, tile_best), best));
    maximum = ct::max(maximum, tile_maximum);
  }
  output_view.store(
      ct::reshape(ct::select(invalid != 0, invalid_id, best), ct::shape{1_ic}),
      row);
}

}  // namespace

absl::StatusOr<cuda::Buffer> ArgmaxTokens(cuda::Executor& executor,
                                          const cuda::Buffer& logits, int rows,
                                          int logical_vocab_size,
                                          int padded_vocab_size) {
  if (rows <= 0 || logical_vocab_size <= 0 ||
      padded_vocab_size < logical_vocab_size) {
    return absl::InvalidArgumentError(
        "argmax requires rows > 0 and 0 < logical vocabulary <= padded width");
  }
  if (&logits.executor() != &executor) {
    return absl::InvalidArgumentError(
        "argmax logits belong to a different CUDA Executor");
  }
  const size_t width = static_cast<size_t>(padded_vocab_size);
  if (width > std::numeric_limits<size_t>::max() / sizeof(float) ||
      static_cast<size_t>(rows) >
          std::numeric_limits<size_t>::max() / sizeof(float) / width) {
    return absl::InvalidArgumentError(
        "argmax input byte size overflows size_t");
  }
  const size_t expected = static_cast<size_t>(rows) * width * sizeof(float);
  if (logits.size_bytes() != expected) {
    return absl::InvalidArgumentError(
        absl::StrCat("argmax FP32 logits have ", logits.size_bytes(),
                     " bytes; expected ", expected));
  }
  ASSIGN_OR_RETURN(auto output,
                   cuda::Buffer::Allocate(
                       executor, static_cast<size_t>(rows) * sizeof(int32_t)));
  ArgmaxTokensKernel<<<rows, 1, 0, executor.stream()>>>(
      static_cast<const float*>(logits.data()), rows, logical_vocab_size,
      padded_vocab_size, static_cast<int32_t*>(output.data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "ArgmaxTokensKernel launch"));
  return std::move(output);
}

}  // namespace pluto::weight_analysis
