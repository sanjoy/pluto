#include "src/llm/experiments/mlp_automaton/top_transitions.h"

#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cstddef>
#include <limits>
#include <utility>

#include "absl/status/status.h"

namespace pluto::llm::mlp_automaton {
namespace {

constexpr int kVocabularyTile = 256;

// A TopTransition packs an int and a float. Both scalar fields have stride
// two in their respective typed view, so the kernel can write the public
// record layout directly instead of materializing another temporary array.
template <class T>
__tile__ auto TransitionFieldView(T* data, int rows) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto mapping = ct::layout_strided_mapping{ct::extents{rows}, ct::extents{2}};
  return ct::partition_view{ct::tensor_span{data, mapping}, ct::shape{1_ic}};
}

// One program owns a row. Two coalesced passes retain only tile-sized running
// candidates and the winner, never a full [rows,vocabulary] probability array.
__tile_global__ void TopTransitionsKernel(const float* __restrict__ logits,
                                          int rows, int logical_vocab,
                                          int padded_vocab, int* output_tokens,
                                          float* output_probabilities) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto mapping = ct::layout_strided_mapping{ct::extents{rows, logical_vocab},
                                            ct::extents{padded_vocab, 1}};
  auto logits_view = ct::partition_view{ct::tensor_span{logits, mapping},
                                        ct::shape{1_ic, 256_ic}};
  auto token_view = TransitionFieldView(output_tokens, rows);
  auto probability_view = TransitionFieldView(output_probabilities, rows);
  const int row = ct::bid().x;
  const int tiles = 1 + (logical_vocab - 1) / kVocabularyTile;
  constexpr float kLargestFiniteFloat = 3.402823466e+38f;
  auto best =
      ct::full<ct::tile<float, ct::shape<1, 256>>>(-kLargestFiniteFloat);
  auto no_id = ct::full<ct::tile<int, ct::shape<1, 256>>>(0x7fffffff);
  auto best_id = no_id;
  auto invalid = ct::zeros<ct::tile<int, ct::shape<1, 256>>>();
  for (int tile = 0; tile < tiles; ++tile) {
    auto ids =
        ct::iota<ct::tile<int, ct::shape<1, 256>>>() + tile * kVocabularyTile;
    auto values = logits_view.load_masked(row, tile);
    auto logical = ids < logical_vocab;
    auto nonfinite = (values != values) | (values > kLargestFiniteFloat) |
                     (values < -kLargestFiniteFloat);
    invalid = ct::max(invalid, ct::element_cast<int>(logical & nonfinite));
    auto wins =
        logical & ((values > best) | ((values == best) & (ids < best_id)));
    best = ct::select(wins, values, best);
    best_id = ct::select(wins, ids, best_id);
  }
  if (static_cast<int>(ct::reduce_max(invalid, 1_ic)) != 0) {
    token_view.store(ct::full<ct::tile<int, ct::shape<1>>>(-1), row);
    auto nan = ct::element_bitcast<float>(
        ct::full<ct::tile<int, ct::shape<1>>>(0x7fc00000));
    probability_view.store(nan, row);
    return;
  }
  auto maximum = ct::reduce_max(best, 1_ic);
  auto winner =
      ct::reduce_min(ct::select(best == maximum, best_id, no_id), 1_ic);
  auto sums = ct::zeros<ct::tile<float, ct::shape<1, 256>>>();
  for (int tile = 0; tile < tiles; ++tile) {
    auto ids =
        ct::iota<ct::tile<int, ct::shape<1, 256>>>() + tile * kVocabularyTile;
    // Masked loads return zero outside the vocabulary. Explicitly remove
    // those lanes' exponentials, otherwise they would dilute the softmax.
    sums =
        sums + ct::select(ids < logical_vocab,
                          ct::exp(logits_view.load_masked(row, tile) - maximum),
                          ct::zeros<decltype(sums)>());
  }
  // At the maximum the unnormalized probability is exactly exp(0)=1.
  auto probability = 1.0f / ct::sum(sums, 1_ic);
  token_view.store(ct::reshape(winner, ct::shape{1_ic}), row);
  probability_view.store(ct::reshape(probability, ct::shape{1_ic}), row);
}

}  // namespace

absl::StatusOr<cuda::Buffer> ReadTopTransitions(cuda::Executor& executor,
                                                const cuda::Buffer& fp32_logits,
                                                int rows, int logical_vocab,
                                                int padded_vocab) {
  if (rows <= 0 || logical_vocab <= 0 || padded_vocab < logical_vocab) {
    return absl::InvalidArgumentError(
        "top transitions require positive rows and 0 < logical_vocab <= "
        "padded_vocab");
  }
  if (&fp32_logits.executor() != &executor) {
    return absl::InvalidArgumentError(
        "top transitions input belongs to a different executor");
  }
  const size_t row_bytes = static_cast<size_t>(padded_vocab) * sizeof(float);
  if (static_cast<size_t>(padded_vocab) >
          std::numeric_limits<size_t>::max() / sizeof(float) ||
      static_cast<size_t>(rows) >
          std::numeric_limits<size_t>::max() / row_bytes ||
      static_cast<size_t>(rows) >
          std::numeric_limits<size_t>::max() / sizeof(TopTransition)) {
    return absl::InvalidArgumentError("top transitions byte size overflows");
  }
  if (fp32_logits.size_bytes() != static_cast<size_t>(rows) * row_bytes) {
    return absl::InvalidArgumentError(
        "top transitions require an exact rows*padded_vocab FP32 buffer");
  }
  auto output = cuda::Buffer::Allocate(
      executor, static_cast<size_t>(rows) * sizeof(TopTransition));
  if (!output.ok())
    return output.status();
  TopTransitionsKernel<<<rows, 1, 0, executor.stream()>>>(
      static_cast<const float*>(fp32_logits.data()), rows, logical_vocab,
      padded_vocab, static_cast<int*>(output->data()),
      static_cast<float*>(output->data()) + 1);
  auto status =
      cuda::CudaStatus(cudaGetLastError(), "TopTransitionsKernel launch");
  if (!status.ok())
    return status;
  return std::move(*output);
}

}  // namespace pluto::llm::mlp_automaton
