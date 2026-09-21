#include "src/llm/vocabulary_readout.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <algorithm>
#include <cstddef>
#include <limits>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

constexpr int kRowsPerChunk = 16;
constexpr int kProjectionTokens = 8;
constexpr int kProjectionColumns = 32;
constexpr int kVocabularyTile = 256;

// Each program owns eight dot products. BF16 activations are widened before
// FP32 multiply-add; the FP32 embedding is never rounded to a narrower type.
// Fixed column traversal and a tile reduction avoid atomics. Masked loads
// exclude vocabulary padding and handle any positive embedding width.
template <class Element>
__tile_global__ void EmbeddingLogitsKernel(const Element* activations,
                                           const float* embedding, int rows,
                                           int width, int vocab_size,
                                           float* logits) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto activation_view =
      ct::partition_view{ct::tensor_span{activations, ct::extents{rows, width}},
                         ct::shape{1_ic, 32_ic}};
  auto embedding_view = ct::partition_view{
      ct::tensor_span{embedding, ct::extents{vocab_size, width}},
      ct::shape{8_ic, 32_ic}};
  auto logits_view =
      ct::partition_view{ct::tensor_span{logits, ct::extents{rows, vocab_size}},
                         ct::shape{1_ic, 8_ic}};
  const int token_tile = ct::bid().x;
  const int row = ct::bid().y;
  const int width_tiles = 1 + (width - 1) / kProjectionColumns;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<8, 32>>>();
  for (int tile = 0; tile < width_tiles; ++tile) {
    auto input =
        ct::element_cast<float>(activation_view.load_masked(row, tile));
    accumulator = ct::fma(input, embedding_view.load_masked(token_tile, tile),
                          accumulator);
  }
  logits_view.store_masked(ct::transpose(ct::sum(accumulator, 1_ic)), row,
                           token_tile);
}

// Scalar fields in the packed TopThreeTokens output have a six-element row
// stride: three IDs followed by three probabilities. Separate typed views
// preserve that public layout without an intermediate output-format kernel.
template <class T>
__tile__ auto TopThreeFieldView(T* data, int rows) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto mapping =
      ct::layout_strided_mapping{ct::extents{rows, 3}, ct::extents{6, 1}};
  return ct::partition_view{ct::tensor_span{data, mapping},
                            ct::shape{1_ic, 1_ic}};
}

// Three fixed-order argmax reductions select distinct IDs, then a fourth pass
// computes the full softmax denominator. This deliberately favors obvious
// tie/normalization semantics over a more intricate top-k merge algorithm.
__tile_global__ void TopThreeKernel(const float* logits, int rows,
                                    int vocab_size, int row_stride,
                                    int* output_tokens,
                                    float* output_probabilities) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  // The logical extent, not the physical row stride, masks the final load.
  // Even nonfinite padding is never read into a vocabulary reduction.
  auto mapping = ct::layout_strided_mapping{ct::extents{rows, vocab_size},
                                            ct::extents{row_stride, 1}};
  auto logits_view = ct::partition_view{ct::tensor_span{logits, mapping},
                                        ct::shape{1_ic, 256_ic}};
  auto token_view = TopThreeFieldView(output_tokens, rows);
  auto probability_view = TopThreeFieldView(output_probabilities, rows);
  const int row = ct::bid().x;
  const int tiles = 1 + (vocab_size - 1) / kVocabularyTile;
  constexpr float kLargestFiniteFloat = 3.402823466e+38f;
  auto no_id = ct::full<ct::tile<int, ct::shape<1, 256>>>(0x7fffffff);
  auto chosen0 = ct::full<ct::tile<int, ct::shape<1, 1>>>(-1);
  auto chosen1 = chosen0;
  auto maximum0 = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  auto maximum1 = maximum0;
  auto maximum2 = maximum0;
  auto invalid_row = ct::zeros<ct::tile<int, ct::shape<1, 1>>>();
  for (int rank = 0; rank < 3; ++rank) {
    auto best =
        ct::full<ct::tile<float, ct::shape<1, 256>>>(-kLargestFiniteFloat);
    auto best_id = no_id;
    auto invalid = ct::zeros<ct::tile<int, ct::shape<1, 256>>>();
    for (int tile = 0; tile < tiles; ++tile) {
      auto ids =
          ct::iota<ct::tile<int, ct::shape<1, 256>>>() + tile * kVocabularyTile;
      auto values = logits_view.load_masked(row, tile);
      auto logical = ids < vocab_size;
      auto nonfinite = (values != values) | (values > kLargestFiniteFloat) |
                       (values < -kLargestFiniteFloat);
      invalid = ct::max(invalid, ct::element_cast<int>(logical & nonfinite));
      auto eligible = logical & (ids != chosen0) & (ids != chosen1);
      auto wins =
          eligible & ((values > best) | ((values == best) & (ids < best_id)));
      best = ct::select(wins, values, best);
      best_id = ct::select(wins, ids, best_id);
    }
    invalid_row = ct::max(invalid_row, ct::reduce_max(invalid, 1_ic));
    auto maximum = ct::reduce_max(best, 1_ic);
    auto winner =
        ct::reduce_min(ct::select(best == maximum, best_id, no_id), 1_ic);
    token_view.store(winner, row, rank);
    if (rank == 0) {
      chosen0 = winner;
      maximum0 = maximum;
    } else if (rank == 1) {
      chosen1 = winner;
      maximum1 = maximum;
    } else {
      maximum2 = maximum;
    }
  }
  if (static_cast<int>(invalid_row) != 0) {
    auto invalid_id = ct::full<ct::tile<int, ct::shape<1, 1>>>(-1);
    auto nan = ct::element_bitcast<float>(
        ct::full<ct::tile<int, ct::shape<1, 1>>>(0x7fc00000));
    for (int rank = 0; rank < 3; ++rank) {
      token_view.store(invalid_id, row, rank);
      probability_view.store(nan, row, rank);
    }
    return;
  }
  auto sums = ct::zeros<ct::tile<float, ct::shape<1, 256>>>();
  for (int tile = 0; tile < tiles; ++tile) {
    auto ids =
        ct::iota<ct::tile<int, ct::shape<1, 256>>>() + tile * kVocabularyTile;
    sums = sums +
           ct::select(ids < vocab_size,
                      ct::exp(logits_view.load_masked(row, tile) - maximum0),
                      ct::zeros<decltype(sums)>());
  }
  auto denominator = ct::sum(sums, 1_ic);
  probability_view.store(1.0f / denominator, row, 0);
  probability_view.store(ct::exp(maximum1 - maximum0) / denominator, row, 1);
  probability_view.store(ct::exp(maximum2 - maximum0) / denominator, row, 2);
}

absl::StatusOr<size_t> CheckedBytes(size_t rows, size_t columns,
                                    size_t element_bytes) {
  if (columns == 0 || element_bytes == 0 ||
      columns > std::numeric_limits<size_t>::max() / element_bytes ||
      rows > std::numeric_limits<size_t>::max() / element_bytes / columns)
    return absl::InvalidArgumentError("readout tensor byte size overflows");
  return rows * columns * element_bytes;
}

absl::Status ValidateMatrixPrefix(cuda::Executor& executor,
                                  const cuda::Buffer& buffer, int rows,
                                  int columns, size_t element_bytes,
                                  const char* description) {
  ASSIGN_OR_RETURN(const size_t row_bytes,
                   CheckedBytes(1, columns, element_bytes));
  ASSIGN_OR_RETURN(const size_t required,
                   CheckedBytes(rows, columns, element_bytes));
  if (&buffer.executor() != &executor)
    return absl::InvalidArgumentError(
        absl::StrCat(description, " belongs to another executor"));
  if (buffer.size_bytes() < required || buffer.size_bytes() % row_bytes != 0)
    return absl::InvalidArgumentError(
        absl::StrCat(description, " must contain the requested whole rows"));
  return absl::OkStatus();
}

template <class Element>
absl::Status ProjectChunk(cuda::Executor& executor,
                          const cuda::Buffer& activations,
                          const cuda::Buffer& embedding, int start, int rows,
                          int vocab_size, int width,
                          const cuda::Buffer& logits) {
  // Subtract before rounding up so even INT_MAX vocabulary counts cannot
  // overflow signed dimension arithmetic.
  const int blocks = (vocab_size - 1) / kProjectionTokens + 1;
  EmbeddingLogitsKernel<<<dim3(blocks, rows), 1, 0, executor.stream()>>>(
      static_cast<const Element*>(activations.data()) +
          static_cast<size_t>(start) * width,
      static_cast<const float*>(embedding.data()), rows, width, vocab_size,
      static_cast<float*>(logits.data()));
  return cuda::CudaStatus(cudaGetLastError(), "EmbeddingLogitsKernel launch");
}

}  // namespace

absl::StatusOr<cuda::Buffer> ReadTopThreeTokens(cuda::Executor& executor,
                                                const cuda::Buffer& fp32_logits,
                                                int rows, int logical_vocab,
                                                int row_stride) {
  if (rows <= 0 || logical_vocab < 3 || row_stride < logical_vocab)
    return absl::InvalidArgumentError(
        "top-three requires rows > 0, vocab >= 3, and stride >= vocab");
  RETURN_IF_ERROR(ValidateMatrixPrefix(executor, fp32_logits, rows, row_stride,
                                       sizeof(float), "readout logits"));
  ASSIGN_OR_RETURN(const size_t output_bytes,
                   CheckedBytes(rows, 1, sizeof(TopThreeTokens)));
  ASSIGN_OR_RETURN(auto output, cuda::Buffer::Allocate(executor, output_bytes));
  TopThreeKernel<<<rows, 1, 0, executor.stream()>>>(
      static_cast<const float*>(fp32_logits.data()), rows, logical_vocab,
      row_stride, static_cast<int*>(output.data()),
      static_cast<float*>(output.data()) + 3);
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "TopThreeKernel launch"));
  return output;
}

absl::StatusOr<cuda::Buffer> ReadEmbeddingNeighbors(
    cuda::Executor& executor, const cuda::Buffer& activations, DataType storage,
    const cuda::Buffer& fp32_embedding, int rows, int logical_vocab,
    int width) {
  if (rows <= 0 || logical_vocab < 3 || width <= 0)
    return absl::InvalidArgumentError(
        "embedding readout requires rows > 0, vocab >= 3, and width > 0");
  if (storage != DataType::FP32 && storage != DataType::BF16)
    return absl::UnimplementedError(
        "embedding readout supports physical FP32 or BF16 activations");
  const size_t element_bytes =
      storage == DataType::BF16 ? sizeof(__nv_bfloat16) : sizeof(float);
  RETURN_IF_ERROR(ValidateMatrixPrefix(executor, activations, rows, width,
                                       element_bytes, "readout activations"));
  RETURN_IF_ERROR(ValidateMatrixPrefix(executor, fp32_embedding, logical_vocab,
                                       width, sizeof(float),
                                       "readout embedding"));
  ASSIGN_OR_RETURN(const size_t output_bytes,
                   CheckedBytes(rows, 1, sizeof(TopThreeTokens)));
  ASSIGN_OR_RETURN(auto output, cuda::Buffer::Allocate(executor, output_bytes));
  ASSIGN_OR_RETURN(
      const size_t logits_bytes,
      CheckedBytes(std::min(rows, kRowsPerChunk), logical_vocab, sizeof(float)));
  ASSIGN_OR_RETURN(auto logits, cuda::Buffer::Allocate(executor, logits_bytes));
  for (int start = 0; start < rows;) {
    const int chunk_rows = std::min(kRowsPerChunk, rows - start);
    if (storage == DataType::BF16)
      RETURN_IF_ERROR(ProjectChunk<__nv_bfloat16>(
          executor, activations, fp32_embedding, start, chunk_rows,
          logical_vocab, width, logits));
    else
      RETURN_IF_ERROR(ProjectChunk<float>(executor, activations, fp32_embedding,
                                          start, chunk_rows, logical_vocab,
                                          width, logits));
    auto* chunk_output = static_cast<TopThreeTokens*>(output.data()) + start;
    TopThreeKernel<<<chunk_rows, 1, 0, executor.stream()>>>(
        static_cast<const float*>(logits.data()), chunk_rows, logical_vocab,
        logical_vocab, reinterpret_cast<int*>(chunk_output),
        reinterpret_cast<float*>(chunk_output) + 3);
    RETURN_IF_ERROR(
        cuda::CudaStatus(cudaGetLastError(), "TopThreeKernel launch"));
    start += chunk_rows;
  }
  // The temporary's stream-ordered free follows every projection/reduction.
  return output;
}

}  // namespace pluto::llm
