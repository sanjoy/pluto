#include "src/llm/layers/embedding.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layers/util.h"
#include "src/llm/token_order.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

template <class Activation>
using MmaType = std::conditional_t<std::is_same_v<Activation, float>, __half,
                                   __nv_bfloat16>;

// The imported table stays in its checkpoint storage format. Reading the
// device ID here avoids a host transfer; masking invalid IDs guarantees that
// malformed input cannot address memory outside the shared table.
template <class Weight>
__tile_global__ void ImportedEmbeddingKernel(const Weight* weights,
                                             const int32_t* token, int vocab,
                                             int width, __nv_bfloat16* output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto index = ct::partition_view{ct::tensor_span{token, ct::extents{1}},
                                  ct::shape{1_ic}};
  auto result = ct::partition_view{ct::tensor_span{output, ct::extents{width}},
                                   ct::shape{256_ic}};
  const int id = static_cast<int>(index.load(0));
  if (id < 0 || id >= vocab) {
    result.store_masked(ct::zeros<ct::tile<__nv_bfloat16, ct::shape<256>>>(),
                        ct::bid().x);
    return;
  }
  auto row = ct::partition_view{
      ct::tensor_span{weights + static_cast<int64_t>(id) * width,
                      ct::extents{width}},
      ct::shape{256_ic}};
  result.store_masked(
      ct::element_cast<__nv_bfloat16>(row.load_masked(ct::bid().x)),
      ct::bid().x);
}

template <class Activation>
__tile_global__ void EmbeddingForwardKernel(const int* __restrict__ tokens,
                                            const float* __restrict__ table,
                                            int rows, int padded_vocab_size,
                                            int embedding_dim,
                                            Activation* __restrict__ output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto token_view = ct::partition_view{
      ct::tensor_span{tokens, ct::extents{rows}}, ct::shape{1_ic}};
  auto table_view = ct::partition_view{
      ct::tensor_span{table, ct::extents{padded_vocab_size, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  const int width_tiles = (embedding_dim - 1) / internal::kDenseTile + 1;
  const int block = ct::bid().x;
  const int row = block / width_tiles;
  const int width_tile = block % width_tiles;
  const int token = static_cast<int>(token_view.load(row));
  // A narrow model stores exactly embedding_dim channels per row. Mask the
  // tail rather than reading the next row or adding trainable padding lanes.
  output_view.store_masked(
      ct::element_cast<Activation>(table_view.load_masked(token, width_tile)),
      row, width_tile);
}

constexpr int kKeyTile = 256;

// Bitonic sorting network entirely within one tile. Exchanging the middle
// dimension swaps partners whose lane IDs differ in Distance's bit. The
// compiler maps these tile operations to registers/shuffles; no global scratch
// or thread-level synchronization is needed for the local sort.
template <int Sequence, int Distance>
__tile__ auto SortKeyTile(
    ::cuda::tiles::tile<uint64_t, ::cuda::tiles::shape<kKeyTile>> keys) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  constexpr int groups = kKeyTile / (2 * Distance);
  auto grouped = ct::reshape(keys, ct::shape<groups, 2, Distance>{});
  auto first = ct::extract(grouped, ct::shape<groups, 1, Distance>{}, 0, 0, 0);
  auto second = ct::extract(grouped, ct::shape<groups, 1, Distance>{}, 0, 1, 0);
  auto partner =
      ct::reshape(ct::cat(second, first, 1_ic), ct::shape<kKeyTile>{});
  auto lanes = ct::iota<ct::tile<int, ct::shape<kKeyTile>>>();
  auto take_min = ((lanes & Sequence) == 0) == ((lanes & Distance) == 0);
  auto ordered =
      ct::select(take_min, ct::min(keys, partner), ct::max(keys, partner));
  if constexpr (Distance > 1)
    return SortKeyTile<Sequence, Distance / 2>(ordered);
  else if constexpr (Sequence < kKeyTile)
    return SortKeyTile<Sequence * 2, Sequence>(ordered);
  else
    return ordered;
}

// The low bits make every key unique: sorting groups equal tokens and puts
// their contributions in input-row order, independently of GPU scheduling.
// Pad only the temporary tile with the largest key, never the stored tensor.
__tile_global__ void EmbeddingRowKeysKernel(const int* __restrict__ tokens,
                                            int rows, uint64_t* keys) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto token_view = ct::partition_view{
      ct::tensor_span{tokens, ct::extents{rows}}, ct::shape<kKeyTile>{}};
  auto key_view = ct::partition_view{ct::tensor_span{keys, ct::extents{rows}},
                                     ct::shape<kKeyTile>{}};
  const int block = ct::bid().x;
  auto row = ct::iota<ct::tile<int, ct::shape<kKeyTile>>>() + block * kKeyTile;
  auto token = ct::element_cast<uint64_t>(token_view.load_masked(block));
  auto key = (token << 32) | ct::element_cast<uint64_t>(row);
  auto sentinel = ct::full<ct::tile<uint64_t, ct::shape<kKeyTile>>>(UINT64_MAX);
  key_view.store_masked(
      SortKeyTile<2, 1>(ct::select(row < rows, key, sentinel)), block);
}

// Merge pairs of sorted runs. Each unique key binary-searches its insertion
// point in the other run, so every output slot has exactly one writer. Both
// sides can scatter in parallel without atomics. Global passes are ordered by
// the executor stream, and need only two O(rows) ping-pong buffers.
__tile_global__ void MergeEmbeddingKeysKernel(const uint64_t* input,
                                              int64_t rows, int64_t run,
                                              uint64_t* output) {
  namespace ct = ::cuda::tiles;
  auto index = ct::iota<ct::tile<int64_t, ct::shape<kKeyTile>>>() +
               static_cast<int64_t>(ct::bid().x) * kKeyTile;
  auto valid = index < rows;
  auto key = ct::load_masked(input + index, valid);
  auto pair = (index / (2 * run)) * (2 * run);
  auto other =
      pair + ct::select((index / run) % 2 == 0, ct::full<decltype(index)>(run),
                        ct::zeros<decltype(index)>());
  auto low = ct::zeros<decltype(index)>();
  auto high =
      ct::max(ct::min(other + run, ct::full<decltype(index)>(rows)) - other,
              ct::zeros<decltype(index)>());
  for (int64_t span = run; span > 0; span /= 2) {
    auto searching = valid & (low < high);
    auto middle = (low + high) / 2;
    auto candidate = ct::load_masked(input + other + middle, searching);
    auto go_right = candidate < key;
    low = ct::select(searching & go_right, middle + 1, low);
    high = ct::select(searching & !go_right, middle, high);
  }
  ct::store_masked(output + pair + index % run + low, key, valid);
}

__tile_global__ void EmbeddingBackwardKernel(
    const uint64_t* __restrict__ sorted_keys,
    const float* __restrict__ output_gradient, int rows, int padded_vocab_size,
    int embedding_dim, float* __restrict__ table_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto key_view = ct::partition_view{
      ct::tensor_span{sorted_keys, ct::extents{rows}}, ct::shape{1_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  auto table_view = ct::partition_view{
      ct::tensor_span{table_gradient,
                      ct::extents{padded_vocab_size, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  const int width_tiles = (embedding_dim - 1) / internal::kDenseTile + 1;
  const int block = ct::bid().x;
  const int start = block / width_tiles;
  const int width_tile = block % width_tiles;
  const uint64_t key = static_cast<uint64_t>(key_view.load(start));
  const uint64_t token = key >> 32;
  if (token >= static_cast<uint64_t>(padded_vocab_size))
    return;
  if (start > 0 &&
      (static_cast<uint64_t>(key_view.load(start - 1)) >> 32) == token)
    return;

  // Only the first row of each segment owns its table tile. Start from the
  // existing gradient, which may already contain the tied LM head's gradient.
  // A floating-point atomic scatter is not equivalent: its addition order
  // changes with scheduling and can perturb every subsequent optimizer step.
  auto accumulator =
      table_view.load_masked(static_cast<int>(token), width_tile);
  for (int index = start; index < rows; ++index) {
    const uint64_t next = static_cast<uint64_t>(key_view.load(index));
    if ((next >> 32) != token)
      break;
    const int row = static_cast<int>(next & 0xffffffffULL);
    accumulator = accumulator + gradient_view.load_masked(row, width_tile);
  }
  table_view.store_masked(accumulator, static_cast<int>(token), width_tile);
}

// These are compute tiles, not model dimensions. Reusing each operand across a
// 64x64 output tile avoids the many repeated loads of the old 16x16 products.
// K is traversed in a fixed order and each result tile has a single writer;
// no atomics or scheduling-dependent split-K reduction are needed. All views
// are masked, including embedding widths smaller than a compute tile.
constexpr int kLmHeadTile = 64;

template <class Activation>
__tile_global__ void LanguageModelingHeadForwardKernel(
    const Activation* __restrict__ input, const float* __restrict__ table,
    int rows, int padded_vocab_size, int stored_vocab_size, int embedding_dim,
    float* __restrict__ output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{rows, embedding_dim}},
      ct::shape{64_ic, 64_ic}};
  auto table_view = ct::partition_view{
      ct::tensor_span{table, ct::extents{stored_vocab_size, embedding_dim}},
      ct::shape{64_ic, 64_ic}};
  // Output padding is independent of table storage. Masked table loads supply
  // zero for nonexistent rows without creating trainable padding parameters.
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{rows, padded_vocab_size}},
      ct::shape{64_ic, 64_ic}};
  const int vocabulary_tiles = (padded_vocab_size - 1) / kLmHeadTile + 1;
  const int width_tiles = (embedding_dim - 1) / kLmHeadTile + 1;
  const int block = ct::bid().x;
  const int row_tile = block / vocabulary_tiles;
  const int vocabulary_tile = block % vocabulary_tiles;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<64, 64>>>();
  for (int dimension_tile = 0; dimension_tile < width_tiles; ++dimension_tile) {
    auto hidden = ct::element_cast<MmaType<Activation>>(
        input_view.load_masked(row_tile, dimension_tile));
    auto embedding_transposed =
        ct::transpose(ct::element_cast<MmaType<Activation>>(
            table_view.load_masked(vocabulary_tile, dimension_tile)));
    accumulator = ct::mma(hidden, embedding_transposed, accumulator);
  }
  output_view.store_masked(accumulator, row_tile, vocabulary_tile);
}

__tile_global__ void MaskPaddedLogitsKernel(float* __restrict__ logits,
                                            int rows, int vocab_size,
                                            int padded_vocab_size) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto logits_view = ct::partition_view{
      ct::tensor_span{logits, ct::extents{rows, padded_vocab_size}},
      ct::shape{1_ic, 16_ic}};
  const int row = ct::bid().x;
  const int last_tile = padded_vocab_size / internal::kDenseTile - 1;
  auto token_ids = ct::iota<ct::tile<int, ct::shape<1, 16>>>() +
                   last_tile * internal::kDenseTile;
  auto values = logits_view.load(row, last_tile);
  auto negative_infinity =
      ct::full<ct::tile<float, ct::shape<1, 16>>>(-3.402823466e+38f);
  logits_view.store(
      ct::select(token_ids < vocab_size, values, negative_infinity), row,
      last_tile);
}

template <class Activation, bool CanonicalOrder = false>
__tile_global__ void LanguageModelingHeadInputGradientKernel(
    const float* __restrict__ output_gradient, const float* __restrict__ table,
    int rows, int padded_vocab_size, int stored_vocab_size, int embedding_dim,
    float* __restrict__ input_gradient, const int32_t* __restrict__ token_order,
    int vocab_size) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, padded_vocab_size}},
      ct::shape{64_ic, 64_ic}};
  auto table_view = ct::partition_view{
      ct::tensor_span{table, ct::extents{stored_vocab_size, embedding_dim}},
      ct::shape{64_ic, 64_ic}};
  auto input_gradient_view = ct::partition_view{
      ct::tensor_span{input_gradient, ct::extents{rows, embedding_dim}},
      ct::shape{64_ic, 64_ic}};
  const int width_tiles = (embedding_dim - 1) / kLmHeadTile + 1;
  const int vocabulary_tiles = (padded_vocab_size - 1) / kLmHeadTile + 1;
  const int block = ct::bid().x;
  const int row_tile = block / width_tiles;
  const int dimension_tile = block % width_tiles;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<64, 64>>>();
  for (int vocabulary_tile = 0; vocabulary_tile < vocabulary_tiles;
       ++vocabulary_tile) {
    if constexpr (CanonicalOrder) {
      // Gather both operands into exactly the same 64-wide K layout as the
      // identity path. Reordering only tiles, or only one operand, is not
      // sufficient: floating-point MMA must see the same products in the
      // same lanes and accumulation order after a vocabulary renaming.
      auto rank = ct::iota<ct::tile<int, ct::shape<64>>>() +
                  vocabulary_tile * kLmHeadTile;
      auto token = ct::select(
          rank < vocab_size,
          ct::load_masked(token_order + rank, rank < vocab_size), rank);
      auto input_row = ct::reshape(
          ct::iota<ct::tile<int, ct::shape<64>>>() + row_tile * kLmHeadTile,
          ct::shape{64_ic, 1_ic});
      auto token_column = ct::reshape(token, ct::shape{1_ic, 64_ic});
      auto gradient = ct::element_cast<MmaType<Activation>>(ct::load_masked(
          output_gradient +
              ct::element_cast<int64_t>(input_row) * padded_vocab_size +
              token_column,
          (input_row < rows) & (token_column < padded_vocab_size)));
      auto token_row = ct::reshape(token, ct::shape{64_ic, 1_ic});
      auto dimension = ct::reshape(ct::iota<ct::tile<int, ct::shape<64>>>() +
                                       dimension_tile * kLmHeadTile,
                                   ct::shape{1_ic, 64_ic});
      auto embeddings = ct::element_cast<MmaType<Activation>>(ct::load_masked(
          table + ct::element_cast<int64_t>(token_row) * embedding_dim +
              dimension,
          (token_row < stored_vocab_size) & (dimension < embedding_dim)));
      accumulator = ct::mma(gradient, embeddings, accumulator);
    } else {
      auto gradient = ct::element_cast<MmaType<Activation>>(
          gradient_view.load_masked(row_tile, vocabulary_tile));
      auto embeddings = ct::element_cast<MmaType<Activation>>(
          table_view.load_masked(vocabulary_tile, dimension_tile));
      accumulator = ct::mma(gradient, embeddings, accumulator);
    }
  }
  input_gradient_view.store_masked(accumulator, row_tile, dimension_tile);
}

template <class Activation>
__tile_global__ void LanguageModelingHeadWeightGradientKernel(
    const Activation* __restrict__ input,
    const float* __restrict__ output_gradient, int rows, int padded_vocab_size,
    int stored_vocab_size, int embedding_dim,
    float* __restrict__ table_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{rows, embedding_dim}},
      ct::shape{64_ic, 64_ic}};
  auto output_gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, padded_vocab_size}},
      ct::shape{64_ic, 64_ic}};
  auto table_gradient_view = ct::partition_view{
      ct::tensor_span{table_gradient,
                      ct::extents{stored_vocab_size, embedding_dim}},
      ct::shape{64_ic, 64_ic}};
  const int width_tiles = (embedding_dim - 1) / kLmHeadTile + 1;
  const int row_tiles = (rows - 1) / kLmHeadTile + 1;
  const int block = ct::bid().x;
  const int vocabulary_tile = block / width_tiles;
  const int dimension_tile = block % width_tiles;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<64, 64>>>();
  for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
    auto gradient_transposed =
        ct::transpose(ct::element_cast<MmaType<Activation>>(
            output_gradient_view.load_masked(row_tile, vocabulary_tile)));
    auto hidden = ct::element_cast<MmaType<Activation>>(
        input_view.load_masked(row_tile, dimension_tile));
    accumulator = ct::mma(gradient_transposed, hidden, accumulator);
  }
  table_gradient_view.store_masked(
      table_gradient_view.load_masked(vocabulary_tile, dimension_tile) +
          accumulator,
      vocabulary_tile, dimension_tile);
}

template <class Activation>
__tile_global__ void PositionEmbeddingForwardKernel(
    const Activation* __restrict__ input, const float* __restrict__ positions,
    int rows, int context_length, int embedding_dim,
    Activation* __restrict__ output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  auto position_view = ct::partition_view{
      ct::tensor_span{positions, ct::extents{context_length, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  const int width_tiles = (embedding_dim - 1) / internal::kDenseTile + 1;
  const int block = ct::bid().x;
  const int row = block / width_tiles;
  const int width_tile = block % width_tiles;
  auto sum = ct::element_cast<float>(input_view.load_masked(row, width_tile)) +
             position_view.load_masked(row % context_length, width_tile);
  output_view.store_masked(ct::element_cast<Activation>(sum), row, width_tile);
}

__tile_global__ void PositionEmbeddingBackwardKernel(
    const float* __restrict__ output_gradient, int rows, int context_length,
    int embedding_dim, float* __restrict__ position_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  auto position_view = ct::partition_view{
      ct::tensor_span{position_gradient,
                      ct::extents{context_length, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  const int width_tiles = (embedding_dim - 1) / internal::kDenseTile + 1;
  const int block = ct::bid().x;
  const int position = block / width_tiles;
  const int width_tile = block % width_tiles;
  // One writer per position/width tile, with a fixed sequence-row order.
  // Unvisited positions stay untouched, including for a partial context.
  auto accumulator = position_view.load_masked(position, width_tile);
  // Use a wider loop counter: the final stride may exceed INT_MAX even
  // though every visited row fits the validated int-sized input.
  for (int64_t row = position; row < rows; row += context_length) {
    accumulator = accumulator +
                  gradient_view.load_masked(static_cast<int>(row), width_tile);
  }
  position_view.store_masked(accumulator, position, width_tile);
}

absl::Status CopyNormalInitialization(cuda::Executor& executor, Buffer& weight,
                                      float standard_deviation, uint64_t seed,
                                      const char* operation) {
  if (!(standard_deviation > 0.0f)) {
    return absl::InvalidArgumentError(
        "initialization standard deviation must be positive");
  }
  std::mt19937_64 random(seed);
  std::normal_distribution<float> distribution(0.0f, standard_deviation);
  ASSIGN_OR_RETURN(auto values,
                   cuda::PageLockedHostArray<float>::Allocate(
                       executor, weight.size_bytes() / sizeof(float)));
  for (float& value : values)
    value = distribution(random);
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(weight.data(), values.data(), weight.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      operation));
  // The pinned staging buffer is freed after this executor's queued upload.
  return absl::OkStatus();
}

}  // namespace

EmbeddingLookupLayer::EmbeddingLookupLayer(
    cuda::Executor& executor, int vocab_size, int padded_vocab_size,
    int stored_vocab_size, int embedding_dim, DataType data_type, Buffer weight,
    std::optional<Buffer> gradient, int sequence_length,
    std::optional<MatrixStorage> imported_storage)
    : vocab_size_(vocab_size),
      padded_vocab_size_(padded_vocab_size),
      stored_vocab_size_(stored_vocab_size),
      embedding_dim_(embedding_dim),
      sequence_length_(sequence_length),
      output_type_(data_type),
      executor_(executor),
      weight_(std::move(weight)),
      gradient_(std::move(gradient)),
      imported_storage_(imported_storage) {}

absl::StatusOr<std::unique_ptr<EmbeddingLookupLayer>>
EmbeddingLookupLayer::Create(cuda::Executor& executor, int vocab_size,
                             int embedding_dim, DataType data_type,
                             int sequence_length, bool pad_vocabulary) {
  RETURN_IF_ERROR(internal::ValidateComputeType(data_type));
  if (sequence_length <= 0)
    return absl::InvalidArgumentError("sequence_length must be positive");
  if (vocab_size <= 0)
    return absl::InvalidArgumentError("vocab_size must be positive");
  RETURN_IF_ERROR(
      internal::ValidatePositiveExtent(embedding_dim, "embedding_dim"));
  const int64_t padded_extent = (int64_t{vocab_size} + 15) / 16 * 16;
  if (padded_extent > std::numeric_limits<int>::max())
    return absl::InvalidArgumentError("padded vocabulary exceeds int range");
  const int padded_vocab_size = static_cast<int>(padded_extent);
  const int stored_vocab_size = pad_vocabulary ? padded_vocab_size : vocab_size;
  if (int64_t{stored_vocab_size} * embedding_dim >
      std::numeric_limits<int>::max())
    return absl::InvalidArgumentError(
        "embedding exceeds the backend's 32-bit element-count limit");
  const size_t bytes =
      static_cast<size_t>(stored_vocab_size) * embedding_dim * sizeof(float);
  ASSIGN_OR_RETURN(auto weight, Buffer::Allocate(executor, bytes));
  ASSIGN_OR_RETURN(auto gradient, Buffer::Allocate(executor, bytes));
  for (Buffer* buffer : {&weight, &gradient}) {
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemsetAsync(buffer->data(), 0, buffer->size_bytes(),
                        executor.stream()),
        "cudaMemsetAsync(embedding parameter)"));
  }
  return absl::WrapUnique(new EmbeddingLookupLayer(
      executor, vocab_size, padded_vocab_size, stored_vocab_size, embedding_dim,
      data_type, std::move(weight), std::move(gradient), sequence_length));
}

absl::StatusOr<std::unique_ptr<EmbeddingLookupLayer>>
EmbeddingLookupLayer::Create(cuda::Executor& executor, Buffer weights,
                             MatrixStorage storage, int vocab_size,
                             int embedding_dim) {
  if (vocab_size <= 0 || vocab_size > kMaximumDimension || embedding_dim <= 0 ||
      embedding_dim > kMaximumDimension)
    return absl::InvalidArgumentError(
        "imported embedding dimensions must be in [1, 1048576]");
  if (storage != MatrixStorage::kBFloat16 && storage != MatrixStorage::kFloat32)
    return absl::InvalidArgumentError(
        "imported embedding requires BF16 or FP32 weights");
  ASSIGN_OR_RETURN(size_t element_bytes, MatrixElementBytes(storage));
  RETURN_IF_ERROR(internal::ValidateBuffer(
      executor, weights, size_t(vocab_size) * embedding_dim * element_bytes,
      "imported embedding weights"));
  const int padded_vocab_size = (vocab_size + 15) / 16 * 16;
  return absl::WrapUnique(new EmbeddingLookupLayer(
      executor, vocab_size, padded_vocab_size, vocab_size, embedding_dim,
      DataType::BF16, std::move(weights), std::nullopt, 1, storage));
}

absl::Status EmbeddingLookupLayer::InitializeIdentity(float scale) {
  if (imported_storage_)
    return absl::UnimplementedError(
        "cannot initialize a frozen imported embedding table");
  ASSIGN_OR_RETURN(
      auto values,
      cuda::PageLockedHostArray<float>::Allocate(
          executor_, static_cast<size_t>(stored_vocab_size_) * embedding_dim_));
  std::fill(values.begin(), values.end(), 0.0f);
  for (int index = 0; index < std::min(vocab_size_, embedding_dim_); ++index)
    values[static_cast<size_t>(index) * embedding_dim_ + index] = scale;
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(weight_.data(), values.data(), weight_.size_bytes(),
                      cudaMemcpyHostToDevice, executor_.stream()),
      "cudaMemcpyAsync(identity embedding)"));
  return absl::OkStatus();
}

absl::Status EmbeddingLookupLayer::InitializeNormal(float standard_deviation,
                                                    uint64_t seed) {
  if (imported_storage_)
    return absl::UnimplementedError(
        "cannot initialize a frozen imported embedding table");
  return CopyNormalInitialization(executor_, weight_, standard_deviation, seed,
                                  "cudaMemcpyAsync(normal embedding)");
}

absl::StatusOr<FwdResult> EmbeddingLookupLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks*) const {
  BackwardState state;
  RETURN_IF_ERROR(
      internal::ValidateExecutor(executor_, executor, "EmbeddingLookupLayer"));
  if (inputs.size() != 1) {
    return absl::InvalidArgumentError(
        "EmbeddingLookupLayer fwd expects token IDs");
  }
  if (imported_storage_) {
    RETURN_IF_ERROR(internal::ValidateBuffer(executor, inputs[0],
                                             sizeof(int32_t),
                                             "imported embedding token input"));
    ASSIGN_OR_RETURN(auto output,
                     Buffer::Allocate(executor, size_t(embedding_dim_) *
                                                    sizeof(__nv_bfloat16)));
    const int blocks = 1 + (embedding_dim_ - 1) / 256;
    if (*imported_storage_ == MatrixStorage::kBFloat16) {
      ImportedEmbeddingKernel<<<blocks, 1, 0, executor.stream()>>>(
          static_cast<const __nv_bfloat16*>(weight_.data()),
          static_cast<const int32_t*>(inputs[0].data()), vocab_size_,
          embedding_dim_, static_cast<__nv_bfloat16*>(output.data()));
    } else {
      ImportedEmbeddingKernel<<<blocks, 1, 0, executor.stream()>>>(
          static_cast<const float*>(weight_.data()),
          static_cast<const int32_t*>(inputs[0].data()), vocab_size_,
          embedding_dim_, static_cast<__nv_bfloat16*>(output.data()));
    }
    RETURN_IF_ERROR(
        cuda::CudaStatus(cudaGetLastError(), "ImportedEmbeddingKernel launch"));
    return FwdResult{{std::move(output)}, {}};
  }
  ASSIGN_OR_RETURN(int rows,
                   internal::ElementCount(executor, inputs[0], sizeof(int),
                                          "embedding token input"));
  ASSIGN_OR_RETURN(
      auto output,
      Buffer::Allocate(executor,
                       static_cast<size_t>(rows) * embedding_dim_ *
                           internal::ActivationElementBytes(output_type_)));
  const int blocks = rows * internal::MaskedTileCount(embedding_dim_);
  if (output_type_ == DataType::BF16) {
    EmbeddingForwardKernel<__nv_bfloat16><<<blocks, 1, 0, executor.stream()>>>(
        static_cast<const int*>(inputs[0].data()),
        static_cast<const float*>(weight_.data()), rows, stored_vocab_size_,
        embedding_dim_, static_cast<__nv_bfloat16*>(output.data()));
  } else {
    EmbeddingForwardKernel<float><<<blocks, 1, 0, executor.stream()>>>(
        static_cast<const int*>(inputs[0].data()),
        static_cast<const float*>(weight_.data()), rows, stored_vocab_size_,
        embedding_dim_, static_cast<float*>(output.data()));
  }
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "EmbeddingForwardKernel launch"));
  state.intermediates = {inputs[0]};
  state.children.clear();
  return FwdResult{{std::move(output)}, std::move(state)};
}

absl::StatusOr<BufferVec> EmbeddingLookupLayer::bwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    BackwardState state, LayerHooks*) {
  RETURN_IF_ERROR(
      internal::ValidateExecutor(executor_, executor, "EmbeddingLookupLayer"));
  if (imported_storage_)
    return absl::UnimplementedError(
        "frozen imported embeddings do not support backward");
  if (output_gradients.size() != 1 || state.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "EmbeddingLookupLayer bwd received an incompatible gradient or state");
  }
  ASSIGN_OR_RETURN(int rows,
                   internal::ElementCount(executor, state.intermediates[0],
                                          sizeof(int), "embedding token input"));
  RETURN_IF_ERROR(internal::ValidateBuffer(
      executor, output_gradients[0],
      static_cast<size_t>(rows) * embedding_dim_ * sizeof(float),
      "embedding output gradient"));
  // Sorting integer keys is deterministic and uses only linear scratch space.
  // All scratch is stream-ordered; no host transfer or synchronization is
  // needed.
  ASSIGN_OR_RETURN(
      auto keys,
      Buffer::Allocate(executor, static_cast<size_t>(rows) * sizeof(uint64_t)));
  ASSIGN_OR_RETURN(
      auto sorted_keys,
      Buffer::Allocate(executor, static_cast<size_t>(rows) * sizeof(uint64_t)));
  auto* keys_ptr = static_cast<uint64_t*>(keys.data());
  auto* sorted_ptr = static_cast<uint64_t*>(sorted_keys.data());
  const int key_blocks = 1 + (rows - 1) / kKeyTile;
  EmbeddingRowKeysKernel<<<key_blocks, 1, 0, executor.stream()>>>(
      static_cast<const int*>(state.intermediates[0].data()), rows, keys_ptr);
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "EmbeddingRowKeysKernel launch"));
  for (int64_t run = kKeyTile; run < rows; run *= 2) {
    MergeEmbeddingKeysKernel<<<key_blocks, 1, 0, executor.stream()>>>(
        keys_ptr, rows, run, sorted_ptr);
    RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(),
                                     "MergeEmbeddingKeysKernel launch"));
    std::swap(keys_ptr, sorted_ptr);
  }
  EmbeddingBackwardKernel<<<rows * internal::MaskedTileCount(embedding_dim_), 1,
                            0, executor.stream()>>>(
      keys_ptr, static_cast<const float*>(output_gradients[0].data()), rows,
      stored_vocab_size_, embedding_dim_,
      static_cast<float*>(gradient_->data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "EmbeddingBackwardKernel launch"));
  return BufferVec{};
}

absl::StatusOr<std::unique_ptr<LanguageModelingHeadLayer>>
LanguageModelingHeadLayer::Create(EmbeddingLookupLayer* embedding,
                                  absl::Span<const int32_t> token_order) {
  if (embedding == nullptr) {
    return absl::InvalidArgumentError(
        "LanguageModelingHeadLayer requires a non-null embedding");
  }
  if (embedding->imported_storage_)
    return absl::UnimplementedError(
        "LanguageModelingHeadLayer requires trainable FP32 master embeddings; "
        "use an imported linear layer for a frozen output projection");
  ASSIGN_OR_RETURN(auto device_order,
                   CopyTokenOrderToDevice(embedding->executor_,
                                          embedding->vocab_size_, token_order));
  return absl::WrapUnique(
      new LanguageModelingHeadLayer(embedding, std::move(device_order)));
}

absl::StatusOr<FwdResult> LanguageModelingHeadLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks*) const {
  BackwardState state;
  RETURN_IF_ERROR(internal::ValidateExecutor(embedding_->executor_, executor,
                                             "LanguageModelingHeadLayer"));
  if (inputs.size() != 1) {
    return absl::InvalidArgumentError(
        "LanguageModelingHeadLayer fwd expects one input");
  }
  ASSIGN_OR_RETURN(
      int rows, internal::ActivationRows(
                    executor, inputs[0], embedding_->embedding_dim_,
                    embedding_->output_type_, "language-modeling-head input"));
  ASSIGN_OR_RETURN(
      auto output,
      Buffer::Allocate(executor, static_cast<size_t>(rows) *
                                     embedding_->padded_vocab_size_ *
                                     sizeof(float)));
  const int blocks = ((rows - 1) / kLmHeadTile + 1) *
                     ((embedding_->padded_vocab_size_ - 1) / kLmHeadTile + 1);
  if (embedding_->output_type_ == DataType::BF16) {
    LanguageModelingHeadForwardKernel<__nv_bfloat16>
        <<<blocks, 1, 0, executor.stream()>>>(
            static_cast<const __nv_bfloat16*>(inputs[0].data()),
            static_cast<const float*>(embedding_->weight_.data()), rows,
            embedding_->padded_vocab_size_, embedding_->stored_vocab_size_,
            embedding_->embedding_dim_, static_cast<float*>(output.data()));
  } else {
    LanguageModelingHeadForwardKernel<float>
        <<<blocks, 1, 0, executor.stream()>>>(
            static_cast<const float*>(inputs[0].data()),
            static_cast<const float*>(embedding_->weight_.data()), rows,
            embedding_->padded_vocab_size_, embedding_->stored_vocab_size_,
            embedding_->embedding_dim_, static_cast<float*>(output.data()));
  }
  MaskPaddedLogitsKernel<<<rows, 1, 0, executor.stream()>>>(
      static_cast<float*>(output.data()), rows, embedding_->vocab_size_,
      embedding_->padded_vocab_size_);
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(),
                                   "language-modeling-head forward launch"));
  state.intermediates = {inputs[0]};
  state.children.clear();
  return FwdResult{{std::move(output)}, std::move(state)};
}

absl::StatusOr<BufferVec> LanguageModelingHeadLayer::bwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    BackwardState state, LayerHooks*) {
  RETURN_IF_ERROR(internal::ValidateExecutor(embedding_->executor_, executor,
                                             "LanguageModelingHeadLayer"));
  if (output_gradients.size() != 1 || state.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "LanguageModelingHeadLayer bwd received an incompatible gradient or "
        "state");
  }
  ASSIGN_OR_RETURN(
      int rows, internal::MatrixRows(executor, output_gradients[0],
                                     embedding_->padded_vocab_size_,
                                     "language-modeling-head output gradient"));
  RETURN_IF_ERROR(internal::ValidateBuffer(
      executor, state.intermediates[0],
      static_cast<size_t>(rows) * embedding_->embedding_dim_ *
          internal::ActivationElementBytes(embedding_->output_type_),
      "language-modeling-head saved input"));
  ASSIGN_OR_RETURN(auto input_gradient,
                   Buffer::Allocate(executor, static_cast<size_t>(rows) *
                                                  embedding_->embedding_dim_ *
                                                  sizeof(float)));
  const int width_tiles = (embedding_->embedding_dim_ - 1) / kLmHeadTile + 1;
  const int input_blocks = ((rows - 1) / kLmHeadTile + 1) * width_tiles;
  const int weight_blocks =
      ((embedding_->stored_vocab_size_ - 1) / kLmHeadTile + 1) * width_tiles;
  if (embedding_->output_type_ == DataType::BF16) {
    if (token_order_) {
      LanguageModelingHeadInputGradientKernel<__nv_bfloat16, true>
          <<<input_blocks, 1, 0, executor.stream()>>>(
              static_cast<const float*>(output_gradients[0].data()),
              static_cast<const float*>(embedding_->weight_.data()), rows,
              embedding_->padded_vocab_size_, embedding_->stored_vocab_size_,
              embedding_->embedding_dim_,
              static_cast<float*>(input_gradient.data()),
              static_cast<const int32_t*>(token_order_->data()),
              embedding_->vocab_size_);
    } else {
      LanguageModelingHeadInputGradientKernel<__nv_bfloat16>
          <<<input_blocks, 1, 0, executor.stream()>>>(
              static_cast<const float*>(output_gradients[0].data()),
              static_cast<const float*>(embedding_->weight_.data()), rows,
              embedding_->padded_vocab_size_, embedding_->stored_vocab_size_,
              embedding_->embedding_dim_,
              static_cast<float*>(input_gradient.data()), nullptr,
              embedding_->vocab_size_);
    }
    LanguageModelingHeadWeightGradientKernel<__nv_bfloat16>
        <<<weight_blocks, 1, 0, executor.stream()>>>(
            static_cast<const __nv_bfloat16*>(state.intermediates[0].data()),
            static_cast<const float*>(output_gradients[0].data()), rows,
            embedding_->padded_vocab_size_, embedding_->stored_vocab_size_,
            embedding_->embedding_dim_,
            static_cast<float*>(embedding_->gradient_->data()));
  } else {
    if (token_order_) {
      LanguageModelingHeadInputGradientKernel<float, true>
          <<<input_blocks, 1, 0, executor.stream()>>>(
              static_cast<const float*>(output_gradients[0].data()),
              static_cast<const float*>(embedding_->weight_.data()), rows,
              embedding_->padded_vocab_size_, embedding_->stored_vocab_size_,
              embedding_->embedding_dim_,
              static_cast<float*>(input_gradient.data()),
              static_cast<const int32_t*>(token_order_->data()),
              embedding_->vocab_size_);
    } else {
      LanguageModelingHeadInputGradientKernel<float>
          <<<input_blocks, 1, 0, executor.stream()>>>(
              static_cast<const float*>(output_gradients[0].data()),
              static_cast<const float*>(embedding_->weight_.data()), rows,
              embedding_->padded_vocab_size_, embedding_->stored_vocab_size_,
              embedding_->embedding_dim_,
              static_cast<float*>(input_gradient.data()), nullptr,
              embedding_->vocab_size_);
    }
    LanguageModelingHeadWeightGradientKernel<float>
        <<<weight_blocks, 1, 0, executor.stream()>>>(
            static_cast<const float*>(state.intermediates[0].data()),
            static_cast<const float*>(output_gradients[0].data()), rows,
            embedding_->padded_vocab_size_, embedding_->stored_vocab_size_,
            embedding_->embedding_dim_,
            static_cast<float*>(embedding_->gradient_->data()));
  }
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(),
                                   "language-modeling-head backward launch"));
  return BufferVec{std::move(input_gradient)};
}

PositionEmbeddingLayer::PositionEmbeddingLayer(cuda::Executor& executor,
                                               int context_length,
                                               int embedding_dim,
                                               DataType data_type,
                                               Buffer weight, Buffer gradient)
    : context_length_(context_length),
      embedding_dim_(embedding_dim),
      output_type_(data_type),
      executor_(executor),
      weight_(std::move(weight)),
      gradient_(std::move(gradient)) {}

absl::StatusOr<std::unique_ptr<PositionEmbeddingLayer>>
PositionEmbeddingLayer::Create(cuda::Executor& executor, int context_length,
                               int embedding_dim, DataType data_type) {
  RETURN_IF_ERROR(internal::ValidateComputeType(data_type));
  if (context_length <= 0)
    return absl::InvalidArgumentError("context_length must be positive");
  RETURN_IF_ERROR(
      internal::ValidatePositiveExtent(embedding_dim, "embedding_dim"));
  const size_t bytes =
      static_cast<size_t>(context_length) * embedding_dim * sizeof(float);
  ASSIGN_OR_RETURN(auto weight, Buffer::Allocate(executor, bytes));
  ASSIGN_OR_RETURN(auto gradient, Buffer::Allocate(executor, bytes));
  for (Buffer* buffer : {&weight, &gradient}) {
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemsetAsync(buffer->data(), 0, buffer->size_bytes(),
                        executor.stream()),
        "cudaMemsetAsync(position parameter)"));
  }
  return absl::WrapUnique(new PositionEmbeddingLayer(
      executor, context_length, embedding_dim, data_type, std::move(weight),
      std::move(gradient)));
}

absl::Status PositionEmbeddingLayer::InitializeNormal(float standard_deviation,
                                                      uint64_t seed) {
  return CopyNormalInitialization(executor_, weight_, standard_deviation, seed,
                                  "cudaMemcpyAsync(normal positions)");
}

absl::StatusOr<FwdResult> PositionEmbeddingLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks*) const {
  BackwardState state;
  RETURN_IF_ERROR(internal::ValidateExecutor(executor_, executor,
                                             "PositionEmbeddingLayer"));
  if (inputs.size() != 1) {
    return absl::InvalidArgumentError(
        "PositionEmbeddingLayer fwd expects one input");
  }
  ASSIGN_OR_RETURN(int rows, internal::ActivationRows(
                                 executor, inputs[0], embedding_dim_,
                                 output_type_, "position-embedding input"));
  ASSIGN_OR_RETURN(auto output,
                   Buffer::Allocate(executor, inputs[0].size_bytes()));
  const int blocks = rows * internal::MaskedTileCount(embedding_dim_);
  if (output_type_ == DataType::BF16) {
    PositionEmbeddingForwardKernel<__nv_bfloat16>
        <<<blocks, 1, 0, executor.stream()>>>(
            static_cast<const __nv_bfloat16*>(inputs[0].data()),
            static_cast<const float*>(weight_.data()), rows, context_length_,
            embedding_dim_, static_cast<__nv_bfloat16*>(output.data()));
  } else {
    PositionEmbeddingForwardKernel<float><<<blocks, 1, 0, executor.stream()>>>(
        static_cast<const float*>(inputs[0].data()),
        static_cast<const float*>(weight_.data()), rows, context_length_,
        embedding_dim_, static_cast<float*>(output.data()));
  }
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(),
                                   "PositionEmbeddingForwardKernel launch"));
  state.intermediates.clear();
  state.children.clear();
  return FwdResult{{std::move(output)}, std::move(state)};
}

absl::StatusOr<BufferVec> PositionEmbeddingLayer::bwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    BackwardState state, LayerHooks*) {
  RETURN_IF_ERROR(internal::ValidateExecutor(executor_, executor,
                                             "PositionEmbeddingLayer"));
  if (output_gradients.size() != 1 || !state.intermediates.empty()) {
    return absl::InvalidArgumentError(
        "PositionEmbeddingLayer bwd received an incompatible gradient or "
        "state");
  }
  ASSIGN_OR_RETURN(int rows, internal::MatrixRows(
                                 executor, output_gradients[0], embedding_dim_,
                                 "position-embedding output gradient"));
  PositionEmbeddingBackwardKernel<<<std::min(rows, context_length_) *
                                        internal::MaskedTileCount(
                                            embedding_dim_),
                                    1, 0, executor.stream()>>>(
      static_cast<const float*>(output_gradients[0].data()), rows,
      context_length_, embedding_dim_, static_cast<float*>(gradient_.data()));
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(),
                                   "PositionEmbeddingBackwardKernel launch"));
  return BufferVec{output_gradients[0]};
}

}  // namespace pluto::llm
