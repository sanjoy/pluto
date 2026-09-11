#include "src/llm/layers/embedding.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
// We use CUB directly, not Thrust algorithms. Its iterator traits otherwise
// pull in Thrust's CUDA execution policy, which contains unconditional C++
// exception handlers. Select the CPP policy for those unused Thrust facilities;
// cub::DeviceRadixSort still runs CUDA kernels on the supplied executor stream.
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#include <cub/device/device_radix_sort.cuh>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layers/internal.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

template <class Activation>
using MmaType = std::conditional_t<std::is_same_v<Activation, float>, __half,
                                   __nv_bfloat16>;

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
  const int width_tiles = embedding_dim / internal::kDenseTile;
  const int block = ct::bid().x;
  const int row = block / width_tiles;
  const int width_tile = block % width_tiles;
  const int token = static_cast<int>(token_view.load(row));
  output_view.store(
      ct::element_cast<Activation>(table_view.load(token, width_tile)), row,
      width_tile);
}

// The low bits make every key unique: radix sorting groups equal tokens and
// puts their contributions in input-row order, independently of GPU scheduling.
__tile_global__ void EmbeddingRowKeysKernel(const int* __restrict__ tokens,
                                            int rows, uint64_t* keys) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto token_view = ct::partition_view{
      ct::tensor_span{tokens, ct::extents{rows}}, ct::shape{1_ic}};
  auto key_view = ct::partition_view{ct::tensor_span{keys, ct::extents{rows}},
                                     ct::shape{1_ic}};
  const int row = ct::bid().x;
  const int token = static_cast<int>(token_view.load(row));
  key_view.store(
      ct::full<ct::tile<uint64_t, ct::shape<1>>>(
          (static_cast<uint64_t>(token) << 32) | static_cast<uint64_t>(row)),
      row);
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
  const int width_tiles = embedding_dim / internal::kDenseTile;
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
  auto accumulator = table_view.load(static_cast<int>(token), width_tile);
  for (int index = start; index < rows; ++index) {
    const uint64_t next = static_cast<uint64_t>(key_view.load(index));
    if ((next >> 32) != token)
      break;
    const int row = static_cast<int>(next & 0xffffffffULL);
    accumulator = accumulator + gradient_view.load(row, width_tile);
  }
  table_view.store(accumulator, static_cast<int>(token), width_tile);
}

template <class Activation>
__tile_global__ void LanguageModelingHeadForwardKernel(
    const Activation* __restrict__ input, const float* __restrict__ table,
    int rows, int padded_vocab_size, int embedding_dim,
    float* __restrict__ output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{rows, embedding_dim}},
      ct::shape{16_ic, 16_ic}};
  auto table_view = ct::partition_view{
      ct::tensor_span{table, ct::extents{padded_vocab_size, embedding_dim}},
      ct::shape{16_ic, 16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{rows, padded_vocab_size}},
      ct::shape{16_ic, 16_ic}};
  const int vocabulary_tiles = padded_vocab_size / internal::kDenseTile;
  const int width_tiles = embedding_dim / internal::kDenseTile;
  const int block = ct::bid().x;
  const int row_tile = block / vocabulary_tiles;
  const int vocabulary_tile = block % vocabulary_tiles;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  for (int dimension_tile = 0; dimension_tile < width_tiles; ++dimension_tile) {
    auto hidden = ct::element_cast<MmaType<Activation>>(
        input_view.load(row_tile, dimension_tile));
    auto embedding_transposed =
        ct::transpose(ct::element_cast<MmaType<Activation>>(
            table_view.load(vocabulary_tile, dimension_tile)));
    accumulator = ct::mma(hidden, embedding_transposed, accumulator);
  }
  output_view.store(accumulator, row_tile, vocabulary_tile);
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

template <class Activation>
__tile_global__ void LanguageModelingHeadInputGradientKernel(
    const float* __restrict__ output_gradient, const float* __restrict__ table,
    int rows, int padded_vocab_size, int embedding_dim,
    float* __restrict__ input_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, padded_vocab_size}},
      ct::shape{16_ic, 16_ic}};
  auto table_view = ct::partition_view{
      ct::tensor_span{table, ct::extents{padded_vocab_size, embedding_dim}},
      ct::shape{16_ic, 16_ic}};
  auto input_gradient_view = ct::partition_view{
      ct::tensor_span{input_gradient, ct::extents{rows, embedding_dim}},
      ct::shape{16_ic, 16_ic}};
  const int width_tiles = embedding_dim / internal::kDenseTile;
  const int vocabulary_tiles = padded_vocab_size / internal::kDenseTile;
  const int block = ct::bid().x;
  const int row_tile = block / width_tiles;
  const int dimension_tile = block % width_tiles;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  for (int vocabulary_tile = 0; vocabulary_tile < vocabulary_tiles;
       ++vocabulary_tile) {
    auto gradient = ct::element_cast<MmaType<Activation>>(
        gradient_view.load(row_tile, vocabulary_tile));
    auto embeddings = ct::element_cast<MmaType<Activation>>(
        table_view.load(vocabulary_tile, dimension_tile));
    accumulator = ct::mma(gradient, embeddings, accumulator);
  }
  input_gradient_view.store(accumulator, row_tile, dimension_tile);
}

template <class Activation>
__tile_global__ void LanguageModelingHeadWeightGradientKernel(
    const Activation* __restrict__ input,
    const float* __restrict__ output_gradient, int rows, int padded_vocab_size,
    int embedding_dim, float* __restrict__ table_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{rows, embedding_dim}},
      ct::shape{16_ic, 16_ic}};
  auto output_gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, padded_vocab_size}},
      ct::shape{16_ic, 16_ic}};
  auto table_gradient_view = ct::partition_view{
      ct::tensor_span{table_gradient,
                      ct::extents{padded_vocab_size, embedding_dim}},
      ct::shape{16_ic, 16_ic}};
  const int width_tiles = embedding_dim / internal::kDenseTile;
  const int row_tiles = rows / internal::kDenseTile;
  const int block = ct::bid().x;
  const int vocabulary_tile = block / width_tiles;
  const int dimension_tile = block % width_tiles;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
    auto gradient_transposed =
        ct::transpose(ct::element_cast<MmaType<Activation>>(
            output_gradient_view.load(row_tile, vocabulary_tile)));
    auto hidden = ct::element_cast<MmaType<Activation>>(
        input_view.load(row_tile, dimension_tile));
    accumulator = ct::mma(gradient_transposed, hidden, accumulator);
  }
  table_gradient_view.store(
      table_gradient_view.load(vocabulary_tile, dimension_tile) + accumulator,
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
  const int width_tiles = embedding_dim / internal::kDenseTile;
  const int block = ct::bid().x;
  const int row = block / width_tiles;
  const int width_tile = block % width_tiles;
  auto sum = ct::element_cast<float>(input_view.load(row, width_tile)) +
             position_view.load(row % context_length, width_tile);
  output_view.store(ct::element_cast<Activation>(sum), row, width_tile);
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
  const int width_tiles = embedding_dim / internal::kDenseTile;
  const int block = ct::bid().x;
  const int position = block / width_tiles;
  const int width_tile = block % width_tiles;
  // One writer per position/width tile, with a fixed sequence-row order.
  // Unvisited positions stay untouched, including for a partial context.
  auto accumulator = position_view.load(position, width_tile);
  // Use a wider loop counter: the final stride may exceed INT_MAX even
  // though every visited row fits the validated int-sized input.
  for (int64_t row = position; row < rows; row += context_length) {
    accumulator =
        accumulator + gradient_view.load(static_cast<int>(row), width_tile);
  }
  position_view.store(accumulator, position, width_tile);
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
  ASSIGN_OR_RETURN(auto values, cuda::PageLockedHostArray<float>::Allocate(
                                    weight.size_bytes() / sizeof(float)));
  for (float& value : values)
    value = distribution(random);
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(weight.data(), values.data(), weight.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      operation));
  return executor.Synchronize();
}

}  // namespace

EmbeddingLookupLayer::EmbeddingLookupLayer(
    cuda::Executor& executor, int vocab_size, int padded_vocab_size,
    int embedding_dim, DataType data_type, Buffer weight, Buffer gradient)
    : vocab_size_(vocab_size),
      padded_vocab_size_(padded_vocab_size),
      embedding_dim_(embedding_dim),
      output_type_(data_type),
      executor_(executor),
      weight_(std::move(weight)),
      gradient_(std::move(gradient)) {}

absl::StatusOr<std::unique_ptr<EmbeddingLookupLayer>>
EmbeddingLookupLayer::Create(cuda::Executor& executor, int vocab_size,
                             int embedding_dim, DataType data_type) {
  RETURN_IF_ERROR(internal::ValidateComputeType(data_type));
  if (vocab_size <= 0)
    return absl::InvalidArgumentError("vocab_size must be positive");
  RETURN_IF_ERROR(
      internal::ValidateTiledExtent(embedding_dim, "embedding_dim"));
  const int padded_vocab_size = internal::RoundUpToTile(vocab_size);
  const size_t bytes =
      static_cast<size_t>(padded_vocab_size) * embedding_dim * sizeof(float);
  ASSIGN_OR_RETURN(auto weight, Buffer::Allocate(executor, bytes));
  ASSIGN_OR_RETURN(auto gradient, Buffer::Allocate(executor, bytes));
  for (Buffer* buffer : {&weight, &gradient}) {
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemsetAsync(buffer->data(), 0, buffer->size_bytes(),
                        executor.stream()),
        "cudaMemsetAsync(embedding parameter)"));
  }
  return std::unique_ptr<EmbeddingLookupLayer>(new EmbeddingLookupLayer(
      executor, vocab_size, padded_vocab_size, embedding_dim, data_type,
      std::move(weight), std::move(gradient)));
}

absl::Status EmbeddingLookupLayer::InitializeIdentity(float scale) {
  ASSIGN_OR_RETURN(auto values, cuda::PageLockedHostArray<float>::Allocate(
                                    static_cast<size_t>(padded_vocab_size_) *
                                    embedding_dim_));
  std::fill(values.begin(), values.end(), 0.0f);
  for (int index = 0; index < std::min(vocab_size_, embedding_dim_); ++index)
    values[static_cast<size_t>(index) * embedding_dim_ + index] = scale;
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(weight_.data(), values.data(), weight_.size_bytes(),
                      cudaMemcpyHostToDevice, executor_.stream()),
      "cudaMemcpyAsync(identity embedding)"));
  return executor_.Synchronize();
}

absl::Status EmbeddingLookupLayer::InitializeNormal(float standard_deviation,
                                                    uint64_t seed) {
  return CopyNormalInitialization(executor_, weight_, standard_deviation, seed,
                                  "cudaMemcpyAsync(normal embedding)");
}

absl::StatusOr<Buffer> EmbeddingLookupLayer::fwd(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    Tape* tape) const {
  RETURN_IF_ERROR(
      internal::ValidateExecutor(executor_, executor, "EmbeddingLookupLayer"));
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "EmbeddingLookupLayer fwd expects token IDs and a non-null tape");
  }
  ASSIGN_OR_RETURN(int rows,
                   internal::ElementCount(executor, inputs[0], sizeof(int),
                                          "embedding token input"));
  ASSIGN_OR_RETURN(
      auto output,
      Buffer::Allocate(executor,
                       static_cast<size_t>(rows) * embedding_dim_ *
                           internal::ActivationElementBytes(output_type_)));
  const int blocks = rows * internal::TileCount(embedding_dim_);
  if (output_type_ == DataType::BF16) {
    EmbeddingForwardKernel<__nv_bfloat16><<<blocks, 1, 0, executor.stream()>>>(
        static_cast<const int*>(inputs[0].data()),
        static_cast<const float*>(weight_.data()), rows, padded_vocab_size_,
        embedding_dim_, static_cast<__nv_bfloat16*>(output.data()));
  } else {
    EmbeddingForwardKernel<float><<<blocks, 1, 0, executor.stream()>>>(
        static_cast<const int*>(inputs[0].data()),
        static_cast<const float*>(weight_.data()), rows, padded_vocab_size_,
        embedding_dim_, static_cast<float*>(output.data()));
  }
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "EmbeddingForwardKernel launch"));
  tape->intermediates = {inputs[0]};
  tape->children.clear();
  return std::move(output);
}

absl::StatusOr<BufferVec> EmbeddingLookupLayer::bwd(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    Tape tape) {
  RETURN_IF_ERROR(
      internal::ValidateExecutor(executor_, executor, "EmbeddingLookupLayer"));
  if (output_gradients.size() != 1 || tape.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "EmbeddingLookupLayer bwd received an incompatible gradient or tape");
  }
  ASSIGN_OR_RETURN(int rows,
                   internal::ElementCount(executor, tape.intermediates[0],
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
  size_t sort_bytes = 0;
  RETURN_IF_ERROR(cuda::CudaStatus(
      cub::DeviceRadixSort::SortKeys(nullptr, sort_bytes, keys_ptr, sorted_ptr,
                                     rows, 0, 64, executor.stream()),
      "query embedding sort scratch"));
  ASSIGN_OR_RETURN(auto scratch, Buffer::Allocate(executor, sort_bytes));
  EmbeddingRowKeysKernel<<<rows, 1, 0, executor.stream()>>>(
      static_cast<const int*>(tape.intermediates[0].data()), rows, keys_ptr);
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "EmbeddingRowKeysKernel launch"));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cub::DeviceRadixSort::SortKeys(scratch.data(), sort_bytes, keys_ptr,
                                     sorted_ptr, rows, 0, 64,
                                     executor.stream()),
      "sort embedding rows"));
  EmbeddingBackwardKernel<<<rows * internal::TileCount(embedding_dim_), 1, 0,
                            executor.stream()>>>(
      sorted_ptr, static_cast<const float*>(output_gradients[0].data()), rows,
      padded_vocab_size_, embedding_dim_,
      static_cast<float*>(gradient_.data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "EmbeddingBackwardKernel launch"));
  return BufferVec{};
}

absl::StatusOr<std::unique_ptr<LanguageModelingHeadLayer>>
LanguageModelingHeadLayer::Create(EmbeddingLookupLayer* embedding) {
  if (embedding == nullptr) {
    return absl::InvalidArgumentError(
        "LanguageModelingHeadLayer requires a non-null embedding");
  }
  return std::unique_ptr<LanguageModelingHeadLayer>(
      new LanguageModelingHeadLayer(embedding));
}

absl::StatusOr<Buffer> LanguageModelingHeadLayer::fwd(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    Tape* tape) const {
  RETURN_IF_ERROR(internal::ValidateExecutor(embedding_->executor_, executor,
                                             "LanguageModelingHeadLayer"));
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "LanguageModelingHeadLayer fwd expects one input and a non-null tape");
  }
  ASSIGN_OR_RETURN(
      int rows, internal::ActivationRows(
                    executor, inputs[0], embedding_->embedding_dim_,
                    embedding_->output_type_, "language-modeling-head input"));
  RETURN_IF_ERROR(
      internal::ValidateTiledExtent(rows, "language-modeling-head rows"));
  ASSIGN_OR_RETURN(
      auto output,
      Buffer::Allocate(executor, static_cast<size_t>(rows) *
                                     embedding_->padded_vocab_size_ *
                                     sizeof(float)));
  const int blocks = internal::TileCount(rows) *
                     internal::TileCount(embedding_->padded_vocab_size_);
  if (embedding_->output_type_ == DataType::BF16) {
    LanguageModelingHeadForwardKernel<__nv_bfloat16>
        <<<blocks, 1, 0, executor.stream()>>>(
            static_cast<const __nv_bfloat16*>(inputs[0].data()),
            static_cast<const float*>(embedding_->weight_.data()), rows,
            embedding_->padded_vocab_size_, embedding_->embedding_dim_,
            static_cast<float*>(output.data()));
  } else {
    LanguageModelingHeadForwardKernel<float>
        <<<blocks, 1, 0, executor.stream()>>>(
            static_cast<const float*>(inputs[0].data()),
            static_cast<const float*>(embedding_->weight_.data()), rows,
            embedding_->padded_vocab_size_, embedding_->embedding_dim_,
            static_cast<float*>(output.data()));
  }
  MaskPaddedLogitsKernel<<<rows, 1, 0, executor.stream()>>>(
      static_cast<float*>(output.data()), rows, embedding_->vocab_size_,
      embedding_->padded_vocab_size_);
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(),
                                   "language-modeling-head forward launch"));
  tape->intermediates = {inputs[0]};
  tape->children.clear();
  return std::move(output);
}

absl::StatusOr<BufferVec> LanguageModelingHeadLayer::bwd(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    Tape tape) {
  RETURN_IF_ERROR(internal::ValidateExecutor(embedding_->executor_, executor,
                                             "LanguageModelingHeadLayer"));
  if (output_gradients.size() != 1 || tape.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "LanguageModelingHeadLayer bwd received an incompatible gradient or "
        "tape");
  }
  ASSIGN_OR_RETURN(
      int rows, internal::MatrixRows(executor, output_gradients[0],
                                     embedding_->padded_vocab_size_,
                                     "language-modeling-head output gradient"));
  RETURN_IF_ERROR(internal::ValidateBuffer(
      executor, tape.intermediates[0],
      static_cast<size_t>(rows) * embedding_->embedding_dim_ *
          internal::ActivationElementBytes(embedding_->output_type_),
      "language-modeling-head saved input"));
  ASSIGN_OR_RETURN(auto input_gradient,
                   Buffer::Allocate(executor, static_cast<size_t>(rows) *
                                                  embedding_->embedding_dim_ *
                                                  sizeof(float)));
  const int input_blocks = internal::TileCount(rows) *
                           internal::TileCount(embedding_->embedding_dim_);
  const int weight_blocks =
      internal::TileCount(embedding_->padded_vocab_size_) *
      internal::TileCount(embedding_->embedding_dim_);
  if (embedding_->output_type_ == DataType::BF16) {
    LanguageModelingHeadInputGradientKernel<__nv_bfloat16>
        <<<input_blocks, 1, 0, executor.stream()>>>(
            static_cast<const float*>(output_gradients[0].data()),
            static_cast<const float*>(embedding_->weight_.data()), rows,
            embedding_->padded_vocab_size_, embedding_->embedding_dim_,
            static_cast<float*>(input_gradient.data()));
    LanguageModelingHeadWeightGradientKernel<__nv_bfloat16>
        <<<weight_blocks, 1, 0, executor.stream()>>>(
            static_cast<const __nv_bfloat16*>(tape.intermediates[0].data()),
            static_cast<const float*>(output_gradients[0].data()), rows,
            embedding_->padded_vocab_size_, embedding_->embedding_dim_,
            static_cast<float*>(embedding_->gradient_.data()));
  } else {
    LanguageModelingHeadInputGradientKernel<float>
        <<<input_blocks, 1, 0, executor.stream()>>>(
            static_cast<const float*>(output_gradients[0].data()),
            static_cast<const float*>(embedding_->weight_.data()), rows,
            embedding_->padded_vocab_size_, embedding_->embedding_dim_,
            static_cast<float*>(input_gradient.data()));
    LanguageModelingHeadWeightGradientKernel<float>
        <<<weight_blocks, 1, 0, executor.stream()>>>(
            static_cast<const float*>(tape.intermediates[0].data()),
            static_cast<const float*>(output_gradients[0].data()), rows,
            embedding_->padded_vocab_size_, embedding_->embedding_dim_,
            static_cast<float*>(embedding_->gradient_.data()));
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
      internal::ValidateTiledExtent(embedding_dim, "embedding_dim"));
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
  return std::unique_ptr<PositionEmbeddingLayer>(new PositionEmbeddingLayer(
      executor, context_length, embedding_dim, data_type, std::move(weight),
      std::move(gradient)));
}

absl::Status PositionEmbeddingLayer::InitializeNormal(float standard_deviation,
                                                      uint64_t seed) {
  return CopyNormalInitialization(executor_, weight_, standard_deviation, seed,
                                  "cudaMemcpyAsync(normal positions)");
}

absl::StatusOr<Buffer> PositionEmbeddingLayer::fwd(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    Tape* tape) const {
  RETURN_IF_ERROR(internal::ValidateExecutor(executor_, executor,
                                             "PositionEmbeddingLayer"));
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "PositionEmbeddingLayer fwd expects one input and a non-null tape");
  }
  ASSIGN_OR_RETURN(int rows, internal::ActivationRows(
                                 executor, inputs[0], embedding_dim_,
                                 output_type_, "position-embedding input"));
  ASSIGN_OR_RETURN(auto output,
                   Buffer::Allocate(executor, inputs[0].size_bytes()));
  const int blocks = rows * internal::TileCount(embedding_dim_);
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
  tape->intermediates.clear();
  tape->children.clear();
  return std::move(output);
}

absl::StatusOr<BufferVec> PositionEmbeddingLayer::bwd(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    Tape tape) {
  RETURN_IF_ERROR(internal::ValidateExecutor(executor_, executor,
                                             "PositionEmbeddingLayer"));
  if (output_gradients.size() != 1 || !tape.intermediates.empty()) {
    return absl::InvalidArgumentError(
        "PositionEmbeddingLayer bwd received an incompatible gradient or "
        "tape");
  }
  ASSIGN_OR_RETURN(int rows, internal::MatrixRows(
                                 executor, output_gradients[0], embedding_dim_,
                                 "position-embedding output gradient"));
  PositionEmbeddingBackwardKernel<<<std::min(rows, context_length_) *
                                        internal::TileCount(embedding_dim_),
                                    1, 0, executor.stream()>>>(
      static_cast<const float*>(output_gradients[0].data()), rows,
      context_length_, embedding_dim_, static_cast<float*>(gradient_.data()));
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(),
                                   "PositionEmbeddingBackwardKernel launch"));
  return BufferVec{output_gradients[0]};
}

}  // namespace pluto::llm
