#include "src/llm/layers/embedding.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "src/gpu/buffer.h"
#include "src/llm/layers/internal.h"

namespace pluto::llm {

using internal::CudaStatus;
using internal::ElementCount;
using internal::kDenseTile;
using internal::MatrixRows;
using internal::TileCount;
using internal::ValidateBuffer;
using internal::ValidateFp16;
using internal::ValidateTiledExtent;

namespace {
__tile_global__ void EmbeddingForwardKernel(
    const int* __restrict__ tokens, const float* __restrict__ table,
    int rows, int vocabulary_size, int embedding_dim,
    float* __restrict__ output) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto token_view = ct::partition_view{
      ct::tensor_span{tokens, ct::extents{rows}}, ct::shape{1_ic}};
  auto table_view = ct::partition_view{
      ct::tensor_span{table, ct::extents{vocabulary_size, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};

  const int width_tiles = embedding_dim / kDenseTile;
  const int block = ct::bid().x;
  const int row = block / width_tiles;
  const int width_tile = block % width_tiles;
  const int token = static_cast<int>(token_view.load(row));
  auto master = table_view.load(token, width_tile);
  auto fp16 = ct::element_cast<__half>(master);
  output_view.store(ct::element_cast<float>(fp16), row, width_tile);
}

__tile_global__ void EmbeddingBackwardKernel(
    const int* __restrict__ tokens, const float* __restrict__ output_gradient,
    int rows, int embedding_dim, float learning_rate,
    float* __restrict__ table) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto token_view = ct::partition_view{
      ct::tensor_span{tokens, ct::extents{rows}}, ct::shape{1_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};

  const int width_tiles = embedding_dim / kDenseTile;
  const int block = ct::bid().x;
  const int row = block / width_tiles;
  const int width_tile = block % width_tiles;
  const int token = static_cast<int>(token_view.load(row));
  auto offsets = ct::iota<ct::tile<int, ct::shape<1, 16>>>() +
                 width_tile * kDenseTile;
  auto pointers = table + token * embedding_dim + offsets;
  ct::atomic_sub<ct::memory_order::relaxed>(
      pointers, gradient_view.load(row, width_tile) * learning_rate);
}

// The language-modeling head ties its projection matrix to the embedding
// table. Embeddings are laid out [vocabulary, model width], so the forward pass
// multiplies hidden states by the table's transpose.
__tile_global__ void LanguageModelingHeadForwardKernel(
    const float* __restrict__ input, const float* __restrict__ table,
    int rows, int vocabulary_size, int embedding_dim,
    float* __restrict__ output) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{rows, embedding_dim}},
      ct::shape{16_ic, 16_ic}};
  auto table_view = ct::partition_view{
      ct::tensor_span{table, ct::extents{vocabulary_size, embedding_dim}},
      ct::shape{16_ic, 16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{rows, vocabulary_size}},
      ct::shape{16_ic, 16_ic}};

  const int vocabulary_tiles = vocabulary_size / kDenseTile;
  const int width_tiles = embedding_dim / kDenseTile;
  const int block = ct::bid().x;
  const int batch_tile = block / vocabulary_tiles;
  const int vocabulary_tile = block % vocabulary_tiles;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  for (int dimension_tile = 0; dimension_tile < width_tiles;
       ++dimension_tile) {
    auto hidden =
        ct::element_cast<__half>(input_view.load(batch_tile, dimension_tile));
    auto embeddings_transposed = ct::transpose(ct::element_cast<__half>(
        table_view.load(vocabulary_tile, dimension_tile)));
    accumulator = ct::mma(hidden, embeddings_transposed, accumulator);
  }
  output_view.store(accumulator, batch_tile, vocabulary_tile);
}

__tile_global__ void LanguageModelingHeadInputGradientKernel(
    const float* __restrict__ output_gradient,
    const float* __restrict__ table, int rows, int vocabulary_size,
    int embedding_dim, float* __restrict__ input_gradient) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, vocabulary_size}},
      ct::shape{16_ic, 16_ic}};
  auto table_view = ct::partition_view{
      ct::tensor_span{table, ct::extents{vocabulary_size, embedding_dim}},
      ct::shape{16_ic, 16_ic}};
  auto input_gradient_view = ct::partition_view{
      ct::tensor_span{input_gradient, ct::extents{rows, embedding_dim}},
      ct::shape{16_ic, 16_ic}};

  const int width_tiles = embedding_dim / kDenseTile;
  const int vocabulary_tiles = vocabulary_size / kDenseTile;
  const int block = ct::bid().x;
  const int batch_tile = block / width_tiles;
  const int dimension_tile = block % width_tiles;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  for (int vocabulary_tile = 0; vocabulary_tile < vocabulary_tiles;
       ++vocabulary_tile) {
    auto gradient = ct::element_cast<__half>(
        gradient_view.load(batch_tile, vocabulary_tile));
    auto embeddings = ct::element_cast<__half>(
        table_view.load(vocabulary_tile, dimension_tile));
    accumulator = ct::mma(gradient, embeddings, accumulator);
  }
  input_gradient_view.store(accumulator, batch_tile, dimension_tile);
}

__tile_global__ void LanguageModelingHeadWeightUpdateKernel(
    const float* __restrict__ input,
    const float* __restrict__ output_gradient, float learning_rate,
    int rows, int vocabulary_size, int embedding_dim,
    float* __restrict__ table) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{rows, embedding_dim}},
      ct::shape{16_ic, 16_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, vocabulary_size}},
      ct::shape{16_ic, 16_ic}};
  auto table_view = ct::partition_view{
      ct::tensor_span{table, ct::extents{vocabulary_size, embedding_dim}},
      ct::shape{16_ic, 16_ic}};

  const int width_tiles = embedding_dim / kDenseTile;
  const int batch_tiles = rows / kDenseTile;
  const int block = ct::bid().x;
  const int vocabulary_tile = block / width_tiles;
  const int dimension_tile = block % width_tiles;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  for (int batch_tile = 0; batch_tile < batch_tiles; ++batch_tile) {
    auto gradient_transposed = ct::transpose(ct::element_cast<__half>(
        gradient_view.load(batch_tile, vocabulary_tile)));
    auto hidden =
        ct::element_cast<__half>(input_view.load(batch_tile, dimension_tile));
    accumulator = ct::mma(gradient_transposed, hidden, accumulator);
  }
  auto old_table = table_view.load(vocabulary_tile, dimension_tile);
  table_view.store(old_table - learning_rate * accumulator, vocabulary_tile,
                   dimension_tile);
}

__tile_global__ void PositionEmbeddingForwardKernel(
    const float* __restrict__ input, const float* __restrict__ positions,
    int rows, int context_length, int embedding_dim,
    float* __restrict__ output) {
  namespace ct = cuda::tiles;
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

  const int width_tiles = embedding_dim / kDenseTile;
  const int block = ct::bid().x;
  const int row = block / width_tiles;
  const int width_tile = block % width_tiles;
  output_view.store(input_view.load(row, width_tile) +
                        position_view.load(row % context_length, width_tile),
                    row, width_tile);
}

__tile_global__ void PositionEmbeddingBackwardKernel(
    const float* __restrict__ output_gradient, float learning_rate,
    int rows, int context_length, int embedding_dim,
    float* __restrict__ positions) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  const int width_tiles = embedding_dim / kDenseTile;
  const int block = ct::bid().x;
  const int row = block / width_tiles;
  const int width_tile = block % width_tiles;
  auto offsets = ct::iota<ct::tile<int, ct::shape<1, 16>>>() +
                 width_tile * kDenseTile;
  auto pointers = positions + (row % context_length) * embedding_dim + offsets;
  ct::atomic_sub<ct::memory_order::relaxed>(
      pointers, gradient_view.load(row, width_tile) * learning_rate);
}

// One program handles one query and one head. Keys and values are streamed
// through registers while an online maximum and normalizer keep the softmax
// stable. No score/probability matrix is written to global memory.

}  // namespace

EmbeddingLookupLayer::EmbeddingLookupLayer(
    int vocab_size, int embedding_dim, DataType data_type, float learning_rate,
    cudaStream_t stream, Buffer weight)
    : vocab_size_(vocab_size),
      embedding_dim_(embedding_dim),
      output_type_(data_type),
      learning_rate_(learning_rate),
      stream_(stream),
      weight_(std::move(weight)) {}

absl::StatusOr<std::unique_ptr<EmbeddingLookupLayer>>
EmbeddingLookupLayer::Create(int vocab_size, int embedding_dim,
                             DataType data_type, float learning_rate,
                             cudaStream_t stream) {
  if (auto status = ValidateFp16(data_type); !status.ok()) return status;
  if (learning_rate < 0.0f) {
    return absl::InvalidArgumentError("learning rate must be non-negative");
  }
  if (auto status = ValidateTiledExtent(vocab_size, "vocab_size");
      !status.ok()) return status;
  if (auto status = ValidateTiledExtent(embedding_dim, "embedding_dim");
      !status.ok()) return status;
  auto weight = Buffer::Allocate(
      static_cast<size_t>(vocab_size) * embedding_dim * sizeof(float), stream);
  if (!weight.ok()) return weight.status();
  if (auto status = CudaStatus(cudaMemsetAsync(weight->data(), 0,
                                               weight->size_bytes(), stream),
                               "cudaMemsetAsync(embedding table)");
      !status.ok()) {
    return status;
  }
  return std::unique_ptr<EmbeddingLookupLayer>(new EmbeddingLookupLayer(
      vocab_size, embedding_dim, data_type, learning_rate, stream,
      *std::move(weight)));
}

absl::Status EmbeddingLookupLayer::InitializeIdentity(float scale) {
  std::vector<float> identity(
      static_cast<size_t>(vocab_size_) * embedding_dim_, 0.0f);
  for (int index = 0; index < std::min(vocab_size_, embedding_dim_); ++index) {
    identity[static_cast<size_t>(index) * embedding_dim_ + index] = scale;
  }
  return CudaStatus(cudaMemcpyAsync(weight_.data(), identity.data(),
                                    weight_.size_bytes(),
                                    cudaMemcpyHostToDevice, stream_),
                    "cudaMemcpyAsync(identity embedding)");
}

absl::StatusOr<Buffer> EmbeddingLookupLayer::fwd(
    absl::Span<const Buffer> inputs, Tape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "EmbeddingLookupLayer fwd expects one input and a non-null tape");
  }
  auto rows = ElementCount(inputs[0], sizeof(int), stream_,
                           "embedding token input");
  if (!rows.ok()) return rows.status();
  auto output = Buffer::Allocate(
      static_cast<size_t>(*rows) * embedding_dim_ * sizeof(float), stream_);
  if (!output.ok()) return output.status();
  tape->intermediates = {inputs[0]};
  tape->children.clear();
  EmbeddingForwardKernel<<<*rows * TileCount(embedding_dim_), 1, 0, stream_>>>(
      static_cast<const int*>(inputs[0].data()),
      static_cast<const float*>(weight_.data()),
      *rows, vocab_size_, embedding_dim_,
      static_cast<float*>(output->data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "EmbeddingForwardKernel launch");
      !status.ok()) {
    return status;
  }
  return *std::move(output);
}

absl::StatusOr<BufferVec> EmbeddingLookupLayer::bwd(
    absl::Span<const Buffer> output_gradients, Tape tape) {
  if (output_gradients.size() != 1 || tape.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "EmbeddingLookupLayer bwd received an incompatible gradient or tape");
  }
  auto rows = ElementCount(tape.intermediates[0], sizeof(int), stream_,
                           "embedding token input");
  if (!rows.ok()) return rows.status();
  if (auto status = ValidateBuffer(
          output_gradients[0],
          static_cast<size_t>(*rows) * embedding_dim_ * sizeof(float), stream_,
          "embedding output gradient");
      !status.ok()) return status;
  EmbeddingBackwardKernel<<<*rows * TileCount(embedding_dim_), 1, 0, stream_>>>(
      static_cast<const int*>(tape.intermediates[0].data()),
      static_cast<const float*>(output_gradients[0].data()), *rows,
      embedding_dim_, learning_rate_,
      static_cast<float*>(weight_.data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "EmbeddingBackwardKernel launch");
      !status.ok()) {
    return status;
  }
  // Integer token IDs are not differentiable.
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
    absl::Span<const Buffer> inputs, Tape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "LanguageModelingHeadLayer fwd expects one input and a non-null tape");
  }
  auto rows = MatrixRows(inputs[0], embedding_->embedding_dim_,
                         embedding_->stream_, "language-modeling-head input");
  if (!rows.ok()) return rows.status();
  if (auto status = ValidateTiledExtent(*rows, "language-modeling-head rows");
      !status.ok()) return status;
  auto output = Buffer::Allocate(
      static_cast<size_t>(*rows) * embedding_->vocab_size_ * sizeof(float),
      embedding_->stream_);
  if (!output.ok()) return output.status();
  tape->intermediates = {inputs[0]};
  tape->children.clear();
  LanguageModelingHeadForwardKernel<<<
      TileCount(*rows) * TileCount(embedding_->vocab_size_), 1, 0,
      embedding_->stream_>>>(
      static_cast<const float*>(inputs[0].data()),
      static_cast<const float*>(embedding_->weight_.data()),
      *rows, embedding_->vocab_size_, embedding_->embedding_dim_,
      static_cast<float*>(output->data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "LanguageModelingHeadForwardKernel launch");
      !status.ok()) {
    return status;
  }
  return *std::move(output);
}

absl::StatusOr<BufferVec> LanguageModelingHeadLayer::bwd(
    absl::Span<const Buffer> output_gradients, Tape tape) {
  if (output_gradients.size() != 1 || tape.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "LanguageModelingHeadLayer bwd received an incompatible gradient or "
        "tape");
  }
  auto rows = MatrixRows(output_gradients[0], embedding_->vocab_size_,
                         embedding_->stream_,
                         "language-modeling-head output gradient");
  if (!rows.ok()) return rows.status();
  if (auto status = ValidateBuffer(
          tape.intermediates[0],
          static_cast<size_t>(*rows) * embedding_->embedding_dim_ *
              sizeof(float),
          embedding_->stream_, "language-modeling-head saved input");
      !status.ok()) return status;
  auto input_gradient = Buffer::Allocate(
      static_cast<size_t>(*rows) * embedding_->embedding_dim_ * sizeof(float),
      embedding_->stream_);
  if (!input_gradient.ok()) return input_gradient.status();
  LanguageModelingHeadInputGradientKernel<<<
      TileCount(*rows) * TileCount(embedding_->embedding_dim_), 1, 0,
      embedding_->stream_>>>(
      static_cast<const float*>(output_gradients[0].data()),
      static_cast<const float*>(embedding_->weight_.data()),
      *rows, embedding_->vocab_size_, embedding_->embedding_dim_,
      static_cast<float*>(input_gradient->data()));
  if (embedding_->learning_rate_ != 0.0f) {
    LanguageModelingHeadWeightUpdateKernel<<<
        TileCount(embedding_->vocab_size_) *
            TileCount(embedding_->embedding_dim_),
        1, 0, embedding_->stream_>>>(
        static_cast<const float*>(tape.intermediates[0].data()),
        static_cast<const float*>(output_gradients[0].data()),
        embedding_->learning_rate_, *rows, embedding_->vocab_size_,
        embedding_->embedding_dim_,
        static_cast<float*>(embedding_->weight_.data()));
  }
  if (auto status = CudaStatus(cudaGetLastError(),
                               "language-modeling-head backward launch");
      !status.ok()) {
    return status;
  }
  return BufferVec{*std::move(input_gradient)};
}

PositionEmbeddingLayer::PositionEmbeddingLayer(
    int context_length, int embedding_dim, DataType data_type,
    float learning_rate, cudaStream_t stream, Buffer weight)
    : context_length_(context_length),
      embedding_dim_(embedding_dim),
      output_type_(data_type),
      learning_rate_(learning_rate),
      stream_(stream),
      weight_(std::move(weight)) {}

absl::StatusOr<std::unique_ptr<PositionEmbeddingLayer>>
PositionEmbeddingLayer::Create(int context_length, int embedding_dim,
                               DataType data_type, float learning_rate,
                               cudaStream_t stream) {
  if (auto status = ValidateFp16(data_type); !status.ok()) return status;
  if (learning_rate < 0.0f) {
    return absl::InvalidArgumentError("learning rate must be non-negative");
  }
  if (context_length <= 0) {
    return absl::InvalidArgumentError("context_length must be positive");
  }
  if (auto status = ValidateTiledExtent(embedding_dim, "embedding_dim");
      !status.ok()) return status;
  auto weight = Buffer::Allocate(
      static_cast<size_t>(context_length) * embedding_dim * sizeof(float),
      stream);
  if (!weight.ok()) return weight.status();
  if (auto status = CudaStatus(cudaMemsetAsync(weight->data(), 0,
                                               weight->size_bytes(), stream),
                               "cudaMemsetAsync(position embedding)");
      !status.ok()) {
    return status;
  }
  return std::unique_ptr<PositionEmbeddingLayer>(new PositionEmbeddingLayer(
      context_length, embedding_dim, data_type, learning_rate, stream,
      *std::move(weight)));
}

absl::StatusOr<Buffer> PositionEmbeddingLayer::fwd(
    absl::Span<const Buffer> inputs, Tape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "PositionEmbeddingLayer fwd expects one input and a non-null tape");
  }
  auto rows = MatrixRows(inputs[0], embedding_dim_, stream_,
                         "position-embedding input");
  if (!rows.ok()) return rows.status();
  auto output = Buffer::Allocate(inputs[0].size_bytes(), stream_);
  if (!output.ok()) return output.status();
  tape->intermediates.clear();
  tape->children.clear();
  PositionEmbeddingForwardKernel<<<*rows * TileCount(embedding_dim_), 1, 0,
                                   stream_>>>(
      static_cast<const float*>(inputs[0].data()),
      static_cast<const float*>(weight_.data()),
      *rows, context_length_, embedding_dim_,
      static_cast<float*>(output->data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "PositionEmbeddingForwardKernel launch");
      !status.ok()) {
    return status;
  }
  return *std::move(output);
}

absl::StatusOr<BufferVec> PositionEmbeddingLayer::bwd(
    absl::Span<const Buffer> output_gradients, Tape tape) {
  if (output_gradients.size() != 1 || !tape.intermediates.empty() ||
      !tape.children.empty()) {
    return absl::InvalidArgumentError(
        "PositionEmbeddingLayer bwd received an incompatible gradient or "
        "tape");
  }
  auto rows = MatrixRows(output_gradients[0], embedding_dim_, stream_,
                         "position-embedding output gradient");
  if (!rows.ok()) return rows.status();
  PositionEmbeddingBackwardKernel<<<*rows * TileCount(embedding_dim_), 1, 0,
                                    stream_>>>(
      static_cast<const float*>(output_gradients[0].data()), learning_rate_,
      *rows, context_length_, embedding_dim_,
      static_cast<float*>(weight_.data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "PositionEmbeddingBackwardKernel launch");
      !status.ok()) {
    return status;
  }
  // The additive path has derivative one and can share the upstream buffer.
  return BufferVec{output_gradients[0]};
}


}  // namespace pluto::llm
