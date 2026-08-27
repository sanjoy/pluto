#include "src/llm/layers/attention.h"

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
__tile_global__ void FlashAttentionForwardKernel(
    const float* __restrict__ input, int rows, int context_length,
    int num_heads, int embedding_dim, float scale,
    float* __restrict__ output) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};

  const int head_dimension = embedding_dim / num_heads;
  const int head_tiles = head_dimension / kDenseTile;
  const int block = ct::bid().x;
  const int output_tile = block % head_tiles;
  const int row_and_head = block / head_tiles;
  const int row = row_and_head / num_heads;
  const int head = row_and_head % num_heads;
  const int sequence_start = (row / context_length) * context_length;
  const int query_position = row % context_length;
  const int head_tile_start = head * head_tiles;
  auto maximum =
      ct::full<ct::tile<float, ct::shape<1, 1>>>(-3.402823466e+38f);
  auto normalizer = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<1, 16>>>();

  for (int key_position = 0; key_position <= query_position; ++key_position) {
    const int key_row = sequence_start + key_position;
    auto score = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
    for (int dimension_tile = 0; dimension_tile < head_tiles;
         ++dimension_tile) {
      const int tile = head_tile_start + dimension_tile;
      auto query = ct::element_cast<float>(
          ct::element_cast<__half>(input_view.load(row, tile)));
      auto key = ct::element_cast<float>(
          ct::element_cast<__half>(input_view.load(key_row, tile)));
      score = score + ct::sum(query * key, 1_ic);
    }
    score = score * scale;
    auto value = ct::element_cast<float>(ct::element_cast<__half>(
        input_view.load(key_row, head_tile_start + output_tile)));
    auto new_maximum = ct::max(maximum, score);
    auto old_scale = ct::exp(maximum - new_maximum);
    auto new_scale = ct::exp(score - new_maximum);
    accumulator = accumulator * old_scale + value * new_scale;
    normalizer = normalizer * old_scale + new_scale;
    maximum = new_maximum;
  }
  output_view.store(accumulator / normalizer, row,
                    head_tile_start + output_tile);
}

// FlashAttention backward recomputes the causal softmax probabilities instead
// of loading a saved quadratic matrix. Q, K, and V alias the same activation,
// so their three contributions are atomically accumulated into one gradient.
__tile_global__ void FlashAttentionBackwardKernel(
    const float* __restrict__ input, const float* __restrict__ output,
    const float* __restrict__ output_gradient, int rows, int context_length,
    int num_heads, int embedding_dim, float scale,
    float* __restrict__ input_gradient) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};

  const int head_dimension = embedding_dim / num_heads;
  const int head_tiles = head_dimension / kDenseTile;
  const int block = ct::bid().x;
  const int output_tile = block % head_tiles;
  const int row_and_head = block / head_tiles;
  const int row = row_and_head / num_heads;
  const int head = row_and_head % num_heads;
  const int sequence_start = (row / context_length) * context_length;
  const int query_position = row % context_length;
  const int head_tile_start = head * head_tiles;
  auto delta = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  for (int dimension_tile = 0; dimension_tile < head_tiles;
       ++dimension_tile) {
    const int tile = head_tile_start + dimension_tile;
    delta = delta + ct::sum(gradient_view.load(row, tile) *
                                output_view.load(row, tile),
                            1_ic);
  }
  auto maximum =
      ct::full<ct::tile<float, ct::shape<1, 1>>>(-3.402823466e+38f);
  auto normalizer = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();

  for (int key_position = 0; key_position <= query_position; ++key_position) {
    const int key_row = sequence_start + key_position;
    auto score = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
    for (int dimension_tile = 0; dimension_tile < head_tiles;
         ++dimension_tile) {
      const int tile = head_tile_start + dimension_tile;
      auto query = ct::element_cast<float>(
          ct::element_cast<__half>(input_view.load(row, tile)));
      auto key = ct::element_cast<float>(
          ct::element_cast<__half>(input_view.load(key_row, tile)));
      score = score + ct::sum(query * key, 1_ic);
    }
    score = score * scale;
    auto new_maximum = ct::max(maximum, score);
    normalizer = normalizer * ct::exp(maximum - new_maximum) +
                 ct::exp(score - new_maximum);
    maximum = new_maximum;
  }

  auto offsets = ct::iota<ct::tile<int, ct::shape<1, 16>>>() +
                 (head_tile_start + output_tile) * kDenseTile;
  auto query_gradient_pointers =
      input_gradient + row * embedding_dim + offsets;
  auto query = ct::element_cast<float>(ct::element_cast<__half>(
      input_view.load(row, head_tile_start + output_tile)));
  auto d_output = gradient_view.load(row, head_tile_start + output_tile);
  for (int key_position = 0; key_position <= query_position; ++key_position) {
    const int key_row = sequence_start + key_position;
    auto score = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
    auto gradient_dot_key = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
    for (int dimension_tile = 0; dimension_tile < head_tiles;
         ++dimension_tile) {
      const int tile = head_tile_start + dimension_tile;
      auto full_query = ct::element_cast<float>(
          ct::element_cast<__half>(input_view.load(row, tile)));
      auto full_key = ct::element_cast<float>(
          ct::element_cast<__half>(input_view.load(key_row, tile)));
      score = score + ct::sum(full_query * full_key, 1_ic);
      gradient_dot_key =
          gradient_dot_key +
          ct::sum(gradient_view.load(row, tile) * full_key, 1_ic);
    }
    score = score * scale;
    auto probability = ct::exp(score - maximum) / normalizer;
    auto d_score = probability * (gradient_dot_key - delta);
    auto key = ct::element_cast<float>(ct::element_cast<__half>(
        input_view.load(key_row, head_tile_start + output_tile)));
    auto key_gradient_pointers =
        input_gradient + key_row * embedding_dim + offsets;
    ct::atomic_add<ct::memory_order::relaxed>(query_gradient_pointers,
                                               d_score * key * scale);
    ct::atomic_add<ct::memory_order::relaxed>(key_gradient_pointers,
                                               d_score * query * scale);
    ct::atomic_add<ct::memory_order::relaxed>(key_gradient_pointers,
                                               probability * d_output);
  }
}


}  // namespace

absl::StatusOr<std::unique_ptr<AttentionLayer>> AttentionLayer::Create(
    int context_length, int num_heads, int embedding_dim, DataType data_type,
    cudaStream_t stream) {
  if (auto status = ValidateFp16(data_type); !status.ok()) return status;
  if (context_length <= 0 || num_heads <= 0 || embedding_dim <= 0) {
    return absl::InvalidArgumentError(
        "attention dimensions must all be positive");
  }
  if (embedding_dim % num_heads != 0) {
    return absl::InvalidArgumentError(
        "embedding_dim must be divisible by num_heads");
  }
  if (auto status = ValidateTiledExtent(embedding_dim / num_heads,
                                        "attention head dimension");
      !status.ok()) return status;
  return std::unique_ptr<AttentionLayer>(new AttentionLayer(
      context_length, num_heads, embedding_dim, data_type, stream));
}

absl::StatusOr<Buffer> AttentionLayer::fwd(
    absl::Span<const Buffer> inputs, Tape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "AttentionLayer fwd expects one input and a non-null tape");
  }
  auto rows = MatrixRows(inputs[0], embedding_dim_, stream_, "attention input");
  if (!rows.ok()) return rows.status();
  if (*rows % context_length_ != 0) {
    return absl::InvalidArgumentError(
        "attention rows must be divisible by context_length");
  }
  auto output = Buffer::Allocate(inputs[0].size_bytes(), stream_);
  if (!output.ok()) return output.status();
  const int head_dimension = embedding_dim_ / num_heads_;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dimension));
  FlashAttentionForwardKernel<<<
      *rows * num_heads_ * TileCount(head_dimension), 1, 0, stream_>>>(
      static_cast<const float*>(inputs[0].data()),
      *rows, context_length_, num_heads_, embedding_dim_, scale,
      static_cast<float*>(output->data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "FlashAttentionForwardKernel launch");
      !status.ok()) {
    return status;
  }
  tape->intermediates = {inputs[0], *output};
  tape->children.clear();
  return *std::move(output);
}

absl::StatusOr<BufferVec> AttentionLayer::bwd(
    absl::Span<const Buffer> output_gradients, Tape tape) {
  if (output_gradients.size() != 1 || tape.intermediates.size() != 2) {
    return absl::InvalidArgumentError(
        "AttentionLayer bwd received an incompatible gradient or tape");
  }
  auto rows = MatrixRows(output_gradients[0], embedding_dim_, stream_,
                         "attention output gradient");
  if (!rows.ok()) return rows.status();
  const size_t activation_bytes = output_gradients[0].size_bytes();
  for (const Buffer& saved : tape.intermediates) {
    if (auto status = ValidateBuffer(saved, activation_bytes, stream_,
                                     "attention saved activation");
        !status.ok()) return status;
  }
  auto input_gradient = Buffer::Allocate(activation_bytes, stream_);
  if (!input_gradient.ok()) return input_gradient.status();
  if (auto status = CudaStatus(
          cudaMemsetAsync(input_gradient->data(), 0,
                          input_gradient->size_bytes(), stream_),
          "cudaMemsetAsync(attention input gradient)");
      !status.ok()) {
    return status;
  }
  const int head_dimension = embedding_dim_ / num_heads_;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dimension));
  FlashAttentionBackwardKernel<<<
      *rows * num_heads_ * TileCount(head_dimension), 1, 0, stream_>>>(
      static_cast<const float*>(tape.intermediates[0].data()),
      static_cast<const float*>(tape.intermediates[1].data()),
      static_cast<const float*>(output_gradients[0].data()),
      *rows, context_length_, num_heads_, embedding_dim_, scale,
      static_cast<float*>(input_gradient->data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "FlashAttentionBackwardKernel launch");
      !status.ok()) {
    return status;
  }
  return BufferVec{*std::move(input_gradient)};
}


}  // namespace pluto::llm
