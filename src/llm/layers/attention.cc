#include "src/llm/layers/attention.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/common/status_macros.h"
#include "src/llm/layers/internal.h"

namespace pluto::llm {
namespace {

template <class Activation>
__tile_global__ void FlashAttentionForwardKernel(
    const Activation* __restrict__ qkv, int rows, int context_length,
    int num_heads, int embedding_dim, float scale,
    Activation* __restrict__ output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto qkv_view = ct::partition_view{
      ct::tensor_span{qkv, ct::extents{rows, 3 * embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};

  const int dimension_tiles = embedding_dim / internal::kDenseTile;
  const int head_dimension = embedding_dim / num_heads;
  const int head_tiles = head_dimension / internal::kDenseTile;
  const int block = ct::bid().x;
  const int output_tile = block % head_tiles;
  const int row_and_head = block / head_tiles;
  const int row = row_and_head / num_heads;
  const int head = row_and_head % num_heads;
  const int sequence_start = (row / context_length) * context_length;
  const int query_position = row % context_length;
  const int head_tile_start = head * head_tiles;
  auto maximum = ct::full<ct::tile<float, ct::shape<1, 1>>>(-3.402823466e+38f);
  auto normalizer = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<1, 16>>>();

  for (int key_position = 0; key_position <= query_position; ++key_position) {
    const int key_row = sequence_start + key_position;
    auto score = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
    for (int dimension_tile = 0; dimension_tile < head_tiles;
         ++dimension_tile) {
      const int tile = head_tile_start + dimension_tile;
      auto query = ct::element_cast<float>(qkv_view.load(row, tile));
      auto key = ct::element_cast<float>(
          qkv_view.load(key_row, dimension_tiles + tile));
      score = score + ct::sum(query * key, 1_ic);
    }
    score = score * scale;
    auto value = ct::element_cast<float>(qkv_view.load(
        key_row, 2 * dimension_tiles + head_tile_start + output_tile));
    auto new_maximum = ct::max(maximum, score);
    auto old_scale = ct::exp(maximum - new_maximum);
    auto new_scale = ct::exp(score - new_maximum);
    accumulator = accumulator * old_scale + value * new_scale;
    normalizer = normalizer * old_scale + new_scale;
    maximum = new_maximum;
  }
  output_view.store(ct::element_cast<Activation>(accumulator / normalizer), row,
                    head_tile_start + output_tile);
}

template <class Activation>
__tile_global__ void FlashAttentionBackwardKernel(
    const Activation* __restrict__ qkv, const Activation* __restrict__ output,
    const float* __restrict__ output_gradient, int rows, int context_length,
    int num_heads, int embedding_dim, float scale,
    float* __restrict__ qkv_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto qkv_view = ct::partition_view{
      ct::tensor_span{qkv, ct::extents{rows, 3 * embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{rows, embedding_dim}},
      ct::shape{1_ic, 16_ic}};

  const int dimension_tiles = embedding_dim / internal::kDenseTile;
  const int packed_dimension = 3 * embedding_dim;
  const int head_dimension = embedding_dim / num_heads;
  const int head_tiles = head_dimension / internal::kDenseTile;
  const int block = ct::bid().x;
  const int output_tile = block % head_tiles;
  const int row_and_head = block / head_tiles;
  const int row = row_and_head / num_heads;
  const int head = row_and_head % num_heads;
  const int sequence_start = (row / context_length) * context_length;
  const int query_position = row % context_length;
  const int head_tile_start = head * head_tiles;

  auto delta = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  for (int dimension_tile = 0; dimension_tile < head_tiles; ++dimension_tile) {
    const int tile = head_tile_start + dimension_tile;
    delta = delta +
            ct::sum(gradient_view.load(row, tile) *
                        ct::element_cast<float>(output_view.load(row, tile)),
                    1_ic);
  }
  auto maximum = ct::full<ct::tile<float, ct::shape<1, 1>>>(-3.402823466e+38f);
  auto normalizer = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  for (int key_position = 0; key_position <= query_position; ++key_position) {
    const int key_row = sequence_start + key_position;
    auto score = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
    for (int dimension_tile = 0; dimension_tile < head_tiles;
         ++dimension_tile) {
      const int tile = head_tile_start + dimension_tile;
      score =
          score + ct::sum(ct::element_cast<float>(qkv_view.load(row, tile)) *
                              ct::element_cast<float>(qkv_view.load(
                                  key_row, dimension_tiles + tile)),
                          1_ic);
    }
    score = score * scale;
    auto new_maximum = ct::max(maximum, score);
    normalizer = normalizer * ct::exp(maximum - new_maximum) +
                 ct::exp(score - new_maximum);
    maximum = new_maximum;
  }

  const int tile = head_tile_start + output_tile;
  auto lane_offsets =
      ct::iota<ct::tile<int, ct::shape<1, 16>>>() + tile * internal::kDenseTile;
  auto query_gradient_pointers =
      qkv_gradient + row * packed_dimension + lane_offsets;
  auto query = ct::element_cast<float>(qkv_view.load(row, tile));
  auto d_output = gradient_view.load(row, tile);
  for (int key_position = 0; key_position <= query_position; ++key_position) {
    const int key_row = sequence_start + key_position;
    auto score = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
    auto d_probability = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
    for (int dimension_tile = 0; dimension_tile < head_tiles;
         ++dimension_tile) {
      const int full_tile = head_tile_start + dimension_tile;
      score = score +
              ct::sum(ct::element_cast<float>(qkv_view.load(row, full_tile)) *
                          ct::element_cast<float>(qkv_view.load(
                              key_row, dimension_tiles + full_tile)),
                      1_ic);
      d_probability =
          d_probability +
          ct::sum(gradient_view.load(row, full_tile) *
                      ct::element_cast<float>(qkv_view.load(
                          key_row, 2 * dimension_tiles + full_tile)),
                  1_ic);
    }
    score = score * scale;
    auto probability = ct::exp(score - maximum) / normalizer;
    auto d_score = probability * (d_probability - delta);
    auto key =
        ct::element_cast<float>(qkv_view.load(key_row, dimension_tiles + tile));
    auto key_gradient_pointers = qkv_gradient + key_row * packed_dimension +
                                 embedding_dim + lane_offsets;
    auto value_gradient_pointers = qkv_gradient + key_row * packed_dimension +
                                   2 * embedding_dim + lane_offsets;
    ct::atomic_add<ct::memory_order::relaxed>(query_gradient_pointers,
                                              d_score * key * scale);
    ct::atomic_add<ct::memory_order::relaxed>(key_gradient_pointers,
                                              d_score * query * scale);
    ct::atomic_add<ct::memory_order::relaxed>(value_gradient_pointers,
                                              probability * d_output);
  }
}

}  // namespace

absl::StatusOr<std::unique_ptr<AttentionLayer>> AttentionLayer::Create(
    int context_length, int num_heads, int embedding_dim, DataType data_type,
    cuda::Executor* executor) {
  if (executor == nullptr) {
    return absl::InvalidArgumentError(
        "AttentionLayer requires a non-null CUDA Executor");
  }
  RETURN_IF_ERROR(internal::ValidateComputeType(data_type));
  if (context_length <= 0 || num_heads <= 0 || embedding_dim <= 0) {
    return absl::InvalidArgumentError(
        "attention dimensions must all be positive");
  }
  if (embedding_dim % num_heads != 0) {
    return absl::InvalidArgumentError(
        "embedding_dim must be divisible by num_heads");
  }
  RETURN_IF_ERROR(internal::ValidateTiledExtent(embedding_dim / num_heads,
                                                "attention head dimension"));
  return std::unique_ptr<AttentionLayer>(new AttentionLayer(
      context_length, num_heads, embedding_dim, data_type, executor));
}

absl::StatusOr<Buffer> AttentionLayer::fwd(absl::Span<const Buffer> inputs,
                                           Tape* tape,
                                           cuda::Executor* executor) const {
  RETURN_IF_ERROR(
      internal::ValidateExecutor(executor_, executor, "AttentionLayer"));
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "AttentionLayer fwd expects packed Q/K/V and a non-null tape");
  }
  ASSIGN_OR_RETURN(int rows, internal::ActivationRows(
                                 inputs[0], 3 * embedding_dim_, output_type_,
                                 executor, "attention packed Q/K/V input"));
  if (rows % context_length_ != 0) {
    return absl::InvalidArgumentError(
        "attention rows must be divisible by context_length");
  }
  ASSIGN_OR_RETURN(
      auto output,
      Buffer::Allocate(static_cast<size_t>(rows) * embedding_dim_ *
                           internal::ActivationElementBytes(output_type_),
                       executor));
  const int head_dimension = embedding_dim_ / num_heads_;
  const int blocks = rows * num_heads_ * internal::TileCount(head_dimension);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dimension));
  if (output_type_ == DataType::BF16) {
    FlashAttentionForwardKernel<__nv_bfloat16>
        <<<blocks, 1, 0, executor->stream()>>>(
            static_cast<const __nv_bfloat16*>(inputs[0].data()), rows,
            context_length_, num_heads_, embedding_dim_, scale,
            static_cast<__nv_bfloat16*>(output.data()));
  } else {
    FlashAttentionForwardKernel<float><<<blocks, 1, 0, executor->stream()>>>(
        static_cast<const float*>(inputs[0].data()), rows, context_length_,
        num_heads_, embedding_dim_, scale, static_cast<float*>(output.data()));
  }
  RETURN_IF_ERROR(internal::CudaStatus(cudaGetLastError(),
                                       "FlashAttentionForwardKernel launch"));
  tape->intermediates = {inputs[0], output};
  tape->children.clear();
  return std::move(output);
}

absl::StatusOr<BufferVec> AttentionLayer::bwd(
    absl::Span<const Buffer> output_gradients, Tape tape,
    cuda::Executor* executor) {
  RETURN_IF_ERROR(
      internal::ValidateExecutor(executor_, executor, "AttentionLayer"));
  if (output_gradients.size() != 1 || tape.intermediates.size() != 2) {
    return absl::InvalidArgumentError(
        "AttentionLayer bwd received an incompatible gradient or tape");
  }
  ASSIGN_OR_RETURN(int rows,
                   internal::MatrixRows(output_gradients[0], embedding_dim_,
                                        executor, "attention output gradient"));
  RETURN_IF_ERROR(internal::ValidateBuffer(
      tape.intermediates[0],
      static_cast<size_t>(rows) * 3 * embedding_dim_ *
          internal::ActivationElementBytes(output_type_),
      executor, "attention saved Q/K/V"));
  RETURN_IF_ERROR(internal::ValidateBuffer(
      tape.intermediates[1],
      static_cast<size_t>(rows) * embedding_dim_ *
          internal::ActivationElementBytes(output_type_),
      executor, "attention saved output"));
  ASSIGN_OR_RETURN(auto qkv_gradient,
                   Buffer::Allocate(static_cast<size_t>(rows) * 3 *
                                        embedding_dim_ * sizeof(float),
                                    executor));
  RETURN_IF_ERROR(internal::CudaStatus(
      cudaMemsetAsync(qkv_gradient.data(), 0, qkv_gradient.size_bytes(),
                      executor->stream()),
      "cudaMemsetAsync(attention Q/K/V gradient)"));
  const int head_dimension = embedding_dim_ / num_heads_;
  const int blocks = rows * num_heads_ * internal::TileCount(head_dimension);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dimension));
  if (output_type_ == DataType::BF16) {
    FlashAttentionBackwardKernel<__nv_bfloat16>
        <<<blocks, 1, 0, executor->stream()>>>(
            static_cast<const __nv_bfloat16*>(tape.intermediates[0].data()),
            static_cast<const __nv_bfloat16*>(tape.intermediates[1].data()),
            static_cast<const float*>(output_gradients[0].data()), rows,
            context_length_, num_heads_, embedding_dim_, scale,
            static_cast<float*>(qkv_gradient.data()));
  } else {
    FlashAttentionBackwardKernel<float><<<blocks, 1, 0, executor->stream()>>>(
        static_cast<const float*>(tape.intermediates[0].data()),
        static_cast<const float*>(tape.intermediates[1].data()),
        static_cast<const float*>(output_gradients[0].data()), rows,
        context_length_, num_heads_, embedding_dim_, scale,
        static_cast<float*>(qkv_gradient.data()));
  }
  RETURN_IF_ERROR(internal::CudaStatus(cudaGetLastError(),
                                       "FlashAttentionBackwardKernel launch"));
  return BufferVec{std::move(qkv_gradient)};
}

}  // namespace pluto::llm
