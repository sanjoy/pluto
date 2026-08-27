#include "src/llm/layers.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <algorithm>
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

namespace pluto::llm {
namespace {

constexpr int kMatrixElementCount = kModelWidth * kModelWidth;
constexpr int kDenseTile = 16;
constexpr int kDenseTilesPerAxis = kModelWidth / kDenseTile;

absl::Status CudaStatus(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return absl::OkStatus();
  return absl::InternalError(
      absl::StrCat(operation, " failed: ", cudaGetErrorName(error), ": ",
                   cudaGetErrorString(error)));
}

absl::Status ValidateFp16(DataType data_type) {
  if (data_type == DataType::FP16) return absl::OkStatus();
  return absl::UnimplementedError(
      "FP8 requires an explicit scaling policy; this cuTile backend currently "
      "implements FP16 compute with FP32 master weights");
}

absl::Status ValidateBuffer(const Buffer& buffer, size_t expected_bytes,
                            cudaStream_t stream, const char* name) {
  if (buffer.size_bytes() != expected_bytes) {
    return absl::InvalidArgumentError(absl::StrCat(
        name, " has ", buffer.size_bytes(), " bytes; expected ",
        expected_bytes));
  }
  if (buffer.stream() != stream) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " belongs to a different CUDA stream"));
  }
  return absl::OkStatus();
}

__tile_global__ void EmbeddingForwardKernel(
    const int* __restrict__ tokens, const float* __restrict__ table,
    float* __restrict__ output) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto token_view = ct::partition_view{
      ct::tensor_span{tokens, ct::extents{256_ic}}, ct::shape{1_ic}};
  auto table_view = ct::partition_view{
      ct::tensor_span{table, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 256_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 256_ic}};

  const int row = ct::bid().x;
  const int token = static_cast<int>(token_view.load(row));
  auto master = table_view.load(token, 0);
  auto fp16 = ct::element_cast<__half>(master);
  output_view.store(ct::element_cast<float>(fp16), row, 0);
}

__tile_global__ void EmbeddingBackwardKernel(
    const int* __restrict__ tokens, const float* __restrict__ output_gradient,
    float learning_rate, float* __restrict__ table) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto token_view = ct::partition_view{
      ct::tensor_span{tokens, ct::extents{256_ic}}, ct::shape{1_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 256_ic}};

  const int row = ct::bid().x;
  const int token = static_cast<int>(token_view.load(row));
  auto offsets = ct::iota<ct::tile<int, ct::shape<1, 256>>>();
  auto pointers = table + token * kModelWidth + offsets;
  ct::atomic_sub<ct::memory_order::relaxed>(
      pointers, gradient_view.load(row, 0) * learning_rate);
}

// The language-modeling head ties its projection matrix to the embedding
// table. Embeddings are laid out [vocabulary, model width], so the forward pass
// multiplies hidden states by the table's transpose.
__tile_global__ void LanguageModelingHeadForwardKernel(
    const float* __restrict__ input, const float* __restrict__ table,
    float* __restrict__ output) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto table_view = ct::partition_view{
      ct::tensor_span{table, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};

  const int block = ct::bid().x;
  const int batch_tile = block / kDenseTilesPerAxis;
  const int vocabulary_tile = block % kDenseTilesPerAxis;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  for (int dimension_tile = 0; dimension_tile < kDenseTilesPerAxis;
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
    const float* __restrict__ table, float* __restrict__ input_gradient) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto table_view = ct::partition_view{
      ct::tensor_span{table, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto input_gradient_view = ct::partition_view{
      ct::tensor_span{input_gradient, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};

  const int block = ct::bid().x;
  const int batch_tile = block / kDenseTilesPerAxis;
  const int dimension_tile = block % kDenseTilesPerAxis;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  for (int vocabulary_tile = 0; vocabulary_tile < kDenseTilesPerAxis;
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
    float* __restrict__ table) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto table_view = ct::partition_view{
      ct::tensor_span{table, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};

  const int block = ct::bid().x;
  const int vocabulary_tile = block / kDenseTilesPerAxis;
  const int dimension_tile = block % kDenseTilesPerAxis;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  for (int batch_tile = 0; batch_tile < kDenseTilesPerAxis; ++batch_tile) {
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
    float* __restrict__ output) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 256_ic}};
  auto position_view = ct::partition_view{
      ct::tensor_span{positions, ct::extents{16_ic, 256_ic}},
      ct::shape{1_ic, 256_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 256_ic}};

  const int row = ct::bid().x;
  output_view.store(input_view.load(row, 0) +
                        position_view.load(row % kContextLength, 0),
                    row, 0);
}

__tile_global__ void PositionEmbeddingBackwardKernel(
    const float* __restrict__ output_gradient, float learning_rate,
    float* __restrict__ positions) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 256_ic}};
  const int row = ct::bid().x;
  auto offsets = ct::iota<ct::tile<int, ct::shape<1, 256>>>();
  auto pointers = positions + (row % kContextLength) * kModelWidth + offsets;
  ct::atomic_sub<ct::memory_order::relaxed>(
      pointers, gradient_view.load(row, 0) * learning_rate);
}

// One program handles one query and one head. Keys and values are streamed
// through registers while an online maximum and normalizer keep the softmax
// stable. No score/probability matrix is written to global memory.
__tile_global__ void FlashAttentionForwardKernel(
    const float* __restrict__ input, float* __restrict__ output) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 64_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 64_ic}};

  const int block = ct::bid().x;
  const int row = block / kAttentionHeads;
  const int head = block % kAttentionHeads;
  const int sequence_start = (row / kContextLength) * kContextLength;
  const int query_position = row % kContextLength;
  constexpr float kScale = 0.125f;

  auto query = ct::element_cast<float>(
      ct::element_cast<__half>(input_view.load(row, head)));
  auto maximum =
      ct::full<ct::tile<float, ct::shape<1, 1>>>(-3.402823466e+38f);
  auto normalizer = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<1, 64>>>();

  for (int key_position = 0; key_position <= query_position; ++key_position) {
    const int key_row = sequence_start + key_position;
    auto key = ct::element_cast<float>(
        ct::element_cast<__half>(input_view.load(key_row, head)));
    auto value = key;
    auto score = ct::sum(query * key, 1_ic) * kScale;
    auto new_maximum = ct::max(maximum, score);
    auto old_scale = ct::exp(maximum - new_maximum);
    auto new_scale = ct::exp(score - new_maximum);
    accumulator = accumulator * old_scale + value * new_scale;
    normalizer = normalizer * old_scale + new_scale;
    maximum = new_maximum;
  }
  output_view.store(accumulator / normalizer, row, head);
}

// FlashAttention backward recomputes the causal softmax probabilities instead
// of loading a saved quadratic matrix. Q, K, and V alias the same activation,
// so their three contributions are atomically accumulated into one gradient.
__tile_global__ void FlashAttentionBackwardKernel(
    const float* __restrict__ input, const float* __restrict__ output,
    const float* __restrict__ output_gradient,
    float* __restrict__ input_gradient) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 64_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 64_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 64_ic}};

  const int block = ct::bid().x;
  const int row = block / kAttentionHeads;
  const int head = block % kAttentionHeads;
  const int sequence_start = (row / kContextLength) * kContextLength;
  const int query_position = row % kContextLength;
  constexpr float kScale = 0.125f;

  auto query = ct::element_cast<float>(
      ct::element_cast<__half>(input_view.load(row, head)));
  auto output_row = output_view.load(row, head);
  auto d_output = gradient_view.load(row, head);
  auto delta = ct::sum(d_output * output_row, 1_ic);
  auto maximum =
      ct::full<ct::tile<float, ct::shape<1, 1>>>(-3.402823466e+38f);
  auto normalizer = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();

  for (int key_position = 0; key_position <= query_position; ++key_position) {
    const int key_row = sequence_start + key_position;
    auto key = ct::element_cast<float>(
        ct::element_cast<__half>(input_view.load(key_row, head)));
    auto score = ct::sum(query * key, 1_ic) * kScale;
    auto new_maximum = ct::max(maximum, score);
    normalizer = normalizer * ct::exp(maximum - new_maximum) +
                 ct::exp(score - new_maximum);
    maximum = new_maximum;
  }

  auto offsets = ct::iota<ct::tile<int, ct::shape<1, 64>>>();
  auto query_gradient_pointers =
      input_gradient + row * kModelWidth + head * kAttentionHeadDimension +
      offsets;
  for (int key_position = 0; key_position <= query_position; ++key_position) {
    const int key_row = sequence_start + key_position;
    auto key = ct::element_cast<float>(
        ct::element_cast<__half>(input_view.load(key_row, head)));
    auto score = ct::sum(query * key, 1_ic) * kScale;
    auto probability = ct::exp(score - maximum) / normalizer;
    auto d_score =
        probability * (ct::sum(d_output * key, 1_ic) - delta);
    auto key_gradient_pointers =
        input_gradient +
        key_row * kModelWidth + head * kAttentionHeadDimension + offsets;
    ct::atomic_add<ct::memory_order::relaxed>(query_gradient_pointers,
                                               d_score * key * kScale);
    ct::atomic_add<ct::memory_order::relaxed>(key_gradient_pointers,
                                               d_score * query * kScale);
    ct::atomic_add<ct::memory_order::relaxed>(key_gradient_pointers,
                                               probability * d_output);
  }
}

__tile_global__ void LayerNormForwardKernel(
    const float* __restrict__ input, float epsilon,
    float* __restrict__ output) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 256_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 256_ic}};
  const int row = ct::bid().x;
  auto values = input_view.load(row, 0);
  auto mean = ct::sum(values, 1_ic) / static_cast<float>(kModelWidth);
  auto centered = values - mean;
  auto variance = ct::sum(centered * centered, 1_ic) /
                  static_cast<float>(kModelWidth);
  output_view.store(centered * ct::rsqrt(variance + epsilon), row, 0);
}

__tile_global__ void LayerNormBackwardKernel(
    const float* __restrict__ input,
    const float* __restrict__ output_gradient, float epsilon,
    float* __restrict__ input_gradient) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 256_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 256_ic}};
  auto input_gradient_view = ct::partition_view{
      ct::tensor_span{input_gradient, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 256_ic}};
  const int row = ct::bid().x;
  auto values = input_view.load(row, 0);
  auto d_output = gradient_view.load(row, 0);
  auto mean = ct::sum(values, 1_ic) / static_cast<float>(kModelWidth);
  auto centered = values - mean;
  auto inverse_stddev = ct::rsqrt(
      ct::sum(centered * centered, 1_ic) /
          static_cast<float>(kModelWidth) +
      epsilon);
  auto normalized = centered * inverse_stddev;
  auto gradient_sum = ct::sum(d_output, 1_ic);
  auto projected_sum = ct::sum(d_output * normalized, 1_ic);
  input_gradient_view.store(
      inverse_stddev *
          (d_output - gradient_sum / static_cast<float>(kModelWidth) -
           normalized * projected_sum / static_cast<float>(kModelWidth)),
      row, 0);
}

__tile_global__ void GeluForwardKernel(const float* __restrict__ input,
                                        float* __restrict__ output) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  const int block = ct::bid().x;
  const int row_tile = block / kDenseTilesPerAxis;
  const int column_tile = block % kDenseTilesPerAxis;
  auto x = input_view.load(row_tile, column_tile);
  constexpr float kSqrtTwoOverPi = 0.7978845608f;
  auto inner = kSqrtTwoOverPi * (x + 0.044715f * x * x * x);
  output_view.store(0.5f * x * (1.0f + ct::tanh(inner)), row_tile,
                    column_tile);
}

__tile_global__ void GeluBackwardKernel(
    const float* __restrict__ input,
    const float* __restrict__ output_gradient,
    float* __restrict__ input_gradient) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto input_gradient_view = ct::partition_view{
      ct::tensor_span{input_gradient, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  const int block = ct::bid().x;
  const int row_tile = block / kDenseTilesPerAxis;
  const int column_tile = block % kDenseTilesPerAxis;
  auto x = input_view.load(row_tile, column_tile);
  constexpr float kSqrtTwoOverPi = 0.7978845608f;
  auto inner = kSqrtTwoOverPi * (x + 0.044715f * x * x * x);
  auto tanh_inner = ct::tanh(inner);
  auto derivative =
      0.5f * (1.0f + tanh_inner) +
      0.5f * x * (1.0f - tanh_inner * tanh_inner) * kSqrtTwoOverPi *
          (1.0f + 3.0f * 0.044715f * x * x);
  input_gradient_view.store(
      gradient_view.load(row_tile, column_tile) * derivative, row_tile,
      column_tile);
}

__tile_global__ void AddKernel(const float* __restrict__ left,
                                const float* __restrict__ right,
                                float* __restrict__ output) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto left_view = ct::partition_view{
      ct::tensor_span{left, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto right_view = ct::partition_view{
      ct::tensor_span{right, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  const int block = ct::bid().x;
  const int row_tile = block / kDenseTilesPerAxis;
  const int column_tile = block % kDenseTilesPerAxis;
  output_view.store(left_view.load(row_tile, column_tile) +
                        right_view.load(row_tile, column_tile),
                    row_tile, column_tile);
}

__tile_global__ void CrossEntropyForwardKernel(
    const float* __restrict__ logits, const int* __restrict__ targets,
    float* __restrict__ losses) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto logits_view = ct::partition_view{
      ct::tensor_span{logits, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 256_ic}};
  auto target_view = ct::partition_view{
      ct::tensor_span{targets, ct::extents{256_ic}}, ct::shape{1_ic}};
  auto loss_view = ct::partition_view{
      ct::tensor_span{losses, ct::extents{256_ic}}, ct::shape{1_ic}};

  const int row = ct::bid().x;
  const int target = static_cast<int>(target_view.load(row));
  auto row_logits = logits_view.load(row, 0);
  auto maximum = ct::reduce_max(row_logits, 1_ic);
  auto exponentials = ct::exp(row_logits - maximum);
  auto denominator = ct::sum(exponentials, 1_ic);
  auto token_ids = ct::iota<ct::tile<int, ct::shape<1, 256>>>();
  auto one_hot = ct::element_cast<float>(token_ids == target);
  auto target_logit = ct::sum(row_logits * one_hot, 1_ic);
  auto loss = ct::log(denominator) + maximum - target_logit;
  loss_view.store(ct::reshape(loss, ct::shape{1_ic}), row);
}

__tile_global__ void CrossEntropyBackwardKernel(
    const float* __restrict__ logits, const int* __restrict__ targets,
    float* __restrict__ logits_gradient) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto logits_view = ct::partition_view{
      ct::tensor_span{logits, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 256_ic}};
  auto target_view = ct::partition_view{
      ct::tensor_span{targets, ct::extents{256_ic}}, ct::shape{1_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{logits_gradient, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 256_ic}};

  const int row = ct::bid().x;
  const int target = static_cast<int>(target_view.load(row));
  auto row_logits = logits_view.load(row, 0);
  auto maximum = ct::reduce_max(row_logits, 1_ic);
  auto exponentials = ct::exp(row_logits - maximum);
  auto denominator = ct::sum(exponentials, 1_ic);
  auto token_ids = ct::iota<ct::tile<int, ct::shape<1, 256>>>();
  auto one_hot = ct::element_cast<float>(token_ids == target);
  gradient_view.store(
      (exponentials / denominator - one_hot) /
          static_cast<float>(kBatchSize),
      row, 0);
}

__tile_global__ void DenseForwardKernel(
    const float* __restrict__ input, const float* __restrict__ matrix,
    const float* __restrict__ bias, float* __restrict__ output) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto matrix_view = ct::partition_view{
      ct::tensor_span{matrix, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto bias_view = ct::partition_view{
      ct::tensor_span{bias, ct::extents{256_ic}}, ct::shape{16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};

  const int block = ct::bid().x;
  const int batch_tile = block / kDenseTilesPerAxis;
  const int output_tile = block % kDenseTilesPerAxis;
  auto accumulator = ct::broadcast(bias_view.load(output_tile),
                                   ct::shape{16_ic, 16_ic});
  for (int inner_tile = 0; inner_tile < kDenseTilesPerAxis; ++inner_tile) {
    auto left =
        ct::element_cast<__half>(input_view.load(batch_tile, inner_tile));
    auto right =
        ct::element_cast<__half>(matrix_view.load(inner_tile, output_tile));
    accumulator = ct::mma(left, right, accumulator);
  }
  output_view.store(accumulator, batch_tile, output_tile);
}

__tile_global__ void DenseInputGradientKernel(
    const float* __restrict__ output_gradient,
    const float* __restrict__ matrix, float* __restrict__ input_gradient) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto matrix_view = ct::partition_view{
      ct::tensor_span{matrix, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto input_gradient_view = ct::partition_view{
      ct::tensor_span{input_gradient, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};

  const int block = ct::bid().x;
  const int batch_tile = block / kDenseTilesPerAxis;
  const int input_tile = block % kDenseTilesPerAxis;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  for (int output_tile = 0; output_tile < kDenseTilesPerAxis;
       ++output_tile) {
    auto gradient = ct::element_cast<__half>(
        gradient_view.load(batch_tile, output_tile));
    auto matrix_transposed = ct::transpose(ct::element_cast<__half>(
        matrix_view.load(input_tile, output_tile)));
    accumulator = ct::mma(gradient, matrix_transposed, accumulator);
  }
  input_gradient_view.store(accumulator, batch_tile, input_tile);
}

__tile_global__ void DenseWeightUpdateKernel(
    const float* __restrict__ input,
    const float* __restrict__ output_gradient, float learning_rate,
    float* __restrict__ matrix) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto matrix_view = ct::partition_view{
      ct::tensor_span{matrix, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};

  const int block = ct::bid().x;
  const int input_tile = block / kDenseTilesPerAxis;
  const int output_tile = block % kDenseTilesPerAxis;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  for (int batch_tile = 0; batch_tile < kDenseTilesPerAxis; ++batch_tile) {
    auto input_transposed = ct::transpose(ct::element_cast<__half>(
        input_view.load(batch_tile, input_tile)));
    auto gradient = ct::element_cast<__half>(
        gradient_view.load(batch_tile, output_tile));
    accumulator = ct::mma(input_transposed, gradient, accumulator);
  }
  auto old_matrix = matrix_view.load(input_tile, output_tile);
  matrix_view.store(old_matrix - learning_rate * accumulator, input_tile,
                    output_tile);
}

__tile_global__ void DenseBiasUpdateKernel(
    const float* __restrict__ output_gradient, float learning_rate,
    float* __restrict__ bias) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto bias_view = ct::partition_view{
      ct::tensor_span{bias, ct::extents{256_ic}}, ct::shape{16_ic}};

  const int output_tile = ct::bid().x;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<1, 16>>>();
  for (int batch_tile = 0; batch_tile < kDenseTilesPerAxis; ++batch_tile) {
    accumulator = accumulator +
                  ct::sum(gradient_view.load(batch_tile, output_tile), 0_ic);
  }
  auto old_bias = bias_view.load(output_tile);
  bias_view.store(old_bias - learning_rate *
                                 ct::reshape(accumulator, ct::shape{16_ic}),
                  output_tile);
}

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
  if (vocab_size <= 0 || embedding_dim <= 0) {
    return absl::InvalidArgumentError(
        "vocab_size and embedding_dim must be positive");
  }
  if (vocab_size != kVocabularySize || embedding_dim != kModelWidth) {
    return absl::UnimplementedError(absl::StrCat(
        "the cuTile backend currently supports vocab_size=", kVocabularySize,
        " and embedding_dim=", kModelWidth));
  }
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
  if (auto status = ValidateBuffer(inputs[0], kBatchSize * sizeof(int),
                                   stream_, "embedding token input");
      !status.ok()) {
    return status;
  }
  auto output = Buffer::Allocate(
      kBatchSize * embedding_dim_ * sizeof(float), stream_);
  if (!output.ok()) return output.status();
  tape->intermediates = {inputs[0]};
  tape->children.clear();
  EmbeddingForwardKernel<<<kBatchSize, 1, 0, stream_>>>(
      static_cast<const int*>(inputs[0].data()),
      static_cast<const float*>(weight_.data()),
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
  if (auto status =
          ValidateBuffer(output_gradients[0],
                         kBatchSize * embedding_dim_ * sizeof(float), stream_,
                         "embedding output gradient");
      !status.ok()) {
    return status;
  }
  EmbeddingBackwardKernel<<<kBatchSize, 1, 0, stream_>>>(
      static_cast<const int*>(tape.intermediates[0].data()),
      static_cast<const float*>(output_gradients[0].data()), learning_rate_,
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
  if (auto status = ValidateBuffer(
          inputs[0], kBatchSize * embedding_->embedding_dim_ * sizeof(float),
          embedding_->stream_, "language-modeling-head input");
      !status.ok()) {
    return status;
  }
  auto output = Buffer::Allocate(
      kBatchSize * embedding_->vocab_size_ * sizeof(float),
      embedding_->stream_);
  if (!output.ok()) return output.status();
  tape->intermediates = {inputs[0]};
  tape->children.clear();
  LanguageModelingHeadForwardKernel<<<
      kDenseTilesPerAxis * kDenseTilesPerAxis, 1, 0, embedding_->stream_>>>(
      static_cast<const float*>(inputs[0].data()),
      static_cast<const float*>(embedding_->weight_.data()),
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
  if (auto status = ValidateBuffer(
          output_gradients[0],
          kBatchSize * embedding_->vocab_size_ * sizeof(float),
          embedding_->stream_, "language-modeling-head output gradient");
      !status.ok()) {
    return status;
  }
  auto input_gradient = Buffer::Allocate(
      kBatchSize * embedding_->embedding_dim_ * sizeof(float),
      embedding_->stream_);
  if (!input_gradient.ok()) return input_gradient.status();
  LanguageModelingHeadInputGradientKernel<<<
      kDenseTilesPerAxis * kDenseTilesPerAxis, 1, 0, embedding_->stream_>>>(
      static_cast<const float*>(output_gradients[0].data()),
      static_cast<const float*>(embedding_->weight_.data()),
      static_cast<float*>(input_gradient->data()));
  if (embedding_->learning_rate_ != 0.0f) {
    LanguageModelingHeadWeightUpdateKernel<<<
        kDenseTilesPerAxis * kDenseTilesPerAxis, 1, 0, embedding_->stream_>>>(
        static_cast<const float*>(tape.intermediates[0].data()),
        static_cast<const float*>(output_gradients[0].data()),
        embedding_->learning_rate_,
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
  if (context_length != kContextLength || embedding_dim != kModelWidth) {
    return absl::UnimplementedError(absl::StrCat(
        "the cuTile position backend currently supports context_length=",
        kContextLength, " and embedding_dim=", kModelWidth));
  }
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
  if (auto status = ValidateBuffer(
          inputs[0], kBatchSize * embedding_dim_ * sizeof(float), stream_,
          "position-embedding input");
      !status.ok()) {
    return status;
  }
  auto output = Buffer::Allocate(
      kBatchSize * embedding_dim_ * sizeof(float), stream_);
  if (!output.ok()) return output.status();
  tape->intermediates.clear();
  tape->children.clear();
  PositionEmbeddingForwardKernel<<<kBatchSize, 1, 0, stream_>>>(
      static_cast<const float*>(inputs[0].data()),
      static_cast<const float*>(weight_.data()),
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
  if (auto status = ValidateBuffer(
          output_gradients[0],
          kBatchSize * embedding_dim_ * sizeof(float), stream_,
          "position-embedding output gradient");
      !status.ok()) {
    return status;
  }
  PositionEmbeddingBackwardKernel<<<kBatchSize, 1, 0, stream_>>>(
      static_cast<const float*>(output_gradients[0].data()), learning_rate_,
      static_cast<float*>(weight_.data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "PositionEmbeddingBackwardKernel launch");
      !status.ok()) {
    return status;
  }
  // The additive path has derivative one and can share the upstream buffer.
  return BufferVec{output_gradients[0]};
}

absl::StatusOr<std::unique_ptr<AttentionLayer>> AttentionLayer::Create(
    int context_length, int num_heads, int embedding_dim, DataType data_type,
    cudaStream_t stream) {
  if (auto status = ValidateFp16(data_type); !status.ok()) return status;
  if (context_length != kContextLength || num_heads != kAttentionHeads ||
      embedding_dim != kModelWidth) {
    return absl::UnimplementedError(absl::StrCat(
        "the FlashAttention specialization requires context_length=",
        kContextLength, ", num_heads=", kAttentionHeads,
        ", and embedding_dim=", kModelWidth));
  }
  return std::unique_ptr<AttentionLayer>(new AttentionLayer(
      context_length, num_heads, embedding_dim, data_type, stream));
}

absl::StatusOr<Buffer> AttentionLayer::fwd(
    absl::Span<const Buffer> inputs, Tape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "AttentionLayer fwd expects one input and a non-null tape");
  }
  if (auto status = ValidateBuffer(
          inputs[0], kBatchSize * embedding_dim_ * sizeof(float), stream_,
          "attention input");
      !status.ok()) {
    return status;
  }
  auto output = Buffer::Allocate(
      kBatchSize * embedding_dim_ * sizeof(float), stream_);
  if (!output.ok()) return output.status();
  FlashAttentionForwardKernel<<<kBatchSize * num_heads_, 1, 0, stream_>>>(
      static_cast<const float*>(inputs[0].data()),
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
  const size_t activation_bytes =
      static_cast<size_t>(kBatchSize) * embedding_dim_ * sizeof(float);
  if (auto status = ValidateBuffer(output_gradients[0], activation_bytes,
                                   stream_, "attention output gradient");
      !status.ok()) {
    return status;
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
  FlashAttentionBackwardKernel<<<kBatchSize * num_heads_, 1, 0, stream_>>>(
      static_cast<const float*>(tape.intermediates[0].data()),
      static_cast<const float*>(tape.intermediates[1].data()),
      static_cast<const float*>(output_gradients[0].data()),
      static_cast<float*>(input_gradient->data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "FlashAttentionBackwardKernel launch");
      !status.ok()) {
    return status;
  }
  return BufferVec{*std::move(input_gradient)};
}

absl::StatusOr<std::unique_ptr<LayerNormLayer>> LayerNormLayer::Create(
    int embedding_dim, float epsilon, DataType data_type,
    cudaStream_t stream) {
  if (auto status = ValidateFp16(data_type); !status.ok()) return status;
  if (epsilon <= 0.0f) {
    return absl::InvalidArgumentError("layer-norm epsilon must be positive");
  }
  if (embedding_dim != kModelWidth) {
    return absl::UnimplementedError(absl::StrCat(
        "the cuTile layer-norm backend currently supports embedding_dim=",
        kModelWidth));
  }
  return std::unique_ptr<LayerNormLayer>(
      new LayerNormLayer(embedding_dim, epsilon, data_type, stream));
}

absl::StatusOr<Buffer> LayerNormLayer::fwd(
    absl::Span<const Buffer> inputs, Tape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "LayerNormLayer fwd expects one input and a non-null tape");
  }
  const size_t activation_bytes =
      static_cast<size_t>(kBatchSize) * embedding_dim_ * sizeof(float);
  if (auto status = ValidateBuffer(inputs[0], activation_bytes, stream_,
                                   "layer-norm input");
      !status.ok()) {
    return status;
  }
  auto output = Buffer::Allocate(activation_bytes, stream_);
  if (!output.ok()) return output.status();
  tape->intermediates = {inputs[0]};
  tape->children.clear();
  LayerNormForwardKernel<<<kBatchSize, 1, 0, stream_>>>(
      static_cast<const float*>(inputs[0].data()), epsilon_,
      static_cast<float*>(output->data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "LayerNormForwardKernel launch");
      !status.ok()) {
    return status;
  }
  return *std::move(output);
}

absl::StatusOr<BufferVec> LayerNormLayer::bwd(
    absl::Span<const Buffer> output_gradients, Tape tape) {
  if (output_gradients.size() != 1 || tape.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "LayerNormLayer bwd received an incompatible gradient or tape");
  }
  const size_t activation_bytes =
      static_cast<size_t>(kBatchSize) * embedding_dim_ * sizeof(float);
  if (auto status = ValidateBuffer(output_gradients[0], activation_bytes,
                                   stream_, "layer-norm output gradient");
      !status.ok()) {
    return status;
  }
  auto input_gradient = Buffer::Allocate(activation_bytes, stream_);
  if (!input_gradient.ok()) return input_gradient.status();
  LayerNormBackwardKernel<<<kBatchSize, 1, 0, stream_>>>(
      static_cast<const float*>(tape.intermediates[0].data()),
      static_cast<const float*>(output_gradients[0].data()), epsilon_,
      static_cast<float*>(input_gradient->data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "LayerNormBackwardKernel launch");
      !status.ok()) {
    return status;
  }
  return BufferVec{*std::move(input_gradient)};
}

absl::StatusOr<std::unique_ptr<GeluLayer>> GeluLayer::Create(
    DataType data_type, cudaStream_t stream) {
  if (auto status = ValidateFp16(data_type); !status.ok()) return status;
  return std::unique_ptr<GeluLayer>(new GeluLayer(data_type, stream));
}

absl::StatusOr<Buffer> GeluLayer::fwd(absl::Span<const Buffer> inputs,
                                       Tape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "GeluLayer fwd expects one input and a non-null tape");
  }
  constexpr size_t kActivationBytes =
      static_cast<size_t>(kBatchSize) * kModelWidth * sizeof(float);
  if (auto status = ValidateBuffer(inputs[0], kActivationBytes, stream_,
                                   "GELU input");
      !status.ok()) {
    return status;
  }
  auto output = Buffer::Allocate(kActivationBytes, stream_);
  if (!output.ok()) return output.status();
  tape->intermediates = {inputs[0]};
  tape->children.clear();
  GeluForwardKernel<<<kDenseTilesPerAxis * kDenseTilesPerAxis, 1, 0,
                      stream_>>>(static_cast<const float*>(inputs[0].data()),
                                 static_cast<float*>(output->data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "GeluForwardKernel launch");
      !status.ok()) {
    return status;
  }
  return *std::move(output);
}

absl::StatusOr<BufferVec> GeluLayer::bwd(
    absl::Span<const Buffer> output_gradients, Tape tape) {
  if (output_gradients.size() != 1 || tape.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "GeluLayer bwd received an incompatible gradient or tape");
  }
  constexpr size_t kActivationBytes =
      static_cast<size_t>(kBatchSize) * kModelWidth * sizeof(float);
  if (auto status = ValidateBuffer(output_gradients[0], kActivationBytes,
                                   stream_, "GELU output gradient");
      !status.ok()) {
    return status;
  }
  auto input_gradient = Buffer::Allocate(kActivationBytes, stream_);
  if (!input_gradient.ok()) return input_gradient.status();
  GeluBackwardKernel<<<kDenseTilesPerAxis * kDenseTilesPerAxis, 1, 0,
                       stream_>>>(
      static_cast<const float*>(tape.intermediates[0].data()),
      static_cast<const float*>(output_gradients[0].data()),
      static_cast<float*>(input_gradient->data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "GeluBackwardKernel launch");
      !status.ok()) {
    return status;
  }
  return BufferVec{*std::move(input_gradient)};
}

ResidualLayer::ResidualLayer(std::unique_ptr<Layer> layer)
    : layer_(std::move(layer)) {
  for (Buffer& weight : layer_->weights()) weights_.push_back(weight);
}

absl::StatusOr<Buffer> ResidualLayer::fwd(
    absl::Span<const Buffer> inputs, Tape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "ResidualLayer fwd expects one input and a non-null tape");
  }
  Tape child_tape;
  auto branch = layer_->fwd(inputs, &child_tape);
  if (!branch.ok()) return branch.status();
  if (branch->size_bytes() != inputs[0].size_bytes() ||
      branch->stream() != inputs[0].stream()) {
    return absl::InvalidArgumentError(
        "ResidualLayer branch changed the activation shape or stream");
  }
  auto output = Buffer::Allocate(inputs[0].size_bytes(), inputs[0].stream());
  if (!output.ok()) return output.status();
  tape->intermediates = {inputs[0]};
  tape->children = {std::move(child_tape)};
  AddKernel<<<kDenseTilesPerAxis * kDenseTilesPerAxis, 1, 0,
              inputs[0].stream()>>>(
      static_cast<const float*>(inputs[0].data()),
      static_cast<const float*>(branch->data()),
      static_cast<float*>(output->data()));
  if (auto status = CudaStatus(cudaGetLastError(), "AddKernel(residual) launch");
      !status.ok()) {
    return status;
  }
  return *std::move(output);
}

absl::StatusOr<BufferVec> ResidualLayer::bwd(
    absl::Span<const Buffer> output_gradients, Tape tape) {
  if (output_gradients.size() != 1 || tape.intermediates.size() != 1 ||
      tape.children.size() != 1) {
    return absl::InvalidArgumentError(
        "ResidualLayer bwd received an incompatible gradient or tape");
  }
  auto branch_gradient =
      layer_->bwd(output_gradients, std::move(tape.children[0]));
  if (!branch_gradient.ok()) return branch_gradient.status();
  if (branch_gradient->size() != 1 ||
      branch_gradient->front().size_bytes() !=
          output_gradients[0].size_bytes()) {
    return absl::InvalidArgumentError(
        "ResidualLayer branch returned an incompatible input gradient");
  }
  auto input_gradient = Buffer::Allocate(output_gradients[0].size_bytes(),
                                         output_gradients[0].stream());
  if (!input_gradient.ok()) return input_gradient.status();
  AddKernel<<<kDenseTilesPerAxis * kDenseTilesPerAxis, 1, 0,
              output_gradients[0].stream()>>>(
      static_cast<const float*>(output_gradients[0].data()),
      static_cast<const float*>(branch_gradient->front().data()),
      static_cast<float*>(input_gradient->data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "AddKernel(residual gradient) launch");
      !status.ok()) {
    return status;
  }
  return BufferVec{*std::move(input_gradient)};
}

FullyConnectedLayer::FullyConnectedLayer(DataType data_type,
                                         float learning_rate,
                                         cudaStream_t stream, Buffer matrix,
                                         Buffer bias)
    : output_type_(data_type),
      learning_rate_(learning_rate),
      stream_(stream),
      weights_{std::move(matrix), std::move(bias)} {}

absl::StatusOr<std::unique_ptr<FullyConnectedLayer>>
FullyConnectedLayer::Create(DataType data_type, float learning_rate,
                            cudaStream_t stream) {
  if (auto status = ValidateFp16(data_type); !status.ok()) return status;
  if (learning_rate < 0.0f) {
    return absl::InvalidArgumentError("learning rate must be non-negative");
  }
  auto matrix = Buffer::Allocate(kMatrixElementCount * sizeof(float), stream);
  if (!matrix.ok()) return matrix.status();
  auto bias = Buffer::Allocate(kModelWidth * sizeof(float), stream);
  if (!bias.ok()) return bias.status();
  if (auto status = CudaStatus(cudaMemsetAsync(matrix->data(), 0,
                                               matrix->size_bytes(), stream),
                               "cudaMemsetAsync(dense matrix)");
      !status.ok()) {
    return status;
  }
  if (auto status = CudaStatus(
          cudaMemsetAsync(bias->data(), 0, bias->size_bytes(), stream),
          "cudaMemsetAsync(dense bias)");
      !status.ok()) {
    return status;
  }
  return std::unique_ptr<FullyConnectedLayer>(new FullyConnectedLayer(
      data_type, learning_rate, stream, *std::move(matrix), *std::move(bias)));
}

absl::Status FullyConnectedLayer::InitializeIdentity(float scale) {
  std::vector<float> identity(kMatrixElementCount, 0.0f);
  for (int index = 0; index < kModelWidth; ++index) {
    identity[index * kModelWidth + index] = scale;
  }
  return CudaStatus(cudaMemcpyAsync(weights_[0].data(), identity.data(),
                                    weights_[0].size_bytes(),
                                    cudaMemcpyHostToDevice, stream_),
                    "cudaMemcpyAsync(identity matrix)");
}

absl::StatusOr<Buffer> FullyConnectedLayer::fwd(
    absl::Span<const Buffer> inputs, Tape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "FullyConnectedLayer fwd expects one input and a non-null tape");
  }
  if (auto status =
          ValidateBuffer(inputs[0],
                         kBatchSize * kModelWidth * sizeof(float), stream_,
                         "dense input");
      !status.ok()) {
    return status;
  }
  auto output =
      Buffer::Allocate(kBatchSize * kModelWidth * sizeof(float), stream_);
  if (!output.ok()) return output.status();
  tape->intermediates = {inputs[0]};
  tape->children.clear();
  DenseForwardKernel<<<kDenseTilesPerAxis * kDenseTilesPerAxis, 1, 0,
                       stream_>>>(
      static_cast<const float*>(inputs[0].data()),
      static_cast<const float*>(weights_[0].data()),
      static_cast<const float*>(weights_[1].data()),
      static_cast<float*>(output->data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "DenseForwardKernel launch");
      !status.ok()) {
    return status;
  }
  return *std::move(output);
}

absl::StatusOr<BufferVec> FullyConnectedLayer::bwd(
    absl::Span<const Buffer> output_gradients, Tape tape) {
  if (output_gradients.size() != 1 || tape.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "FullyConnectedLayer bwd received an incompatible gradient or tape");
  }
  if (auto status =
          ValidateBuffer(output_gradients[0],
                         kBatchSize * kModelWidth * sizeof(float), stream_,
                         "dense output gradient");
      !status.ok()) {
    return status;
  }
  auto input_gradient =
      Buffer::Allocate(kBatchSize * kModelWidth * sizeof(float), stream_);
  if (!input_gradient.ok()) return input_gradient.status();
  DenseInputGradientKernel<<<kDenseTilesPerAxis * kDenseTilesPerAxis, 1, 0,
                             stream_>>>(
      static_cast<const float*>(output_gradients[0].data()),
      static_cast<const float*>(weights_[0].data()),
      static_cast<float*>(input_gradient->data()));
  if (learning_rate_ != 0.0f) {
    DenseWeightUpdateKernel<<<kDenseTilesPerAxis * kDenseTilesPerAxis, 1, 0,
                              stream_>>>(
        static_cast<const float*>(tape.intermediates[0].data()),
        static_cast<const float*>(output_gradients[0].data()), learning_rate_,
        static_cast<float*>(weights_[0].data()));
    DenseBiasUpdateKernel<<<kDenseTilesPerAxis, 1, 0, stream_>>>(
        static_cast<const float*>(output_gradients[0].data()), learning_rate_,
        static_cast<float*>(weights_[1].data()));
  }
  if (auto status = CudaStatus(cudaGetLastError(),
                               "dense backward kernel launch");
      !status.ok()) {
    return status;
  }
  return BufferVec{*std::move(input_gradient)};
}

absl::StatusOr<std::unique_ptr<CrossEntropyLossLayer>>
CrossEntropyLossLayer::Create(DataType data_type, cudaStream_t stream) {
  if (auto status = ValidateFp16(data_type); !status.ok()) return status;
  return std::unique_ptr<CrossEntropyLossLayer>(
      new CrossEntropyLossLayer(data_type, stream));
}

absl::StatusOr<Buffer> CrossEntropyLossLayer::fwd(
    absl::Span<const Buffer> inputs, Tape* tape) {
  if (inputs.size() != 2 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "CrossEntropyLossLayer fwd expects logits, targets, and a non-null "
        "tape");
  }
  if (auto status =
          ValidateBuffer(inputs[0],
                         kBatchSize * kVocabularySize * sizeof(float), stream_,
                         "cross-entropy logits");
      !status.ok()) {
    return status;
  }
  if (auto status = ValidateBuffer(inputs[1], kBatchSize * sizeof(int),
                                   stream_, "cross-entropy targets");
      !status.ok()) {
    return status;
  }
  auto losses = Buffer::Allocate(kBatchSize * sizeof(float), stream_);
  if (!losses.ok()) return losses.status();
  tape->intermediates = {inputs[0], inputs[1]};
  tape->children.clear();
  CrossEntropyForwardKernel<<<kBatchSize, 1, 0, stream_>>>(
      static_cast<const float*>(inputs[0].data()),
      static_cast<const int*>(inputs[1].data()),
      static_cast<float*>(losses->data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "CrossEntropyForwardKernel launch");
      !status.ok()) {
    return status;
  }
  return *std::move(losses);
}

absl::StatusOr<BufferVec> CrossEntropyLossLayer::bwd(
    absl::Span<const Buffer> output_gradients, Tape tape) {
  if (!output_gradients.empty() || tape.intermediates.size() != 2) {
    return absl::InvalidArgumentError(
        "terminal CrossEntropyLossLayer bwd expects no upstream gradient and "
        "a matching tape");
  }
  auto logits_gradient = Buffer::Allocate(
      kBatchSize * kVocabularySize * sizeof(float), stream_);
  if (!logits_gradient.ok()) return logits_gradient.status();
  CrossEntropyBackwardKernel<<<kBatchSize, 1, 0, stream_>>>(
      static_cast<const float*>(tape.intermediates[0].data()),
      static_cast<const int*>(tape.intermediates[1].data()),
      static_cast<float*>(logits_gradient->data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "CrossEntropyBackwardKernel launch");
      !status.ok()) {
    return status;
  }
  return BufferVec{*std::move(logits_gradient)};
}

ComposedLayer::ComposedLayer(DataType data_type,
                             std::vector<std::unique_ptr<Layer>> layers)
    : output_type_(data_type), layers_(std::move(layers)) {
  for (const auto& layer : layers_) {
    for (Buffer& weight : layer->weights()) weights_.push_back(weight);
  }
}

absl::StatusOr<Buffer> ComposedLayer::fwd(absl::Span<const Buffer> inputs,
                                           Tape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "ComposedLayer fwd expects one input and a non-null tape");
  }
  tape->intermediates.clear();
  tape->children.clear();
  Buffer activation = inputs.front();
  for (auto& layer : layers_) {
    Tape child_tape;
    BufferVec child_inputs = {activation};
    auto output = layer->fwd(child_inputs, &child_tape);
    if (!output.ok()) return output.status();
    activation = *std::move(output);
    tape->children.push_back(std::move(child_tape));
  }
  return activation;
}

absl::StatusOr<BufferVec> ComposedLayer::bwd(
    absl::Span<const Buffer> output_gradients, Tape tape) {
  if (output_gradients.size() != 1 || tape.children.size() != layers_.size()) {
    return absl::InvalidArgumentError(
        "ComposedLayer bwd received an incompatible gradient or tape");
  }
  Buffer gradient = output_gradients.front();
  for (size_t index = layers_.size(); index-- > 0;) {
    BufferVec child_gradients = {gradient};
    auto input_gradients = layers_[index]->bwd(
        child_gradients, std::move(tape.children[index]));
    if (!input_gradients.ok()) return input_gradients.status();
    if (index == 0 && input_gradients->empty()) return BufferVec{};
    if (input_gradients->size() != 1) {
      return absl::InternalError(
          "a composed unary layer returned multiple input gradients");
    }
    gradient = input_gradients->front();
  }
  return BufferVec{gradient};
}

}  // namespace pluto::llm
