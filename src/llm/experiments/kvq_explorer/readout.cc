#include "src/llm/experiments/kvq_explorer/readout.h"

#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <string>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/util/status_macros.h"

namespace pluto::llm::kvq_explorer {
namespace {

constexpr int kProjectionColumns = 256;
constexpr int kVocabularyRows = 8;
constexpr int kEmbeddingColumns = 32;
constexpr int kChunk = 16;

// W is row-major [input_width, 3*width], with three contiguous output slices.
// A program owns 256 adjacent output columns for one input token. Keep the
// diagnostic's full FP32 precision and fixed accumulation order: converting
// operands to tensor-core formats would erase low-order checkpoint bits.
__tile_global__ void ProjectKernel(const float* embedding, const float* matrix,
                                   const float* bias, const int* tokens,
                                   int rows, int width, float* projected) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  const int row = ct::bid().y;
  const int output_tile = ct::bid().x;
  auto tokens_view = ct::partition_view{
      ct::tensor_span{tokens, ct::extents{rows}}, ct::shape{1_ic}};
  const int token = static_cast<int>(tokens_view.load(row));
  auto input_view = ct::partition_view{
      ct::tensor_span{embedding + static_cast<size_t>(token) * width,
                      ct::extents{width}},
      ct::shape{1_ic}};
  auto matrix_view =
      ct::partition_view{ct::tensor_span{matrix, ct::extents{width, 3 * width}},
                         ct::shape{1_ic, 256_ic}};
  auto bias_view = ct::partition_view{
      ct::tensor_span{bias, ct::extents{3 * width}}, ct::shape{256_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{projected, ct::extents{rows, 3 * width}},
      ct::shape{1_ic, 256_ic}};
  auto sum = ct::zeros<ct::tile<float, ct::shape<1, 256>>>();
  for (int input = 0; input < width; ++input)
    sum = ct::fma(input_view.load(input),
                  matrix_view.load_masked(input, output_tile), sum);
  output_view.store_masked(sum + bias_view.load_masked(output_tile), row,
                           output_tile);
}

// A program computes eight vocabulary logits with 32 FP32 partial sums each,
// then a deterministic reduction along the embedding dimension. All three
// projections use the same E. Masking uses the logical vocabulary, so padding
// embeddings cannot contribute to a logit or the softmax denominator.
__tile_global__ void UnembedKernel(const float* embedding,
                                   const float* projected, int rows, int width,
                                   int vocab_size, float* logits) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  const int row = ct::bid().y;  // Flattened [input token, Q/K/V].
  const int vocabulary_tile = ct::bid().x;
  auto embedding_view = ct::partition_view{
      ct::tensor_span{embedding, ct::extents{vocab_size, width}},
      ct::shape{8_ic, 32_ic}};
  auto projected_view = ct::partition_view{
      ct::tensor_span{projected + static_cast<size_t>(row) * width,
                      ct::extents{width}},
      ct::shape{32_ic}};
  auto logits_view =
      ct::partition_view{ct::tensor_span{logits, ct::extents{rows, vocab_size}},
                         ct::shape{1_ic, 8_ic}};
  auto sum = ct::zeros<ct::tile<float, ct::shape<8, 32>>>();
  const int column_tiles = (width + kEmbeddingColumns - 1) / kEmbeddingColumns;
  for (int column_tile = 0; column_tile < column_tiles; ++column_tile)
    sum = ct::fma(embedding_view.load_masked(vocabulary_tile, column_tile),
                  projected_view.load_masked(column_tile), sum);
  logits_view.store_masked(
      ct::reshape(ct::sum(sum, 1_ic), ct::shape{1_ic, 8_ic}), row,
      vocabulary_tile);
}

absl::StatusOr<size_t> FloatBytes(size_t rows, size_t columns) {
  if (columns > std::numeric_limits<size_t>::max() / sizeof(float) ||
      rows > std::numeric_limits<size_t>::max() / sizeof(float) / columns)
    return absl::InvalidArgumentError("tensor byte size overflows");
  return rows * columns * sizeof(float);
}

absl::StatusOr<cuda::Buffer> LoadWeight(cuda::Executor& executor,
                                        const std::filesystem::path& path,
                                        size_t bytes) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error))
    return absl::NotFoundError(absl::StrCat("missing weight: ", path.string()));
  const auto actual = std::filesystem::file_size(path, error);
  if (error || actual != bytes ||
      bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()))
    return absl::DataLossError(absl::StrCat(
        "wrong weight size: ", path.string(), "; expected ", bytes, " bytes"));
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<float>::Allocate(
                                  executor, bytes / sizeof(float)));
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(host.data()), bytes) ||
      input.peek() != std::ifstream::traits_type::eof())
    return absl::DataLossError(
        absl::StrCat("cannot read weight: ", path.string()));
  for (float value : host)
    if (!std::isfinite(value))
      return absl::DataLossError(
          absl::StrCat("nonfinite weight: ", path.string()));
  ASSIGN_OR_RETURN(auto device, cuda::Buffer::Allocate(executor, bytes));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(device.data(), host.data(), bytes, cudaMemcpyHostToDevice,
                      executor.stream()),
      "upload KVQ weight"));
  // Page-locked storage frees on executor after this upload, even on error.
  return device;
}

}  // namespace

absl::StatusOr<cuda::Buffer> ReadTopThree(cuda::Executor& executor,
                                          const cuda::Buffer& logits, int rows,
                                          int vocab_size) {
  if (rows <= 0 || vocab_size < 3)
    return absl::InvalidArgumentError(
        "top-three requires rows > 0 and vocab >= 3");
  ASSIGN_OR_RETURN(const size_t bytes, FloatBytes(rows, vocab_size));
  if (logits.size_bytes() != bytes || &logits.executor() != &executor)
    return absl::InvalidArgumentError("logits shape or executor mismatch");
  return ReadTopThreeTokens(executor, logits, rows, vocab_size, vocab_size);
}

absl::StatusOr<std::unique_ptr<Readout>> Readout::Load(
    cuda::Executor& executor, const std::filesystem::path& checkpoint,
    const Dimensions& dimensions) {
  const auto [vocab, width, blocks] = dimensions;
  if (checkpoint.empty() || vocab < 3 ||
      vocab > std::numeric_limits<int>::max() - 15 || width <= 0 ||
      width > (std::numeric_limits<int>::max() - kProjectionColumns) / 3 ||
      blocks <= 0 || blocks > kGpt2TransformerBlockCount)
    return absl::InvalidArgumentError(
        "invalid KVQ dimensions or empty checkpoint");
  const int padded_vocab = (vocab + 15) / 16 * 16;
  ASSIGN_OR_RETURN(const size_t embedding_bytes,
                   FloatBytes(padded_vocab, width));
  ASSIGN_OR_RETURN(const size_t matrix_bytes, FloatBytes(width, 3 * width));
  ASSIGN_OR_RETURN(const size_t bias_bytes, FloatBytes(1, 3 * width));
  ASSIGN_OR_RETURN(
      auto embedding,
      LoadWeight(executor, checkpoint / "weight_0.bin", embedding_bytes));
  std::vector<BlockWeights> weights;
  for (int block = 0; block < blocks; ++block) {
    ASSIGN_OR_RETURN(
        auto matrix,
        LoadWeight(executor,
                   checkpoint / absl::StrCat("weight_", 4 + 12 * block, ".bin"),
                   matrix_bytes));
    ASSIGN_OR_RETURN(
        auto bias,
        LoadWeight(executor,
                   checkpoint / absl::StrCat("weight_", 5 + 12 * block, ".bin"),
                   bias_bytes));
    weights.push_back({std::move(matrix), std::move(bias)});
  }
  return absl::WrapUnique(
      new Readout(dimensions, std::move(embedding), std::move(weights)));
}

absl::StatusOr<cuda::PageLockedHostArray<TopThree>> Readout::Explore(
    cuda::Executor& executor,
    const cuda::PageLockedHostArray<int>& tokens) const {
  if (&embedding_.executor() != &executor || &tokens.executor() != &executor)
    return absl::InvalidArgumentError(
        "KVQ input or weights use another executor");
  for (int token : tokens)
    if (token < 0 || token >= dimensions_.vocab_size)
      return absl::InvalidArgumentError(
          "input token outside logical vocabulary");
  const size_t stride = 3 * dimensions_.block_count;
  if (tokens.size() > std::numeric_limits<size_t>::max() / stride)
    return absl::InvalidArgumentError("KVQ result length overflows");
  ASSIGN_OR_RETURN(auto result, cuda::PageLockedHostArray<TopThree>::Allocate(
                                    executor, tokens.size() * stride));
  const int width = dimensions_.model_width;
  const int vocab = dimensions_.vocab_size;
  for (size_t start = 0; start < tokens.size(); start += kChunk) {
    const int rows = std::min<size_t>(kChunk, tokens.size() - start);
    ASSIGN_OR_RETURN(auto input,
                     cuda::Buffer::Allocate(executor, rows * sizeof(int)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(input.data(), tokens.data() + start, input.size_bytes(),
                        cudaMemcpyHostToDevice, executor.stream()),
        "upload KVQ tokens"));
    ASSIGN_OR_RETURN(const size_t projection_bytes, FloatBytes(rows, 3 * width));
    ASSIGN_OR_RETURN(const size_t logit_bytes, FloatBytes(3 * rows, vocab));
    ASSIGN_OR_RETURN(auto projected,
                     cuda::Buffer::Allocate(executor, projection_bytes));
    ASSIGN_OR_RETURN(auto logits, cuda::Buffer::Allocate(executor, logit_bytes));
    ASSIGN_OR_RETURN(
        auto compact,
        cuda::PageLockedHostArray<TopThree>::Allocate(executor, 3 * rows));
    for (int block = 0; block < dimensions_.block_count; ++block) {
      ProjectKernel<<<
          dim3((3 * width + kProjectionColumns - 1) / kProjectionColumns, rows),
          1, 0, executor.stream()>>>(
          static_cast<const float*>(embedding_.data()),
          static_cast<const float*>(blocks_[block].matrix.data()),
          static_cast<const float*>(blocks_[block].bias.data()),
          static_cast<const int*>(input.data()), rows, width,
          static_cast<float*>(projected.data()));
      RETURN_IF_ERROR(
          cuda::CudaStatus(cudaGetLastError(), "ProjectKernel launch"));
      UnembedKernel<<<dim3((vocab + kVocabularyRows - 1) / kVocabularyRows,
                           3 * rows),
                      1, 0, executor.stream()>>>(
          static_cast<const float*>(embedding_.data()),
          static_cast<const float*>(projected.data()), 3 * rows, width, vocab,
          static_cast<float*>(logits.data()));
      RETURN_IF_ERROR(
          cuda::CudaStatus(cudaGetLastError(), "UnembedKernel launch"));
      ASSIGN_OR_RETURN(auto top,
                       ReadTopThree(executor, logits, 3 * rows, vocab));
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(compact.data(), top.data(), compact.size_bytes(),
                          cudaMemcpyDeviceToHost, executor.stream()),
          "download KVQ top three"));
      RETURN_IF_ERROR(executor.Synchronize());
      for (int row = 0; row < rows; ++row)
        for (int projection = 0; projection < 3; ++projection) {
          const TopThree& entry = compact[3 * row + projection];
          if (entry.tokens[0] < 0)
            return absl::DataLossError(absl::StrCat(
                "nonfinite projection readout at token ", start + row,
                ", block ", block, ", QKV slice ", projection));
          result[(start + row) * stride + 3 * block + projection] = entry;
        }
    }
  }
  return result;
}

}  // namespace pluto::llm::kvq_explorer
