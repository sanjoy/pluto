#include "src/llm/qwen/embedding_algebra_table.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/llm/qwen/checkpoint.h"
#include "src/util/status_macros.h"

namespace pluto::llm::qwen {
namespace {
namespace ct = ::cuda::tiles;
using namespace ct::literals;

// Each tile owns eight rows. The 256-column loop also handles wide embeddings
// without assuming a power-of-two width or creating very large register tiles.
__tile_global__ void RowNorms(const __nv_bfloat16* weights, int rows, int width,
                              float* output) {
  auto w =
      ct::partition_view{ct::tensor_span{weights, ct::extents{rows, width}},
                         ct::shape{8_ic, 256_ic}};
  auto out = ct::partition_view{ct::tensor_span{output, ct::extents{rows}},
                                ct::shape{8_ic}};
  auto sum = ct::zeros<ct::tile<float, ct::shape<8, 256>>>();
  for (int c = 0; c < (width + 255) / 256; ++c) {
    auto x = ct::element_cast<float>(w.load_masked(ct::bid().x, c));
    sum = sum + x * x;
  }
  out.store_masked(ct::reshape(ct::sum(sum, 1_ic), ct::shape{8_ic}),
                   ct::bid().x);
}

__tile_global__ void EvaluateTerms(const __nv_bfloat16* weights,
                                   const int32_t* terms, int count, int rows,
                                   int width, float* output) {
  auto w =
      ct::partition_view{ct::tensor_span{weights, ct::extents{rows, width}},
                         ct::shape{1_ic, 256_ic}};
  auto ids = ct::partition_view{ct::tensor_span{terms, ct::extents{count}},
                                ct::shape{1_ic}};
  auto signs = ct::partition_view{
      ct::tensor_span{terms + count, ct::extents{count}}, ct::shape{1_ic}};
  auto out = ct::partition_view{ct::tensor_span{output, ct::extents{width}},
                                ct::shape{256_ic}};
  auto sum = ct::zeros<ct::tile<float, ct::shape<1, 256>>>();
  for (int t = 0; t < count; ++t) {
    const int id = static_cast<int>(ids.load(t));
    sum = sum + ct::element_cast<float>(w.load_masked(id, ct::bid().x)) *
                    ct::element_cast<float>(signs.load(t));
  }
  out.store_masked(ct::reshape(sum, ct::shape{256_ic}), ct::bid().x);
}

// Compute distances directly instead of norm(a)^2+norm(b)^2-2*dot(a,b):
// the latter loses precision near an exact self-match and can go negative.
__tile_global__ void RowScores(const __nv_bfloat16* weights, const float* query,
                               int rows, int width, float* output) {
  auto w =
      ct::partition_view{ct::tensor_span{weights, ct::extents{rows, width}},
                         ct::shape{8_ic, 256_ic}};
  auto q = ct::partition_view{ct::tensor_span{query, ct::extents{width}},
                              ct::shape{256_ic}};
  auto dots = ct::partition_view{ct::tensor_span{output, ct::extents{rows}},
                                 ct::shape{8_ic}};
  auto distances = ct::partition_view{
      ct::tensor_span{output + rows, ct::extents{rows}}, ct::shape{8_ic}};
  auto dot = ct::zeros<ct::tile<float, ct::shape<8, 256>>>();
  auto distance = ct::zeros<ct::tile<float, ct::shape<8, 256>>>();
  for (int c = 0; c < (width + 255) / 256; ++c) {
    auto x = ct::element_cast<float>(w.load_masked(ct::bid().x, c));
    auto y = ct::reshape(q.load_masked(c), ct::shape{1_ic, 256_ic});
    auto delta = x - y;
    dot = dot + x * y;
    distance = distance + delta * delta;
  }
  dots.store_masked(ct::reshape(ct::sum(dot, 1_ic), ct::shape{8_ic}),
                    ct::bid().x);
  distances.store_masked(ct::reshape(ct::sum(distance, 1_ic), ct::shape{8_ic}),
                         ct::bid().x);
}

absl::Status CheckDimensions(int rows, int dimensions, int searchable_count) {
  if (rows <= 0 || dimensions <= 0 || searchable_count <= 0 ||
      searchable_count > rows || rows > std::numeric_limits<int>::max() - 7 ||
      dimensions > std::numeric_limits<int>::max() - 255 ||
      static_cast<size_t>(rows) > std::numeric_limits<size_t>::max() /
                                      sizeof(__nv_bfloat16) /
                                      static_cast<size_t>(dimensions))
    return absl::InvalidArgumentError("invalid embedding table dimensions");
  return absl::OkStatus();
}

// Returned arrays are CPU-ready. The caller never observes a pending D2H copy.
absl::StatusOr<cuda::PageLockedHostArray<float>> ReadFloats(
    cuda::Executor& executor, const cuda::Buffer& device, size_t count) {
  ASSIGN_OR_RETURN(auto host,
                   cuda::PageLockedHostArray<float>::Allocate(executor, count));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), device.data(), host.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "download embedding algebra results"));
  RETURN_IF_ERROR(executor.Synchronize());
  return host;
}
}  // namespace

EmbeddingAlgebraTable::EmbeddingAlgebraTable(
    cuda::Executor& executor, cuda::Buffer weights, int rows, int dimensions,
    int searchable_token_count, cuda::PageLockedHostArray<float> squared_norms)
    : executor_(executor),
      weights_(std::move(weights)),
      rows_(rows),
      dimensions_(dimensions),
      searchable_token_count_(searchable_token_count),
      squared_norms_(std::move(squared_norms)) {}

absl::StatusOr<std::unique_ptr<EmbeddingAlgebraTable>>
EmbeddingAlgebraTable::Load(cuda::Executor& executor,
                            const std::filesystem::path& checkpoint,
                            int searchable_token_count) {
  ASSIGN_OR_RETURN(auto config, LoadConfig(checkpoint));
  RETURN_IF_ERROR(CheckDimensions(config.vocab_size, config.hidden_size,
                                  searchable_token_count));
  ASSIGN_OR_RETURN(auto source, SafetensorsCheckpoint::Open(checkpoint));
  ASSIGN_OR_RETURN(auto tensor,
                   source->Tensor("model.language_model.embed_tokens.weight"));
  if (tensor.dtype != TensorDType::kBF16 ||
      tensor.shape !=
          std::vector<int64_t>{config.vocab_size, config.hidden_size})
    return absl::InvalidArgumentError(
        "embed_tokens.weight must be BF16 with shape [vocab_size, "
        "hidden_size]");
  ASSIGN_OR_RETURN(auto device,
                   cuda::Buffer::Allocate(executor, tensor.bytes.size()));
  // Keep host staging bounded even for the 2.5 GB production embedding table.
  // Wait before reusing its bytes and before unmapping the source checkpoint.
  constexpr size_t kChunkBytes = 32 * 1024 * 1024;
  ASSIGN_OR_RETURN(auto staging,
                   cuda::PageLockedHostArray<uint8_t>::Allocate(
                       executor, std::min(kChunkBytes, tensor.bytes.size())));
  for (size_t offset = 0; offset < tensor.bytes.size();) {
    const size_t bytes = std::min(staging.size(), tensor.bytes.size() - offset);
    std::memcpy(staging.data(), tensor.bytes.data() + offset, bytes);
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(static_cast<uint8_t*>(device.data()) + offset,
                        staging.data(), bytes, cudaMemcpyHostToDevice,
                        executor.stream()),
        "upload checkpoint embedding table"));
    RETURN_IF_ERROR(executor.Synchronize());
    offset += bytes;
  }
  return Create(executor, std::move(device), config.vocab_size,
                config.hidden_size, searchable_token_count);
}

absl::StatusOr<std::unique_ptr<EmbeddingAlgebraTable>>
EmbeddingAlgebraTable::Create(cuda::Executor& executor, cuda::Buffer weights,
                              int rows, int dimensions,
                              int searchable_token_count) {
  RETURN_IF_ERROR(CheckDimensions(rows, dimensions, searchable_token_count));
  if (weights.data() == nullptr ||
      weights.size_bytes() !=
          static_cast<size_t>(rows) * dimensions * sizeof(__nv_bfloat16) ||
      &weights.executor() != &executor)
    return absl::InvalidArgumentError(
        "embedding buffer has the wrong size or belongs to another executor");
  ASSIGN_OR_RETURN(auto norms, cuda::Buffer::Allocate(
                                   executor, size_t(rows) * sizeof(float)));
  RowNorms<<<(rows + 7) / 8, 1, 0, executor.stream()>>>(
      static_cast<const __nv_bfloat16*>(weights.data()), rows, dimensions,
      static_cast<float*>(norms.data()));
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(), "embedding row norms"));
  ASSIGN_OR_RETURN(auto host_norms, ReadFloats(executor, norms, rows));
  for (int i = 0; i < rows; ++i)
    if (!std::isfinite(host_norms[i]))
      return absl::InvalidArgumentError(
          absl::StrCat("embedding row ", i,
                       " has nonfinite entries or an overflowing norm"));
  return absl::WrapUnique(
      new EmbeddingAlgebraTable(executor, std::move(weights), rows, dimensions,
                                searchable_token_count, std::move(host_norms)));
}

absl::StatusOr<cuda::PageLockedHostArray<float>>
EmbeddingAlgebraTable::Evaluate(absl::Span<const EmbeddingTerm> terms) const {
  if (terms.empty() || terms.size() > std::numeric_limits<int>::max() / 2)
    return absl::InvalidArgumentError(
        "embedding expression needs at least one term");
  for (const auto& term : terms)
    if (term.token_id < 0 || term.token_id >= searchable_token_count_ ||
        (term.coefficient != 1 && term.coefficient != -1))
      return absl::InvalidArgumentError(
          "invalid embedding expression token or sign");
  ASSIGN_OR_RETURN(
      auto host_terms,
      cuda::PageLockedHostArray<int32_t>::Allocate(executor_, 2 * terms.size()));
  for (size_t i = 0; i < terms.size(); ++i) {
    host_terms[i] = terms[i].token_id;
    host_terms[terms.size() + i] = terms[i].coefficient;
  }
  ASSIGN_OR_RETURN(auto device_terms,
                   cuda::Buffer::Allocate(executor_, host_terms.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(device_terms.data(), host_terms.data(),
                      host_terms.size_bytes(), cudaMemcpyHostToDevice,
                      executor_.stream()),
      "upload embedding expression"));
  ASSIGN_OR_RETURN(
      auto result,
      cuda::Buffer::Allocate(executor_, size_t(dimensions_) * sizeof(float)));
  EvaluateTerms<<<(dimensions_ + 255) / 256, 1, 0, executor_.stream()>>>(
      static_cast<const __nv_bfloat16*>(weights_.data()),
      static_cast<const int32_t*>(device_terms.data()),
      static_cast<int>(terms.size()), rows_, dimensions_,
      static_cast<float*>(result.data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "evaluate embedding expression"));
  ASSIGN_OR_RETURN(auto host_result, ReadFloats(executor_, result, dimensions_));
  for (float value : host_result)
    if (!std::isfinite(value))
      return absl::InvalidArgumentError(
          "embedding expression produced a nonfinite result");
  return host_result;
}

absl::StatusOr<std::vector<EmbeddingMatch>> EmbeddingAlgebraTable::Nearest(
    absl::Span<const float> query, int top_k) const {
  if (query.size() != static_cast<size_t>(dimensions_) || top_k <= 0)
    return absl::InvalidArgumentError(
        "invalid embedding query dimensions or top_k");
  float squared_norm = 0;
  for (float value : query) {
    if (!std::isfinite(value))
      return absl::InvalidArgumentError(
          "embedding query contains a nonfinite value");
    squared_norm += value * value;
  }
  if (!std::isfinite(squared_norm) || squared_norm <= 0)
    return absl::InvalidArgumentError(
        "cosine similarity requires a finite, nonzero query norm");
  ASSIGN_OR_RETURN(auto host_query,
                   cuda::PageLockedHostArray<float>::CopyFrom(executor_, query));
  ASSIGN_OR_RETURN(auto device_query,
                   cuda::Buffer::Allocate(executor_, host_query.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(device_query.data(), host_query.data(),
                      host_query.size_bytes(), cudaMemcpyHostToDevice,
                      executor_.stream()),
      "upload embedding query"));
  ASSIGN_OR_RETURN(
      auto device_scores,
      cuda::Buffer::Allocate(
          executor_, size_t(searchable_token_count_) * 2 * sizeof(float)));
  RowScores<<<(searchable_token_count_ + 7) / 8, 1, 0, executor_.stream()>>>(
      static_cast<const __nv_bfloat16*>(weights_.data()),
      static_cast<const float*>(device_query.data()), searchable_token_count_,
      dimensions_, static_cast<float*>(device_scores.data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "embedding similarity search"));
  ASSIGN_OR_RETURN(auto scores, ReadFloats(executor_, device_scores,
                                           size_t(searchable_token_count_) * 2));
  std::vector<EmbeddingMatch> matches;
  matches.reserve(searchable_token_count_);
  for (int id = 0; id < searchable_token_count_; ++id) {
    if (squared_norms_[id] == 0)
      continue;
    const float cosine =
        scores[id] / std::sqrt(squared_norm) / std::sqrt(squared_norms_[id]);
    const float distance = scores[searchable_token_count_ + id];
    if (!std::isfinite(cosine) || !std::isfinite(distance))
      return absl::InvalidArgumentError(
          "embedding similarity calculation overflowed");
    matches.push_back(
        {id, std::clamp(cosine, -1.0f, 1.0f), std::sqrt(distance)});
  }
  const size_t count = std::min(matches.size(), static_cast<size_t>(top_k));
  std::partial_sort(matches.begin(), matches.begin() + count, matches.end(),
                    [](const EmbeddingMatch& a, const EmbeddingMatch& b) {
                      if (a.cosine_similarity != b.cosine_similarity)
                        return a.cosine_similarity > b.cosine_similarity;
                      return a.token_id < b.token_id;
                    });
  matches.resize(count);
  return matches;
}

}  // namespace pluto::llm::qwen
