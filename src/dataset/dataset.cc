#include "src/dataset/dataset.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "src/common/status_macros.h"
#include "src/cuda/buffer.h"

namespace pluto {
namespace {

static_assert(sizeof(int) == sizeof(int32_t));

absl::Status CudaStatus(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return absl::OkStatus();
  return absl::InternalError(absl::StrCat(operation,
                                          " failed: ", cudaGetErrorName(error),
                                          ": ", cudaGetErrorString(error)));
}

}  // namespace

InMemoryDataSetIterator::InMemoryDataSetIterator(std::vector<int> corpus,
                                                 InMemoryDataSetOptions options,
                                                 cudaStream_t stream,
                                                 cuda::Buffer token_buffer,
                                                 cuda::Buffer target_buffer)
    : corpus_(std::move(corpus)),
      options_(options),
      stream_(stream),
      token_buffer_(std::move(token_buffer)),
      target_buffer_(std::move(target_buffer)),
      host_tokens_(options.batch_size),
      host_targets_(options.batch_size),
      random_(options.seed),
      random_start_(0, corpus_.size() - options.context_length - 1) {}

absl::StatusOr<std::unique_ptr<InMemoryDataSetIterator>>
InMemoryDataSetIterator::Create(absl::Span<const int> tokens,
                                InMemoryDataSetOptions options,
                                cudaStream_t stream) {
  if (stream == nullptr || stream == cudaStreamLegacy ||
      stream == cudaStreamPerThread) {
    return absl::InvalidArgumentError(
        "InMemoryDataSetIterator requires an explicit CUDA stream");
  }
  if (options.batch_size <= 0 || options.context_length <= 0 ||
      options.batch_size % options.context_length != 0) {
    return absl::InvalidArgumentError(
        "batch_size must be positive and divisible by context_length");
  }
  if (tokens.size() <= static_cast<size_t>(options.context_length)) {
    return absl::InvalidArgumentError(
        "the corpus must contain more than context_length tokens");
  }
  if (static_cast<size_t>(options.batch_size) >
      std::numeric_limits<size_t>::max() / sizeof(int)) {
    return absl::InvalidArgumentError("batch_size is too large");
  }
  const size_t buffer_bytes =
      static_cast<size_t>(options.batch_size) * sizeof(int);
  ASSIGN_OR_RETURN(auto token_buffer,
                   cuda::Buffer::Allocate(buffer_bytes, stream));
  ASSIGN_OR_RETURN(auto target_buffer,
                   cuda::Buffer::Allocate(buffer_bytes, stream));
  return std::unique_ptr<InMemoryDataSetIterator>(new InMemoryDataSetIterator(
      std::vector<int>(tokens.begin(), tokens.end()), options, stream,
      std::move(token_buffer), std::move(target_buffer)));
}

absl::StatusOr<TokenBatch> InMemoryDataSetIterator::Next() {
  const size_t sequence_start_count =
      corpus_.size() - static_cast<size_t>(options_.context_length);
  const int sequences_per_batch = options_.batch_size / options_.context_length;
  for (int sequence = 0; sequence < sequences_per_batch; ++sequence) {
    size_t start;
    if (options_.order == InMemoryDataSetOrder::kRandom) {
      start = random_start_(random_);
    } else {
      start = next_sequential_start_;
      next_sequential_start_ =
          (next_sequential_start_ + options_.context_length) %
          sequence_start_count;
    }
    for (int position = 0; position < options_.context_length; ++position) {
      const int row = sequence * options_.context_length + position;
      host_tokens_[row] = corpus_[start + position];
      host_targets_[row] = corpus_[start + position + 1];
    }
  }

  RETURN_IF_ERROR(
      CudaStatus(cudaMemcpyAsync(token_buffer_.data(), host_tokens_.data(),
                                 token_buffer_.size_bytes(),
                                 cudaMemcpyHostToDevice, stream_),
                 "cudaMemcpyAsync(dataset tokens)"));
  RETURN_IF_ERROR(
      CudaStatus(cudaMemcpyAsync(target_buffer_.data(), host_targets_.data(),
                                 target_buffer_.size_bytes(),
                                 cudaMemcpyHostToDevice, stream_),
                 "cudaMemcpyAsync(dataset targets)"));
  return TokenBatch{.tokens = token_buffer_,
                    .targets = target_buffer_,
                    .batch_size = options_.batch_size};
}

absl::Status InMemoryDataSetIterator::Reset() {
  random_.seed(options_.seed);
  random_start_.reset();
  next_sequential_start_ = 0;
  return absl::OkStatus();
}

}  // namespace pluto
