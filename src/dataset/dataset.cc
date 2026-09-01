#include "src/dataset/dataset.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "src/util/status_macros.h"
#include "src/cuda/buffer.h"
#include "src/dataset/tokenizer.h"

namespace pluto {
namespace {

static_assert(sizeof(int) == sizeof(int32_t));

class ScopedFileDescriptor {
 public:
  explicit ScopedFileDescriptor(int descriptor) : descriptor_(descriptor) {}
  ScopedFileDescriptor(const ScopedFileDescriptor&) = delete;
  ScopedFileDescriptor& operator=(const ScopedFileDescriptor&) = delete;
  ~ScopedFileDescriptor() {
    if (descriptor_ >= 0) close(descriptor_);
  }

  int get() const { return descriptor_; }

 private:
  int descriptor_;
};

absl::Status CudaStatus(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return absl::OkStatus();
  return absl::InternalError(absl::StrCat(operation,
                                          " failed: ", cudaGetErrorName(error),
                                          ": ", cudaGetErrorString(error)));
}

}  // namespace

struct TextCorpus::Mapping {
  Mapping(void* mapped_address, size_t mapped_size)
      : address(mapped_address), size(mapped_size) {}

  ~Mapping() { munmap(address, size); }

  void* address;
  size_t size;
};

TextCorpus::TextCorpus(std::shared_ptr<const Mapping> mapping, size_t offset,
                       size_t size)
    : mapping_(std::move(mapping)), offset_(offset), size_(size) {}

absl::string_view TextCorpus::text() const {
  if (size_ == 0) return {};
  const auto* data = static_cast<const char*>(mapping_->address);
  return absl::string_view(data + offset_, size_);
}

absl::StatusOr<TextCorpus> TextCorpus::SubCorpus(size_t offset,
                                                 size_t length) const {
  if (offset > size_) {
    return absl::OutOfRangeError(absl::StrCat("subcorpus offset ", offset,
                                              " exceeds corpus size ", size_));
  }
  const size_t available = size_ - offset;
  if (length == absl::string_view::npos || length > available) {
    length = available;
  }
  return TextCorpus(mapping_, offset_ + offset, length);
}

absl::StatusOr<TextCorpus> LoadTextCorpus(absl::string_view path) {
  if (path.empty()) {
    return absl::InvalidArgumentError("text corpus path must not be empty");
  }
  if (path.find('\0') != absl::string_view::npos) {
    return absl::InvalidArgumentError(
        "text corpus path must not contain a NUL byte");
  }
  const std::string path_string(path);
  ScopedFileDescriptor file(open(path_string.c_str(), O_RDONLY | O_CLOEXEC));
  if (file.get() < 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("cannot open text corpus ", path_string));
  }

  struct stat attributes {};
  if (fstat(file.get(), &attributes) != 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("cannot stat text corpus ", path_string));
  }
  if (attributes.st_size < 0 || static_cast<uintmax_t>(attributes.st_size) >
                                    std::numeric_limits<size_t>::max()) {
    return absl::ResourceExhaustedError(
        absl::StrCat("text corpus is too large to map: ", path_string));
  }
  const size_t size = static_cast<size_t>(attributes.st_size);
  if (size == 0) return TextCorpus(nullptr, 0, 0);

  void* address = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, file.get(), 0);
  if (address == MAP_FAILED) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("cannot mmap text corpus ", path_string));
  }
  return TextCorpus(std::make_shared<TextCorpus::Mapping>(address, size), 0,
                    size);
}

InMemoryDataSetIterator::InMemoryDataSetIterator(cuda::Executor& executor,
                                                 cuda::Buffer corpus,
                                                 size_t corpus_token_count,
                                                 InMemoryDataSetOptions options,
                                                 cuda::Buffer token_buffer,
                                                 cuda::Buffer target_buffer)
    : corpus_(std::move(corpus)),
      corpus_token_count_(corpus_token_count),
      options_(options),
      executor_(executor),
      token_buffer_(std::move(token_buffer)),
      target_buffer_(std::move(target_buffer)),
      random_(options.seed),
      random_start_(0, corpus_token_count_ - options.context_length - 1) {}

absl::StatusOr<std::unique_ptr<InMemoryDataSetIterator>>
InMemoryDataSetIterator::Create(cuda::Executor& executor,
                                absl::Span<const int> tokens,
                                InMemoryDataSetOptions options) {
  return Create(executor, std::vector<int>(tokens.begin(), tokens.end()),
                options);
}

absl::StatusOr<std::unique_ptr<InMemoryDataSetIterator>>
InMemoryDataSetIterator::Create(cuda::Executor& executor,
                                std::vector<int> tokens,
                                InMemoryDataSetOptions options) {
  if (options.batch_size <= 0 || options.context_length <= 0 ||
      options.batch_size % options.context_length != 0) {
    return absl::InvalidArgumentError(
        "batch_size must be positive and divisible by context_length");
  }
  if (tokens.size() <= static_cast<size_t>(options.context_length)) {
    return absl::InvalidArgumentError(
        "the corpus must contain more than context_length tokens");
  }
  if (tokens.size() > std::numeric_limits<size_t>::max() / sizeof(int)) {
    return absl::InvalidArgumentError("the corpus is too large");
  }
  if (static_cast<size_t>(options.batch_size) >
      std::numeric_limits<size_t>::max() / sizeof(int)) {
    return absl::InvalidArgumentError("batch_size is too large");
  }
  const size_t buffer_bytes =
      static_cast<size_t>(options.batch_size) * sizeof(int);
  const size_t corpus_bytes = tokens.size() * sizeof(int);
  ASSIGN_OR_RETURN(auto corpus_buffer,
                   cuda::Buffer::Allocate(executor, corpus_bytes));
  ASSIGN_OR_RETURN(auto token_buffer,
                   cuda::Buffer::Allocate(executor, buffer_bytes));
  ASSIGN_OR_RETURN(auto target_buffer,
                   cuda::Buffer::Allocate(executor, buffer_bytes));
  RETURN_IF_ERROR(CudaStatus(
      cudaMemcpyAsync(corpus_buffer.data(), tokens.data(), corpus_bytes,
                      cudaMemcpyHostToDevice, executor.stream()),
      "cudaMemcpyAsync(dataset corpus)"));
  // The caller-owned span and the by-value vector overload can both cease to
  // exist when Create() returns. Complete this one-time upload before releasing
  // either host source; subsequent Next() calls remain fully asynchronous.
  RETURN_IF_ERROR(executor.Synchronize());
  return std::unique_ptr<InMemoryDataSetIterator>(new InMemoryDataSetIterator(
      executor, std::move(corpus_buffer), tokens.size(), options,
      std::move(token_buffer), std::move(target_buffer)));
}

absl::StatusOr<TokenBatch> InMemoryDataSetIterator::Next() {
  const size_t sequence_start_count =
      corpus_token_count_ - static_cast<size_t>(options_.context_length);
  const int sequences_per_batch = options_.batch_size / options_.context_length;
  const size_t sequence_bytes =
      static_cast<size_t>(options_.context_length) * sizeof(int);
  const auto* corpus = static_cast<const char*>(corpus_.data());
  auto* batch_tokens = static_cast<char*>(token_buffer_.data());
  auto* batch_targets = static_cast<char*>(target_buffer_.data());
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
    const size_t destination_offset =
        static_cast<size_t>(sequence) * sequence_bytes;
    RETURN_IF_ERROR(CudaStatus(
        cudaMemcpyAsync(batch_tokens + destination_offset,
                        corpus + start * sizeof(int), sequence_bytes,
                        cudaMemcpyDeviceToDevice, executor_.stream()),
        "cudaMemcpyAsync(dataset token slice)"));
    RETURN_IF_ERROR(CudaStatus(
        cudaMemcpyAsync(batch_targets + destination_offset,
                        corpus + (start + 1) * sizeof(int), sequence_bytes,
                        cudaMemcpyDeviceToDevice, executor_.stream()),
        "cudaMemcpyAsync(dataset target slice)"));
  }
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

absl::StatusOr<std::unique_ptr<InMemoryDataSetIterator>>
MakeInMemoryDataSetIterator(cuda::Executor& executor, const TextCorpus& corpus,
                            const tokenizer::Gpt2Tokenizer& tokenizer,
                            InMemoryDataSetOptions options) {
  ASSIGN_OR_RETURN(auto tokens, tokenizer.Encode(corpus.text()));
  return InMemoryDataSetIterator::Create(executor, std::move(tokens), options);
}

}  // namespace pluto
