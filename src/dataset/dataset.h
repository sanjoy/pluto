#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"

namespace pluto {

namespace tokenizer {
class Gpt2Tokenizer;
}  // namespace tokenizer

// A cheap, copyable view of an mmap-backed UTF-8 text file.
//
// Copies and subcorpora share ownership of the mapping. The mapping therefore
// remains valid until the final TextCorpus view is destroyed, even if the file
// is renamed or unlinked after LoadTextCorpus() returns.
class TextCorpus {
 public:
  TextCorpus() = default;

  absl::string_view text() const;
  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  // Returns a view into this corpus without copying or creating another mmap.
  absl::StatusOr<TextCorpus> SubCorpus(
      size_t offset, size_t length = absl::string_view::npos) const;

 private:
  struct Mapping;

  TextCorpus(std::shared_ptr<const Mapping> mapping, size_t offset,
             size_t size);

  friend absl::StatusOr<TextCorpus> LoadTextCorpus(absl::string_view path);

  std::shared_ptr<const Mapping> mapping_;
  size_t offset_ = 0;
  size_t size_ = 0;
};

// Memory-maps path read-only. No text bytes are copied into process-owned heap
// storage; an empty file produces a valid empty corpus without calling mmap().
absl::StatusOr<TextCorpus> LoadTextCorpus(absl::string_view path);

struct CorpusSplit {
  TextCorpus training;
  TextCorpus test;
};

// Splits corpus into a training prefix and held-out test suffix while
// preserving temporal order. The boundary is moved to the next newline when
// possible, which avoids leaking overlapping context across the two corpora.
absl::StatusOr<CorpusSplit> SplitCorpus(const TextCorpus& corpus,
                                        double test_fraction);

// One opaque, device-resident batch. The concrete iterator defines the element
// type, row shape, and any internal layout of data. batch_size is the number of
// logical examples rather than a byte or element count.
struct DataBatch {
  cuda::Buffer data;
  int32_t batch_size;
};

// Source of device-resident batches.
//
// Implementations own their staging storage, so returned Buffer handles stay
// alive independently through reference counting. Reset() restores the
// iterator's original sequence, which lets Evaluate() compare the same sample
// before and after training.
class DataSetIterator {
 public:
  virtual ~DataSetIterator() = default;

  virtual absl::StatusOr<DataBatch> Next() = 0;
  virtual absl::Status Reset() = 0;
};

enum class InMemoryDataSetOrder {
  // Independently sample a valid sequence window for every packed sequence.
  kRandom,
  // Visit non-overlapping sequence windows in corpus order, wrapping at end.
  kSequential,
};

struct InMemoryDataSetOptions {
  int batch_size;
  int context_length;
  InMemoryDataSetOrder order = InMemoryDataSetOrder::kRandom;
  uint64_t seed = 0;
};

// Produces next-token batches from a device-resident copy of a page-locked
// token array.
//
// A batch may pack multiple independent sequences, so batch_size must be a
// multiple of context_length. Create() uploads the corpus through executor;
// Next() then assembles each batch entirely with stream-ordered device copies.
// Its DataBatch::data contains 2 * batch_size int32 values: model input tokens
// first, followed by the corresponding one-token-shifted targets. Keeping this
// concrete schema out of DataBatch lets other iterators expose activation
// matrices through the same base interface. Random order is appropriate for
// optimization; sequential order plus Reset() is appropriate for stable
// train/test evaluation.
class InMemoryDataSetIterator final : public DataSetIterator {
 public:
  static absl::StatusOr<std::unique_ptr<InMemoryDataSetIterator>> Create(
      cuda::Executor& executor, cuda::PageLockedHostArray<int> tokens,
      InMemoryDataSetOptions options);

  absl::StatusOr<DataBatch> Next() override;
  absl::Status Reset() override;

  size_t token_count() const { return corpus_token_count_; }

 private:
  InMemoryDataSetIterator(cuda::Executor& executor, cuda::Buffer corpus,
                          size_t corpus_token_count,
                          InMemoryDataSetOptions options,
                          cuda::Buffer data_buffer);

  cuda::Buffer corpus_;
  size_t corpus_token_count_;
  InMemoryDataSetOptions options_;
  cuda::Executor& executor_;
  cuda::Buffer data_buffer_;
  std::mt19937_64 random_;
  std::uniform_int_distribution<size_t> random_start_;
  size_t next_sequential_start_ = 0;
};

// Tokenizes an mmap-backed corpus into page-locked memory and uploads the
// resulting token array to an in-memory dataset iterator's device storage.
absl::StatusOr<std::unique_ptr<InMemoryDataSetIterator>>
MakeInMemoryDataSetIterator(cuda::Executor& executor, const TextCorpus& corpus,
                            const tokenizer::Gpt2Tokenizer& tokenizer,
                            InMemoryDataSetOptions options);

}  // namespace pluto
