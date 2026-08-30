#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"

namespace pluto {

// One next-token language-modeling batch. `tokens` contains model inputs and
// `targets` contains the same sequences shifted left by one token. Both are
// int32 device buffers with `batch_size` elements on the same CUDA executor.
struct TokenBatch {
  cuda::Buffer tokens;
  cuda::Buffer targets;
  int batch_size;
};

// Source of device-resident next-token batches.
//
// Implementations own their staging storage, so returned Buffer handles stay
// alive independently through reference counting. Reset() restores the
// iterator's original sequence, which lets Evaluate() compare the same sample
// before and after training.
class DataSetIterator {
 public:
  virtual ~DataSetIterator() = default;

  virtual absl::StatusOr<TokenBatch> Next() = 0;
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

// Produces next-token batches from a copied, host-resident token vector.
//
// A batch may pack multiple independent sequences, so batch_size must be a
// multiple of context_length. Next() fills reusable host staging vectors and
// asynchronously copies them into reusable device buffers. Random order is
// appropriate for optimization; sequential order plus Reset() is appropriate
// for stable train/test evaluation.
class InMemoryDataSetIterator final : public DataSetIterator {
 public:
  static absl::StatusOr<std::unique_ptr<InMemoryDataSetIterator>> Create(
      absl::Span<const int> tokens, InMemoryDataSetOptions options,
      cuda::Executor& executor);

  absl::StatusOr<TokenBatch> Next() override;
  absl::Status Reset() override;

 private:
  InMemoryDataSetIterator(std::vector<int> corpus,
                          InMemoryDataSetOptions options,
                          cuda::Executor& executor, cuda::Buffer token_buffer,
                          cuda::Buffer target_buffer);

  std::vector<int> corpus_;
  InMemoryDataSetOptions options_;
  cuda::Executor& executor_;
  cuda::Buffer token_buffer_;
  cuda::Buffer target_buffer_;
  std::vector<int> host_tokens_;
  std::vector<int> host_targets_;
  std::mt19937_64 random_;
  std::uniform_int_distribution<size_t> random_start_;
  size_t next_sequential_start_ = 0;
};

}  // namespace pluto
