#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/dataset.h"
#include "src/dataset/tokenizer.h"

namespace pluto {

struct PaddedLineDataSetOptions {
  int batch_size = 1;
  int context_length = 1024;
  int prompt_tokens = 5;
  int eos_token = 50256;
  // Shuffle without replacement at the start of each epoch. Reset() restores
  // the seed, including the first epoch's permutation, for reproducible runs.
  bool shuffle = false;
  uint64_t seed = 0;
};

// A finite corpus of independently tokenized lines, repeated in epochs.
//
// Each sample has exactly context_length input positions: its original tokens
// followed by EOS padding. Targets are shifted by one, with EOS after the last
// text token. Targets for the supplied prompt and all padding are -1, the
// cross-entropy loss's ignored-target sentinel. No BOS token is inserted.
// For L text tokens and P prompt tokens, exactly L - P + 1 targets are scored;
// the final one is EOS. Right padding cannot influence preceding positions in
// causal attention, so padded positions need only be excluded from the loss.
//
// The final batch of each epoch is smaller when necessary; no sample is
// duplicated to fill a batch. Next() then begins a new epoch. All corpus and
// batch device allocations happen in Create(); Next() only queues D2D copies.
// Returned buffers are reused, so consume each batch on the same executor
// before requesting the next one. The executor must outlive this iterator and
// every returned buffer, including the retained CPU token storage.
class PaddedLineDataSetIterator final : public DataSetIterator {
 public:
  // A final newline is allowed. Empty/blank lines, samples shorter than the
  // prompt, invalid vocabulary IDs, and overlength samples are errors; this
  // factory never silently drops, concatenates, or truncates a sample.
  static absl::StatusOr<std::unique_ptr<PaddedLineDataSetIterator>> Create(
      cuda::Executor& executor, absl::string_view corpus_text,
      const tokenizer::Tokenizer& tokenizer, PaddedLineDataSetOptions options);

  absl::StatusOr<DataBatch> Next() override;
  absl::Status Reset() override;

  size_t sample_count() const { return lengths_.size(); }
  size_t batches_per_epoch() const;
  int64_t supervised_row_count() const { return supervised_row_count_; }
  const PaddedLineDataSetOptions& options() const { return options_; }

  // Original, unpadded tokens in corpus order, independent of shuffle order.
  // The returned view remains valid for this iterator's lifetime.
  absl::Span<const int> sample_tokens(size_t index) const;

 private:
  struct BatchBuffers {
    cuda::Buffer inputs;
    cuda::Buffer targets;
  };

  PaddedLineDataSetIterator(
      cuda::Executor& executor, PaddedLineDataSetOptions options,
      cuda::PageLockedHostArray<int> host_inputs, std::vector<int> lengths,
      cuda::Buffer corpus_inputs, cuda::Buffer corpus_targets,
      BatchBuffers full_batch, std::optional<BatchBuffers> partial_batch);
  void BeginEpoch();

  cuda::Executor& executor_;
  PaddedLineDataSetOptions options_;
  cuda::PageLockedHostArray<int> host_inputs_;
  std::vector<int> lengths_;
  cuda::Buffer corpus_inputs_;
  cuda::Buffer corpus_targets_;
  BatchBuffers full_batch_;
  std::optional<BatchBuffers> partial_batch_;
  std::vector<size_t> order_;
  std::mt19937_64 random_;
  size_t next_sample_ = 0;
  int64_t supervised_row_count_ = 0;
};

}  // namespace pluto
