#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "src/cuda/buffer.h"
#include "src/dataset/dataset.h"
#include "src/dataset/tokenizer.h"

namespace pluto::llm::fsm {

struct DataSetOptions {
  int batch_size = 4;  // Independent FSM examples, not flattened token count.
  int context_length = 1024;  // Right-padded width; overlong lines are errors.
  bool shuffle = false;     // Shuffle examples without replacement each epoch.
  uint64_t seed = 17;       // Reset restores this sampling sequence.
  bool answer_only = true;  // Score only the final field plus EOS when true.
  int eos_token =
      50256;  // Also used as right-padding input, never a pad target.
};

// Independently tokenized FSM sentences with per-sentence answer boundaries.
// Every label is checked by executing its FSM before any training occurs.
// With answer_only, descriptions and inputs are context, not prediction
// targets: the last semicolon predicts the first answer token; the last answer
// token predicts EOS. All earlier and padding targets are -1. This avoids
// treating randomly generated transition descriptions as learnable output text.
//
// Corpus and reusable batch buffers live on the GPU. Next() allocates nothing,
// preserves sequence boundaries, and emits a smaller final batch each epoch.
// Consume its buffers on the same executor before calling Next() again. The
// executor must outlive the iterator and all returned buffer handles.
class FsmDataSetIterator final : public DataSetIterator {
 public:
  static absl::StatusOr<std::unique_ptr<FsmDataSetIterator>> Create(
      cuda::Executor& executor, absl::string_view corpus_text,
      const tokenizer::Tokenizer& tokenizer, DataSetOptions options);

  absl::StatusOr<DataBatch> Next() override;
  absl::Status Reset() override;
  size_t sample_count() const { return supervised_rows_.size(); }
  size_t batches_per_epoch() const;
  int max_tokens() const { return max_tokens_; }
  int64_t supervised_row_count() const;

 private:
  struct BatchBuffers {
    cuda::Buffer inputs;
    cuda::Buffer targets;
  };
  FsmDataSetIterator(cuda::Executor& executor, DataSetOptions options,
                     cuda::Buffer inputs, cuda::Buffer targets,
                     std::vector<int> supervised_rows, int max_tokens,
                     BatchBuffers full_batch,
                     std::optional<BatchBuffers> partial_batch);
  void BeginEpoch();

  cuda::Executor& executor_;
  DataSetOptions options_;
  cuda::Buffer corpus_inputs_;
  cuda::Buffer corpus_targets_;
  std::vector<int> supervised_rows_;
  int max_tokens_;
  BatchBuffers full_batch_;
  std::optional<BatchBuffers> partial_batch_;
  std::vector<size_t> order_;
  std::mt19937_64 random_;
  size_t next_sample_ = 0;
};

}  // namespace pluto::llm::fsm
