#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"

namespace pluto::llm::qwen {

// One signed input-token embedding in a left-to-right vector expression.
struct EmbeddingTerm {
  int token_id;
  int coefficient;  // Exactly +1 or -1.
};

// A vocabulary row ranked by cosine similarity; L2 uses unnormalized vectors.
struct EmbeddingMatch {
  int token_id;
  float cosine_similarity;
  float l2_distance;
};

// An embedding-only GPU view of a checkpoint, not a decoder or an LM head.
// The executor must outlive the table and every returned page-locked array.
// Instances use one stream and are not intended for concurrent calls.
class EmbeddingAlgebraTable final {
 public:
  // Loads only embed_tokens.weight. Search excludes model padding beyond the
  // tokenizer vocabulary, even when the checkpoint contains extra rows.
  static absl::StatusOr<std::unique_ptr<EmbeddingAlgebraTable>> Load(
      cuda::Executor& executor, const std::filesystem::path& checkpoint,
      int searchable_token_count);

  // Shares immutable, row-major BF16 weights, primarily for independent tests.
  // A finite zero row is allowed but cannot participate in cosine ranking.
  static absl::StatusOr<std::unique_ptr<EmbeddingAlgebraTable>> Create(
      cuda::Executor& executor, cuda::Buffer bf16_weights, int rows,
      int dimensions, int searchable_token_count);

  // Sums embeddings in FP32. A zero result is valid (for example, king-king).
  absl::StatusOr<cuda::PageLockedHostArray<float>> Evaluate(
      absl::Span<const EmbeddingTerm> terms) const;

  // Returns at most top_k nonzero rows, including expression inputs. Cosine
  // similarity is not a probability. Exact ties prefer the lower token ID.
  absl::StatusOr<std::vector<EmbeddingMatch>> Nearest(
      absl::Span<const float> query, int top_k = 3) const;

  int dimensions() const { return dimensions_; }
  int vocab_size() const { return rows_; }

 private:
  EmbeddingAlgebraTable(cuda::Executor& executor, cuda::Buffer weights,
                        int rows, int dimensions, int searchable_token_count,
                        cuda::PageLockedHostArray<float> squared_norms);

  cuda::Executor& executor_;
  cuda::Buffer weights_;
  int rows_;
  int dimensions_;
  int searchable_token_count_;
  cuda::PageLockedHostArray<float> squared_norms_;
};

}  // namespace pluto::llm::qwen
