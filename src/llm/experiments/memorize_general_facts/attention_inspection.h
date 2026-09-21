#pragma once

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/dataset/detokenizer.h"
#include "src/llm/layer.h"
#include "src/llm/layer_hooks.h"

namespace pluto::llm {

// Records read-only attention probabilities for a single-sample inference
// forward pass. Print reports each head separately and then forgets the pass,
// so one instance can be reused while generating successive tokens. Buffers
// are retained by reference; the hooks never synchronize or modify GPU data.
// The Executor must outlive this object and its captured callback references.
class AttentionProbabilityInspector final {
 public:
  explicit AttentionProbabilityInspector(cuda::Executor& executor);
  AttentionProbabilityInspector(const AttentionProbabilityInspector&) = delete;
  AttentionProbabilityInspector& operator=(
      const AttentionProbabilityInspector&) = delete;
  AttentionProbabilityInspector(AttentionProbabilityInspector&&) = delete;
  AttentionProbabilityInspector& operator=(AttentionProbabilityInspector&&) =
      delete;

  // Borrow for one complete model forward. Call Print only after that forward
  // succeeds and generation selects an actual, non-EOS output token.
  LayerHooks& layer_hooks() { return hooks_; }

  // input_prefix is the unpadded input that produced produced_token. The
  // output token's zero-based position is input_prefix.size(): token 6 was
  // predicted from positions 0..5, so its attention report is a 6x6 lower
  // triangle. The new token itself has not yet been fed back into the model.
  // Future/padding rows and columns and the masked upper triangle are omitted.
  // Per-head leading squares are downloaded through pinned storage, with one
  // synchronization before printing. Successful printing clears the saved
  // buffers and scope numbering; failures leave the record available to retry.
  absl::Status Print(cuda::Executor& executor,
                     const tokenizer::Detokenizer& detokenizer,
                     absl::Span<const int> input_prefix, int produced_token,
                     std::ostream& output);

 private:
  struct Scope {
    std::string path;
    size_t next_child = 0;
  };
  struct Entry {
    std::string path;
    size_t heads;
    size_t sequence;
    cuda::Buffer probabilities;
  };

  absl::Status ValidateExecutor(const cuda::Executor& executor) const;
  std::string NextPath(absl::string_view name);
  absl::Status Record(cuda::Executor& executor, absl::string_view name,
                      const ActivationType& type,
                      const cuda::Buffer& probabilities);

  cuda::Executor& executor_;
  LayerHooks hooks_;
  size_t next_root_ = 0;
  std::vector<Scope> scopes_;
  std::vector<Entry> entries_;
};

}  // namespace pluto::llm
