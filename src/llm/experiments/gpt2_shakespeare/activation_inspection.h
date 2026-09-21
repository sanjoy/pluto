#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/dataset/detokenizer.h"
#include "src/llm/layer.h"
#include "src/llm/layer_hooks.h"

namespace pluto::llm {

// Read-only vocabulary projections of one forward pass. Hidden vectors are
// scored by raw activation * token_embedding^T, with no extra normalization or
// temperature. Vocabulary-width FP32 outputs are already logits. Each result
// contains the top three probabilities from the FULL logical vocabulary, not
// a softmax renormalized over only the winning tokens. Other output spaces
// have no canonical vocabulary projection and are omitted from the report.
//
// Only compact top-three results survive a callback; activations are neither
// copied to the CPU nor modified. Print downloads all results using pinned
// storage and one stream synchronization. Create a new instance for each
// inspected forward pass. The executor must outlive it and its buffer handles.
class NeighboringVocabInspector final {
 public:
  // token_embedding is a row-major FP32 table with at least vocab_size rows;
  // extra rows are padding and never appear in the results. Inspect the first
  // positions flattened token rows in [batch, sequence, feature] outputs,
  // excluding any trailing causal padding. Rank-two/scalar outputs and shapes
  // ambiguous between hidden vectors and logits are omitted.
  // position_offset changes printed numbering only, not which rows are read.
  // min_prob is an inclusive full-vocabulary probability threshold in [0, 1],
  // not a percentage. Its default is 0.01 (1%); zero keeps all top-three
  // tokens.
  static absl::StatusOr<std::unique_ptr<NeighboringVocabInspector>> Create(
      cuda::Executor& executor, const Buffer& token_embedding, int vocab_size,
      int model_width, int positions, int position_offset = 0,
      double min_prob = 0.01);

  // Borrow the callbacks for Layer::fwd. The inspector must outlive every
  // call using this reference. Moving the owning unique_ptr is safe; moving
  // or copying the inspector itself is disabled to keep callback captures
  // valid.
  LayerHooks& layer_hooks() { return hooks_; }

  // Group output by token position, preserving forward callback order within
  // each group. Paths include sibling indices, so repeated transformer blocks
  // are distinct. Token bytes are quoted and escaped (including partial UTF-8
  // tokens), making whitespace/control characters visible and unambiguous.
  // input_tokens contains exactly the inspected window's token IDs, not the
  // full prompt or causal padding. Each heading includes its decoded input
  // token; position_offset affects numbering, not indexing into this span.
  // Only top-three candidates with probability >= min_prob are printed, with
  // their original probabilities. Unsupported/invalid rows, layers with no
  // surviving candidates, and completely empty position sections are omitted.
  absl::Status Print(cuda::Executor& executor,
                     const tokenizer::Detokenizer& detokenizer,
                     absl::Span<const int> input_tokens,
                     std::ostream& output) const;

 private:
  struct Scope {
    std::string name;
    std::string path;
    size_t next_child = 0;
  };
  struct Entry {
    std::string label;
    Buffer neighbors;
  };

  NeighboringVocabInspector(cuda::Executor& executor, Buffer token_embedding,
                            int vocab_size, int model_width, int positions,
                            int position_offset, int embedding_rows,
                            double min_prob)
      : executor_(executor),
        token_embedding_(std::move(token_embedding)),
        vocab_size_(vocab_size),
        model_width_(model_width),
        positions_(positions),
        position_offset_(position_offset),
        embedding_rows_(embedding_rows),
        min_prob_(min_prob),
        hooks_{.activation_hook =
                   [this](cuda::Executor& executor, absl::string_view name,
                          absl::Span<const ActivationType> types,
                          absl::Span<Buffer> activations) {
                     return activation_hook(executor, name, types, activations);
                   },
               .enter_combinator =
                   [this](cuda::Executor& executor, absl::string_view name) {
                     return enter_combinator(executor, name);
                   },
               .exit_combinator =
                   [this](cuda::Executor& executor) {
                     return exit_combinator(executor);
                   }} {}

  NeighboringVocabInspector(const NeighboringVocabInspector&) = delete;
  NeighboringVocabInspector& operator=(const NeighboringVocabInspector&) =
      delete;
  NeighboringVocabInspector(NeighboringVocabInspector&&) = delete;
  NeighboringVocabInspector& operator=(NeighboringVocabInspector&&) = delete;

  absl::Status activation_hook(
      cuda::Executor& executor, absl::string_view layer_name,
      absl::Span<const ActivationType> activation_types,
      absl::Span<Buffer> activations);
  absl::Status enter_combinator(cuda::Executor& executor,
                                absl::string_view layer_name);
  absl::Status exit_combinator(cuda::Executor& executor);

  absl::Status ValidateExecutor(const cuda::Executor& executor) const;
  std::string NextPath(absl::string_view name);

  cuda::Executor& executor_;
  Buffer token_embedding_;
  int vocab_size_;
  int model_width_;
  int positions_;
  int position_offset_;
  int embedding_rows_;
  double min_prob_;
  LayerHooks hooks_;
  size_t next_root_ = 0;
  std::vector<Scope> scopes_;
  // Combinators publish their activation immediately AFTER exit_combinator.
  // Keep its already-assigned path so this callback does not consume another
  // sibling index or appear outside the subtree it just completed.
  std::optional<Scope> exited_scope_;
  std::vector<Entry> entries_;
};

}  // namespace pluto::llm
