#pragma once

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "src/cuda/executor.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/layer.h"

namespace pluto::llm::weight_sensitivity {

// Whole-corpus scores, with one entry per original sentence (not per token).
struct CompletionScores {
  // One iff every scored suffix token and EOS is predicted exactly correctly.
  std::vector<uint8_t> exact;
  // Number of supervised suffix/EOS positions; prompt and padding are ignored.
  int64_t scored_tokens = 0;
  // Incorrect teacher-forced predictions, NOT errors along a greedy rollout.
  int64_t token_errors = 0;
  // Scored rows containing nonfinite logits; each is also a token error.
  int64_t nonfinite_rows = 0;
};

// Resets and scores one complete epoch in corpus order. Requires an unshuffled
// iterator, a causal deterministic model, and FP32 vocabulary logits. All work
// uses executor; only predicted IDs and targets are copied to pinned CPU
// memory. Nonfinite logits count as incorrect predictions instead of aborting
// an ablation. Invalid dimensions, targets, or execution errors still fail.
//
// Scoring every teacher-forced suffix position is sufficient to decide exact
// greedy completion, including EOS: until the first wrong prediction, greedy
// decoding and teacher forcing have identical prefixes. Thus either all
// predictions agree, or both fail at that first divergence. This equivalence
// does not extend to the number of later erroneous tokens: token_errors is
// strictly a teacher-forced diagnostic, not a greedy-trajectory error count.
absl::StatusOr<CompletionScores> EvaluateCompletions(
    cuda::Executor& executor, const Layer& model,
    PaddedLineDataSetIterator& data, int vocabulary_size);

}  // namespace pluto::llm::weight_sensitivity
