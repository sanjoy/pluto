#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::llm::one_shot_memorizer {

// Full real-vocabulary logits for the next token of the supplied history.
// Called synchronously; the borrowed history must not be retained. Neither
// gold text nor a target token is supplied to this model-evaluation callback.
using GreedySuffixLogits = std::function<absl::StatusOr<std::vector<float>>(
    absl::Span<const int> history)>;

// Side information for an exploratory suffix decoder, not known token order.
struct GreedySuffixReadoutOptions {
  int vocabulary_size = 0;  // Excludes padded/non-vocabulary logit columns.
  int eos_token = -1;       // Valid vocabulary ID that terminates generation.
  // Distinct valid IDs including EOS; treated as a set, never as an ordering.
  absl::Span<const int> selected_token_ids;
  int prompt_token_count = 5;  // Initial history contains this many EOS IDs.
  // Positive cap on non-EOS suffix tokens, including the seeded first token.
  // Zero derives the frozen heuristic 2 * selected_non_EOS_count, not a gold
  // length. One final next-token query may still produce EOS at this cap.
  int max_non_eos_tokens = 0;
};

// One unrestricted greedy prediction, including a possible cap-rejected token.
struct GreedySuffixStep {
  int token;               // Stable full-vocabulary top-1; ties use lowest ID.
  double log_probability;  // Full-vocabulary log-softmax of that prediction.
  bool appended;           // False only for non-EOS predicted beyond the cap.
};

// A continuation beginning with one selected non-EOS token after dummy EOS
// history. The first token is seeded, not predicted and NOT scored.
struct GreedySuffixCandidate {
  int start_token;
  std::vector<int> token_ids;  // Includes the seed and EOS if generation ended.
  std::vector<GreedySuffixStep> steps;  // All queries, including cap rejection.
  double log_score = 0;  // Sum for appended successors plus EOS; seed scores 0.
  bool terminated_with_eos = false;
  bool all_tokens_selected = true;  // No masking; this is checked afterward.
  size_t distinct_selected_covered = 0;  // Excludes EOS, ignores repetitions.
  bool covers_selected_tokens = false;
  bool hit_token_limit =
      false;  // Next non-EOS prediction could not be appended.

  bool accepted() const {
    return terminated_with_eos && all_tokens_selected && covers_selected_tokens;
  }
};

// Every attempted start and the ranking of accepted candidates, if any.
struct GreedySuffixReadout {
  std::vector<GreedySuffixCandidate> candidates;  // Ascending start-token ID.
  // Indices into candidates, greatest log_score first; ties use start-token ID.
  // Empty is a valid negative result, not an error and not a fallback answer.
  std::vector<size_t> accepted_order;
  size_t model_query_count = 0;
  int max_non_eos_tokens = 0;  // Resolved generation cap used for every start.
};

// For each selected non-EOS ID s, starts with [EOS] * prompt_token_count + [s]
// and greedily generates using the actual previous predictions. The model's
// top-1 is taken over its ENTIRE vocabulary: selected IDs never mask logits.
// Repetitions are allowed. Accept only EOS-terminated candidates containing
// every selected ID and no outside ID; rank them by the score described above.
// These scores are conditional on different seeds/dummy histories, not
// calibrated probabilities of reconstructing the training text.
//
// This CPU-only control logic does not infer a prompt or read a corpus. The
// callback may evaluate a GPU model. Rejects malformed options, empty callback,
// wrong logit vector lengths, and nonfinite logits (even at unselected IDs).
absl::StatusOr<GreedySuffixReadout> ReadGreedySuffix(
    const GreedySuffixReadoutOptions& options,
    const GreedySuffixLogits& logits);

}  // namespace pluto::llm::one_shot_memorizer
