#pragma once

#include <cstddef>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::llm::one_shot_memorizer {

// Agreement between a CPU first-update prediction and an observed checkpoint.
struct FirstAdamUpdateScore {
  double squared_error = 0;          // Sum of squared predicted-weight errors.
  size_t sign_mismatches = 0;        // Unequal signs of changes from initial.
  size_t bit_equal_coordinates = 0;  // Equal FP32 weight bits, including zeros.
};

// Predicts initial - learning_rate * (g / (abs(g) + epsilon)) using FP32
// operands and a materialized FP32 normalized gradient, then sums squared
// errors in double precision. The compiler may contract the final multiply
// and subtract; bit identities depend on that compiler arithmetic policy.
// This algebraic first-Adam approximation assumes fresh zero moments and zero
// weight decay. It does NOT reproduce the GPU optimizer's
// moment/bias-correction intermediates, so correct CPU gradients need not
// predict bit-identical GPU weights. Sign comparisons use actual rounded weight
// changes; an update that rounds away has sign zero, and positive/negative zero
// have the same sign.
//
// All arrays must be nonempty, equally sized and finite; rate and epsilon must
// be finite and positive. Arithmetic overflow is an error, not a poor score.
// No model, corpus, token labels, CUDA executor, or GPU work is involved.
absl::StatusOr<FirstAdamUpdateScore> ScoreFirstAdamUpdate(
    absl::Span<const float> initial, absl::Span<const float> observed,
    absl::Span<const float> gradients, float learning_rate, float epsilon);

// One unordered prompt hypothesis; each distinct ID appears exactly once.
struct FirstUpdatePromptSet {
  std::vector<int> token_ids;  // Five ascending compact vocabulary IDs.
  int removed_token = -1;      // Original ID replaced, or -1 for the base set.
  int added_token = -1;        // Suffix ID inserted, or -1 for the base set.
};

// Returns the original five-ID set, followed by every distinct set obtained
// by replacing one original ID with one distinct non-EOS suffix ID. The base
// set comes first; subsequent sets follow ascending removed-ID/added-ID order.
// Repeated suffix IDs are deduplicated, EOS is ignored, and replacements that
// duplicate a retained prompt ID or reproduce the base set are skipped. An
// empty suffix (or one containing only EOS/existing prompt IDs) yields only
// the base set. Input order does not affect output order.
//
// Requires exactly five distinct non-EOS prompt IDs, a positive vocabulary
// size, and every ID/EOS in range. This is an exploratory one-error repair
// hypothesis, not a fact decoder: it reads no weights, labels, text or corpus,
// and cannot recover repeated prompt IDs or multiple missing prompt IDs.
absl::StatusOr<std::vector<FirstUpdatePromptSet>>
BuildOneTokenReplacementPromptSets(absl::Span<const int> prompt_ids,
                                   absl::Span<const int> suffix_ids,
                                   int vocabulary_size, int eos_token);

}  // namespace pluto::llm::one_shot_memorizer
