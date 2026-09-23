#pragma once

#include <cstddef>

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

}  // namespace pluto::llm::one_shot_memorizer
