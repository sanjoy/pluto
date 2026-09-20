#pragma once

#include <random>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::llm {

// A zero temperature requests greedy decoding, not a limiting softmax. Call
// this before allocating model/GPU state to reject negative or nonfinite
// temperatures and negative generation lengths.
absl::Status ValidateGenerationOptions(int generation_tokens,
                                       double temperature);

// Selects a logical vocabulary ID from logits in increasing token-ID order.
// At temperature zero, the largest logit wins, ties go to the smallest ID,
// and random is not advanced. This makes greedy inference independent of the
// caller's RNG state and the number of earlier prompts.
//
// Positive temperatures use the supplied, explicitly seeded generator. The
// same inputs and generator state reproduce samples within the same C++
// runtime/build; distribution algorithms need not match across runtimes.
// Negative infinity masks a token; NaN, positive infinity, empty input, and
// an entirely masked vocabulary are rejected instead of silently picking an ID.
absl::StatusOr<int> SelectNextToken(absl::Span<const float> logits,
                                    double temperature, std::mt19937& random);

}  // namespace pluto::llm
