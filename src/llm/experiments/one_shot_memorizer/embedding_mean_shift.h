#pragma once

#include <cstddef>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::llm::one_shot_memorizer {

// A fixed row-common translation, not a fitted token-specific correction.
struct EmbeddingMeanShift {
  // Row-major embedding after adding shift to every row and rounding to FP32.
  std::vector<float> values;
  // One FP64 mean(reference - embedding) per column, averaged across rows.
  std::vector<double> shift;
  // Squared Euclidean errors to the supplied reference, accumulated in FP64.
  double squared_error_before = 0;
  double squared_error_after = 0;
};

// Matches column means by adding the same vector to every embedding row.
// Uses FP64 differences, sums and addition, with one final FP32 rounding per
// output. A zero shift preserves the original bytes, including signed zero.
// Centered row differences are unchanged except for final FP32 rounding.
// Both arrays must be finite, nonempty, equal-sized row-major matrices whose
// sizes are multiples of a positive width. Rejects nonfinite FP32 results.
// The reference is an oracle input: this is not dataset-only reconstruction.
absl::StatusOr<EmbeddingMeanShift> MatchEmbeddingMeans(
    absl::Span<const float> embedding, absl::Span<const float> reference,
    size_t width);

}  // namespace pluto::llm::one_shot_memorizer
