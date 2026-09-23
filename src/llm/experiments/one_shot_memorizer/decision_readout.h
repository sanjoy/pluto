#pragma once

#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::llm::one_shot_memorizer {

// Real-valued final LayerNorm plus vocabulary head, written in coordinates
// useful for decision constraints. This does not simulate BF16 rounding.
struct DecisionReadout {
  int width = 0;
  int vocab_size = 0;
  // Row-major [vocab_size, width]: center(gamma * embedding[token]).
  std::vector<double> directions;
  // Per-token affine score beta dot embedding[token]; never silently dropped.
  std::vector<double> offsets;
  double epsilon = 1e-5;  // Positive additive constant in LayerNorm variance.
};

// Embeddings are a nonempty row-major [vocab_size, width] matrix. Gamma and
// beta each have width entries. The vocabulary dimension is inferred.
absl::StatusOr<DecisionReadout> MakeDecisionReadout(
    absl::Span<const float> embeddings, int width,
    absl::Span<const float> gamma, absl::Span<const float> beta,
    double epsilon = 1e-5);

// Returns row-major [rows, vocab_size] scores for residuals [rows, width].
// score_j(z) = directions[j] dot center(z) / s + offsets[j], where
// s = sqrt(mean(center(z)^2) + epsilon). Directions made above are centered,
// so their dot product with z equals that with center(z) in exact arithmetic.
// Empty input is allowed. Malformed/nonfinite inputs and overflowing results
// fail instead of producing a partial score matrix.
absl::StatusOr<std::vector<double>> EvaluateDecisionReadout(
    const DecisionReadout& readout, absl::Span<const float> residuals);

}  // namespace pluto::llm::one_shot_memorizer
