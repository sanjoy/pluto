#pragma once

#include <array>
#include <cstddef>
#include <istream>
#include <vector>

#include "absl/status/statusor.h"

namespace pluto::llm::one_shot_memorizer {

// Saved FP32 affine coefficients for one fixed 152-feature quadratic MLP.
struct QuadraticCoefficients {
  std::vector<float> weights;  // Row-major [152,16], feature first.
  std::vector<float> bias;     // [16], the unpenalized affine intercept.
};

// One joint-replacement first failure. These labels and prefixes are report
// metadata, not a source for fitting or choosing replacement coefficients.
struct QuadraticFailure {
  size_t line;          // One-based original corpus line, in [1,1024].
  int target_position;  // Absolute zero-based target position, in [5,1024].
  int predicted_token;  // Actual compact-vocabulary argmax, in [0,4474].
  int expected_token;   // Gold compact token, or EOS; differs from prediction.
  std::vector<int>
      prefix;  // Exactly target_position report-only gold token IDs.
};

// Reads the exact coefficients.tsv schema. Selects quadratic_refit only, with
// every W2/b2 coordinate present exactly once in each of eight blocks. Rejects
// quadratic W1/b1, nonfinite/out-of-range FP32 values, bad indices, duplicates,
// incomplete selected tensors, unknown conditions/tensors, and malformed rows.
// Known other conditions are shape/value validated but need not be complete.
// No CUDA operations or checkpoint access are performed.
absl::StatusOr<std::array<QuadraticCoefficients, 8>> ReadQuadraticCoefficients(
    std::istream& input);

// Reads the exact first_failures.tsv schema, selecting quadratic_refit/all.
// Validates every row, including ignored conditions/selections. Selected corpus
// lines must be unique and agree with the fixed index%5 held-out split; query
// position must equal target_position-1. Prefix length and compact IDs are
// checked. At least one selected failure is required, but no fixed failure
// count is assumed. Returned rows retain file order. The caller must separately
// verify prefixes/expected tokens against its corpus and model outputs.
absl::StatusOr<std::vector<QuadraticFailure>> ReadQuadraticFailures(
    std::istream& input);

}  // namespace pluto::llm::one_shot_memorizer
