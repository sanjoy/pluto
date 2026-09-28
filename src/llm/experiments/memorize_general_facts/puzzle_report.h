#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace pluto::llm::memorize_general_facts {

// One fact/position's captured A3 residual and next-token label for reporting.
// Prompt and padding positions are included, but only labeled rows are scored.
struct PuzzlePoint {
  // Zero-based index of the sentence in PuzzleReportData::facts.
  int fact_index = -1;
  // Zero-based input-token position, in [0, context_length), including padding.
  int position = -1;
  // Input ID in the model's vocabulary (compact when enabled); EOS for padding.
  int input_token = -1;
  // Expected next-token ID (not a prediction), including terminal EOS.
  // Uses the input vocabulary; -1 excludes masked prompt and padding rows.
  int target_token = -1;
  // True beyond the fact's real input tokens; such rows must have target -1.
  // The last real token predicting terminal EOS is not padding.
  bool padding = false;
  // Full residual after A3's residual addition, before that block's MLP.
  // Exactly model_width finite values, not a projection; captured BF16 values
  // are expanded exactly to float by the caller.
  std::vector<float> coordinates;
};

// CPU-side corpus capture shared by the collision audit and coordinate plots.
// Retains every fact/position pair, including masked prompt and padding rows.
struct PuzzleReportData {
  // Positive hidden dimension; every point has this many residual coordinates.
  unsigned int model_width = 0;
  // Positive number of input positions per fact, including padded positions.
  unsigned int context_length = 0;
  // Nonempty list of original sentences in dataset order, used as plot labels.
  std::vector<std::string> facts;
  // Exactly one point per (fact_index, position); vector order is unrestricted.
  std::vector<PuzzlePoint> points;
};

// Counts from a successful exact-vector collision audit over all scored rows.
// Equal full vectors must share a target; this does not prove MLP learnability.
struct PuzzleSeparation {
  // Number of rows with target_token >= 0, including continuation and EOS.
  int64_t scored_points = 0;
  // Distinct full residual vectors among scored rows, across facts/positions;
  // equality is exact numerically, with +0 and -0 treated as the same value.
  int64_t unique_vectors = 0;
  // Distinct expected token IDs among scored rows, not the vocabulary size.
  int64_t distinct_targets = 0;
};

// Checks finite coordinates, token metadata, and complete fact/position
// coverage without imposing any separation requirement on the hidden states.
absl::Status ValidatePuzzleReportData(const PuzzleReportData& data);

// Audits exact numerical equality of finite full-width vectors, treating +0
// and -0 as equal. All scored positions share the same comparison domain:
// identical vectors may have the same target but never different targets.
// Masked rows are validated but excluded; this does not prove that full-prefix
// next-token prediction is unambiguous, or that two-dimensional plots separate.
absl::StatusOr<PuzzleSeparation> VerifyPuzzleSeparation(
    const PuzzleReportData& data);

// Writes a standalone interactive HTML report of raw coordinate pairs. Checks
// complete coverage and recomputes the supplied audit before opening the file.
absl::Status WritePuzzleHtml(const PuzzleReportData& data,
                             const PuzzleSeparation& separation,
                             absl::string_view path);

}  // namespace pluto::llm::memorize_general_facts
