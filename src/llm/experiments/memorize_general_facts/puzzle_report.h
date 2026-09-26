#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace pluto::llm::memorize_general_facts {

struct PuzzlePoint {
  int fact_index = -1;
  int position = -1;
  int input_token = -1;
  // -1 denotes a masked prompt or padding row, excluded from the audit.
  int target_token = -1;
  bool padding = false;
  // Actual residual coordinates after the third attention layer (A3), before
  // any MLP. BF16 activations are expanded exactly to float by the caller.
  std::vector<float> coordinates;
};

struct PuzzleReportData {
  int model_width = 0;
  int context_length = 0;
  std::vector<std::string> facts;
  // Exactly one point for each (fact_index, position), including padding.
  std::vector<PuzzlePoint> points;
};

struct PuzzleSeparation {
  int64_t scored_points = 0;
  int64_t unique_vectors = 0;
  int64_t distinct_targets = 0;
};

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
