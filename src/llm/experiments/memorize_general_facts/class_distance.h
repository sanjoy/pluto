#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/llm/experiments/memorize_general_facts/puzzle_report.h"

namespace pluto::llm::memorize_general_facts {

// The closest two observed states from one unordered pair of target classes.
// Witness indices refer to the supplied PuzzleReportData::points vector.
struct ClassDistancePair {
  int first_token;      // Smaller target ID; not necessarily contiguous.
  int second_token;     // Larger target ID, always different from first_token.
  double distance;      // Minimum Euclidean distance, not squared distance.
  size_t first_point;   // A state whose target is first_token.
  size_t second_point;  // A state whose target is second_token.
};

// A complete finite-corpus comparison of distinct scored output-token classes.
// Every class pair contributes once regardless of either class's frequency.
struct ClassDistanceAnalysis {
  int64_t scored_points = 0;   // Suffix and EOS rows, excluding masked rows.
  int64_t target_classes = 0;  // Observed target IDs, not vocabulary size.
  // Exactly target_classes choose 2 entries, sorted by the pair of token IDs.
  std::vector<ClassDistancePair> pairs;
};

// Exact exhaustive search over scored vectors across all facts and positions.
// Uses direct FP64 coordinate differences, without normalizing or projecting.
// Zero distances are valid (cross-class collisions); fewer than two classes
// produce no pairs. Complexity is O(scored_points^2 * model_width), with one
// distance calculation per cross-class state pair and O(target_classes^2)
// output memory. Tied witnesses use the first pair in capture order.
absl::StatusOr<ClassDistanceAnalysis> ComputeClassDistances(
    const PuzzleReportData& data);

// Computes the analysis and writes standalone linear/log-distance histograms,
// quantiles, and the 50 closest class pairs with their witness facts/positions.
// Labels are indexed by model token ID; an empty span prints numeric IDs only.
// This is a descriptive finite-corpus statistic, not a separability proof.
absl::Status WriteClassDistanceHtml(const PuzzleReportData& data,
                                    absl::Span<const std::string> token_labels,
                                    absl::string_view path);

}  // namespace pluto::llm::memorize_general_facts
