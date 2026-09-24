#pragma once

#include <array>
#include <compare>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace pluto::llm::discretized::generator {

// Diagnostics and provenance are independent of the model's transition data.
struct VerificationResult {
  int64_t samples = 0;
  int64_t targets = 0;
  int64_t errors = 0;
  int64_t explicit_eos = 0;
  auto operator<=>(const VerificationResult&) const = default;
};

struct CompactionRecord {
  int boundary = 0;
  std::array<int, 2> seed_ids{};
  // Present only when optional representative-vector hints were supplied.
  std::optional<double> euclidean_distance;
  int64_t induced_compactions = 0;
  auto operator<=>(const CompactionRecord&) const = default;
};

enum class CompactionPhase {
  kNearest,
  kNearestPassComplete,
  kPointwise,
  kPointwisePassComplete,
  kExhaustive,
  kExhaustivePassComplete,
};

struct CompactionProgress {
  CompactionPhase phase = CompactionPhase::kNearest;
  int pass = 0;
  int64_t states = 0;
  std::vector<int> states_per_stage;
  int64_t attempted = 0;
  int64_t accepted = 0;
  int64_t compactions = 0;
  double seconds = 0;
  auto operator<=>(const CompactionProgress&) const = default;
};

enum class CompactionStoppingReason {
  kPassLimit,
  kAttemptLimit,
  kNearestCandidatesExhausted,
  kNoCompatiblePair,
};

struct CompactionSearchStatistics {
  CompactionStoppingReason stopping_reason =
      CompactionStoppingReason::kPassLimit;
  bool pairwise_compaction_complete = false;
  bool global_minimum_proven = false;
  int nearest_neighbors = 0;
  int64_t exhaustive_pair_limit = 0;
  int64_t remaining_pairs_before_sweep = 0;
  std::vector<CompactionProgress> history;
  double seconds = 0;
  auto operator<=>(const CompactionSearchStatistics&) const = default;
};

struct RelabelStatistics {
  std::vector<int> eligible_layers;
  std::vector<int> skipped_layers;
  int64_t changed_states = 0;
  bool layer_boundaries_preserved = true;
  bool vocabulary_aligned_final_boundaries = false;
  std::optional<int> last_attention_base;
  std::optional<int> final_state_base;
  std::optional<int> reserved_range_size;
  auto operator<=>(const RelabelStatistics&) const = default;
};

struct ModelStatistics {
  int64_t captured_samples = 0;
  int64_t scored_targets = 0;
  int64_t exact_states = 0;
  int64_t states = 0;
  bool membership_complete = false;
  std::optional<int64_t> membership_original_states;
  std::vector<int> states_per_stage;
  int64_t attempted_seeds = 0;
  int64_t accepted_seeds = 0;
  int64_t state_compactions = 0;
  int64_t cached_rejections = 0;
  std::vector<CompactionRecord> accepted_compactions;
  std::optional<VerificationResult> verification;
  std::optional<CompactionSearchStatistics> compaction_search;
  std::optional<RelabelStatistics> pointwise_relabeling;
  auto operator<=>(const ModelStatistics&) const = default;
};

struct CaptureProgress {
  int64_t samples = 0;
  int64_t total_samples = 0;
  int64_t rows = 0;
  bool greedy_verified = false;
};

enum class GenerationPhase { kBaseline, kGenerated };
struct GenerationProgress {
  GenerationPhase phase = GenerationPhase::kBaseline;
  int64_t states = 0;
  VerificationResult verification;
  std::string output;
};

using ProgressEvent =
    std::variant<CaptureProgress, CompactionProgress, GenerationProgress>;

}  // namespace pluto::llm::discretized::generator
