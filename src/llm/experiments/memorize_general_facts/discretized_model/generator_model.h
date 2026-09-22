#pragma once

#include <array>
#include <compare>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace pluto::llm::discretized::generator {

// Original tokenizer identity and exact bytes for one compact vocabulary ID.
// Its index in ModelMetadata::vocabulary is the compact ID.
struct VocabularyToken {
  int original_id = 0;
  std::string bytes;
  auto operator<=>(const VocabularyToken&) const = default;
};

struct ModelMetadata {
  int width = 0;   // Number of BF16 channels in each captured residual vector.
  int layers = 0;  // Number of attention/MLP transformer blocks.
  int vocab_size = 0;
  int eos_token = -1;
  int prompt_tokens = 0;  // Initial tokens supplied to autonomous verification.
  std::vector<VocabularyToken> vocabulary;
  auto operator<=>(const ModelMetadata&) const = default;
};

// Host-side observation of a single native execution. Boundary vectors are
// indexed [residual boundary][real token position][channel], without padding.
struct ExecutionSample {
  std::vector<int> tokens;
  std::vector<int> predictions;
  std::vector<std::vector<std::vector<uint16_t>>> boundaries;
  auto operator<=>(const ExecutionSample&) const = default;
};

struct Sample {
  std::vector<int> tokens;  // Complete fact, excluding the final predicted EOS.
  auto operator<=>(const Sample&) const = default;
};

struct SymbolicState {
  int id = 0;        // Globally unique hidden-state ID, above vocabulary IDs.
  int boundary = 0;  // Entry=0; block l attention=2*l+1, MLP=2*l+2.
  std::vector<uint16_t> bits;  // Representative's exact BF16 storage bits.
  // Original IDs in this equivalence class; absent when no map is available.
  std::optional<std::vector<int>> members;
  auto operator<=>(const SymbolicState&) const = default;
};

struct EntryTransition {
  int token = 0;
  int position = 0;
  int output = 0;
  auto operator<=>(const EntryTransition&) const = default;
};

struct AttentionTransition {
  std::vector<int> prefix;  // Entire ordered causal input-state history.
  int output = 0;
  auto operator<=>(const AttentionTransition&) const = default;
};

// Pointwise state mapping. For the language modeling head, output is a compact
// vocabulary ID; for an MLP, it is a hidden-state ID at the next boundary.
struct StateTransition {
  int input = 0;
  int output = 0;
  auto operator<=>(const StateTransition&) const = default;
};

struct TransformerTransitions {
  std::vector<AttentionTransition> attention;
  std::vector<StateTransition> mlp;
  auto operator<=>(const TransformerTransitions&) const = default;
};

struct StateRelabeling {
  int old_id = 0;
  int new_id = 0;
  int boundary = 0;
  auto operator<=>(const StateRelabeling&) const = default;
};

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
  double euclidean_distance =
      0;  // Distance between the chosen representatives.
  int64_t induced_compactions =
      0;  // Further compactions forced by this seed compaction.
  auto operator<=>(const CompactionRecord&) const = default;
};

enum class CompactionPhase {
  kNearest,
  kNearestPassComplete,
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

// Finite transition system consumed by compaction and code generation. The
// expected corpus is verification-only and never consulted by a transition.
struct SymbolicModel {
  ModelMetadata metadata;
  std::vector<SymbolicState> states;
  std::vector<Sample> samples;
  std::vector<EntryTransition> entry;
  std::vector<TransformerTransitions> transformers;
  std::vector<StateTransition> language_modeling_head;
  std::vector<StateRelabeling> state_relabeling;
  ModelStatistics stats;
  auto operator<=>(const SymbolicModel&) const = default;
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
  std::string output;  // Set only after publishing the generated directory.
};

using ProgressEvent =
    std::variant<CaptureProgress, CompactionProgress, GenerationProgress>;

}  // namespace pluto::llm::discretized::generator
