#pragma once

#include <compare>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_model.h"

namespace pluto::llm::discretized::generator {

enum class TransitionRepresentation {
  kEmpty,
  kGuardedAffine,
  kSparseAffineSupportMask,
  kAffineRangesAndSwitch,
  kGuardedOutputArray,
  kExactTokenPositionSwitch,
  kPackedSupportPatternsAndExceptions,
  kSharedSuffixControlFlow,
};

enum class AttentionStrategy { kHybrid, kControlFlow };

// Measurements of a generated transition. Optional measurements apply only
// to the named representation; absence is distinct from a measured zero.
struct TransitionStatistics {
  TransitionRepresentation representation = TransitionRepresentation::kEmpty;
  size_t rows = 0;
  size_t source_bytes = 0;
  std::optional<AttentionStrategy> strategy;
  std::optional<int64_t> table_bytes;
  std::optional<int64_t> affine_ranges;
  std::optional<int64_t> switch_cases;
  std::optional<int64_t> supported_span;
  std::optional<bool> named_anchor;
  std::optional<bool> named_outputs;
  std::optional<int64_t> tokens;
  std::optional<int64_t> patterns;
  std::optional<int64_t> position_bits;
  std::optional<int64_t> token_only_defaults;
  std::optional<int64_t> position_exceptions;
  std::optional<bool> named_exception_tokens;
  std::optional<int64_t> flat_key_scalars;
  std::optional<int64_t> flat_scalars;
  std::optional<int64_t> trie_nodes;
  std::optional<int64_t> nodes;
  std::optional<int64_t> edges;
  std::optional<int64_t> unary_nodes;
  std::optional<int64_t> branch_nodes;
  std::optional<int64_t> branch_edges;
  std::optional<int64_t> control_blocks;
  std::optional<int64_t> helpers;
  std::optional<int64_t> helper_node_limit;
  std::optional<int64_t> entry_cases;
  std::optional<int64_t> scalar_estimate;
  std::optional<int64_t> literal_sequence_patterns;
  std::optional<int64_t> literal_sequence_steps;
  std::optional<int64_t> literal_sequence_calls;
  std::optional<int64_t> literal_sequence_word_bits;
  auto operator<=>(const TransitionStatistics&) const = default;
};

// A complete generated C++ function and measurements of its representation.
struct RenderedTransition {
  std::string source;
  TransitionStatistics stats;
};

// One shared suffix program. A missing output rejects a history ending here;
// edges consume one more input symbol, in increasing symbol order.
struct AttentionNode {
  std::optional<int> output;
  std::vector<std::pair<int, int>> edges;
  auto operator<=>(const AttentionNode&) const = default;
};

// Minimal acyclic recognizer for a finite ordered-history -> output mapping.
// Nodes are control locations, never additional residual-state classes.
struct AttentionProgram {
  std::vector<AttentionNode> nodes;
  int root;
  size_t rows;
  size_t key_scalars;
  size_t trie_nodes;
  auto operator<=>(const AttentionProgram&) const = default;
};

// Shared by all transition emitters; rejects C++ keywords, reserved
// identifiers, and text that could escape a function declaration.
absl::Status ValidateTransitionFunctionName(absl::string_view name);

absl::StatusOr<AttentionProgram> BuildAttention(
    absl::Span<const AttentionTransition> rows);
std::optional<int> EvaluateAttention(const AttentionProgram& program,
                                     absl::Span<const int> history);

// Emits exact literal tests, sharing suffix code and splitting helpers to keep
// compiler control-flow graphs bounded. Hybrid emission also shares long runs
// of literal symbol/output pairs; control_flow uses only branches.
absl::StatusOr<RenderedTransition> RenderAttention(
    absl::string_view name, absl::Span<const AttentionTransition> rows,
    int chunk_size = 256, absl::string_view strategy = "hybrid");

}  // namespace pluto::llm::discretized::generator
