#pragma once

#include <compare>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model.h"

// Test-only access to the attention compiler's intermediate recognizer.
// Model capture, compaction, and code-generation clients must use
// CapturedModel.
namespace pluto::llm::discretized::generator::internal {

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
  auto operator<=>(const AttentionProgram&) const = default;
};

absl::StatusOr<AttentionProgram> BuildAttention(
    absl::Span<const AttentionTransition> rows);
std::optional<int> EvaluateAttention(const AttentionProgram& program,
                                     absl::Span<const int> history);

}  // namespace pluto::llm::discretized::generator::internal
