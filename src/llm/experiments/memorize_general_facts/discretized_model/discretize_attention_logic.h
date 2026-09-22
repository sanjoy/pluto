#pragma once

#include <compare>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_model.h"

namespace pluto::llm::discretized::generator {

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

// Emits exact literal tests, sharing suffix code and splitting helpers to keep
// compiler control-flow graphs bounded. Hybrid emission also shares long runs
// of literal symbol/output pairs; control_flow uses only branches.
absl::StatusOr<std::string> RenderAttention(
    absl::string_view name, absl::Span<const AttentionTransition> rows,
    int chunk_size = 256, absl::string_view strategy = "hybrid");

}  // namespace pluto::llm::discretized::generator
