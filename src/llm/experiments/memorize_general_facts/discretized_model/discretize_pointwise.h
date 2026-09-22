#pragma once

#include <map>
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_model.h"

namespace pluto::llm::discretized::generator {

using TokenNames = std::map<int, std::string>;

// Renames symbols within boundaries to expose simple pointwise maps. No states
// are compacted and no layer is bypassed. Includes all original/new IDs in the
// returned model's state_relabeling field; the input remains unchanged.
absl::StatusOr<SymbolicModel> RelabelMlpOutputs(const SymbolicModel& model);

absl::StatusOr<std::optional<int>> EvaluatePointwise(
    absl::Span<const StateTransition> rows, int state);
absl::StatusOr<std::optional<int>> EvaluateEntry(
    absl::Span<const EntryTransition> rows, int token, int position);

// Chooses exact guarded affine maps, support masks, switches or narrow arrays.
// All absent inputs stay unsupported, including negative strong IDs.
absl::StatusOr<std::string> RenderPointwise(
    absl::string_view name, absl::Span<const StateTransition> rows,
    const TokenNames* token_names = nullptr);
absl::StatusOr<std::string> RenderEntry(absl::string_view name,
                                        absl::Span<const EntryTransition> rows,
                                        const TokenNames& token_names);

}  // namespace pluto::llm::discretized::generator
