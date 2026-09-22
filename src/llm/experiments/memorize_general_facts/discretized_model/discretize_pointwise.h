#pragma once

#include <map>
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_attention_logic.h"

namespace pluto::llm::discretized::generator {

using TokenNames = std::map<int, std::string>;

// Renames symbols within boundaries to expose simple pointwise maps. No states
// are merged and no layer is bypassed. Includes all original/new IDs in the
// returned model's state_relabeling field; the input remains unchanged.
absl::StatusOr<Json> RelabelMlpOutputs(const Json& model);

absl::StatusOr<std::optional<int>> EvaluatePointwise(const Json& rows,
                                                     int state);
absl::StatusOr<std::optional<int>> EvaluateEntry(const Json& rows, int token,
                                                 int position);

// Chooses exact guarded affine maps, support masks, switches or narrow arrays.
// All absent inputs stay unsupported, including negative strong IDs.
absl::StatusOr<RenderedTransition> RenderPointwise(
    absl::string_view name, const Json& rows,
    const TokenNames* token_names = nullptr);
absl::StatusOr<RenderedTransition> RenderEntry(absl::string_view name,
                                               const Json& rows,
                                               const TokenNames& token_names);

}  // namespace pluto::llm::discretized::generator
