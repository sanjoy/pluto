#pragma once

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/serialized_cpp_program.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/utils.h"

namespace pluto::llm::discretized::generator {

// Lower one hidden-state map without crossing layer boundaries. Vocabulary
// names are supplied only for language-modeling-head outputs. Compact output
// chooses guarded affine maps, support masks, switches, or narrow arrays.
absl::StatusOr<SerializedCppProgram> RenderMap(
    const CapturedMap& map, absl::string_view name,
    const TokenNames* token_names = nullptr, bool compact = true);

}  // namespace pluto::llm::discretized::generator
