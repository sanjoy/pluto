#pragma once

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/serialized_cpp_program.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/utils.h"

namespace pluto::llm::discretized::generator {

// Lower the observed (token, absolute position) -> hidden-state mapping.
// Every unobserved pair remains unsupported, with either representation.
absl::StatusOr<SerializedCppProgram> RenderPositionEmbedding(
    const CapturedPositionEmbedding& embedding, absl::string_view name,
    const TokenNames& token_names, bool compact = true);

}  // namespace pluto::llm::discretized::generator
