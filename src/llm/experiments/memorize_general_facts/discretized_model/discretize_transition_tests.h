#pragma once

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_io.h"

namespace pluto::llm::discretized::generator {

// Independently replay source transitions and emit per-boundary gtest fixtures.
// These expected states are never linked into production inference.
absl::StatusOr<std::string> RenderTransitionTest(
    const Json& model, const std::vector<std::string>& token_names);

}  // namespace pluto::llm::discretized::generator
