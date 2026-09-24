#pragma once

#include <map>
#include <string>

#include "absl/status/statusor.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_state_vectors.h"

namespace pluto::llm::discretized::generator {

// Emits an inspection-only CPU printer and sharded, exact BF16 vector data.
// Every state must name its original members, and the archive must cover those
// members exactly. The generated printer never participates in inference.
absl::StatusOr<std::map<std::string, std::string>> RenderStateVectors(
    const CapturedModel& model, const CapturedStateVectors& vectors);

}  // namespace pluto::llm::discretized::generator
