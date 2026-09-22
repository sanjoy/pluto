#pragma once

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/code_generator.h"

namespace pluto::llm::discretized::generator {

// Lowers each boundary independently and installs its translation unit.
// Compact controls implementation choices, not the captured state partition.
absl::Status RenderLayerSources(const CapturedModel& model,
                                const std::vector<std::string>& token_names,
                                bool compact, FileMap& files);

}  // namespace pluto::llm::discretized::generator
