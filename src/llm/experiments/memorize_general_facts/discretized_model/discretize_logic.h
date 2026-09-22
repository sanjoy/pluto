#pragma once

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_emit.h"

namespace pluto::llm::discretized::generator {

// Replaces each boundary's plain implementation with an exact compact program.
// Does not compaction symbols or simplify across transformer boundaries.
absl::Status RenderCompact(const SymbolicModel& model,
                           const std::vector<std::string>& token_names,
                           FileMap& files);

}  // namespace pluto::llm::discretized::generator
