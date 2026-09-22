#pragma once

#include "absl/status/statusor.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generate_discretized_model.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_model.h"

namespace pluto::llm::discretized::generator {

// Captures only real corpus positions and interns exact native BF16 states.
// No trace file is created: samples pass directly to the symbolic model
// builder.
absl::StatusOr<SymbolicModel> CaptureCheckpoint(
    const GeneratorOptions& options);

}  // namespace pluto::llm::discretized::generator
