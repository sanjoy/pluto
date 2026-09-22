#pragma once

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/serialized_cpp_program.h"

namespace pluto::llm::discretized::generator {

// Lower one causal-history -> hidden-state mapping. Unknown histories remain
// unsupported. Compact output shares suffix control flow only within this
// layer; the alternative emits sorted rows with exact binary search.
absl::StatusOr<SerializedCppProgram> RenderAttention(
    const CapturedCausalAttention& attention, absl::string_view name,
    bool compact = true, int chunk_size = 256,
    absl::string_view strategy = "hybrid");

}  // namespace pluto::llm::discretized::generator
