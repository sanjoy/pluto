#pragma once

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/serialized_cpp_program.h"

namespace pluto::llm::discretized::generator {

// Lower one causal-history -> hidden-state mapping. Unknown histories remain
// unsupported. The default compact form emits independent MatchState<ID>
// predicates with flat ORs of exact history matches, grouped by length.
// Acceptance checks every element: uniqueness among recorded examples cannot
// validate unseen histories. Legacy hybrid/control_flow recognizers and plain
// sorted lookup remain available for equivalence tests and alternatives.
absl::StatusOr<SerializedCppProgram> RenderAttention(
    const CapturedCausalAttention& attention, absl::string_view name,
    bool compact = true, int chunk_size = 256,
    absl::string_view strategy = "state_matchers");

}  // namespace pluto::llm::discretized::generator
