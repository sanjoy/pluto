// Private declarations for the generated CLI and its tests; do not edit.
#pragma once
#include <iosfwd>

#include "src/llm/experiments/memorize_general_facts/discretized_model/runtime.h"

namespace pluto::llm::discretized::gen::internal {
// Encodes only captured text prefixes; independent of production inference.
absl::StatusOr<std::vector<DiscreteToken>> EncodeGeneratedPrompt(
    absl::string_view text);
// Checks complete corpus continuations using independently linked fixtures.
absl::Status VerifyGeneratedModel(const DiscreteModel& model,
                                  std::ostream& output);
}  // namespace pluto::llm::discretized::gen::internal
