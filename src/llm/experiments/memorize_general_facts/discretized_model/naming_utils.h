#pragma once

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace pluto::llm::discretized::generator {

// Checks a generated transition function's unqualified ASCII identifier.
// Requires a leading letter and rejects C++ keywords, double underscores, and
// characters that could escape a declaration. Shared by all transition
// emitters.
absl::Status ValidateTransitionFunctionName(absl::string_view name);

}  // namespace pluto::llm::discretized::generator
