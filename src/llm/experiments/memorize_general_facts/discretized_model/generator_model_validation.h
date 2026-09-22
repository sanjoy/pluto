#pragma once

#include "absl/status/status.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_io.h"

namespace pluto::llm::discretized::generator::internal {
// Certificate validation only needs immutable tables; full validation also
// checks corpus and vocabulary metadata needed for emission and evaluation.
absl::Status ValidateTables(const Json& model, bool full);
}  // namespace pluto::llm::discretized::generator::internal
