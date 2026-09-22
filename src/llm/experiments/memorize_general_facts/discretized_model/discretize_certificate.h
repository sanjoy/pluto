#pragma once

#include "absl/status/statusor.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_io.h"

namespace pluto::llm::discretized::generator {

// Independent backward collision proof over immutable transition tables. It
// reads no reducer state, search history, or rejection cache. "Inconclusive"
// is not failure: this sufficient argument need not prove every irreducible
// model. A successful proof is pairwise, not global-partition minimality.
absl::StatusOr<Json> CertifyModel(const Json& model);

}  // namespace pluto::llm::discretized::generator
