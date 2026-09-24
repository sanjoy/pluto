#pragma once

#include "absl/status/statusor.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model.h"

namespace pluto::llm::discretized::generator {

// Shares each bijective MLP's input/output symbol while keeping the MLP as an
// explicit identity map. Every MLP must cover both of its boundaries exactly;
// attention boundaries from different transformer blocks are never combined.
// Run after ordinary compaction and relabeling. Original memberships are kept
// so shared symbols can still be traced to vectors on both sides of the MLP.
// A fully paired model is returned unchanged; partial pairing is unsupported.
absl::StatusOr<CapturedModel> CompactMlpPairs(const CapturedModel& model);

}  // namespace pluto::llm::discretized::generator
