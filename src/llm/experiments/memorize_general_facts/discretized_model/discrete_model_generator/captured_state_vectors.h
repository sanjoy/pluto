#pragma once

#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/state_vector_hints.h"

namespace pluto::llm::discretized::generator {

// Immutable provenance from exact capture, kept separate from the symbolic
// model. Compaction unions CapturedState::members rather than copying vectors;
// code generation resolves those original IDs here to emit the complete list.
// Repeated corpus occurrences of the same exact vector at one boundary share
// an original state, but numerically distinct vectors are never averaged away.
struct CapturedStateVectors {
  // Original (pre-compaction/pre-relabeling) state ID -> exact BF16 channels.
  // These keys never change, even when the current model's state IDs do.
  StateVectorHints original_states;
};

}  // namespace pluto::llm::discretized::generator
