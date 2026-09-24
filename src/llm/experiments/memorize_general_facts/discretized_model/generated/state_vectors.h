// Private generated vector-inspection API; do not edit.
#pragma once
#include <cstdint>
#include <iosfwd>

#include "src/llm/experiments/memorize_general_facts/discretized_model/runtime.h"
namespace pluto::llm::discretized::gen::internal {
// Immutable original-vector rows, backed by static generated arrays.
struct StateVectorShard {
  absl::Span<const int> original_ids;  // One capture-state ID per row.
  absl::Span<const int>
      original_boundaries;           // Original residual stage per row.
  absl::Span<const uint16_t> words;  // Row-major exact BF16 words.
};
// Prints every original activation in one compacted hidden state.
// Unknown IDs fail without output; formatting on output is preserved.
absl::Status PrintState(DiscreteHiddenState state, std::ostream& output);
StateVectorShard GeneratedStateVectorShard0();
StateVectorShard GeneratedStateVectorShard1();
StateVectorShard GeneratedStateVectorShard2();
StateVectorShard GeneratedStateVectorShard3();
StateVectorShard GeneratedStateVectorShard4();
StateVectorShard GeneratedStateVectorShard5();
StateVectorShard GeneratedStateVectorShard6();
StateVectorShard GeneratedStateVectorShard7();
StateVectorShard GeneratedStateVectorShard8();
StateVectorShard GeneratedStateVectorShard9();
StateVectorShard GeneratedStateVectorShard10();
StateVectorShard GeneratedStateVectorShard11();
StateVectorShard GeneratedStateVectorShard12();
StateVectorShard GeneratedStateVectorShard13();
}  // namespace pluto::llm::discretized::gen::internal
