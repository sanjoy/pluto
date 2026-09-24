#pragma once

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace pluto::llm::discretized::generator {

// BF16 vectors indexed by hidden-state ID. During exact capture these preserve
// every original vector; CapturedStateVectors retains that immutable archive
// for inspection. When passed to a compactor, they only guide candidate
// ordering; transitions and validity are symbolic. A supplied hint map must
// cover every state in the model being compacted.
// IDs belong to that particular input model: Export(), CompactModel(), and
// RelabelMlpOutputs() can renumber states, so this map cannot be reused with
// their output without explicitly translating its keys to the new state IDs.
using StateVectorHints = absl::flat_hash_map<int, std::vector<uint16_t>>;

}  // namespace pluto::llm::discretized::generator
