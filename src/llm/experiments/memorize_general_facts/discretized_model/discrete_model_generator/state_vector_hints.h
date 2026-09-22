#pragma once

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace pluto::llm::discretized::generator {

// Optional capture-time BF16 representative vectors, indexed by hidden-state
// ID. They only guide candidate ordering; transitions and validity are
// symbolic. A supplied map must cover every state in the model being compacted.
// IDs belong to that particular input model: Export(), CompactModel(), and
// RelabelMlpOutputs() can renumber states, so this map cannot be reused with
// their output without explicitly translating its keys to the new state IDs.
using StateVectorHints = absl::flat_hash_map<int, std::vector<uint16_t>>;

}  // namespace pluto::llm::discretized::generator
