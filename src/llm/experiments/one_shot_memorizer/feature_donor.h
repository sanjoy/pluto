#pragma once

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/experiments/one_shot_memorizer/feature_subset.h"

namespace pluto::llm::one_shot_memorizer {

// What happens to channels outside the donor subset. Neither option changes
// the recipient's separately supplied residual stream or any model weights.
enum class FeatureDonorBackground { kSparse, kIntact };

// Constructs one counterfactual physical BF16 row. Selected channels come from
// donor; unselected channels are +0 for kSparse, or recipient for kIntact.
// Copies raw bits, preserving retained signed zeros, NaNs and infinities, and
// never changes either source. Widths must match and be in [1,64]; rejects mask
// bits outside the row and unknown background values.
absl::StatusOr<std::vector<uint16_t>> MakeFeatureDonorRow(
    absl::Span<const uint16_t> recipient, absl::Span<const uint16_t> donor,
    FeatureSubset subset, FeatureDonorBackground background);

}  // namespace pluto::llm::one_shot_memorizer
