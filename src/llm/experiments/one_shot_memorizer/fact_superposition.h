#pragma once

#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::llm::one_shot_memorizer {

enum class FactSuperposition { kSum, kMean };

// Combine independently trained updates from a shared initialization:
//   SUM  = initial + ((only_a - initial) + (only_b - initial))
//   MEAN = initial + 0.5 * ((only_a - initial) + (only_b - initial)).
// Every operation above uses FP64, followed by exactly one FP32 result cast.
// This is a fixed construction, not a fit: targets and joint weights are not
// inputs. All arrays must be nonempty, finite, and equally sized. Rejects a
// result outside finite FP32; representable underflow is allowed. The caller
// separately validates checkpoint shapes, mappings and matched provenance.
absl::StatusOr<std::vector<float>> SuperposeFactParameters(
    absl::Span<const float> initial, absl::Span<const float> only_a,
    absl::Span<const float> only_b, FactSuperposition method);

}  // namespace pluto::llm::one_shot_memorizer
