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

// An ordered comparison of three observed FP32 endpoints. These vectors
// describe parameter changes, not exclusive causal ownership or information.
struct FactSuperpositionDecomposition {
  std::vector<double> total;                // joint - sum.
  std::vector<double> gradient_trajectory;  // joint - frozen.
  std::vector<double> optimizer_history;    // frozen - sum.
  // Largest |total - (gradient_trajectory + optimizer_history)| in FP64.
  double maximum_absolute_closure_error = 0;
};

// Compare an actual joint endpoint, a joint replay with fixed gradients, and
// the SUM of separately replayed component updates. Subtraction/addition use
// FP64, with no output cast. The telescoping identity is exact over the reals;
// widely separated FP32 exponents can still produce nonzero FP64 closure error,
// which is reported rather than rejected. Component norms can cancel and must
// not be interpreted as fractions of the total effect. The first component
// includes any change in the recomputed gradients, including changes caused by
// BF16 rounding boundaries, not just smooth adaptation of learned features.
// Requires equally sized, nonempty, finite arrays. Replay provenance and
// parameter layout are the caller's responsibility.
absl::StatusOr<FactSuperpositionDecomposition> DecomposeFactSuperposition(
    absl::Span<const float> joint, absl::Span<const float> frozen,
    absl::Span<const float> sum);

}  // namespace pluto::llm::one_shot_memorizer
