#include "src/llm/experiments/one_shot_memorizer/fact_superposition.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "absl/status/status.h"

namespace pluto::llm::one_shot_memorizer {

absl::StatusOr<std::vector<float>> SuperposeFactParameters(
    absl::Span<const float> initial, absl::Span<const float> only_a,
    absl::Span<const float> only_b, FactSuperposition method) {
  if (initial.empty() || only_a.size() != initial.size() ||
      only_b.size() != initial.size())
    return absl::InvalidArgumentError(
        "superposition requires equal nonempty arrays");
  if (method != FactSuperposition::kSum && method != FactSuperposition::kMean)
    return absl::InvalidArgumentError("unknown fact superposition method");
  const double scale = method == FactSuperposition::kSum ? 1.0 : 0.5;
  std::vector<float> result(initial.size());
  for (size_t i = 0; i < initial.size(); ++i) {
    if (!std::isfinite(initial[i]) || !std::isfinite(only_a[i]) ||
        !std::isfinite(only_b[i]))
      return absl::InvalidArgumentError(
          "superposition parameters must be finite");
    const double base = initial[i];
    const double delta_a = static_cast<double>(only_a[i]) - base;
    const double delta_b = static_cast<double>(only_b[i]) - base;
    const double value = base + scale * (delta_a + delta_b);
    if (!std::isfinite(value) ||
        std::abs(value) > std::numeric_limits<float>::max())
      return absl::OutOfRangeError("superposition exceeds finite FP32");
    result[i] = static_cast<float>(value);
  }
  return result;
}

absl::StatusOr<FactSuperpositionDecomposition> DecomposeFactSuperposition(
    absl::Span<const float> joint, absl::Span<const float> frozen,
    absl::Span<const float> sum) {
  if (joint.empty() || frozen.size() != joint.size() ||
      sum.size() != joint.size())
    return absl::InvalidArgumentError(
        "decomposition requires equal nonempty arrays");
  FactSuperpositionDecomposition result;
  result.total.resize(joint.size());
  result.gradient_trajectory.resize(joint.size());
  result.optimizer_history.resize(joint.size());
  for (size_t i = 0; i < joint.size(); ++i) {
    if (!std::isfinite(joint[i]) || !std::isfinite(frozen[i]) ||
        !std::isfinite(sum[i]))
      return absl::InvalidArgumentError(
          "decomposition parameters must be finite");
    const double actual = joint[i], fixed = frozen[i], additive = sum[i];
    result.total[i] = actual - additive;
    result.gradient_trajectory[i] = actual - fixed;
    result.optimizer_history[i] = fixed - additive;
    const double closure = result.total[i] - (result.gradient_trajectory[i] +
                                              result.optimizer_history[i]);
    if (!std::isfinite(result.total[i]) ||
        !std::isfinite(result.gradient_trajectory[i]) ||
        !std::isfinite(result.optimizer_history[i]) || !std::isfinite(closure))
      return absl::OutOfRangeError("decomposition exceeds finite FP64");
    result.maximum_absolute_closure_error =
        std::max(result.maximum_absolute_closure_error, std::abs(closure));
  }
  return result;
}

}  // namespace pluto::llm::one_shot_memorizer
