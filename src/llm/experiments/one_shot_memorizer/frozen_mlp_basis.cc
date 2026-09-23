#include "src/llm/experiments/one_shot_memorizer/frozen_mlp_basis.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

// Compensated sums keep small projection terms and centered squared deviations
// from being lost when the fitting rows or input coordinates differ in scale.
class CompensatedSum {
 public:
  void Add(double value) {
    const double next = sum_ + value;
    correction_ += std::abs(sum_) >= std::abs(value) ? (sum_ - next) + value
                                                     : (value - next) + sum_;
    sum_ = next;
  }

  double value() const { return sum_ + correction_; }

 private:
  double sum_ = 0;
  double correction_ = 0;
};

absl::StatusOr<float> ToCoefficient(double value, int feature) {
  if (!std::isfinite(value) ||
      std::abs(value) > std::numeric_limits<float>::max())
    return absl::OutOfRangeError(absl::StrCat(
        "standardized coefficient overflows FP32 at feature ", feature));
  const float result = static_cast<float>(value);
  if (value != 0 && result == 0)
    return absl::OutOfRangeError(absl::StrCat(
        "standardized coefficient underflows FP32 at feature ", feature));
  return result;
}

}  // namespace

absl::StatusOr<FrozenMlpBasis> StandardizeFrozenMlpBasis(
    absl::Span<const float> fitting_inputs,
    absl::Span<const float> initial_weights, int input_dim, int feature_width) {
  if (input_dim <= 0 || feature_width <= 0)
    return absl::InvalidArgumentError("basis dimensions must be positive");
  const size_t width = static_cast<size_t>(input_dim);
  const size_t features = static_cast<size_t>(feature_width);
  if (width > std::numeric_limits<size_t>::max() / features ||
      initial_weights.size() != width * features || fitting_inputs.empty() ||
      fitting_inputs.size() % width != 0)
    return absl::InvalidArgumentError(
        "inconsistent frozen-basis matrix shapes");
  for (float value : fitting_inputs)
    if (!std::isfinite(value))
      return absl::InvalidArgumentError("fitting inputs must be finite");
  for (float value : initial_weights)
    if (!std::isfinite(value))
      return absl::InvalidArgumentError("initial directions must be finite");

  const size_t rows = fitting_inputs.size() / width;
  FrozenMlpBasis result;
  result.weights.resize(initial_weights.size());
  result.bias.resize(features);
  result.projection_means.resize(features);
  result.projection_standard_deviations.resize(features);
  // Store just one feature's projections, not a rows-by-features matrix.
  std::vector<double> projections(rows);
  for (int feature = 0; feature < feature_width; ++feature) {
    CompensatedSum projection_sum;
    for (size_t row = 0; row < rows; ++row) {
      CompensatedSum dot;
      for (size_t coordinate = 0; coordinate < width; ++coordinate)
        dot.Add(static_cast<double>(fitting_inputs[row * width + coordinate]) *
                initial_weights[coordinate * features + feature]);
      projections[row] = dot.value();
      projection_sum.Add(projections[row]);
    }
    // Even a compensated sum followed by division can round a constant mean
    // away from its repeated value. Do not mistake that rounding for variance.
    bool varies = false;
    for (double projection : projections)
      varies |= projection != projections.front();
    if (!varies)
      return absl::InvalidArgumentError(
          absl::StrCat("constant fitting projection at feature ", feature));
    const double mean = projection_sum.value() / static_cast<double>(rows);
    // Center before squaring instead of subtracting two large raw moments.
    CompensatedSum squared_deviations;
    for (double projection : projections) {
      const double delta = projection - mean;
      squared_deviations.Add(delta * delta);
    }
    const double variance =
        squared_deviations.value() / static_cast<double>(rows);
    const double sigma = std::sqrt(variance);
    if (!std::isfinite(mean) || !std::isfinite(sigma))
      return absl::OutOfRangeError(
          absl::StrCat("non-finite fitting moments at feature ", feature));
    if (sigma == 0)
      return absl::InvalidArgumentError(
          absl::StrCat("constant fitting projection at feature ", feature));
    result.projection_means[feature] = mean;
    result.projection_standard_deviations[feature] = sigma;
    for (size_t coordinate = 0; coordinate < width; ++coordinate) {
      ASSIGN_OR_RETURN(
          result.weights[coordinate * features + feature],
          ToCoefficient(
              initial_weights[coordinate * features + feature] / sigma,
              feature));
    }
    ASSIGN_OR_RETURN(result.bias[feature],
                     ToCoefficient(-mean / sigma, feature));
  }
  return result;
}

}  // namespace pluto::llm::one_shot_memorizer
