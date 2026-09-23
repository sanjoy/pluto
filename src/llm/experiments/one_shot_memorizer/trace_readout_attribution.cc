#include "src/llm/experiments/one_shot_memorizer/trace_readout_attribution.h"

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

// Compensated accumulation makes small terms less sensitive to dimension
// ordering. This does not pretend that floating-point telescoping is exact;
// the remaining closure error is returned explicitly to the caller.
class Sum {
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

absl::Status ValidateVector(absl::Span<const float> values, size_t width) {
  if (values.size() != width)
    return absl::InvalidArgumentError(
        "readout attribution vector widths differ");
  for (float value : values)
    if (!std::isfinite(value))
      return absl::InvalidArgumentError(
          "readout attribution inputs must be finite");
  return absl::OkStatus();
}

absl::Status ValidateResult(const TraceReadoutAttribution& result) {
  for (double value :
       {result.final_mean, result.final_variance,
        result.normalization_denominator, result.beta_contribution,
        result.ideal_margin, result.accounted_margin,
        result.accounting_residual, result.actual_normalized_margin,
        result.normalization_residual})
    if (!std::isfinite(value))
      return absl::OutOfRangeError("readout attribution arithmetic overflowed");
  for (const auto* values : {&result.embedding_difference, &result.direction,
                             &result.actual_dimension_contributions})
    for (double value : *values)
      if (!std::isfinite(value))
        return absl::OutOfRangeError(
            "readout attribution coordinate overflowed");
  for (const auto& boundary : result.boundaries) {
    if (!std::isfinite(boundary.total))
      return absl::OutOfRangeError(
          "readout attribution boundary total overflowed");
    for (double value : boundary.dimension_contributions)
      if (!std::isfinite(value))
        return absl::OutOfRangeError(
            "readout attribution boundary coordinate overflowed");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<TraceReadoutAttribution> ComputeTraceReadoutAttribution(
    absl::Span<const absl::Span<const float>> residual_rows,
    absl::Span<const float> gamma, absl::Span<const float> beta,
    absl::Span<const float> target_embedding,
    absl::Span<const float> rival_embedding,
    absl::Span<const float> actual_normalized_output, double epsilon) {
  const size_t width = gamma.size();
  if (width == 0 || residual_rows.empty() || !std::isfinite(epsilon) ||
      epsilon <= 0)
    return absl::InvalidArgumentError(
        "readout attribution requires nonempty rows/width and positive finite "
        "epsilon");
  if (width > std::vector<double>().max_size() ||
      residual_rows.size() >
          std::vector<ResidualMarginContribution>().max_size())
    return absl::OutOfRangeError(
        "readout attribution dimensions exceed supported storage");
  for (const auto values : {gamma, beta, target_embedding, rival_embedding,
                            actual_normalized_output})
    RETURN_IF_ERROR(ValidateVector(values, width));
  for (const auto row : residual_rows)
    RETURN_IF_ERROR(ValidateVector(row, width));

  TraceReadoutAttribution result;
  const auto final_row = residual_rows.back();
  Sum mean;
  for (float value : final_row)
    mean.Add(value);
  result.final_mean = mean.value() / width;
  Sum variance;
  for (float value : final_row) {
    const double centered = static_cast<double>(value) - result.final_mean;
    variance.Add(centered * centered);
  }
  result.final_variance = variance.value() / width;
  result.normalization_denominator = std::sqrt(result.final_variance + epsilon);
  if (!std::isfinite(result.normalization_denominator) ||
      result.normalization_denominator <= 0)
    return absl::OutOfRangeError(
        "readout attribution normalization scale overflowed");

  result.embedding_difference.resize(width);
  result.direction.resize(width);
  result.actual_dimension_contributions.resize(width);
  Sum weighted_embedding_mean;
  Sum beta_margin;
  for (size_t dim = 0; dim < width; ++dim) {
    // Subtract in double: two finite FP32 values can have an FP32-overflowing
    // difference. BF16-expanded embeddings are a subset of these safe inputs.
    const double difference =
        static_cast<double>(target_embedding[dim]) - rival_embedding[dim];
    result.embedding_difference[dim] = difference;
    result.direction[dim] = gamma[dim] * difference;
    weighted_embedding_mean.Add(result.direction[dim]);
    beta_margin.Add(beta[dim] * difference);
  }
  const double direction_mean = weighted_embedding_mean.value() / width;
  for (double& value : result.direction)
    value = (value - direction_mean) / result.normalization_denominator;
  result.beta_contribution = beta_margin.value();

  // LayerNorm centers the final residual. Moving that centering to the
  // direction gives a dot product with each uncentered residual/delta. Holding
  // THIS final denominator fixed is what makes the following sum telescope;
  // re-normalizing each intermediate would instead be a different experiment.
  Sum accounted;
  for (size_t boundary = 0; boundary < residual_rows.size(); ++boundary) {
    ResidualMarginContribution contribution;
    contribution.dimension_contributions.resize(width);
    Sum total;
    for (size_t dim = 0; dim < width; ++dim) {
      double value = residual_rows[boundary][dim];
      if (boundary != 0)
        value -= static_cast<double>(residual_rows[boundary - 1][dim]);
      const double term = result.direction[dim] * value;
      contribution.dimension_contributions[dim] = term;
      total.Add(term);
    }
    contribution.total = total.value();
    accounted.Add(contribution.total);
    result.boundaries.push_back(std::move(contribution));
  }
  accounted.Add(result.beta_contribution);
  result.accounted_margin = accounted.value();

  Sum ideal;
  Sum actual;
  for (size_t dim = 0; dim < width; ++dim) {
    const double centered =
        static_cast<double>(final_row[dim]) - result.final_mean;
    const double normalized =
        centered / result.normalization_denominator * gamma[dim] + beta[dim];
    ideal.Add(normalized * result.embedding_difference[dim]);
    const double term = static_cast<double>(actual_normalized_output[dim]) *
                        result.embedding_difference[dim];
    result.actual_dimension_contributions[dim] = term;
    actual.Add(term);
  }
  result.ideal_margin = ideal.value();
  result.accounting_residual = result.accounted_margin - result.ideal_margin;
  result.actual_normalized_margin = actual.value();
  result.normalization_residual =
      result.actual_normalized_margin - result.ideal_margin;
  RETURN_IF_ERROR(ValidateResult(result));
  return result;
}

}  // namespace pluto::llm::one_shot_memorizer
