#include "src/llm/experiments/one_shot_memorizer/attention_factorial.h"

#include <cmath>
#include <limits>

#include "absl/status/status.h"

namespace pluto::llm::one_shot_memorizer {

absl::StatusOr<AttentionFactorialDecomposition> DecomposeAttentionFactorial(
    const AttentionFactorialCorners& corners) {
  const size_t width = corners.front().size();
  if (width == 0)
    return absl::InvalidArgumentError("attention corners must be nonempty");
  for (const auto corner : corners) {
    if (corner.size() != width)
      return absl::InvalidArgumentError("attention corner widths differ");
    for (double value : corner)
      if (!std::isfinite(value))
        return absl::InvalidArgumentError("nonfinite attention corner");
  }
  AttentionFactorialDecomposition result;
  result.total.resize(width);
  result.routing.resize(width);
  result.values.resize(width);
  result.interaction.resize(width);
  for (size_t channel = 0; channel < width; ++channel) {
    result.total[channel] = corners[3][channel] - corners[0][channel];
    result.routing[channel] = corners[1][channel] - corners[0][channel];
    result.values[channel] = corners[2][channel] - corners[0][channel];
    result.interaction[channel] = corners[3][channel] - corners[1][channel] -
                                  corners[2][channel] + corners[0][channel];
    if (!std::isfinite(result.total[channel]) ||
        !std::isfinite(result.routing[channel]) ||
        !std::isfinite(result.values[channel]) ||
        !std::isfinite(result.interaction[channel]))
      return absl::OutOfRangeError("attention finite difference overflowed");
  }
  return result;
}

absl::StatusOr<IdealAttentionDecomposition> DecomposeIdealAttention(
    absl::Span<const float> recipient_probabilities,
    absl::Span<const float> donor_probabilities,
    absl::Span<const float> recipient_values,
    absl::Span<const float> donor_values, size_t width,
    size_t shared_prefix_length) {
  const size_t rows = recipient_probabilities.size();
  if (rows == 0 || donor_probabilities.size() != rows || width == 0 ||
      shared_prefix_length > rows ||
      rows > std::numeric_limits<size_t>::max() / width ||
      recipient_values.size() != rows * width ||
      donor_values.size() != recipient_values.size())
    return absl::InvalidArgumentError("invalid ideal attention shapes");
  for (const auto probabilities :
       {recipient_probabilities, donor_probabilities}) {
    double sum = 0;
    for (float probability : probabilities) {
      if (!std::isfinite(probability) || probability < 0 || probability > 1)
        return absl::InvalidArgumentError("invalid attention probability");
      sum += probability;
    }
    if (std::abs(sum - 1) > 1e-4)
      return absl::InvalidArgumentError(
          "attention probabilities do not sum to one");
  }
  for (size_t i = 0; i < recipient_values.size(); ++i) {
    if (!std::isfinite(recipient_values[i]) || !std::isfinite(donor_values[i]))
      return absl::InvalidArgumentError("nonfinite attention value");
    if (i < shared_prefix_length * width &&
        recipient_values[i] != donor_values[i])
      return absl::InvalidArgumentError("shared attention value rows differ");
  }
  IdealAttentionDecomposition result;
  result.routing.resize(width);
  result.values.resize(width);
  result.interaction.resize(width);
  result.shared_prefix_routing.resize(width);
  // Accumulate in channel/key order, retaining the reported float probability
  // values exactly. FlashAttention multiplies unnormalized probabilities and
  // divides later; this ideal calculation deliberately does not claim to
  // reproduce that operation's finite-precision rounding.
  for (size_t channel = 0; channel < width; ++channel)
    for (size_t key = 0; key < rows; ++key) {
      const double p = recipient_probabilities[key];
      const double delta_p = static_cast<double>(donor_probabilities[key]) - p;
      const double v = recipient_values[key * width + channel];
      const double delta_v =
          static_cast<double>(donor_values[key * width + channel]) - v;
      result.routing[channel] += delta_p * v;
      result.values[channel] += p * delta_v;
      result.interaction[channel] += delta_p * delta_v;
      if (key < shared_prefix_length)
        result.shared_prefix_routing[channel] += delta_p * v;
    }
  return result;
}

}  // namespace pluto::llm::one_shot_memorizer
