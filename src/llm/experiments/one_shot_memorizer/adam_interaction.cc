#include "src/llm/experiments/one_shot_memorizer/adam_interaction.h"

#include <cmath>

#include "absl/status/status.h"

namespace pluto::llm::one_shot_memorizer {

absl::StatusOr<TwoStepAdamInteraction> ExplainTwoStepAdamInteraction(
    float a, float b, const TwoStepAdamConfig& config) {
  if (!std::isfinite(a) || !std::isfinite(b) ||
      !(config.beta1 >= 0 && config.beta1 < 1) ||
      !(config.beta2 >= 0 && config.beta2 < 1) || !(config.epsilon > 0) ||
      !std::isfinite(config.epsilon) || !(config.second_rate > 0) ||
      !std::isfinite(config.second_rate))
    return absl::InvalidArgumentError("invalid two-step Adam inputs");

  // At t=2, cancel (1-beta) against the bias correction (1-beta^2).
  // The first moment is linear: the shared numerator equals x + y. Only
  // the history-dependent denominators prevent the separate updates adding.
  const double x = config.beta1 * a / (1 + config.beta1);
  const double y = b / (1 + config.beta1);
  const double rms_a = std::sqrt(config.beta2 / (1 + config.beta2)) *
                       std::abs(static_cast<double>(a));
  const double rms_b =
      std::abs(static_cast<double>(b)) / std::sqrt(1 + config.beta2);
  TwoStepAdamInteraction result;
  result.a_denominator = rms_a + config.epsilon;
  result.b_denominator = rms_b + config.epsilon;
  result.joint_denominator = std::hypot(rms_a, rms_b) + config.epsilon;
  // Quotient differences avoid forming 1/epsilon before multiplying a zero
  // numerator. The sign convention follows weights, which SUBTRACT updates.
  result.a_term = config.second_rate *
                  (x / result.a_denominator - x / result.joint_denominator);
  result.b_term = config.second_rate *
                  (y / result.b_denominator - y / result.joint_denominator);
  result.difference = result.a_term + result.b_term;
  if (!std::isfinite(result.a_denominator) ||
      !std::isfinite(result.b_denominator) ||
      !std::isfinite(result.joint_denominator) ||
      !std::isfinite(result.a_term) || !std::isfinite(result.b_term) ||
      !std::isfinite(result.difference))
    return absl::OutOfRangeError("two-step Adam result exceeds finite FP64");
  return result;
}

}  // namespace pluto::llm::one_shot_memorizer
