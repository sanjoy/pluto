#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "scripts/weight_analysis/autoregressive_probe.h"

namespace pluto::weight_analysis {

absl::Status ValidateGenerationOptions(int steps, double temperature,
                                       int seed) {
  if (steps < 0 || !std::isfinite(temperature) || temperature <= 0 ||
      seed < 0 || seed == std::numeric_limits<int>::max()) {
    return absl::InvalidArgumentError(
        "steps must be nonnegative, temperature finite/positive, and seed in "
        "0..INT_MAX-1 (production initializes mt19937 with seed+1)");
  }
  return absl::OkStatus();
}

absl::StatusOr<PredictionWindow> FillPredictionInput(
    absl::Span<const int> history, int context_length, int vocab_size,
    absl::Span<int> output) {
  if (history.empty() || context_length <= 0 || vocab_size <= 1 ||
      output.size() != static_cast<size_t>(context_length)) {
    return absl::InvalidArgumentError(
        "invalid prediction context or output shape");
  }
  for (int token : history) {
    if (token < 0 || token >= vocab_size) {
      return absl::InvalidArgumentError(
          "history contains an invalid logical token ID");
    }
  }
  const size_t length = std::min(history.size(), output.size());
  const size_t start = history.size() - length;
  std::copy(history.begin() + start, history.end(), output.begin());
  std::fill(output.begin() + length, output.end(), history.back());
  return PredictionWindow{start, length, length - 1};
}

absl::StatusOr<SamplingDecision> SampleProductionLogits(
    absl::Span<const float> logits, double temperature, std::mt19937& random) {
  if (logits.size() < 2 || logits.size() > std::numeric_limits<int>::max() ||
      !std::isfinite(temperature) || temperature <= 0) {
    return absl::InvalidArgumentError(
        "invalid logical logits or sampling temperature");
  }
  for (float value : logits) {
    if (!std::isfinite(value)) {
      return absl::InvalidArgumentError(
          "sampling requires finite logical logits");
    }
  }
  const auto maximum_position = std::max_element(logits.begin(), logits.end());
  const float maximum = *maximum_position;
  std::vector<double> weights(logits.size());
  for (size_t token = 0; token < logits.size(); ++token) {
    // Do NOT promote either operand before this subtraction. The production
    // expression subtracts two floats, then promotes for division by double.
    weights[token] = std::exp((logits[token] - maximum) / temperature);
  }
  std::discrete_distribution<int> sample(weights.begin(), weights.end());
  auto canonical_random = random;
  const double uniform = std::generate_canonical<double, 53>(canonical_random);
  auto words_random = random;
  const std::array<uint32_t, 2> words{static_cast<uint32_t>(words_random()),
                                      static_cast<uint32_t>(words_random())};
  const int selected = sample(random);
  if (random != canonical_random || random != words_random) {
    return absl::FailedPreconditionError(
        "standard discrete_distribution does not use the recorded canonical "
        "RNG draw");
  }
  auto probabilities = sample.probabilities();
  std::vector<double> cdf(probabilities.size());
  std::partial_sum(probabilities.begin(), probabilities.end(), cdf.begin());
  // libstdc++ explicitly fixes the final cumulative endpoint to exactly one.
  cdf.back() = 1.0;
  const int reconstructed = static_cast<int>(
      std::lower_bound(cdf.begin(), cdf.end(), uniform) - cdf.begin());
  if (selected < 0 || selected >= static_cast<int>(logits.size()) ||
      reconstructed != selected || !std::isfinite(uniform) || uniform < 0 ||
      uniform >= 1) {
    return absl::FailedPreconditionError(
        "recorded normalized CDF does not explain the actual sampled token");
  }
  int rank = 1;
  for (size_t token = 0; token < logits.size(); ++token) {
    if (logits[token] > logits[selected] ||
        (logits[token] == logits[selected] &&
         token < static_cast<size_t>(selected))) {
      ++rank;
    }
  }
  return SamplingDecision{selected,
                          logits[selected],
                          rank,
                          static_cast<int>(maximum_position - logits.begin()),
                          probabilities[selected],
                          uniform,
                          selected == 0 ? 0 : cdf[selected - 1],
                          cdf[selected],
                          words};
}

}  // namespace pluto::weight_analysis
