#include "src/llm/sampling.h"

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "absl/status/status.h"
#include "src/util/status_macros.h"

namespace pluto::llm {

absl::Status ValidateGenerationOptions(int generation_tokens,
                                       double temperature) {
  if (generation_tokens < 0 || !std::isfinite(temperature) ||
      temperature < 0.0) {
    return absl::InvalidArgumentError(
        "generation_tokens must be non-negative and temperature finite and "
        "non-negative (zero selects greedy decoding)");
  }
  return absl::OkStatus();
}

absl::StatusOr<int> SelectNextToken(absl::Span<const float> logits,
                                    double temperature, std::mt19937& random) {
  RETURN_IF_ERROR(ValidateGenerationOptions(0, temperature));
  if (logits.empty() ||
      logits.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return absl::InvalidArgumentError(
        "logits must have a nonempty int-sized vocabulary");
  }
  int best = 0;
  for (int token = 0; token < static_cast<int>(logits.size()); ++token) {
    if (std::isnan(logits[token]) ||
        logits[token] == std::numeric_limits<float>::infinity()) {
      return absl::InvalidArgumentError(
          "logits contain NaN or positive infinity");
    }
    // Strict comparison deliberately preserves the first (lowest-ID) tie,
    // including +0 versus -0. No random draw participates in greedy decoding.
    if (logits[token] > logits[best])
      best = token;
  }
  if (!std::isfinite(logits[best])) {
    return absl::InvalidArgumentError(
        "all logits are masked by negative infinity");
  }
  if (temperature == 0.0)
    return best;

  std::vector<double> probabilities(logits.size());
  for (size_t token = 0; token < logits.size(); ++token) {
    // Subtract in double so extreme finite FP32 logits cannot overflow the
    // subtraction. The maximum has weight one, even at tiny temperatures.
    probabilities[token] = std::exp(
        (static_cast<double>(logits[token]) - logits[best]) / temperature);
  }
  std::discrete_distribution<int> sample(probabilities.begin(),
                                         probabilities.end());
  return sample(random);
}

}  // namespace pluto::llm
