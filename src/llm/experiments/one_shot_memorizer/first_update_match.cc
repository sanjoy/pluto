#include "src/llm/experiments/one_shot_memorizer/first_update_match.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <utility>
#include <vector>

#include "absl/status/status.h"

namespace pluto::llm::one_shot_memorizer {

absl::StatusOr<FirstAdamUpdateScore> ScoreFirstAdamUpdate(
    absl::Span<const float> initial, absl::Span<const float> observed,
    absl::Span<const float> gradients, float learning_rate, float epsilon) {
  static_assert(sizeof(float) == sizeof(uint32_t) &&
                std::numeric_limits<float>::is_iec559);
  if (initial.empty() || initial.size() != observed.size() ||
      initial.size() != gradients.size())
    return absl::InvalidArgumentError(
        "first-update arrays must have the same nonzero length");
  if (!std::isfinite(learning_rate) || learning_rate <= 0 ||
      !std::isfinite(epsilon) || epsilon <= 0)
    return absl::InvalidArgumentError(
        "first-update learning rate and epsilon must be finite and positive");
  FirstAdamUpdateScore score;
  for (size_t i = 0; i < initial.size(); ++i) {
    if (!std::isfinite(initial[i]) || !std::isfinite(observed[i]) ||
        !std::isfinite(gradients[i]))
      return absl::InvalidArgumentError("first-update arrays must be finite");
    const float denominator = std::abs(gradients[i]) + epsilon;
    if (!std::isfinite(denominator))
      return absl::OutOfRangeError("first-update denominator overflows FP32");
    // Keep the validated prototype's two FP32 statements. Do not force extra
    // rounding that would prohibit the compiler's usual multiply/subtract
    // contraction: that would change candidate scores from the prototype.
    const float update = gradients[i] / (std::abs(gradients[i]) + epsilon);
    const float prediction = initial[i] - learning_rate * update;
    if (!std::isfinite(update) || !std::isfinite(prediction))
      return absl::OutOfRangeError("first-update prediction overflows FP32");
    const double error = static_cast<double>(prediction) - observed[i];
    score.squared_error += error * error;
    if (!std::isfinite(score.squared_error))
      return absl::OutOfRangeError(
          "first-update squared error overflows double");
    // Comparison is equivalent to sign(weight - initial), without overflowing
    // a subtraction between widely separated finite FP32 endpoints.
    const int predicted_sign =
        (prediction > initial[i]) - (prediction < initial[i]);
    const int observed_sign =
        (observed[i] > initial[i]) - (observed[i] < initial[i]);
    score.sign_mismatches += predicted_sign != observed_sign;
    score.bit_equal_coordinates += std::bit_cast<uint32_t>(prediction) ==
                                   std::bit_cast<uint32_t>(observed[i]);
  }
  return score;
}

absl::StatusOr<std::vector<FirstUpdatePromptSet>>
BuildOneTokenReplacementPromptSets(absl::Span<const int> prompt_ids,
                                   absl::Span<const int> suffix_ids,
                                   int vocabulary_size, int eos_token) {
  if (vocabulary_size <= 0 || eos_token < 0 || eos_token >= vocabulary_size ||
      prompt_ids.size() != 5)
    return absl::InvalidArgumentError(
        "prompt repair requires five IDs and a valid vocabulary/EOS");
  std::vector<int> original(prompt_ids.begin(), prompt_ids.end());
  for (int token : original)
    if (token < 0 || token >= vocabulary_size || token == eos_token)
      return absl::InvalidArgumentError("invalid prompt repair token ID");
  std::sort(original.begin(), original.end());
  if (std::adjacent_find(original.begin(), original.end()) != original.end())
    return absl::InvalidArgumentError("prompt repair IDs must be distinct");
  std::set<int> donors;
  for (int token : suffix_ids) {
    if (token < 0 || token >= vocabulary_size)
      return absl::InvalidArgumentError("invalid suffix repair token ID");
    if (token != eos_token)
      donors.insert(token);
  }
  std::vector<FirstUpdatePromptSet> result{{original, -1, -1}};
  std::set<std::vector<int>> seen{original};
  for (size_t slot = 0; slot < original.size(); ++slot)
    for (int added : donors) {
      auto ids = original;
      const int removed = ids[slot];
      ids[slot] = added;
      std::sort(ids.begin(), ids.end());
      if (std::adjacent_find(ids.begin(), ids.end()) == ids.end() &&
          seen.insert(ids).second)
        result.push_back({std::move(ids), removed, added});
    }
  return result;
}

}  // namespace pluto::llm::one_shot_memorizer
