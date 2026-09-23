#include "src/llm/experiments/one_shot_memorizer/embedding_mean_shift.h"

#include <cmath>
#include <limits>

#include "absl/status/status.h"

namespace pluto::llm::one_shot_memorizer {

absl::StatusOr<EmbeddingMeanShift> MatchEmbeddingMeans(
    absl::Span<const float> embedding, absl::Span<const float> reference,
    size_t width) {
  if (width == 0 || embedding.empty() || embedding.size() != reference.size() ||
      embedding.size() % width != 0)
    return absl::InvalidArgumentError(
        "embedding mean shift requires equal, "
        "nonempty rectangular matrices");
  EmbeddingMeanShift result;
  result.shift.assign(width, 0);
  for (size_t i = 0; i < embedding.size(); ++i) {
    if (!std::isfinite(embedding[i]) || !std::isfinite(reference[i]))
      return absl::InvalidArgumentError(
          "embedding mean shift requires finite "
          "input and reference values");
    const double difference = static_cast<double>(reference[i]) - embedding[i];
    result.shift[i % width] += difference;
    result.squared_error_before += difference * difference;
  }
  const size_t rows = embedding.size() / width;
  for (double& shift : result.shift)
    shift /= rows;
  result.values.reserve(embedding.size());
  for (size_t i = 0; i < embedding.size(); ++i) {
    const double value =
        static_cast<double>(embedding[i]) + result.shift[i % width];
    if (!std::isfinite(value) ||
        std::abs(value) > std::numeric_limits<float>::max())
      return absl::OutOfRangeError("embedding mean shift overflows FP32");
    // Do not turn -0 into +0 when no arithmetic change was requested.
    const float rounded =
        result.shift[i % width] == 0 ? embedding[i] : static_cast<float>(value);
    result.values.push_back(rounded);
    const double difference = static_cast<double>(reference[i]) - rounded;
    result.squared_error_after += difference * difference;
  }
  return result;
}

}  // namespace pluto::llm::one_shot_memorizer
