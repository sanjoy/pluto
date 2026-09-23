#include "src/llm/experiments/one_shot_memorizer/embedding_sign_readout.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

absl::Status ValidateEndpoints(absl::Span<const float> initial,
                               absl::Span<const float> trained, int width) {
  if (width <= 0 || initial.empty() || initial.size() != trained.size() ||
      initial.size() % width != 0)
    return absl::InvalidArgumentError(
        "embeddings must have the same nonempty [vocabulary_size, width] "
        "shape");
  if (initial.size() / width > std::numeric_limits<int>::max())
    return absl::OutOfRangeError("embedding row IDs do not fit in int");
  for (size_t i = 0; i < initial.size(); ++i)
    if (!std::isfinite(initial[i]) || !std::isfinite(trained[i]))
      return absl::InvalidArgumentError(
          "embeddings must contain finite values");
  return absl::OkStatus();
}

float RoundBf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000u;
  return std::bit_cast<float>(bits);
}

EmbeddingSignScores Score(absl::Span<const double> matrix, int width) {
  const size_t rows = matrix.size() / width;
  EmbeddingSignScores result;
  result.mean.resize(width, 0);
  result.scores.resize(rows, 0);
  for (size_t row = 0; row < rows; ++row)
    for (int col = 0; col < width; ++col)
      result.mean[col] += matrix[row * width + col] / rows;
  for (size_t row = 0; row < rows; ++row) {
    for (int col = 0; col < width; ++col)
      result.scores[row] += matrix[row * width + col] * result.mean[col];
    if (result.scores[row] < 0)
      result.selected_ids.push_back(static_cast<int>(row));
  }
  return result;
}

}  // namespace

absl::StatusOr<EmbeddingSignReadout> ComputeEmbeddingSignReadout(
    absl::Span<const float> initial, absl::Span<const float> trained,
    int width) {
  RETURN_IF_ERROR(ValidateEndpoints(initial, trained, width));

  std::vector<double> delta_fp32(initial.size()), delta_bf16(initial.size());
  std::vector<double> trained_fp32(initial.size()),
      trained_bf16(initial.size());
  for (size_t i = 0; i < initial.size(); ++i) {
    const float initial_rounded = RoundBf16(initial[i]);
    const float trained_rounded = RoundBf16(trained[i]);
    if (!std::isfinite(initial_rounded) || !std::isfinite(trained_rounded))
      return absl::OutOfRangeError("embedding endpoint overflows BF16");
    // Subtract in double: even two finite FP32 endpoints can overflow FP32.
    delta_fp32[i] = static_cast<double>(trained[i]) - initial[i];
    delta_bf16[i] = static_cast<double>(trained_rounded) - initial_rounded;
    trained_fp32[i] = trained[i];
    trained_bf16[i] = trained_rounded;
  }
  return EmbeddingSignReadout{
      Score(delta_fp32, width), Score(delta_bf16, width),
      Score(trained_fp32, width), Score(trained_bf16, width)};
}

absl::StatusOr<std::vector<EmbeddingResidualCandidate>>
RankEmbeddingResidualCandidates(absl::Span<const float> initial,
                                absl::Span<const float> trained, int width) {
  RETURN_IF_ERROR(ValidateEndpoints(initial, trained, width));
  std::vector<double> delta(initial.size());
  for (size_t i = 0; i < initial.size(); ++i)
    delta[i] = static_cast<double>(trained[i]) - initial[i];
  const auto sign = Score(delta, width);
  std::vector<EmbeddingResidualCandidate> candidates;
  for (size_t row = 0; row < sign.scores.size(); ++row) {
    if (sign.scores[row] < 0)
      continue;
    double squared_norm = 0;
    for (int column = 0; column < width; ++column) {
      const double residual = delta[row * width + column] - sign.mean[column];
      squared_norm += residual * residual;
    }
    candidates.push_back({static_cast<int>(row), squared_norm});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) {
              return a.residual_squared_norm != b.residual_squared_norm
                         ? a.residual_squared_norm > b.residual_squared_norm
                         : a.row_id < b.row_id;
            });
  return candidates;
}

}  // namespace pluto::llm::one_shot_memorizer
