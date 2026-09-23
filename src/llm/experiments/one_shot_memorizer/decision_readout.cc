#include "src/llm/experiments/one_shot_memorizer/decision_readout.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

absl::Status Validate(const DecisionReadout& readout) {
  if (readout.width <= 0 || readout.vocab_size <= 0 ||
      !std::isfinite(readout.epsilon) || readout.epsilon <= 0)
    return absl::InvalidArgumentError(
        "invalid decision readout dimensions/epsilon");
  const size_t width = readout.width, vocabulary = readout.vocab_size;
  if (vocabulary > std::numeric_limits<size_t>::max() / width ||
      readout.directions.size() != vocabulary * width ||
      readout.offsets.size() != vocabulary)
    return absl::InvalidArgumentError(
        "decision readout storage does not match shape");
  for (double value : readout.directions)
    if (!std::isfinite(value))
      return absl::InvalidArgumentError(
          "decision readout directions must be finite");
  for (double value : readout.offsets)
    if (!std::isfinite(value))
      return absl::InvalidArgumentError(
          "decision readout offsets must be finite");
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<DecisionReadout> MakeDecisionReadout(
    absl::Span<const float> embeddings, int width,
    absl::Span<const float> gamma, absl::Span<const float> beta,
    double epsilon) {
  if (width <= 0 || embeddings.empty() || embeddings.size() % width != 0 ||
      gamma.size() != static_cast<size_t>(width) ||
      beta.size() != static_cast<size_t>(width) || !std::isfinite(epsilon) ||
      epsilon <= 0)
    return absl::InvalidArgumentError(
        "invalid decision readout parameter shape/epsilon");
  const size_t vocabulary = embeddings.size() / width;
  if (vocabulary > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      embeddings.size() > std::vector<double>().max_size())
    return absl::OutOfRangeError(
        "decision readout shape exceeds supported range");
  for (const auto values : {embeddings, gamma, beta})
    for (float value : values)
      if (!std::isfinite(value))
        return absl::InvalidArgumentError(
            "decision readout parameters must be finite");
  DecisionReadout readout{.width = width,
                          .vocab_size = static_cast<int>(vocabulary),
                          .directions = std::vector<double>(embeddings.size()),
                          .offsets = std::vector<double>(vocabulary),
                          .epsilon = epsilon};
  for (size_t token = 0; token < vocabulary; ++token) {
    const size_t first = token * width;
    double mean = 0, offset = 0;
    for (int dim = 0; dim < width; ++dim) {
      const double embedding = embeddings[first + dim];
      readout.directions[first + dim] = embedding * gamma[dim];
      mean += readout.directions[first + dim];
      offset += embedding * beta[dim];
    }
    mean /= width;
    for (int dim = 0; dim < width; ++dim)
      readout.directions[first + dim] -= mean;
    readout.offsets[token] = offset;
  }
  RETURN_IF_ERROR(Validate(readout));
  return readout;
}

absl::StatusOr<std::vector<double>> EvaluateDecisionReadout(
    const DecisionReadout& readout, absl::Span<const float> residuals) {
  RETURN_IF_ERROR(Validate(readout));
  const size_t width = readout.width, vocabulary = readout.vocab_size;
  if (residuals.size() % width != 0)
    return absl::InvalidArgumentError(
        "decision residual storage does not match width");
  const size_t rows = residuals.size() / width;
  if (rows > std::vector<double>().max_size() / vocabulary)
    return absl::OutOfRangeError(
        "decision score matrix exceeds supported range");
  for (float value : residuals)
    if (!std::isfinite(value))
      return absl::InvalidArgumentError("decision residuals must be finite");
  std::vector<double> scores(rows * vocabulary), centered(width);
  for (size_t row = 0; row < rows; ++row) {
    double mean = 0;
    for (size_t dim = 0; dim < width; ++dim)
      mean += residuals[row * width + dim];
    mean /= width;
    double variance = 0;
    for (size_t dim = 0; dim < width; ++dim) {
      centered[dim] = static_cast<double>(residuals[row * width + dim]) - mean;
      variance += centered[dim] * centered[dim];
    }
    const double scale = std::sqrt(variance / width + readout.epsilon);
    if (!std::isfinite(scale) || scale <= 0)
      return absl::OutOfRangeError("decision normalization scale overflowed");
    // Normalize before taking the dot product: large, finite FP32 residuals
    // must not overflow merely because normalization was deferred.
    for (double& value : centered)
      value /= scale;
    for (size_t token = 0; token < vocabulary; ++token) {
      double score = readout.offsets[token];
      for (size_t dim = 0; dim < width; ++dim)
        score += readout.directions[token * width + dim] * centered[dim];
      if (!std::isfinite(score))
        return absl::OutOfRangeError("decision score overflowed");
      scores[row * vocabulary + token] = score;
    }
  }
  return scores;
}

}  // namespace pluto::llm::one_shot_memorizer
