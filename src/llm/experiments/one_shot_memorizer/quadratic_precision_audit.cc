#include "src/llm/experiments/one_shot_memorizer/quadratic_precision_audit.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

constexpr int kWidth = 16;
constexpr int kFeatures = kWidth + kWidth * (kWidth + 1) / 2;
constexpr size_t kHeldStride = 5;
constexpr double kRidge = 1e-6;

bool IsFiniteBf16(float value) {
  return std::isfinite(value) && (std::bit_cast<uint32_t>(value) & 0xffff) == 0;
}

float RoundBf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fff + ((bits >> 16) & 1);
  return std::bit_cast<float>(bits & 0xffff0000);
}

std::vector<float> SelectFitting(absl::Span<const float> values, size_t width,
                                 absl::Span<const uint8_t> fitting_rows) {
  std::vector<float> result;
  for (size_t row = 0; row < fitting_rows.size(); ++row)
    if (fitting_rows[row])
      result.insert(result.end(), values.begin() + row * width,
                    values.begin() + (row + 1) * width);
  return result;
}

double RelativeNorm(double numerator, double denominator) {
  return denominator > 0  ? numerator / denominator
         : numerator == 0 ? 0
                          : std::numeric_limits<double>::infinity();
}

absl::StatusOr<QuadraticPrecisionErrors> ScoreGroup(
    absl::Span<const double> predictions, absl::Span<const float> targets,
    absl::Span<const uint8_t> fitting_rows, bool fitting) {
  QuadraticPrecisionErrors result;
  std::array<double, kWidth> means{};
  for (size_t row = 0; row < fitting_rows.size(); ++row) {
    if (static_cast<bool>(fitting_rows[row]) != fitting)
      continue;
    ++result.rows;
    for (int coordinate = 0; coordinate < kWidth; ++coordinate)
      means[coordinate] += targets[row * kWidth + coordinate];
  }
  if (result.rows == 0)
    return absl::InvalidArgumentError("quadratic audit has an empty row group");
  for (double& mean : means)
    mean /= static_cast<double>(result.rows);
  double error_norm = 0, target_norm = 0, centered_norm = 0;
  for (size_t row = 0; row < fitting_rows.size(); ++row) {
    if (static_cast<bool>(fitting_rows[row]) != fitting)
      continue;
    for (int coordinate = 0; coordinate < kWidth; ++coordinate) {
      const size_t index = row * kWidth + coordinate;
      error_norm = std::hypot(error_norm, predictions[index] - targets[index]);
      target_norm =
          std::hypot(target_norm, static_cast<double>(targets[index]));
      centered_norm =
          std::hypot(centered_norm, targets[index] - means[coordinate]);
    }
  }
  if (!std::isfinite(error_norm) || !std::isfinite(target_norm) ||
      !std::isfinite(centered_norm))
    return absl::OutOfRangeError("quadratic audit error norm overflowed");
  result.rmse =
      error_norm / std::sqrt(static_cast<double>(result.rows) * kWidth);
  result.relative_rmse = RelativeNorm(error_norm, target_norm);
  result.centered_relative_rmse = RelativeNorm(error_norm, centered_norm);
  return result;
}

absl::StatusOr<QuadraticPrecisionEvaluation> Evaluate(
    const ClosedFormMap& map, absl::Span<const float> features,
    absl::Span<const float> targets, absl::Span<const uint8_t> fitting_rows) {
  ASSIGN_OR_RETURN(auto predictions, ApplyClosedFormMap(map, features));
  QuadraticPrecisionEvaluation result;
  ASSIGN_OR_RETURN(result.fitting,
                   ScoreGroup(predictions, targets, fitting_rows, true));
  ASSIGN_OR_RETURN(result.held,
                   ScoreGroup(predictions, targets, fitting_rows, false));
  return result;
}

absl::StatusOr<double> RoundCoefficient(double value, bool bf16) {
  if (!std::isfinite(value) ||
      std::abs(value) > std::numeric_limits<float>::max())
    return absl::OutOfRangeError("quadratic audit coefficient exceeds FP32");
  float rounded = static_cast<float>(value);
  if (bf16)
    rounded = RoundBf16(rounded);
  if (!std::isfinite(rounded))
    return absl::OutOfRangeError("quadratic audit coefficient exceeds BF16");
  // Underflow to zero is intentional here: the audit measures this rounding.
  return static_cast<double>(rounded);
}

absl::StatusOr<QuadraticPrecisionVariant> FitVariant(
    absl::Span<const float> features, absl::Span<const float> targets,
    absl::Span<const uint8_t> fitting_rows) {
  const auto fitting_features =
      SelectFitting(features, kFeatures, fitting_rows);
  const auto fitting_targets = SelectFitting(targets, kWidth, fitting_rows);
  QuadraticPrecisionVariant result;
  ASSIGN_OR_RETURN(result.fitted_map,
                   FitAffineMap(fitting_features, fitting_targets, kFeatures,
                                kWidth, {.ridge = kRidge}));
  ASSIGN_OR_RETURN(result.fp64_parameters,
                   Evaluate(result.fitted_map, features, targets, fitting_rows));
  auto rounded = result.fitted_map;
  for (double& weight : rounded.weights) {
    ASSIGN_OR_RETURN(weight, RoundCoefficient(weight, false));
  }
  for (double& bias : rounded.biases) {
    ASSIGN_OR_RETURN(bias, RoundCoefficient(bias, false));
  }
  ASSIGN_OR_RETURN(result.fp32_parameters,
                   Evaluate(rounded, features, targets, fitting_rows));
  for (double& weight : rounded.weights) {
    ASSIGN_OR_RETURN(weight, RoundCoefficient(weight, true));
  }
  ASSIGN_OR_RETURN(result.bf16_weights_fp32_bias,
                   Evaluate(rounded, features, targets, fitting_rows));
  return result;
}

}  // namespace

absl::StatusOr<std::vector<float>> MakeQuadraticAuditFeatures(
    absl::Span<const float> normalized_inputs,
    QuadraticProductPrecision precision) {
  if (precision != QuadraticProductPrecision::kFp32 &&
      precision != QuadraticProductPrecision::kBf16)
    return absl::InvalidArgumentError("unknown quadratic product precision");
  if (normalized_inputs.empty() || normalized_inputs.size() % kWidth != 0)
    return absl::InvalidArgumentError(
        "quadratic audit inputs require [rows,16]");
  const size_t rows = normalized_inputs.size() / kWidth;
  if (rows > std::numeric_limits<size_t>::max() / kFeatures)
    return absl::OutOfRangeError("quadratic audit feature shape overflow");
  for (float value : normalized_inputs)
    if (!IsFiniteBf16(value))
      return absl::InvalidArgumentError(
          "quadratic audit inputs must be finite exact BF16");
  std::vector<float> result(rows * kFeatures);
  for (size_t row = 0; row < rows; ++row) {
    for (int coordinate = 0; coordinate < kWidth; ++coordinate)
      result[row * kFeatures + coordinate] =
          normalized_inputs[row * kWidth + coordinate];
    size_t column = kWidth;
    for (int i = 0; i < kWidth; ++i)
      for (int j = i; j < kWidth; ++j) {
        float product = normalized_inputs[row * kWidth + i] *
                        normalized_inputs[row * kWidth + j];
        if (!std::isfinite(product))
          return absl::OutOfRangeError("quadratic product exceeds FP32");
        if (precision == QuadraticProductPrecision::kBf16)
          product = RoundBf16(product);
        if (!std::isfinite(product))
          return absl::OutOfRangeError("quadratic product exceeds BF16");
        result[row * kFeatures + column++] = product;
      }
  }
  return result;
}

absl::StatusOr<QuadraticPrecisionAudit> AuditQuadraticPrecision(
    absl::Span<const float> normalized_inputs,
    absl::Span<const float> target_updates,
    absl::Span<const size_t> sentence_lengths) {
  if (normalized_inputs.empty() || normalized_inputs.size() % kWidth != 0 ||
      target_updates.size() != normalized_inputs.size() ||
      sentence_lengths.empty())
    return absl::InvalidArgumentError(
        "inconsistent quadratic audit matrix shapes");
  for (float value : target_updates)
    if (!IsFiniteBf16(value))
      return absl::InvalidArgumentError(
          "quadratic audit targets must be finite exact BF16");
  const size_t rows = normalized_inputs.size() / kWidth;
  std::vector<uint8_t> fitting_rows;
  fitting_rows.reserve(rows);
  QuadraticPrecisionAudit result;
  for (size_t sentence = 0; sentence < sentence_lengths.size(); ++sentence) {
    const size_t length = sentence_lengths[sentence];
    if (length == 0 || length > rows - fitting_rows.size())
      return absl::InvalidArgumentError(
          "invalid quadratic audit sentence lengths");
    const bool fitting = sentence % kHeldStride != 0;
    fitting_rows.insert(fitting_rows.end(), length, fitting);
    if (fitting)
      ++result.fitting_sentences;
    else
      ++result.held_sentences;
  }
  if (fitting_rows.size() != rows || result.fitting_sentences == 0 ||
      result.held_sentences == 0)
    return absl::InvalidArgumentError(
        "quadratic audit needs complete nonempty sentence groups");
  ASSIGN_OR_RETURN(auto features,
                   MakeQuadraticAuditFeatures(normalized_inputs,
                                              QuadraticProductPrecision::kFp32));
  ASSIGN_OR_RETURN(result.unrounded_products,
                   FitVariant(features, target_updates, fitting_rows));
  ASSIGN_OR_RETURN(features,
                   MakeQuadraticAuditFeatures(normalized_inputs,
                                              QuadraticProductPrecision::kBf16));
  ASSIGN_OR_RETURN(result.bf16_products,
                   FitVariant(features, target_updates, fitting_rows));
  return result;
}

}  // namespace pluto::llm::one_shot_memorizer
