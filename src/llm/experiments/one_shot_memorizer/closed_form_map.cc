#include "src/llm/experiments/one_shot_memorizer/closed_form_map.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <utility>

#include "absl/status/status.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

absl::Status ValidateDimensions(int input_dim, int output_dim) {
  if (input_dim <= 0 || output_dim <= 0)
    return absl::InvalidArgumentError("Affine-map dimensions must be positive");
  if (static_cast<size_t>(input_dim) >
      std::numeric_limits<size_t>::max() / output_dim)
    return absl::OutOfRangeError("Affine-map coefficient count overflows");
  return absl::OkStatus();
}

double ColumnNorm(const std::vector<double>& matrix, size_t rows, size_t cols,
                  size_t first_row, size_t column) {
  // Scaled sum of squares avoids overflow/underflow in squared entries.
  double scale = 0, sum_squares = 1;
  for (size_t row = first_row; row < rows; ++row) {
    const double value = std::abs(matrix[row * cols + column]);
    if (value == 0)
      continue;
    if (scale < value) {
      const double ratio = scale / value;
      sum_squares = 1 + sum_squares * ratio * ratio;
      scale = value;
    } else {
      const double ratio = value / scale;
      sum_squares += ratio * ratio;
    }
  }
  return scale == 0 ? 0 : scale * std::sqrt(sum_squares);
}

// Factor the centered/augmented design A with column pivoting and apply the
// same Householder reflectors to every target column. The upper triangle of
// A becomes R; implicit-unit reflector vectors occupy the lower triangle.
absl::Status PivotedQr(std::vector<double>& a, std::vector<double>& y,
                       size_t rows, size_t input_dim, size_t output_dim,
                       double relative_tolerance,
                       std::vector<size_t>& permutation) {
  permutation.resize(input_dim);
  std::iota(permutation.begin(), permutation.end(), size_t{0});
  double reference_norm = 0;
  for (size_t k = 0; k < input_dim; ++k) {
    size_t pivot = k;
    double norm = 0;
    // Recompute norms after each reflector rather than using unstable
    // squared-norm downdates. Width is at most tens in the intended probes.
    for (size_t j = k; j < input_dim; ++j) {
      const double candidate = ColumnNorm(a, rows, input_dim, k, j);
      if (!std::isfinite(candidate))
        return absl::OutOfRangeError("QR column norm overflowed");
      if (candidate > norm) {
        norm = candidate;
        pivot = j;
      }
    }
    if (k == 0)
      reference_norm = norm;
    if (norm == 0 || norm <= relative_tolerance * reference_norm)
      return absl::FailedPreconditionError(
          "Affine design has numerical rank " + std::to_string(k) +
          " below input_dim; use positive ridge or adjust rank tolerance");
    if (pivot != k) {
      for (size_t row = 0; row < rows; ++row)
        std::swap(a[row * input_dim + k], a[row * input_dim + pivot]);
      std::swap(permutation[k], permutation[pivot]);
    }
    const double original = a[k * input_dim + k];
    const double diagonal = -std::copysign(norm, original);
    const double leading = original - diagonal;
    const double tau = (diagonal - original) / diagonal;
    a[k * input_dim + k] = diagonal;
    for (size_t row = k + 1; row < rows; ++row)
      a[row * input_dim + k] /= leading;
    for (size_t j = k + 1; j < input_dim; ++j) {
      double dot = a[k * input_dim + j];
      for (size_t row = k + 1; row < rows; ++row)
        dot += a[row * input_dim + k] * a[row * input_dim + j];
      dot *= tau;
      a[k * input_dim + j] -= dot;
      for (size_t row = k + 1; row < rows; ++row)
        a[row * input_dim + j] -= a[row * input_dim + k] * dot;
    }
    for (size_t j = 0; j < output_dim; ++j) {
      double dot = y[k * output_dim + j];
      for (size_t row = k + 1; row < rows; ++row)
        dot += a[row * input_dim + k] * y[row * output_dim + j];
      dot *= tau;
      y[k * output_dim + j] -= dot;
      for (size_t row = k + 1; row < rows; ++row)
        y[row * output_dim + j] -= a[row * input_dim + k] * dot;
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::vector<double>> ApplyClosedFormMap(
    const ClosedFormMap& map, absl::Span<const float> inputs) {
  const auto dimensions = ValidateDimensions(map.input_dim, map.output_dim);
  if (!dimensions.ok())
    return dimensions;
  const size_t p = map.input_dim, q = map.output_dim;
  if (map.weights.size() != p * q || map.biases.size() != q ||
      inputs.size() % p != 0)
    return absl::InvalidArgumentError(
        "Affine-map or input matrix shape mismatch");
  const size_t n = inputs.size() / p;
  if (n > std::numeric_limits<size_t>::max() / q)
    return absl::OutOfRangeError("Affine output element count overflows");
  for (double weight : map.weights)
    if (!std::isfinite(weight))
      return absl::InvalidArgumentError("Affine weights must be finite");
  for (double bias : map.biases)
    if (!std::isfinite(bias))
      return absl::InvalidArgumentError("Affine biases must be finite");
  for (float value : inputs)
    if (!std::isfinite(value))
      return absl::InvalidArgumentError("Affine inputs must be finite");
  std::vector<double> result(n * q);
  for (size_t row = 0; row < n; ++row) {
    for (size_t j = 0; j < q; ++j) {
      double value = map.biases[j];
      for (size_t k = 0; k < p; ++k)
        value +=
            static_cast<double>(inputs[row * p + k]) * map.weights[k * q + j];
      if (!std::isfinite(value))
        return absl::OutOfRangeError("Affine prediction overflowed");
      result[row * q + j] = value;
    }
  }
  return result;
}

absl::StatusOr<ClosedFormMap> FitAffineMap(absl::Span<const float> inputs,
                                           absl::Span<const float> targets,
                                           int input_dim, int output_dim,
                                           const AffineMapOptions& options) {
  const auto dimensions = ValidateDimensions(input_dim, output_dim);
  if (!dimensions.ok())
    return dimensions;
  if (!std::isfinite(options.ridge) || options.ridge < 0 ||
      !std::isfinite(options.relative_rank_tolerance) ||
      options.relative_rank_tolerance < 0 ||
      options.relative_rank_tolerance >= 1)
    return absl::InvalidArgumentError(
        "Invalid ridge or relative rank tolerance");
  const size_t p = input_dim, q = output_dim;
  if (inputs.empty() || inputs.size() % p != 0)
    return absl::InvalidArgumentError("Inputs must be nonempty [n,input_dim]");
  const size_t n = inputs.size() / p;
  if (n > std::numeric_limits<size_t>::max() / q || targets.size() != n * q)
    return absl::InvalidArgumentError("Targets must have shape [n,output_dim]");
  if (options.ridge == 0 && n <= p)
    return absl::FailedPreconditionError(
        "Unregularized centered fit needs more samples than input dimensions");
  const size_t extra = options.ridge > 0 ? p : 0;
  if (n > std::numeric_limits<size_t>::max() - extra ||
      n + extra > std::numeric_limits<size_t>::max() / p ||
      n + extra > std::numeric_limits<size_t>::max() / q)
    return absl::OutOfRangeError("Augmented affine design dimensions overflow");
  const size_t rows = n + extra;
  const double scaled_ridge = static_cast<double>(n) * options.ridge;
  if (!std::isfinite(scaled_ridge))
    return absl::OutOfRangeError("Sample-scaled ridge overflowed");
  std::vector<double> input_mean(p), target_mean(q);
  for (size_t row = 0; row < n; ++row) {
    for (size_t j = 0; j < p; ++j) {
      const float value = inputs[row * p + j];
      if (!std::isfinite(value))
        return absl::InvalidArgumentError("Affine inputs must be finite");
      input_mean[j] += value;
    }
    for (size_t j = 0; j < q; ++j) {
      const float value = targets[row * q + j];
      if (!std::isfinite(value))
        return absl::InvalidArgumentError("Affine targets must be finite");
      target_mean[j] += value;
    }
  }
  for (double& mean : input_mean)
    mean /= n;
  for (double& mean : target_mean)
    mean /= n;
  std::vector<double> a(rows * p), y(rows * q);
  for (size_t row = 0; row < n; ++row) {
    for (size_t j = 0; j < p; ++j)
      a[row * p + j] = static_cast<double>(inputs[row * p + j]) - input_mean[j];
    for (size_t j = 0; j < q; ++j)
      y[row * q + j] =
          static_cast<double>(targets[row * q + j]) - target_mean[j];
  }
  if (extra != 0)
    for (size_t j = 0; j < p; ++j)
      a[(n + j) * p + j] = std::sqrt(scaled_ridge);
  std::vector<size_t> permutation;
  const auto qr =
      PivotedQr(a, y, rows, p, q, options.relative_rank_tolerance, permutation);
  if (!qr.ok())
    return qr;
  std::vector<double> solution(p * q);
  for (size_t row = p; row-- > 0;) {
    for (size_t j = 0; j < q; ++j) {
      double value = y[row * q + j];
      for (size_t k = row + 1; k < p; ++k)
        value -= a[row * p + k] * solution[k * q + j];
      solution[row * q + j] = value / a[row * p + row];
      if (!std::isfinite(solution[row * q + j]))
        return absl::OutOfRangeError("Affine fitted coefficient overflowed");
    }
  }
  ClosedFormMap map;
  map.input_dim = input_dim;
  map.output_dim = output_dim;
  map.sample_count = n;
  map.numerical_rank = p;
  map.qr_diagonal_magnitudes.reserve(p);
  for (size_t j = 0; j < p; ++j)
    map.qr_diagonal_magnitudes.push_back(std::abs(a[j * p + j]));
  map.options = options;
  map.weights.resize(p * q);
  map.biases = target_mean;
  for (size_t row = 0; row < p; ++row)
    for (size_t j = 0; j < q; ++j)
      map.weights[permutation[row] * q + j] = solution[row * q + j];
  for (size_t j = 0; j < q; ++j)
    for (size_t k = 0; k < p; ++k)
      map.biases[j] -= input_mean[k] * map.weights[k * q + j];
  auto predictions = ApplyClosedFormMap(map, inputs);
  if (!predictions.ok())
    return predictions.status();
  double error_norm = 0, target_norm = 0;
  for (size_t i = 0; i < targets.size(); ++i) {
    error_norm = std::hypot(error_norm, (*predictions)[i] - targets[i]);
    target_norm = std::hypot(target_norm, static_cast<double>(targets[i]));
  }
  map.rmse = error_norm / std::sqrt(static_cast<double>(targets.size()));
  map.target_rms = target_norm / std::sqrt(static_cast<double>(targets.size()));
  map.relative_rmse = target_norm > 0 ? error_norm / target_norm
                      : error_norm == 0
                          ? 0
                          : std::numeric_limits<double>::infinity();
  return map;
}

}  // namespace pluto::llm::one_shot_memorizer
