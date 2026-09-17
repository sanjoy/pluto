#include "src/llm/experiments/ntk/kernel_regression.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/util/status_macros.h"

namespace pluto::llm::ntk {
namespace {

absl::Status ValidateValues(absl::Span<const double> values,
                            absl::string_view description) {
  for (size_t index = 0; index < values.size(); ++index)
    if (!std::isfinite(values[index]))
      return absl::InvalidArgumentError(absl::StrCat(
          description, " contains a nonfinite value at index ", index));
  return absl::OkStatus();
}

absl::Status ValidateTrainingInputs(const Matrix& kernel,
                                    absl::Span<const double> initial_values,
                                    absl::Span<const double> targets) {
  RETURN_IF_ERROR(ValidateMatrix(kernel));
  if (kernel.rows == 0 || kernel.rows != kernel.columns)
    return absl::InvalidArgumentError(
        "training kernel must be nonempty and square");
  if (initial_values.size() != kernel.rows || targets.size() != kernel.rows)
    return absl::InvalidArgumentError(
        "initial values and targets must match training kernel rows");
  RETURN_IF_ERROR(ValidateValues(initial_values, "initial values"));
  RETURN_IF_ERROR(ValidateValues(targets, "targets"));
  constexpr double kSymmetryTolerance =
      64 * std::numeric_limits<double>::epsilon();
  for (size_t row = 0; row < kernel.rows; ++row)
    for (size_t column = 0; column < row; ++column) {
      const double left = kernel(row, column);
      const double right = kernel(column, row);
      const double scale = std::max(std::abs(left), std::abs(right));
      if (std::abs(left - right) > kSymmetryTolerance * scale)
        return absl::InvalidArgumentError(absl::StrCat(
            "training kernel is not symmetric at (", row, ", ", column, ")"));
    }
  return absl::OkStatus();
}

absl::Status NumericalFailure(absl::string_view operation, size_t row) {
  return absl::OutOfRangeError(
      absl::StrCat(operation, " produced nonfinite arithmetic at row ", row));
}

// Callers validate all shapes once. Reuse the prediction storage for every GD
// step, and never update alpha until ALL rows have used the old coefficients.
absl::Status PredictInto(const Matrix& kernel,
                         absl::Span<const double> initial_values,
                         absl::Span<const double> coefficients,
                         absl::Span<double> predictions) {
  for (size_t row = 0; row < kernel.rows; ++row) {
    double value = initial_values[row];
    for (size_t column = 0; column < kernel.columns; ++column)
      value = std::fma(kernel(row, column), coefficients[column], value);
    if (!std::isfinite(value))
      return NumericalFailure("kernel prediction", row);
    predictions[row] = value;
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateMatrix(const Matrix& matrix) {
  if (matrix.columns != 0 &&
      matrix.rows > std::numeric_limits<size_t>::max() / matrix.columns)
    return absl::InvalidArgumentError("matrix dimensions overflow size_t");
  if (matrix.values.size() != matrix.rows * matrix.columns)
    return absl::InvalidArgumentError(
        "matrix storage does not match its row and column dimensions");
  return ValidateValues(matrix.values, "matrix");
}

absl::StatusOr<std::vector<double>> FitRidge(
    const Matrix& training_kernel, absl::Span<const double> initial_values,
    absl::Span<const double> targets, double ridge) {
  RETURN_IF_ERROR(
      ValidateTrainingInputs(training_kernel, initial_values, targets));
  if (!std::isfinite(ridge) || ridge <= 0)
    return absl::InvalidArgumentError("ridge must be finite and positive");

  const size_t count = training_kernel.rows;
  Matrix lower{count, count,
               std::vector<double>(training_kernel.values.size())};
  // Cholesky constructs K + ridge I = L L^T. Work only in the lower triangle;
  // averaging the two entries makes tolerated symmetry roundoff unbiased.
  // Do not silently add jitter: changing ridge changes the experiment.
  for (size_t row = 0; row < count; ++row)
    for (size_t column = 0; column <= row; ++column) {
      double value = training_kernel(row, column);
      if (row == column)
        value += ridge;
      else
        value += (training_kernel(column, row) - value) * 0.5;
      for (size_t inner = 0; inner < column; ++inner)
        value = std::fma(-lower(row, inner), lower(column, inner), value);
      if (!std::isfinite(value))
        return NumericalFailure("Cholesky factorization", row);
      if (row == column) {
        if (value <= 0)
          return absl::InvalidArgumentError(absl::StrCat(
              "regularized training kernel is not numerically positive "
              "definite at row ",
              row, "; increase ridge or check the kernel"));
        lower(row, column) = std::sqrt(value);
      } else {
        lower(row, column) = value / lower(column, column);
        if (!std::isfinite(lower(row, column)))
          return NumericalFailure("Cholesky factorization", row);
      }
    }

  // First solve L z = targets - f0, then L^T alpha = z. Reusing the same
  // vector is safe because each triangular solve only reads solved entries.
  std::vector<double> coefficients(count);
  for (size_t row = 0; row < count; ++row) {
    double value = targets[row] - initial_values[row];
    for (size_t column = 0; column < row; ++column)
      value = std::fma(-lower(row, column), coefficients[column], value);
    coefficients[row] = value / lower(row, row);
    if (!std::isfinite(coefficients[row]))
      return NumericalFailure("ridge forward substitution", row);
  }
  for (size_t row = count; row-- > 0;) {
    double value = coefficients[row];
    for (size_t column = row + 1; column < count; ++column)
      value = std::fma(-lower(column, row), coefficients[column], value);
    coefficients[row] = value / lower(row, row);
    if (!std::isfinite(coefficients[row]))
      return NumericalFailure("ridge backward substitution", row);
  }
  return coefficients;
}

absl::StatusOr<std::vector<double>> Predict(
    const Matrix& cross_kernel, absl::Span<const double> query_initial,
    absl::Span<const double> coefficients) {
  RETURN_IF_ERROR(ValidateMatrix(cross_kernel));
  if (query_initial.size() != cross_kernel.rows ||
      coefficients.size() != cross_kernel.columns)
    return absl::InvalidArgumentError(
        "query initial values and coefficients must match cross-kernel "
        "rows and columns");
  RETURN_IF_ERROR(ValidateValues(query_initial, "query initial values"));
  RETURN_IF_ERROR(ValidateValues(coefficients, "coefficients"));
  std::vector<double> predictions(cross_kernel.rows);
  RETURN_IF_ERROR(PredictInto(cross_kernel, query_initial, coefficients,
                              absl::MakeSpan(predictions)));
  return predictions;
}

absl::StatusOr<std::vector<double>> FitGradientDescent(
    const Matrix& training_kernel, absl::Span<const double> initial_values,
    absl::Span<const double> targets, double learning_rate, int steps) {
  RETURN_IF_ERROR(
      ValidateTrainingInputs(training_kernel, initial_values, targets));
  if (!std::isfinite(learning_rate) || learning_rate <= 0)
    return absl::InvalidArgumentError(
        "learning rate must be finite and positive");
  if (steps < 0)
    return absl::InvalidArgumentError(
        "gradient descent steps must be nonnegative");

  const size_t count = training_kernel.rows;
  std::vector<double> coefficients(count, 0.0);
  std::vector<double> predictions(count);
  for (int step = 0; step < steps; ++step) {
    RETURN_IF_ERROR(PredictInto(training_kernel, initial_values, coefficients,
                                absl::MakeSpan(predictions)));
    for (size_t row = 0; row < count; ++row) {
      const double residual = predictions[row] - targets[row];
      coefficients[row] =
          std::fma(-learning_rate, residual / static_cast<double>(count),
                   coefficients[row]);
      if (!std::isfinite(coefficients[row]))
        return NumericalFailure("gradient descent update", row);
    }
  }
  return coefficients;
}

}  // namespace pluto::llm::ntk
