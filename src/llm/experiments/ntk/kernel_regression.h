#pragma once

#include <cstddef>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::llm::ntk {

// Dense, row-major CPU matrix. Indexing is unchecked; numerical entry points
// validate dimensions, storage, and finiteness before accessing any elements.
struct Matrix {
  size_t rows = 0;
  size_t columns = 0;
  std::vector<double> values;

  double& operator()(size_t row, size_t column) {
    return values[row * columns + column];
  }
  const double& operator()(size_t row, size_t column) const {
    return values[row * columns + column];
  }
};

// Empty matrices are structurally valid. Fitting separately requires a
// nonempty square matrix. Rejects dimension overflow and NaN/infinite values.
absl::Status ValidateMatrix(const Matrix& matrix);

// Fits the residual targets around a NONZERO initial function:
//   (training_kernel + ridge * I) * coefficients = targets - initial_values.
// For a feature Gram matrix this is kernel ridge regression with unnormalized
// squared loss (1/2 sum of squared errors), not mean loss. Thus ridge is added
// directly, without multiplying by the number of observations.
//
// Requires finite ridge > 0 and a symmetric matrix (64 double epsilons of
// relative entrywise tolerance). Symmetric roundoff is averaged. Cholesky
// requires the REGULARIZED matrix to be numerically positive definite; this
// routine does not separately certify that the input kernel is PSD. Singular
// PSD kernels are valid when ridge makes the factorization well conditioned.
absl::StatusOr<std::vector<double>> FitRidge(
    const Matrix& training_kernel, absl::Span<const double> initial_values,
    absl::Span<const double> targets, double ridge);

// Returns query_initial + cross_kernel * coefficients. Rows are query output
// coordinates and columns are training output coordinates, in exactly the
// same ordering used while fitting. Zero query rows are supported.
absl::StatusOr<std::vector<double>> Predict(
    const Matrix& cross_kernel, absl::Span<const double> query_initial,
    absl::Span<const double> coefficients);

// Full-batch gradient descent in a fixed feature model, expressed in kernel
// coefficients. Starts at alpha=0 and repeats:
//   alpha -= learning_rate * (initial_values + K * alpha - targets) / N.
// This corresponds to mean squared loss 1/(2N) sum(error^2), with no ridge or
// weight decay. It is NOT gradient descent on alpha as independent model
// parameters (which would multiply the residual by K again).
//
// Each row is one scalar output observation. For a multi-output/block kernel,
// flatten sample/output-coordinate pairs consistently; N is the total number
// of scalar coordinates, including all outputs, not just the sample count.
// Requires finite learning_rate > 0, steps >= 0, and a nonempty symmetric K.
// No PSD/stability-bound test is performed. Nonfinite arithmetic is reported
// as an error; finite divergence remains possible for an excessive rate.
absl::StatusOr<std::vector<double>> FitGradientDescent(
    const Matrix& training_kernel, absl::Span<const double> initial_values,
    absl::Span<const double> targets, double learning_rate, int steps);

}  // namespace pluto::llm::ntk
