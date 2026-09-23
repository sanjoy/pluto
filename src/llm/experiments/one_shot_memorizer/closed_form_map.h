#pragma once

#include <cstddef>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::llm::one_shot_memorizer {

struct AffineMapOptions {
  // Minimize ||XW+1*b-Y||_F^2/n + ridge*||W||_F^2. The affine bias is
  // unpenalized. Zero requires a full-column-rank centered input matrix.
  double ridge = 0;
  // Pivoted QR rejects a remaining column norm <= this fraction of the
  // largest initial column norm. Applies to the ridge-augmented matrix too.
  double relative_rank_tolerance = 1e-12;
};

struct ClosedFormMap {
  int input_dim = 0;
  int output_dim = 0;
  size_t sample_count = 0;
  // Rank of the fitted centered/augmented design, not an estimate of the
  // original input rank when ridge regularization is present.
  size_t numerical_rank = 0;
  // |R_jj| from column-pivoted QR, in pivot order. These describe the centered
  // design, augmented by sqrt(n*ridge)*I when ridge is positive. Their spread
  // is a rank/conditioning diagnostic, not the matrix's condition number.
  std::vector<double> qr_diagonal_magnitudes;
  AffineMapOptions options;
  std::vector<double> weights;  // Row-major [input_dim, output_dim].
  std::vector<double> biases;   // [output_dim].
  double rmse = 0;              // sqrt(SSE/(n*output_dim)).
  double target_rms = 0;
  // ||prediction-Y||_F / ||Y||_F, using uncentered targets. Zero if both
  // norms are zero; infinity if only the target norm is zero.
  double relative_rmse = 0;
};

// Fits an affine map to finite row-major float matrices X[n,input_dim] and
// Y[n,output_dim], using double-precision column-pivoted Householder QR.
// X/Y are centered first; positive ridge appends sqrt(n*ridge)*I to X and
// zero rows to Y. The intercept is then mean(Y)-mean(X)*W. No gradient
// updates, normal equations, feature rescaling, or external solver are used.
// Returns FailedPrecondition for insufficient/numerically dependent columns.
absl::StatusOr<ClosedFormMap> FitAffineMap(
    absl::Span<const float> inputs, absl::Span<const float> targets,
    int input_dim, int output_dim, const AffineMapOptions& options = {});

// Evaluates XW+b in double precision, preserving input row order. Empty
// input is allowed; malformed matrices and nonfinite coefficients/data fail.
absl::StatusOr<std::vector<double>> ApplyClosedFormMap(
    const ClosedFormMap& map, absl::Span<const float> inputs);

}  // namespace pluto::llm::one_shot_memorizer
