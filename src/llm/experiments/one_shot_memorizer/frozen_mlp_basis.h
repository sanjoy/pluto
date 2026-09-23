#pragma once

#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::llm::one_shot_memorizer {

// A fixed expansion basis standardized using only the supplied fitting inputs.
// This is a construction from initial directions, not a fit to teacher outputs
// or labels. The CPU helper neither takes nor infers either of those targets.
struct FrozenMlpBasis {
  // FP32 master expansion matrix, row-major [input_dim, feature_width].
  std::vector<float> weights;
  // FP32 master expansion bias, with one value per feature.
  std::vector<float> bias;
  // Fitting-set means of X * initial_weights, before standardization.
  std::vector<double> projection_means;
  // Population standard deviations of the same unstandardized projections.
  std::vector<double> projection_standard_deviations;
};

// fitting_inputs contains only fitting rows, row-major [rows, input_dim]; the
// caller must not include held-out rows. initial_weights has shape [input_dim,
// feature_width]. In the frozen-BF16 experiment, these are effective step-0
// BF16 directions decoded to float, not the checkpoint's unrounded FP32
// masters.
//
// For each direction a, computes mu = mean(X*a), sigma = stddev(X*a), then
// returns a/sigma and -mu/sigma. An initial bias is unnecessary: subtracting
// the projection mean cancels it. Population moments and intermediate
// arithmetic use FP64; output masters use FP32. Any later GPU BF16 rounding is
// intentional, so neither FP32 nor BF16 evaluation is promised to have exactly
// unit variance.
//
// Rejects invalid shapes, non-finite values, constant projections, and output
// coefficients that overflow FP32 or round a nonzero coefficient to zero.
absl::StatusOr<FrozenMlpBasis> StandardizeFrozenMlpBasis(
    absl::Span<const float> fitting_inputs,
    absl::Span<const float> initial_weights, int input_dim, int feature_width);

}  // namespace pluto::llm::one_shot_memorizer
