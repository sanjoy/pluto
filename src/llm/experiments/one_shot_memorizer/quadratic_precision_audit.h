#pragma once

#include <cstddef>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/experiments/one_shot_memorizer/closed_form_map.h"

namespace pluto::llm::one_shot_memorizer {

// Only the precision of the quadratic products changes; the 16 linear
// coordinates are copied. Both variants start from decoded physical BF16.
enum class QuadraticProductPrecision { kFp32, kBf16 };

// CPU-only diagnostic basis, row-major [rows,152]: x[0..15], followed by
// x[i]*x[j] for i=0..15,j=i..15. Products are computed in FP32, optionally
// rounded once to BF16 ties-to-even. No scaling, constant column, or targets
// enter construction. Rejects empty/malformed inputs, values not exactly
// representable in finite BF16, and nonfinite products after either rounding.
absl::StatusOr<std::vector<float>> MakeQuadraticAuditFeatures(
    absl::Span<const float> normalized_inputs,
    QuadraticProductPrecision precision);

// Errors for one sentence group, including every real token row in that group.
struct QuadraticPrecisionErrors {
  size_t rows = 0;           // Number of 16-dimensional output rows scored.
  double rmse = 0;           // sqrt(SSE / (rows * 16)).
  double relative_rmse = 0;  // sqrt(SSE / sum(target^2)).
  // Denominator centers each target coordinate using this group's own mean.
  // A zero denominator yields zero for zero error, otherwise positive infinity.
  double centered_relative_rmse = 0;
};

struct QuadraticPrecisionEvaluation {
  QuadraticPrecisionErrors fitting;  // Sentence indices not divisible by five.
  QuadraticPrecisionErrors held;     // Sentence indices divisible by five.
};

// One independently fitted feature precision, with three arithmetic audits.
// All predictions use FP64 dot products and have NO final output rounding.
// Parameter-rounding audits are NOT tensor-core/MMA or BF16-forward emulation:
// they isolate coefficient rounding, leaving accumulation in FP64.
struct QuadraticPrecisionVariant {
  ClosedFormMap
      fitted_map;  // FP64 ridge fit, retained for rank/weight auditing.
  QuadraticPrecisionEvaluation
      fp64_parameters;  // Original fitted coefficients.
  QuadraticPrecisionEvaluation fp32_parameters;  // W and bias rounded to FP32.
  // W rounded FP64 -> FP32 -> BF16; bias rounded FP64 -> FP32 only.
  QuadraticPrecisionEvaluation bf16_weights_fp32_bias;
};

// Separate penalty-free diagnostic on unrounded products. It uses the same
// fitting rows and FP64 pivoted QR, with no parameter or output rounding.
struct QuadraticUnregularizedFit {
  ClosedFormMap fitted_map;                 // ridge=0; intercept unpenalized.
  QuadraticPrecisionEvaluation evaluation;  // Fit and held rows scored apart.
};

struct QuadraticPrecisionAudit {
  size_t fitting_sentences = 0;  // Caller verifies 819 for the full corpus.
  size_t held_sentences = 0;     // Caller verifies 205 for the full corpus.
  QuadraticPrecisionVariant unrounded_products;  // No BF16 product rounding.
  QuadraticPrecisionVariant bf16_products;       // Production feature rounding.
  // FailedPrecondition records rank deficiency/insufficient samples without
  // discarding the primary ridge fits. No retry changes the QR tolerance.
  absl::StatusOr<QuadraticUnregularizedFit> ridge_zero_unrounded;
};

// Fits and scores both variants using the SAME sentence split and fixed
// ridge=1e-6, with an unpenalized intercept. A separate unrounded-product fit
// uses ridge=0 and the solver's unchanged relative rank tolerance (1e-12).
// Inputs and learned branch-update targets have shape [rows,16] and must be
// decoded finite physical BF16.
// sentence_lengths lists positive real-token counts in unshuffled corpus order;
// its sum must equal rows. Every fifth sentence (0,5,10,...) is held out as a
// whole. Targets from those rows never enter fitting or feature construction.
// The caller enforces the experiment's 1,024-sentence dataset identity; smaller
// corpora are supported for unit tests, provided both groups are nonempty.
//
// This conditional diagnostic retains learned normalized inputs and outputs.
// It does not reconstruct a model from text or establish downstream completion
// accuracy. No CUDA runtime, executor, GPU allocation, or model mutation
// occurs.
absl::StatusOr<QuadraticPrecisionAudit> AuditQuadraticPrecision(
    absl::Span<const float> normalized_inputs,
    absl::Span<const float> target_updates,
    absl::Span<const size_t> sentence_lengths);

}  // namespace pluto::llm::one_shot_memorizer
