#pragma once

#include "src/llm/experiments/one_shot_memorizer/closed_form_map.h"
#include "src/llm/experiments/one_shot_memorizer/token_codes.h"

namespace pluto::llm::one_shot_memorizer {

struct LabelProjectionOptions {
  // Zero fits all sentences; N>=2 excludes sentence indices divisible by N.
  int held_sentence_stride = 5;
  // Desired residual is this positive multiple of the target token's code.
  double code_scale = 0.25;
  AffineMapOptions affine{.ridge = 1e-6};
};

// Construct a final MLP projection from fixed features and corpus token labels,
// NOT from the original MLP's outputs. For each fitting row i, solve
//   features[i] W + b ~= code_scale * codes[label[i]] - center(residual[i]).
// Centering removes the common-coordinate component that final LayerNorm
// discards. Features and incoming residuals still depend on a learned backbone;
// this is label-only module reconstruction, not dataset-only model training.
// Inputs/residuals are row-major, labels and sentence_indices identify each
// supervised row, and the codebook is assigned before the sentence split.
// Held rows are validated but never used to fit means, coefficients or biases.
absl::StatusOr<ClosedFormMap> FitTokenCodeProjection(
    absl::Span<const float> features, int feature_width,
    absl::Span<const float> residuals, absl::Span<const int> labels,
    absl::Span<const int> sentence_indices, const TokenCodes& codes,
    const LabelProjectionOptions& options = {});

}  // namespace pluto::llm::one_shot_memorizer
