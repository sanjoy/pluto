#include "src/llm/experiments/one_shot_memorizer/label_projection.h"

#include <cmath>
#include <limits>
#include <vector>

#include "absl/status/status.h"

namespace pluto::llm::one_shot_memorizer {

absl::StatusOr<ClosedFormMap> FitTokenCodeProjection(
    absl::Span<const float> features, int feature_width,
    absl::Span<const float> residuals, absl::Span<const int> labels,
    absl::Span<const int> sentence_indices, const TokenCodes& codes,
    const LabelProjectionOptions& options) {
  if (feature_width <= 0 || codes.width <= 0 || codes.vocab_size <= 0 ||
      labels.empty() || sentence_indices.size() != labels.size() ||
      labels.size() > std::numeric_limits<size_t>::max() / feature_width ||
      labels.size() > std::numeric_limits<size_t>::max() / codes.width ||
      static_cast<size_t>(codes.vocab_size) >
          std::numeric_limits<size_t>::max() / codes.width ||
      features.size() != labels.size() * feature_width ||
      residuals.size() != labels.size() * codes.width ||
      codes.values.size() !=
          static_cast<size_t>(codes.vocab_size) * codes.width)
    return absl::InvalidArgumentError("malformed token-code fit matrices");
  if (!std::isfinite(options.code_scale) || options.code_scale <= 0 ||
      options.held_sentence_stride < 0 || options.held_sentence_stride == 1)
    return absl::InvalidArgumentError("invalid token-code fit options");
  for (const auto values :
       {features, residuals, absl::Span<const float>(codes.values)})
    for (float value : values)
      if (!std::isfinite(value))
        return absl::InvalidArgumentError(
            "token-code fit values must be finite");
  std::vector<float> fitting_inputs, fitting_targets;
  fitting_inputs.reserve(features.size());
  fitting_targets.reserve(residuals.size());
  for (size_t row = 0; row < labels.size(); ++row) {
    const int label = labels[row], sentence = sentence_indices[row];
    if (label < 0 || label >= codes.vocab_size || sentence < 0)
      return absl::InvalidArgumentError(
          "invalid token label or sentence index");
    if (options.held_sentence_stride != 0 &&
        sentence % options.held_sentence_stride == 0)
      continue;
    fitting_inputs.insert(fitting_inputs.end(),
                          features.begin() + row * feature_width,
                          features.begin() + (row + 1) * feature_width);
    double mean = 0;
    for (int dim = 0; dim < codes.width; ++dim)
      mean += residuals[row * codes.width + dim];
    mean /= codes.width;
    for (int dim = 0; dim < codes.width; ++dim) {
      const double target =
          options.code_scale *
              codes.values[static_cast<size_t>(label) * codes.width + dim] -
          (residuals[row * codes.width + dim] - mean);
      const float stored = static_cast<float>(target);
      if (!std::isfinite(stored))
        return absl::OutOfRangeError("token-code fit target overflows float");
      fitting_targets.push_back(stored);
    }
  }
  if (fitting_inputs.empty())
    return absl::InvalidArgumentError("sentence split leaves no fitting rows");
  return FitAffineMap(fitting_inputs, fitting_targets, feature_width,
                      codes.width, options.affine);
}

}  // namespace pluto::llm::one_shot_memorizer
