#include "src/llm/experiments/one_shot_memorizer/feature_donor.h"

#include "absl/status/status.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {

absl::StatusOr<std::vector<uint16_t>> MakeFeatureDonorRow(
    absl::Span<const uint16_t> recipient, absl::Span<const uint16_t> donor,
    FeatureSubset subset, FeatureDonorBackground background) {
  if (recipient.size() != donor.size())
    return absl::InvalidArgumentError("recipient and donor widths must match");
  if (background != FeatureDonorBackground::kSparse &&
      background != FeatureDonorBackground::kIntact)
    return absl::InvalidArgumentError("unknown feature donor background");
  ASSIGN_OR_RETURN(auto result, ApplyBf16FeatureSubset(donor, subset));
  if (background == FeatureDonorBackground::kIntact)
    for (size_t channel = 0; channel < recipient.size(); ++channel)
      if (!(subset & (FeatureSubset{1} << channel)))
        result[channel] = recipient[channel];
  return result;
}

}  // namespace pluto::llm::one_shot_memorizer
