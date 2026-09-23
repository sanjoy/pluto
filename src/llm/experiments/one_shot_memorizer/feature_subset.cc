#include "src/llm/experiments/one_shot_memorizer/feature_subset.h"

#include <algorithm>
#include <numeric>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"

namespace pluto::llm::one_shot_memorizer {

absl::StatusOr<FeatureSubset> ParseFeatureSubset(absl::string_view text,
                                                 int feature_count) {
  if (feature_count < 1 || feature_count > 64)
    return absl::InvalidArgumentError("feature count must be in [1,64]");
  if (text == "-")
    return FeatureSubset{0};
  if (text.empty())
    return absl::InvalidArgumentError("empty feature subset");
  FeatureSubset result = 0;
  for (size_t begin = 0;;) {
    const size_t end = text.find(',', begin);
    const auto field =
        text.substr(begin, end == text.npos ? text.npos : end - begin);
    int feature;
    if (field.empty() ||
        !std::all_of(field.begin(), field.end(),
                     [](char ch) { return ch >= '0' && ch <= '9'; }) ||
        !absl::SimpleAtoi(field, &feature) || feature >= feature_count)
      return absl::InvalidArgumentError(
          "feature IDs must be unsigned decimal values inside the row");
    const FeatureSubset bit = FeatureSubset{1} << feature;
    if (result & bit)
      return absl::InvalidArgumentError("duplicate feature ID");
    result |= bit;
    if (end == text.npos)
      return result;
    begin = end + 1;
  }
}

absl::StatusOr<std::vector<FeatureSubset>> EnumerateFeatureSubsets(
    int feature_count, int cardinality) {
  if (feature_count < 1 || feature_count > 64 || cardinality < 0 ||
      cardinality > feature_count)
    return absl::InvalidArgumentError("invalid feature count or cardinality");

  // Binomial coefficients grow monotonically up to the middle. Check after
  // each exact recurrence step: its previous value is at most one million,
  // so multiplying by at most 64 cannot overflow this uint64_t accumulator.
  uint64_t count = 1;
  const int smaller = std::min(cardinality, feature_count - cardinality);
  for (int i = 1; i <= smaller; ++i) {
    count = count * static_cast<uint64_t>(feature_count - i + 1) / i;
    if (count > kMaxMaterializedFeatureSubsets)
      return absl::ResourceExhaustedError(
          "feature subset count exceeds the materialization limit");
  }

  std::vector<FeatureSubset> result;
  result.reserve(static_cast<size_t>(count));
  if (cardinality == 0) {
    result.push_back(0);
    return result;
  }
  std::vector<int> features(cardinality);
  std::iota(features.begin(), features.end(), 0);
  while (true) {
    FeatureSubset subset = 0;
    for (int feature : features)
      subset |= FeatureSubset{1} << feature;
    result.push_back(subset);

    // Advance the rightmost feature that can move, then reset its suffix.
    // This also handles cardinality==feature_count without ever shifting 64.
    int position = cardinality - 1;
    while (position >= 0 &&
           features[position] == feature_count - cardinality + position)
      --position;
    if (position < 0)
      break;
    ++features[position];
    for (int next = position + 1; next < cardinality; ++next)
      features[next] = features[next - 1] + 1;
  }
  return result;
}

absl::StatusOr<std::vector<uint16_t>> ApplyBf16FeatureSubset(
    absl::Span<const uint16_t> row, FeatureSubset subset) {
  if (row.empty() || row.size() > 64)
    return absl::InvalidArgumentError(
        "BF16 feature row width must be in [1,64]");
  // Shifting by the integer width is undefined, so width 64 needs no test:
  // every possible bit already addresses a valid feature in that case.
  if (row.size() < 64 && (subset >> row.size()) != 0)
    return absl::InvalidArgumentError("feature mask contains bits outside row");
  std::vector<uint16_t> result(row.begin(), row.end());
  for (size_t feature = 0; feature < row.size(); ++feature)
    if (!(subset & (FeatureSubset{1} << feature)))
      result[feature] = 0;
  return result;
}

}  // namespace pluto::llm::one_shot_memorizer
