#include "src/llm/experiments/memorize_general_facts/activation_collision_audit/audit.h"

#include <algorithm>
#include <tuple>
#include <utility>

namespace pluto::llm::activation_collision_audit {

absl::Status Audit::Add(absl::Span<const uint16_t> bits,
                        Occurrence occurrence) {
  if (width_ <= 0)
    return absl::InvalidArgumentError("audit width must be positive");
  if (bits.size() != static_cast<size_t>(width_))
    return absl::InvalidArgumentError(
        "activation bitvector size does not match audit width");
  if (occurrence.sample < 0 || occurrence.position < 0 || occurrence.target < 0)
    return absl::InvalidArgumentError(
        "sample, position, and target must be nonnegative");
  for (uint16_t word : bits)
    // BF16's eight exponent bits are bits 7..14. An all-ones exponent denotes
    // either infinity or NaN, regardless of its sign and mantissa payload.
    if ((word & 0x7f80) == 0x7f80)
      return absl::InvalidArgumentError("activation contains nonfinite BF16");

  groups_[std::vector<uint16_t>(bits.begin(), bits.end())].push_back(
      occurrence);
  ++total_rows_;
  return absl::OkStatus();
}

Summary Audit::Summarize() const {
  Summary summary;
  summary.total_rows = total_rows_;
  summary.unique_vectors = groups_.size();
  for (const auto& [bits, occurrences] : groups_) {
    if (occurrences.size() > 1)
      ++summary.repeated_groups;
    absl::flat_hash_map<int, int64_t> target_counts;
    int64_t largest_count = 0;
    for (const auto& occurrence : occurrences)
      largest_count =
          std::max(largest_count, ++target_counts[occurrence.target]);
    // A deterministic function of this vector must return one label for every
    // occurrence. Choosing its most frequent target is optimal for this group.
    summary.minimum_errors +=
        static_cast<int64_t>(occurrences.size()) - largest_count;
    if (target_counts.size() < 2)
      continue;
    ++summary.conflicting_groups;
    summary.conflicting_rows += occurrences.size();
    CollisionGroup group{bits, occurrences};
    std::sort(group.occurrences.begin(), group.occurrences.end(),
              [](const Occurrence& a, const Occurrence& b) {
                return std::tie(a.sample, a.position, a.target) <
                       std::tie(b.sample, b.position, b.target);
              });
    summary.conflicts.push_back(std::move(group));
  }
  std::sort(summary.conflicts.begin(), summary.conflicts.end(),
            [](const CollisionGroup& a, const CollisionGroup& b) {
              return a.bits < b.bits;
            });
  return summary;
}

}  // namespace pluto::llm::activation_collision_audit
