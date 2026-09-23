#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace pluto::llm::one_shot_memorizer {

// Bit j retains feature j. A zero mask retains no features; the highest bit
// addresses feature 63. These helpers know nothing about targets or searches.
using FeatureSubset = uint64_t;

// Bound materialization rather than permitting an accidental exponential
// allocation. This includes all triples and quadruples of 64 features.
inline constexpr size_t kMaxMaterializedFeatureSubsets = 1'000'000;

// Parses "-" for no retained features, or comma-separated unsigned decimal
// IDs. Rejects whitespace, signs, empty fields, duplicate numeric IDs
// (including alternate leading-zero spellings), and IDs outside [0,
// feature_count). feature_count must be in [1,64]; ordering in the input has no
// significance.
absl::StatusOr<FeatureSubset> ParseFeatureSubset(absl::string_view text,
                                                 int feature_count);

// Every subset with exactly cardinality retained features, in lexicographic
// order of their ascending feature-ID lists (not numerical bitmask order).
// Cardinality zero returns the single zero mask; feature_count is in [1,64]
// and cardinality is in [0,feature_count]. Returns ResourceExhausted before
// allocating if the binomial count exceeds kMaxMaterializedFeatureSubsets.
absl::StatusOr<std::vector<FeatureSubset>> EnumerateFeatureSubsets(
    int feature_count, int cardinality);

// Copies a physical BF16 row, replacing unretained entries with +0 bits. This
// is a pure bit operation: retained signed zeros, NaNs, and infinities are
// preserved exactly, without numeric conversions or finite-value checks.
// Rejects empty rows, widths above 64, and mask bits outside the supplied row.
absl::StatusOr<std::vector<uint16_t>> ApplyBf16FeatureSubset(
    absl::Span<const uint16_t> row, FeatureSubset subset);

}  // namespace pluto::llm::one_shot_memorizer
