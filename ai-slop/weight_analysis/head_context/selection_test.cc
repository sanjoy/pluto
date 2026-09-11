#include "ai-slop/weight_analysis/head_context/selection.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::weight_analysis::head_context {
namespace {

constexpr Geometry kSmall{8, 4, 2, 3};
constexpr Selection kSelection{3, 1, 1, 2, QueryScope::kSelectedQuery, 0.5f};

TEST(HeadContextSelectionTest, ExhaustiveFiniteHalfMatchesNearestValueOracle) {
  // This independent oracle searches all representable positive BF16 values
  // instead of duplicating the implementation's bit-shift/exponent branches.
  std::vector<double> numbers;
  for (uint32_t bits = 0; bits < 0x7f80; ++bits) {
    numbers.push_back(std::bit_cast<float>(bits << 16));
  }
  for (uint32_t bits = 0; bits <= 0xffff; ++bits) {
    if ((bits & 0x7f80) == 0x7f80) {
      EXPECT_FALSE(ScaleBf16(bits, 0.5f).ok());
      continue;
    }
    const double half = numbers[bits & 0x7fff] * 0.5;
    size_t upper = std::lower_bound(numbers.begin(), numbers.end(), half) -
                   numbers.begin();
    const size_t lower = upper == 0 ? 0 : upper - 1;
    const double below = half - numbers[lower];
    const double above = numbers[upper] - half;
    const size_t nearest =
        above < below || (above == below && upper % 2 == 0) ? upper : lower;
    auto scaled = ScaleBf16(bits, 0.5f);
    ASSERT_TRUE(scaled.ok());
    EXPECT_EQ(*scaled, (bits & 0x8000) | nearest) << bits;
    EXPECT_EQ(*ScaleBf16(bits, 1.0f), bits);
    EXPECT_EQ(*ScaleBf16(bits, 0.0f), 0);
  }
}

TEST(HeadContextSelectionTest, RejectsEveryNonfiniteEncodingAndUndeclaredDose) {
  for (uint32_t bits = 0; bits <= 0xffff; ++bits) {
    if ((bits & 0x7f80) != 0x7f80) continue;
    for (float scale : {0.0f, 0.5f, 1.0f}) {
      EXPECT_FALSE(ScaleBf16(bits, scale).ok());
    }
  }
  for (float scale :
       {-1.0f, 0.25f, 2.0f, std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()}) {
    EXPECT_FALSE(ScaleBf16(0x3f80, scale).ok());
  }
}

TEST(HeadContextSelectionTest, ThreeScopesTouchOnlyDeclaredSequenceAndHead) {
  std::vector<uint16_t> original(8 * 6, 0x3f80);
  original[0] = 0x8000;  // Unselected negative zero must retain its bits.
  for (auto scope : {QueryScope::kSelectedQuery, QueryScope::kOtherQueries,
                     QueryScope::kAllQueries}) {
    for (float scale : {0.0f, 0.5f, 1.0f}) {
      Selection selection = kSelection;
      selection.scope = scope;
      selection.scale = scale;
      std::vector<uint16_t> output(original.size(), 0xffff);
      ASSERT_TRUE(
          ScaleContext(original, absl::MakeSpan(output), 8, kSmall, selection)
              .ok());
      for (int row = 0; row < 8; ++row) {
        for (int lane = 0; lane < 6; ++lane) {
          const bool query_selected = row % 4 == 2;
          const bool applies =
              row >= 4 && lane >= 3 &&
              (scope == QueryScope::kAllQueries ||
               (scope == QueryScope::kSelectedQuery && query_selected) ||
               (scope == QueryScope::kOtherQueries && !query_selected));
          const uint16_t expected = applies ? (scale == 0      ? 0
                                               : scale == 0.5f ? 0x3f00
                                                               : 0x3f80)
                                            : original[row * 6 + lane];
          EXPECT_EQ(output[row * 6 + lane], expected) << row << ',' << lane;
        }
      }
    }
  }
  EXPECT_EQ(original[0], 0x8000);
  EXPECT_EQ(std::count(original.begin(), original.end(), 0x3f80), 47);
}

TEST(HeadContextSelectionTest, OtherQueryScopeHasEmptyEffectForLengthOne) {
  const Geometry geometry{1, 1, 1, 1};
  const Selection selection{0, 0, 0, 0, QueryScope::kOtherQueries, 0};
  std::vector<uint16_t> original{0x8000}, output{0};
  ASSERT_TRUE(
      ScaleContext(original, absl::MakeSpan(output), 1, geometry, selection)
          .ok());
  EXPECT_EQ(output, original);
}

TEST(HeadContextSelectionTest, SelectionGeometryOverflowAndScopeAreValidated) {
  EXPECT_TRUE(ValidateSelection(kSmall, kSelection, 8).ok());
  for (int rows : {-1, 0, 7, std::numeric_limits<int>::max()}) {
    EXPECT_FALSE(ValidateSelection(kSmall, kSelection, rows).ok());
  }
  for (Geometry geometry :
       {Geometry{0, 4, 2, 3}, Geometry{8, 0, 2, 3}, Geometry{8, 4, 0, 3},
        Geometry{8, 4, 2, 0},
        Geometry{8, 4, std::numeric_limits<int>::max(), 2}}) {
    EXPECT_FALSE(ValidateSelection(geometry, kSelection, 8).ok());
  }
  for (int field = 0; field < 4; ++field) {
    for (int value : {-1, 999}) {
      Selection selection = kSelection;
      switch (field) {
        case 0:
          selection.block = value;
          break;
        case 1:
          selection.head = value;
          break;
        case 2:
          selection.sequence = value;
          break;
        case 3:
          selection.query_position = value;
          break;
      }
      EXPECT_FALSE(ValidateSelection(kSmall, selection, 8).ok());
    }
  }
  Selection selection = kSelection;
  selection.scope = static_cast<QueryScope>(99);
  EXPECT_FALSE(ValidateSelection(kSmall, selection, 8).ok());
}

TEST(HeadContextSelectionTest,
     ShapeAliasingAndNonfiniteErrorsDoNotModifyOutput) {
  std::vector<uint16_t> original(48, 0x3f80), output(48, 0x1234);
  EXPECT_FALSE(ScaleContext(original, absl::MakeSpan(output).subspan(1), 8,
                            kSmall, kSelection)
                   .ok());
  EXPECT_EQ(std::count(output.begin(), output.end(), 0x1234), 48);
  EXPECT_FALSE(
      ScaleContext(original, absl::MakeSpan(original), 8, kSmall, kSelection)
          .ok());
  std::vector<uint16_t> overlap(49, 0x3f80);
  EXPECT_FALSE(ScaleContext(absl::MakeConstSpan(overlap).subspan(0, 48),
                            absl::MakeSpan(overlap).subspan(1), 8, kSmall,
                            kSelection)
                   .ok());
  original.back() = 0x7f80;  // Even an unselected nonfinite value is invalid.
  EXPECT_FALSE(
      ScaleContext(original, absl::MakeSpan(output), 8, kSmall, kSelection)
          .ok());
  EXPECT_EQ(std::count(output.begin(), output.end(), 0x1234), 48);
}

}  // namespace
}  // namespace pluto::weight_analysis::head_context
