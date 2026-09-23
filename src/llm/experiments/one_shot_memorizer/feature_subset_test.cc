#include "src/llm/experiments/one_shot_memorizer/feature_subset.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

TEST(FeatureSubsetTest, ParsesEmptySetAndUnorderedDecimalIds) {
  ASSERT_TRUE(ParseFeatureSubset("-", 1).ok());
  EXPECT_EQ(*ParseFeatureSubset("-", 1), 0u);
  auto result = ParseFeatureSubset("03,0,2", 4);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(*result, 13u);
}

TEST(FeatureSubsetTest, ParsesHighestBitWithoutSignedShift) {
  auto result = ParseFeatureSubset("63,0", 64);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(*result, (uint64_t{1} << 63) | 1);
  EXPECT_FALSE(ParseFeatureSubset("63", 63).ok());
}

TEST(FeatureSubsetTest, RejectsMalformedDuplicateAndOverflowIds) {
  for (const char* text :
       {"", ",", "1,", ",1", "1,,2", "-1", "+1", " 1", "1 ", "1\t", "1\n",
        "0x1", "64", "1,1", "1,01", "999999999999999999999999999999999"})
    EXPECT_EQ(ParseFeatureSubset(text, 64).status().code(),
              absl::StatusCode::kInvalidArgument)
        << text;
  for (int width : {-1, 0, 65, std::numeric_limits<int>::max()})
    EXPECT_FALSE(ParseFeatureSubset("-", width).ok());
}

TEST(FeatureSubsetTest, EnumeratesCanonicalChannelCombinationOrder) {
  auto result = EnumerateFeatureSubsets(4, 2);
  ASSERT_TRUE(result.ok()) << result.status();
  // [0,1], [0,2], [0,3], [1,2], [1,3], [2,3], not integer-mask order.
  EXPECT_EQ(*result, (std::vector<FeatureSubset>{3, 5, 9, 6, 10, 12}));
}

TEST(FeatureSubsetTest, HandlesZeroFullAndSingleFeatureCardinalities) {
  for (int width : {1, 2, 63, 64}) {
    auto none = EnumerateFeatureSubsets(width, 0);
    ASSERT_TRUE(none.ok()) << none.status();
    EXPECT_EQ(*none, (std::vector<FeatureSubset>{0}));
    auto all = EnumerateFeatureSubsets(width, width);
    ASSERT_TRUE(all.ok()) << all.status();
    ASSERT_EQ(all->size(), 1u);
    const uint64_t expected =
        width == 64 ? ~uint64_t{0} : (uint64_t{1} << width) - 1;
    EXPECT_EQ(all->front(), expected);
  }
  auto singles = EnumerateFeatureSubsets(64, 1);
  ASSERT_TRUE(singles.ok()) << singles.status();
  ASSERT_EQ(singles->size(), 64u);
  EXPECT_EQ(singles->back(), uint64_t{1} << 63);
}

TEST(FeatureSubsetTest, EverySmallCombinationIsUniqueAndHasExactCardinality) {
  for (int width = 1; width <= 10; ++width) {
    for (int cardinality = 0; cardinality <= width; ++cardinality) {
      auto result = EnumerateFeatureSubsets(width, cardinality);
      ASSERT_TRUE(result.ok()) << result.status();
      auto sorted = *result;
      std::sort(sorted.begin(), sorted.end());
      EXPECT_EQ(std::adjacent_find(sorted.begin(), sorted.end()), sorted.end());
      std::vector<FeatureSubset> brute_force;
      for (FeatureSubset mask = 0; mask < (FeatureSubset{1} << width); ++mask)
        if (std::popcount(mask) == cardinality)
          brute_force.push_back(mask);
      EXPECT_EQ(sorted, brute_force);
    }
  }
}

TEST(FeatureSubsetTest, SupportsPairsTriplesAndQuadsAtWidth64) {
  for (const auto& [cardinality, count] :
       {std::pair{2, 2016u}, std::pair{3, 41664u}, std::pair{4, 635376u}}) {
    auto result = EnumerateFeatureSubsets(64, cardinality);
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(result->size(), count);
    EXPECT_EQ(std::popcount(result->front()), cardinality);
    EXPECT_EQ(std::popcount(result->back()), cardinality);
  }
}

TEST(FeatureSubsetTest, RejectsInvalidAndExcessiveEnumerations) {
  for (const auto& [width, cardinality] :
       {std::pair{0, 0}, std::pair{65, 0}, std::pair{1, -1}, std::pair{1, 2}})
    EXPECT_EQ(EnumerateFeatureSubsets(width, cardinality).status().code(),
              absl::StatusCode::kInvalidArgument);
  // Check both symmetry and a huge central coefficient without overflowing.
  for (int cardinality : {5, 32, 59})
    EXPECT_EQ(EnumerateFeatureSubsets(64, cardinality).status().code(),
              absl::StatusCode::kResourceExhausted);
  auto nearly_full = EnumerateFeatureSubsets(64, 63);
  ASSERT_TRUE(nearly_full.ok()) << nearly_full.status();
  EXPECT_EQ(nearly_full->size(), 64u);
}

TEST(FeatureSubsetTest, PreservesKeptRawBitsAndDoesNotMutateSource) {
  const std::vector<uint16_t> row{0x8000, 0x7fc1, 0xff80, 0x3f80, 0x0000};
  auto result = ApplyBf16FeatureSubset(row, 0b00111);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(*result, (std::vector<uint16_t>{0x8000, 0x7fc1, 0xff80, 0, 0}));
  EXPECT_EQ(row, (std::vector<uint16_t>{0x8000, 0x7fc1, 0xff80, 0x3f80, 0}));
  auto zero = ApplyBf16FeatureSubset(row, 0);
  ASSERT_TRUE(zero.ok()) << zero.status();
  EXPECT_EQ(*zero, std::vector<uint16_t>(5, 0));
}

TEST(FeatureSubsetTest, MasksWidth64AndHighestBitExactly) {
  std::vector<uint16_t> row(64, 0x8000);
  row.back() = 0x7fff;
  auto highest = ApplyBf16FeatureSubset(row, uint64_t{1} << 63);
  ASSERT_TRUE(highest.ok()) << highest.status();
  EXPECT_EQ(highest->back(), 0x7fff);
  EXPECT_TRUE(std::all_of(highest->begin(), highest->end() - 1,
                          [](uint16_t value) { return value == 0; }));
  auto identity = ApplyBf16FeatureSubset(row, ~uint64_t{0});
  ASSERT_TRUE(identity.ok()) << identity.status();
  EXPECT_EQ(*identity, row);
}

TEST(FeatureSubsetTest, RejectsInvalidRowWidthsAndOutOfRowBits) {
  for (size_t width : {0u, 65u})
    EXPECT_EQ(
        ApplyBf16FeatureSubset(std::vector<uint16_t>(width), 0).status().code(),
        absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ApplyBf16FeatureSubset(std::vector<uint16_t>(63), uint64_t{1} << 63)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_FALSE(ApplyBf16FeatureSubset(std::vector<uint16_t>(1), 2).ok());
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
