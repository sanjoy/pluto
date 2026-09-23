#include "src/llm/experiments/one_shot_memorizer/token_codes.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

std::vector<uint32_t> SignMasks(const TokenCodes& codes) {
  std::vector<uint32_t> masks;
  for (int token = 0; token < codes.vocab_size; ++token) {
    uint32_t mask = 0;
    for (int coordinate = 0; coordinate < codes.width; ++coordinate)
      if (codes.values[token * codes.width + coordinate] > 0)
        mask |= uint32_t{1} << coordinate;
    masks.push_back(mask);
  }
  return masks;
}

std::vector<std::vector<float>> SortedRows(const TokenCodes& codes) {
  std::vector<std::vector<float>> rows;
  for (int token = 0; token < codes.vocab_size; ++token) {
    const auto begin = codes.values.begin() + token * codes.width;
    rows.emplace_back(begin, begin + codes.width);
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

TEST(TokenCodesTest, FullVocabularyIsBalancedDistinctAndExactlyRepresentable) {
  auto codes = MakeBalancedTokenCodes(4475);
  ASSERT_TRUE(codes.ok()) << codes.status();
  EXPECT_EQ(codes->width, 16);
  EXPECT_EQ(codes->vocab_size, 4475);
  ASSERT_EQ(codes->values.size(), 4475u * 16);
  for (int token = 0; token < codes->vocab_size; ++token) {
    double sum = 0, squared_norm = 0;
    int positives = 0;
    for (int coordinate = 0; coordinate < codes->width; ++coordinate) {
      const float value = codes->values[token * codes->width + coordinate];
      EXPECT_TRUE(value == 1 || value == -1);
      // The lower half of the FP32 word is zero: conversion to BF16 is exact.
      EXPECT_EQ(std::bit_cast<uint32_t>(value) & UINT32_C(0xffff), 0u);
      sum += value;
      squared_norm += value * value;
      positives += value > 0;
    }
    EXPECT_EQ(sum, 0);
    EXPECT_EQ(squared_norm, 16);
    EXPECT_EQ(positives, 8);
  }
  const auto masks = SignMasks(*codes);
  EXPECT_EQ(std::set<uint32_t>(masks.begin(), masks.end()).size(), 4475u);

  auto repeated = MakeBalancedTokenCodes(4475, 16, 0);
  auto other_seed = MakeBalancedTokenCodes(4475, 16, 1);
  ASSERT_TRUE(repeated.ok());
  ASSERT_TRUE(other_seed.ok());
  EXPECT_EQ(repeated->values, codes->values);
  EXPECT_NE(other_seed->values, codes->values);
}

TEST(TokenCodesTest, SmallCodebookHasSpecifiedCrossPlatformOrderAndCapacity) {
  auto codes = MakeBalancedTokenCodes(6, 4, 0);
  ASSERT_TRUE(codes.ok()) << codes.status();
  EXPECT_EQ(SignMasks(*codes), (std::vector<uint32_t>{10, 6, 12, 9, 3, 5}));
  auto prefix = MakeBalancedTokenCodes(3, 4, 0);
  ASSERT_TRUE(prefix.ok());
  EXPECT_EQ(prefix->values, (std::vector<float>(codes->values.begin(),
                                                codes->values.begin() + 12)));
  EXPECT_FALSE(MakeBalancedTokenCodes(7, 4).ok());
  EXPECT_TRUE(MakeBalancedTokenCodes(2, 2).ok());
  EXPECT_FALSE(MakeBalancedTokenCodes(3, 2).ok());
  EXPECT_TRUE(MakeBalancedTokenCodes(1, 20).ok());
  EXPECT_FALSE(MakeBalancedTokenCodes(184757, 20).ok());
}

TEST(TokenCodesTest, RejectsInvalidBalancedDimensionsAndVocabulary) {
  for (int width : {-2, 0, 1, 3, 19, 21, 22, std::numeric_limits<int>::max()})
    EXPECT_FALSE(MakeBalancedTokenCodes(1, width).ok()) << width;
  for (int vocabulary : {-1, 0, 12871, std::numeric_limits<int>::max()})
    EXPECT_FALSE(MakeBalancedTokenCodes(vocabulary, 16).ok()) << vocabulary;
}

TEST(TokenCodesTest, NormalizesEachRowsCoordinatesRatherThanAcrossTokens) {
  const std::vector<float> input = {1, 2, 3, 4, 12, 14, 16, 18, -1, -2, -3, -4};
  auto codes = NormalizeTokenCodes(input, 4);
  ASSERT_TRUE(codes.ok()) << codes.status();
  EXPECT_EQ(codes->width, 4);
  EXPECT_EQ(codes->vocab_size, 3);
  ASSERT_EQ(codes->values.size(), input.size());
  for (int row = 0; row < 3; ++row) {
    double sum = 0, squared_norm = 0;
    for (int coordinate = 0; coordinate < 4; ++coordinate) {
      const float value = codes->values[row * 4 + coordinate];
      sum += value;
      squared_norm += static_cast<double>(value) * value;
    }
    EXPECT_NEAR(sum, 0, 1e-6);
    EXPECT_NEAR(squared_norm, 4, 1e-6);
  }
  for (int coordinate = 0; coordinate < 4; ++coordinate) {
    EXPECT_FLOAT_EQ(codes->values[coordinate], codes->values[4 + coordinate]);
    EXPECT_FLOAT_EQ(codes->values[coordinate], -codes->values[8 + coordinate]);
    EXPECT_NEAR(
        codes->values[coordinate],
        (static_cast<double>(input[coordinate]) - 2.5) / std::sqrt(1.25), 1e-7);
  }
  EXPECT_EQ(input,
            (std::vector<float>{1, 2, 3, 4, 12, 14, 16, 18, -1, -2, -3, -4}));
}

TEST(TokenCodesTest, NormalizationHandlesOddWidthsAndExtremeFiniteValues) {
  auto odd = NormalizeTokenCodes(std::vector<float>{1, 2, 3}, 3);
  ASSERT_TRUE(odd.ok()) << odd.status();
  EXPECT_NEAR(odd->values[0], -std::sqrt(1.5), 1e-7);
  EXPECT_EQ(odd->values[1], 0);
  EXPECT_NEAR(odd->values[2], std::sqrt(1.5), 1e-7);
  const float largest = std::numeric_limits<float>::max();
  const float smallest = std::numeric_limits<float>::denorm_min();
  auto extremes = NormalizeTokenCodes(
      std::vector<float>{-largest, largest, -smallest, smallest}, 2);
  ASSERT_TRUE(extremes.ok()) << extremes.status();
  EXPECT_EQ(extremes->values, (std::vector<float>{-1, 1, -1, 1}));
}

TEST(TokenCodesTest, NormalizationRejectsMalformedNonfiniteAndConstantRows) {
  EXPECT_FALSE(NormalizeTokenCodes({}, 2).ok());
  EXPECT_FALSE(NormalizeTokenCodes(std::vector<float>{1, 2}, 0).ok());
  EXPECT_FALSE(NormalizeTokenCodes(std::vector<float>{1, 2}, -2).ok());
  EXPECT_FALSE(NormalizeTokenCodes(std::vector<float>{1, 2, 3}, 2).ok());
  EXPECT_FALSE(NormalizeTokenCodes(std::vector<float>{1, 2}, 1).ok());
  EXPECT_FALSE(NormalizeTokenCodes(std::vector<float>{0, 0}, 2).ok());
  EXPECT_FALSE(NormalizeTokenCodes(std::vector<float>{1, 2, 5, 5}, 2).ok());
  for (float nonfinite : {std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()})
    EXPECT_FALSE(NormalizeTokenCodes(std::vector<float>{1, nonfinite}, 2).ok());
}

TEST(TokenCodesTest, PermutationIsAReproducibleWholeRowBijection) {
  const TokenCodes original{
      .width = 3,
      .vocab_size = 6,
      .values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 0, 0, 0, -3, -2, -1}};
  const auto saved = original.values;
  auto permuted = PermuteTokenCodes(original, 0);
  auto repeated = PermuteTokenCodes(original, 0);
  auto other_seed = PermuteTokenCodes(original, 1);
  ASSERT_TRUE(permuted.ok()) << permuted.status();
  ASSERT_TRUE(repeated.ok());
  ASSERT_TRUE(other_seed.ok());
  EXPECT_EQ(permuted->width, original.width);
  EXPECT_EQ(permuted->vocab_size, original.vocab_size);
  EXPECT_EQ(SortedRows(*permuted), SortedRows(original));
  EXPECT_EQ(repeated->values, permuted->values);
  EXPECT_NE(permuted->values, other_seed->values);
  EXPECT_EQ(original.values, saved);
  // The same seed and population size use the same swaps as balanced masks.
  const std::vector<int> source_rows = {4, 2, 5, 3, 0, 1};
  for (size_t row = 0; row < source_rows.size(); ++row)
    for (int coordinate = 0; coordinate < original.width; ++coordinate)
      EXPECT_EQ(
          permuted->values[row * original.width + coordinate],
          original.values[source_rows[row] * original.width + coordinate]);
}

TEST(TokenCodesTest, PermutationAcceptsSingletonAndRejectsInvalidMetadata) {
  const TokenCodes singleton{.width = 2, .vocab_size = 1, .values = {3, 3}};
  auto permuted = PermuteTokenCodes(singleton, UINT64_MAX);
  ASSERT_TRUE(permuted.ok());
  EXPECT_EQ(permuted->values, singleton.values);
  EXPECT_FALSE(PermuteTokenCodes(TokenCodes{}).ok());
  EXPECT_FALSE(
      PermuteTokenCodes({.width = 0, .vocab_size = 1, .values = {1}}).ok());
  EXPECT_FALSE(
      PermuteTokenCodes({.width = -1, .vocab_size = 1, .values = {1}}).ok());
  EXPECT_FALSE(
      PermuteTokenCodes({.width = 2, .vocab_size = 0, .values = {1, 2}}).ok());
  EXPECT_FALSE(
      PermuteTokenCodes({.width = 2, .vocab_size = -1, .values = {1, 2}}).ok());
  EXPECT_FALSE(
      PermuteTokenCodes({.width = 2, .vocab_size = 2, .values = {1, 2}}).ok());
  EXPECT_FALSE(
      PermuteTokenCodes({.width = 2, .vocab_size = 1, .values = {1, 2, 3}})
          .ok());
  EXPECT_FALSE(PermuteTokenCodes({.width = std::numeric_limits<int>::max(),
                                  .vocab_size = std::numeric_limits<int>::max(),
                                  .values = {1}})
                   .ok());
  for (float nonfinite : {std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()})
    EXPECT_FALSE(PermuteTokenCodes(
                     {.width = 2, .vocab_size = 1, .values = {1, nonfinite}})
                     .ok());
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
