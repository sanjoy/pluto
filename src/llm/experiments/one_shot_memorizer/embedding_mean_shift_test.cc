#include "src/llm/experiments/one_shot_memorizer/embedding_mean_shift.h"

#include <bit>
#include <cstdint>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

TEST(EmbeddingMeanShiftTest, UsesSeparateColumnMeansNotGlobalMean) {
  const std::vector<float> source{1, 20, 3, 40};
  const std::vector<float> reference{4, 15, 6, 35};
  const auto result = MatchEmbeddingMeans(source, reference, 2);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->shift, (std::vector<double>{3, -5}));
  EXPECT_EQ(result->values, reference);
  EXPECT_DOUBLE_EQ(result->squared_error_before, 68);
  EXPECT_DOUBLE_EQ(result->squared_error_after, 0);
  EXPECT_EQ(source, (std::vector<float>{1, 20, 3, 40}));
  EXPECT_EQ(reference, (std::vector<float>{4, 15, 6, 35}));
}

TEST(EmbeddingMeanShiftTest, RemovesMeanErrorNotCenteredRowDifferences) {
  const auto result = MatchEmbeddingMeans({1, 10, 5, 30}, {5, 16, 5, 16}, 2);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->shift, (std::vector<double>{2, -4}));
  EXPECT_EQ(result->values, (std::vector<float>{3, 6, 7, 26}));
  EXPECT_DOUBLE_EQ(result->squared_error_before, 248);
  EXPECT_DOUBLE_EQ(result->squared_error_after, 208);
}

TEST(EmbeddingMeanShiftTest, ZeroShiftIsByteIdenticalIncludingSignedZero) {
  const std::vector<float> source{-0.0f, 0.0f, -2.25f, 3.5f};
  const auto result = MatchEmbeddingMeans(source, source, 2);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->shift, (std::vector<double>{0, 0}));
  for (size_t i = 0; i < source.size(); ++i)
    EXPECT_EQ(std::bit_cast<uint32_t>(result->values[i]),
              std::bit_cast<uint32_t>(source[i]));
  EXPECT_DOUBLE_EQ(result->squared_error_before, 0);
  EXPECT_DOUBLE_EQ(result->squared_error_after, 0);
}

TEST(EmbeddingMeanShiftTest, UsesDoubleDifferencesAndOneOutputRounding) {
  // FP32 subtraction would erase the unit before the differences cancel.
  const auto result = MatchEmbeddingMeans({33554432, 0}, {1, 33554432}, 1);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->shift, (std::vector<double>{0.5}));
  EXPECT_EQ(result->values, (std::vector<float>{33554432, 0.5}));
}

TEST(EmbeddingMeanShiftTest, RejectsEmptyMismatchedAndNonrectangularShapes) {
  EXPECT_FALSE(MatchEmbeddingMeans({}, {}, 1).ok());
  EXPECT_FALSE(MatchEmbeddingMeans({1}, {1}, 0).ok());
  EXPECT_FALSE(MatchEmbeddingMeans({1, 2}, {1}, 1).ok());
  EXPECT_FALSE(MatchEmbeddingMeans({1, 2, 3}, {1, 2, 3}, 2).ok());
  EXPECT_FALSE(MatchEmbeddingMeans({1}, {1}, 2).ok());
}

TEST(EmbeddingMeanShiftTest, RejectsNonfiniteInputsAndFloatOverflow) {
  for (float invalid : {std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
    EXPECT_FALSE(MatchEmbeddingMeans({invalid}, {0}, 1).ok());
    EXPECT_FALSE(MatchEmbeddingMeans({0}, {invalid}, 1).ok());
  }
  const float maximum = std::numeric_limits<float>::max();
  const auto overflow =
      MatchEmbeddingMeans({maximum, -maximum}, {maximum, maximum}, 1);
  ASSERT_FALSE(overflow.ok());
  EXPECT_EQ(overflow.status().code(), absl::StatusCode::kOutOfRange);
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
