#include <cstddef>
#include <limits>

#include "gtest/gtest.h"
#include "src/llm/experiments/one_shot_memorizer/quadratic_features.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

TEST(QuadraticFeaturesShapeTest, CountsColumnsAndSampleAxesWithoutAGpu) {
  static_assert(QuadraticFeaturesLayer::kOutputWidth == 16 + 16 * 17 / 2);
  for (int context : {1, 3, 1024})
    for (int batch : {1, 2, 31}) {
      const size_t rows = static_cast<size_t>(batch) * context;
      auto size = QuadraticFeaturesOutputBytes(rows * 16 * 2, context);
      ASSERT_TRUE(size.ok()) << size.status();
      EXPECT_EQ(*size, rows * 152 * 2);
    }
}

TEST(QuadraticFeaturesShapeTest, RejectsEmptyPartialAndMisalignedSamples) {
  EXPECT_FALSE(QuadraticFeaturesOutputBytes(0, 1).ok());
  EXPECT_FALSE(QuadraticFeaturesOutputBytes(31, 1).ok());
  EXPECT_FALSE(QuadraticFeaturesOutputBytes(33, 1).ok());
  EXPECT_FALSE(QuadraticFeaturesOutputBytes(32, 2).ok());
  EXPECT_FALSE(QuadraticFeaturesOutputBytes(5 * 32, 3).ok());
  EXPECT_FALSE(QuadraticFeaturesOutputBytes(32, 0).ok());
  EXPECT_FALSE(QuadraticFeaturesOutputBytes(32, -1).ok());
}

TEST(QuadraticFeaturesShapeTest, ChecksIntIndexBoundaryBeforeAllocation) {
  const size_t max_rows = std::numeric_limits<int>::max() / 152;
  auto boundary = QuadraticFeaturesOutputBytes(max_rows * 32, 1);
  ASSERT_TRUE(boundary.ok()) << boundary.status();
  EXPECT_EQ(*boundary, max_rows * 304);
  EXPECT_FALSE(QuadraticFeaturesOutputBytes((max_rows + 1) * 32, 1).ok());
  EXPECT_FALSE(
      QuadraticFeaturesOutputBytes(32, std::numeric_limits<int>::max()).ok());
  EXPECT_FALSE(
      QuadraticFeaturesOutputBytes(std::numeric_limits<size_t>::max(), 1).ok());
  const size_t large_aligned = std::numeric_limits<size_t>::max() / 32 * 32;
  EXPECT_FALSE(QuadraticFeaturesOutputBytes(large_aligned, 1).ok());
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
