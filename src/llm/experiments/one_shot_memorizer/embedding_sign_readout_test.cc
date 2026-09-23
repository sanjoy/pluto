#include "src/llm/experiments/one_shot_memorizer/embedding_sign_readout.h"

#include <cmath>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

TEST(EmbeddingSignReadoutTest, UsesAllRowsAndSelectsOnlyNegativeScores) {
  const std::vector<float> initial(6, 0);
  const std::vector<float> trained{2, 0, 2, 0, -1, 0};
  auto result = ComputeEmbeddingSignReadout(initial, trained, 2);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->delta_fp32.mean, (std::vector<double>{1, 0}));
  EXPECT_EQ(result->delta_fp32.scores, (std::vector<double>{2, 2, -1}));
  EXPECT_EQ(result->delta_fp32.selected_ids, (std::vector<int>{2}));
  EXPECT_EQ(result->delta_bf16.selected_ids, result->delta_fp32.selected_ids);
  EXPECT_EQ(result->trained_fp32.scores, result->delta_fp32.scores);
  EXPECT_EQ(result->trained_bf16.scores, result->delta_fp32.scores);
}

TEST(EmbeddingSignReadoutTest, DistinguishesUpdatesFromTrainedWeights) {
  auto result = ComputeEmbeddingSignReadout({3, 3, 3}, {5, 5, 2}, 1);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->delta_fp32.selected_ids, (std::vector<int>{2}));
  EXPECT_TRUE(result->trained_fp32.selected_ids.empty());
  EXPECT_EQ(result->trained_fp32.mean, (std::vector<double>{4}));
}

TEST(EmbeddingSignReadoutTest, ZeroIsNotNegative) {
  auto result = ComputeEmbeddingSignReadout({0, 0}, {-1, 1}, 1);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->delta_fp32.scores, (std::vector<double>{0, 0}));
  EXPECT_TRUE(result->delta_fp32.selected_ids.empty());
  result = ComputeEmbeddingSignReadout({2, 3}, {2, 3}, 1);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->delta_fp32.selected_ids.empty());
}

TEST(EmbeddingSignReadoutTest, RoundsBf16EndpointsWithTiesToEven) {
  auto result =
      ComputeEmbeddingSignReadout({0, 0}, {1.00390625f, 1.01171875f}, 2);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->trained_bf16.mean, (std::vector<double>{1, 1.015625}));
  // Rounding the difference instead would keep a nonzero difference here.
  result = ComputeEmbeddingSignReadout({1}, {1.00390625f}, 1);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->delta_bf16.mean, (std::vector<double>{0}));
  EXPECT_GT(result->delta_fp32.mean[0], 0);
}

TEST(EmbeddingSignReadoutTest, RejectsMalformedShapes) {
  EXPECT_FALSE(ComputeEmbeddingSignReadout({}, {}, 2).ok());
  EXPECT_FALSE(ComputeEmbeddingSignReadout({1}, {1}, 0).ok());
  EXPECT_FALSE(ComputeEmbeddingSignReadout({1}, {1}, -1).ok());
  EXPECT_FALSE(ComputeEmbeddingSignReadout({1, 2, 3}, {1, 2, 3}, 2).ok());
  EXPECT_FALSE(ComputeEmbeddingSignReadout({1}, {1, 2}, 1).ok());
}

TEST(EmbeddingSignReadoutTest, RejectsNonfiniteEndpointsOnEitherSide) {
  for (float invalid : {std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
    EXPECT_EQ(ComputeEmbeddingSignReadout({invalid}, {0}, 1).status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(ComputeEmbeddingSignReadout({0}, {invalid}, 1).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(EmbeddingSignReadoutTest, RejectsOverflowInBf16Conversion) {
  const float huge = std::numeric_limits<float>::max();
  EXPECT_EQ(ComputeEmbeddingSignReadout({0}, {huge}, 1).status().code(),
            absl::StatusCode::kOutOfRange);
  EXPECT_EQ(ComputeEmbeddingSignReadout({-huge}, {0}, 1).status().code(),
            absl::StatusCode::kOutOfRange);
}

TEST(EmbeddingSignReadoutTest, SubtractsAndAccumulatesInDouble) {
  auto result = ComputeEmbeddingSignReadout({-3e38f}, {3e38f}, 1);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(std::isfinite(result->delta_fp32.scores[0]));
  EXPECT_GT(result->delta_fp32.mean[0], std::numeric_limits<float>::max());
  EXPECT_TRUE(std::isfinite(result->delta_bf16.scores[0]));
}

TEST(EmbeddingSignReadoutTest,
     CommonOrthogonalCoordinateChangePreservesScores) {
  auto original =
      ComputeEmbeddingSignReadout({0, 0, 0, 0, 0, 0}, {2, 1, 2, 0, -1, 0}, 2);
  // Rotate 90 degrees: (x, y) -> (-y, x), applied to every row.
  auto rotated =
      ComputeEmbeddingSignReadout({0, 0, 0, 0, 0, 0}, {-1, 2, 0, 2, 0, -1}, 2);
  ASSERT_TRUE(original.ok()) << original.status();
  ASSERT_TRUE(rotated.ok()) << rotated.status();
  EXPECT_EQ(original->delta_fp32.scores, rotated->delta_fp32.scores);
  EXPECT_EQ(original->delta_fp32.selected_ids,
            rotated->delta_fp32.selected_ids);
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
