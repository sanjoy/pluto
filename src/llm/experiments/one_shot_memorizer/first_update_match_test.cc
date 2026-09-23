#include "src/llm/experiments/one_shot_memorizer/first_update_match.h"

#include <cmath>
#include <limits>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

TEST(FirstUpdateMatchTest, ZeroGradientsPreserveWeights) {
  auto result =
      ScoreFirstAdamUpdate({1, -2, 0}, {1, -2, 0}, {0, 0, 0}, .1f, 1e-8f);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->squared_error, 0);
  EXPECT_EQ(result->sign_mismatches, 0u);
  EXPECT_EQ(result->bit_equal_coordinates, 3u);
}

TEST(FirstUpdateMatchTest, PositiveNegativeAndSmallGradientsUseEpsilon) {
  auto result = ScoreFirstAdamUpdate({1, -1}, {.75f, -.75f}, {1, -1}, .5f, 1);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->squared_error, 0);
  EXPECT_EQ(result->sign_mismatches, 0u);
  EXPECT_EQ(result->bit_equal_coordinates, 2u);
  // Epsilon is not discarded, even for tiny gradients: g/(|g|+epsilon)=+/-1/2.
  result = ScoreFirstAdamUpdate({0, 0}, {-.125f, .125f}, {1e-8f, -1e-8f}, .25f,
                                1e-8f);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->squared_error, 0);
  EXPECT_EQ(result->sign_mismatches, 0u);
  EXPECT_EQ(result->bit_equal_coordinates, 2u);
}

TEST(FirstUpdateMatchTest, ComparesWeightUpdateSignsNotGradientSigns) {
  // Predictions are (-.5,+.5,0), but observations move (+.5,0,-.5).
  auto result =
      ScoreFirstAdamUpdate({0, 0, 0}, {.5f, 0, -.5f}, {1, -1, 0}, 1, 1);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->squared_error, 1.5);
  EXPECT_EQ(result->sign_mismatches, 3u);
  EXPECT_EQ(result->bit_equal_coordinates, 0u);
}

TEST(FirstUpdateMatchTest, RoundedAwayUpdateHasZeroSign) {
  auto result = ScoreFirstAdamUpdate({1}, {1}, {1}, 0x1p-30f, 1);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->squared_error, 0);
  EXPECT_EQ(result->sign_mismatches, 0u);
  EXPECT_EQ(result->bit_equal_coordinates, 1u);
}

TEST(FirstUpdateMatchTest, SignedZeroBitsDifferButUpdateSignsDoNot) {
  auto result = ScoreFirstAdamUpdate({0}, {-0.0f}, {0}, 1, 1);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->squared_error, 0);
  EXPECT_EQ(result->sign_mismatches, 0u);
  EXPECT_EQ(result->bit_equal_coordinates, 0u);
  result = ScoreFirstAdamUpdate({-0.0f}, {-0.0f}, {0}, 1, 1);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->bit_equal_coordinates, 1u);
}

TEST(FirstUpdateMatchTest, UsesFp32NormalizedGradient) {
  // The normalized gradient is stored as FP32 before the prediction. These
  // exact expected bits do not depend on final multiply/subtract contraction.
  const float observed = -1.0f / 3.0f;
  auto result = ScoreFirstAdamUpdate({0}, {observed}, {1}, 1, 2);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->squared_error, 0);
  EXPECT_EQ(result->bit_equal_coordinates, 1u);
}

TEST(FirstUpdateMatchTest, SquaredErrorSubtractsInDouble) {
  const float huge = 3e38f;
  auto result = ScoreFirstAdamUpdate({huge}, {-huge}, {0}, 1, 1);
  ASSERT_TRUE(result.ok()) << result.status();
  const double error = 2 * static_cast<double>(huge);
  EXPECT_EQ(result->squared_error, error * error);
  EXPECT_TRUE(std::isfinite(result->squared_error));
  EXPECT_EQ(result->sign_mismatches, 1u);
}

TEST(FirstUpdateMatchTest, RejectsEmptyAndMismatchedShapes) {
  EXPECT_EQ(ScoreFirstAdamUpdate({}, {}, {}, 1, 1).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ScoreFirstAdamUpdate({0}, {}, {0}, 1, 1).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ScoreFirstAdamUpdate({0}, {0}, {}, 1, 1).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ScoreFirstAdamUpdate({0, 1}, {0}, {0}, 1, 1).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(FirstUpdateMatchTest, RejectsNonfiniteInputsInEveryArray) {
  for (float bad : {std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN()}) {
    EXPECT_EQ(ScoreFirstAdamUpdate({bad}, {0}, {0}, 1, 1).status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(ScoreFirstAdamUpdate({0}, {bad}, {0}, 1, 1).status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(ScoreFirstAdamUpdate({0}, {0}, {bad}, 1, 1).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(FirstUpdateMatchTest, RejectsInvalidRateAndEpsilon) {
  for (float bad : {0.0f, -0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN()}) {
    EXPECT_EQ(ScoreFirstAdamUpdate({0}, {0}, {0}, bad, 1).status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(ScoreFirstAdamUpdate({0}, {0}, {0}, 1, bad).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(FirstUpdateMatchTest, RejectsIntermediateAndPredictionOverflow) {
  const float huge = std::numeric_limits<float>::max();
  EXPECT_EQ(ScoreFirstAdamUpdate({0}, {0}, {huge}, 1, huge).status().code(),
            absl::StatusCode::kOutOfRange);
  EXPECT_EQ(
      ScoreFirstAdamUpdate({huge}, {0}, {-1}, huge, 1e-8f).status().code(),
      absl::StatusCode::kOutOfRange);
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
