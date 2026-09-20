#include "src/llm/sampling.h"

#include <limits>
#include <random>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm {
namespace {

TEST(SamplingTest, ZeroTemperatureIsGreedyAndDoesNotAdvanceRandom) {
  const std::vector<float> logits{-3.0f, 8.0f, 2.0f};
  std::mt19937 random(17);
  const auto initial = random;
  for (int repeat = 0; repeat < 100; ++repeat) {
    auto token = SelectNextToken(logits, 0.0, random);
    ASSERT_TRUE(token.ok()) << token.status();
    EXPECT_EQ(*token, 1);
    EXPECT_EQ(random, initial);
  }
}

TEST(SamplingTest, GreedyTiesUseLowestLogicalTokenId) {
  std::mt19937 random(100);
  EXPECT_EQ(*SelectNextToken({-1.0f, 4.0f, 4.0f, 4.0f}, 0.0, random), 1);
  EXPECT_EQ(*SelectNextToken({-0.0f, +0.0f}, -0.0, random), 0);
  EXPECT_EQ(*SelectNextToken({7.0f}, 0.0, random), 0);
}

TEST(SamplingTest, GreedyIsIndependentOfSeedAndPreviousSampling) {
  std::mt19937 a(17), b(991);
  const std::vector<float> logits{2.0f, -3.0f, 7.0f, 7.0f};
  for (int repeat = 0; repeat < 50; ++repeat) {
    ASSERT_TRUE(SelectNextToken(logits, 0.8, a).ok());
    EXPECT_EQ(*SelectNextToken(logits, 0.0, a),
              *SelectNextToken(logits, 0.0, b));
  }
}

TEST(SamplingTest, PositiveTemperatureReplaysWithTheSameSeed) {
  std::mt19937 a(17), b(17);
  const std::vector<float> logits{0.1f, 0.3f, -0.2f, 0.4f};
  for (int repeat = 0; repeat < 100; ++repeat) {
    EXPECT_EQ(*SelectNextToken(logits, 0.8, a),
              *SelectNextToken(logits, 0.8, b));
  }
  EXPECT_EQ(a, b);
}

TEST(SamplingTest, MaskedTokensAreNeverChosen) {
  std::mt19937 random(17);
  const float masked = -std::numeric_limits<float>::infinity();
  for (double temperature : {0.0, 0.8, 100.0}) {
    for (int repeat = 0; repeat < 20; ++repeat) {
      EXPECT_EQ(*SelectNextToken({masked, 1.0f, masked}, temperature, random),
                1);
    }
  }
}

TEST(SamplingTest, RejectsInvalidInputsWithoutConsumingRandom) {
  std::mt19937 random(17);
  const auto initial = random;
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (const std::vector<float>& logits : {std::vector<float>{},
                                           {-inf, -inf},
                                           {0.0f, inf},
                                           {nan, 0.0f},
                                           {0.0f, nan}}) {
    EXPECT_FALSE(SelectNextToken(logits, 0.0, random).ok());
    EXPECT_FALSE(SelectNextToken(logits, 1.0, random).ok());
  }
  for (double invalid : {-1.0, std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_FALSE(SelectNextToken({1.0f}, invalid, random).ok());
    EXPECT_FALSE(ValidateGenerationOptions(1, invalid).ok());
  }
  EXPECT_FALSE(ValidateGenerationOptions(-1, 0.0).ok());
  EXPECT_TRUE(ValidateGenerationOptions(0, 0.0).ok());
  EXPECT_EQ(random, initial);
}

TEST(SamplingTest, ExtremeFiniteLogitsAndTinyTemperatureAreWellDefined) {
  std::mt19937 random(17);
  const float largest = std::numeric_limits<float>::max();
  EXPECT_EQ(*SelectNextToken({-largest, largest},
                             std::numeric_limits<double>::min(), random),
            1);
}

}  // namespace
}  // namespace pluto::llm
