#include "src/llm/experiments/one_shot_memorizer/adam_interaction.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

// Independent iterative definition: update both moments, apply both bias
// corrections, and then subtract the update. It never uses the closed form.
double Replay(double a, double b, const TwoStepAdamConfig& config) {
  double weight = 0, first = 0, second = 0;
  int step = 0;
  for (double gradient : {a, b}) {
    ++step;
    first = config.beta1 * first + (1 - config.beta1) * gradient;
    second = config.beta2 * second + (1 - config.beta2) * gradient * gradient;
    const double rate = step == 1 ? 6e-6 : config.second_rate;
    weight -= rate * (first / (1 - std::pow(config.beta1, step))) /
              (std::sqrt(second / (1 - std::pow(config.beta2, step))) +
               config.epsilon);
  }
  return weight;
}

TEST(AdamInteractionTest, MatchesIndependentIterativeDefinition) {
  const std::array<float, 13> gradients{0,
                                        -0.0f,
                                        1e-40f,
                                        -1e-20f,
                                        1e-9f,
                                        -1e-8f,
                                        0.001f,
                                        -0.3f,
                                        1,
                                        -1,
                                        7,
                                        -100,
                                        std::numeric_limits<float>::max()};
  for (double beta1 : {0.0, 0.5, static_cast<double>(0.9f)})
    for (double beta2 : {0.1, 0.7, static_cast<double>(0.99f)})
      for (double epsilon : {1e-8, 0.1}) {
        const TwoStepAdamConfig config{beta1, beta2, epsilon, 1.2e-5};
        for (float a : gradients)
          for (float b : gradients) {
            const auto result = ExplainTwoStepAdamInteraction(a, b, config);
            ASSERT_TRUE(result.ok()) << result.status();
            const double expected =
                Replay(a, b, config) -
                (Replay(a, 0, config) + Replay(0, b, config));
            EXPECT_NEAR(result->difference, expected, 2e-18)
                << a << "," << b << "," << beta1 << "," << beta2;
          }
      }
}

TEST(AdamInteractionTest, OneAbsentFactHasNoInteraction) {
  const TwoStepAdamConfig config{.second_rate = 1.2e-5};
  for (float value : {-2.0f, 0.0f, 1e-30f, 1.0f})
    for (bool first : {false, true}) {
      const auto result = ExplainTwoStepAdamInteraction(
          first ? value : 0, first ? 0 : value, config);
      ASSERT_TRUE(result.ok());
      EXPECT_DOUBLE_EQ(result->difference, 0);
    }
}

TEST(AdamInteractionTest, SignedTermsCanCancel) {
  const auto result = ExplainTwoStepAdamInteraction(
      1, -1, {.beta1 = 0.9, .beta2 = 0.99, .second_rate = 1.2e-5});
  ASSERT_TRUE(result.ok());
  EXPECT_GT(result->a_term, 0);
  EXPECT_LT(result->b_term, 0);
  EXPECT_LT(std::abs(result->difference), std::abs(result->a_term));
  EXPECT_LT(std::abs(result->difference), std::abs(result->b_term));
}

TEST(AdamInteractionTest, FirstMomentMomentumIsNotRequired) {
  const auto result = ExplainTwoStepAdamInteraction(
      1, 1, {.beta1 = 0, .beta2 = 0.99, .second_rate = 1.2e-5});
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->a_term, 0);
  EXPECT_GT(result->b_term, 0);
}

TEST(AdamInteractionTest, FixedSharedDenominatorRestoresLinearity) {
  // This is the exact-real linear alternative, not another trained model.
  // The scale may vary by time/coordinate, but not by which facts contributed.
  const std::array<double, 4> a{2, -3, 0, 8}, b{-4, 5, 9, 0};
  auto replay = [&](bool use_a, bool use_b, double beta) {
    double first = 0, result = 0;
    for (int step = 0; step < 4; ++step) {
      const double gradient = (use_a ? a[step] : 0) + (use_b ? b[step] : 0);
      first = beta * first + (1 - beta) * gradient;
      result -= (step + 1) * first / std::pow(2.0, step + 1);
    }
    return result;
  };
  for (double beta : {0.0, 0.5})
    EXPECT_EQ(replay(true, true, beta),
              replay(true, false, beta) + replay(false, true, beta));
}

TEST(AdamInteractionTest, RejectsInvalidInputsAndOverflow) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  const TwoStepAdamConfig valid{.second_rate = 1.2e-5};
  EXPECT_FALSE(ExplainTwoStepAdamInteraction(nan, 1, valid).ok());
  EXPECT_FALSE(ExplainTwoStepAdamInteraction(1, infinity, valid).ok());
  for (double bad : {-1.0, 1.0, nan, infinity}) {
    auto config = valid;
    config.beta1 = bad;
    EXPECT_FALSE(ExplainTwoStepAdamInteraction(1, 1, config).ok());
    config = valid;
    config.beta2 = bad;
    EXPECT_FALSE(ExplainTwoStepAdamInteraction(1, 1, config).ok());
  }
  for (double bad : {-1.0, 0.0, nan, infinity}) {
    auto config = valid;
    config.epsilon = bad;
    EXPECT_FALSE(ExplainTwoStepAdamInteraction(1, 1, config).ok());
    config = valid;
    config.second_rate = bad;
    EXPECT_FALSE(ExplainTwoStepAdamInteraction(1, 1, config).ok());
  }
  EXPECT_FALSE(ExplainTwoStepAdamInteraction(
                   std::numeric_limits<float>::max(), 1,
                   {.beta2 = 0, .epsilon = 1e-308, .second_rate = 1})
                   .ok());
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
