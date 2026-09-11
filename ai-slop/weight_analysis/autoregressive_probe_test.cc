#include "ai-slop/weight_analysis/autoregressive_probe.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::weight_analysis {
namespace {

TEST(AutoregressiveProbeOptionsTest, ValidatesBeforeAnyGeneration) {
  EXPECT_TRUE(ValidateGenerationOptions(0, .8, 17).ok());
  EXPECT_TRUE(ValidateGenerationOptions(512, 1, 0).ok());
  EXPECT_TRUE(ValidateGenerationOptions(1, std::numeric_limits<double>::min(),
                                        std::numeric_limits<int>::max() - 1)
                  .ok());
  EXPECT_FALSE(ValidateGenerationOptions(-1, .8, 17).ok());
  EXPECT_FALSE(ValidateGenerationOptions(1, .8, -1).ok());
  EXPECT_FALSE(
      ValidateGenerationOptions(1, .8, std::numeric_limits<int>::max()).ok());
  for (double temperature : {0., -1., std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_FALSE(ValidateGenerationOptions(1, temperature, 17).ok());
  }
}

TEST(AutoregressiveProbeContextTest, PadsWithLastIdAndSelectsLastRealRow) {
  const std::vector<int> history{8, 3, 11};
  std::array<int, 8> output;
  auto window = FillPredictionInput(history, 8, 20, absl::MakeSpan(output));
  ASSERT_TRUE(window.ok()) << window.status();
  EXPECT_EQ(window->start, 0);
  EXPECT_EQ(window->length, 3);
  EXPECT_EQ(window->output_row, 2);
  EXPECT_EQ(output, (std::array<int, 8>{8, 3, 11, 11, 11, 11, 11, 11}));
  EXPECT_EQ(history, (std::vector<int>{8, 3, 11}));
}

TEST(AutoregressiveProbeContextTest, SlidesExactlyAtContextBoundary) {
  std::vector<int> history{0, 1, 2, 3};
  std::array<int, 4> output;
  auto exact = FillPredictionInput(history, 4, 20, absl::MakeSpan(output));
  ASSERT_TRUE(exact.ok());
  EXPECT_EQ(exact->start, 0);
  EXPECT_EQ(exact->length, 4);
  EXPECT_EQ(exact->output_row, 3);
  history.push_back(4);
  auto slid = FillPredictionInput(history, 4, 20, absl::MakeSpan(output));
  ASSERT_TRUE(slid.ok());
  EXPECT_EQ(slid->start, 1);
  EXPECT_EQ(output, (std::array<int, 4>{1, 2, 3, 4}));
  history.insert(history.end(), {5, 6, 7, 8});
  auto later = FillPredictionInput(history, 4, 20, absl::MakeSpan(output));
  ASSERT_TRUE(later.ok());
  EXPECT_EQ(later->start, 5);
  EXPECT_EQ(output, (std::array<int, 4>{5, 6, 7, 8}));
  EXPECT_EQ(history.size(), 9);
}

TEST(AutoregressiveProbeContextTest,
     RejectsInvalidIdsEvenOutsideCroppedWindow) {
  std::array<int, 4> output{90, 90, 90, 90};
  for (const auto& history : std::vector<std::vector<int>>{
           {}, {-1}, {10}, {-1, 0, 1, 2, 3, 4}, {0, 1, 2, 3, 4, 10}}) {
    EXPECT_FALSE(
        FillPredictionInput(history, 4, 10, absl::MakeSpan(output)).ok());
    EXPECT_EQ(output, (std::array<int, 4>{90, 90, 90, 90}));
  }
  const std::array<int, 1> history{0};
  EXPECT_FALSE(
      FillPredictionInput(history, 0, 10, absl::MakeSpan(output)).ok());
  EXPECT_FALSE(
      FillPredictionInput(history, 5, 10, absl::MakeSpan(output)).ok());
  EXPECT_FALSE(FillPredictionInput(history, 4, 1, absl::MakeSpan(output)).ok());
}

TEST(AutoregressiveProbeSamplingTest,
     ExactlyMatchesUninstrumentedProductionExpression) {
  for (int vocabulary : {2, 17, 50257}) {
    std::vector<float> logits(vocabulary);
    for (int token = 0; token < vocabulary; ++token)
      logits[token] = static_cast<float>((token * 137) % 997 - 498) / 37.0f;
    for (double temperature : {.0001, .8, 1., 5.}) {
      for (uint32_t seed : {0u, 18u, 12345u}) {
        std::mt19937 actual_random(seed), expected_random(seed);
        for (int step = 0; step < 12; ++step) {
          const float maximum = *std::max_element(logits.begin(), logits.end());
          std::vector<double> probabilities(vocabulary);
          for (int token = 0; token < vocabulary; ++token) {
            probabilities[token] =
                std::exp((logits[token] - maximum) / temperature);
          }
          std::discrete_distribution<int> production(probabilities.begin(),
                                                     probabilities.end());
          const int expected = production(expected_random);
          auto measured =
              SampleProductionLogits(logits, temperature, actual_random);
          ASSERT_TRUE(measured.ok()) << measured.status();
          EXPECT_EQ(measured->token_id, expected);
          EXPECT_EQ(actual_random, expected_random);
          EXPECT_DOUBLE_EQ(measured->sampled_probability,
                           production.probabilities()[expected]);
          EXPECT_LE(measured->uniform, measured->cdf_upper);
          EXPECT_GE(measured->uniform, measured->cdf_lower);
          EXPECT_EQ(measured->raw_logit, logits[expected]);
        }
      }
    }
  }
}

TEST(AutoregressiveProbeSamplingTest,
     RecordsExactTwoWordUniformAndStableTieRanks) {
  const std::array<float, 5> logits{0, 0, 0, 0, 0};
  std::mt19937 random(18);
  for (int step = 0; step < 100; ++step) {
    const auto before = random;
    auto measured = SampleProductionLogits(logits, .8, random);
    ASSERT_TRUE(measured.ok()) << measured.status();
    auto expected = before;
    const uint32_t low = static_cast<uint32_t>(expected());
    const uint32_t high = static_cast<uint32_t>(expected());
    EXPECT_EQ(measured->rng_words[0], low);
    EXPECT_EQ(measured->rng_words[1], high);
    const double uniform =
        (static_cast<double>(low) + static_cast<double>(high) * 4294967296.0) /
        18446744073709551616.0;
    EXPECT_DOUBLE_EQ(measured->uniform, uniform);
    EXPECT_EQ(measured->raw_rank, measured->token_id + 1);
    EXPECT_EQ(measured->raw_argmax_token_id, 0);
    EXPECT_DOUBLE_EQ(measured->sampled_probability, .2);
  }
}

TEST(AutoregressiveProbeSamplingTest,
     ExtremeFiniteInputsAndUnderflowRemainValid) {
  const std::array<float, 4> logits{-std::numeric_limits<float>::max(), -1e20f,
                                    0, std::numeric_limits<float>::max()};
  std::mt19937 random(18);
  auto measured = SampleProductionLogits(logits, .8, random);
  ASSERT_TRUE(measured.ok()) << measured.status();
  EXPECT_EQ(measured->token_id, 3);
  EXPECT_EQ(measured->raw_rank, 1);
  EXPECT_DOUBLE_EQ(measured->sampled_probability, 1);
}

TEST(AutoregressiveProbeSamplingTest,
     InvalidInputsNeverAdvanceTheRandomEngine) {
  for (const auto& logits : std::vector<std::vector<float>>{
           {},
           {1},
           {0, std::numeric_limits<float>::infinity()},
           {0, -std::numeric_limits<float>::infinity()},
           {std::numeric_limits<float>::quiet_NaN(), 0}}) {
    std::mt19937 random(18), before(random);
    EXPECT_FALSE(SampleProductionLogits(logits, .8, random).ok());
    EXPECT_EQ(random, before);
  }
  const std::array<float, 2> logits{0, 1};
  for (double temperature : {0., -1., std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
    std::mt19937 random(18), before(random);
    EXPECT_FALSE(SampleProductionLogits(logits, temperature, random).ok());
    EXPECT_EQ(random, before);
  }
}

}  // namespace
}  // namespace pluto::weight_analysis
