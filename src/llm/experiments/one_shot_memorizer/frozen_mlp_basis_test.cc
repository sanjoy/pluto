#include "src/llm/experiments/one_shot_memorizer/frozen_mlp_basis.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

std::vector<double> Evaluate(const FrozenMlpBasis& basis,
                             const std::vector<float>& inputs, int width,
                             int features) {
  std::vector<double> result;
  for (size_t row = 0; row < inputs.size() / width; ++row)
    for (int feature = 0; feature < features; ++feature) {
      double value = basis.bias[feature];
      for (int coordinate = 0; coordinate < width; ++coordinate)
        value += static_cast<double>(inputs[row * width + coordinate]) *
                 basis.weights[coordinate * features + feature];
      result.push_back(value);
    }
  return result;
}

TEST(FrozenMlpBasisTest, ComputesKnownPopulationMomentsAndLayout) {
  // Projections are [3,-2], [5,2], [7,6]: means [5,2], stddevs
  // [sqrt(8/3),sqrt(32/3)]. The matrix is [input,feature], not transposed.
  const std::vector<float> inputs{1, 2, 3, 2, 5, 2};
  const std::vector<float> weights{1, 2, 1, -2};
  auto result = StandardizeFrozenMlpBasis(inputs, weights, 2, 2);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->projection_means, (std::vector<double>{5, 2}));
  EXPECT_DOUBLE_EQ(result->projection_standard_deviations[0],
                   std::sqrt(8. / 3));
  EXPECT_DOUBLE_EQ(result->projection_standard_deviations[1],
                   std::sqrt(32. / 3));
  EXPECT_FLOAT_EQ(result->weights[0], 1 / std::sqrt(8. / 3));
  EXPECT_FLOAT_EQ(result->weights[1], 2 / std::sqrt(32. / 3));
  EXPECT_FLOAT_EQ(result->weights[2], 1 / std::sqrt(8. / 3));
  EXPECT_FLOAT_EQ(result->weights[3], -2 / std::sqrt(32. / 3));
  EXPECT_FLOAT_EQ(result->bias[0], -5 / std::sqrt(8. / 3));
  EXPECT_FLOAT_EQ(result->bias[1], -2 / std::sqrt(32. / 3));
  const auto values = Evaluate(*result, inputs, 2, 2);
  for (int feature = 0; feature < 2; ++feature) {
    double mean = 0, second_moment = 0;
    for (int row = 0; row < 3; ++row) {
      const double value = values[row * 2 + feature];
      mean += value / 3;
      second_moment += value * value / 3;
    }
    EXPECT_NEAR(mean, 0, 3e-7);
    EXPECT_NEAR(second_moment, 1, 3e-7);
  }
}

TEST(FrozenMlpBasisTest, UsesOnlySuppliedFittingRowsAndDoesNotMutateThem) {
  const std::vector<float> inputs{1, 3};
  const std::vector<float> weights{2};
  auto result = StandardizeFrozenMlpBasis(inputs, weights, 1, 1);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->weights, std::vector<float>{1});
  EXPECT_EQ(result->bias, std::vector<float>{-2});
  // A held-out row is evaluated with the already frozen basis, not included in
  // its statistics. Adding it to fitting would produce a different basis.
  EXPECT_EQ(Evaluate(*result, {101}, 1, 1), std::vector<double>{99});
  auto contaminated = StandardizeFrozenMlpBasis({1, 3, 101}, weights, 1, 1);
  ASSERT_TRUE(contaminated.ok()) << contaminated.status();
  EXPECT_NE(result->weights, contaminated->weights);
  EXPECT_EQ(inputs, (std::vector<float>{1, 3}));
  EXPECT_EQ(weights, std::vector<float>{2});
}

TEST(FrozenMlpBasisTest, PositiveDirectionScalingCancelsAndNegativeFlipsSign) {
  const std::vector<float> inputs{-2, 0, 3};
  auto original = StandardizeFrozenMlpBasis(inputs, {2}, 1, 1);
  auto positive = StandardizeFrozenMlpBasis(inputs, {16}, 1, 1);
  auto negative = StandardizeFrozenMlpBasis(inputs, {-2}, 1, 1);
  ASSERT_TRUE(original.ok()) << original.status();
  ASSERT_TRUE(positive.ok()) << positive.status();
  ASSERT_TRUE(negative.ok()) << negative.status();
  EXPECT_EQ(original->weights, positive->weights);
  EXPECT_EQ(original->bias, positive->bias);
  EXPECT_FLOAT_EQ(original->weights[0], -negative->weights[0]);
  EXPECT_FLOAT_EQ(original->bias[0], -negative->bias[0]);
}

TEST(FrozenMlpBasisTest, TranslatedInputsHaveSameStandardizedProjections) {
  auto original = StandardizeFrozenMlpBasis({-1, 1}, {4}, 1, 1);
  auto shifted = StandardizeFrozenMlpBasis({7, 9}, {4}, 1, 1);
  ASSERT_TRUE(original.ok()) << original.status();
  ASSERT_TRUE(shifted.ok()) << shifted.status();
  EXPECT_EQ(original->weights, shifted->weights);
  EXPECT_EQ(Evaluate(*original, {-1, 1}, 1, 1),
            Evaluate(*shifted, {7, 9}, 1, 1));
}

TEST(FrozenMlpBasisTest, ReorderingRowsPreservesMomentsWithinFp64Tolerance) {
  std::vector<float> inputs{100000, 1, -100000, 3, 0.25f, -0.125f, 17};
  auto original = StandardizeFrozenMlpBasis(inputs, {3}, 1, 1);
  ASSERT_TRUE(original.ok()) << original.status();
  std::reverse(inputs.begin(), inputs.end());
  auto reversed = StandardizeFrozenMlpBasis(inputs, {3}, 1, 1);
  ASSERT_TRUE(reversed.ok()) << reversed.status();
  EXPECT_NEAR(original->projection_means[0], reversed->projection_means[0],
              1e-12);
  EXPECT_NEAR(original->projection_standard_deviations[0],
              reversed->projection_standard_deviations[0], 1e-9);
  EXPECT_EQ(original->weights, reversed->weights);
  EXPECT_EQ(original->bias, reversed->bias);
  auto repeated = StandardizeFrozenMlpBasis(inputs, {3}, 1, 1);
  ASSERT_TRUE(repeated.ok()) << repeated.status();
  EXPECT_EQ(reversed->projection_means, repeated->projection_means);
  EXPECT_EQ(reversed->projection_standard_deviations,
            repeated->projection_standard_deviations);
}

TEST(FrozenMlpBasisTest, RejectsInvalidDimensionsAndShapes) {
  EXPECT_FALSE(StandardizeFrozenMlpBasis({1, 2}, {1}, 0, 1).ok());
  EXPECT_FALSE(StandardizeFrozenMlpBasis({1, 2}, {1}, 1, -1).ok());
  EXPECT_FALSE(StandardizeFrozenMlpBasis({}, {1}, 1, 1).ok());
  EXPECT_FALSE(StandardizeFrozenMlpBasis({1, 2, 3}, {1, 2}, 2, 1).ok());
  EXPECT_FALSE(StandardizeFrozenMlpBasis({1, 2}, {1}, 1, 2).ok());
  EXPECT_FALSE(StandardizeFrozenMlpBasis({1}, {1},
                                         std::numeric_limits<int>::max(),
                                         std::numeric_limits<int>::max())
                   .ok());
}

TEST(FrozenMlpBasisTest, RejectsZeroAndOtherConstantFittingProjections) {
  EXPECT_FALSE(StandardizeFrozenMlpBasis({1, 2}, {0}, 1, 1).ok());
  EXPECT_FALSE(StandardizeFrozenMlpBasis({1, 1}, {2}, 1, 1).ok());
  EXPECT_FALSE(StandardizeFrozenMlpBasis({1}, {2}, 1, 1).ok());
  EXPECT_FALSE(
      StandardizeFrozenMlpBasis(std::vector<float>(7, 1.3f), {1.7f}, 1, 1)
          .ok());
  EXPECT_FALSE(StandardizeFrozenMlpBasis({1, 2, 2, 1}, {1, 1}, 2, 1).ok());
  // A single constant feature rejects the entire basis, not only that column.
  EXPECT_FALSE(StandardizeFrozenMlpBasis({1, 2}, {1, 0}, 1, 2).ok());
}

TEST(FrozenMlpBasisTest, RejectsNonFiniteInputsOrDirections) {
  for (float value : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()}) {
    EXPECT_FALSE(StandardizeFrozenMlpBasis({value, 1}, {1}, 1, 1).ok());
    EXPECT_FALSE(StandardizeFrozenMlpBasis({1, 2}, {value}, 1, 1).ok());
  }
}

TEST(FrozenMlpBasisTest, RejectsUnrepresentableFp32Coefficients) {
  const float tiny = std::numeric_limits<float>::denorm_min();
  auto overflow = StandardizeFrozenMlpBasis({-tiny, tiny}, {1}, 1, 1);
  EXPECT_EQ(overflow.status().code(), absl::StatusCode::kOutOfRange);
  const float large = std::numeric_limits<float>::max();
  auto underflow = StandardizeFrozenMlpBasis({-large, -large, large, large},
                                             {tiny, large}, 2, 1);
  EXPECT_EQ(underflow.status().code(), absl::StatusCode::kOutOfRange);
}

TEST(FrozenMlpBasisTest, HandlesLargeProductsInDoubleWithoutFloatOverflow) {
  const float large = std::numeric_limits<float>::max();
  auto result = StandardizeFrozenMlpBasis({-large, large}, {large}, 1, 1);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(std::isfinite(result->projection_standard_deviations[0]));
  EXPECT_GT(result->weights[0], 0);
  const auto values = Evaluate(*result, {-large, large}, 1, 1);
  EXPECT_NEAR(values[0], -1, 1e-6);
  EXPECT_NEAR(values[1], 1, 1e-6);
}

TEST(FrozenMlpBasisTest, SupportsExperimentDimensions) {
  std::vector<float> inputs(16 * 3), weights(16 * 64);
  for (int row = 0; row < 3; ++row)
    for (int coordinate = 0; coordinate < 16; ++coordinate)
      inputs[row * 16 + coordinate] = row + coordinate;
  for (int coordinate = 0; coordinate < 16; ++coordinate)
    for (int feature = 0; feature < 64; ++feature)
      weights[coordinate * 64 + feature] = (feature + 1) * (coordinate + 1);
  auto result = StandardizeFrozenMlpBasis(inputs, weights, 16, 64);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->weights.size(), 1024u);
  EXPECT_EQ(result->bias.size(), 64u);
  EXPECT_EQ(result->projection_means.size(), 64u);
  EXPECT_EQ(result->projection_standard_deviations.size(), 64u);
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
