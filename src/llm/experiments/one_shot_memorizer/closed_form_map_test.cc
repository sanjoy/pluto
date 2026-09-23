#include "src/llm/experiments/one_shot_memorizer/closed_form_map.h"

#include <cmath>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

TEST(ClosedFormMapTest, RecoversExactAffineCoefficientsAfterColumnPivoting) {
  const std::vector<float> x = {1,  0,  0,   0, 100, 0,    0,  0, 10, 1, 100,
                                10, -1, 200, 0, 2,   -100, 30, 0, 0,  0};
  const std::vector<double> weights = {2, -1, 3, 4, -2, 0.5};
  const std::vector<double> biases = {7, -3};
  std::vector<float> y;
  for (size_t row = 0; row < x.size() / 3; ++row) {
    for (int j = 0; j < 2; ++j) {
      double value = biases[j];
      for (int k = 0; k < 3; ++k)
        value += x[row * 3 + k] * weights[k * 2 + j];
      y.push_back(value);
    }
  }
  auto map = FitAffineMap(x, y, 3, 2);
  ASSERT_TRUE(map.ok()) << map.status();
  EXPECT_EQ(map->sample_count, 7u);
  EXPECT_EQ(map->numerical_rank, 3u);
  ASSERT_EQ(map->qr_diagonal_magnitudes.size(), 3u);
  for (double diagonal : map->qr_diagonal_magnitudes) {
    EXPECT_TRUE(std::isfinite(diagonal));
    EXPECT_GT(diagonal, 0);
  }
  ASSERT_EQ(map->weights.size(), weights.size());
  for (size_t i = 0; i < weights.size(); ++i)
    EXPECT_NEAR(map->weights[i], weights[i], 1e-11);
  for (size_t i = 0; i < biases.size(); ++i)
    EXPECT_NEAR(map->biases[i], biases[i], 1e-11);
  EXPECT_LT(map->relative_rmse, 1e-12);
  auto predictions = ApplyClosedFormMap(*map, std::vector<float>{3, -2, 1});
  ASSERT_TRUE(predictions.ok());
  EXPECT_NEAR((*predictions)[0], 5.0, 1e-10);
  EXPECT_NEAR((*predictions)[1], -13.5, 1e-10);
}

TEST(ClosedFormMapTest, Reconstructs64To16LinearProjectionWithAffineBias) {
  constexpr int p = 64, q = 16, n = 65;
  std::vector<float> x(n * p), y(n * q);
  for (int row = 0; row < p; ++row)
    x[row * p + row] = 1;
  for (int row = 0; row < n; ++row)
    for (int j = 0; j < q; ++j)
      y[row * q + j] =
          j / 16.0f + (row < p ? (row - 32) * (j + 1) / 128.0f : 0.0f);
  auto map = FitAffineMap(x, y, p, q);
  ASSERT_TRUE(map.ok()) << map.status();
  for (int k = 0; k < p; ++k)
    for (int j = 0; j < q; ++j)
      EXPECT_NEAR(map->weights[k * q + j], (k - 32) * (j + 1) / 128.0, 1e-11);
  for (int j = 0; j < q; ++j)
    EXPECT_NEAR(map->biases[j], j / 16.0, 1e-11);
  EXPECT_LT(map->relative_rmse, 1e-12);
}

TEST(ClosedFormMapTest, QrDiagonalsDescribePivotedDesignAndScaleWithInputs) {
  const std::vector<float> x = {-1, -2, -1, 2, 1, -2, 1, 2};
  const std::vector<float> y = {-5, 3, -3, 5};
  auto map = FitAffineMap(x, y, 2, 1);
  ASSERT_TRUE(map.ok()) << map.status();
  ASSERT_EQ(map->qr_diagonal_magnitudes.size(), 2u);
  // Orthogonal centered columns have norms 2 and 4: pivoting selects 4 first.
  EXPECT_NEAR(map->qr_diagonal_magnitudes[0], 4, 1e-12);
  EXPECT_NEAR(map->qr_diagonal_magnitudes[1], 2, 1e-12);
  auto scaled_x = x;
  for (float& value : scaled_x)
    value *= 3;
  auto scaled = FitAffineMap(scaled_x, y, 2, 1);
  ASSERT_TRUE(scaled.ok());
  for (size_t j = 0; j < 2; ++j)
    EXPECT_NEAR(scaled->qr_diagonal_magnitudes[j],
                3 * map->qr_diagonal_magnitudes[j], 1e-12);
}

TEST(ClosedFormMapTest, QrDiagonalsIncludeRidgeAndScaleWithSampleDuplication) {
  const std::vector<float> x = {-1, -2, -1, 2, 1, -2, 1, 2};
  const std::vector<float> y = {-5, 3, -3, 5};
  auto duplicated_x = x;
  auto duplicated_y = y;
  duplicated_x.insert(duplicated_x.end(), x.begin(), x.end());
  duplicated_y.insert(duplicated_y.end(), y.begin(), y.end());
  for (double ridge : {0.0, 0.25}) {
    AffineMapOptions options;
    options.ridge = ridge;
    auto map = FitAffineMap(x, y, 2, 1, options);
    auto duplicated = FitAffineMap(duplicated_x, duplicated_y, 2, 1, options);
    ASSERT_TRUE(map.ok()) << map.status();
    ASSERT_TRUE(duplicated.ok()) << duplicated.status();
    ASSERT_EQ(map->qr_diagonal_magnitudes.size(), 2u);
    ASSERT_EQ(duplicated->qr_diagonal_magnitudes.size(), 2u);
    EXPECT_NEAR(map->qr_diagonal_magnitudes[0], std::sqrt(16 + 4 * ridge),
                1e-12);
    EXPECT_NEAR(map->qr_diagonal_magnitudes[1], std::sqrt(4 + 4 * ridge),
                1e-12);
    for (size_t j = 0; j < 2; ++j)
      EXPECT_NEAR(duplicated->qr_diagonal_magnitudes[j],
                  std::sqrt(2.0) * map->qr_diagonal_magnitudes[j], 1e-12);
  }
}

TEST(ClosedFormMapTest, NonlinearTargetLeavesAuditableResidual) {
  auto map = FitAffineMap(std::vector<float>{-2, -1, 0, 1, 2},
                          std::vector<float>{4, 1, 0, 1, 4}, 1, 1);
  ASSERT_TRUE(map.ok());
  EXPECT_NEAR(map->weights[0], 0, 1e-12);
  EXPECT_NEAR(map->biases[0], 2, 1e-12);
  EXPECT_NEAR(map->rmse, std::sqrt(14.0 / 5), 1e-12);
  EXPECT_NEAR(map->target_rms, std::sqrt(34.0 / 5), 1e-12);
  EXPECT_NEAR(map->relative_rmse, std::sqrt(14.0 / 34), 1e-12);
}

TEST(ClosedFormMapTest,
     SingularDesignRequiresRidgeAndHasCorrectMinimumNormLimit) {
  const std::vector<float> x = {0, 0, 1, 2, 2, 4, 3, 6};
  const std::vector<float> y = {1, 4, 7, 10};
  EXPECT_FALSE(FitAffineMap(x, y, 2, 1).ok());
  AffineMapOptions options;
  options.ridge = 0.5;
  auto regularized = FitAffineMap(x, y, 2, 1, options);
  ASSERT_TRUE(regularized.ok()) << regularized.status();
  EXPECT_NEAR(regularized->weights[0], 5.0 / 9, 1e-12);
  EXPECT_NEAR(regularized->weights[1], 10.0 / 9, 1e-12);
  EXPECT_NEAR(regularized->biases[0], 4.0 / 3, 1e-12);
  options.ridge = 1e-10;
  auto tiny_ridge = FitAffineMap(x, y, 2, 1, options);
  ASSERT_TRUE(tiny_ridge.ok()) << tiny_ridge.status();
  EXPECT_NEAR(tiny_ridge->weights[0], 0.6, 1e-5);
  EXPECT_NEAR(tiny_ridge->weights[1], 1.2, 1e-5);
  EXPECT_LT(tiny_ridge->relative_rmse, 1e-8);
}

TEST(ClosedFormMapTest, RidgeUsesMeanLossAndDoesNotPenalizeTheBias) {
  AffineMapOptions options;
  options.ridge = 1;
  auto map = FitAffineMap(std::vector<float>{-1, 1}, std::vector<float>{3, 7},
                          1, 1, options);
  auto repeated =
      FitAffineMap(std::vector<float>{-1, 1, -1, 1},
                   std::vector<float>{103, 107, 103, 107}, 1, 1, options);
  ASSERT_TRUE(map.ok());
  ASSERT_TRUE(repeated.ok());
  EXPECT_NEAR(map->weights[0], 1, 1e-12);
  EXPECT_NEAR(map->biases[0], 5, 1e-12);
  EXPECT_NEAR(repeated->weights[0], 1, 1e-12);
  EXPECT_NEAR(repeated->biases[0], 105, 1e-12);
}

TEST(ClosedFormMapTest,
     RankToleranceIsConfigurableWithoutSquaringConditionNumber) {
  const float tiny = 1e-9f;
  const std::vector<float> x = {-1, -tiny, -1, tiny, 1, -tiny, 1, tiny};
  const std::vector<float> y = {-1, 1, -1, 1};
  AffineMapOptions options;
  options.relative_rank_tolerance = 1e-8;
  EXPECT_FALSE(FitAffineMap(x, y, 2, 1, options).ok());
  options.relative_rank_tolerance = 1e-12;
  auto map = FitAffineMap(x, y, 2, 1, options);
  ASSERT_TRUE(map.ok()) << map.status();
  EXPECT_LT(map->relative_rmse, 1e-12);
  EXPECT_NEAR(map->weights[0], 0, 1e-12);
  EXPECT_NEAR(map->weights[1] * tiny, 1, 1e-12);
}

TEST(ClosedFormMapTest, ZeroTargetsAndRidgeConstantDesignHaveDefinedMetrics) {
  auto zero = FitAffineMap(std::vector<float>{-1, 0, 1},
                           std::vector<float>{0, 0, 0}, 1, 1);
  ASSERT_TRUE(zero.ok());
  EXPECT_DOUBLE_EQ(zero->rmse, 0);
  EXPECT_DOUBLE_EQ(zero->target_rms, 0);
  EXPECT_DOUBLE_EQ(zero->relative_rmse, 0);
  EXPECT_FALSE(
      FitAffineMap(std::vector<float>{3, 7}, std::vector<float>{4, 8}, 2, 2)
          .ok());
  AffineMapOptions options;
  options.ridge = 0.01;
  auto constant = FitAffineMap(std::vector<float>{3, 7},
                               std::vector<float>{4, 8}, 2, 2, options);
  ASSERT_TRUE(constant.ok()) << constant.status();
  for (double weight : constant->weights)
    EXPECT_DOUBLE_EQ(weight, 0);
  EXPECT_EQ(constant->biases, (std::vector<double>{4, 8}));
  EXPECT_DOUBLE_EQ(constant->relative_rmse, 0);
  auto empty = ApplyClosedFormMap(*constant, {});
  ASSERT_TRUE(empty.ok());
  EXPECT_TRUE(empty->empty());
}

TEST(ClosedFormMapTest, RejectsInvalidShapesOptionsAndNonfiniteData) {
  const std::vector<float> x = {-1, 0, 1}, y = {0, 1, 2};
  EXPECT_FALSE(FitAffineMap({}, {}, 1, 1).ok());
  EXPECT_FALSE(FitAffineMap(x, y, 0, 1).ok());
  EXPECT_FALSE(FitAffineMap(x, y, 1, -1).ok());
  EXPECT_FALSE(FitAffineMap(x, y, 2, 1).ok());
  EXPECT_FALSE(FitAffineMap(x, y, 1, 2).ok());
  EXPECT_FALSE(
      FitAffineMap(
          std::vector<float>{-1, 0, std::numeric_limits<float>::quiet_NaN()}, y,
          1, 1)
          .ok());
  EXPECT_FALSE(
      FitAffineMap(
          x, std::vector<float>{0, 1, std::numeric_limits<float>::infinity()},
          1, 1)
          .ok());
  AffineMapOptions options;
  options.ridge = -1;
  EXPECT_FALSE(FitAffineMap(x, y, 1, 1, options).ok());
  options.ridge = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(FitAffineMap(x, y, 1, 1, options).ok());
  options.ridge = std::numeric_limits<double>::max();
  EXPECT_FALSE(FitAffineMap(x, y, 1, 1, options).ok());
  options.ridge = 0;
  options.relative_rank_tolerance = 1;
  EXPECT_FALSE(FitAffineMap(x, y, 1, 1, options).ok());
  options.relative_rank_tolerance = -1;
  EXPECT_FALSE(FitAffineMap(x, y, 1, 1, options).ok());
  options.relative_rank_tolerance = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(FitAffineMap(x, y, 1, 1, options).ok());
  auto map = FitAffineMap(x, y, 1, 1);
  ASSERT_TRUE(map.ok());
  auto corrupt = *map;
  corrupt.weights.clear();
  EXPECT_FALSE(ApplyClosedFormMap(corrupt, x).ok());
  corrupt = *map;
  corrupt.biases[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(ApplyClosedFormMap(corrupt, x).ok());
  corrupt = *map;
  corrupt.weights[0] = std::numeric_limits<double>::max();
  EXPECT_FALSE(ApplyClosedFormMap(corrupt, std::vector<float>{2}).ok());
  EXPECT_FALSE(
      ApplyClosedFormMap(
          *map, std::vector<float>{std::numeric_limits<float>::infinity()})
          .ok());
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
