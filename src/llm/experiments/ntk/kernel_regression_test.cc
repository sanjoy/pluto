#include "src/llm/experiments/ntk/kernel_regression.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace pluto::llm::ntk {
namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kMaximum = std::numeric_limits<double>::max();

TEST(KernelMatrixTest, IndexingUsesRowMajorStorage) {
  Matrix matrix{2, 3, {1, 2, 3, 4, 5, 6}};
  EXPECT_TRUE(ValidateMatrix(matrix).ok());
  EXPECT_DOUBLE_EQ(matrix(1, 0), 4);
  matrix(0, 2) = 7;
  const Matrix& immutable = matrix;
  EXPECT_DOUBLE_EQ(immutable(0, 2), 7);
  EXPECT_DOUBLE_EQ(matrix.values[2], 7);
}

TEST(KernelMatrixTest, AcceptsEmptyAndRectangularMatrices) {
  EXPECT_TRUE(ValidateMatrix({}).ok());
  EXPECT_TRUE(ValidateMatrix({0, 3, {}}).ok());
  EXPECT_TRUE(ValidateMatrix({3, 0, {}}).ok());
  EXPECT_TRUE(ValidateMatrix({1, 2, {1, 2}}).ok());
}

TEST(KernelMatrixTest, RejectsStorageMismatchAndOverflowBeforeIndexing) {
  for (const Matrix& invalid :
       std::vector<Matrix>{{1, 2, {1}},
                           {1, 1, {1, 2}},
                           {0, 2, {1}},
                           {std::numeric_limits<size_t>::max(), 2, {}},
                           {2, std::numeric_limits<size_t>::max(), {}}})
    EXPECT_TRUE(absl::IsInvalidArgument(ValidateMatrix(invalid)));
}

TEST(KernelMatrixTest, RejectsNonfiniteEntries) {
  for (double invalid : {kNaN, kInfinity, -kInfinity})
    EXPECT_TRUE(absl::IsInvalidArgument(ValidateMatrix({1, 1, {invalid}})));
}

TEST(KernelRidgeTest, FitsResidualAroundNonzeroInitialFunction) {
  const Matrix kernel{2, 2, {2, 1, 1, 2}};
  const std::vector<double> initial{1, -2};
  auto coefficients = FitRidge(kernel, initial, {4, -5}, 1);
  ASSERT_TRUE(coefficients.ok()) << coefficients.status();
  ASSERT_EQ(coefficients->size(), 2);
  EXPECT_NEAR((*coefficients)[0], 1.5, 1e-14);
  EXPECT_NEAR((*coefficients)[1], -1.5, 1e-14);

  auto training_predictions = Predict(kernel, initial, *coefficients);
  ASSERT_TRUE(training_predictions.ok()) << training_predictions.status();
  EXPECT_NEAR((*training_predictions)[0], 2.5, 1e-14);
  EXPECT_NEAR((*training_predictions)[1], -3.5, 1e-14);

  const Matrix cross_kernel{3, 2, {1, 0, 0, 2, -1, 1}};
  auto held_out = Predict(cross_kernel, {10, -1, 0}, *coefficients);
  ASSERT_TRUE(held_out.ok()) << held_out.status();
  EXPECT_NEAR((*held_out)[0], 11.5, 1e-14);
  EXPECT_NEAR((*held_out)[1], -4, 1e-14);
  EXPECT_NEAR((*held_out)[2], -3, 1e-14);
}

TEST(KernelRidgeTest, RidgeUsesSumLossRatherThanMeanLossScaling) {
  auto coefficients = FitRidge({2, 2, {2, 0, 0, 2}}, {2, 2}, {8, 8}, 2);
  ASSERT_TRUE(coefficients.ok()) << coefficients.status();
  EXPECT_DOUBLE_EQ((*coefficients)[0], 1.5);
  EXPECT_DOUBLE_EQ((*coefficients)[1], 1.5);
}

TEST(KernelRidgeTest, HandlesSingularPositiveSemidefiniteKernel) {
  const Matrix kernel{2, 2, {1, 1, 1, 1}};
  auto coefficients = FitRidge(kernel, {0, 0}, {1, 3}, 0.5);
  ASSERT_TRUE(coefficients.ok()) << coefficients.status();
  EXPECT_NEAR((*coefficients)[0], -1.2, 1e-14);
  EXPECT_NEAR((*coefficients)[1], 2.8, 1e-14);
  auto predictions = Predict(kernel, {0, 0}, *coefficients);
  ASSERT_TRUE(predictions.ok()) << predictions.status();
  EXPECT_NEAR((*predictions)[0], 1.6, 1e-14);
  EXPECT_NEAR((*predictions)[1], 1.6, 1e-14);
}

TEST(KernelRidgeTest, ZeroKernelCannotChangeInitialFunction) {
  const Matrix kernel{2, 2, {0, 0, 0, 0}};
  auto coefficients = FitRidge(kernel, {2, -3}, {3, 4}, 0.5);
  ASSERT_TRUE(coefficients.ok()) << coefficients.status();
  EXPECT_NEAR((*coefficients)[0], 2, 1e-14);
  EXPECT_NEAR((*coefficients)[1], 14, 1e-14);
  auto predictions = Predict(kernel, {2, -3}, *coefficients);
  ASSERT_TRUE(predictions.ok()) << predictions.status();
  EXPECT_EQ(*predictions, (std::vector<double>{2, -3}));
}

TEST(KernelRidgeTest, ChecksPositiveDefinitenessAfterRegularization) {
  const Matrix indefinite{2, 2, {1, 2, 2, 1}};
  auto insufficient = FitRidge(indefinite, {0, 0}, {1, -1}, 0.25);
  EXPECT_TRUE(absl::IsInvalidArgument(insufficient.status()));

  // The API certifies the solve, not whether this is a legitimate Gram matrix.
  auto sufficient = FitRidge(indefinite, {0, 0}, {1, -1}, 2);
  ASSERT_TRUE(sufficient.ok()) << sufficient.status();
  EXPECT_NEAR((*sufficient)[0], 1, 1e-14);
  EXPECT_NEAR((*sufficient)[1], -1, 1e-14);
}

TEST(KernelRidgeTest, AveragesTinyRelativeSymmetryRoundoff) {
  const double upper = 1 + 8 * std::numeric_limits<double>::epsilon();
  const double average = 1 + (upper - 1) * 0.5;
  auto coefficients = FitRidge({2, 2, {2, upper, 1, 2}}, {0, 0}, {1, 3}, 1);
  ASSERT_TRUE(coefficients.ok()) << coefficients.status();
  EXPECT_NEAR(3 * (*coefficients)[0] + average * (*coefficients)[1], 1, 1e-14);
  EXPECT_NEAR(average * (*coefficients)[0] + 3 * (*coefficients)[1], 3, 1e-14);
}

TEST(KernelFitValidationTest, RequiresNonemptySquareSymmetricTrainingKernel) {
  for (const Matrix& invalid : std::vector<Matrix>{
           {},
           {1, 2, {1, 2}},
           {2, 2, {1, 2, 0, 1}},
           // Relative validation must not treat tiny but grossly asymmetric
           // entries as zero merely because their absolute size is small.
           {2, 2, {1e-200, 1e-200, 0, 1e-200}},
           {2, 2, {1, 2, 2}},
           {1, 1, {kNaN}}}) {
    const std::vector<double> values(invalid.rows);
    EXPECT_TRUE(
        absl::IsInvalidArgument(FitRidge(invalid, values, values, 1).status()));
    EXPECT_TRUE(absl::IsInvalidArgument(
        FitGradientDescent(invalid, values, values, 1, 1).status()));
  }
}

TEST(KernelFitValidationTest, RejectsMismatchedAndNonfiniteVectors) {
  const Matrix kernel{2, 2, {1, 0, 0, 1}};
  for (const std::vector<double>& invalid : std::vector<std::vector<double>>{
           {}, {0}, {0, 0, 0}, {kNaN, 0}, {0, kInfinity}}) {
    EXPECT_TRUE(
        absl::IsInvalidArgument(FitRidge(kernel, invalid, {0, 0}, 1).status()));
    EXPECT_TRUE(
        absl::IsInvalidArgument(FitRidge(kernel, {0, 0}, invalid, 1).status()));
    EXPECT_TRUE(absl::IsInvalidArgument(
        FitGradientDescent(kernel, invalid, {0, 0}, 1, 1).status()));
    EXPECT_TRUE(absl::IsInvalidArgument(
        FitGradientDescent(kernel, {0, 0}, invalid, 1, 1).status()));
  }
}

TEST(KernelFitValidationTest, RejectsInvalidRidgeRateAndStepCount) {
  const Matrix kernel{1, 1, {1}};
  for (double invalid : {0.0, -1.0, kNaN, kInfinity, -kInfinity}) {
    EXPECT_TRUE(
        absl::IsInvalidArgument(FitRidge(kernel, {0}, {1}, invalid).status()));
    EXPECT_TRUE(absl::IsInvalidArgument(
        FitGradientDescent(kernel, {0}, {1}, invalid, 1).status()));
  }
  EXPECT_TRUE(absl::IsInvalidArgument(
      FitGradientDescent(kernel, {0}, {1}, 1, -1).status()));
}

TEST(KernelRidgeTest, ReportsNonfiniteFactorizationAndResidualArithmetic) {
  EXPECT_TRUE(absl::IsOutOfRange(
      FitRidge({1, 1, {kMaximum}}, {0}, {1}, kMaximum).status()));
  EXPECT_TRUE(absl::IsOutOfRange(
      FitRidge({1, 1, {1}}, {-kMaximum}, {kMaximum}, 1).status()));
}

TEST(KernelPredictionTest, AllowsEmptyQueriesAndZeroFeatureBaseline) {
  auto empty = Predict({0, 2, {}}, {}, {1, 2});
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_TRUE(empty->empty());
  auto baseline = Predict({2, 0, {}}, {1, 2}, {});
  ASSERT_TRUE(baseline.ok()) << baseline.status();
  EXPECT_EQ(*baseline, (std::vector<double>{1, 2}));
}

TEST(KernelPredictionTest, ValidatesShapesAndFiniteness) {
  const Matrix kernel{1, 2, {1, 2}};
  EXPECT_TRUE(absl::IsInvalidArgument(Predict(kernel, {}, {1, 2}).status()));
  EXPECT_TRUE(absl::IsInvalidArgument(Predict(kernel, {0}, {1}).status()));
  EXPECT_TRUE(
      absl::IsInvalidArgument(Predict(kernel, {kNaN}, {1, 2}).status()));
  EXPECT_TRUE(
      absl::IsInvalidArgument(Predict(kernel, {0}, {1, kInfinity}).status()));
  EXPECT_TRUE(
      absl::IsInvalidArgument(Predict({1, 1, {kInfinity}}, {0}, {1}).status()));
  EXPECT_TRUE(
      absl::IsInvalidArgument(Predict({2, 2, {1}}, {0, 0}, {1, 2}).status()));
  EXPECT_TRUE(
      absl::IsOutOfRange(Predict({1, 1, {kMaximum}}, {0}, {2}).status()));
}

TEST(KernelGradientDescentTest, ZeroStepsRetainsInitialFunction) {
  const Matrix kernel{2, 2, {2, 1, 1, 2}};
  auto coefficients = FitGradientDescent(kernel, {1, -2}, {4, -5}, 0.1, 0);
  ASSERT_TRUE(coefficients.ok()) << coefficients.status();
  EXPECT_EQ(*coefficients, (std::vector<double>{0, 0}));
  auto predictions = Predict(kernel, {1, -2}, *coefficients);
  ASSERT_TRUE(predictions.ok()) << predictions.status();
  EXPECT_EQ(*predictions, (std::vector<double>{1, -2}));
}

TEST(KernelGradientDescentTest, MatchesDiagonalClosedFormAndHeldOutPrediction) {
  const Matrix kernel{2, 2, {2, 0, 0, 4}};
  constexpr int kSteps = 5;
  // alpha_i(t) = (y_i-f0_i)/K_ii * (1-(1-lr*K_ii/N)^t).
  const double expected0 = 3 * (1 - std::pow(0.75, kSteps));
  const double expected1 = 2 * (1 - std::pow(0.5, kSteps));
  auto coefficients = FitGradientDescent(kernel, {1, -1}, {7, 7}, 0.25, kSteps);
  ASSERT_TRUE(coefficients.ok()) << coefficients.status();
  EXPECT_NEAR((*coefficients)[0], expected0, 1e-14);
  EXPECT_NEAR((*coefficients)[1], expected1, 1e-14);
  auto held_out = Predict({1, 2, {1, 2}}, {10}, *coefficients);
  ASSERT_TRUE(held_out.ok()) << held_out.status();
  EXPECT_NEAR((*held_out)[0], 10 + expected0 + 2 * expected1, 1e-14);
}

TEST(KernelGradientDescentTest, UpdatesAllCoordinatesSimultaneously) {
  const Matrix kernel{2, 2, {2, 1, 1, 2}};
  // alpha_1=[0.5,1]; f_1=[2,2.5]; alpha_2=[0,0.75]. Updating alpha
  // in-place during the matrix product would instead produce different values.
  auto coefficients = FitGradientDescent(kernel, {0, 0}, {1, 2}, 1, 2);
  ASSERT_TRUE(coefficients.ok()) << coefficients.status();
  EXPECT_DOUBLE_EQ((*coefficients)[0], 0);
  EXPECT_DOUBLE_EQ((*coefficients)[1], 0.75);
}

TEST(KernelGradientDescentTest, SingularKernelDoesNotImplyFailure) {
  const Matrix kernel{2, 2, {1, 1, 1, 1}};
  auto coefficients = FitGradientDescent(kernel, {0, 0}, {1, 3}, 1, 1);
  ASSERT_TRUE(coefficients.ok()) << coefficients.status();
  EXPECT_EQ(*coefficients, (std::vector<double>{0.5, 1.5}));
  auto predictions = Predict(kernel, {0, 0}, *coefficients);
  ASSERT_TRUE(predictions.ok()) << predictions.status();
  EXPECT_EQ(*predictions, (std::vector<double>{2, 2}));
}

TEST(KernelGradientDescentTest, FullBlockKernelKeepsCrossOutputInteractions) {
  // Two examples with two outputs each, flattened as [x0.a,x0.b,x1.a,x1.b].
  // All four coordinates share one feature with derivative [1,2,3,4]. This
  // outer-product kernel has nonzero cross-output blocks; treating outputs
  // as independent scalar models would lose those interactions.
  const Matrix kernel{
      4, 4, {1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12, 4, 8, 12, 16}};
  const std::vector<double> initial{10, 20, 30, 40};
  auto coefficients =
      FitGradientDescent(kernel, initial, {11, 20, 30, 40}, 0.4, 1);
  ASSERT_TRUE(coefficients.ok()) << coefficients.status();
  EXPECT_EQ(*coefficients, (std::vector<double>{0.1, 0, 0, 0}));
  auto predictions = Predict(kernel, initial, *coefficients);
  ASSERT_TRUE(predictions.ok()) << predictions.status();
  for (size_t row = 0; row < 4; ++row)
    EXPECT_NEAR((*predictions)[row], initial[row] + 0.1 * (row + 1), 1e-14);
  // A held-out example with two outputs has a 2x4 cross-kernel, not 2x2.
  auto held_out =
      Predict({2, 4, {5, 10, 15, 20, 6, 12, 18, 24}}, {50, 60}, *coefficients);
  ASSERT_TRUE(held_out.ok()) << held_out.status();
  EXPECT_NEAR((*held_out)[0], 50.5, 1e-14);
  EXPECT_NEAR((*held_out)[1], 60.6, 1e-14);
}

TEST(KernelGradientDescentTest, ReportsNonfiniteUpdatesAndPredictions) {
  EXPECT_TRUE(absl::IsOutOfRange(
      FitGradientDescent({1, 1, {1}}, {0}, {2}, kMaximum, 1).status()));
  EXPECT_TRUE(absl::IsOutOfRange(
      FitGradientDescent({1, 1, {kMaximum}}, {0}, {2}, 1, 2).status()));
  EXPECT_TRUE(absl::IsOutOfRange(
      FitGradientDescent({1, 1, {1}}, {-kMaximum}, {kMaximum}, 1, 1).status()));
}

}  // namespace
}  // namespace pluto::llm::ntk
