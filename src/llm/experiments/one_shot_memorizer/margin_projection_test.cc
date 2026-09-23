#include "src/llm/experiments/one_shot_memorizer/margin_projection.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

// Deliberately separate from the solver's separation oracle. Tests enumerate
// every target/rival pair, not only the constraints selected by cutting planes.
double MinimumMargin(const MarginProjectionResult& fit,
                     const std::vector<float>& features,
                     const std::vector<float>& residuals,
                     const std::vector<int>& labels,
                     const std::vector<double>& decoder, int vocab_size) {
  double minimum = std::numeric_limits<double>::infinity();
  for (size_t row = 0; row < labels.size(); ++row) {
    std::vector<double> activation(fit.output_dim);
    for (int dim = 0; dim < fit.output_dim; ++dim) {
      activation[dim] = fit.biases[dim] + residuals[row * fit.output_dim + dim];
      for (int feature = 0; feature < fit.input_dim; ++feature)
        activation[dim] += features[row * fit.input_dim + feature] *
                           fit.weights[feature * fit.output_dim + dim];
    }
    for (int token = 0; token < vocab_size; ++token) {
      if (token == labels[row])
        continue;
      double margin = 0;
      for (int dim = 0; dim < fit.output_dim; ++dim)
        margin += (decoder[labels[row] * fit.output_dim + dim] -
                   decoder[token * fit.output_dim + dim]) *
                  activation[dim];
      minimum = std::min(minimum, margin);
    }
  }
  return minimum;
}

TEST(MarginProjectionTest, ConstructsProjectionFromFeaturesResidualsAndLabels) {
  const std::vector<float> x{-1, 1, -0.75, 0.75};
  const std::vector<float> residuals{0.75, -0.25, 0.25, 0.75,
                                     0.50, -0.10, 0.10, 0.50};
  const std::vector<int> labels{0, 1, 0, 1};
  const std::vector<double> decoder{1, -1, -1, 1};
  std::vector<MarginProjectionProgress> progress;
  MarginProjectionOptions options;
  options.center_coefficients = true;
  options.progress = [&](const MarginProjectionProgress& value) {
    progress.push_back(value);
  };
  auto result =
      FitMarginProjection(x, 1, residuals, labels, decoder, 2, 2, options);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->outcome, MarginProjectionOutcome::kVerifiedPositiveMargin);
  EXPECT_EQ(result->progress.correct_count, labels.size());
  const double measured =
      MinimumMargin(*result, x, residuals, labels, decoder, 2);
  EXPECT_GT(measured, options.acceptance_tolerance);
  EXPECT_NEAR(measured, result->progress.minimum_margin, 1e-12);
  EXPECT_NEAR(result->weights[0] + result->weights[1], 0, 1e-8);
  EXPECT_NEAR(result->biases[0] + result->biases[1], 0, 1e-8);
  for (const auto& coefficients : {result->weights, result->biases})
    for (double value : coefficients)
      EXPECT_LE(std::abs(value), options.coefficient_bound);
  ASSERT_GE(progress.size(), 2u);
  EXPECT_EQ(progress.front().round, 0);
  EXPECT_EQ(progress.front().checked_round, 0);
  EXPECT_EQ(progress.back().status, "verified_positive_margin");
  EXPECT_EQ(progress.back().correct_count, labels.size());
  EXPECT_EQ(progress.back().round, progress.back().checked_round);
}

TEST(MarginProjectionTest, RetainsZeroProjectionWhenResidualAlreadySeparates) {
  auto result = FitMarginProjection(
      std::vector<float>{1, -1}, 1, std::vector<float>{2, -2},
      std::vector<int>{0, 1}, std::vector<double>{1, -1}, 1, 2);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->outcome, MarginProjectionOutcome::kVerifiedPositiveMargin);
  EXPECT_EQ(result->progress.round, 0);
  EXPECT_EQ(result->progress.checked_round, 0);
  EXPECT_EQ(result->progress.cut_count, 0u);
  EXPECT_EQ(result->progress.solver_status, -1);
  EXPECT_EQ(result->weights, std::vector<double>{0});
  EXPECT_EQ(result->biases, std::vector<double>{0});
  EXPECT_DOUBLE_EQ(result->progress.minimum_margin, 4);
}

TEST(MarginProjectionTest, ContradictoryTargetsCannotHavePositiveCommonMargin) {
  auto result = FitMarginProjection(
      std::vector<float>{0, 0}, 1, std::vector<float>{0, 0},
      std::vector<int>{0, 1}, std::vector<double>{1, -1}, 1, 2);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->outcome, MarginProjectionOutcome::kBoundedNoPositiveMargin);
  EXPECT_LE(result->progress.lp_objective, 0);
  EXPECT_LE(result->progress.minimum_margin, 0);
  // Ties favor the smaller vocabulary ID, but ties never certify success.
  EXPECT_EQ(result->progress.correct_count, 1u);
}

TEST(MarginProjectionTest, ChecksRivalClassesAbsentFromObservedLabels) {
  // Class 0 can defeat class 1 only on z>0, where unseen class 2 defeats it.
  // A fitter checking only observed labels would incorrectly accept this.
  const std::vector<float> x{0}, residuals{0};
  const std::vector<int> labels{0};
  const std::vector<double> decoder{1, -1, 2};
  MarginProjectionOptions options;
  options.max_new_cuts = 1;
  auto result =
      FitMarginProjection(x, 1, residuals, labels, decoder, 1, 3, options);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->outcome, MarginProjectionOutcome::kBoundedNoPositiveMargin);
  EXPECT_EQ(result->progress.cut_count, 2u);
  EXPECT_NEAR(MinimumMargin(*result, x, residuals, labels, decoder, 3),
              result->progress.minimum_margin, 1e-12);
}

TEST(MarginProjectionTest,
     CoefficientBoundRestrictsTheClaimAndDeltaCanBeNegative) {
  const std::vector<float> x{0}, residuals{-3};
  const std::vector<int> labels{0};
  const std::vector<double> decoder{1, -1};
  MarginProjectionOptions options;
  auto small =
      FitMarginProjection(x, 1, residuals, labels, decoder, 1, 2, options);
  ASSERT_TRUE(small.ok()) << small.status();
  EXPECT_EQ(small->outcome, MarginProjectionOutcome::kBoundedNoPositiveMargin);
  EXPECT_NEAR(small->progress.lp_objective, -4, 1e-8);
  EXPECT_NEAR(small->progress.minimum_margin, -4, 1e-8);
  options.coefficient_bound = 4;
  auto large =
      FitMarginProjection(x, 1, residuals, labels, decoder, 1, 2, options);
  ASSERT_TRUE(large.ok()) << large.status();
  EXPECT_EQ(large->outcome, MarginProjectionOutcome::kVerifiedPositiveMargin);
  EXPECT_GT(large->progress.minimum_margin, options.acceptance_tolerance);
}

TEST(MarginProjectionTest, BudgetExhaustionDoesNotClaimInfeasibility) {
  const std::vector<float> x{0}, residuals{0};
  const std::vector<int> labels{0};
  const std::vector<double> decoder{1, -1, 2};
  MarginProjectionOptions options;
  options.max_rounds = 1;
  options.max_new_cuts = 1;
  auto round_limited =
      FitMarginProjection(x, 1, residuals, labels, decoder, 1, 3, options);
  ASSERT_TRUE(round_limited.ok()) << round_limited.status();
  EXPECT_EQ(round_limited->outcome, MarginProjectionOutcome::kRoundLimit);
  EXPECT_EQ(round_limited->progress.checked_round, 1);
  EXPECT_GT(round_limited->progress.lp_objective, 0);
  EXPECT_LT(round_limited->progress.minimum_margin, 0);
  options.max_rounds = 3;
  options.max_total_cuts = 1;
  auto cut_limited =
      FitMarginProjection(x, 1, residuals, labels, decoder, 1, 3, options);
  ASSERT_TRUE(cut_limited.ok()) << cut_limited.status();
  EXPECT_EQ(cut_limited->outcome, MarginProjectionOutcome::kCutLimit);
  EXPECT_EQ(cut_limited->progress.cut_count, 1u);
}

TEST(MarginProjectionTest, PositiveButInsufficientMarginIsNotNumericalFailure) {
  // The best bounded bias is 1, so the exact maximum margin is 0.5. Both
  // classification and the LP are fine; only the requested robustness fails.
  MarginProjectionOptions options;
  options.acceptance_tolerance = 0.75;
  auto result = FitMarginProjection(
      std::vector<float>{0}, 1, std::vector<float>{0}, std::vector<int>{0},
      std::vector<double>{0.25, -0.25}, 1, 2, options);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->outcome,
            MarginProjectionOutcome::kBoundedMarginBelowTolerance);
  EXPECT_EQ(result->progress.correct_count, 1u);
  EXPECT_DOUBLE_EQ(result->progress.minimum_margin, 0.5);
  EXPECT_DOUBLE_EQ(result->progress.lp_objective, 0.5);
  EXPECT_EQ(result->progress.status, "bounded_margin_below_tolerance");
}

TEST(MarginProjectionTest,
     BothSimplexStrategiesHandleFeasibleAndNegativeOptima) {
  for (bool dual_simplex : {false, true}) {
    SCOPED_TRACE(dual_simplex);
    MarginProjectionOptions options;
    options.dual_simplex = dual_simplex;
    auto feasible = FitMarginProjection(
        std::vector<float>{-1, 1}, 1, std::vector<float>{0, 0},
        std::vector<int>{0, 1}, std::vector<double>{1, -1}, 1, 2, options);
    ASSERT_TRUE(feasible.ok()) << feasible.status();
    EXPECT_EQ(feasible->outcome,
              MarginProjectionOutcome::kVerifiedPositiveMargin);
    EXPECT_GT(feasible->progress.minimum_margin, options.acceptance_tolerance);
    EXPECT_GT(MinimumMargin(*feasible, {-1, 1}, {0, 0}, {0, 1}, {1, -1}, 2),
              options.acceptance_tolerance);
    auto negative = FitMarginProjection(
        std::vector<float>{0}, 1, std::vector<float>{-3}, std::vector<int>{0},
        std::vector<double>{1, -1}, 1, 2, options);
    ASSERT_TRUE(negative.ok()) << negative.status();
    EXPECT_EQ(negative->outcome,
              MarginProjectionOutcome::kBoundedNoPositiveMargin);
    EXPECT_NEAR(negative->progress.lp_objective, -4, 1e-8);
    EXPECT_NEAR(negative->progress.minimum_margin, -4, 1e-8);
  }
}

TEST(MarginProjectionTest, IsDeterministicWhenSeveralRowsAndRivalsTie) {
  const std::vector<float> x{-1, 1, -1, 1}, residuals(8, 0);
  const std::vector<int> labels{0, 1, 0, 1};
  const std::vector<double> decoder{1, -1, -1, 1, 0, 0};
  MarginProjectionOptions options;
  options.center_coefficients = true;
  options.max_new_cuts = 1;
  auto first =
      FitMarginProjection(x, 1, residuals, labels, decoder, 2, 3, options);
  auto second =
      FitMarginProjection(x, 1, residuals, labels, decoder, 2, 3, options);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(first->outcome, MarginProjectionOutcome::kVerifiedPositiveMargin);
  EXPECT_EQ(first->outcome, second->outcome);
  EXPECT_EQ(first->weights, second->weights);
  EXPECT_EQ(first->biases, second->biases);
  EXPECT_EQ(first->progress.cut_count, second->progress.cut_count);
  EXPECT_EQ(first->progress.minimum_margin, second->progress.minimum_margin);
}

TEST(MarginProjectionTest, RejectsBadShapesLabelsNonfiniteValuesAndGauge) {
  const std::vector<float> x{0}, residuals{0};
  const std::vector<int> labels{0};
  const std::vector<double> decoder{1, -1};
  EXPECT_FALSE(
      FitMarginProjection(x, 0, residuals, labels, decoder, 1, 2).ok());
  EXPECT_FALSE(
      FitMarginProjection({}, 1, residuals, labels, decoder, 1, 2).ok());
  EXPECT_FALSE(FitMarginProjection(x, 1, {}, labels, decoder, 1, 2).ok());
  EXPECT_FALSE(FitMarginProjection(x, 1, residuals, {}, decoder, 1, 2).ok());
  EXPECT_FALSE(
      FitMarginProjection(x, 1, residuals, labels, decoder, 1, 1).ok());
  EXPECT_FALSE(FitMarginProjection(x, 1, residuals, labels, decoder,
                                   std::numeric_limits<int>::max(), 2)
                   .ok());
  EXPECT_FALSE(
      FitMarginProjection(x, 1, residuals, std::vector<int>{2}, decoder, 1, 2)
          .ok());
  EXPECT_FALSE(
      FitMarginProjection(x, 1, residuals, std::vector<int>{-1}, decoder, 1, 2)
          .ok());
  const float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(FitMarginProjection(std::vector<float>{nan}, 1, residuals,
                                   labels, decoder, 1, 2)
                   .ok());
  EXPECT_FALSE(
      FitMarginProjection(x, 1, std::vector<float>{nan}, labels, decoder, 1, 2)
          .ok());
  EXPECT_FALSE(FitMarginProjection(x, 1, residuals, labels,
                                   std::vector<double>{1, nan}, 1, 2)
                   .ok());
  MarginProjectionOptions options;
  options.center_coefficients = true;
  EXPECT_FALSE(
      FitMarginProjection(x, 1, residuals, labels, decoder, 1, 2, options)
          .ok());
}

TEST(MarginProjectionTest, RejectsInvalidLimitsIncludingUnlimitedTimeout) {
  const std::vector<float> x{0}, residuals{0};
  const std::vector<int> labels{0};
  const std::vector<double> decoder{1, -1};
  auto invalid = [&](const MarginProjectionOptions& options) {
    EXPECT_FALSE(
        FitMarginProjection(x, 1, residuals, labels, decoder, 1, 2, options)
            .ok());
  };
  MarginProjectionOptions options;
  options.per_solve_timeout_seconds = 0;
  invalid(options);
  options.per_solve_timeout_seconds = -1;
  invalid(options);
  options = {};
  options.max_rounds = 0;
  invalid(options);
  options.max_rounds = std::numeric_limits<int>::max();
  invalid(options);
  options = {};
  options.max_new_cuts = 0;
  invalid(options);
  options = {};
  options.max_total_cuts = 0;
  invalid(options);
  options.max_total_cuts = std::numeric_limits<int>::max();
  invalid(options);
  options = {};
  options.coefficient_bound = 0;
  invalid(options);
  options.coefficient_bound = std::numeric_limits<double>::infinity();
  invalid(options);
  options.coefficient_bound = 1e100;
  invalid(options);
  options = {};
  options.margin_cap = 0;
  invalid(options);
  options.margin_cap = 1e100;
  invalid(options);
  options = {};
  options.acceptance_tolerance = 0;
  invalid(options);
  options.acceptance_tolerance = options.margin_cap;
  invalid(options);
  options.acceptance_tolerance = std::numeric_limits<double>::quiet_NaN();
  invalid(options);
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
