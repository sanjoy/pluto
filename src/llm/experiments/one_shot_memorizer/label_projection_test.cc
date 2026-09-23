#include "src/llm/experiments/one_shot_memorizer/label_projection.h"

#include <limits>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

const TokenCodes kCodes{2, 2, {1, -1, -1, 1}};

TEST(LabelProjectionTest, RecoversLabelCodesAfterAddingResidualAndCentering) {
  const std::vector<float> x{-1, 1, -1, 1};
  // Common-mode shifts are deliberately large; the classifier ignores them.
  const std::vector<float> residuals{10.5, 9.5, -4.5, -3.5, 2.5, 1.5, 2.5, 3.5};
  const std::vector<int> labels{0, 1, 0, 1}, sentences{0, 1, 2, 3};
  LabelProjectionOptions options;
  options.held_sentence_stride = 0;
  options.code_scale = 0.25;
  options.affine.ridge = 0;
  auto fit = FitTokenCodeProjection(x, 1, residuals, labels, sentences, kCodes,
                                    options);
  ASSERT_TRUE(fit.ok()) << fit.status();
  auto projected = ApplyClosedFormMap(*fit, x);
  ASSERT_TRUE(projected.ok()) << projected.status();
  for (size_t row = 0; row < labels.size(); ++row) {
    const double mean = (residuals[row * 2] + residuals[row * 2 + 1]) / 2.0;
    for (int dim = 0; dim < 2; ++dim)
      EXPECT_NEAR((*projected)[row * 2 + dim] + residuals[row * 2 + dim] - mean,
                  0.25 * kCodes.values[labels[row] * 2 + dim], 1e-10);
  }
}

TEST(LabelProjectionTest, WholeSentenceHoldoutDoesNotAffectAnyFitStatistic) {
  const std::vector<float> x{100, 1, -1, 100, 1, -1};
  const std::vector<float> r(12, 0);
  const std::vector<int> y{0, 0, 1, 1, 0, 1}, group{0, 1, 2, 0, 3, 4};
  LabelProjectionOptions options;
  options.held_sentence_stride = 5;
  options.affine.ridge = 0;
  auto fit = FitTokenCodeProjection(x, 1, r, y, group, kCodes, options);
  ASSERT_TRUE(fit.ok()) << fit.status();
  auto changed_x = x;
  auto changed_r = r;
  auto changed_y = y;
  for (int row : {0, 3}) {
    changed_x[row] = -999;
    changed_r[2 * row] = 800;
    changed_y[row] = 1 - changed_y[row];
  }
  auto changed = FitTokenCodeProjection(changed_x, 1, changed_r, changed_y,
                                        group, kCodes, options);
  ASSERT_TRUE(changed.ok()) << changed.status();
  EXPECT_EQ(fit->sample_count, 4u);
  EXPECT_EQ(fit->weights, changed->weights);
  EXPECT_EQ(fit->biases, changed->biases);
  EXPECT_EQ(fit->target_rms, changed->target_rms);
}

TEST(LabelProjectionTest, RejectsMalformedDataAndEmptyFittingSplit) {
  const std::vector<float> x{-1, 1, -1, 1}, r(8, 0);
  const std::vector<int> y{0, 1, 0, 1}, group{1, 1, 1, 1};
  EXPECT_FALSE(FitTokenCodeProjection(x, 0, r, y, group, kCodes).ok());
  EXPECT_FALSE(FitTokenCodeProjection(x, 1, {}, y, group, kCodes).ok());
  EXPECT_FALSE(FitTokenCodeProjection(x, 1, r, y, {}, kCodes).ok());
  auto bad_y = y;
  bad_y[0] = 2;
  EXPECT_FALSE(FitTokenCodeProjection(x, 1, r, bad_y, group, kCodes).ok());
  auto bad_x = x;
  bad_x[0] = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(FitTokenCodeProjection(bad_x, 1, r, y, group, kCodes).ok());
  EXPECT_FALSE(
      FitTokenCodeProjection(x, 1, r, y, std::vector<int>(4, 0), kCodes).ok());
  LabelProjectionOptions options;
  options.held_sentence_stride = 1;
  EXPECT_FALSE(FitTokenCodeProjection(x, 1, r, y, group, kCodes, options).ok());
  options.held_sentence_stride = 0;
  options.code_scale = 0;
  EXPECT_FALSE(FitTokenCodeProjection(x, 1, r, y, group, kCodes, options).ok());
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
