#include "src/llm/experiments/one_shot_memorizer/associative_readout.h"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

constexpr std::array<ReadoutMode, 3> kModes = {
    ReadoutMode::kMeanDot, ReadoutMode::kNearestCentroid,
    ReadoutMode::kSharedCovarianceLda};

TEST(AssociativeReadoutTest, AllModesClassifySeparatedVectors) {
  const std::vector<float> x = {-2, -1, -3, -1, 2, 1, 3, 1};
  const std::vector<int> y = {2, 2, 5, 5};
  for (ReadoutMode mode : kModes) {
    ReadoutOptions options;
    options.mode = mode;
    auto model = BuildReadout(x, y, 2, 8, options);
    ASSERT_TRUE(model.ok()) << model.status();
    EXPECT_EQ(model->seen_classes, (std::vector<int>{2, 5}));
    EXPECT_EQ(model->training_sample_count, 4u);
    auto evaluation = EvaluateReadout(*model, x, y);
    ASSERT_TRUE(evaluation.ok()) << evaluation.status();
    EXPECT_EQ(evaluation->predictions, y);
    EXPECT_EQ(evaluation->correct_count, 4u);
    EXPECT_DOUBLE_EQ(evaluation->accuracy, 1.0);
    EXPECT_EQ(evaluation->singleton_target_count, 0u);
    EXPECT_EQ(evaluation->unseen_target_count, 0u);
  }
}

TEST(AssociativeReadoutTest,
     NearestCentroidUsesBiasAndTiesChooseSmallestSeenId) {
  const std::vector<float> x = {1, 10};
  const std::vector<int> y = {4, 2};
  auto dot = BuildReadout(x, y, 1, 6);
  ReadoutOptions options;
  options.mode = ReadoutMode::kNearestCentroid;
  auto nearest = BuildReadout(x, y, 1, 6, options);
  ASSERT_TRUE(dot.ok());
  ASSERT_TRUE(nearest.ok());
  auto dot_result =
      EvaluateReadout(*dot, std::vector<float>{1}, std::vector<int>{4});
  auto nearest_result =
      EvaluateReadout(*nearest, std::vector<float>{1}, std::vector<int>{4});
  ASSERT_TRUE(dot_result.ok());
  ASSERT_TRUE(nearest_result.ok());
  EXPECT_EQ(dot_result->predictions[0], 2);
  EXPECT_EQ(nearest_result->predictions[0], 4);
  EXPECT_FLOAT_EQ(nearest->biases[4], -0.5f);
  EXPECT_FLOAT_EQ(nearest->biases[2], -50.0f);
  auto tie =
      EvaluateReadout(*nearest, std::vector<float>{5.5f}, std::vector<int>{2});
  ASSERT_TRUE(tie.ok());
  EXPECT_EQ(tie->predictions[0], 2);
  auto dot_tie =
      EvaluateReadout(*dot, std::vector<float>{0}, std::vector<int>{2});
  ASSERT_TRUE(dot_tie.ok());
  EXPECT_EQ(dot_tie->predictions[0], 2);
}

TEST(AssociativeReadoutTest, LdaUsesPooledWithinClassCovarianceDividedByN) {
  const std::vector<float> x = {0, 2, 4, 6};
  const std::vector<int> y = {0, 0, 2, 2};
  ReadoutOptions options;
  options.mode = ReadoutMode::kSharedCovarianceLda;
  options.ridge = 0.5;
  auto model = BuildReadout(x, y, 1, 3, options);
  ASSERT_TRUE(model.ok()) << model.status();
  // Means 1 and 5; within scatter 4; covariance 4/4 + 0.5 = 1.5.
  EXPECT_NEAR(model->weights[0], 1.0 / 1.5, 1e-6);
  EXPECT_NEAR(model->weights[2], 5.0 / 1.5, 1e-6);
  EXPECT_NEAR(model->biases[0], -0.5 / 1.5, 1e-6);
  EXPECT_NEAR(model->biases[2], -12.5 / 1.5, 1e-6);
}

TEST(AssociativeReadoutTest,
     CovarianceChangesClassificationAlongNoisyDirections) {
  const std::vector<float> x = {-10, 0, 10, 0, 0, -1, 0, 1,
                                -8,  2, 12, 2, 2, 1,  2, 3};
  const std::vector<int> y = {1, 1, 1, 1, 3, 3, 3, 3};
  ReadoutOptions options;
  options.mode = ReadoutMode::kNearestCentroid;
  options.ridge = 0.1;
  auto nearest = BuildReadout(x, y, 2, 4, options);
  options.mode = ReadoutMode::kSharedCovarianceLda;
  auto lda = BuildReadout(x, y, 2, 4, options);
  ASSERT_TRUE(nearest.ok());
  ASSERT_TRUE(lda.ok());
  const std::vector<float> query = {3, 0};
  auto a = EvaluateReadout(*nearest, query, std::vector<int>{1});
  auto b = EvaluateReadout(*lda, query, std::vector<int>{1});
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());
  EXPECT_EQ(a->predictions[0], 3);
  EXPECT_EQ(b->predictions[0], 1);
}

TEST(AssociativeReadoutTest, RidgeMakesSingularWithinClassCovarianceSolvable) {
  const std::vector<float> x = {1, 2, 1, 2, -1, -2, -1, -2};
  const std::vector<int> y = {0, 0, 1, 1};
  ReadoutOptions options;
  options.mode = ReadoutMode::kSharedCovarianceLda;
  options.ridge = 0.25;
  auto model = BuildReadout(x, y, 2, 2, options);
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_FLOAT_EQ(model->weights[0], 4);
  EXPECT_FLOAT_EQ(model->weights[1], 8);
  auto evaluation = EvaluateReadoutLeaveOneOut(x, y, 2, 2, options);
  ASSERT_TRUE(evaluation.ok()) << evaluation.status();
  EXPECT_EQ(evaluation->correct_count, 4u);
}

TEST(AssociativeReadoutTest, UnseenClassesAreDisabledAndReportedAsMisses) {
  auto model = BuildReadout(std::vector<float>{1, 0, 0, 1},
                            std::vector<int>{2, 5}, 2, 8);
  ASSERT_TRUE(model.ok());
  EXPECT_EQ(model->singleton_class_count, 2u);
  EXPECT_EQ(model->biases[0], -std::numeric_limits<float>::infinity());
  EXPECT_FLOAT_EQ(model->weights[0], 0);
  auto evaluation =
      EvaluateReadout(*model, std::vector<float>{-1, -1}, std::vector<int>{7});
  ASSERT_TRUE(evaluation.ok());
  EXPECT_EQ(evaluation->predictions[0], 2);  // Zero unseen rows cannot win.
  EXPECT_EQ(evaluation->unseen_target_count, 1u);
  EXPECT_EQ(evaluation->correct_count, 0u);
}

TEST(AssociativeReadoutTest, LeaveOneOutRemovesSingletonSelfVotes) {
  const std::vector<float> x = {1, 0, 0, 1};
  const std::vector<int> y = {2, 5};
  for (ReadoutMode mode : kModes) {
    ReadoutOptions options;
    options.mode = mode;
    auto loo = EvaluateReadoutLeaveOneOut(x, y, 2, 8, options);
    ASSERT_TRUE(loo.ok()) << loo.status();
    EXPECT_TRUE(loo->leave_one_out);
    EXPECT_EQ(loo->predictions, (std::vector<int>{5, 2}));
    EXPECT_EQ(loo->singleton_target_count, 2u);
    EXPECT_EQ(loo->unseen_target_count, 2u);
    EXPECT_EQ(loo->correct_count, 0u);
    EXPECT_EQ(loo->seen_class_count, 2u);
  }
}

TEST(AssociativeReadoutTest, LeaveOneOutUpdatesMatchLiteralRefits) {
  const std::vector<float> x = {-2, -1,   -1.5f, -2,   -2.4f, -0.5f, 2,
                                1,  1.5f, 2,     2.4f, 0.5f,  0,     3};
  const std::vector<int> y = {1, 1, 1, 3, 3, 3, 5};
  for (ReadoutMode mode : kModes) {
    for (bool normalize : {false, true}) {
      ReadoutOptions options;
      options.mode = mode;
      options.ridge = 0.2;
      options.normalize_inputs = normalize;
      auto loo = EvaluateReadoutLeaveOneOut(x, y, 2, 6, options);
      ASSERT_TRUE(loo.ok()) << loo.status();
      for (size_t omitted = 0; omitted < y.size(); ++omitted) {
        std::vector<float> train_x;
        std::vector<int> train_y;
        for (size_t i = 0; i < y.size(); ++i) {
          if (i == omitted)
            continue;
          train_x.insert(train_x.end(), x.begin() + 2 * i,
                         x.begin() + 2 * i + 2);
          train_y.push_back(y[i]);
        }
        auto refit = BuildReadout(train_x, train_y, 2, 6, options);
        ASSERT_TRUE(refit.ok()) << refit.status();
        auto prediction = EvaluateReadout(
            *refit, absl::Span<const float>(x).subspan(2 * omitted, 2),
            absl::Span<const int>(y).subspan(omitted, 1));
        ASSERT_TRUE(prediction.ok()) << prediction.status();
        EXPECT_EQ(loo->predictions[omitted], prediction->predictions[0])
            << "mode=" << static_cast<int>(mode) << " normalize=" << normalize
            << " omitted=" << omitted;
      }
      EXPECT_EQ(loo->unseen_target_count, 1u);
    }
  }
}

TEST(AssociativeReadoutTest,
     LdaRankOneUpdateMatchesRefitsWithCorrelatedFeatures) {
  ReadoutOptions options;
  options.mode = ReadoutMode::kSharedCovarianceLda;
  options.ridge = 0.03;
  for (int seed = 1; seed <= 5; ++seed) {
    std::vector<float> x;
    std::vector<int> y;
    for (int i = 0; i < 12; ++i) {
      y.push_back(i % 3);
      for (int j = 0; j < 3; ++j)
        x.push_back(std::sin((i + 1) * (j + 2) * 0.73 + seed) + 0.6 * (i % 3));
    }
    auto loo = EvaluateReadoutLeaveOneOut(x, y, 3, 4, options);
    ASSERT_TRUE(loo.ok()) << loo.status();
    for (size_t omitted = 0; omitted < y.size(); ++omitted) {
      std::vector<float> train_x;
      std::vector<int> train_y;
      for (size_t i = 0; i < y.size(); ++i) {
        if (i == omitted)
          continue;
        train_x.insert(train_x.end(), x.begin() + 3 * i, x.begin() + 3 * i + 3);
        train_y.push_back(y[i]);
      }
      auto refit = BuildReadout(train_x, train_y, 3, 4, options);
      ASSERT_TRUE(refit.ok()) << refit.status();
      auto prediction = EvaluateReadout(
          *refit, absl::Span<const float>(x).subspan(3 * omitted, 3),
          absl::Span<const int>(y).subspan(omitted, 1));
      ASSERT_TRUE(prediction.ok()) << prediction.status();
      EXPECT_EQ(loo->predictions[omitted], prediction->predictions[0])
          << "seed=" << seed << " omitted=" << omitted;
    }
  }
}

TEST(AssociativeReadoutTest, OptionalInputNormalizationIsAppliedAtInference) {
  ReadoutOptions options;
  options.normalize_inputs = true;
  auto model = BuildReadout(std::vector<float>{20, 0, 0, 0.3f},
                            std::vector<int>{0, 1}, 2, 2, options);
  ASSERT_TRUE(model.ok());
  EXPECT_FLOAT_EQ(model->weights[0], 1);
  EXPECT_FLOAT_EQ(model->weights[3], 1);
  auto result =
      EvaluateReadout(*model, std::vector<float>{0.1f, 5}, std::vector<int>{1});
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->predictions[0], 1);
  EXPECT_FALSE(
      EvaluateReadout(*model, std::vector<float>{0, 0}, std::vector<int>{1})
          .ok());
  EXPECT_FALSE(
      BuildReadout(std::vector<float>{0, 0}, std::vector<int>{0}, 2, 2, options)
          .ok());
  // Unequal prototype lengths give unequal centroid biases. Omitting query
  // normalization would incorrectly choose class 1 for this small input.
  options.mode = ReadoutMode::kNearestCentroid;
  auto centroid = BuildReadout(
      std::vector<float>{20, 0, std::sqrt(3.0f), 1, -std::sqrt(3.0f), 1},
      std::vector<int>{0, 1, 1}, 2, 2, options);
  ASSERT_TRUE(centroid.ok());
  auto normalized = EvaluateReadout(*centroid, std::vector<float>{0.09f, 0.01f},
                                    std::vector<int>{0});
  ASSERT_TRUE(normalized.ok());
  EXPECT_EQ(normalized->predictions[0], 0);
}

TEST(AssociativeReadoutTest,
     GroupHoldoutRemovesClassesRepeatedOnlyWithinGroup) {
  const std::vector<float> x = {1, 0, 1.1f, 0, 0, 1, 0, 1.1f};
  const std::vector<int> y = {0, 0, 1, 1};
  const std::vector<int> groups = {8, 8, 3, 3};
  for (ReadoutMode mode : kModes) {
    ReadoutOptions options;
    options.mode = mode;
    auto evaluation = EvaluateReadoutLeaveGroupOut(x, y, groups, 2, 2, options);
    ASSERT_TRUE(evaluation.ok()) << evaluation.status();
    EXPECT_TRUE(evaluation->leave_group_out);
    EXPECT_FALSE(evaluation->leave_one_out);
    EXPECT_EQ(evaluation->predictions, (std::vector<int>{1, 1, 0, 0}));
    EXPECT_EQ(evaluation->correct_count, 0u);
    EXPECT_EQ(evaluation->singleton_target_count, 0u);
    EXPECT_EQ(evaluation->unseen_target_count, 4u);
    EXPECT_EQ(evaluation->seen_class_count, 2u);
  }
}

TEST(AssociativeReadoutTest,
     GroupHoldoutMatchesLiteralRefitsInOriginalRowOrder) {
  const std::vector<float> x = {1,    0,  0,  1,     1,  0.1f, 0,
                                1.2f, -1, -1, -1.1f, -1, -2,   -1};
  const std::vector<int> y = {1, 2, 1, 2, 3, 3, 4};
  const std::vector<int> groups = {9, 3, 9, 4, 3, 4, 4};
  for (ReadoutMode mode : kModes) {
    ReadoutOptions options;
    options.mode = mode;
    options.ridge = 0.2;
    options.normalize_inputs = true;
    auto held_out = EvaluateReadoutLeaveGroupOut(x, y, groups, 2, 6, options);
    ASSERT_TRUE(held_out.ok()) << held_out.status();
    EXPECT_EQ(held_out->sample_count, y.size());
    EXPECT_EQ(held_out->seen_class_count, 4u);
    EXPECT_EQ(held_out->singleton_target_count, 1u);
    EXPECT_EQ(held_out->unseen_target_count, 3u);
    size_t correct = 0;
    for (int held_group : {3, 4, 9}) {
      std::vector<float> train_x;
      std::vector<int> train_y;
      for (size_t i = 0; i < y.size(); ++i) {
        if (groups[i] == held_group)
          continue;
        train_x.insert(train_x.end(), x.begin() + 2 * i, x.begin() + 2 * i + 2);
        train_y.push_back(y[i]);
      }
      auto refit = BuildReadout(train_x, train_y, 2, 6, options);
      ASSERT_TRUE(refit.ok()) << refit.status();
      for (size_t i = 0; i < y.size(); ++i) {
        if (groups[i] != held_group)
          continue;
        auto expected = EvaluateReadout(
            *refit, absl::Span<const float>(x).subspan(2 * i, 2),
            absl::Span<const int>(y).subspan(i, 1));
        ASSERT_TRUE(expected.ok()) << expected.status();
        EXPECT_EQ(held_out->predictions[i], expected->predictions[0]);
        correct += expected->correct_count;
      }
    }
    EXPECT_EQ(held_out->correct_count, correct);
    EXPECT_DOUBLE_EQ(held_out->accuracy,
                     static_cast<double>(correct) / y.size());
  }
}

TEST(AssociativeReadoutTest, GroupHoldoutValidatesGroupsAndDimensions) {
  const std::vector<float> x = {1, 0, 0, 1};
  const std::vector<int> y = {0, 1};
  EXPECT_FALSE(
      EvaluateReadoutLeaveGroupOut(x, y, std::vector<int>{3}, 2, 2).ok());
  EXPECT_FALSE(
      EvaluateReadoutLeaveGroupOut(x, y, std::vector<int>{3, 3}, 2, 2).ok());
  EXPECT_FALSE(
      EvaluateReadoutLeaveGroupOut(x, y, std::vector<int>{3, -1}, 2, 2).ok());
  EXPECT_FALSE(
      EvaluateReadoutLeaveGroupOut(x, y, std::vector<int>{3, 4}, 3, 2).ok());
  EXPECT_FALSE(EvaluateReadoutLeaveGroupOut({}, {}, {}, 2, 2).ok());
}

TEST(AssociativeReadoutTest, RejectsMalformedInputsAndReadoutArtifacts) {
  const std::vector<float> x = {1, 2};
  const std::vector<int> y = {0};
  EXPECT_FALSE(BuildReadout({}, {}, 2, 2).ok());
  EXPECT_FALSE(BuildReadout(x, y, 0, 2).ok());
  EXPECT_FALSE(BuildReadout(x, y, 2, 0).ok());
  EXPECT_FALSE(BuildReadout(x, y, 3, 2).ok());
  EXPECT_FALSE(BuildReadout(x, std::vector<int>{-1}, 2, 2).ok());
  EXPECT_FALSE(BuildReadout(x, std::vector<int>{2}, 2, 2).ok());
  EXPECT_FALSE(
      BuildReadout(
          std::vector<float>{1, std::numeric_limits<float>::infinity()}, y, 2,
          2)
          .ok());
  EXPECT_FALSE(EvaluateReadoutLeaveOneOut(x, y, 2, 2).ok());
  ReadoutOptions options;
  options.ridge = 0;
  EXPECT_FALSE(BuildReadout(x, y, 2, 2, options).ok());
  options.ridge = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(BuildReadout(x, y, 2, 2, options).ok());
  options.ridge = 0.1;
  options.mode = static_cast<ReadoutMode>(99);
  EXPECT_FALSE(BuildReadout(x, y, 2, 2, options).ok());
  auto model = BuildReadout(x, y, 2, 2);
  ASSERT_TRUE(model.ok());
  auto corrupt = *model;
  corrupt.biases[1] = 0;
  EXPECT_FALSE(EvaluateReadout(corrupt, x, y).ok());
  corrupt = *model;
  corrupt.weights[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(EvaluateReadout(corrupt, x, y).ok());
  corrupt = *model;
  corrupt.seen_classes.push_back(0);
  EXPECT_FALSE(EvaluateReadout(corrupt, x, y).ok());
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
