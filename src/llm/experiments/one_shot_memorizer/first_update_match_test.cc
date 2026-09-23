#include "src/llm/experiments/one_shot_memorizer/first_update_match.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <vector>

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

TEST(FirstUpdatePromptSetTest, EnumeratesAllFiftyFiveReplacements) {
  const std::vector<int> prompt{4, 2, 0, 3, 1};
  const std::vector<int> suffix{5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  auto sets = BuildOneTokenReplacementPromptSets(prompt, suffix, 17, 16);
  ASSERT_TRUE(sets.ok()) << sets.status();
  ASSERT_EQ(sets->size(), 56u);
  const std::vector<int> original{0, 1, 2, 3, 4};
  EXPECT_EQ((*sets)[0].token_ids, original);
  EXPECT_EQ((*sets)[0].removed_token, -1);
  EXPECT_EQ((*sets)[0].added_token, -1);
  std::set<std::vector<int>> unique;
  for (size_t i = 0; i < sets->size(); ++i) {
    const auto& set = (*sets)[i];
    EXPECT_TRUE(unique.insert(set.token_ids).second);
    EXPECT_TRUE(std::is_sorted(set.token_ids.begin(), set.token_ids.end()));
    EXPECT_EQ(set.token_ids.size(), 5u);
    EXPECT_EQ(std::adjacent_find(set.token_ids.begin(), set.token_ids.end()),
              set.token_ids.end());
    EXPECT_EQ(std::count(set.token_ids.begin(), set.token_ids.end(), 16), 0);
    if (i == 0)
      continue;
    const int expected_removed = (i - 1) / 11;
    const int expected_added = 5 + (i - 1) % 11;
    EXPECT_EQ(set.removed_token, expected_removed);
    EXPECT_EQ(set.added_token, expected_added);
    auto expected = original;
    expected[expected_removed] = expected_added;
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(set.token_ids, expected);
  }
}

TEST(FirstUpdatePromptSetTest, DeduplicatesSuffixAndIgnoresEosAndExistingIds) {
  auto repeated = BuildOneTokenReplacementPromptSets(
      {4, 3, 2, 1, 0}, {6, 5, 5, 6, 9, 0, 1, 9}, 10, 9);
  auto distinct =
      BuildOneTokenReplacementPromptSets({0, 1, 2, 3, 4}, {5, 6}, 10, 9);
  ASSERT_TRUE(repeated.ok()) << repeated.status();
  ASSERT_TRUE(distinct.ok()) << distinct.status();
  ASSERT_EQ(repeated->size(), 11u);
  ASSERT_EQ(repeated->size(), distinct->size());
  for (size_t i = 0; i < repeated->size(); ++i) {
    EXPECT_EQ((*repeated)[i].token_ids, (*distinct)[i].token_ids);
    EXPECT_EQ((*repeated)[i].removed_token, (*distinct)[i].removed_token);
    EXPECT_EQ((*repeated)[i].added_token, (*distinct)[i].added_token);
  }
}

TEST(FirstUpdatePromptSetTest, KeepsBaseWhenThereAreNoNewNonEosIds) {
  for (const std::vector<int>& suffix :
       {std::vector<int>{}, std::vector<int>{9, 9},
        std::vector<int>{0, 1, 4}}) {
    auto sets =
        BuildOneTokenReplacementPromptSets({0, 1, 2, 3, 4}, suffix, 10, 9);
    ASSERT_TRUE(sets.ok()) << sets.status();
    EXPECT_EQ(sets->size(), 1u);
    EXPECT_EQ((*sets)[0].token_ids, (std::vector<int>{0, 1, 2, 3, 4}));
  }
}

TEST(FirstUpdatePromptSetTest, RejectsMalformedPromptAndVocabulary) {
  for (const std::vector<int>& prompt :
       {std::vector<int>{}, std::vector<int>{0, 1, 2, 3},
        std::vector<int>{0, 1, 2, 3, 4, 5}, std::vector<int>{0, 0, 1, 2, 3},
        std::vector<int>{-1, 0, 1, 2, 3}, std::vector<int>{0, 1, 2, 3, 10},
        std::vector<int>{0, 1, 2, 3, 9}})
    EXPECT_EQ(
        BuildOneTokenReplacementPromptSets(prompt, {5}, 10, 9).status().code(),
        absl::StatusCode::kInvalidArgument);
  for (int vocabulary : {-1, 0})
    EXPECT_FALSE(
        BuildOneTokenReplacementPromptSets({0, 1, 2, 3, 4}, {}, vocabulary, 0)
            .ok());
  for (int eos : {-1, 10})
    EXPECT_FALSE(
        BuildOneTokenReplacementPromptSets({0, 1, 2, 3, 4}, {}, 10, eos).ok());
  for (int token : {-1, 10})
    EXPECT_FALSE(
        BuildOneTokenReplacementPromptSets({0, 1, 2, 3, 4}, {token}, 10, 9)
            .ok());
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
