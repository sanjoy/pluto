#include "src/llm/experiments/one_shot_memorizer/greedy_suffix_readout.h"

#include <cmath>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

std::vector<float> Scores(int winner, int vocabulary = 5,
                          float confidence = 8) {
  std::vector<float> values(vocabulary, 0);
  values[winner] = confidence;
  return values;
}

TEST(GreedySuffixReadoutTest, PreservesActualHistoryAndInfersRepeatedTokens) {
  const std::vector<int> selected{0, 1, 2, 3};
  const GreedySuffixReadoutOptions options{
      .vocabulary_size = 5, .eos_token = 0, .selected_token_ids = selected};
  std::vector<std::vector<int>> histories;
  auto result = ReadGreedySuffix(options, [&](absl::Span<const int> history) {
    histories.emplace_back(history.begin(), history.end());
    if (history[5] != 1)
      return Scores(0);
    const std::vector<int> suffix{1, 2, 1, 3, 0};
    return Scores(suffix[history.size() - 5]);
  });
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->max_non_eos_tokens, 6);
  ASSERT_EQ(result->accepted_order, (std::vector<size_t>{0}));
  const auto& best = result->candidates[0];
  EXPECT_EQ(best.token_ids, (std::vector<int>{1, 2, 1, 3, 0}));
  EXPECT_EQ(best.distinct_selected_covered, 3u);
  EXPECT_EQ(histories[0], (std::vector<int>{0, 0, 0, 0, 0, 1}));
  EXPECT_EQ(histories[2], (std::vector<int>{0, 0, 0, 0, 0, 1, 2, 1}));
  EXPECT_EQ(result->model_query_count, 6u);
  // Exactly four successors are scored; the seeded initial1 is not scored.
  EXPECT_NEAR(best.log_score, -4 * std::log(1 + 4 * std::exp(-8.0)), 1e-14);
}

TEST(GreedySuffixReadoutTest, EosWithoutCoverageIsAReportedNegativeResult) {
  const std::vector<int> selected{2, 0, 1};
  auto result = ReadGreedySuffix(
      {.vocabulary_size = 5, .eos_token = 0, .selected_token_ids = selected},
      [](absl::Span<const int>) { return Scores(0); });
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->accepted_order.empty());
  ASSERT_EQ(result->candidates.size(), 2u);
  for (const auto& candidate : result->candidates) {
    EXPECT_TRUE(candidate.terminated_with_eos);
    EXPECT_TRUE(candidate.all_tokens_selected);
    EXPECT_FALSE(candidate.covers_selected_tokens);
    EXPECT_EQ(candidate.distinct_selected_covered, 1u);
    EXPECT_FALSE(candidate.hit_token_limit);
  }
}

TEST(GreedySuffixReadoutTest, NeverMasksAnOutsideSelectedTopOne) {
  const std::vector<int> selected{0, 1, 2};
  auto result = ReadGreedySuffix(
      {.vocabulary_size = 5, .eos_token = 0, .selected_token_ids = selected},
      [](absl::Span<const int> history) {
        if (history.size() > 6)
          return Scores(0);
        auto values = Scores(3, 5, 10);
        values[2] = 9;  // Would win incorrectly if ID3 were masked away.
        return values;
      });
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->accepted_order.empty());
  for (const auto& candidate : result->candidates) {
    EXPECT_EQ(candidate.token_ids[1], 3);
    EXPECT_FALSE(candidate.all_tokens_selected);
    EXPECT_TRUE(candidate.terminated_with_eos);
  }
}

TEST(GreedySuffixReadoutTest,
     CapsRepeatingGenerationAndDoesNotScoreRejectedStep) {
  const std::vector<int> selected{0, 1, 2};
  auto result = ReadGreedySuffix(
      {.vocabulary_size = 5, .eos_token = 0, .selected_token_ids = selected},
      [](absl::Span<const int> history) { return Scores(history.back()); });
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->accepted_order.empty());
  EXPECT_EQ(result->model_query_count, 8u);
  for (const auto& candidate : result->candidates) {
    EXPECT_EQ(candidate.token_ids.size(), 4u);
    ASSERT_EQ(candidate.steps.size(), 4u);
    EXPECT_FALSE(candidate.steps.back().appended);
    EXPECT_TRUE(candidate.hit_token_limit);
    EXPECT_FALSE(candidate.terminated_with_eos);
    EXPECT_NEAR(candidate.log_score, -3 * std::log(1 + 4 * std::exp(-8.0)),
                1e-14);
  }
}

TEST(GreedySuffixReadoutTest, AllowsEosImmediatelyAfterTheNonEosCap) {
  const std::vector<int> selected{0, 1, 2};
  auto result = ReadGreedySuffix(
      {.vocabulary_size = 5,
       .eos_token = 0,
       .selected_token_ids = selected,
       .max_non_eos_tokens = 2},
      [](absl::Span<const int> history) {
        return Scores(history.size() == 6 ? 3 - history.back() : 0);
      });
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->accepted_order.size(), 2u);
  for (const auto& candidate : result->candidates) {
    EXPECT_EQ(candidate.token_ids.size(), 3u);
    EXPECT_TRUE(candidate.terminated_with_eos);
    EXPECT_FALSE(candidate.hit_token_limit);
  }
}

TEST(GreedySuffixReadoutTest, SingleSelectedTokenAllowsEosAtTheSmallestCap) {
  const std::vector<int> selected{0, 1};
  auto result =
      ReadGreedySuffix({.vocabulary_size = 5,
                        .eos_token = 0,
                        .selected_token_ids = selected,
                        .max_non_eos_tokens = 1},
                       [](absl::Span<const int>) { return Scores(0); });
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->model_query_count, 1u);
  ASSERT_EQ(result->accepted_order, (std::vector<size_t>{0}));
  EXPECT_EQ(result->candidates[0].token_ids, (std::vector<int>{1, 0}));
  EXPECT_FALSE(result->candidates[0].hit_token_limit);
}

TEST(GreedySuffixReadoutTest, RanksAcceptedCandidatesBySuccessorScores) {
  const std::vector<int> selected{0, 1, 2};
  auto result = ReadGreedySuffix(
      {.vocabulary_size = 5, .eos_token = 0, .selected_token_ids = selected},
      [](absl::Span<const int> history) {
        const int winner = history.size() == 6 ? 3 - history.back() : 0;
        return Scores(winner, 5, history[5] == 2 ? 9 : 3);
      });
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->accepted_order, (std::vector<size_t>{1, 0}));
  EXPECT_GT(result->candidates[1].log_score, result->candidates[0].log_score);
}

TEST(GreedySuffixReadoutTest, EqualScoresUseAscendingStartNotSuppliedSetOrder) {
  const std::vector<int> selected{3, 0, 2, 1};
  auto result = ReadGreedySuffix(
      {.vocabulary_size = 5, .eos_token = 0, .selected_token_ids = selected},
      [](absl::Span<const int> history) {
        // Every competing exponential underflows to zero, making all summed
        // scores EXACTLY zero despite different winner positions in the sum.
        return Scores(history.size() == 8 ? 0 : history.back() % 3 + 1, 5,
                      1000);
      });
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->accepted_order, (std::vector<size_t>{0, 1, 2}));
  for (size_t index = 0; index < result->candidates.size(); ++index)
    EXPECT_EQ(result->candidates[index].start_token,
              static_cast<int>(index) + 1);
}

TEST(GreedySuffixReadoutTest, LogitTiesUseTheLowestVocabularyId) {
  const std::vector<int> selected{0, 1, 2};
  auto result = ReadGreedySuffix(
      {.vocabulary_size = 5, .eos_token = 0, .selected_token_ids = selected},
      [](absl::Span<const int> history) {
        if (history.size() > 6)
          return Scores(0);
        auto values = Scores(1);
        values[2] = values[1];
        return values;
      });
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->accepted_order, (std::vector<size_t>{1}));
  for (const auto& candidate : result->candidates)
    EXPECT_EQ(candidate.steps.front().token, 1);
}

TEST(GreedySuffixReadoutTest, RejectsMalformedIdsWithoutCallingModel) {
  int calls = 0;
  const auto logits = [&](absl::Span<const int>) {
    ++calls;
    return Scores(0);
  };
  for (const auto& selected :
       {std::vector<int>{}, std::vector<int>{0}, std::vector<int>{1, 2},
        std::vector<int>{0, 0, 1}, std::vector<int>{-1, 0},
        std::vector<int>{0, 5}})
    EXPECT_EQ(ReadGreedySuffix({.vocabulary_size = 5,
                                .eos_token = 0,
                                .selected_token_ids = selected},
                               logits)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(calls, 0);
}

TEST(GreedySuffixReadoutTest, RejectsBadOptionsAndEmptyCallback) {
  const std::vector<int> selected{0, 1};
  const auto logits = [](absl::Span<const int>) { return Scores(0); };
  const GreedySuffixReadoutOptions valid{
      .vocabulary_size = 5, .eos_token = 0, .selected_token_ids = selected};
  EXPECT_FALSE(ReadGreedySuffix(valid, {}).ok());
  for (int invalid : {-1, 0}) {
    auto options = valid;
    options.vocabulary_size = invalid;
    EXPECT_FALSE(ReadGreedySuffix(options, logits).ok());
    options = valid;
    options.prompt_token_count = invalid;
    EXPECT_FALSE(ReadGreedySuffix(options, logits).ok());
  }
  auto options = valid;
  options.eos_token = 5;
  EXPECT_FALSE(ReadGreedySuffix(options, logits).ok());
  options = valid;
  options.max_non_eos_tokens = -1;
  EXPECT_FALSE(ReadGreedySuffix(options, logits).ok());
  options = valid;
  options.prompt_token_count = std::numeric_limits<int>::max();
  EXPECT_EQ(ReadGreedySuffix(options, logits).status().code(),
            absl::StatusCode::kOutOfRange);
}

TEST(GreedySuffixReadoutTest, RejectsWrongShapeAndNonfiniteUnselectedLogits) {
  const std::vector<int> selected{0, 1};
  const GreedySuffixReadoutOptions options{
      .vocabulary_size = 5, .eos_token = 0, .selected_token_ids = selected};
  for (int size : {0, 4, 6})
    EXPECT_EQ(ReadGreedySuffix(options,
                               [size](absl::Span<const int>) {
                                 return std::vector<float>(size, 0);
                               })
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  for (float invalid : {std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()})
    EXPECT_EQ(ReadGreedySuffix(options,
                               [invalid](absl::Span<const int>) {
                                 auto values = Scores(0);
                                 values[4] = invalid;
                                 return values;
                               })
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
}

TEST(GreedySuffixReadoutTest, PropagatesCallbackErrors) {
  const std::vector<int> selected{0, 1};
  auto result = ReadGreedySuffix(
      {.vocabulary_size = 5, .eos_token = 0, .selected_token_ids = selected},
      [](absl::Span<const int>) -> absl::StatusOr<std::vector<float>> {
        return absl::FailedPreconditionError("deliberate model failure");
      });
  EXPECT_EQ(result.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(result.status().message(), "deliberate model failure");
}

TEST(GreedySuffixReadoutTest, StableNormalizationAtFiniteFloatExtremes) {
  const std::vector<int> selected{0, 1};
  auto result = ReadGreedySuffix(
      {.vocabulary_size = 3, .eos_token = 0, .selected_token_ids = selected},
      [](absl::Span<const int>) {
        const float maximum = std::numeric_limits<float>::max();
        return std::vector<float>{maximum, maximum, -maximum};
      });
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->accepted_order.size(), 1u);
  EXPECT_NEAR(result->candidates[0].log_score, -std::log(2.0), 1e-14);
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
