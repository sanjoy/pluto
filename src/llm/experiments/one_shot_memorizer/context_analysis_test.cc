#include "src/llm/experiments/one_shot_memorizer/context_analysis.h"

#include <algorithm>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

using Corpus = std::vector<std::vector<int>>;

constexpr ContextAnalysisOptions kOneTokenPrompt{.prompt_tokens = 1,
                                                 .eos_token = 9};

void ExpectWindow(const ContextWindowStats& stats, int window, int64_t contexts,
                  int64_t conflicts, int64_t targets, int64_t errors) {
  EXPECT_EQ(stats.window, window);
  EXPECT_EQ(stats.contexts, contexts);
  EXPECT_EQ(stats.conflicting_contexts, conflicts);
  EXPECT_EQ(stats.targets, targets);
  EXPECT_EQ(stats.irreducible_top1_errors, errors);
}

TEST(ContextAnalysisTest, LongerSuffixSeparatesDifferentNextTokens) {
  const Corpus corpus{{1, 2, 3}, {4, 2, 5}};
  const auto analysis = AnalyzeContexts(corpus, kOneTokenPrompt);
  ASSERT_TRUE(analysis.ok()) << analysis.status();
  EXPECT_EQ(analysis->longest_prefix, 3);
  EXPECT_EQ(analysis->shortest_exact_window, 2);
  ASSERT_EQ(analysis->windows.size(), 4);
  ExpectWindow(analysis->windows[0], 0, 1, 1, 6, 4);
  ExpectWindow(analysis->windows[1], 1, 5, 1, 6, 1);
  ExpectWindow(analysis->windows[2], 2, 6, 0, 6, 0);
  ExpectWindow(analysis->windows[3], 3, 6, 0, 6, 0);
  ASSERT_EQ(analysis->targets.size(), 6);
  const std::vector<int> expected_windows{1, 2, 1, 1, 2, 1};
  const std::vector<int> expected_tokens{2, 3, 9, 2, 5, 9};
  for (size_t index = 0; index < analysis->targets.size(); ++index) {
    SCOPED_TRACE(index);
    const auto& target = analysis->targets[index];
    EXPECT_EQ(target.sentence_index, index / 3);
    EXPECT_EQ(target.target_index, index % 3 + 1);
    EXPECT_EQ(target.target_token, expected_tokens[index]);
    EXPECT_EQ(target.shortest_sufficient_window, expected_windows[index]);
    EXPECT_EQ(target.shortest_unique_window, expected_windows[index]);
  }
}

TEST(ContextAnalysisTest, MasksFiveTokenPromptAndScoresExactlyOneEos) {
  const Corpus corpus{{1, 2, 3, 4, 5, 6, 7}};
  const auto analysis = AnalyzeContexts(corpus, {.eos_token = 0});
  ASSERT_TRUE(analysis.ok()) << analysis.status();
  EXPECT_EQ(analysis->longest_prefix, 7);
  ASSERT_EQ(analysis->windows.size(), 8);
  ExpectWindow(analysis->windows[0], 0, 1, 1, 3, 2);
  ExpectWindow(analysis->windows[1], 1, 3, 0, 3, 0);
  ASSERT_EQ(analysis->targets.size(), 3);
  EXPECT_EQ(analysis->targets[0].target_index, 5);
  EXPECT_EQ(analysis->targets[0].target_token, 6);
  EXPECT_EQ(analysis->targets[1].target_index, 6);
  EXPECT_EQ(analysis->targets[1].target_token, 7);
  EXPECT_EQ(analysis->targets[2].target_index, 7);
  EXPECT_EQ(analysis->targets[2].target_token, 0);

  const auto model = ContextModel::Build(corpus, 1, {.eos_token = 0});
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ(model->Predict(std::vector<int>{1}).status().code(),
            absl::StatusCode::kNotFound);
  const auto continuation = model->Predict(std::vector<int>{5});
  ASSERT_TRUE(continuation.ok()) << continuation.status();
  EXPECT_EQ(continuation->token, 6);
  const auto eos = model->Predict(corpus[0]);
  ASSERT_TRUE(eos.ok()) << eos.status();
  EXPECT_EQ(eos->token, 0);
  EXPECT_EQ(eos->occurrences, 1);
}

TEST(ContextAnalysisTest, PromptOnlySentenceNeedsNoHistoryToPredictEos) {
  const Corpus corpus{{1, 2, 3, 4, 5}};
  const auto analysis = AnalyzeContexts(corpus);
  ASSERT_TRUE(analysis.ok()) << analysis.status();
  EXPECT_EQ(analysis->shortest_exact_window, 0);
  ASSERT_EQ(analysis->targets.size(), 1);
  EXPECT_EQ(analysis->targets[0].target_token, 50256);
  EXPECT_EQ(analysis->targets[0].shortest_sufficient_window, 0);
  EXPECT_EQ(analysis->targets[0].shortest_unique_window, 0);
  for (int window = 0; window <= 5; ++window)
    ExpectWindow(analysis->windows[window], window, 1, 0, 1, 0);
}

TEST(ContextAnalysisTest, SameLabelDuplicatesAreSufficientButNeverUnique) {
  const Corpus corpus{{1, 2}, {1, 2}};
  const auto analysis = AnalyzeContexts(corpus, kOneTokenPrompt);
  ASSERT_TRUE(analysis.ok()) << analysis.status();
  EXPECT_EQ(analysis->shortest_exact_window, 1);
  ExpectWindow(analysis->windows[0], 0, 1, 1, 4, 2);
  ExpectWindow(analysis->windows[1], 1, 2, 0, 4, 0);
  ExpectWindow(analysis->windows[2], 2, 2, 0, 4, 0);
  ASSERT_EQ(analysis->targets.size(), 4);
  for (const auto& target : analysis->targets) {
    EXPECT_EQ(target.shortest_sufficient_window, 1);
    EXPECT_EQ(target.shortest_unique_window, -1);
  }
}

TEST(ContextAnalysisTest, RepeatedContextsWithinOneSentenceCountAsOccurrences) {
  const Corpus corpus{{1, 2, 1, 2}};
  const auto analysis = AnalyzeContexts(corpus, kOneTokenPrompt);
  ASSERT_TRUE(analysis.ok()) << analysis.status();
  EXPECT_EQ(analysis->shortest_exact_window, 3);
  ExpectWindow(analysis->windows[1], 1, 2, 1, 4, 1);
  ExpectWindow(analysis->windows[2], 2, 3, 1, 4, 1);
  ExpectWindow(analysis->windows[3], 3, 4, 0, 4, 0);
  ASSERT_EQ(analysis->targets.size(), 4);
  EXPECT_EQ(analysis->targets[0].shortest_sufficient_window, 1);
  EXPECT_EQ(analysis->targets[0].shortest_unique_window, 2);
  EXPECT_EQ(analysis->targets[1].shortest_sufficient_window, 3);
  EXPECT_EQ(analysis->targets[1].shortest_unique_window, 3);
  EXPECT_EQ(analysis->targets[2].shortest_sufficient_window, 1);
  EXPECT_EQ(analysis->targets[2].shortest_unique_window, 2);
  EXPECT_EQ(analysis->targets[3].shortest_sufficient_window, 3);
}

TEST(ContextAnalysisTest, RetainsLeftBoundaryWhenWindowExceedsPrefixLength) {
  const Corpus corpus{{1}, {2, 1, 3}};
  const auto analysis = AnalyzeContexts(corpus, kOneTokenPrompt);
  ASSERT_TRUE(analysis.ok()) << analysis.status();
  EXPECT_EQ(analysis->shortest_exact_window, 2);
  ExpectWindow(analysis->windows[1], 1, 3, 1, 4, 1);
  ExpectWindow(analysis->windows[2], 2, 4, 0, 4, 0);
  ASSERT_EQ(analysis->targets.size(), 4);
  EXPECT_EQ(analysis->targets[0].target_index, 1);
  EXPECT_EQ(analysis->targets[0].target_token, 9);
  EXPECT_EQ(analysis->targets[0].shortest_sufficient_window, 2);
  EXPECT_EQ(analysis->targets[0].shortest_unique_window, 2);

  const auto model = ContextModel::Build(corpus, 2, kOneTokenPrompt);
  ASSERT_TRUE(model.ok()) << model.status();
  const auto at_boundary = model->Predict(std::vector<int>{1});
  const auto longer_history = model->Predict(std::vector<int>{2, 1});
  ASSERT_TRUE(at_boundary.ok()) << at_boundary.status();
  ASSERT_TRUE(longer_history.ok()) << longer_history.status();
  EXPECT_EQ(at_boundary->token, 9);
  EXPECT_EQ(longer_history->token, 3);
}

TEST(ContextAnalysisTest, ConflictingIdenticalPrefixesCannotBeSolved) {
  const Corpus corpus{{1, 2}, {1, 3}};
  const auto analysis = AnalyzeContexts(corpus, kOneTokenPrompt);
  ASSERT_TRUE(analysis.ok()) << analysis.status();
  EXPECT_EQ(analysis->shortest_exact_window, -1);
  ExpectWindow(analysis->windows.back(), 2, 3, 1, 4, 1);
  EXPECT_EQ(analysis->targets[0].shortest_sufficient_window, -1);
  EXPECT_EQ(analysis->targets[0].shortest_unique_window, -1);
  EXPECT_EQ(analysis->targets[2].shortest_sufficient_window, -1);
  EXPECT_EQ(analysis->targets[2].shortest_unique_window, -1);
}

TEST(ContextModelTest, ZeroOrderUsesEmpiricalCountsAndLowestTokenTieBreak) {
  const Corpus corpus{{1, 2, 3}, {4, 2, 5}};
  const auto model = ContextModel::Build(corpus, 0, kOneTokenPrompt);
  ASSERT_TRUE(model.ok()) << model.status();
  for (const auto& history : Corpus{{}, {100, 101}, {9}}) {
    const auto prediction = model->Predict(history);
    ASSERT_TRUE(prediction.ok()) << prediction.status();
    EXPECT_EQ(prediction->token, 2);
    EXPECT_DOUBLE_EQ(prediction->probability, 2.0 / 6);
    EXPECT_EQ(prediction->occurrences, 6);
    EXPECT_EQ(prediction->distinct_next_tokens, 4);
  }
  ASSERT_TRUE(model->Probability({}, 3).ok());
  EXPECT_DOUBLE_EQ(*model->Probability({}, 3), 1.0 / 6);
  EXPECT_DOUBLE_EQ(*model->Probability({}, 9), 2.0 / 6);
  EXPECT_DOUBLE_EQ(*model->Probability({}, 100), 0);
}

TEST(ContextModelTest, IgnoresEarlierHistoryAndDoesNotBackOffOnMissingContext) {
  const Corpus corpus{{1, 2, 3}, {4, 2, 5}};
  const auto model = ContextModel::Build(corpus, 2, kOneTokenPrompt);
  ASSERT_TRUE(model.ok()) << model.status();
  const auto prediction = model->Predict(std::vector<int>{100, 101, 1, 2});
  ASSERT_TRUE(prediction.ok()) << prediction.status();
  EXPECT_EQ(prediction->token, 3);
  EXPECT_DOUBLE_EQ(prediction->probability, 1);
  EXPECT_EQ(model->Predict(std::vector<int>{99, 2}).status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(model->Predict(std::vector<int>{2}).status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(model->Predict({}).status().code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(model->Probability(std::vector<int>{99, 2}, 3).status().code(),
            absl::StatusCode::kNotFound);
}

TEST(ContextAnalysisTest, CorpusOrderDoesNotChangeWindowsOrPredictions) {
  Corpus corpus{{1, 2, 3}, {4, 2, 5}, {1, 2, 3}};
  const auto original = AnalyzeContexts(corpus, kOneTokenPrompt);
  ASSERT_TRUE(original.ok()) << original.status();
  Corpus reversed = corpus;
  std::reverse(reversed.begin(), reversed.end());
  // Also move a different sentence into the first position; the duplicate
  // endpoints alone would make a reversal vacuous.
  std::rotate(reversed.begin(), reversed.begin() + 1, reversed.end());
  const auto reordered = AnalyzeContexts(reversed, kOneTokenPrompt);
  ASSERT_TRUE(reordered.ok()) << reordered.status();
  ASSERT_EQ(original->windows.size(), reordered->windows.size());
  EXPECT_EQ(original->shortest_exact_window, reordered->shortest_exact_window);
  for (size_t window = 0; window < original->windows.size(); ++window) {
    const auto& expected = original->windows[window];
    ExpectWindow(reordered->windows[window], expected.window, expected.contexts,
                 expected.conflicting_contexts, expected.targets,
                 expected.irreducible_top1_errors);
    const auto first = ContextModel::Build(corpus, window, kOneTokenPrompt);
    const auto second = ContextModel::Build(reversed, window, kOneTokenPrompt);
    ASSERT_TRUE(first.ok()) << first.status();
    ASSERT_TRUE(second.ok()) << second.status();
    for (const auto& sentence : corpus) {
      for (size_t length = 1; length <= sentence.size(); ++length) {
        const auto history = absl::MakeConstSpan(sentence).first(length);
        const auto first_prediction = first->Predict(history);
        const auto second_prediction = second->Predict(history);
        ASSERT_TRUE(first_prediction.ok()) << first_prediction.status();
        ASSERT_TRUE(second_prediction.ok()) << second_prediction.status();
        EXPECT_EQ(first_prediction->token, second_prediction->token);
        EXPECT_DOUBLE_EQ(first_prediction->probability,
                         second_prediction->probability);
        EXPECT_EQ(first_prediction->occurrences,
                  second_prediction->occurrences);
        EXPECT_EQ(first_prediction->distinct_next_tokens,
                  second_prediction->distinct_next_tokens);
      }
    }
  }
}

TEST(ContextAnalysisTest, RejectsInvalidCorpusAndOptions) {
  for (const auto& corpus :
       std::vector<Corpus>{{}, {{}}, {{1, -2}}, {{1, 9}}}) {
    EXPECT_EQ(AnalyzeContexts(corpus, kOneTokenPrompt).status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(ContextModel::Build(corpus, 1, kOneTokenPrompt).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
  const Corpus corpus{{1, 2}};
  for (const auto& options : std::vector<ContextAnalysisOptions>{
           {.prompt_tokens = 0},
           {.prompt_tokens = -1},
           {.prompt_tokens = 3},
           {.prompt_tokens = 1, .eos_token = -1}}) {
    EXPECT_EQ(AnalyzeContexts(corpus, options).status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(ContextModel::Build(corpus, 1, options).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
  EXPECT_EQ(ContextModel::Build(corpus, -1, kOneTokenPrompt).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(ContextModelTest, RejectsNegativeQueryTokensEvenOutsideTheSuffix) {
  const Corpus corpus{{1, 2}};
  const auto model = ContextModel::Build(corpus, 1, kOneTokenPrompt);
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ(model->Predict(std::vector<int>{-1, 1}).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(model->Probability(std::vector<int>{-1, 1}, 2).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(model->Probability(std::vector<int>{1}, -1).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
