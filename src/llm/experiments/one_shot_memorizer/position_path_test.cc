#include "src/llm/experiments/one_shot_memorizer/position_path.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

double LiteralScore(const std::vector<int>& path,
                    const std::vector<double>& scores) {
  const int n = path.size();
  double sum = 0;
  for (int i = 0; i < n; ++i)
    sum += scores[(i * n + path[i]) * (n + 1) + (i == n - 1 ? n : path[i + 1])];
  return sum;
}

TEST(PositionPathTest, OneNodeUsesOnlyTerminalEdge) {
  auto result = SolvePositionPaths(1, {1234.0, -7.5});
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->size(), 1u);
  EXPECT_EQ((*result)[0].nodes, (std::vector<int>{0}));
  EXPECT_DOUBLE_EQ((*result)[0].score, -7.5);
}

TEST(PositionPathTest, PositionAndTerminalScoresDetermineOrder) {
  const int n = 3;
  std::vector<double> scores(n * n * (n + 1), -100);
  scores[(0 * n + 2) * (n + 1) + 0] = -1;
  scores[(1 * n + 0) * (n + 1) + 1] = -2;
  scores[(2 * n + 1) * (n + 1) + n] = -3;
  // A huge early terminal or self edge cannot terminate/repeat a node.
  scores[(0 * n + 2) * (n + 1) + n] = 10000;
  scores[(1 * n + 0) * (n + 1) + 0] = 10000;
  auto result = SolvePositionPaths(n, scores);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->size(), 2u);
  EXPECT_EQ((*result)[0].nodes, (std::vector<int>{2, 0, 1}));
  EXPECT_DOUBLE_EQ((*result)[0].score, -6);
  EXPECT_LT((*result)[1].score, (*result)[0].score);
}

TEST(PositionPathTest, TerminalScoreCanReverseBestPath) {
  std::vector<double> scores(2 * 2 * 3, 0);
  scores[(1 * 2 + 0) * 3 + 2] = -9;
  auto first = SolvePositionPaths(2, scores);
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_EQ((*first)[0].nodes, (std::vector<int>{0, 1}));
  scores[(1 * 2 + 0) * 3 + 2] = 9;
  auto second = SolvePositionPaths(2, scores);
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ((*second)[0].nodes, (std::vector<int>{1, 0}));
}

TEST(PositionPathTest, TwoBestAgreeWithExhaustiveEnumerationIncludingTies) {
  std::mt19937 rng(812731);
  for (int n = 1; n <= 8; ++n)
    for (int trial = 0; trial < 3; ++trial) {
      SCOPED_TRACE(::testing::Message() << "n=" << n << " trial=" << trial);
      std::vector<double> scores(n * n * (n + 1));
      for (double& score : scores)
        score = trial == 0 ? 0 : -double(rng() % 1000000) / 1000;
      auto result = SolvePositionPaths(n, scores);
      ASSERT_TRUE(result.ok()) << result.status();
      ASSERT_EQ(result->size(), n == 1 ? 1u : 2u);
      std::vector<int> permutation(n);
      std::iota(permutation.begin(), permutation.end(), 0);
      std::vector<double> exhaustive;
      do {
        exhaustive.push_back(LiteralScore(permutation, scores));
      } while (std::next_permutation(permutation.begin(), permutation.end()));
      std::sort(exhaustive.begin(), exhaustive.end(), std::greater<double>());
      std::set<std::vector<int>> unique;
      for (size_t rank = 0; rank < result->size(); ++rank) {
        const auto& path = (*result)[rank];
        EXPECT_NEAR(path.score, exhaustive[rank], 1e-8);
        EXPECT_NEAR(path.score, LiteralScore(path.nodes, scores), 1e-8);
        EXPECT_TRUE(unique.insert(path.nodes).second);
        auto sorted = path.nodes;
        std::sort(sorted.begin(), sorted.end());
        EXPECT_EQ(sorted, permutation);
      }
      auto repeated = SolvePositionPaths(n, scores);
      ASSERT_TRUE(repeated.ok()) << repeated.status();
      for (size_t rank = 0; rank < result->size(); ++rank)
        EXPECT_EQ((*result)[rank].nodes, (*repeated)[rank].nodes);
    }
}

TEST(PositionPathTest, RejectsBadNodeCountsAndShape) {
  for (int n : {-1, 0, kMaxPositionPathNodes + 1})
    EXPECT_EQ(SolvePositionPaths(n, {}).status().code(),
              absl::StatusCode::kInvalidArgument);
  for (const auto& scores : {std::vector<double>{}, std::vector<double>{0},
                             std::vector<double>{0, 0, 0}})
    EXPECT_EQ(SolvePositionPaths(1, scores).status().code(),
              absl::StatusCode::kInvalidArgument);
}

TEST(PositionPathTest, RejectsNonfiniteScoresEvenOnUnusedEdges) {
  for (double value : {std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_EQ(SolvePositionPaths(1, {value, 0}).status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(SolvePositionPaths(1, {0, value}).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(PositionPathTest, RejectsOverflowInTransitionsAndTerminal) {
  for (int n : {2, 3})
    for (double value : {std::numeric_limits<double>::max(),
                         -std::numeric_limits<double>::max()}) {
      std::vector<double> scores(n * n * (n + 1), value);
      EXPECT_EQ(SolvePositionPaths(n, scores).status().code(),
                absl::StatusCode::kOutOfRange);
    }
}

TEST(PositionPathScoresTest, PositionAwarePreservesEveryEntry) {
  std::vector<double> scores(2 * 2 * 3);
  std::iota(scores.begin(), scores.end(), -6.0);
  auto prepared = PreparePositionPathScores(
      2, scores, PositionPathScoreMode::kPositionAware);
  ASSERT_TRUE(prepared.ok()) << prepared.status();
  EXPECT_EQ(*prepared, scores);
}

TEST(PositionPathScoresTest, FixedPositionRepeatsExactlyOnePlane) {
  const std::vector<double> plane{1, 2, 3, 4, 5, 6};
  auto prepared = PreparePositionPathScores(
      2, plane, PositionPathScoreMode::kFixedPosition);
  ASSERT_TRUE(prepared.ok()) << prepared.status();
  EXPECT_EQ(*prepared,
            (std::vector<double>{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}));
}

TEST(PositionPathScoresTest, MeanAveragesLogScoresNotProbabilities) {
  const std::vector<double> scores{0,  -2, -4, -6, -8,  -10,
                                   -2, -4, -6, -8, -10, -12};
  auto prepared = PreparePositionPathScores(
      2, scores, PositionPathScoreMode::kMeanPositions);
  ASSERT_TRUE(prepared.ok()) << prepared.status();
  EXPECT_EQ(*prepared, (std::vector<double>{-1, -3, -5, -7, -9, -11, -1, -3, -5,
                                            -7, -9, -11}));
}

TEST(PositionPathScoresTest, RejectsBadShapeModeNonfiniteAndOverflow) {
  for (auto mode : {PositionPathScoreMode::kPositionAware,
                    PositionPathScoreMode::kFixedPosition,
                    PositionPathScoreMode::kMeanPositions}) {
    EXPECT_EQ(PreparePositionPathScores(0, {}, mode).status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(PreparePositionPathScores(17, {}, mode).status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(PreparePositionPathScores(2, {0}, mode).status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(PreparePositionPathScores(
                  1, {std::numeric_limits<double>::quiet_NaN(), 0}, mode)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  EXPECT_EQ(PreparePositionPathScores(1, {0, 0},
                                      static_cast<PositionPathScoreMode>(999))
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(PreparePositionPathScores(
                2, std::vector<double>(12, std::numeric_limits<double>::max()),
                PositionPathScoreMode::kMeanPositions)
                .status()
                .code(),
            absl::StatusCode::kOutOfRange);
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
