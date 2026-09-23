#include "src/llm/experiments/one_shot_memorizer/decision_readout.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

TEST(DecisionReadoutTest, MatchesLiteralLayerNormAndHeadWithArbitraryRowMeans) {
  const std::vector<float> embeddings{1,   2,  -3,   4,    -2, 1,
                                      0.5, -1, 0.25, -0.5, 2,  3};
  const std::vector<float> gamma{2, -0.5, 0, 1.25};
  const std::vector<float> beta{0.25, -1, 2, -0.5};
  const std::vector<float> residuals{1,  3,  5,  -1, 1001, 1003, 1005, 999,
                                     -7, -7, -7, -7, -4,   2,    1,    8};
  constexpr double epsilon = 0.03125;
  auto readout = MakeDecisionReadout(embeddings, 4, gamma, beta, epsilon);
  ASSERT_TRUE(readout.ok()) << readout.status();
  EXPECT_EQ(readout->width, 4);
  EXPECT_EQ(readout->vocab_size, 3);
  EXPECT_EQ(readout->epsilon, epsilon);
  auto scores = EvaluateDecisionReadout(*readout, residuals);
  ASSERT_TRUE(scores.ok()) << scores.status();
  ASSERT_EQ(scores->size(), 12u);
  for (int token = 0; token < 3; ++token) {
    double direction_sum = 0;
    for (int dim = 0; dim < 4; ++dim)
      direction_sum += readout->directions[token * 4 + dim];
    EXPECT_NEAR(direction_sum, 0, 1e-14);
  }
  for (int row = 0; row < 4; ++row) {
    double mean = 0;
    for (int dim = 0; dim < 4; ++dim)
      mean += residuals[row * 4 + dim];
    mean /= 4;
    double variance = 0;
    for (int dim = 0; dim < 4; ++dim) {
      const double centered = residuals[row * 4 + dim] - mean;
      variance += centered * centered / 4;
    }
    for (int token = 0; token < 3; ++token) {
      double literal = 0;
      for (int dim = 0; dim < 4; ++dim) {
        const double norm = gamma[dim] * (residuals[row * 4 + dim] - mean) /
                                std::sqrt(variance + epsilon) +
                            beta[dim];
        literal += norm * embeddings[token * 4 + dim];
      }
      EXPECT_NEAR((*scores)[row * 3 + token], literal, 1e-12);
    }
  }
  for (int token = 0; token < 3; ++token) {
    EXPECT_DOUBLE_EQ((*scores)[token], (*scores)[3 + token]);
    EXPECT_DOUBLE_EQ((*scores)[6 + token], readout->offsets[token]);
  }
}

TEST(DecisionReadoutTest, ZeroBetaAllowsRemovingThePositiveNormalizationScale) {
  const std::vector<float> embeddings{1, -1, 2, -3, 2, 1, 0, -1, -1};
  const std::vector<float> gamma{2, -0.5, 1.5}, beta(3);
  const std::vector<float> residuals{5, -1, 2, 1, 0, -2, -6, 2, -1};
  auto readout = MakeDecisionReadout(embeddings, 3, gamma, beta);
  ASSERT_TRUE(readout.ok()) << readout.status();
  auto scores = EvaluateDecisionReadout(*readout, residuals);
  ASSERT_TRUE(scores.ok()) << scores.status();
  for (int row = 0; row < 3; ++row) {
    std::vector<double> raw(3);
    for (int token = 0; token < 3; ++token)
      for (int dim = 0; dim < 3; ++dim)
        raw[token] +=
            readout->directions[token * 3 + dim] * residuals[row * 3 + dim];
    const auto first = scores->begin() + row * 3;
    EXPECT_EQ(std::max_element(first, first + 3) - first,
              std::max_element(raw.begin(), raw.end()) - raw.begin());
  }
}

TEST(DecisionReadoutTest, NonzeroBetaChangesTheWinnerAndCannotBeDropped) {
  const std::vector<float> embeddings{1, 0, 0, 1};
  const std::vector<float> gamma{1, 1}, beta{0, 3}, zeros(2);
  const std::vector<float> residuals{1, -1};
  auto original = MakeDecisionReadout(embeddings, 2, gamma, beta);
  auto without_beta = MakeDecisionReadout(embeddings, 2, gamma, zeros);
  ASSERT_TRUE(original.ok());
  ASSERT_TRUE(without_beta.ok());
  EXPECT_EQ(original->directions, without_beta->directions);
  EXPECT_EQ(original->offsets, (std::vector<double>{0, 3}));
  auto original_scores = EvaluateDecisionReadout(*original, residuals);
  auto unbiased_scores = EvaluateDecisionReadout(*without_beta, residuals);
  ASSERT_TRUE(original_scores.ok());
  ASSERT_TRUE(unbiased_scores.ok());
  EXPECT_GT((*original_scores)[1], (*original_scores)[0]);
  EXPECT_GT((*unbiased_scores)[0], (*unbiased_scores)[1]);
}

TEST(DecisionReadoutTest,
     HandlesOneCoordinateEmptyRowsAndExtremeFiniteResiduals) {
  auto scalar =
      MakeDecisionReadout(std::vector<float>{2, -3}, 1, std::vector<float>{5},
                          std::vector<float>{7});
  ASSERT_TRUE(scalar.ok());
  EXPECT_EQ(scalar->directions, (std::vector<double>{0, 0}));
  auto scalar_scores =
      EvaluateDecisionReadout(*scalar, std::vector<float>{-100, 80});
  ASSERT_TRUE(scalar_scores.ok());
  EXPECT_EQ(*scalar_scores, (std::vector<double>{14, -21, 14, -21}));
  auto empty = EvaluateDecisionReadout(*scalar, {});
  ASSERT_TRUE(empty.ok());
  EXPECT_TRUE(empty->empty());
  auto pair =
      MakeDecisionReadout(std::vector<float>{1, -1}, 2,
                          std::vector<float>{1, 1}, std::vector<float>{0, 0});
  ASSERT_TRUE(pair.ok());
  const float largest = std::numeric_limits<float>::max();
  auto extreme =
      EvaluateDecisionReadout(*pair, std::vector<float>{largest, -largest});
  ASSERT_TRUE(extreme.ok()) << extreme.status();
  EXPECT_NEAR((*extreme)[0], 2, 1e-14);
}

TEST(DecisionReadoutTest, RejectsMalformedAndNonfiniteParameters) {
  const std::vector<float> embeddings{1, -1}, gamma{1, 1}, beta{0, 0};
  for (int width : {-1, 0, 1, 3, std::numeric_limits<int>::max()})
    EXPECT_FALSE(MakeDecisionReadout(embeddings, width, gamma, beta).ok());
  EXPECT_FALSE(MakeDecisionReadout({}, 2, gamma, beta).ok());
  EXPECT_FALSE(
      MakeDecisionReadout(embeddings, 2, std::vector<float>{1}, beta).ok());
  EXPECT_FALSE(
      MakeDecisionReadout(embeddings, 2, gamma, std::vector<float>{0}).ok());
  for (double epsilon : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()})
    EXPECT_FALSE(MakeDecisionReadout(embeddings, 2, gamma, beta, epsilon).ok());
  for (float bad : {std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN()}) {
    const std::vector<float> invalid{bad, 0};
    EXPECT_FALSE(MakeDecisionReadout(invalid, 2, gamma, beta).ok());
    EXPECT_FALSE(MakeDecisionReadout(embeddings, 2, invalid, beta).ok());
    EXPECT_FALSE(MakeDecisionReadout(embeddings, 2, gamma, invalid).ok());
  }
}

TEST(DecisionReadoutTest,
     RejectsMalformedResidualsReadoutAndArithmeticOverflow) {
  const std::vector<float> residuals{1, -1};
  auto readout =
      MakeDecisionReadout(std::vector<float>{1, -1}, 2,
                          std::vector<float>{1, 1}, std::vector<float>{0, 0});
  ASSERT_TRUE(readout.ok());
  EXPECT_FALSE(EvaluateDecisionReadout(*readout, std::vector<float>{1}).ok());
  for (float bad : {std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN()})
    EXPECT_FALSE(
        EvaluateDecisionReadout(*readout, std::vector<float>{bad, 1}).ok());
  EXPECT_FALSE(EvaluateDecisionReadout({}, residuals).ok());
  auto invalid = *readout;
  invalid.vocab_size = std::numeric_limits<int>::max();
  EXPECT_FALSE(EvaluateDecisionReadout(invalid, residuals).ok());
  invalid = *readout;
  invalid.directions.pop_back();
  EXPECT_FALSE(EvaluateDecisionReadout(invalid, residuals).ok());
  invalid = *readout;
  invalid.offsets.clear();
  EXPECT_FALSE(EvaluateDecisionReadout(invalid, residuals).ok());
  invalid = *readout;
  invalid.epsilon = 0;
  EXPECT_FALSE(EvaluateDecisionReadout(invalid, residuals).ok());
  invalid = *readout;
  invalid.directions[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(EvaluateDecisionReadout(invalid, residuals).ok());
  invalid = *readout;
  invalid.offsets[0] = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(EvaluateDecisionReadout(invalid, residuals).ok());
  invalid = *readout;
  invalid.directions = {std::numeric_limits<double>::max(),
                        -std::numeric_limits<double>::max()};
  const auto overflow = EvaluateDecisionReadout(invalid, residuals);
  EXPECT_FALSE(overflow.ok());
  EXPECT_EQ(overflow.status().code(), absl::StatusCode::kOutOfRange);
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
