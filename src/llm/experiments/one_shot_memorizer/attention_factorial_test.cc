#include "src/llm/experiments/one_shot_memorizer/attention_factorial.h"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

TEST(AttentionFactorialTest, ExactStoredCornersKeepInteractionSeparate) {
  const std::array<double, 2> rr{1, 8}, dr{4, 4}, rd{6, 9}, dd{20, 2};
  const auto result = DecomposeAttentionFactorial({rr, dr, rd, dd});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->total, (std::vector<double>{19, -6}));
  EXPECT_EQ(result->routing, (std::vector<double>{3, -4}));
  EXPECT_EQ(result->values, (std::vector<double>{5, 1}));
  EXPECT_EQ(result->interaction, (std::vector<double>{11, -3}));
  for (size_t i = 0; i < 2; ++i)
    EXPECT_EQ(result->total[i],
              result->routing[i] + result->values[i] + result->interaction[i]);
}

TEST(AttentionFactorialTest, ComponentsMayCancelWithZeroTotalChange) {
  const std::array<double, 1> rr{0}, dr{10}, rd{-6}, dd{0};
  const auto result = DecomposeAttentionFactorial({rr, dr, rd, dd});
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->total[0], 0);
  EXPECT_EQ(result->routing[0], 10);
  EXPECT_EQ(result->values[0], -6);
  EXPECT_EQ(result->interaction[0], -4);
}

TEST(AttentionFactorialTest, IdentityAndAdditiveCorners) {
  const std::array<double, 1> rr{2}, dr{3}, rd{4}, dd{5};
  const auto additive = DecomposeAttentionFactorial({rr, dr, rd, dd});
  ASSERT_TRUE(additive.ok());
  EXPECT_EQ(additive->interaction[0], 0);
  const auto identity = DecomposeAttentionFactorial({rr, rr, rr, rr});
  ASSERT_TRUE(identity.ok());
  EXPECT_EQ(identity->total[0], 0);
  EXPECT_EQ(identity->routing[0], 0);
  EXPECT_EQ(identity->values[0], 0);
  EXPECT_EQ(identity->interaction[0], 0);
}

TEST(AttentionFactorialTest, InvalidCornerShapesNumbersAndOverflow) {
  const std::array<double, 1> a{1},
      nan{std::numeric_limits<double>::quiet_NaN()},
      inf{std::numeric_limits<double>::infinity()},
      large{std::numeric_limits<double>::max()},
      negative{-std::numeric_limits<double>::max()};
  EXPECT_FALSE(DecomposeAttentionFactorial({}).ok());
  EXPECT_FALSE(DecomposeAttentionFactorial({a, {}, a, a}).ok());
  EXPECT_FALSE(DecomposeAttentionFactorial({a, nan, a, a}).ok());
  EXPECT_FALSE(DecomposeAttentionFactorial({a, a, inf, a}).ok());
  EXPECT_FALSE(DecomposeAttentionFactorial({negative, a, a, large}).ok());
}

TEST(AttentionFactorialTest, IdealSplitMatchesFourRealArithmeticCorners) {
  const std::array<float, 2> pr{.75f, .25f}, pd{.25f, .75f};
  const std::array<float, 4> vr{2, 4, 10, 20}, vd{2, 4, 14, 28};
  const auto result = DecomposeIdealAttention(pr, pd, vr, vd, 2, 1);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->routing, (std::vector<double>{4, 8}));
  EXPECT_EQ(result->values, (std::vector<double>{1, 2}));
  EXPECT_EQ(result->interaction, (std::vector<double>{2, 4}));
  EXPECT_EQ(result->shared_prefix_routing, (std::vector<double>{-1, -2}));
  // Shared values can contribute to changed output through routing alone.
  const std::array<double, 2> rr{4, 8}, dr{8, 16}, rd{5, 10}, dd{11, 22};
  const auto observed = DecomposeAttentionFactorial({rr, dr, rd, dd});
  ASSERT_TRUE(observed.ok());
  EXPECT_EQ(observed->routing, result->routing);
  EXPECT_EQ(observed->values, result->values);
  EXPECT_EQ(observed->interaction, result->interaction);
}

TEST(AttentionFactorialTest, IdealChecksShapesSharedRowsAndFiniteValues) {
  const std::array<float, 2> p{.5f, .5f}, v{1, 2}, other{1, 3}, badp{.5f, .4f},
      negative{-.1f, 1.1f}, nan{1, std::numeric_limits<float>::quiet_NaN()};
  EXPECT_FALSE(DecomposeIdealAttention({}, {}, {}, {}, 1, 0).ok());
  EXPECT_FALSE(DecomposeIdealAttention(p, p, v, v, 0, 0).ok());
  EXPECT_FALSE(DecomposeIdealAttention(p, p, v, v, 2, 0).ok());
  EXPECT_FALSE(DecomposeIdealAttention(p, p, v, v, 1, 3).ok());
  EXPECT_FALSE(DecomposeIdealAttention(p, p, v, other, 1, 2).ok());
  EXPECT_FALSE(DecomposeIdealAttention(p, badp, v, v, 1, 0).ok());
  EXPECT_FALSE(DecomposeIdealAttention(p, negative, v, v, 1, 0).ok());
  EXPECT_FALSE(DecomposeIdealAttention(p, p, v, nan, 1, 0).ok());
  EXPECT_FALSE(DecomposeIdealAttention(p, nan, v, v, 1, 0).ok());
  EXPECT_TRUE(DecomposeIdealAttention(p, p, v, other, 1, 1).ok());
}

TEST(AttentionFactorialTest, IdealIdentityHasZeroComponents) {
  const std::array<float, 2> p{.25f, .75f}, v{2, -3};
  const auto result = DecomposeIdealAttention(p, p, v, v, 1, 2);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->routing[0], 0);
  EXPECT_EQ(result->values[0], 0);
  EXPECT_EQ(result->interaction[0], 0);
  EXPECT_EQ(result->shared_prefix_routing[0], 0);
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
