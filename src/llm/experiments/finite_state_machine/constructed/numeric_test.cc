#include "src/llm/experiments/finite_state_machine/constructed/numeric.h"

#include <cmath>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::fsm::constructed {
namespace {

TEST(AffineTest, AppliesSparseWeightsBiasesAndRepeatedCoefficients) {
  Affine affine("example", 3, 2);
  affine.Add(0, 0, 2);
  affine.Add(0, 0, -0.5);
  affine.Add(0, 2, -1);
  affine.Add(1, 1, 3);
  affine.Add(1, 2, 0);
  affine.Bias(0, 4);
  affine.Bias(1, -2);

  EXPECT_EQ(affine.Apply({2, 3, 5}), (Vector{2, 7}));
  EXPECT_EQ(affine.name(), "example");
  EXPECT_EQ(affine.input_size(), 3);
  EXPECT_EQ(affine.output_size(), 2);
  ASSERT_EQ(affine.coefficients().size(), size_t{4});
  EXPECT_EQ(affine.coefficients()[0].row, 0);
  EXPECT_EQ(affine.coefficients()[0].column, 0);
  EXPECT_EQ(affine.coefficients()[0].value, 2);
  EXPECT_EQ(affine.coefficients()[1].value, -0.5);
  EXPECT_EQ(affine.bias()[0], 4);
  EXPECT_EQ(affine.bias()[1], -2);
}

TEST(AffineTest, BiasSetsInsteadOfAddingAndMayProvideAConstantMap) {
  Affine affine("constant", 0, 2);
  affine.Bias(0, 1);
  affine.Bias(0, 5);
  affine.Bias(1, -3);
  EXPECT_EQ(affine.Apply({}), (Vector{5, -3}));
  EXPECT_TRUE(affine.coefficients().empty());
}

TEST(NumericTest, ReluAndResidualAdditionAreOrdinaryNumericalOperations) {
  const Vector input = {-2, 0, 0.125, 4};
  EXPECT_EQ(Relu(input), (Vector{0, 0, 0.125, 4}));
  EXPECT_EQ(input[0], -2);
  EXPECT_EQ(AddVectors(input, {3, -1, 0.875, 2}), (Vector{1, -1, 1, 6}));
}

TEST(NumericTest, SparsifyKeepsSortedNonzerosWithoutQuantization) {
  EXPECT_EQ(Sparsify({0, -0.0, 3.5, 0, -2}), (SparseVector{{2, 3.5}, {4, -2}}));
  EXPECT_EQ(Sparsify({1e-100}), (SparseVector{{0, 1e-100}}));
  EXPECT_TRUE(Sparsify({0, 0}).empty());
  EXPECT_TRUE(Sparsify({}).empty());
}

TEST(AttentionTest, ReturnsASoftMixtureRatherThanTheWinningValue) {
  const std::vector<MemoryEntry> memory = {{{}, {{0, 2}}},
                                           {{{0, 1}}, {{0, 6}}}};
  const AttentionResult result = Attend({1}, memory, 1);
  const double probability = std::exp(1.0) / (1 + std::exp(1.0));
  ASSERT_EQ(result.output.size(), size_t{1});
  EXPECT_NEAR(result.output[0], (1 - probability) * 2 + probability * 6, 1e-14);
  EXPECT_GT(result.output[0], 2);
  EXPECT_LT(result.output[0], 6);
  EXPECT_EQ(result.winning_index, size_t{1});
  EXPECT_NEAR(result.winning_probability, probability, 1e-15);
}

TEST(AttentionTest, IsStableUnderLargePositiveAndNegativeScoreShifts) {
  const std::vector<MemoryEntry> original = {{{}, {{0, 2}}},
                                             {{{0, 1}}, {{0, 6}}}};
  const AttentionResult expected = Attend({1}, original, 1);
  for (const double shift : {-1000.0, 1000.0}) {
    const std::vector<MemoryEntry> shifted = {{{{0, shift}}, {{0, 2}}},
                                              {{{0, shift + 1}}, {{0, 6}}}};
    const AttentionResult result = Attend({1}, shifted, 1);
    EXPECT_DOUBLE_EQ(result.output[0], expected.output[0]);
    EXPECT_DOUBLE_EQ(result.winning_probability, expected.winning_probability);
    EXPECT_EQ(result.winning_index, expected.winning_index);
  }
}

TEST(AttentionTest, SparseDotProductsIgnoreAbsentCoordinates) {
  const std::vector<MemoryEntry> memory = {
      {{{0, 100}, {1, 3}, {2, 200}, {3, 2}}, {{0, 1}, {2, 4}}},
      {{{0, -900}, {1, 0.5}}, {{1, 2}, {2, -2}}}};
  // The first dot product is 2*3 - 3*2 = 0; the second is 2*0.5 = 1.
  const AttentionResult result = Attend({0, 2, 0, -3}, memory, 4);
  const double probability = std::exp(1.0) / (1 + std::exp(1.0));
  ASSERT_EQ(result.output.size(), size_t{4});
  EXPECT_NEAR(result.output[0], 1 - probability, 1e-15);
  EXPECT_NEAR(result.output[1], 2 * probability, 1e-15);
  EXPECT_NEAR(result.output[2], 4 * (1 - probability) - 2 * probability, 1e-15);
  EXPECT_EQ(result.output[3], 0);
  EXPECT_EQ(result.winning_index, size_t{1});
}

TEST(AttentionTest, ZeroQueryGivesUniformAttentionAndFirstIndexWinsTies) {
  const std::vector<MemoryEntry> memory = {
      {{{0, 8}}, {{0, 1}}}, {{{0, -3}}, {{0, 5}}}, {{}, {{0, 12}}}};
  const AttentionResult result = Attend({0}, memory, 1);
  EXPECT_DOUBLE_EQ(result.output[0], 6);
  EXPECT_EQ(result.winning_index, size_t{0});
  EXPECT_DOUBLE_EQ(result.winning_probability, 1.0 / 3);
}

TEST(AttentionTest, OneMemoryEntryCopiesItsValueIncludingImplicitZeros) {
  const std::vector<MemoryEntry> memory = {{{{0, -23}}, {{1, 4}, {3, -6}}}};
  const AttentionResult result = Attend({7}, memory, 5);
  EXPECT_EQ(result.output, (Vector{0, 4, 0, -6, 0}));
  EXPECT_EQ(result.winning_index, size_t{0});
  EXPECT_EQ(result.winning_probability, 1);
}

TEST(AttentionTest, SharpAttentionStillIncludesTheLosingValue) {
  const std::vector<MemoryEntry> memory = {{{{0, 15}}, {{0, 1}}},
                                           {{{0, 14}}, {}}};
  const AttentionResult result = Attend({16}, memory, 1);
  const double probability = 1 / (1 + std::exp(-16.0));
  EXPECT_DOUBLE_EQ(result.output[0], probability);
  EXPECT_DOUBLE_EQ(result.winning_probability, probability);
  EXPECT_GT(result.output[0], 0.999);
  EXPECT_LT(result.output[0], 1);
}

TEST(AttentionTest, EmptyCoordinatesAndOutputsStillHaveValidProbabilities) {
  const std::vector<MemoryEntry> memory = {{{}, {}}, {{}, {}}};
  const AttentionResult result = Attend({}, memory, 0);
  EXPECT_TRUE(result.output.empty());
  EXPECT_EQ(result.winning_index, size_t{0});
  EXPECT_EQ(result.winning_probability, 0.5);
}

}  // namespace
}  // namespace pluto::llm::fsm::constructed
