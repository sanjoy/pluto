#include "src/llm/experiments/one_shot_memorizer/fact_superposition.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

TEST(FactSuperpositionTest, AppliesBothFixedFormulasAndIsSymmetric) {
  const std::vector<float> initial{1, -2, 0, 8}, a{2, -5, -3, 10},
      b{4, -1, 2, 6};
  const auto sum =
      SuperposeFactParameters(initial, a, b, FactSuperposition::kSum);
  const auto mean =
      SuperposeFactParameters(initial, a, b, FactSuperposition::kMean);
  ASSERT_TRUE(sum.ok()) << sum.status();
  ASSERT_TRUE(mean.ok()) << mean.status();
  EXPECT_EQ(*sum, (std::vector<float>{5, -4, -1, 8}));
  EXPECT_EQ(*mean, (std::vector<float>{3, -3, -0.5, 8}));
  for (auto method : {FactSuperposition::kSum, FactSuperposition::kMean}) {
    const auto forward = SuperposeFactParameters(initial, a, b, method);
    const auto reverse = SuperposeFactParameters(initial, b, a, method);
    ASSERT_TRUE(forward.ok());
    ASSERT_TRUE(reverse.ok());
    EXPECT_EQ(*forward, *reverse);
  }
}

TEST(FactSuperpositionTest, OneUnchangedComponentIsExactStepOneControl) {
  const std::vector<float> initial{1.0f, -0.125f, 0.0f, 2.0f};
  const std::vector<float> trained{1.0001f, -0.124f, 0.000006f, 1.9999f};
  const auto result = SuperposeFactParameters(initial, trained, initial,
                                              FactSuperposition::kSum);
  ASSERT_TRUE(result.ok()) << result.status();
  for (size_t i = 0; i < initial.size(); ++i)
    EXPECT_EQ(std::bit_cast<uint32_t>((*result)[i]),
              std::bit_cast<uint32_t>(trained[i]));
}

TEST(FactSuperpositionTest, UsesDoubleDeltasAndOnlyOneFinalFloatRounding) {
  // FP32 subtraction would lose the unit correction before cancellation.
  const auto cancellation = SuperposeFactParameters(
      {33554432.0f}, {1.0f}, {33554432.0f}, FactSuperposition::kSum);
  ASSERT_TRUE(cancellation.ok()) << cancellation.status();
  EXPECT_FLOAT_EQ((*cancellation)[0], 1);
  const float next = std::nextafter(1.0f, 2.0f);
  const float next_next = std::nextafter(next, 2.0f);
  const auto even_down =
      SuperposeFactParameters({0}, {1}, {next}, FactSuperposition::kMean);
  const auto even_up = SuperposeFactParameters({0}, {next}, {next_next},
                                               FactSuperposition::kMean);
  ASSERT_TRUE(even_down.ok()) << even_down.status();
  ASSERT_TRUE(even_up.ok()) << even_up.status();
  EXPECT_FLOAT_EQ((*even_down)[0], 1);
  EXPECT_FLOAT_EQ((*even_up)[0], next_next);
}

TEST(FactSuperpositionTest, RejectsInvalidShapeModeAndNonfiniteValues) {
  EXPECT_FALSE(
      SuperposeFactParameters({}, {}, {}, FactSuperposition::kSum).ok());
  EXPECT_FALSE(
      SuperposeFactParameters({1}, {}, {1}, FactSuperposition::kSum).ok());
  EXPECT_FALSE(
      SuperposeFactParameters({1}, {1}, {}, FactSuperposition::kSum).ok());
  EXPECT_FALSE(
      SuperposeFactParameters({1}, {1}, {1}, static_cast<FactSuperposition>(3))
          .ok());
  for (float bad : {std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN()})
    for (auto method : {FactSuperposition::kSum, FactSuperposition::kMean}) {
      EXPECT_FALSE(SuperposeFactParameters({bad}, {1}, {1}, method).ok());
      EXPECT_FALSE(SuperposeFactParameters({1}, {bad}, {1}, method).ok());
      EXPECT_FALSE(SuperposeFactParameters({1}, {1}, {bad}, method).ok());
    }
}

TEST(FactSuperpositionTest, ChecksFinalRangeNotIntermediateFloatRange) {
  const float largest = std::numeric_limits<float>::max();
  EXPECT_FALSE(SuperposeFactParameters({-largest}, {largest}, {largest},
                                       FactSuperposition::kSum)
                   .ok());
  const auto mean = SuperposeFactParameters({-largest}, {largest}, {largest},
                                            FactSuperposition::kMean);
  ASSERT_TRUE(mean.ok()) << mean.status();
  EXPECT_FLOAT_EQ((*mean)[0], largest);
  const auto small =
      SuperposeFactParameters({0}, {std::numeric_limits<float>::denorm_min()},
                              {0}, FactSuperposition::kMean);
  ASSERT_TRUE(small.ok()) << small.status();
  EXPECT_FLOAT_EQ((*small)[0], 0);
}

TEST(FactSuperpositionTest, DecomposesSignedCoordinatesWithoutChangingInputs) {
  const std::vector<float> joint{1, -2, 0, 8}, frozen{2, -5, -3, 10},
      sum{4, -1, 2, 6};
  const auto actual = DecomposeFactSuperposition(joint, frozen, sum);
  ASSERT_TRUE(actual.ok()) << actual.status();
  EXPECT_EQ(actual->total, (std::vector<double>{-3, -1, -2, 2}));
  EXPECT_EQ(actual->gradient_trajectory, (std::vector<double>{-1, 3, 3, -2}));
  EXPECT_EQ(actual->optimizer_history, (std::vector<double>{-2, -4, -5, 4}));
  EXPECT_DOUBLE_EQ(actual->maximum_absolute_closure_error, 0);
  EXPECT_EQ(joint, (std::vector<float>{1, -2, 0, 8}));
  EXPECT_EQ(frozen, (std::vector<float>{2, -5, -3, 10}));
  EXPECT_EQ(sum, (std::vector<float>{4, -1, 2, 6}));
}

TEST(FactSuperpositionTest, DecompositionComponentsCanExceedTotalByCancelling) {
  const auto actual = DecomposeFactSuperposition({1}, {100}, {0});
  ASSERT_TRUE(actual.ok()) << actual.status();
  EXPECT_DOUBLE_EQ(actual->total[0], 1);
  EXPECT_DOUBLE_EQ(actual->gradient_trajectory[0], -99);
  EXPECT_DOUBLE_EQ(actual->optimizer_history[0], 100);
  EXPECT_GT(std::abs(actual->gradient_trajectory[0]),
            std::abs(actual->total[0]));
  EXPECT_GT(std::abs(actual->optimizer_history[0]), std::abs(actual->total[0]));
  EXPECT_DOUBLE_EQ(actual->maximum_absolute_closure_error, 0);
}

TEST(FactSuperpositionTest, UnchangedEndpointsHaveNoEffect) {
  const std::vector<float> weights{0, -0.0f, 1, -2.5f};
  const auto actual = DecomposeFactSuperposition(weights, weights, weights);
  ASSERT_TRUE(actual.ok()) << actual.status();
  const std::vector<double> zero(weights.size(), 0);
  EXPECT_EQ(actual->total, zero);
  EXPECT_EQ(actual->gradient_trajectory, zero);
  EXPECT_EQ(actual->optimizer_history, zero);
  EXPECT_DOUBLE_EQ(actual->maximum_absolute_closure_error, 0);
}

TEST(FactSuperpositionTest, ExtremeExponentsReportRatherThanHideClosureError) {
  const float large = std::numeric_limits<float>::max();
  // FP64 cannot retain the tiny 1/2 terms alongside an exponent near 128.
  // The real identity still holds; its computed components do not close.
  const auto actual =
      DecomposeFactSuperposition({1, 2}, {large, -large}, {0, 0});
  ASSERT_TRUE(actual.ok()) << actual.status();
  EXPECT_EQ(actual->total, (std::vector<double>{1, 2}));
  for (size_t i = 0; i < actual->total.size(); ++i)
    EXPECT_DOUBLE_EQ(
        actual->gradient_trajectory[i] + actual->optimizer_history[i], 0);
  EXPECT_DOUBLE_EQ(actual->maximum_absolute_closure_error, 2);
  // Unlike the constructed FP32 weights, these differences remain FP64 and
  // may legitimately exceed the largest finite FP32 value.
  const auto wide = DecomposeFactSuperposition({large}, {0}, {-large});
  ASSERT_TRUE(wide.ok()) << wide.status();
  EXPECT_DOUBLE_EQ(wide->total[0], 2 * static_cast<double>(large));
  EXPECT_DOUBLE_EQ(wide->maximum_absolute_closure_error, 0);
}

TEST(FactSuperpositionTest, DecompositionRejectsBadShapesAndNonfiniteInputs) {
  EXPECT_FALSE(DecomposeFactSuperposition({}, {}, {}).ok());
  EXPECT_FALSE(DecomposeFactSuperposition({1}, {}, {1}).ok());
  EXPECT_FALSE(DecomposeFactSuperposition({1}, {1}, {}).ok());
  EXPECT_FALSE(DecomposeFactSuperposition({1}, {1, 2}, {1}).ok());
  for (float bad : {std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN()}) {
    EXPECT_FALSE(DecomposeFactSuperposition({bad}, {1}, {1}).ok());
    EXPECT_FALSE(DecomposeFactSuperposition({1}, {bad}, {1}).ok());
    EXPECT_FALSE(DecomposeFactSuperposition({1}, {1}, {bad}).ok());
  }
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
