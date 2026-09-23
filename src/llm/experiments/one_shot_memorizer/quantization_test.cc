#include "src/llm/experiments/one_shot_memorizer/quantization.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

TEST(QuantizationTest, RoundsHalfwayCodesAwayFromZero) {
  const std::vector<float> input{-3, -2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5, 3};
  auto quantized = QuantizeTensorSymmetric(input, 3);
  ASSERT_TRUE(quantized.ok()) << quantized.status();
  EXPECT_EQ(quantized->bits, 3);
  EXPECT_EQ(quantized->maximum_code, 3);
  EXPECT_EQ(quantized->scale, 1);
  EXPECT_EQ(quantized->values,
            (std::vector<float>{-3, -3, -2, -1, 0, 1, 2, 3, 3}));
  EXPECT_EQ(input,
            (std::vector<float>{-3, -2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5, 3}));

  auto ternary =
      QuantizeTensorSymmetric(std::vector<float>{-3, -1.5, 0, 1.5, 3}, 2);
  ASSERT_TRUE(ternary.ok());
  EXPECT_EQ(ternary->maximum_code, 1);
  EXPECT_EQ(ternary->scale, 3);
  EXPECT_EQ(ternary->values, (std::vector<float>{-3, -3, 0, 3, 3}));
}

TEST(QuantizationTest, EveryIntegerCodeRoundTripsWithOneUnusedCode) {
  for (int bits : {2, 3, 4, 8, 16}) {
    SCOPED_TRACE(bits);
    const int maximum_code = (1 << (bits - 1)) - 1;
    std::vector<float> codes;
    for (int code = -maximum_code; code <= maximum_code; ++code)
      codes.push_back(static_cast<float>(code));
    auto quantized = QuantizeTensorSymmetric(codes, bits);
    ASSERT_TRUE(quantized.ok()) << quantized.status();
    EXPECT_EQ(quantized->values.size(), static_cast<size_t>((1 << bits) - 1));
    EXPECT_EQ(quantized->maximum_code, maximum_code);
    EXPECT_EQ(quantized->scale, 1);
    EXPECT_EQ(quantized->values, codes);
  }
}

TEST(QuantizationTest, IsSymmetricDeterministicAndCanonicallyZero) {
  const std::vector<float> input{-7.25, -5.1, -1.8, -0.01, -0.0,
                                 0.0,   0.01, 1.8,  5.1,   7.25};
  for (int bits : {2, 4, 8, 16}) {
    SCOPED_TRACE(bits);
    auto first = QuantizeTensorSymmetric(input, bits);
    auto second = QuantizeTensorSymmetric(input, bits);
    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(second.ok());
    EXPECT_EQ(first->scale, second->scale);
    EXPECT_EQ(first->scale, 7.25 / first->maximum_code);
    for (size_t index = 0; index < input.size(); ++index) {
      EXPECT_EQ(std::bit_cast<uint32_t>(first->values[index]),
                std::bit_cast<uint32_t>(second->values[index]));
      EXPECT_EQ(first->values[index], -first->values[input.size() - index - 1]);
      if (first->values[index] != 0)
        continue;
      EXPECT_EQ(std::bit_cast<uint32_t>(first->values[index]), 0u);
    }
  }
}

TEST(QuantizationTest, ZeroTensorHasZeroScaleWithoutDivision) {
  for (int bits : {2, 8, 16}) {
    auto quantized =
        QuantizeTensorSymmetric(std::vector<float>{0, -0.0, 0}, bits);
    ASSERT_TRUE(quantized.ok()) << quantized.status();
    EXPECT_EQ(quantized->scale, 0);
    EXPECT_EQ(quantized->bits, bits);
    EXPECT_EQ(quantized->maximum_code, (1 << (bits - 1)) - 1);
    for (float value : quantized->values)
      EXPECT_EQ(std::bit_cast<uint32_t>(value), 0u);
  }
}

TEST(QuantizationTest, ErrorIsAtMostHalfStepPlusFp32StorageRounding) {
  std::vector<float> input;
  for (int index = -997; index <= 997; ++index)
    input.push_back(static_cast<float>(index * 3.125 / 997));
  for (int bits : {2, 3, 4, 8, 16}) {
    SCOPED_TRACE(bits);
    auto quantized = QuantizeTensorSymmetric(input, bits);
    ASSERT_TRUE(quantized.ok()) << quantized.status();
    ASSERT_EQ(quantized->values.size(), input.size());
    const double storage_tolerance =
        3.125 * std::numeric_limits<float>::epsilon();
    for (size_t index = 0; index < input.size(); ++index) {
      const double error = std::abs(static_cast<double>(input[index]) -
                                    quantized->values[index]);
      EXPECT_LE(error, quantized->scale / 2 + storage_tolerance);
      const double code =
          std::round(quantized->values[index] / quantized->scale);
      EXPECT_LE(std::abs(code), quantized->maximum_code);
      const double expected = std::round(input[index] / quantized->scale);
      EXPECT_EQ(code, expected);
    }
  }
}

TEST(QuantizationTest, HandlesFiniteFp32ExtremesAndSubnormalScales) {
  const float largest = std::numeric_limits<float>::max();
  const float smallest = std::numeric_limits<float>::denorm_min();
  for (int bits : {2, 4, 8, 16}) {
    SCOPED_TRACE(bits);
    auto large = QuantizeTensorSymmetric(
        std::vector<float>{-largest, -1, 0, 1, largest}, bits);
    ASSERT_TRUE(large.ok()) << large.status();
    EXPECT_EQ(large->values.front(), -largest);
    EXPECT_EQ(large->values.back(), largest);
    EXPECT_TRUE(std::isfinite(large->scale));
    for (float value : large->values)
      EXPECT_TRUE(std::isfinite(value));
    auto small = QuantizeTensorSymmetric(
        std::vector<float>{-smallest, 0, smallest}, bits);
    ASSERT_TRUE(small.ok()) << small.status();
    EXPECT_GT(small->scale, 0);
    EXPECT_EQ(small->scale,
              static_cast<double>(smallest) / small->maximum_code);
    EXPECT_EQ(small->values, (std::vector<float>{-smallest, 0, smallest}));
  }
}

TEST(QuantizationTest, RejectsEmptyNonfiniteInputAndInvalidBits) {
  EXPECT_FALSE(QuantizeTensorSymmetric({}, 8).ok());
  for (int bits : {std::numeric_limits<int>::min(), -1, 0, 1, 17,
                   std::numeric_limits<int>::max()})
    EXPECT_FALSE(QuantizeTensorSymmetric(std::vector<float>{1}, bits).ok());
  for (float value : {std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()})
    EXPECT_FALSE(
        QuantizeTensorSymmetric(std::vector<float>{0, value, 1}, 8).ok());
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
