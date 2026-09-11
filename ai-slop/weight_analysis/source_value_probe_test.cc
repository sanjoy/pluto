#include "ai-slop/weight_analysis/source_value_probe.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include "gtest/gtest.h"
#include "src/llm/recipes/gpt2.h"

namespace pluto::weight_analysis {
namespace {
constexpr int kContext = llm::kGpt2ContextLength;

TEST(SourceValueValidationTest, AcceptsBoundarySitesAndFutureNullControl) {
  EXPECT_TRUE(ValidateSourceValueSelection({0, 0, 0, 0, 0, 0}, kContext).ok());
  EXPECT_TRUE(
      ValidateSourceValueSelection({7, 7, 1, 1023, 1023, 0.5f}, 2 * kContext)
          .ok());
  EXPECT_TRUE(
      ValidateSourceValueSelection({1, 2, 0, 0, 1023, 1}, kContext).ok());
  const int max_rows = (std::numeric_limits<int>::max() /
                        llm::kGpt2FeedForwardWidth / kContext) *
                       kContext;
  EXPECT_TRUE(ValidateSourceValueSelection(
                  {7, 7, max_rows / kContext - 1, 1023, 0, 0.5f}, max_rows)
                  .ok());
  EXPECT_FALSE(
      ValidateSourceValueSelection({0, 0, 0, 0, 0, 1}, max_rows + kContext)
          .ok());
}

TEST(SourceValueValidationTest, RejectsEachInvalidCoordinateAndShape) {
  for (int block : {-1, 8}) {
    EXPECT_FALSE(
        ValidateSourceValueSelection({block, 0, 0, 0, 0, 1}, kContext).ok());
  }
  for (int head : {-1, 8}) {
    EXPECT_FALSE(
        ValidateSourceValueSelection({0, head, 0, 0, 0, 1}, kContext).ok());
  }
  for (int sequence : {-1, 2}) {
    EXPECT_FALSE(
        ValidateSourceValueSelection({0, 0, sequence, 0, 0, 1}, 2 * kContext)
            .ok());
  }
  for (int position : {-1, kContext}) {
    EXPECT_FALSE(
        ValidateSourceValueSelection({0, 0, 0, position, 0, 1}, kContext).ok());
    EXPECT_FALSE(
        ValidateSourceValueSelection({0, 0, 0, 0, position, 1}, kContext).ok());
  }
  for (int rows :
       {-1024, -1, 0, 1, 1023, 1025, std::numeric_limits<int>::max()}) {
    EXPECT_FALSE(ValidateSourceValueSelection({0, 0, 0, 0, 0, 1}, rows).ok());
  }
}

TEST(SourceValueValidationTest, RejectsUnspecifiedAndNonfiniteDoses) {
  for (float scale :
       {-1.0f, 0.25f, 0.75f, 2.0f, std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()}) {
    EXPECT_FALSE(
        ValidateSourceValueSelection({0, 0, 0, 0, 0, scale}, kContext).ok());
    EXPECT_FALSE(ScaleSourceBf16(0x3f80, scale).ok());
  }
}

TEST(SourceBf16Test, SignedZerosFiniteExtremesAndSubnormalTies) {
  for (uint16_t bits : {0x0000, 0x8000, 0x0001, 0x8001, 0x7f7f, 0xff7f}) {
    ASSERT_TRUE(ScaleSourceBf16(bits, 1).ok());
    EXPECT_EQ(*ScaleSourceBf16(bits, 1), bits);
    EXPECT_EQ(*ScaleSourceBf16(bits, 0), 0);
  }
  EXPECT_EQ(*ScaleSourceBf16(0x8000, 0.5f), 0x8000);
  EXPECT_EQ(*ScaleSourceBf16(0x0001, 0.5f), 0x0000);
  EXPECT_EQ(*ScaleSourceBf16(0x8001, 0.5f), 0x8000);
  EXPECT_EQ(*ScaleSourceBf16(0x0003, 0.5f), 0x0002);
  EXPECT_EQ(*ScaleSourceBf16(0x0005, 0.5f), 0x0002);
  EXPECT_EQ(*ScaleSourceBf16(0x007f, 0.5f), 0x0040);
  EXPECT_EQ(*ScaleSourceBf16(0x0080, 0.5f), 0x0040);
  EXPECT_EQ(*ScaleSourceBf16(0x00ff, 0.5f), 0x0080);
  EXPECT_EQ(*ScaleSourceBf16(0x0100, 0.5f), 0x0080);
  EXPECT_EQ(*ScaleSourceBf16(0x3f80, 0.5f), 0x3f00);
  EXPECT_EQ(*ScaleSourceBf16(0xbf80, 0.5f), 0xbf00);
  EXPECT_EQ(*ScaleSourceBf16(0x7f7f, 0.5f), 0x7eff);
}

// Decode with double arithmetic and powers of two, not the implementation's
// integer shift/round rule. Every BF16 value, and half of it, is exact here.
double Magnitude(uint16_t bits) {
  const int exponent = (bits >> 7) & 0xff;
  const int significand = bits & 0x7f;
  if (exponent == 0) return std::ldexp(static_cast<double>(significand), -133);
  return std::ldexp(static_cast<double>(128 + significand), exponent - 134);
}

TEST(SourceBf16Test, ExhaustiveFiniteHalfIsNearestWithEvenTies) {
  for (uint32_t raw = 0; raw <= 0xffff; ++raw) {
    const uint16_t bits = raw;
    if ((bits & 0x7f80) == 0x7f80) continue;
    ASSERT_EQ(*ScaleSourceBf16(bits, 1), bits);
    ASSERT_EQ(*ScaleSourceBf16(bits, 0), 0);
    const auto half = ScaleSourceBf16(bits, 0.5f);
    ASSERT_TRUE(half.ok());
    ASSERT_EQ(*half & 0x8000, bits & 0x8000);
    const uint16_t result = *half & 0x7fff;
    const double exact = Magnitude(bits) * 0.5;
    const double distance = std::abs(Magnitude(result) - exact);
    for (int adjacent :
         {static_cast<int>(result) - 1, static_cast<int>(result) + 1}) {
      if (adjacent < 0 || adjacent >= 0x7f80) continue;
      const double alternative = std::abs(Magnitude(adjacent) - exact);
      ASSERT_LE(distance, alternative) << "BF16 bits=" << bits;
      if (distance == alternative) {
        ASSERT_EQ(result & 1, 0) << "BF16 tie bits=" << bits;
      }
    }
  }
}

TEST(SourceBf16Test, RejectsEveryInfinityAndNanEncodingEvenAtZeroDose) {
  for (uint32_t raw = 0; raw <= 0xffff; ++raw) {
    if ((raw & 0x7f80) != 0x7f80) continue;
    for (float scale : {0.0f, 0.5f, 1.0f}) {
      EXPECT_FALSE(ScaleSourceBf16(raw, scale).ok());
    }
  }
}

}  // namespace
}  // namespace pluto::weight_analysis
