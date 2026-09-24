#include "src/llm/experiments/memorize_general_facts/permutation_trace/trace_util.h"

#include <bit>
#include <cstring>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::permutation_trace {
namespace {

template <typename T>
std::vector<uint8_t> Bytes(std::initializer_list<T> values) {
  std::vector<uint8_t> result(values.size() * sizeof(T));
  std::memcpy(result.data(), values.begin(), result.size());
  return result;
}

TEST(TraceUtilTest, RejectsIncompleteInvalidAndNonEosPreservingPermutations) {
  const auto valid = ParsePermutation("1 0 2\n", 3, 2);
  ASSERT_TRUE(valid.ok()) << valid.status();
  EXPECT_EQ(*valid, (std::vector<int>{1, 0, 2}));
  for (const char* text : {"1 0", "1 1 2", "1 0 3", "1 -1 2", "1 x 2", "2 1 0"})
    EXPECT_FALSE(ParsePermutation(text, 3, 2).ok()) << text;
}

TEST(TraceUtilTest, AlignsRowsColumnsAndIdsWithoutMovingPadding) {
  const std::vector<int> permutation{1, 0, 2};
  auto rows = Compare({"embedding", "fp32", {3, 2}, "vocab_rows"},
                      Bytes<float>({1, 2, 3, 4, 5, 6}),
                      Bytes<float>({3, 4, 1, 2, 5, 6}), permutation);
  ASSERT_TRUE(rows.ok()) << rows.status();
  EXPECT_EQ(rows->mismatches, 0);
  const float inf = -std::numeric_limits<float>::infinity();
  auto columns =
      Compare({"logits", "fp32", {2, 4}, "vocab_columns"},
              Bytes<float>({1, 2, 3, inf, 4, 5, 6, inf}),
              Bytes<float>({2, 1, 3, inf, 5, 4, 6, inf}), permutation);
  ASSERT_TRUE(columns.ok()) << columns.status();
  EXPECT_EQ(columns->mismatches, 0);
  EXPECT_EQ(columns->l2, 0);
  auto tokens = Compare({"targets", "int32", {5}, "token_ids"},
                        Bytes<int32_t>({0, -1, 2, 1, 0}),
                        Bytes<int32_t>({1, -1, 2, 0, 1}), permutation);
  ASSERT_TRUE(tokens.ok()) << tokens.status();
  EXPECT_EQ(tokens->mismatches, 0);
}

TEST(TraceUtilTest, EmptyPermutationChecksLiteralDeterminism) {
  auto result =
      Compare({"embedding", "fp32", {2, 2}, "vocab_rows"},
              Bytes<float>({1, 2, 3, 4}), Bytes<float>({3, 4, 1, 2}), {});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->mismatches, 4);
  EXPECT_EQ(result->max_abs, 2);
  EXPECT_EQ(result->l2, 4);
}

TEST(TraceUtilTest, ReportsBitwiseSignedZeroWithoutNumericalDifference) {
  auto result = Compare({"gradient", "fp32", {2}}, Bytes<float>({0.0f, 1.0f}),
                        Bytes<float>({-0.0f, 1.0f}), {});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->mismatches, 1);
  EXPECT_EQ(result->max_abs, 0);
  EXPECT_EQ(result->l2, 0);
}

TEST(TraceUtilTest, DecodesBf16AndRejectsMalformedDescriptions) {
  auto result =
      Compare({"activation", "bf16", {2}}, Bytes<uint16_t>({0x3f80, 0x4000}),
              Bytes<uint16_t>({0x4000, 0x4040}), {});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->mismatches, 2);
  EXPECT_EQ(result->max_abs, 1);
  EXPECT_NEAR(result->l2, 1.4142135623730951, 1e-15);
  EXPECT_FALSE(Compare({"x", "fp32", {3}}, Bytes<float>({1, 2}),
                       Bytes<float>({1, 2}), {})
                   .ok());
  EXPECT_FALSE(Compare({"x", "bf16", {-2, 2}}, Bytes<uint16_t>({1, 2}),
                       Bytes<uint16_t>({1, 2}), {})
                   .ok());
  EXPECT_FALSE(Compare({"x", "fp32", {2}, "vocab_columns"},
                       Bytes<float>({1, 2}), Bytes<float>({1, 2}), {0, 0})
                   .ok());
  EXPECT_EQ(ShapeText({32, 27, 4480}), "32,27,4480");
}

}  // namespace
}  // namespace pluto::llm::permutation_trace
