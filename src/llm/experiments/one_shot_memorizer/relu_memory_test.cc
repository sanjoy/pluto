#include "src/llm/experiments/one_shot_memorizer/relu_memory.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

ReluMemoryCode CodeForTest(int token) {
  ReluMemoryCode code;
  for (size_t bit = 0; bit < code.size(); ++bit)
    code[bit] = (static_cast<unsigned>(token) & (1u << bit)) ? 1.0f : -1.0f;
  return code;
}

float Dot(const ReluMemoryCode& a, const ReluMemoryCode& b) {
  float sum = 0.0f;
  for (size_t bit = 0; bit < a.size(); ++bit)
    sum += a[bit] * b[bit];
  return sum;
}

TEST(ReluMemoryTest, ConstructsFloatWeightsAndRecallsEntireSuffixIncludingEos) {
  const std::vector<std::vector<int>> sentences = {{1, 2, 3, 4, 5, 6, 7},
                                                   {8, 2, 3, 4, 9, 10, 7}};
  auto model = BuildReluMemory(sentences, 20, 19, 5, 5);
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_TRUE(ValidateReluMemory(*model).ok());
  ASSERT_EQ(model->units.size(), 6);
  for (const auto& sentence : sentences) {
    auto generated = ReluMemoryGreedyContinuation(
        *model, absl::Span<const int>(sentence).first(5), 10);
    ASSERT_TRUE(generated.ok()) << generated.status();
    std::vector<int> expected(sentence.begin() + 5, sentence.end());
    expected.push_back(19);
    EXPECT_EQ(*generated, expected);
  }
  auto first =
      ReluMemoryGreedyContinuation(*model, std::vector<int>{1, 2, 3, 4, 5}, 1);
  ASSERT_TRUE(first.ok());
  EXPECT_EQ(*first, (std::vector<int>{6}));
  EXPECT_FALSE(
      ReluMemoryGreedyContinuation(*model, std::vector<int>{1, 2, 3, 4, 5}, 0)
          .ok());
}

TEST(ReluMemoryTest, RefusesConflictsUntilWindowDistinguishesTheirContexts) {
  const std::vector<std::vector<int>> sentences = {{1, 2, 3, 4}, {6, 2, 3, 5}};
  auto conflict = BuildReluMemory(sentences, 8, 7, 1, 2);
  EXPECT_EQ(conflict.status().code(), absl::StatusCode::kInvalidArgument);
  auto model = BuildReluMemory(sentences, 8, 7, 1, 3);
  ASSERT_TRUE(model.ok()) << model.status();
  auto a = ReluMemoryGreedyContinuation(*model, std::vector<int>{1}, 10);
  auto b = ReluMemoryGreedyContinuation(*model, std::vector<int>{6}, 10);
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());
  EXPECT_EQ(*a, (std::vector<int>{2, 3, 4, 7}));
  EXPECT_EQ(*b, (std::vector<int>{2, 3, 5, 7}));
  // EOS is a target and must also participate in conflict detection.
  EXPECT_FALSE(BuildReluMemory({{1, 2}, {1, 2, 3}}, 8, 7, 1, 8).ok());
}

TEST(ReluMemoryTest, DeduplicatesKeysAndProducesCanonicalNumericWeights) {
  auto a = BuildReluMemory({{1, 3, 4}, {2, 3, 4}}, 8, 7, 2, 2);
  auto b = BuildReluMemory({{2, 3, 4}, {1, 3, 4}, {2, 3, 4}}, 8, 7, 2, 2);
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());
  ASSERT_EQ(a->units.size(), 3);
  ASSERT_EQ(a->units.size(), b->units.size());
  for (size_t i = 0; i < a->units.size(); ++i) {
    EXPECT_EQ(a->units[i].input_biases, b->units[i].input_biases);
    EXPECT_EQ(a->units[i].output_code, b->units[i].output_code);
  }
}

TEST(ReluMemoryTest, TruncatesOldContextButRejectsInvalidOlderTokens) {
  auto model = BuildReluMemory({{1, 3, 4}}, 10, 9, 2, 2);
  ASSERT_TRUE(model.ok());
  auto token = ReluMemoryNextToken(*model, std::vector<int>{7, 8, 1, 3});
  ASSERT_TRUE(token.ok()) << token.status();
  EXPECT_EQ(*token, 4);
  EXPECT_EQ(
      ReluMemoryNextToken(*model, std::vector<int>{7, 8, 2, 3}).status().code(),
      absl::StatusCode::kNotFound);
  EXPECT_EQ(ReluMemoryNextToken(*model, std::vector<int>{-1, 8, 1, 3})
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_FALSE(ReluMemoryNextToken(*model, std::vector<int>{9, 8, 1, 3}).ok());
  EXPECT_FALSE(ReluMemoryNextToken(*model, std::vector<int>{10, 8, 1, 3}).ok());
  EXPECT_FALSE(ReluMemoryNextToken(*model, std::vector<int>{3}).ok());
}

TEST(ReluMemoryTest, PaddingCannotAliasRealTokens) {
  auto model = BuildReluMemory({{0, 1}}, 4, 3, 1, 3);
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ(model->units[0].input_biases,
            (std::vector<float>{1, -1, 1, -1, 0, 0}));
  auto generated = ReluMemoryGreedyContinuation(*model, std::vector<int>{0}, 4);
  ASSERT_TRUE(generated.ok());
  EXPECT_EQ(*generated, (std::vector<int>{1, 3}));
  EXPECT_FALSE(ReluMemoryNextToken(*model, std::vector<int>{-1}).ok());
  auto terminal = BuildReluMemory({{0}}, 2, 1, 1, 1);
  ASSERT_TRUE(terminal.ok());
  auto eos = ReluMemoryNextToken(*terminal, std::vector<int>{0});
  ASSERT_TRUE(eos.ok());
  EXPECT_EQ(*eos, 1);
}

TEST(ReluMemoryTest, Float32ArithmeticHasExactUnitSupportAtLargestTokenIds) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::digits == 24);
  auto model = BuildReluMemory({{65533, 65534}}, 65536, 65535, 1, 1);
  ASSERT_TRUE(model.ok());
  auto output = EvaluateReluMemory(*model, std::vector<int>{65533});
  ASSERT_TRUE(output.ok()) << output.status();
  EXPECT_EQ(*output, CodeForTest(65534));
  EXPECT_FLOAT_EQ(Dot(*output, CodeForTest(65534)), 16.0f);
  EXPECT_FLOAT_EQ(Dot(*output, CodeForTest(65535)), 14.0f);
  // An unseen query one integer away from a key lies exactly at tent support's
  // boundary, so no unit is active and the model must reject it.
  EXPECT_EQ(EvaluateReluMemory(*model, std::vector<int>{65532}).status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(EvaluateReluMemory(*model, std::vector<int>{0}).status().code(),
            absl::StatusCode::kNotFound);
}

TEST(ReluMemoryTest, BitDecoderMatchesExhaustiveDotProductsAndBreaksTies) {
  auto model = BuildReluMemory({{1}}, 13, 12, 1, 1);
  ASSERT_TRUE(model.ok());
  for (int token = 0; token < model->vocabulary_size; ++token) {
    const auto code = CodeForTest(token);
    auto decoded = DecodeReluMemoryOutput(*model, code);
    ASSERT_TRUE(decoded.ok());
    float best = -std::numeric_limits<float>::infinity();
    int argmax = -1;
    for (int candidate = 0; candidate < model->vocabulary_size; ++candidate) {
      const float score = Dot(code, CodeForTest(candidate));
      if (score > best) {
        best = score;
        argmax = candidate;
      }
    }
    EXPECT_EQ(*decoded, argmax);
    EXPECT_EQ(*decoded, token);
  }
  ReluMemoryCode zero = {};
  auto tied = DecodeReluMemoryOutput(*model, zero);
  ASSERT_TRUE(tied.ok());
  EXPECT_EQ(*tied, 0);
  zero[0] = 0.5f;  // Every odd ID ties; the lowest is 1.
  auto odd = DecodeReluMemoryOutput(*model, zero);
  ASSERT_TRUE(odd.ok());
  EXPECT_EQ(*odd, 1);
  EXPECT_FALSE(DecodeReluMemoryOutput(*model, CodeForTest(13)).ok());
  zero[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(DecodeReluMemoryOutput(*model, zero).ok());
  zero.fill(std::numeric_limits<float>::max());
  EXPECT_FALSE(DecodeReluMemoryOutput(*model, zero).ok());
}

TEST(ReluMemoryTest, CountsStoredAndGeneratedWeightsSeparately) {
  auto model = BuildReluMemory({{1, 3, 4}, {2, 3, 4}}, 8, 7, 2, 2);
  ASSERT_TRUE(model.ok());
  auto counts = GetReluMemoryStorageCounts(*model);
  ASSERT_TRUE(counts.ok());
  EXPECT_EQ(counts->first_hidden_units, 12);
  EXPECT_EQ(counts->second_hidden_units, 3);
  EXPECT_EQ(counts->data_dependent_float_count, 60);
  EXPECT_EQ(counts->data_dependent_bytes, 240);
  EXPECT_EQ(counts->fixed_sparse_weight_count, 24);
  EXPECT_EQ(counts->fixed_bias_count, 3);
  EXPECT_EQ(counts->fixed_decoder_weight_count, 128);
}

TEST(ReluMemoryTest, RejectsInvalidCorpusMetadataAndTokens) {
  EXPECT_FALSE(BuildReluMemory({}, 8, 7, 1, 1).ok());
  EXPECT_FALSE(BuildReluMemory({{1}}, 0, 0, 1, 1).ok());
  EXPECT_FALSE(BuildReluMemory({{1}}, 65537, 7, 1, 1).ok());
  EXPECT_FALSE(BuildReluMemory({{1}}, 8, 8, 1, 1).ok());
  EXPECT_FALSE(BuildReluMemory({{1}}, 8, -1, 1, 1).ok());
  EXPECT_FALSE(BuildReluMemory({{1}}, 8, 7, 1, 0).ok());
  EXPECT_FALSE(BuildReluMemory({{1}}, 8, 7, 0, 1).ok());
  EXPECT_FALSE(BuildReluMemory({{1}}, 8, 7, 2, 1).ok());
  EXPECT_FALSE(BuildReluMemory({{-1}}, 8, 7, 1, 1).ok());
  EXPECT_FALSE(BuildReluMemory({{8}}, 8, 7, 1, 1).ok());
  EXPECT_FALSE(BuildReluMemory({{7}}, 8, 7, 1, 1).ok());
  EXPECT_FALSE(
      BuildReluMemory({{1}}, 8, 7, 1, std::numeric_limits<size_t>::max()).ok());
}

TEST(ReluMemoryTest, ValidatesNumericArtifactsBeforeInference) {
  auto built = BuildReluMemory({{1, 2}}, 8, 7, 1, 3);
  ASSERT_TRUE(built.ok());
  ReluMemory model = *built;
  model.units[0].input_biases.pop_back();
  EXPECT_FALSE(ValidateReluMemory(model).ok());
  model = *built;
  model.units[0].input_biases[0] = 0;
  EXPECT_FALSE(ValidateReluMemory(model).ok());
  model = *built;
  model.units[0].input_biases[4] = -1.5f;
  model.units[0].input_biases[5] = 1.5f;
  EXPECT_FALSE(ValidateReluMemory(model).ok());
  model = *built;
  model.units[0].input_biases[1] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(ValidateReluMemory(model).ok());
  model = *built;
  model.units[0].input_biases[0] = -1;
  model.units[0].input_biases[1] = 1;  // Real token before remaining padding.
  EXPECT_FALSE(ValidateReluMemory(model).ok());
  model = *built;
  model.units[0].output_code[0] = 0;
  EXPECT_FALSE(ValidateReluMemory(model).ok());
  model = *built;
  model.units[0].output_code = CodeForTest(8);
  EXPECT_FALSE(ValidateReluMemory(model).ok());
  model = *built;
  std::reverse(model.units.begin(), model.units.end());
  EXPECT_FALSE(ValidateReluMemory(model).ok());
  model = *built;
  model.units.push_back(model.units.back());
  EXPECT_FALSE(ValidateReluMemory(model).ok());
  EXPECT_FALSE(EvaluateReluMemory(model, std::vector<int>{1}).ok());
  model.units.clear();
  EXPECT_FALSE(GetReluMemoryStorageCounts(model).ok());
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
