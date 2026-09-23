#include "src/llm/experiments/one_shot_memorizer/projected_relu_memory.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {
constexpr int kVocabulary = 64;
constexpr int kEos = 63;

const std::vector<std::vector<int>> kSentences{
    {1, 2, 3, 4, 5, 0, 6, 0, 7, 8, 9}, {4, 5, 6, 7, 8, 3, 3, 1, 12, 4}};

void Set64(std::string& bytes, size_t offset, uint64_t value) {
  for (int i = 0; i < 8; ++i)
    bytes[offset + i] = static_cast<char>((value >> (8 * i)) & 255);
}

TEST(ProjectedReluMemoryTest, CompletesSentencesWithRepeatsZeroAndEos) {
  auto model = BuildProjectedReluMemory(kSentences, kVocabulary, kEos);
  ASSERT_TRUE(model.ok()) << model.status();
  for (const auto& sentence : kSentences) {
    const auto prefix = absl::MakeConstSpan(sentence).first(5);
    std::vector<int> expected(sentence.begin() + 5, sentence.end());
    expected.push_back(kEos);
    auto output =
        ProjectedReluGreedyContinuation(*model, prefix, expected.size() + 1);
    ASSERT_TRUE(output.ok()) << output.status();
    EXPECT_EQ(*output, expected);
  }
  auto zero = ProjectedReluNextToken(*model, {kSentences[0].data(), 5});
  ASSERT_TRUE(zero.ok()) << zero.status();
  EXPECT_EQ(*zero, 0);  // A zero token is not the unsupported-query sentinel.
}

TEST(ProjectedReluMemoryTest, DeterministicAcrossCorpusOrderAndDuplicates) {
  auto a = BuildProjectedReluMemory(kSentences, kVocabulary, kEos);
  auto reversed = kSentences;
  std::reverse(reversed.begin(), reversed.end());
  reversed.push_back(kSentences[0]);
  auto b = BuildProjectedReluMemory(reversed, kVocabulary, kEos);
  ASSERT_TRUE(a.ok()) << a.status();
  ASSERT_TRUE(b.ok()) << b.status();
  auto first = SerializeProjectedReluMemory(*a);
  auto second = SerializeProjectedReluMemory(*b);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(*first, *second);
  EXPECT_EQ(a->projection_attempts, 1u);
  EXPECT_EQ(a->projection,
            (std::array<double, 9>{904624, 615925, 607568, 819693, 554140,
                                   959211, 340706, 437053, 625348}));
}

TEST(ProjectedReluMemoryTest, RejectsSameTargetCorpusHashCollisionAndRetries) {
  // These distinct five-token keys collide under the FIRST seed-0 projection.
  // Both request EOS, yet the protocol still requires distinct unit hashes.
  const std::vector<int> a{61, 157, 0, 0, 1};
  const std::vector<int> b{0, 0, 14, 411, 1};
  auto model = BuildProjectedReluMemory({a, b}, 1024, 1023);
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_EQ(model->units.size(), 2u);
  EXPECT_GT(model->projection_attempts, 1u);
  EXPECT_NE(model->units[0].positive_bias, model->units[1].positive_bias);
  for (const auto& prefix : {a, b}) {
    auto output = ProjectedReluNextToken(*model, prefix);
    ASSERT_TRUE(output.ok()) << output.status();
    EXPECT_EQ(*output, 1023);
  }
}

TEST(ProjectedReluMemoryTest, UnseenContextCanAliasAStoredHash) {
  const std::vector<int> trained{61, 157, 0, 0, 1};
  const std::vector<int> unseen{0, 0, 14, 411, 1};
  ASSERT_NE(trained, unseen);
  auto model = BuildProjectedReluMemory({trained}, 1024, 1023);
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_EQ(model->projection_attempts, 1u);
  // Deliberate documented limitation: no full keys remain to distinguish them.
  auto alias = ProjectedReluNextToken(*model, unseen);
  ASSERT_TRUE(alias.ok()) << alias.status();
  EXPECT_EQ(*alias, 1023);
  auto absent = unseen;
  ++absent.back();
  EXPECT_EQ(ProjectedReluNextToken(*model, absent).status().code(),
            absl::StatusCode::kNotFound);
}

TEST(ProjectedReluMemoryTest, ExactAtMaximumVocabularyBound) {
  const std::vector<int> prefix(9, 65535);
  auto model = BuildProjectedReluMemory({prefix}, 65536, 65534);
  ASSERT_TRUE(model.ok()) << model.status();
  double hash = 0;
  for (double coefficient : model->projection)
    hash += coefficient * 65535;
  model->units = {{-hash, hash, 0}};
  ASSERT_TRUE(ValidateProjectedReluMemory(*model).ok());
  auto output = ProjectedReluNextToken(*model, prefix);
  ASSERT_TRUE(output.ok()) << output.status();
  EXPECT_EQ(*output, 0);
  auto near = prefix;
  near.back() -= 2;  // Avoid the reserved EOS ID 65534.
  EXPECT_EQ(ProjectedReluNextToken(*model, near).status().code(),
            absl::StatusCode::kNotFound);
}

TEST(ProjectedReluMemoryTest, LeftPaddingAndTruncationArePartOfTheKey) {
  const std::vector<int> long_sentence{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  auto model = BuildProjectedReluMemory({long_sentence}, 64, 63);
  ASSERT_TRUE(model.ok()) << model.status();
  const std::vector<int> changed_old{44, 45, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  auto next = ProjectedReluNextToken(*model, changed_old);
  ASSERT_TRUE(next.ok()) << next.status();
  EXPECT_EQ(*next, 12);
  auto short_next = ProjectedReluNextToken(*model, {long_sentence.data(), 5});
  ASSERT_TRUE(short_next.ok()) << short_next.status();
  EXPECT_EQ(*short_next, 6);
}

TEST(ProjectedReluMemoryTest, RejectsConflictingTargetsAndInvalidCorpora) {
  EXPECT_FALSE(BuildProjectedReluMemory({}, 64, 63).ok());
  EXPECT_FALSE(BuildProjectedReluMemory({{1, 2, 3, 4}}, 64, 63).ok());
  EXPECT_FALSE(BuildProjectedReluMemory({{1, 2, 3, 4, 63}}, 64, 63).ok());
  EXPECT_FALSE(BuildProjectedReluMemory({{1, 2, 3, 4, -1}}, 64, 63).ok());
  EXPECT_FALSE(BuildProjectedReluMemory({{1, 2, 3, 4, 64}}, 64, 63).ok());
  EXPECT_FALSE(BuildProjectedReluMemory(kSentences, 0, 0).ok());
  EXPECT_FALSE(BuildProjectedReluMemory(kSentences, 65537, 0).ok());
  EXPECT_FALSE(BuildProjectedReluMemory(kSentences, 64, -1).ok());
  EXPECT_FALSE(BuildProjectedReluMemory(kSentences, 64, 64).ok());
  EXPECT_FALSE(
      BuildProjectedReluMemory({{1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 7}}, 64, 63)
          .ok());
}

TEST(ProjectedReluMemoryTest, RejectsInvalidPrefixesAndHandlesGenerationCap) {
  auto model = BuildProjectedReluMemory(kSentences, 64, 63);
  ASSERT_TRUE(model.ok()) << model.status();
  for (const auto& prefix : std::vector<std::vector<int>>{{},
                                                          {1, 2, 3, 4},
                                                          {1, 2, 3, 4, -1},
                                                          {1, 2, 3, 4, 63},
                                                          {1, 2, 3, 4, 64}})
    EXPECT_EQ(ProjectedReluNextToken(*model, prefix).status().code(),
              absl::StatusCode::kInvalidArgument);
  const auto prefix = absl::MakeConstSpan(kSentences[0]).first(5);
  EXPECT_FALSE(ProjectedReluGreedyContinuation(*model, prefix, 0).ok());
  auto partial = ProjectedReluGreedyContinuation(*model, prefix, 2);
  ASSERT_TRUE(partial.ok()) << partial.status();
  EXPECT_EQ(*partial, (std::vector<int>{0, 6}));
}

TEST(ProjectedReluMemoryTest, RejectsMalformedNumericWeights) {
  auto source = BuildProjectedReluMemory(kSentences, 64, 63);
  ASSERT_TRUE(source.ok()) << source.status();
  for (double invalid : {0.0, -1.0, 1.5, double((1 << 20) + 1),
                         std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    auto model = *source;
    model.projection[0] = invalid;
    EXPECT_FALSE(ValidateProjectedReluMemory(model).ok());
  }
  for (uint32_t attempts : {0u, 257u}) {
    auto model = *source;
    model.projection_attempts = attempts;
    EXPECT_FALSE(ValidateProjectedReluMemory(model).ok());
  }
  auto wrong_draw = *source;
  wrong_draw.projection[0] += 1;
  EXPECT_FALSE(ValidateProjectedReluMemory(wrong_draw).ok());
  wrong_draw = *source;
  ++wrong_draw.projection_attempts;
  EXPECT_FALSE(ValidateProjectedReluMemory(wrong_draw).ok());
  for (int kind = 0; kind < 6; ++kind) {
    auto model = *source;
    if (kind == 0)
      model.units.clear();
    if (kind == 1)
      model.units[0].negative_bias += 1;
    if (kind == 2)
      model.units[0].output_weight = 64;
    if (kind == 3)
      model.units[0].output_weight = .5;
    if (kind == 4)
      model.units[1] = model.units[0];
    if (kind == 5)
      std::swap(model.units[0], model.units[1]);
    EXPECT_FALSE(ValidateProjectedReluMemory(model).ok());
    EXPECT_FALSE(SerializeProjectedReluMemory(model).ok());
  }
}

TEST(ProjectedReluMemoryTest, ReloadsExactWeightsAndRejectsCorruptArtifacts) {
  auto model = BuildProjectedReluMemory(kSentences, 64, 63);
  ASSERT_TRUE(model.ok()) << model.status();
  auto bytes = SerializeProjectedReluMemory(*model);
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  auto loaded = DeserializeProjectedReluMemory(*bytes);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  auto again = SerializeProjectedReluMemory(*loaded);
  ASSERT_TRUE(again.ok()) << again.status();
  EXPECT_EQ(*again, *bytes);
  for (size_t cut = 0; cut < bytes->size(); ++cut)
    EXPECT_FALSE(DeserializeProjectedReluMemory(bytes->substr(0, cut)).ok());
  EXPECT_FALSE(DeserializeProjectedReluMemory(*bytes + 'x').ok());
  const size_t header = std::string("PLUTO_PROJECTED_RELU_V1\n").size();
  for (int kind = 0; kind < 6; ++kind) {
    std::string malformed = *bytes;
    if (kind == 0)
      malformed[0] = 'x';
    if (kind == 1)
      malformed[header + 12] = 8;  // Wrong window.
    if (kind == 2)
      malformed[header + 16] = 1;  // Wrong seed.
    if (kind == 3)
      Set64(malformed, header + 28, UINT64_MAX);
    if (kind == 4)
      Set64(malformed, header + 36, std::bit_cast<uint64_t>(.5));
    if (kind == 5)
      Set64(malformed, header + 36 + 9 * 8 + 16,
            std::bit_cast<uint64_t>(std::numeric_limits<double>::quiet_NaN()));
    EXPECT_FALSE(DeserializeProjectedReluMemory(malformed).ok());
  }
  auto completion = ProjectedReluGreedyContinuation(
      *loaded, {kSentences[0].data(), 5}, kSentences[0].size());
  ASSERT_TRUE(completion.ok()) << completion.status();
  std::vector<int> expected(kSentences[0].begin() + 5, kSentences[0].end());
  expected.push_back(63);
  EXPECT_EQ(*completion, expected);
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
