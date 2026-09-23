#include "src/llm/experiments/one_shot_memorizer/model_io.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {
TEST(ModelIoTest, RoundTripPredictionsAndCanonicalBytes) {
  auto model = BuildModel({{1, 2, 3}, {4, 2, 3}, {1, 5}}, 8, 7);
  ASSERT_TRUE(model.ok()) << model.status();
  auto bytes = SerializeModel(*model);
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  auto restored = DeserializeModel(*bytes);
  ASSERT_TRUE(restored.ok()) << restored.status();
  EXPECT_EQ(*SerializeModel(*restored), *bytes);
  auto answer = GreedyContinuation(*restored, std::vector<int>{4}, 10);
  ASSERT_TRUE(answer.ok()) << answer.status();
  EXPECT_EQ(*answer, (std::vector<int>{2, 3, 7}));
}

TEST(ModelIoTest, RejectsEveryTruncationAndTrailingGarbage) {
  auto model = BuildModel({{1, 2}}, 8, 7);
  ASSERT_TRUE(model.ok());
  auto bytes = SerializeModel(*model);
  ASSERT_TRUE(bytes.ok());
  for (size_t length = 0; length < bytes->size(); ++length)
    EXPECT_FALSE(DeserializeModel(bytes->substr(0, length)).ok()) << length;
  EXPECT_FALSE(DeserializeModel(*bytes + "x").ok());
  (*bytes)[0] = '!';
  EXPECT_FALSE(DeserializeModel(*bytes).ok());
}

TEST(ModelIoTest, BoundsUntrustedDimensionsBeforeAllocation) {
  auto model = BuildModel({{1}}, 8, 7);
  ASSERT_TRUE(model.ok());
  auto bytes = SerializeModel(*model);
  ASSERT_TRUE(bytes.ok());
  // Header: magic(13), vocabulary(4), EOS(4), initial/trie/sentence counts(24),
  // then state count(8). Replacing it by UINT64_MAX must not allocate it.
  bytes->replace(45, 8, std::string(8, '\xff'));
  EXPECT_FALSE(DeserializeModel(*bytes).ok());
}

TEST(ModelIoTest, RejectsSemanticallyInvalidSerializedWeights) {
  auto model = BuildModel({{1}}, 8, 7);
  ASSERT_TRUE(model.ok());
  auto bytes = SerializeModel(*model);
  ASSERT_TRUE(bytes.ok());
  // The first state's suffix mass is at header(53) + terminal_count(8).
  // It must equal its terminal count because the first state has no edges.
  (*bytes)[61] = '\x02';
  EXPECT_FALSE(DeserializeModel(*bytes).ok());
}
}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
