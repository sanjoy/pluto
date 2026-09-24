#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/executor.h"
#include "src/dataset/plain_text_tokenizer.h"
#include "src/llm/experiments/memorize_general_facts/token_corpus.h"

namespace pluto::llm::memorize_general_facts {
namespace {

TEST(TokenCorpusGpuTest, SubstitutesExactRowsWithoutRetokenizing) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  tokenizer::PlainTextTokenizer base;
  auto adapter = TokenCorpusTokenizer::Create(**executor, base, " a\r\nbc\r\n",
                                              "3 4\n5 6\n", 0);
  ASSERT_TRUE(adapter.ok()) << adapter.status();
  EXPECT_EQ((*adapter)->vocab_size(), base.vocab_size());
  auto ids = (*adapter)->Encode(**executor, " a");
  ASSERT_TRUE(ids.ok()) << ids.status();
  EXPECT_EQ((std::vector<int>(ids->begin(), ids->end())),
            (std::vector<int>{3, 4}));
  auto second = (*adapter)->Encode(**executor, "bc");
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ((std::vector<int>(second->begin(), second->end())),
            (std::vector<int>{5, 6}));
  EXPECT_FALSE((*adapter)->Encode(**executor, "a").ok());
  EXPECT_FALSE((*adapter)->Encode(**executor, " b").ok());
}

TEST(TokenCorpusGpuTest, RejectsChangedLengthsAndConflictingDuplicateLines) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  tokenizer::PlainTextTokenizer base;
  EXPECT_FALSE(
      TokenCorpusTokenizer::Create(**executor, base, "abc", "1 2", 0).ok());
  EXPECT_FALSE(
      TokenCorpusTokenizer::Create(**executor, base, "ab\nab", "1 2\n2 1", 0)
          .ok());
  EXPECT_TRUE(
      TokenCorpusTokenizer::Create(**executor, base, "ab\nab", "1 2\n1 2", 0)
          .ok());
  EXPECT_FALSE(
      TokenCorpusTokenizer::Create(**executor, base, "ab\n \n", "1 2\n1", 0)
          .ok());
}

}  // namespace
}  // namespace pluto::llm::memorize_general_facts
