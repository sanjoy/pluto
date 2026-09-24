#include "src/llm/experiments/memorize_general_facts/token_corpus.h"

#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace pluto::llm::memorize_general_facts {
namespace {

TEST(TokenCorpusTest, PreservesExactIdsAndAcceptsWhitespaceAndCrLf) {
  const auto rows = ParseTokenCorpus(" 3\t0 1\r\n2 4\r\n", {3, 2}, 6, 5);
  ASSERT_TRUE(rows.ok()) << rows.status();
  EXPECT_EQ(*rows, (std::vector<std::vector<int>>{{3, 0, 1}, {2, 4}}));
  EXPECT_TRUE(ParseTokenCorpus("1", {1}, 6, 5).ok());
}

TEST(TokenCorpusTest, RejectsWrongRowsLengthsAndBlankRows) {
  for (const auto* text : {"", "1 2", "1\n2\n3", "1\n\n", "1 2\n3"}) {
    SCOPED_TRACE(text);
    EXPECT_EQ(ParseTokenCorpus(text, {1, 1}, 6, 5).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(TokenCorpusTest, RejectsOutOfRangeEosMalformedAndOverflowIds) {
  for (const auto* text : {"-1", "6", "5", "word", "1.0", "0x1", "1,2",
                           "2147483648", "9999999999999999999999999"}) {
    SCOPED_TRACE(text);
    EXPECT_EQ(ParseTokenCorpus(text, {1}, 6, 5).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(TokenCorpusTest, RejectsInvalidVocabularyAndEos) {
  EXPECT_FALSE(ParseTokenCorpus("1", {1}, 0, 0).ok());
  EXPECT_FALSE(ParseTokenCorpus("1", {1}, 6, -1).ok());
  EXPECT_FALSE(ParseTokenCorpus("1", {1}, 6, 6).ok());
}

}  // namespace
}  // namespace pluto::llm::memorize_general_facts
