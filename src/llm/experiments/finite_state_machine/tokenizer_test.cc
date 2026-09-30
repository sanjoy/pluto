#include "src/llm/experiments/finite_state_machine/tokenizer.h"

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace pluto::llm::fsm {
namespace {

class FsmTokenizerTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }

  void TearDown() override {
    if (executor_ != nullptr)
      EXPECT_TRUE(executor_->Synchronize().ok());
  }

  void ExpectEncoding(absl::string_view text,
                      const std::vector<int>& expected) {
    auto encoded = tokenizer_.Encode(*executor_, text);
    ASSERT_TRUE(encoded.ok()) << encoded.status();
    ASSERT_EQ(encoded->size(), expected.size());
    for (size_t index = 0; index < expected.size(); ++index)
      EXPECT_EQ((*encoded)[index], expected[index]) << "token index " << index;
    auto decoded = tokenizer_.Decode(encoded->span());
    ASSERT_TRUE(decoded.ok()) << decoded.status();
    EXPECT_EQ(*decoded, text);
  }

  std::unique_ptr<cuda::Executor> executor_;
  FsmTokenizer tokenizer_;
};

TEST(FsmTokenizerDecodeTest, ExposesTheExactVocabularyThroughBothInterfaces) {
  const FsmTokenizer tokenizer;
  const tokenizer::Tokenizer& encoder = tokenizer;
  const tokenizer::Detokenizer& decoder = tokenizer;
  EXPECT_EQ(encoder.vocab_size(), 1029);
  EXPECT_EQ(decoder.vocab_size(), 1029);
  EXPECT_EQ(kStateCount, 1000);
  EXPECT_EQ(kLetterOffset, 1000);
  EXPECT_EQ(kSemicolonToken, 1026);
  EXPECT_EQ(kOutputSeparatorToken, 1027);
  EXPECT_EQ(kErrorToken, 1028);
}

TEST_F(FsmTokenizerTest, EncodesTheRequestedSuccessExampleWithoutExtraTokens) {
  ExpectEncoding("000X999;X>999", {0, 1023, 999, 1026, 1023, 1027, 999});
}

TEST_F(FsmTokenizerTest, EncodesTheRequestedErrorExampleWithoutExtraTokens) {
  ExpectEncoding("000X999;Y>ERR", {0, 1023, 999, 1026, 1024, 1027, 1028});
}

TEST_F(FsmTokenizerTest, MapsEveryStateToItsNumberAndPreservesLeadingZeroes) {
  std::string text;
  std::vector<int> expected;
  for (int state = 0; state < 1000; ++state) {
    if (state != 0) {
      text += ';';
      expected.push_back(1026);
    }
    const std::string digits = std::to_string(state);
    text.append(3 - digits.size(), '0');
    text += digits;
    expected.push_back(state);
  }
  ExpectEncoding(text, expected);
  ExpectEncoding("000007042999", {0, 7, 42, 999});
}

TEST_F(FsmTokenizerTest, MapsTheAlphabetAndBothDelimiters) {
  std::vector<int> expected;
  for (int letter = 0; letter < 26; ++letter)
    expected.push_back(1000 + letter);
  expected.push_back(1026);
  expected.push_back(1027);
  ExpectEncoding("ABCDEFGHIJKLMNOPQRSTUVWXYZ;>", expected);
}

TEST_F(FsmTokenizerTest, DistinguishesInputLettersErrFromTheErrorOutput) {
  ExpectEncoding("ERR", {1028});
  ExpectEncoding(";ERR>ERR", {1026, 1004, 1017, 1017, 1027, 1028});
  ExpectEncoding("000E001;001R002;002R003;ERR>003",
                 {0, 1004, 1, 1026, 1, 1017, 2, 1026, 2, 1017, 3, 1026, 1004,
                  1017, 1017, 1027, 3});
  ExpectEncoding("ERRY", {1004, 1017, 1017, 1024});
  ExpectEncoding(";AERR>ERR", {1026, 1000, 1004, 1017, 1017, 1027, 1028});
}

TEST_F(FsmTokenizerTest, RoundTripsSentencesAndCompleteTokenPrefixes) {
  for (absl::string_view text :
       {"", "000", "000X", "000X999;", "000X999;X", "000X999;X>",
        "000A001;001B002;AB>002", "000A001;001B002;ABC>ERR"}) {
    SCOPED_TRACE(text);
    auto encoded = tokenizer_.Encode(*executor_, text);
    ASSERT_TRUE(encoded.ok()) << encoded.status();
    auto decoded = tokenizer_.Decode(encoded->span());
    ASSERT_TRUE(decoded.ok()) << decoded.status();
    EXPECT_EQ(*decoded, text);
  }
  ExpectEncoding("000X999;Y>", {0, 1023, 999, 1026, 1024, 1027});
  ExpectEncoding("", {});
}

TEST_F(FsmTokenizerTest, RoundTripsAllVocabularyIdsTogether) {
  std::vector<int> ids;
  for (int token = 0; token < 1029; ++token)
    ids.push_back(token);
  auto decoded = tokenizer_.Decode(ids);
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  ExpectEncoding(*decoded, ids);
}

TEST_F(FsmTokenizerTest,
       RejectsIncompleteNumericTokensAndUnsupportedCharacters) {
  const std::vector<std::string> invalid{
      "0",           "00",          "0000",
      "00000",       "000A01",      "000A001;A>01",
      "000 A001",    "000A001\n",   "000A001\r",
      "000A001\t",   "000a001",     "000A-01",
      "000A001,",    "000A001:ERR", std::string("000\0A001", 8),
      "000A\xc3\xa9"};
  for (const std::string& text : invalid) {
    SCOPED_TRACE(text);
    auto encoded = tokenizer_.Encode(*executor_, text);
    ASSERT_FALSE(encoded.ok());
    EXPECT_EQ(encoded.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(encoded.status().message().find("at byte "), std::string::npos);
  }
}

TEST_F(FsmTokenizerTest, ReportsTheBadOffsetAndEscapesItsSnippet) {
  const auto partial = tokenizer_.Encode(*executor_, "000A01");
  ASSERT_FALSE(partial.ok());
  EXPECT_NE(partial.status().message().find("at byte 4: \"01\""),
            std::string::npos);
  const auto whitespace = tokenizer_.Encode(*executor_, "000\n");
  ASSERT_FALSE(whitespace.ok());
  EXPECT_NE(whitespace.status().message().find("at byte 3: \"\\n\""),
            std::string::npos);
}

TEST(FsmTokenizerDecodeTest, DecodesEmptyAndIndividualSpecialTokens) {
  const FsmTokenizer tokenizer;
  auto empty = tokenizer.Decode({});
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_EQ(*empty, "");
  auto symbols = tokenizer.Decode({1026, 1027, 1028});
  ASSERT_TRUE(symbols.ok()) << symbols.status();
  EXPECT_EQ(*symbols, ";>ERR");
  auto states = tokenizer.Decode({0, 9, 99, 999});
  ASSERT_TRUE(states.ok()) << states.status();
  EXPECT_EQ(*states, "000009099999");
}

TEST(FsmTokenizerDecodeTest, RejectsEveryOutOfRangeIdWithItsIndex) {
  const FsmTokenizer tokenizer;
  for (int invalid : {-1, 1029, std::numeric_limits<int>::min(),
                      std::numeric_limits<int>::max()}) {
    SCOPED_TRACE(invalid);
    auto decoded = tokenizer.Decode({0, invalid});
    ASSERT_FALSE(decoded.ok());
    EXPECT_EQ(decoded.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(decoded.status().message().find("at index 1"), std::string::npos);
  }
}

}  // namespace
}  // namespace pluto::llm::fsm
