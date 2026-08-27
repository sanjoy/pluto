#include "src/tokenization/plain_text_tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::tokenization {
namespace {

TEST(PlainTextTokenizerTest, RoundTripsTextAndArbitraryBytes) {
  const PlainTextTokenizer tokenizer;
  const std::string input{"To be, or not to be.\n\0\xFF", 23};

  const std::vector<uint32_t> tokens = tokenizer.Encode(input);
  ASSERT_EQ(tokens.size(), input.size());
  EXPECT_EQ(tokens[0], static_cast<uint32_t>('T'));
  EXPECT_EQ(tokens.back(), 255u);

  const auto decoded = tokenizer.Decode(tokens);
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(*decoded, input);
}

TEST(PlainTextTokenizerTest, RejectsTokensThatAreNotBytes) {
  const PlainTextTokenizer tokenizer;
  const std::vector<uint32_t> tokens = {65, 256};
  const auto decoded = tokenizer.Decode(tokens);

  EXPECT_FALSE(decoded.ok());
  EXPECT_EQ(decoded.status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::tokenization
