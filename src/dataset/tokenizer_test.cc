#include "src/dataset/tokenizer.h"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/executor.h"
#include "src/dataset/detokenizer.h"

namespace pluto::tokenizer {
namespace {

std::filesystem::path TokenizerDirectory() {
  const char* directory = std::getenv("PLUTO_GPT2_TOKENIZER_DIR");
  EXPECT_NE(directory, nullptr)
      << "set PLUTO_GPT2_TOKENIZER_DIR to the saved GPT-2 tokenizer";
  return directory == nullptr ? std::filesystem::path() : directory;
}

TEST(Gpt2TokenizerTest, MatchesReferenceTokenIds) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  auto tokenizer = Gpt2Tokenizer::Load(TokenizerDirectory());
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();

  EXPECT_EQ((*tokenizer)->vocab_size(), 50257);
  EXPECT_EQ((*tokenizer)->eos_token_id(), 50256);

  auto ascii = (*tokenizer)->Encode(**executor, "Hello, world!");
  ASSERT_TRUE(ascii.ok()) << ascii.status();
  EXPECT_EQ(std::vector<int>(ascii->begin(), ascii->end()),
            (std::vector<int>{15496, 11, 995, 0}));

  auto unicode = (*tokenizer)->Encode(**executor, "Hello, 🌍!");
  ASSERT_TRUE(unicode.ok()) << unicode.status();
  EXPECT_EQ(std::vector<int>(unicode->begin(), unicode->end()),
            (std::vector<int>{15496, 11, 12520, 234, 235, 0}));

  auto spaces = (*tokenizer)->Encode(**executor, "  leading spaces");
  ASSERT_TRUE(spaces.ok()) << spaces.status();
  EXPECT_EQ(std::vector<int>(spaces->begin(), spaces->end()),
            (std::vector<int>{220, 3756, 9029}));
}

TEST(Gpt2TokenizerTest, RecognizesTheEosToken) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  auto tokenizer = Gpt2Tokenizer::Load(TokenizerDirectory());
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
  auto encoded = (*tokenizer)->Encode(**executor, "a<|endoftext|>b");
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  EXPECT_EQ(std::vector<int>(encoded->begin(), encoded->end()),
            (std::vector<int>{64, 50256, 65}));
}

TEST(Gpt2TokenizerTest, DetokenizerRoundTripsBytes) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  auto tokenizer = Gpt2Tokenizer::Load(TokenizerDirectory());
  auto detokenizer = Gpt2Detokenizer::Load(TokenizerDirectory());
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
  ASSERT_TRUE(detokenizer.ok()) << detokenizer.status();

  const std::vector<std::string> samples = {
      "", "Hello, world!", "  tabs\tand newlines\n", "naïve café 🌍",
      "before<|endoftext|>after"};
  for (const std::string& sample : samples) {
    auto encoded = (*tokenizer)->Encode(**executor, sample);
    ASSERT_TRUE(encoded.ok()) << encoded.status();
    auto decoded = (*detokenizer)->Decode(encoded->span());
    ASSERT_TRUE(decoded.ok()) << decoded.status();
    EXPECT_EQ(*decoded, sample);
  }
}

TEST(Gpt2DetokenizerTest, RejectsUnknownTokenId) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  auto detokenizer = Gpt2Detokenizer::Load(TokenizerDirectory());
  ASSERT_TRUE(detokenizer.ok()) << detokenizer.status();
  const std::vector<int> invalid = {-1};
  EXPECT_FALSE((*detokenizer)->Decode(invalid).ok());
}

}  // namespace
}  // namespace pluto::tokenizer
