#include "src/dataset/tokenizer.h"

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/executor.h"
#include "src/dataset/detokenizer.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/dataset/gpt2_tokenizer.h"

namespace pluto::tokenizer {
namespace {

static_assert(std::is_abstract_v<Tokenizer>);
static_assert(std::is_abstract_v<Detokenizer>);
static_assert(std::has_virtual_destructor_v<Tokenizer>);
static_assert(std::has_virtual_destructor_v<Detokenizer>);
static_assert(std::is_base_of_v<Tokenizer, Gpt2Tokenizer>);
static_assert(std::is_base_of_v<Detokenizer, Gpt2Detokenizer>);

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

  const std::vector<std::string> samples = {"",
                                            "Hello, world!",
                                            "  tabs\tand newlines\n",
                                            "naïve café 🌍",
                                            std::string{"before\0after", 12},
                                            "before<|endoftext|>after",
                                            std::string{"\xED\x9F\xBF", 3},
                                            std::string{"\xEE\x80\x80", 3},
                                            std::string{"\xF4\x8F\xBF\xBF", 4}};
  for (const std::string& sample : samples) {
    SCOPED_TRACE(testing::PrintToString(sample));
    auto encoded = (*tokenizer)->Encode(**executor, sample);
    ASSERT_TRUE(encoded.ok()) << encoded.status();
    auto decoded = (*detokenizer)->Decode(encoded->span());
    ASSERT_TRUE(decoded.ok()) << decoded.status();
    EXPECT_EQ(*decoded, sample);
  }
}

TEST(Gpt2TokenizerTest, OwnsConcreteImplementationsThroughAbstractInterfaces) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  auto tokenizer = Gpt2Tokenizer::Load(TokenizerDirectory());
  auto detokenizer = Gpt2Detokenizer::Load(TokenizerDirectory());
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
  ASSERT_TRUE(detokenizer.ok()) << detokenizer.status();
  std::unique_ptr<Tokenizer> encoder = std::move(*tokenizer);
  std::unique_ptr<Detokenizer> decoder = std::move(*detokenizer);
  EXPECT_EQ(encoder->vocab_size(), 50257);
  EXPECT_EQ(decoder->vocab_size(), encoder->vocab_size());
  const Tokenizer& encoder_view = *encoder;
  const Detokenizer& decoder_view = *decoder;

  for (const std::string& sample :
       {std::string{}, std::string{"Hello, 🌍!"}, std::string{"a\0b", 3},
        std::string{"<|endoftext|>"}}) {
    auto tokens = encoder_view.Encode(**executor, sample);
    ASSERT_TRUE(tokens.ok()) << tokens.status();
    EXPECT_EQ(&tokens->executor(), executor->get());
    if (tokens->empty()) {
      EXPECT_EQ(tokens->data(), nullptr);
    } else {
      cudaPointerAttributes attributes{};
      ASSERT_EQ(cudaPointerGetAttributes(&attributes, tokens->data()),
                cudaSuccess);
      EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
    }
    auto decoded = decoder_view.Decode(tokens->span());
    ASSERT_TRUE(decoded.ok()) << decoded.status();
    EXPECT_EQ(*decoded, sample);
  }
  encoder.reset();
  decoder.reset();
}

TEST(Gpt2TokenizerTest, HonorsTheExecutorPassedToEachCall) {
  auto first_executor = cuda::Executor::Create();
  auto second_executor = cuda::Executor::Create();
  ASSERT_TRUE(first_executor.ok()) << first_executor.status();
  ASSERT_TRUE(second_executor.ok()) << second_executor.status();
  auto implementation = Gpt2Tokenizer::Load(TokenizerDirectory());
  ASSERT_TRUE(implementation.ok()) << implementation.status();
  const Tokenizer& tokenizer = **implementation;

  // The vocabulary/BPE cache is shared, but output allocations are not tied to
  // the executor of the first call.
  auto first = tokenizer.Encode(**first_executor, "Hello, world!");
  auto second = tokenizer.Encode(**second_executor, "Hello, world!");
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(&first->executor(), first_executor->get());
  EXPECT_EQ(&second->executor(), second_executor->get());
  EXPECT_EQ(first->span(), second->span());
}

TEST(Gpt2TokenizerTest, BaseInterfacePreservesInvalidUtf8Errors) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  auto implementation = Gpt2Tokenizer::Load(TokenizerDirectory());
  ASSERT_TRUE(implementation.ok()) << implementation.status();
  const Tokenizer& tokenizer = **implementation;
  const std::vector<std::string> malformed = {
      std::string{"\x80", 1},
      std::string{"\xC3", 1},
      std::string{"\xC3\x28", 2},
      std::string{"\xC0\xAF", 2},
      std::string{"\xED\xA0\x80", 3},
      std::string{"\xED\xAF\xBF", 3},
      std::string{"\xED\xB0\x80", 3},
      std::string{"\xED\xBF\xBF", 3},
      std::string{"\xF4\x90\x80\x80", 4},
      std::string{"\xFF", 1}};
  for (const std::string& bytes : malformed) {
    SCOPED_TRACE(testing::PrintToString(bytes));
    auto encoded = tokenizer.Encode(**executor, "valid prefix " + bytes);
    EXPECT_FALSE(encoded.ok());
    EXPECT_EQ(encoded.status().code(), absl::StatusCode::kInvalidArgument);
  }
  // Failed calls must not corrupt the tokenizer's cache or later output.
  auto valid = tokenizer.Encode(**executor, "Hello, world!");
  ASSERT_TRUE(valid.ok()) << valid.status();
  EXPECT_EQ(std::vector<int>(valid->begin(), valid->end()),
            (std::vector<int>{15496, 11, 995, 0}));
}

TEST(Gpt2DetokenizerTest, RejectsUnknownTokenIdsThroughBaseInterface) {
  auto implementation = Gpt2Detokenizer::Load(TokenizerDirectory());
  ASSERT_TRUE(implementation.ok()) << implementation.status();
  const Detokenizer& detokenizer = **implementation;
  for (int invalid :
       {-1, detokenizer.vocab_size(), std::numeric_limits<int>::min(),
        std::numeric_limits<int>::max()}) {
    SCOPED_TRACE(invalid);
    const std::vector<int> tokens = {15496, invalid};
    auto decoded = detokenizer.Decode(tokens);
    EXPECT_FALSE(decoded.ok());
    EXPECT_EQ(decoded.status().code(), absl::StatusCode::kInvalidArgument);
  }
}

}  // namespace
}  // namespace pluto::tokenizer
