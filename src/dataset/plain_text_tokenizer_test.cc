#include "src/dataset/plain_text_tokenizer.h"

#include <cuda_runtime_api.h>

#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/executor.h"
#include "src/dataset/detokenizer.h"
#include "src/dataset/tokenizer.h"

namespace pluto::tokenizer {
namespace {

static_assert(std::is_abstract_v<Tokenizer>);
static_assert(std::is_abstract_v<Detokenizer>);
static_assert(std::has_virtual_destructor_v<Tokenizer>);
static_assert(std::has_virtual_destructor_v<Detokenizer>);
static_assert(std::is_base_of_v<Tokenizer, PlainTextTokenizer>);
static_assert(std::is_base_of_v<Detokenizer, PlainTextTokenizer>);

TEST(PlainTextTokenizerTest, RoundTripsTextAndArbitraryBytesThroughBothBases) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  const PlainTextTokenizer implementation;
  const Tokenizer& encoder = implementation;
  const Detokenizer& decoder = implementation;
  EXPECT_EQ(encoder.vocab_size(), 256);
  EXPECT_EQ(decoder.vocab_size(), 256);

  const std::vector<std::string> samples = {
      "", "To be, or not to be.\n", std::string{"a\0b", 3}, "naïve café 🌍",
      std::string{"\xFF\x80\xC3", 3}};
  for (const std::string& input : samples) {
    SCOPED_TRACE(input.size());
    auto tokens = encoder.Encode(**executor, input);
    ASSERT_TRUE(tokens.ok()) << tokens.status();
    ASSERT_EQ(tokens->size(), input.size());
    EXPECT_EQ(&tokens->executor(), executor->get());
    for (size_t index = 0; index < input.size(); ++index)
      EXPECT_EQ((*tokens)[index], static_cast<unsigned char>(input[index]));
    const auto decoded = decoder.Decode(tokens->span());
    ASSERT_TRUE(decoded.ok()) << decoded.status();
    EXPECT_EQ(*decoded, input);
  }
}

TEST(PlainTextTokenizerTest, EveryByteHasItsIdenticallyNumberedToken) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  const PlainTextTokenizer implementation;
  const Tokenizer& encoder = implementation;
  const Detokenizer& decoder = implementation;
  std::string bytes;
  for (int value = 0; value < 256; ++value)
    bytes.push_back(static_cast<char>(value));

  auto tokens = encoder.Encode(**executor, bytes);
  ASSERT_TRUE(tokens.ok()) << tokens.status();
  ASSERT_EQ(tokens->size(), 256);
  for (int value = 0; value < 256; ++value)
    EXPECT_EQ((*tokens)[value], value);
  cudaPointerAttributes attributes{};
  ASSERT_EQ(cudaPointerGetAttributes(&attributes, tokens->data()), cudaSuccess);
  EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
  auto decoded = decoder.Decode(tokens->span());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(*decoded, bytes);
}

TEST(PlainTextTokenizerTest, OwnsConcreteImplementationsThroughEitherBase) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  std::unique_ptr<Tokenizer> encoder = std::make_unique<PlainTextTokenizer>();
  std::unique_ptr<Detokenizer> decoder = std::make_unique<PlainTextTokenizer>();
  auto encoded = encoder->Encode(**executor, "Hello");
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  auto decoded = decoder->Decode(encoded->span());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(*decoded, "Hello");
  // Destruction through either base must run the concrete destructor.
  encoder.reset();
  decoder.reset();
}

TEST(PlainTextTokenizerTest, HonorsTheExecutorPassedToEachCall) {
  auto first_executor = cuda::Executor::Create();
  auto second_executor = cuda::Executor::Create();
  ASSERT_TRUE(first_executor.ok()) << first_executor.status();
  ASSERT_TRUE(second_executor.ok()) << second_executor.status();
  const PlainTextTokenizer implementation;
  const Tokenizer& tokenizer = implementation;
  auto first = tokenizer.Encode(**first_executor, "same text");
  auto second = tokenizer.Encode(**second_executor, "same text");
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(&first->executor(), first_executor->get());
  EXPECT_EQ(&second->executor(), second_executor->get());
  EXPECT_EQ(first->span(), second->span());
}

TEST(PlainTextTokenizerTest, RejectsNegativeAndOutOfRangeTokens) {
  const PlainTextTokenizer implementation;
  const Detokenizer& decoder = implementation;
  for (int invalid : {-1, 256, std::numeric_limits<int>::min(),
                      std::numeric_limits<int>::max()}) {
    SCOPED_TRACE(invalid);
    const std::vector<int> tokens = {65, invalid, 66};
    const auto decoded = decoder.Decode(tokens);
    EXPECT_FALSE(decoded.ok());
    EXPECT_EQ(decoded.status().code(), absl::StatusCode::kInvalidArgument);
  }
}

}  // namespace
}  // namespace pluto::tokenizer
