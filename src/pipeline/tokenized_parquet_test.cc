#include <cstdlib>
#include <filesystem>
#include <string>

#include "gtest/gtest.h"
#include "src/parquet/fineweb_parquet_reader.h"
#include "src/tokenization/detokenizer.h"
#include "src/tokenization/tokenizer.h"

namespace pluto::pipeline {
namespace {

std::filesystem::path TokenizerDirectory() {
  const char* directory = std::getenv("PLUTO_GPT2_TOKENIZER_DIR");
  EXPECT_NE(directory, nullptr)
      << "set PLUTO_GPT2_TOKENIZER_DIR to the saved GPT-2 tokenizer";
  return directory == nullptr ? std::filesystem::path() : directory;
}

std::filesystem::path FixturePath() {
  const char* runfiles = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  EXPECT_NE(runfiles, nullptr);
  EXPECT_NE(workspace, nullptr);
  return std::filesystem::path(runfiles == nullptr ? "" : runfiles) /
         (workspace == nullptr ? "" : workspace) /
         "src/parquet/testdata/fineweb_sample.parquet";
}

TEST(TokenizedParquetTest, TextProjectionRoundTripsThroughGpt2) {
  auto reader = parquet::FineWebParquetReader::Open(FixturePath());
  auto encoder = tokenizer::Gpt2Tokenizer::Load(TokenizerDirectory());
  auto decoder = tokenizer::Gpt2Detokenizer::Load(TokenizerDirectory());
  ASSERT_TRUE(reader.ok()) << reader.status();
  ASSERT_TRUE(encoder.ok()) << encoder.status();
  ASSERT_TRUE(decoder.ok()) << decoder.status();

  auto texts = (*reader)->ReadTextRows(0, 3);
  ASSERT_TRUE(texts.ok()) << texts.status();
  for (const std::string& text : *texts) {
    auto token_ids = (*encoder)->Encode(text);
    ASSERT_TRUE(token_ids.ok()) << token_ids.status();
    EXPECT_FALSE(token_ids->empty());
    auto decoded = (*decoder)->Decode(*token_ids);
    ASSERT_TRUE(decoded.ok()) << decoded.status();
    EXPECT_EQ(*decoded, text);
  }
}

}  // namespace
}  // namespace pluto::pipeline
