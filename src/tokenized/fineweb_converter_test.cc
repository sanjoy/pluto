#include "src/tokenized/fineweb_converter.h"

#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "src/tokenized/document_file.h"
#include "src/tokenizer/detokenizer.h"
#include "src/tokenizer/tokenizer.h"

namespace pluto::tokenized {
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

TEST(FineWebConverterTest, ConvertsAndRoundTripsEveryFixtureDocument) {
  auto encoder = tokenizer::Gpt2Tokenizer::Load(TokenizerDirectory());
  auto decoder = tokenizer::Gpt2Detokenizer::Load(TokenizerDirectory());
  ASSERT_TRUE(encoder.ok()) << encoder.status();
  ASSERT_TRUE(decoder.ok()) << decoder.status();

  const std::filesystem::path output =
      std::filesystem::path(testing::TempDir()) / "fixture.tokenized";
  // A one-row batch deliberately exercises multiple reads and the fixture's
  // row-group boundary through the same helper used by the production binary.
  FineWebConversionOptions options;
  options.batch_size = 1;
  ASSERT_TRUE(
      ConvertFineWebParquetFile(FixturePath(), output, **encoder, options).ok());

  auto reader = DocumentFileReader::Open(output);
  ASSERT_TRUE(reader.ok()) << reader.status();
  ASSERT_EQ((*reader)->num_documents(), 3u);
  const std::vector<std::string> expected = {
      "Hello, world!", "The quick brown fox.", "naïve café 🌍"};
  for (uint32_t index = 0; index < expected.size(); ++index) {
    auto stored = (*reader)->ReadDocument(index);
    ASSERT_TRUE(stored.ok()) << stored.status();
    std::vector<int> token_ids(stored->begin(), stored->end());
    auto decoded = (*decoder)->Decode(token_ids);
    ASSERT_TRUE(decoded.ok()) << decoded.status();
    EXPECT_EQ(*decoded, expected[index]);
  }
}

TEST(FineWebConverterTest, RejectsZeroBatchSizeWithoutPublishingOutput) {
  auto encoder = tokenizer::Gpt2Tokenizer::Load(TokenizerDirectory());
  ASSERT_TRUE(encoder.ok()) << encoder.status();
  const std::filesystem::path output =
      std::filesystem::path(testing::TempDir()) / "invalid.tokenized";
  FineWebConversionOptions options;
  options.batch_size = 0;
  EXPECT_FALSE(
      ConvertFineWebParquetFile(FixturePath(), output, **encoder, options).ok());
  EXPECT_FALSE(std::filesystem::exists(output));
}

}  // namespace
}  // namespace pluto::tokenized
