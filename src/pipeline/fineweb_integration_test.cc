#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/executor.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/parquet/fineweb_parquet_reader.h"

namespace pluto::pipeline {
namespace {

std::filesystem::path RequiredDirectory(const char* variable) {
  const char* value = std::getenv(variable);
  EXPECT_NE(value, nullptr)
      << "set " << variable << " before running integration tests";
  return value == nullptr ? std::filesystem::path() : value;
}

TEST(FineWebIntegrationTest, SamplesEveryShardAndRoundTripsItsText) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  const std::filesystem::path parquet_directory =
      RequiredDirectory("PLUTO_FINEWEB_PARQUET_DIR");
  const std::filesystem::path tokenizer_directory =
      RequiredDirectory("PLUTO_GPT2_TOKENIZER_DIR");
  ASSERT_TRUE(std::filesystem::is_directory(parquet_directory));

  std::vector<std::filesystem::path> shards;
  for (const auto& entry :
       std::filesystem::directory_iterator(parquet_directory)) {
    if (entry.is_regular_file() && entry.path().extension() == ".parquet")
      shards.push_back(entry.path());
  }
  std::sort(shards.begin(), shards.end());
  ASSERT_FALSE(shards.empty());

  auto encoder = tokenizer::Gpt2Tokenizer::Load(tokenizer_directory);
  auto decoder = tokenizer::Gpt2Detokenizer::Load(tokenizer_directory);
  ASSERT_TRUE(encoder.ok()) << encoder.status();
  ASSERT_TRUE(decoder.ok()) << decoder.status();

  constexpr size_t kRecordsPerShard = 2;
  for (const std::filesystem::path& shard : shards) {
    SCOPED_TRACE(shard.filename().string());
    auto reader = parquet::FineWebParquetReader::Open(shard);
    ASSERT_TRUE(reader.ok()) << reader.status();
    ASSERT_GE((*reader)->num_rows(), static_cast<int64_t>(kRecordsPerShard));

    // ReadRows validates the complete schema. Only two records are materialized
    // from each multi-gigabyte file, keeping this integration test quick.
    auto rows = (*reader)->ReadRows(0, kRecordsPerShard);
    ASSERT_TRUE(rows.ok()) << rows.status();
    ASSERT_EQ(rows->size(), kRecordsPerShard);
    for (const parquet::FineWebRecord& row : *rows) {
      EXPECT_FALSE(row.text.empty());
      EXPECT_FALSE(row.id.empty());
      EXPECT_EQ(row.language, "en");
      EXPECT_GT(row.token_count, 0);

      auto token_ids = (*encoder)->Encode(**executor, row.text);
      ASSERT_TRUE(token_ids.ok()) << token_ids.status();
      EXPECT_FALSE(token_ids->empty());
      auto decoded = (*decoder)->Decode(*token_ids);
      ASSERT_TRUE(decoded.ok()) << decoded.status();
      EXPECT_EQ(*decoded, row.text);
    }
  }
}

}  // namespace
}  // namespace pluto::pipeline
