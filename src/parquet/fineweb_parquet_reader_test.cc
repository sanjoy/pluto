#include "src/parquet/fineweb_parquet_reader.h"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::parquet {
namespace {

std::filesystem::path TestDataPath() {
  const char* runfiles = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  EXPECT_NE(runfiles, nullptr);
  EXPECT_NE(workspace, nullptr);
  return std::filesystem::path(runfiles == nullptr ? "" : runfiles) /
         (workspace == nullptr ? "" : workspace) /
         "src/parquet/testdata/fineweb_sample.parquet";
}

TEST(FineWebParquetReaderTest, ReadsMetadataAndProjectedText) {
  auto reader = FineWebParquetReader::Open(TestDataPath());
  ASSERT_TRUE(reader.ok()) << reader.status();
  EXPECT_EQ((*reader)->num_rows(), 3);
  EXPECT_EQ((*reader)->num_row_groups(), size_t{2});

  auto text = (*reader)->ReadTextRows(0, 3);
  ASSERT_TRUE(text.ok()) << text.status();
  EXPECT_EQ(*text, (std::vector<std::string>{
                       "Hello, world!", "The quick brown fox.", "naïve café 🌍"}));
}

TEST(FineWebParquetReaderTest, ReadsAllColumnsAcrossRowGroups) {
  auto reader = FineWebParquetReader::Open(TestDataPath());
  ASSERT_TRUE(reader.ok()) << reader.status();

  // Row group size is two, so this request exercises boundary crossing.
  auto rows = (*reader)->ReadRows(1, 2);
  ASSERT_TRUE(rows.ok()) << rows.status();
  ASSERT_EQ(rows->size(), size_t{2});

  EXPECT_EQ((*rows)[0].text, "The quick brown fox.");
  EXPECT_EQ((*rows)[0].id, "id-2");
  EXPECT_EQ((*rows)[0].dump, "CC-MAIN-2026-30");
  EXPECT_EQ((*rows)[0].url, "https://example.test/two");
  EXPECT_EQ((*rows)[0].file_path, "s3://example/two.warc.gz");
  EXPECT_EQ((*rows)[0].language, "en");
  EXPECT_DOUBLE_EQ((*rows)[0].language_score, 0.95);
  EXPECT_EQ((*rows)[0].token_count, 5);
  EXPECT_DOUBLE_EQ((*rows)[0].score, 3.5);
  EXPECT_EQ((*rows)[0].int_score, 4);

  EXPECT_EQ((*rows)[1].text, "naïve café 🌍");
  EXPECT_EQ((*rows)[1].id, "id-3");
}

TEST(FineWebParquetReaderTest, RejectsOutOfRangeReads) {
  auto reader = FineWebParquetReader::Open(TestDataPath());
  ASSERT_TRUE(reader.ok()) << reader.status();
  EXPECT_FALSE((*reader)->ReadRows(-1, 1).ok());
  EXPECT_FALSE((*reader)->ReadRows(2, 2).ok());
  EXPECT_TRUE((*reader)->ReadTextRows(3, 0).ok());
}

}  // namespace
}  // namespace pluto::parquet
