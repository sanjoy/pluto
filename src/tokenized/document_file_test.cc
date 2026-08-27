#include "src/tokenized/document_file.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::tokenized {
namespace {

TEST(DocumentFileTest, RoundTripsDocumentsIncludingEmptyAndMaximumToken) {
  const std::filesystem::path path =
      std::filesystem::path(testing::TempDir()) / "round_trip.tokenized";
  auto writer = DocumentFileWriter::Create(path, 3);
  ASSERT_TRUE(writer.ok()) << writer.status();
  EXPECT_EQ((*writer)->expected_documents(), 3u);

  const std::vector<uint16_t> first = {0, 1, 50256};
  const std::vector<uint16_t> empty;
  const std::vector<uint16_t> last = {65535};
  ASSERT_TRUE((*writer)->AddDocument(first).ok());
  ASSERT_TRUE((*writer)->AddDocument(empty).ok());
  ASSERT_TRUE((*writer)->AddDocument(last).ok());
  EXPECT_EQ((*writer)->documents_written(), 3u);
  ASSERT_TRUE((*writer)->Close().ok());

  auto reader = DocumentFileReader::Open(path);
  ASSERT_TRUE(reader.ok()) << reader.status();
  EXPECT_EQ((*reader)->num_documents(), 3u);
  EXPECT_EQ(*(*reader)->document_length(0), 3u);
  EXPECT_EQ(*(*reader)->document_length(1), 0u);
  EXPECT_EQ(*(*reader)->ReadDocument(0), first);
  EXPECT_EQ(*(*reader)->ReadDocument(1), empty);
  EXPECT_EQ(*(*reader)->ReadDocument(2), last);
  EXPECT_FALSE((*reader)->ReadDocument(3).ok());
}

TEST(DocumentFileTest, UsesTheDocumentedLittleEndianLayout) {
  const std::filesystem::path path =
      std::filesystem::path(testing::TempDir()) / "layout.tokenized";
  auto writer = DocumentFileWriter::Create(path, 2);
  ASSERT_TRUE(writer.ok()) << writer.status();
  const std::vector<uint16_t> first = {0x1234, 0xabcd};
  const std::vector<uint16_t> second = {0xffff};
  ASSERT_TRUE((*writer)->AddDocument(first).ok());
  ASSERT_TRUE((*writer)->AddDocument(second).ok());
  ASSERT_TRUE((*writer)->Close().ok());

  std::ifstream input(path, std::ios::binary);
  const std::vector<uint8_t> actual{
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  const std::vector<uint8_t> expected = {
      0x02, 0x00, 0x00, 0x00,  // document count
      0x02, 0x00, 0x00, 0x00,  // first length
      0x01, 0x00, 0x00, 0x00,  // second length
      0x34, 0x12, 0xcd, 0xab, 0xff, 0xff};
  EXPECT_EQ(actual, expected);
}

TEST(DocumentFileTest, DoesNotPublishAnIncompleteFile) {
  const std::filesystem::path path =
      std::filesystem::path(testing::TempDir()) / "incomplete.tokenized";
  {
    auto writer = DocumentFileWriter::Create(path, 2);
    ASSERT_TRUE(writer.ok()) << writer.status();
    const std::vector<uint16_t> only_document = {1};
    ASSERT_TRUE((*writer)->AddDocument(only_document).ok());
    EXPECT_FALSE((*writer)->Close().ok());
  }
  EXPECT_FALSE(std::filesystem::exists(path));
}

TEST(DocumentFileTest, RejectsTruncatedPayload) {
  const std::filesystem::path path =
      std::filesystem::path(testing::TempDir()) / "truncated.tokenized";
  const char bytes[] = {
      0x01, 0x00, 0x00, 0x00,  // one document
      0x02, 0x00, 0x00, 0x00,  // claims two tokens
      0x01, 0x00};              // only one token
  std::ofstream(path, std::ios::binary).write(bytes, sizeof(bytes));
  EXPECT_FALSE(DocumentFileReader::Open(path).ok());
}

}  // namespace
}  // namespace pluto::tokenized
