#include "ai-slop/weight_analysis/token_trace_probe.h"

#include <cuda_runtime_api.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>

#include "ai-slop/weight_analysis/causal_probe.h"
#include "gtest/gtest.h"

namespace pluto::weight_analysis {
namespace {

TEST(TokenTraceValidationTest, PreservesIdsAndAcceptsVocabularyEndpoints) {
  const std::array<int32_t, 4> ids{50256, 0, 257, 257};
  EXPECT_TRUE(ValidateTokenIds(ids, 50257, 4).ok());
  EXPECT_EQ(ids, (std::array<int32_t, 4>{50256, 0, 257, 257}));
  EXPECT_EQ(ValidateTokenIds(ids, 50257, 3).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ValidateTokenIds({}, 50257, 1024).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_FALSE(ValidateTokenIds(ids, 0, 4).ok());
  EXPECT_FALSE(ValidateTokenIds(ids, 50257, -1).ok());
  for (const int32_t invalid : {-1, 50257, std::numeric_limits<int32_t>::max()})
    EXPECT_FALSE(
        ValidateTokenIds(absl::MakeConstSpan(&invalid, 1), 50257, 4).ok());
}

class TokenTraceGpuTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
    std::string pattern =
        (std::filesystem::temp_directory_path() / "token-trace-test.XXXXXX")
            .string();
    char* created = mkdtemp(pattern.data());
    ASSERT_NE(created, nullptr);
    directory_ = created;
  }
  void TearDown() override {
    if (executor_)
      EXPECT_TRUE(executor_->Synchronize().ok());
    // Only this test's exclusively created directory and fixtures are removed.
    if (!directory_.empty())
      std::filesystem::remove_all(directory_);
  }
  std::unique_ptr<cuda::Executor> executor_;
  std::filesystem::path directory_;
};

TEST_F(TokenTraceGpuTest, FileBytesAreExactLittleEndianIdsWithoutRetokenizing) {
  static_assert(std::endian::native == std::endian::little);
  const std::array<uint8_t, 12> bytes{0, 0, 0, 0, 1, 1, 0, 0, 0x50, 0xc4, 0, 0};
  const auto path = directory_ / "tokens.i32";
  ASSERT_TRUE(WriteExclusive(path, bytes.data(), bytes.size()).ok());
  auto ids = ReadTokenIds(path, 50257, 1024);
  ASSERT_TRUE(ids.ok()) << ids.status();
  ASSERT_EQ(ids->size(), 3U);
  EXPECT_EQ((*ids)[0], 0);
  EXPECT_EQ((*ids)[1], 257);
  EXPECT_EQ((*ids)[2], 50256);
  EXPECT_EQ(std::memcmp(ids->data(), bytes.data(), bytes.size()), 0);
  cudaPointerAttributes attributes{};
  ASSERT_EQ(cudaPointerGetAttributes(&attributes, ids->data()), cudaSuccess);
  EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
}

TEST_F(TokenTraceGpuTest, TokenFileRejectsMalformedSizeIdsAndNonregularInputs) {
  const std::array<int32_t, 3> valid{0, 1, 2};
  const auto good = directory_ / "good";
  ASSERT_TRUE(WriteExclusive(good, valid.data(), sizeof(valid)).ok());
  EXPECT_FALSE(ReadTokenIds(good, 50257, 2).ok());
  EXPECT_FALSE(ReadTokenIds(good, 0, 1024).ok());
  EXPECT_FALSE(ReadTokenIds(good, 50257, 0).ok());
  const auto empty = directory_ / "empty";
  ASSERT_TRUE(WriteExclusive(empty, valid.data(), 0).ok());
  EXPECT_FALSE(ReadTokenIds(empty, 50257, 1024).ok());
  const auto partial = directory_ / "partial";
  ASSERT_TRUE(WriteExclusive(partial, valid.data(), 3).ok());
  EXPECT_FALSE(ReadTokenIds(partial, 50257, 1024).ok());
  for (const int32_t value : {-1, 50257}) {
    const auto invalid = directory_ / std::to_string(value);
    ASSERT_TRUE(WriteExclusive(invalid, &value, sizeof(value)).ok());
    EXPECT_FALSE(ReadTokenIds(invalid, 50257, 1024).ok());
  }
  const auto link = directory_ / "symlink";
  std::filesystem::create_symlink(good, link);
  EXPECT_FALSE(ReadTokenIds(link, 50257, 1024).ok());
  EXPECT_FALSE(ReadTokenIds(directory_, 50257, 1024).ok());
  EXPECT_FALSE(ReadTokenIds(directory_ / "missing", 50257, 1024).ok());
  const auto fifo = directory_ / "fifo";
  ASSERT_EQ(mkfifo(fifo.c_str(), 0600), 0);
  // Must reject without waiting for a writer to connect.
  EXPECT_FALSE(ReadTokenIds(fifo, 50257, 1024).ok());
}

TEST_F(TokenTraceGpuTest, ReadsFirstMiddleAndLastRowsWithExactByteOffsets) {
  auto host = cuda::PageLockedHostArray<uint8_t>::Allocate(4 * 8 * 4);
  ASSERT_TRUE(host.ok());
  for (size_t i = 0; i < host->size(); ++i)
    (*host)[i] = i;
  auto buffer = cuda::Buffer::Allocate(*executor_, host->size_bytes());
  ASSERT_TRUE(buffer.ok());
  ASSERT_EQ(cudaMemcpyAsync(buffer->data(), host->data(), host->size_bytes(),
                            cudaMemcpyHostToDevice, executor_->stream()),
            cudaSuccess);
  for (int row = 0; row < 4; ++row) {
    auto actual = ReadSelectedRow(*executor_, *buffer, row, 8, 4);
    ASSERT_TRUE(actual.ok()) << actual.status();
    ASSERT_EQ(actual->size_bytes(), 32U);
    EXPECT_EQ(std::memcmp(actual->data(), host->data() + row * 32, 32), 0);
    cudaPointerAttributes attributes{};
    ASSERT_EQ(cudaPointerGetAttributes(&attributes, actual->data()),
              cudaSuccess);
    EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
  }
  auto bf16_row = ReadSelectedRow(*executor_, *buffer, 7, 8, 2);
  ASSERT_TRUE(bf16_row.ok());
  EXPECT_EQ(bf16_row->size_bytes(), 16U);
  EXPECT_EQ(std::memcmp(bf16_row->data(), host->data() + 112, 16), 0);
}

TEST_F(TokenTraceGpuTest, RejectsInvalidRowsShapesTypesAndDifferentExecutor) {
  auto buffer = cuda::Buffer::Allocate(*executor_, 4 * 8 * 4);
  ASSERT_TRUE(buffer.ok());
  const std::array<std::array<int, 3>, 10> invalid{
      {{-1, 8, 4},
       {4, 8, 4},
       {std::numeric_limits<int>::max(), 8, 4},
       {0, 0, 4},
       {0, -1, 4},
       {0, 7, 4},
       {0, 8, 0},
       {0, 8, 1},
       {0, 8, 8},
       {0, std::numeric_limits<int>::max(), 4}}};
  for (const auto& spec : invalid) {
    EXPECT_EQ(ReadSelectedRow(*executor_, *buffer, spec[0], spec[1], spec[2])
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok());
  EXPECT_EQ(ReadSelectedRow(**other, *buffer, 0, 8, 4).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::weight_analysis
