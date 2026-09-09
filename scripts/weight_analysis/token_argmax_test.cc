#include "scripts/weight_analysis/token_argmax.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <utility>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::weight_analysis {
namespace {

class TokenArgmaxTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }

  void TearDown() override {
    if (executor_) {
      EXPECT_TRUE(executor_->Synchronize().ok());
    }
  }

  absl::StatusOr<cuda::PageLockedHostArray<int32_t>> Run(
      const cuda::PageLockedHostArray<float>& host, int rows, int logical,
      int padded) {
    ASSIGN_OR_RETURN(auto input,
                     cuda::Buffer::Allocate(*executor_, host.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(input.data(), host.data(), host.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "test logits upload"));
    // No intermediate synchronization: the utility must use this exact
    // executor so the earlier H2D and following D2H are correctly ordered.
    ASSIGN_OR_RETURN(auto result,
                     ArgmaxTokens(*executor_, input, rows, logical, padded));
    EXPECT_EQ(&result.executor(), executor_.get());
    EXPECT_EQ(result.size_bytes(), static_cast<size_t>(rows) * sizeof(int32_t));
    ASSIGN_OR_RETURN(auto output,
                     cuda::PageLockedHostArray<int32_t>::Allocate(rows));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(output.data(), result.data(), output.size_bytes(),
                        cudaMemcpyDeviceToHost, executor_->stream()),
        "test token IDs download"));
    RETURN_IF_ERROR(executor_->Synchronize());
    return output;
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(TokenArgmaxTest, NegativeValuesSignedZeroAndMinimumIdTies) {
  auto host = cuda::PageLockedHostArray<float>::Allocate(5 * 7);
  ASSERT_TRUE(host.ok()) << host.status();
  const float lowest = std::numeric_limits<float>::lowest();
  const std::array<float, 35> values = {
      -9,     -2,    -2,     -5,     -6,     100,    200,    -0.0f,  0.0f,
      -1,     -2,    -3,     100,    200,    lowest, lowest, lowest, lowest,
      lowest, 100,   200,    -3,     -4,     -5,     -6,     -1,     100,
      200,    -.25f, -.125f, -.125f, -.375f, -1,     100,    200};
  std::copy(values.begin(), values.end(), host->begin());
  auto result = Run(*host, 5, 5, 7);
  ASSERT_TRUE(result.ok()) << result.status();
  const std::array<int32_t, 5> expected = {1, 0, 0, 4, 1};
  for (int row = 0; row < 5; ++row) EXPECT_EQ((*result)[row], expected[row]);
}

TEST_F(TokenArgmaxTest, TileBoundariesOddStridesAndLogicalVocabularyPadding) {
  const std::array<std::pair<int, int>, 8> widths = {{{1, 1},
                                                      {255, 257},
                                                      {256, 256},
                                                      {257, 261},
                                                      {511, 513},
                                                      {512, 512},
                                                      {513, 519},
                                                      {50257, 50272}}};
  for (auto [logical, padded] : widths) {
    SCOPED_TRACE(logical);
    auto host = cuda::PageLockedHostArray<float>::Allocate(4 * padded);
    ASSERT_TRUE(host.ok()) << host.status();
    std::fill(host->begin(), host->end(), -20.0f);
    for (int row = 0; row < 4; ++row) {
      for (int token = logical; token < padded; ++token) {
        (*host)[row * padded + token] = std::numeric_limits<float>::quiet_NaN();
      }
    }
    (*host)[logical - 1] = 7;
    (*host)[padded] = 7;
    (*host)[padded + logical - 1] = 7;
    (*host)[2 * padded + logical / 2] = 7;
    (*host)[2 * padded + logical - 1] = 7;
    auto result = Run(*host, 4, logical, padded);
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ((*result)[0], logical - 1);
    EXPECT_EQ((*result)[1], 0);
    EXPECT_EQ((*result)[2], logical / 2);
    EXPECT_EQ((*result)[3], 0);
  }
}

TEST_F(TokenArgmaxTest, EveryNonfiniteLogicalValueInvalidatesItsRow) {
  auto host = cuda::PageLockedHostArray<float>::Allocate(6 * 8);
  ASSERT_TRUE(host.ok()) << host.status();
  std::fill(host->begin(), host->end(), 0.0f);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  (*host)[0] = nan;
  (*host)[8 + 4] = infinity;
  (*host)[16 + 4] = -infinity;
  for (int token = 0; token < 5; ++token) (*host)[24 + token] = nan;
  (*host)[32 + 3] = 9;
  (*host)[32 + 5] = nan;
  (*host)[32 + 6] = infinity;
  (*host)[32 + 7] = -infinity;
  (*host)[40 + 1] = 9;
  (*host)[40 + 4] =
      nan;  // Not merely a check that the winning value is finite.
  auto result = Run(*host, 6, 5, 8);
  ASSERT_TRUE(result.ok()) << result.status();
  for (int row = 0; row < 6; ++row) {
    EXPECT_EQ((*result)[row], row == 4 ? 3 : kInvalidTokenId);
  }
}

TEST_F(TokenArgmaxTest, MultipleRowsAgreeWithIndependentScalarOracle) {
  constexpr int kRows = 37;
  constexpr int kLogical = 777;
  constexpr int kPadded = 789;
  auto host = cuda::PageLockedHostArray<float>::Allocate(kRows * kPadded);
  ASSERT_TRUE(host.ok()) << host.status();
  std::mt19937 random(17);
  std::uniform_int_distribution<int> draw(-120, 120);
  for (float& value : *host) value = static_cast<float>(draw(random)) / 8;
  auto result = Run(*host, kRows, kLogical, kPadded);
  ASSERT_TRUE(result.ok()) << result.status();
  for (int row = 0; row < kRows; ++row) {
    int expected = 0;
    for (int token = 1; token < kLogical; ++token) {
      if ((*host)[row * kPadded + token] > (*host)[row * kPadded + expected]) {
        expected = token;
      }
    }
    EXPECT_EQ((*result)[row], expected) << "row " << row;
  }
}

TEST_F(TokenArgmaxTest, RejectsInvalidDimensionsByteCountsAndExecutor) {
  auto input = cuda::Buffer::Allocate(*executor_, 4 * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();
  for (const auto& shape : std::array<std::array<int, 3>, 7>{{{0, 1, 4},
                                                              {-1, 1, 4},
                                                              {1, 0, 4},
                                                              {1, 5, 4},
                                                              {1, 1, 0},
                                                              {2, 1, 4},
                                                              {1, 1, 3}}}) {
    auto result =
        ArgmaxTokens(*executor_, *input, shape[0], shape[1], shape[2]);
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  }
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  EXPECT_EQ(ArgmaxTokens(**other, *input, 1, 1, 4).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::weight_analysis
