#include "src/llm/experiments/ntk/gram_kernel.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::llm::ntk::internal {
namespace {

class GramKernelTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }

  void TearDown() override {
    if (executor_ != nullptr)
      EXPECT_TRUE(executor_->Synchronize().ok());
  }

  absl::StatusOr<cuda::Buffer> Upload(const std::vector<float>& values) {
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<float>::CopyFrom(
                                    *executor_, absl::MakeConstSpan(values)));
    ASSIGN_OR_RETURN(auto device,
                     cuda::Buffer::Allocate(*executor_, host.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(device.data(), host.data(), host.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "upload Gram test Jacobian"));
    return device;
  }

  std::unique_ptr<cuda::Executor> executor_;
};

// A deliberately simple independent oracle. Long double limits accumulation
// error in the oracle; both factors are converted before multiplication.
double CpuEntry(const std::vector<float>& jacobian, size_t parameters,
                size_t row, size_t column) {
  long double result = 0;
  for (size_t parameter = 0; parameter < parameters; ++parameter)
    result +=
        static_cast<long double>(jacobian[row * parameters + parameter]) *
        static_cast<long double>(jacobian[column * parameters + parameter]);
  return static_cast<double>(result);
}

TEST_F(GramKernelTest, MatchesCpuAcrossThreadAndChunkBoundaries) {
  constexpr size_t kRows = 3;
  for (const size_t parameters : {1, 255, 256, 257, 4095, 4096, 4097, 8205}) {
    SCOPED_TRACE(parameters);
    std::vector<float> values(kRows * parameters);
    for (size_t row = 0; row < kRows; ++row)
      for (size_t parameter = 0; parameter < parameters; ++parameter)
        values[row * parameters + parameter] =
            static_cast<float>(static_cast<int>((parameter + 3 * row) % 17) -
                               8) *
            static_cast<float>(row + 1) / 4;
    auto jacobian = Upload(values);
    ASSERT_TRUE(jacobian.ok()) << jacobian.status();
    auto gram = ComputeGram(*executor_, *jacobian, kRows, parameters);
    ASSERT_TRUE(gram.ok()) << gram.status();
    ASSERT_EQ(gram->rows, kRows);
    ASSERT_EQ(gram->columns, kRows);
    for (size_t row = 0; row < kRows; ++row)
      for (size_t column = 0; column < kRows; ++column) {
        // These quarter-integer products and sums are exactly representable.
        EXPECT_DOUBLE_EQ((*gram)(row, column),
                         CpuEntry(values, parameters, row, column));
        EXPECT_DOUBLE_EQ((*gram)(row, column), (*gram)(column, row));
      }
  }
}

TEST_F(GramKernelTest, MultipliesInDoubleBeforeFp32ProductsCouldOverflow) {
  constexpr float kLarge = 1e30f;
  const std::vector<float> values{kLarge,  kLarge / 2, -kLarge,
                                  -kLarge, kLarge,     kLarge / 2};
  auto jacobian = Upload(values);
  ASSERT_TRUE(jacobian.ok()) << jacobian.status();
  auto gram = ComputeGram(*executor_, *jacobian, 2, 3);
  ASSERT_TRUE(gram.ok()) << gram.status();
  for (size_t row = 0; row < 2; ++row)
    for (size_t column = 0; column < 2; ++column) {
      const double expected = CpuEntry(values, 3, row, column);
      EXPECT_TRUE(std::isfinite((*gram)(row, column)));
      EXPECT_NEAR((*gram)(row, column), expected, std::abs(expected) * 1e-14);
    }
  EXPECT_LT((*gram)(0, 1), 0);
}

TEST_F(GramKernelTest, IsBitwiseRepeatableAndSymmetricWithUnevenChunks) {
  constexpr size_t kRows = 3;
  constexpr size_t kParameters = 8207;
  std::vector<float> values(kRows * kParameters);
  for (size_t index = 0; index < values.size(); ++index)
    values[index] = static_cast<float>(static_cast<int>(index % 101) - 50) /
                    static_cast<float>((index % 13) + 3);
  auto jacobian = Upload(values);
  ASSERT_TRUE(jacobian.ok()) << jacobian.status();
  auto expected = ComputeGram(*executor_, *jacobian, kRows, kParameters);
  ASSERT_TRUE(expected.ok()) << expected.status();
  for (int repetition = 0; repetition < 4; ++repetition) {
    auto actual = ComputeGram(*executor_, *jacobian, kRows, kParameters);
    ASSERT_TRUE(actual.ok()) << actual.status();
    EXPECT_EQ(std::memcmp(actual->values.data(), expected->values.data(),
                          expected->values.size() * sizeof(double)),
              0);
    for (size_t row = 0; row < kRows; ++row)
      for (size_t column = 0; column < kRows; ++column) {
        EXPECT_EQ(std::memcmp(&(*actual)(row, column), &(*actual)(column, row),
                              sizeof(double)),
                  0);
        const double oracle = CpuEntry(values, kParameters, row, column);
        EXPECT_NEAR((*actual)(row, column), oracle,
                    1e-12 * std::max(1.0, std::abs(oracle)));
      }
  }
}

TEST_F(GramKernelTest, RejectsZeroDimensionsMismatchedBytesAndForeignExecutor) {
  auto jacobian = Upload({1, 2, 3});
  ASSERT_TRUE(jacobian.ok()) << jacobian.status();
  EXPECT_TRUE(absl::IsInvalidArgument(
      ComputeGram(*executor_, *jacobian, 0, 3).status()));
  EXPECT_TRUE(absl::IsInvalidArgument(
      ComputeGram(*executor_, *jacobian, 3, 0).status()));
  EXPECT_TRUE(absl::IsInvalidArgument(
      ComputeGram(*executor_, *jacobian, 2, 1).status()));
  auto other_executor = cuda::Executor::Create();
  ASSERT_TRUE(other_executor.ok()) << other_executor.status();
  EXPECT_TRUE(absl::IsInvalidArgument(
      ComputeGram(**other_executor, *jacobian, 1, 3).status()));
}

TEST_F(GramKernelTest, RejectsSizeOverflowAndGridOverflowBeforeAllocating) {
  auto jacobian = Upload({1});
  ASSERT_TRUE(jacobian.ok()) << jacobian.status();
  constexpr size_t kMaximum = std::numeric_limits<size_t>::max();
  EXPECT_TRUE(absl::IsInvalidArgument(
      ComputeGram(*executor_, *jacobian, 2, kMaximum).status()));
  EXPECT_TRUE(absl::IsInvalidArgument(
      ComputeGram(*executor_, *jacobian, 1, kMaximum / sizeof(float) + 1)
          .status()));
  EXPECT_TRUE(absl::IsResourceExhausted(
      ComputeGram(*executor_, *jacobian, 65536, 1).status()));
  // Even these huge logical dimensions use only the one-element fixture.
  constexpr size_t kTooManyChunks =
      static_cast<size_t>(std::numeric_limits<int>::max()) * 4096 + 1;
  EXPECT_TRUE(absl::IsResourceExhausted(
      ComputeGram(*executor_, *jacobian, 1, kTooManyChunks).status()));
}

TEST_F(GramKernelTest, RejectsNonfiniteResults) {
  for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity()}) {
    auto jacobian = Upload({invalid});
    ASSERT_TRUE(jacobian.ok()) << jacobian.status();
    EXPECT_TRUE(absl::IsInvalidArgument(
        ComputeGram(*executor_, *jacobian, 1, 1).status()));
  }
}

}  // namespace
}  // namespace pluto::llm::ntk::internal
