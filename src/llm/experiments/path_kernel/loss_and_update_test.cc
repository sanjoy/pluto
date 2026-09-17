#include <cuda_runtime_api.h>

#include <cmath>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/experiments/path_kernel/path_kernel.h"
#include "src/llm/experiments/path_kernel/update_kernel.h"
#include "src/util/status_macros.h"

namespace pluto::llm::path_kernel {
namespace {

absl::StatusOr<Buffer> Upload(cuda::Executor& executor,
                              const std::vector<float>& values) {
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<float>::CopyFrom(
                                  executor, absl::MakeConstSpan(values)));
  ASSIGN_OR_RETURN(auto buffer, Buffer::Allocate(executor, host.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(buffer.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload path-kernel loss/update test"));
  return buffer;
}

absl::StatusOr<std::vector<float>> Read(cuda::Executor& executor,
                                        const Buffer& buffer) {
  ASSIGN_OR_RETURN(auto host,
                   cuda::PageLockedHostArray<float>::Allocate(
                       executor, buffer.size_bytes() / sizeof(float)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), buffer.data(), buffer.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "read path-kernel loss/update test"));
  RETURN_IF_ERROR(executor.Synchronize());
  return std::vector<float>(host.begin(), host.end());
}

TEST(PathLossTest, LogicalVocabularyExcludesOtherPositionsPaddingAndOutputs) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  auto unused = Upload(**executor, {1000, 1000});
  auto logits = Upload(**executor, {2000, 1, 2, 3, 3000, 4000});
  ASSERT_TRUE(unused.ok());
  ASSERT_TRUE(logits.ok());
  auto result = CrossEntropy({1, 1}, 3, 1)(**executor, {*unused, *logits});
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->gradients.size(), 2);
  const double denominator = std::exp(-2.0) + std::exp(-1.0) + 1;
  EXPECT_DOUBLE_EQ(result->value, 1 + std::log(denominator));
  auto first = Read(**executor, result->gradients[0]);
  auto second = Read(**executor, result->gradients[1]);
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  EXPECT_EQ(*first, (std::vector<float>{0, 0}));
  EXPECT_EQ(*second, (std::vector<float>{
                         0, static_cast<float>(std::exp(-2.0) / denominator),
                         static_cast<float>(std::exp(-1.0) / denominator - 1),
                         static_cast<float>(1 / denominator), 0, 0}));
}

TEST(PathLossTest, StableForLargeLogitsAndMinusInfinityNonTarget) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok());
  auto logits = Upload(**executor,
                       {10000, 9999, -std::numeric_limits<float>::infinity()});
  ASSERT_TRUE(logits.ok());
  auto result = CrossEntropy({0, 0}, 3, 1)(**executor, {*logits});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_DOUBLE_EQ(result->value, 1 + std::log1p(std::exp(-1.0)));
  auto seed = Read(**executor, result->gradients[0]);
  ASSERT_TRUE(seed.ok());
  EXPECT_FLOAT_EQ((*seed)[0], 1.0 / (1.0 + std::exp(-1.0)));
  EXPECT_FLOAT_EQ((*seed)[1], -(*seed)[0]);
  EXPECT_EQ((*seed)[2], 0);
}

TEST(PathLossTest, RejectsInvalidRangesTargetsPrecisionAndNonfiniteLogits) {
  auto executor = cuda::Executor::Create();
  auto foreign = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok());
  ASSERT_TRUE(foreign.ok());
  auto logits = Upload(**executor, {1, 2, 3});
  auto foreign_logits = Upload(**foreign, {1, 2, 3});
  auto bad_size = Buffer::Allocate(**executor, 3);
  ASSERT_TRUE(logits.ok());
  ASSERT_TRUE(foreign_logits.ok());
  ASSERT_TRUE(bad_size.ok());
  EXPECT_FALSE(CrossEntropy({0, 0}, 3, 0)(**executor, {}).ok());
  EXPECT_FALSE(CrossEntropy({1, 0}, 3, 0)(**executor, {*logits}).ok());
  EXPECT_FALSE(CrossEntropy({0, 0}, 0, 0)(**executor, {*logits}).ok());
  EXPECT_FALSE(CrossEntropy({0, 0}, 3, 3)(**executor, {*logits}).ok());
  EXPECT_FALSE(CrossEntropy({0, 1}, 3, 0)(**executor, {*logits}).ok());
  EXPECT_FALSE(CrossEntropy({0, std::numeric_limits<size_t>::max()}, 3, 0)(
                   **executor, {*logits})
                   .ok());
  EXPECT_FALSE(CrossEntropy({0, 0}, 1, 0)(**executor, {*bad_size}).ok());
  EXPECT_FALSE(CrossEntropy({0, 0}, 3, 0)(**executor, {*foreign_logits}).ok());
  for (float value : {std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
    auto nonfinite = Upload(**executor, {value, 1});
    ASSERT_TRUE(nonfinite.ok());
    EXPECT_FALSE(CrossEntropy({0, 0}, 2, 0)(**executor, {*nonfinite}).ok());
  }
}

TEST(PathUpdateTest, SelectsLossRowsAndParameterBlockAveragesExactlyOnce) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok());
  // Five total parameters, block [1,4), three rows. Ignore the huge query
  // derivative in row zero and the unrelated edge parameters in every row.
  auto jacobian = Upload(**executor, {999, 999, 999, 999, 999, 888, 1, 3, 5,
                                      888, 777, 3, 5, 7, 777});
  auto weights = Upload(**executor, {10, 20, 30});
  ASSERT_TRUE(jacobian.ok());
  ASSERT_TRUE(weights.ok());
  ASSERT_TRUE(internal::ApplyGradientDescent(**executor, *weights, *jacobian, 5,
                                             1, 1, 2, 0.5)
                  .ok());
  auto actual = Read(**executor, *weights);
  ASSERT_TRUE(actual.ok());
  EXPECT_EQ(*actual, (std::vector<float>{9, 18, 27}));
}

TEST(PathUpdateTest, RejectsInvalidDimensionsExecutorsAndNonfiniteUpdate) {
  auto executor = cuda::Executor::Create();
  auto foreign = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok());
  ASSERT_TRUE(foreign.ok());
  auto weights = Upload(**executor, {1, 2});
  auto jacobian = Upload(**executor, {1, 2, 3, 4});
  auto foreign_weights = Upload(**foreign, {1, 2});
  ASSERT_TRUE(weights.ok());
  ASSERT_TRUE(jacobian.ok());
  ASSERT_TRUE(foreign_weights.ok());
  EXPECT_FALSE(internal::ApplyGradientDescent(**executor, *weights, *jacobian,
                                              0, 0, 0, 2, 0.1)
                   .ok());
  EXPECT_FALSE(internal::ApplyGradientDescent(**executor, *weights, *jacobian,
                                              2, 1, 0, 2, 0.1)
                   .ok());
  EXPECT_FALSE(internal::ApplyGradientDescent(**executor, *weights, *jacobian,
                                              2, 0, 1, 2, 0.1)
                   .ok());
  EXPECT_FALSE(internal::ApplyGradientDescent(**executor, *weights, *jacobian,
                                              2, 0, 0, 0, 0.1)
                   .ok());
  EXPECT_FALSE(internal::ApplyGradientDescent(**executor, *weights, *jacobian,
                                              2, 0, 0, 2, 0)
                   .ok());
  EXPECT_FALSE(internal::ApplyGradientDescent(**executor, *foreign_weights,
                                              *jacobian, 2, 0, 0, 2, 0.1)
                   .ok());
  EXPECT_EQ(internal::ApplyGradientDescent(**executor, *weights, *jacobian, 2,
                                           0, 0, 2, 1e300)
                .code(),
            absl::StatusCode::kOutOfRange);
}

}  // namespace
}  // namespace pluto::llm::path_kernel
