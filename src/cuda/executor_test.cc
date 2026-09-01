#include "src/cuda/executor.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>

#include "gtest/gtest.h"

namespace pluto::cuda {
namespace {

TEST(ExecutorTest, ConvertsCudaErrorsToStatus) {
  EXPECT_TRUE(CudaStatus(cudaSuccess, "successful operation").ok());

  const absl::Status status =
      CudaStatus(cudaErrorInvalidValue, "failing operation");
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(status.message().find("failing operation failed: "),
            std::size_t{0});
}

TEST(ExecutorTest, OwnsAnExplicitStreamAndSynchronizesIt) {
  auto executor = Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  EXPECT_NE((*executor)->stream(), nullptr);
  EXPECT_NE((*executor)->stream(), cudaStreamLegacy);
  EXPECT_NE((*executor)->stream(), cudaStreamPerThread);
  EXPECT_TRUE((*executor)->Synchronize().ok());
}

}  // namespace
}  // namespace pluto::cuda
