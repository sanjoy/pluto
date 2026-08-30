#include "src/cuda/executor.h"

#include <cuda_runtime_api.h>

#include <memory>

#include "gtest/gtest.h"

namespace pluto::cuda {
namespace {

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
