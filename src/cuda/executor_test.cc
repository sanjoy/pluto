#include "src/cuda/executor.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
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

TEST(ExecutorTest, TrivialStreamIsIndependentAndUsedForImmediateWaits) {
  auto executor = Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  const cudaStream_t stream = (*executor)->trivial_stream();
  EXPECT_NE(stream, nullptr);
  EXPECT_NE(stream, cudaStreamLegacy);
  EXPECT_NE(stream, cudaStreamPerThread);
  EXPECT_NE(stream, (*executor)->stream());
  unsigned int flags = 0;
  ASSERT_EQ(cudaStreamGetFlags(stream, &flags), cudaSuccess);
  EXPECT_EQ(flags, cudaStreamNonBlocking);
  ASSERT_EQ(cudaStreamQuery(stream), cudaSuccess);

  // The stream is not reserved for allocation: any short operation follows
  // the same immediate-wait contract, leaving no deferred work behind.
  int completed = 0;
  const cudaError_t queued = cudaLaunchHostFunc(
      stream, [](void* state) { *static_cast<int*>(state) = 1; }, &completed);
  // Wait even if submission failed, before the callback's stack state expires.
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(queued, cudaSuccess);
  EXPECT_EQ(completed, 1);
  EXPECT_EQ(cudaStreamQuery(stream), cudaSuccess);
}

TEST(ExecutorTest, HostPoolHasSafeCrossStreamReuseAndGpuAccess) {
  auto executor = Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  const cudaMemPool_t pool = (*executor)->host_memory_pool();
  ASSERT_NE(pool, nullptr);

  int opportunistic = 0, follow_events = 1, internal_dependencies = 1;
  ASSERT_EQ(cudaMemPoolGetAttribute(pool, cudaMemPoolReuseAllowOpportunistic,
                                    &opportunistic),
            cudaSuccess);
  ASSERT_EQ(cudaMemPoolGetAttribute(
                pool, cudaMemPoolReuseFollowEventDependencies, &follow_events),
            cudaSuccess);
  ASSERT_EQ(
      cudaMemPoolGetAttribute(pool, cudaMemPoolReuseAllowInternalDependencies,
                              &internal_dependencies),
      cudaSuccess);
  EXPECT_EQ(opportunistic, 1);
  EXPECT_EQ(follow_events, 0);
  EXPECT_EQ(internal_dependencies, 0);
  uint64_t release_threshold = 0;
  ASSERT_EQ(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                    &release_threshold),
            cudaSuccess);
  EXPECT_GT(release_threshold, 0);
  EXPECT_LE(release_threshold, 64 * 1024 * 1024);

  int device;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  cudaMemLocation location{};
  location.type = cudaMemLocationTypeDevice;
  location.id = device;
  cudaMemAccessFlags flags = cudaMemAccessFlagsProtNone;
  ASSERT_EQ(cudaMemPoolGetAccess(&flags, pool, &location), cudaSuccess);
  EXPECT_EQ(flags, cudaMemAccessFlagsProtReadWrite);
}

}  // namespace
}  // namespace pluto::cuda
