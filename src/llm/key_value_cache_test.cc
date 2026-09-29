#include "src/llm/key_value_cache.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"

namespace pluto::llm {
namespace {

static_assert(!std::is_copy_constructible_v<KeyValueCache>);
static_assert(!std::is_copy_assignable_v<KeyValueCache>);

class KeyValueCacheTest : public testing::Test {
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

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(KeyValueCacheTest, OwnsSeparateBuffersWithRequestedShapeAndExecutor) {
  auto cache = KeyValueCache::Create(*executor_, 7, 2, 3);
  ASSERT_TRUE(cache.ok()) << cache.status();
  EXPECT_EQ((*cache)->capacity(), 7);
  EXPECT_EQ((*cache)->key_value_heads(), 2);
  EXPECT_EQ((*cache)->head_dim(), 3);
  EXPECT_EQ((*cache)->position(), 0);
  EXPECT_EQ(&(*cache)->executor(), executor_.get());
  EXPECT_EQ(&(*cache)->keys().executor(), executor_.get());
  EXPECT_EQ(&(*cache)->values().executor(), executor_.get());
  EXPECT_EQ((*cache)->keys().size_bytes(), 7u * 2 * 3 * sizeof(float));
  EXPECT_EQ((*cache)->values().size_bytes(), (*cache)->keys().size_bytes());
  EXPECT_NE((*cache)->keys().data(), nullptr);
  EXPECT_NE((*cache)->values().data(), nullptr);
  EXPECT_NE((*cache)->keys().data(), (*cache)->values().data());

  auto independent = KeyValueCache::Create(*executor_, 7, 2, 3);
  ASSERT_TRUE(independent.ok()) << independent.status();
  EXPECT_NE((*independent)->keys().data(), (*cache)->keys().data());
  EXPECT_NE((*independent)->values().data(), (*cache)->values().data());
  ASSERT_TRUE((*cache)->Advance().ok());
  EXPECT_EQ((*independent)->position(), 0);
}

TEST_F(KeyValueCacheTest, AdvancesUpToCapacityAndCanStartAnotherSequence) {
  auto cache = KeyValueCache::Create(*executor_, 3, 1, 1);
  ASSERT_TRUE(cache.ok()) << cache.status();
  for (int next = 1; next <= 3; ++next) {
    ASSERT_TRUE((*cache)->Advance().ok());
    EXPECT_EQ((*cache)->position(), next);
  }
  EXPECT_EQ((*cache)->Advance().code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ((*cache)->Advance().code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ((*cache)->position(), 3);
  (*cache)->Reset();
  EXPECT_EQ((*cache)->position(), 0);
  (*cache)->Reset();
  ASSERT_TRUE((*cache)->Advance().ok());
  EXPECT_EQ((*cache)->position(), 1);
}

TEST_F(KeyValueCacheTest, ResetPreservesAllocationsAndDoesNotClearBytes) {
  auto cache = KeyValueCache::Create(*executor_, 3, 2, 5);
  ASSERT_TRUE(cache.ok()) << cache.status();
  const size_t bytes = (*cache)->keys().size_bytes();
  void* keys = (*cache)->keys().data();
  void* values = (*cache)->values().data();
  ASSERT_EQ(cudaMemsetAsync(keys, 0x12, bytes, executor_->stream()),
            cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(values, 0x34, bytes, executor_->stream()),
            cudaSuccess);
  ASSERT_TRUE((*cache)->Advance().ok());
  (*cache)->Reset();
  EXPECT_EQ((*cache)->position(), 0);
  EXPECT_EQ((*cache)->keys().data(), keys);
  EXPECT_EQ((*cache)->values().data(), values);

  auto host =
      cuda::PageLockedHostArray<uint8_t>::Allocate(*executor_, 2 * bytes);
  ASSERT_TRUE(host.ok()) << host.status();
  ASSERT_EQ(cudaMemcpyAsync(host->data(), keys, bytes, cudaMemcpyDeviceToHost,
                            executor_->stream()),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(host->data() + bytes, values, bytes,
                            cudaMemcpyDeviceToHost, executor_->stream()),
            cudaSuccess);
  // Cache destruction queues frees after both pending copies. No explicit
  // synchronization is needed to keep the source buffers alive for the copies.
  cache->reset();
  ASSERT_TRUE(executor_->Synchronize().ok());
  for (size_t index = 0; index < bytes; ++index) {
    EXPECT_EQ((*host)[index], 0x12);
    EXPECT_EQ((*host)[bytes + index], 0x34);
  }
}

TEST_F(KeyValueCacheTest, RejectsNonpositiveDimensionsBeforeAllocation) {
  for (int invalid : {0, -1, std::numeric_limits<int>::min()}) {
    EXPECT_EQ(KeyValueCache::Create(*executor_, invalid, 2, 3).status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(KeyValueCache::Create(*executor_, 7, invalid, 3).status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(KeyValueCache::Create(*executor_, 7, 2, invalid).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST_F(KeyValueCacheTest, RejectsByteSizeOverflowBeforeAllocation) {
  constexpr int largest = std::numeric_limits<int>::max();
  auto cache = KeyValueCache::Create(*executor_, largest, largest, largest);
  EXPECT_EQ(cache.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(cache.status().message().find("overflows"),
            absl::string_view::npos);
}

}  // namespace
}  // namespace pluto::llm
