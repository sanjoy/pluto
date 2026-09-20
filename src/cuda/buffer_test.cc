#include "src/cuda/buffer.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>

#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"

namespace pluto::cuda {
namespace {

constexpr size_t kByteCount = 4096;

__global__ void FillBytes(uint8_t* bytes, size_t size, uint8_t value) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    bytes[index] = value;
}

class BufferTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }

  void TearDown() override {
    if (executor_ == nullptr)
      return;
    EXPECT_TRUE(executor_->Synchronize().ok());
    executor_.reset();
  }

  std::unique_ptr<Executor> executor_;
};

static_assert(std::is_copy_constructible_v<Buffer>);
static_assert(std::is_copy_assignable_v<Buffer>);
static_assert(std::is_nothrow_move_constructible_v<Buffer>);
static_assert(std::is_nothrow_move_assignable_v<Buffer>);

TEST_F(BufferTest, CopiesShareStorageUntilTheLastReferenceIsDestroyed) {
  std::optional<Buffer> survivor;
  void* address = nullptr;
  {
    auto original = Buffer::Allocate(*executor_, kByteCount);
    ASSERT_TRUE(original.ok()) << original.status();
    ASSERT_NE(original->data(), nullptr);
    EXPECT_EQ(original->size_bytes(), kByteCount);
    EXPECT_EQ(&original->executor(), executor_.get());

    Buffer copy = *original;
    EXPECT_EQ(copy.data(), original->data());
    EXPECT_EQ(copy.size_bytes(), original->size_bytes());
    EXPECT_EQ(&copy.executor(), &original->executor());
    address = copy.data();
    survivor.emplace(copy);
  }

  // The original StatusOr and its local copy are gone, but the shared
  // allocation remains valid through survivor.
  ASSERT_TRUE(survivor.has_value());
  ASSERT_EQ(survivor->data(), address);
  FillBytes<<<(kByteCount + 255) / 256, 256, 0, executor_->stream()>>>(
      static_cast<uint8_t*>(survivor->data()), kByteCount, 0xa5);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  auto host_bytes =
      PageLockedHostArray<uint8_t>::Allocate(*executor_, kByteCount);
  ASSERT_TRUE(host_bytes.ok()) << host_bytes.status();
  ASSERT_EQ(cudaMemcpyAsync(host_bytes->data(), survivor->data(), kByteCount,
                            cudaMemcpyDeviceToHost, executor_->stream()),
            cudaSuccess);

  // This drops the final reference. cudaFreeAsync is queued after the kernel
  // and copy above, so the host transfer must still complete correctly.
  survivor.reset();
  ASSERT_TRUE(executor_->Synchronize().ok());
  for (const uint8_t byte : *host_bytes)
    EXPECT_EQ(byte, 0xa5);
}

TEST_F(BufferTest, ZeroByteBufferRetainsItsExecutorWithoutAllocatingStorage) {
  {
    auto buffer = Buffer::Allocate(*executor_, 0);
    ASSERT_TRUE(buffer.ok()) << buffer.status();
    EXPECT_EQ(buffer->data(), nullptr);
    EXPECT_EQ(buffer->size_bytes(), 0u);
    EXPECT_EQ(&buffer->executor(), executor_.get());

    Buffer copy = *buffer;
    EXPECT_EQ(copy.data(), nullptr);
    EXPECT_EQ(&copy.executor(), executor_.get());
  }
  EXPECT_TRUE(executor_->Synchronize().ok());
}

}  // namespace
}  // namespace pluto::cuda
