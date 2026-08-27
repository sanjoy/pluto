#include "src/gpu/buffer.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

#include "gtest/gtest.h"

namespace pluto::gpu {
namespace {

constexpr size_t kByteCount = 4096;

__global__ void FillBytes(uint8_t* bytes, size_t size, uint8_t value) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) bytes[index] = value;
}

class BufferTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
              cudaSuccess);
  }

  void TearDown() override {
    if (stream_ == nullptr) return;
    EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream_), cudaSuccess);
  }

  cudaStream_t stream_ = nullptr;
};

static_assert(std::is_copy_constructible_v<Buffer>);
static_assert(std::is_copy_assignable_v<Buffer>);
static_assert(std::is_nothrow_move_constructible_v<Buffer>);
static_assert(std::is_nothrow_move_assignable_v<Buffer>);

TEST(BufferDeathTest, RejectsEveryDefaultCudaStreamHandle) {
  EXPECT_DEATH((void)Buffer::Allocate(32, nullptr),
               "explicitly created CUDA stream");
  EXPECT_DEATH((void)Buffer::Allocate(32, cudaStreamLegacy),
               "explicitly created CUDA stream");
  EXPECT_DEATH((void)Buffer::Allocate(32, cudaStreamPerThread),
               "explicitly created CUDA stream");
}

TEST_F(BufferTest, CopiesShareStorageUntilTheLastReferenceIsDestroyed) {
  std::optional<Buffer> survivor;
  void* address = nullptr;
  {
    auto original = Buffer::Allocate(kByteCount, stream_);
    ASSERT_TRUE(original.ok()) << original.status();
    ASSERT_NE(original->data(), nullptr);
    EXPECT_EQ(original->size_bytes(), kByteCount);
    EXPECT_EQ(original->stream(), stream_);

    Buffer copy = *original;
    EXPECT_EQ(copy.data(), original->data());
    EXPECT_EQ(copy.size_bytes(), original->size_bytes());
    EXPECT_EQ(copy.stream(), original->stream());
    address = copy.data();
    survivor.emplace(copy);
  }

  // The original StatusOr and its local copy are gone, but the shared
  // allocation remains valid through survivor.
  ASSERT_TRUE(survivor.has_value());
  ASSERT_EQ(survivor->data(), address);
  FillBytes<<<(kByteCount + 255) / 256, 256, 0, stream_>>>(
      static_cast<uint8_t*>(survivor->data()), kByteCount, 0xa5);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::array<uint8_t, kByteCount> host_bytes{};
  ASSERT_EQ(cudaMemcpyAsync(host_bytes.data(), survivor->data(), kByteCount,
                            cudaMemcpyDeviceToHost, stream_),
            cudaSuccess);

  // This drops the final reference. cudaFreeAsync is queued after the kernel
  // and copy above, so the host transfer must still complete correctly.
  survivor.reset();
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  for (const uint8_t byte : host_bytes) EXPECT_EQ(byte, 0xa5);
}

TEST_F(BufferTest, ZeroByteBufferRetainsItsStreamWithoutAllocatingStorage) {
  {
    auto buffer = Buffer::Allocate(0, stream_);
    ASSERT_TRUE(buffer.ok()) << buffer.status();
    EXPECT_EQ(buffer->data(), nullptr);
    EXPECT_EQ(buffer->size_bytes(), 0u);
    EXPECT_EQ(buffer->stream(), stream_);

    Buffer copy = *buffer;
    EXPECT_EQ(copy.data(), nullptr);
    EXPECT_EQ(copy.stream(), stream_);
  }
  EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

}  // namespace
}  // namespace pluto::gpu
