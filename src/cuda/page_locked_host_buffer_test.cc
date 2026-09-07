#include <cuda_runtime.h>

#include <cstddef>
#include <limits>

#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"

namespace pluto::cuda {
namespace {

TEST(PageLockedHostBufferTest, SharesPinnedAllocation) {
  auto original = PageLockedHostBuffer::Allocate(64);
  ASSERT_TRUE(original.ok()) << original.status();
  ASSERT_NE(original->data(), nullptr);
  EXPECT_EQ(original->size_bytes(), 64);

  cudaPointerAttributes attributes{};
  ASSERT_EQ(cudaPointerGetAttributes(&attributes, original->data()),
            cudaSuccess);
  EXPECT_EQ(attributes.type, cudaMemoryTypeHost);

  PageLockedHostBuffer copy = *original;
  original = PageLockedHostBuffer();
  EXPECT_NE(copy.data(), nullptr);
  EXPECT_EQ(copy.size_bytes(), 64);
}

TEST(PageLockedHostArrayTest, ProvidesTypedPinnedStorageForAsyncCopies) {
  auto executor = Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  auto input = PageLockedHostArray<int>::Allocate(4);
  auto output = PageLockedHostArray<int>::Allocate(4);
  auto device = Buffer::Allocate(**executor, 4 * sizeof(int));
  ASSERT_TRUE(input.ok()) << input.status();
  ASSERT_TRUE(output.ok()) << output.status();
  ASSERT_TRUE(device.ok()) << device.status();
  EXPECT_EQ(input->data(), input->buffer().data());

  for (size_t index = 0; index < input->size(); ++index) {
    (*input)[index] = static_cast<int>(index + 10);
  }
  ASSERT_EQ(cudaMemcpyAsync(device->data(), input->data(), input->size_bytes(),
                            cudaMemcpyHostToDevice, (*executor)->stream()),
            cudaSuccess);
  ASSERT_EQ(
      cudaMemcpyAsync(output->data(), device->data(), output->size_bytes(),
                      cudaMemcpyDeviceToHost, (*executor)->stream()),
      cudaSuccess);
  ASSERT_TRUE((*executor)->Synchronize().ok());
  EXPECT_EQ(output->span(), input->span());
}

TEST(PageLockedHostArrayTest, RepresentsEmptyArrayWithoutAllocation) {
  auto array = PageLockedHostArray<int>::Allocate(0);
  ASSERT_TRUE(array.ok()) << array.status();
  EXPECT_TRUE(array->empty());
  EXPECT_EQ(array->size(), 0);
  EXPECT_EQ(array->size_bytes(), 0);
  EXPECT_EQ(array->data(), nullptr);
  EXPECT_EQ(array->begin(), array->end());
}

TEST(PageLockedHostArrayTest, RejectsByteSizeOverflow) {
  const auto array = PageLockedHostArray<int>::Allocate(
      std::numeric_limits<size_t>::max() / sizeof(int) + 1);
  EXPECT_FALSE(array.ok());
}

}  // namespace
}  // namespace pluto::cuda
