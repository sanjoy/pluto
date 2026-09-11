#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"

namespace pluto::cuda {
namespace {

// Construction must go through the factory outside the typed wrapper.
static_assert(!std::is_default_constructible_v<PageLockedHostBuffer>);

TEST(PageLockedHostBufferTest, ZeroSizedBufferIsEmpty) {
  auto buffer = PageLockedHostBuffer::Allocate(0);
  ASSERT_TRUE(buffer.ok()) << buffer.status();
  EXPECT_EQ(buffer->data(), nullptr);
  EXPECT_EQ(std::as_const(*buffer).data(), nullptr);
  EXPECT_EQ(buffer->size_bytes(), 0);
}

TEST(PageLockedHostArrayTest, DefaultConstructedArrayIsEmpty) {
  PageLockedHostArray<int> array;
  EXPECT_TRUE(array.empty());
  EXPECT_EQ(array.size_bytes(), 0);
  EXPECT_EQ(array.data(), nullptr);
  EXPECT_EQ(std::as_const(array).data(), nullptr);
  EXPECT_EQ(array.begin(), array.end());
}

TEST(PageLockedHostBufferTest, MovedFromBufferIsEmpty) {
  auto allocated = PageLockedHostBuffer::Allocate(64);
  ASSERT_TRUE(allocated.ok()) << allocated.status();
  void* data = allocated->data();

  PageLockedHostBuffer moved(std::move(*allocated));
  EXPECT_EQ(allocated->data(), nullptr);
  EXPECT_EQ(std::as_const(*allocated).data(), nullptr);
  EXPECT_EQ(allocated->size_bytes(), 0);
  EXPECT_EQ(moved.data(), data);
  EXPECT_EQ(moved.size_bytes(), 64);

  auto assigned = PageLockedHostBuffer::Allocate(0);
  ASSERT_TRUE(assigned.ok()) << assigned.status();
  *assigned = std::move(moved);
  EXPECT_EQ(moved.data(), nullptr);
  EXPECT_EQ(std::as_const(moved).data(), nullptr);
  EXPECT_EQ(moved.size_bytes(), 0);
  EXPECT_EQ(assigned->data(), data);
  EXPECT_EQ(assigned->size_bytes(), 64);
}

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
  original = PageLockedHostBuffer::Allocate(0);
  ASSERT_TRUE(original.ok()) << original.status();
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

  for (size_t index = 0; index < input->size(); ++index)
    (*input)[index] = static_cast<int>(index + 10);
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
