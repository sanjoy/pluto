#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <thread>
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

class PinnedMemoryTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }

  std::unique_ptr<Executor> executor_;
};

using PageLockedHostBufferTest = PinnedMemoryTest;
using PageLockedHostArrayTest = PinnedMemoryTest;

// Holding the compute stream proves that allocating/filling/destroying host
// storage does not wait for it. A bounded callback prevents a broken allocator
// from hanging the test forever. The callback itself makes no CUDA calls.
class ComputeGate {
 public:
  explicit ComputeGate(Executor& executor) : executor_(executor) {}
  ~ComputeGate() {
    Open();
    EXPECT_TRUE(executor_.Synchronize().ok());
  }

  cudaError_t Start() {
    return cudaLaunchHostFunc(executor_.stream(), Wait, this);
  }
  void Open() { open_.store(true); }
  bool expired() const { return expired_.load(); }

 private:
  static void CUDART_CB Wait(void* opaque) {
    auto& gate = *static_cast<ComputeGate*>(opaque);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!gate.open_.load()) {
      if (std::chrono::steady_clock::now() > deadline) {
        gate.expired_.store(true);
        break;
      }
      std::this_thread::yield();
    }
  }

  Executor& executor_;
  std::atomic<bool> open_{false};
  std::atomic<bool> expired_{false};
};

TEST_F(PageLockedHostBufferTest, ZeroSizedBufferRetainsExecutorWithoutStorage) {
  auto buffer = PageLockedHostBuffer::Allocate(*executor_, 0);
  ASSERT_TRUE(buffer.ok()) << buffer.status();
  EXPECT_EQ(buffer->data(), nullptr);
  EXPECT_EQ(std::as_const(*buffer).data(), nullptr);
  EXPECT_EQ(buffer->size_bytes(), 0);
  EXPECT_EQ(&buffer->executor(), executor_.get());
  uint64_t used = 1;
  ASSERT_EQ(cudaMemPoolGetAttribute(executor_->host_memory_pool(),
                                    cudaMemPoolAttrUsedMemCurrent, &used),
            cudaSuccess);
  EXPECT_EQ(used, 0);
}

TEST_F(PageLockedHostArrayTest, DefaultConstructedArrayIsEmpty) {
  PageLockedHostArray<int> array;
  EXPECT_TRUE(array.empty());
  EXPECT_EQ(array.size_bytes(), 0);
  EXPECT_EQ(array.data(), nullptr);
  EXPECT_EQ(std::as_const(array).data(), nullptr);
  EXPECT_EQ(array.begin(), array.end());
}

TEST_F(PageLockedHostBufferTest, MovedFromBufferIsEmpty) {
  auto allocated = PageLockedHostBuffer::Allocate(*executor_, 64);
  ASSERT_TRUE(allocated.ok()) << allocated.status();
  void* data = allocated->data();

  PageLockedHostBuffer moved(std::move(*allocated));
  EXPECT_EQ(allocated->data(), nullptr);
  EXPECT_EQ(std::as_const(*allocated).data(), nullptr);
  EXPECT_EQ(allocated->size_bytes(), 0);
  EXPECT_EQ(moved.data(), data);
  EXPECT_EQ(moved.size_bytes(), 64);
  EXPECT_EQ(&moved.executor(), executor_.get());

  auto assigned = PageLockedHostBuffer::Allocate(*executor_, 0);
  ASSERT_TRUE(assigned.ok()) << assigned.status();
  *assigned = std::move(moved);
  EXPECT_EQ(moved.data(), nullptr);
  EXPECT_EQ(std::as_const(moved).data(), nullptr);
  EXPECT_EQ(moved.size_bytes(), 0);
  EXPECT_EQ(assigned->data(), data);
  EXPECT_EQ(assigned->size_bytes(), 64);
  EXPECT_EQ(&assigned->executor(), executor_.get());
}

TEST_F(PageLockedHostBufferTest, SharesPinnedAllocation) {
  auto original = PageLockedHostBuffer::Allocate(*executor_, 64);
  ASSERT_TRUE(original.ok()) << original.status();
  ASSERT_NE(original->data(), nullptr);
  EXPECT_EQ(original->size_bytes(), 64);

  cudaPointerAttributes attributes{};
  ASSERT_EQ(cudaPointerGetAttributes(&attributes, original->data()),
            cudaSuccess);
  EXPECT_EQ(attributes.type, cudaMemoryTypeHost);

  PageLockedHostBuffer copy = *original;
  original = PageLockedHostBuffer::Allocate(*executor_, 0);
  ASSERT_TRUE(original.ok()) << original.status();
  EXPECT_NE(copy.data(), nullptr);
  EXPECT_EQ(copy.size_bytes(), 64);
  EXPECT_EQ(&copy.executor(), executor_.get());
}

TEST_F(PageLockedHostArrayTest, ProvidesTypedPinnedStorageForAsyncCopies) {
  auto input = PageLockedHostArray<int>::Allocate(*executor_, 4);
  auto output = PageLockedHostArray<int>::Allocate(*executor_, 4);
  auto device = Buffer::Allocate(*executor_, 4 * sizeof(int));
  ASSERT_TRUE(input.ok()) << input.status();
  ASSERT_TRUE(output.ok()) << output.status();
  ASSERT_TRUE(device.ok()) << device.status();
  EXPECT_EQ(input->data(), input->buffer().data());

  for (size_t index = 0; index < input->size(); ++index)
    (*input)[index] = static_cast<int>(index + 10);
  ASSERT_EQ(cudaMemcpyAsync(device->data(), input->data(), input->size_bytes(),
                            cudaMemcpyHostToDevice, executor_->stream()),
            cudaSuccess);
  ASSERT_EQ(
      cudaMemcpyAsync(output->data(), device->data(), output->size_bytes(),
                      cudaMemcpyDeviceToHost, executor_->stream()),
      cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());
  EXPECT_EQ(output->span(), input->span());
}

TEST_F(PageLockedHostArrayTest, CopyFromIsImmediatelyCpuAccessible) {
  const int values[] = {2, 7, 1, 8};
  auto array = PageLockedHostArray<int>::CopyFrom(*executor_, values);
  ASSERT_TRUE(array.ok()) << array.status();
  EXPECT_EQ(array->span(), absl::MakeConstSpan(values));
  EXPECT_EQ(&array->executor(), executor_.get());
}

TEST_F(PageLockedHostArrayTest, RepresentsEmptyArrayWithoutAllocation) {
  auto array = PageLockedHostArray<int>::Allocate(*executor_, 0);
  ASSERT_TRUE(array.ok()) << array.status();
  EXPECT_TRUE(array->empty());
  EXPECT_EQ(array->size(), 0);
  EXPECT_EQ(array->size_bytes(), 0);
  EXPECT_EQ(array->data(), nullptr);
  EXPECT_EQ(array->begin(), array->end());
  EXPECT_EQ(&array->executor(), executor_.get());
}

TEST_F(PageLockedHostArrayTest, RejectsByteSizeOverflow) {
  const auto array = PageLockedHostArray<int>::Allocate(
      *executor_, std::numeric_limits<size_t>::max() / sizeof(int) + 1);
  EXPECT_FALSE(array.ok());
}

TEST_F(PageLockedHostBufferTest, PendingUploadsOutliveTheirLastHostOwner) {
  constexpr int kCopies = 128;
  auto device = Buffer::Allocate(*executor_, kCopies * sizeof(int));
  auto result = PageLockedHostArray<int>::Allocate(*executor_, kCopies);
  ASSERT_TRUE(device.ok()) << device.status();
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_TRUE(executor_->Synchronize().ok());

  ComputeGate gate(*executor_);
  ASSERT_EQ(gate.Start(), cudaSuccess);
  auto* output = static_cast<int*>(device->data());
  for (int index = 0; index < kCopies; ++index) {
    // Allocate and write while compute is held. Destruction queues a free
    // after this copy; the next allocation must not reuse its pending source.
    auto input = PageLockedHostArray<int>::Allocate(*executor_, 1024);
    ASSERT_TRUE(input.ok()) << input.status();
    (*input)[0] = index + 100;
    ASSERT_EQ(cudaMemcpyAsync(output + index, input->data(), sizeof(int),
                              cudaMemcpyHostToDevice, executor_->stream()),
              cudaSuccess);
  }
  EXPECT_FALSE(gate.expired());
  EXPECT_EQ(cudaStreamQuery(executor_->stream()), cudaErrorNotReady);
  gate.Open();
  ASSERT_EQ(
      cudaMemcpyAsync(result->data(), device->data(), result->size_bytes(),
                      cudaMemcpyDeviceToHost, executor_->stream()),
      cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());
  for (int index = 0; index < kCopies; ++index)
    EXPECT_EQ((*result)[index], index + 100);
}

void CUDART_CB MarkCompleted(void* opaque) {
  static_cast<std::atomic<bool>*>(opaque)->store(true);
}

TEST_F(PageLockedHostBufferTest,
       RecyclesCompletedFreesWithoutComputeSynchronize) {
  // Observe completion using a host callback, not a CUDA synchronization or
  // query: opportunistic reuse must work even when the host has not told CUDA
  // that it observed compute completion. With opportunistic reuse disabled,
  // the reserved pool grows by each round's allocation volume on this path.
  uint64_t warm_reserved = 0;
  std::atomic<bool> completed{false};
  for (int round = 0; round < 64; ++round) {
    for (int index = 0; index < 32; ++index) {
      auto buffer = PageLockedHostArray<int>::Allocate(*executor_, 16384);
      ASSERT_TRUE(buffer.ok()) << buffer.status();
      (*buffer)[0] = round + index;
    }
    completed.store(false);
    ASSERT_EQ(
        cudaLaunchHostFunc(executor_->stream(), MarkCompleted, &completed),
        cudaSuccess);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!completed.load() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    // Synchronize on failure so no callback retains a pointer to destroyed
    // stack state, while leaving the successful path genuinely unsynchronized.
    if (!completed.load()) {
      EXPECT_TRUE(executor_->Synchronize().ok());
      FAIL() << "compute callback did not complete";
    }
    uint64_t reserved;
    ASSERT_EQ(
        cudaMemPoolGetAttribute(executor_->host_memory_pool(),
                                cudaMemPoolAttrReservedMemCurrent, &reserved),
        cudaSuccess);
    if (round == 0)
      warm_reserved = reserved;
    EXPECT_LE(reserved, warm_reserved + 32 * 1024 * 1024);
  }
  ASSERT_TRUE(executor_->Synchronize().ok());
  uint64_t used = 1;
  ASSERT_EQ(cudaMemPoolGetAttribute(executor_->host_memory_pool(),
                                    cudaMemPoolAttrUsedMemCurrent, &used),
            cudaSuccess);
  EXPECT_EQ(used, 0);
}

}  // namespace
}  // namespace pluto::cuda
