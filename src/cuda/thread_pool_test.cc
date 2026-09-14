#include "src/cuda/thread_pool.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"

namespace pluto::cuda {
namespace {

using namespace std::chrono_literals;

TEST(ThreadPoolTest, RejectsInvalidWorkerCounts) {
  EXPECT_TRUE(absl::IsInvalidArgument(ThreadPool::Create(0).status()));
  EXPECT_TRUE(absl::IsInvalidArgument(ThreadPool::Create(-2).status()));
}

TEST(ThreadPoolTest, DefaultCreatesOneWorkerPerReportedCore) {
  auto pool = ThreadPool::Create();
  ASSERT_TRUE(pool.ok()) << pool.status();
  EXPECT_EQ(static_cast<unsigned int>((*pool)->size()),
            std::max(1u, std::thread::hardware_concurrency()));
  std::atomic<int> count = 0;
  EXPECT_TRUE((*pool)->ParallelFor([&](Executor&) { ++count; }).ok());
  EXPECT_EQ(count, (*pool)->size());
}

TEST(ThreadPoolTest, SingleWorkerCanRunRepeatedlyAndBeDestroyedUnused) {
  {
    auto unused = ThreadPool::Create(1);
    ASSERT_TRUE(unused.ok()) << unused.status();
  }
  auto pool = ThreadPool::Create(1);
  ASSERT_TRUE(pool.ok()) << pool.status();
  EXPECT_EQ((*pool)->size(), 1);
  int count = 0;
  for (int iteration = 0; iteration < 5; ++iteration)
    ASSERT_TRUE((*pool)->ParallelFor([&](Executor&) { ++count; }).ok());
  EXPECT_EQ(count, 5);
}

TEST(ThreadPoolTest, WorkersRunConcurrentlyAndRetainTheirOwnExecutors) {
  constexpr int kWorkers = 3;
  auto pool = ThreadPool::Create(kWorkers);
  ASSERT_TRUE(pool.ok()) << pool.status();
  int creating_device = -1;
  ASSERT_EQ(cudaGetDevice(&creating_device), cudaSuccess);

  std::mutex mutex;
  std::condition_variable condition;
  int arrived = 0;
  std::map<std::thread::id, std::pair<Executor*, cudaStream_t>> identities;
  const auto caller_thread = std::this_thread::get_id();
  absl::Status status = (*pool)->ParallelFor([&](Executor& executor) {
    int device = -1;
    EXPECT_EQ(cudaGetDevice(&device), cudaSuccess);
    EXPECT_EQ(device, creating_device);
    EXPECT_NE(std::this_thread::get_id(), caller_thread);
    EXPECT_NE(executor.stream(), nullptr);
    EXPECT_NE(executor.stream(), cudaStreamLegacy);
    EXPECT_NE(executor.stream(), cudaStreamPerThread);
    std::unique_lock<std::mutex> lock(mutex);
    EXPECT_TRUE(identities
                    .emplace(std::this_thread::get_id(),
                             std::make_pair(&executor, executor.stream()))
                    .second);
    ++arrived;
    condition.notify_all();
    // A serial implementation must fail, rather than permanently deadlock at
    // the test barrier. Every worker releases the mutex while waiting.
    EXPECT_TRUE(
        condition.wait_for(lock, 10s, [&] { return arrived == kWorkers; }));
  });
  ASSERT_TRUE(status.ok()) << status;
  ASSERT_EQ(identities.size(), static_cast<size_t>(kWorkers));
  for (auto first = identities.begin(); first != identities.end(); ++first)
    for (auto second = identities.begin(); second != first; ++second) {
      EXPECT_NE(first->second.first, second->second.first);
      EXPECT_NE(first->second.second, second->second.second);
    }

  for (int iteration = 0; iteration < 3; ++iteration) {
    std::atomic<int> count = 0;
    status = (*pool)->ParallelFor([&](Executor& executor) {
      const auto found = identities.find(std::this_thread::get_id());
      ASSERT_NE(found, identities.end());
      EXPECT_EQ(found->second.first, &executor);
      EXPECT_EQ(found->second.second, executor.stream());
      ++count;
    });
    ASSERT_TRUE(status.ok()) << status;
    EXPECT_EQ(count, kWorkers);
  }
}

TEST(ThreadPoolTest, CompletesDeviceTransfersBeforeReturning) {
  constexpr int kWorkers = 3;
  constexpr size_t kBytes = 1024 * 1024;
  auto pool = ThreadPool::Create(kWorkers);
  ASSERT_TRUE(pool.ok()) << pool.status();
  // These arrays are destroyed before the pool: their stream-ordered frees
  // still need the worker-owned executors to exist.
  std::vector<PageLockedHostArray<uint8_t>> outputs(kWorkers);
  std::atomic<int> next = 0;
  const absl::Status status =
      (*pool)->ParallelFor([&](Executor& executor) -> absl::Status {
        const int index = next++;
        auto device = Buffer::Allocate(executor, kBytes);
        if (!device.ok())
          return device.status();
        auto output = PageLockedHostArray<uint8_t>::Allocate(executor, kBytes);
        if (!output.ok())
          return output.status();
        outputs[index] = std::move(*output);
        absl::Status status =
            CudaStatus(cudaMemsetAsync(device->data(), index + 1, kBytes,
                                       executor.stream()),
                       "cudaMemsetAsync");
        if (!status.ok())
          return status;
        // device's destruction queues its free after the copy. No explicit
        // synchronization is needed inside the callback or in the reader.
        return CudaStatus(
            cudaMemcpyAsync(outputs[index].data(), device->data(), kBytes,
                            cudaMemcpyDeviceToHost, executor.stream()),
            "cudaMemcpyAsync");
      });
  ASSERT_TRUE(status.ok()) << status;
  ASSERT_EQ(next, kWorkers);
  for (int index = 0; index < kWorkers; ++index) {
    EXPECT_EQ(cudaStreamQuery(outputs[index].executor().stream()), cudaSuccess);
    EXPECT_TRUE(
        std::all_of(outputs[index].begin(), outputs[index].end(),
                    [index](uint8_t value) { return value == index + 1; }));
  }
}

// A bounded CUDA host callback holds queued GPU work open until the test
// releases it. It never calls CUDA, which is forbidden from host callbacks.
struct CompletionGate {
  static void CUDART_CB Wait(void* opaque) {
    auto& gate = *static_cast<CompletionGate*>(opaque);
    std::unique_lock<std::mutex> lock(gate.mutex);
    if (!gate.condition.wait_for(lock, 10s, [&] { return gate.released; }))
      gate.timed_out = true;
    ++gate.completed;
    gate.condition.notify_all();
  }

  std::mutex mutex;
  std::condition_variable condition;
  bool released = false;
  bool returned = false;
  bool timed_out = false;
  int queued = 0;
  int completed = 0;
};

TEST(ThreadPoolTest, CallbackErrorDoesNotSkipWorkersOrPendingGpuWork) {
  constexpr int kWorkers = 2;
  auto pool = ThreadPool::Create(kWorkers);
  ASSERT_TRUE(pool.ok()) << pool.status();
  CompletionGate gate;
  std::atomic<int> next = 0;
  std::vector<Executor*> executors(kWorkers);
  absl::Status result;
  std::thread caller([&] {
    result = (*pool)->ParallelFor([&](Executor& executor) -> absl::Status {
      const int index = next++;
      executors[index] = &executor;
      const absl::Status queued = CudaStatus(
          cudaLaunchHostFunc(executor.stream(), CompletionGate::Wait, &gate),
          "cudaLaunchHostFunc");
      {
        std::lock_guard<std::mutex> lock(gate.mutex);
        ++gate.queued;
        gate.condition.notify_all();
      }
      if (!queued.ok())
        return queued;
      return index == 0 ? absl::InvalidArgumentError("test callback failure")
                        : absl::OkStatus();
    });
    std::lock_guard<std::mutex> lock(gate.mutex);
    gate.returned = true;
    gate.condition.notify_all();
  });
  {
    std::unique_lock<std::mutex> lock(gate.mutex);
    EXPECT_TRUE(gate.condition.wait_for(
        lock, 10s, [&] { return gate.queued == kWorkers; }));
    // CPU callbacks have returned, but their streams remain blocked. Error
    // propagation must not allow ParallelFor to return with that work live.
    EXPECT_FALSE(
        gate.condition.wait_for(lock, 50ms, [&] { return gate.returned; }));
    gate.released = true;
    gate.condition.notify_all();
  }
  caller.join();
  EXPECT_EQ(result, absl::InvalidArgumentError("test callback failure"));
  EXPECT_EQ(next, kWorkers);
  // Also drain explicitly if an implementation regresses: a failing test
  // must not destroy gate while an asynchronous host callback still uses it.
  for (Executor* executor : executors) {
    if (executor == nullptr)
      continue;
    EXPECT_TRUE(executor->Synchronize().ok());
  }
  EXPECT_EQ(gate.completed, kWorkers);
  EXPECT_FALSE(gate.timed_out);
  std::atomic<int> rerun = 0;
  EXPECT_TRUE((*pool)->ParallelFor([&](Executor&) { ++rerun; }).ok());
  EXPECT_EQ(rerun, kWorkers);
}

TEST(ThreadPoolTest, RejectsSamePoolRecursionAndRemainsUsable) {
  auto pool = ThreadPool::Create(2);
  ASSERT_TRUE(pool.ok()) << pool.status();
  std::atomic<int> inner_calls = 0;
  const absl::Status status = (*pool)->ParallelFor([&](Executor&) {
    const auto nested = (*pool)->ParallelFor([&](Executor&) { ++inner_calls; });
    EXPECT_TRUE(absl::IsFailedPrecondition(nested)) << nested;
  });
  EXPECT_TRUE(status.ok()) << status;
  EXPECT_EQ(inner_calls, 0);
  EXPECT_TRUE((*pool)->ParallelFor([](Executor&) {}).ok());
}

TEST(ThreadPoolTest, SerializesConcurrentCallersWithoutLosingDispatches) {
  constexpr int kWorkers = 3;
  auto pool = ThreadPool::Create(kWorkers);
  ASSERT_TRUE(pool.ok()) << pool.status();
  std::mutex mutex;
  std::condition_variable condition;
  int ready = 0;
  std::vector<int> runs;
  auto invoke = [&](int run) {
    {
      std::unique_lock<std::mutex> lock(mutex);
      ++ready;
      condition.notify_all();
      EXPECT_TRUE(condition.wait_for(lock, 10s, [&] { return ready == 2; }));
    }
    const absl::Status status = (*pool)->ParallelFor([&](Executor&) {
      std::lock_guard<std::mutex> lock(mutex);
      runs.push_back(run);
    });
    EXPECT_TRUE(status.ok()) << status;
  };
  std::thread first(invoke, 1);
  std::thread second(invoke, 2);
  first.join();
  second.join();
  ASSERT_EQ(runs.size(), static_cast<size_t>(2 * kWorkers));
  EXPECT_NE(runs.front(), runs.back());
  for (int index = 0; index < kWorkers; ++index) {
    EXPECT_EQ(runs[index], runs.front());
    EXPECT_EQ(runs[index + kWorkers], runs.back());
  }
}

}  // namespace
}  // namespace pluto::cuda
