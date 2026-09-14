#include "src/cuda/thread_pool.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "src/util/status_macros.h"

namespace pluto::cuda {
namespace {

// Detect direct reentrancy before taking call_mutex_: the outer caller holds
// that mutex while waiting for this worker. Waiting again would deadlock.
thread_local const ThreadPool* current_pool = nullptr;

}  // namespace

ThreadPool::ThreadPool(int num_threads, int device)
    : num_threads_(num_threads),
      device_(device),
      remaining_(num_threads),
      statuses_(num_threads) {}

absl::StatusOr<std::unique_ptr<ThreadPool>> ThreadPool::Create(
    int num_threads) {
  if (num_threads == -1) {
    const unsigned int cores =
        std::max(1u, std::thread::hardware_concurrency());
    if (cores > static_cast<unsigned int>(std::numeric_limits<int>::max()))
      return absl::InvalidArgumentError("CPU count exceeds ThreadPool limit");
    num_threads = static_cast<int>(cores);
  }
  if (num_threads <= 0)
    return absl::InvalidArgumentError("thread count must be positive or -1");
  int device;
  RETURN_IF_ERROR(CudaStatus(cudaGetDevice(&device), "cudaGetDevice"));
  auto pool = absl::WrapUnique(new ThreadPool(num_threads, device));
  pool->threads_.reserve(num_threads);
  for (int index = 0; index < num_threads; ++index)
    pool->threads_.emplace_back(
        [pointer = pool.get(), index] { pointer->Worker(index); });

  // A failed worker still participates in the startup barrier. Destruction on
  // failure signals and joins the successful workers before releasing state.
  {
    std::unique_lock lock(pool->mutex_);
    pool->finished_.wait(lock, [&] { return pool->remaining_ == 0; });
    for (const auto& status : pool->statuses_)
      RETURN_IF_ERROR(status);
  }
  return pool;
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard lock(mutex_);
    stopping_ = true;
  }
  work_available_.notify_all();
  for (auto& thread : threads_)
    thread.join();
}

absl::Status ThreadPool::Run(
    absl::FunctionRef<absl::Status(Executor&)> function) {
  if (current_pool == this)
    return absl::FailedPreconditionError(
        "ParallelFor cannot recursively use the same ThreadPool");
  std::lock_guard call_lock(call_mutex_);
  std::unique_lock lock(mutex_);
  function_ = function;
  remaining_ = num_threads_;
  ++generation_;
  work_available_.notify_all();
  finished_.wait(lock, [&] { return remaining_ == 0; });
  function_.reset();
  for (const auto& status : statuses_)
    RETURN_IF_ERROR(status);
  return absl::OkStatus();
}

void ThreadPool::Worker(int index) {
  current_pool = this;
  // CUDA's current device is thread-local and new CPU threads do not inherit
  // the caller's selection. Select it explicitly before creating any streams.
  absl::Status status = CudaStatus(cudaSetDevice(device_), "cudaSetDevice");
  std::unique_ptr<Executor> executor;
  if (status.ok()) {
    auto created = Executor::Create();
    status = created.status();
    if (created.ok())
      executor = std::move(*created);
  }
  {
    std::lock_guard lock(mutex_);
    statuses_[index] = status;
    --remaining_;
    finished_.notify_one();
  }
  if (!status.ok())
    return;

  uint64_t observed_generation = 0;
  for (;;) {
    std::unique_lock lock(mutex_);
    work_available_.wait(
        lock, [&] { return stopping_ || generation_ != observed_generation; });
    if (stopping_)
      break;
    observed_generation = generation_;
    const auto function = *function_;
    lock.unlock();

    status = CudaStatus(cudaSetDevice(device_), "cudaSetDevice");
    if (status.ok())
      status = function(*executor);
    // Even an error-returning callback may have queued GPU work. Drain it
    // before publishing completion or accepting another invocation. Restore
    // the device in case the callback temporarily selected a different one.
    const auto selected = CudaStatus(cudaSetDevice(device_), "cudaSetDevice");
    status.Update(selected);
    if (selected.ok())
      status.Update(executor->Synchronize());

    lock.lock();
    statuses_[index] = std::move(status);
    --remaining_;
    finished_.notify_one();
  }
  // The local Executor drains frees and releases its streams/pool here, on
  // the same worker/device that created them.
}

}  // namespace pluto::cuda
