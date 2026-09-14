#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <type_traits>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/cuda/executor.h"

namespace pluto::cuda {

// A fixed set of persistent CPU workers, each owning one CUDA Executor.
// Executors are created, used, and destroyed on their worker, on the CUDA
// device selected by the thread calling Create(). Workers share that GPU;
// this is not a multi-GPU scheduler.
class ThreadPool final {
 public:
  // -1 selects one worker per logical CPU reported by hardware_concurrency()
  // (one worker if detection is unavailable). Other counts must be positive.
  // Returns only after every worker's Executor has been initialized.
  static absl::StatusOr<std::unique_ptr<ThreadPool>> Create(
      int num_threads = -1);

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ~ThreadPool();

  int size() const { return num_threads_; }

  // Invokes the same callable exactly once on each worker, in parallel, then
  // waits for every callback AND the work queued on its Executor. The callable
  // may return void or absl::Status; all workers finish even if one fails.
  // The first error in worker-index order is returned (including CUDA errors).
  //
  // Each worker reuses its own thread/Executor across calls. The callable and
  // any shared captures must support concurrent invocation. Buffers may outlive
  // a call, but must be destroyed before the pool; never concurrently use a
  // worker's Executor outside its callback.
  //
  // Concurrent calls from external threads are serialized. Recursive calls
  // from this pool's workers are rejected rather than deadlocking. Destruction
  // must not race with a call or occur from a callback. As elsewhere in Pluto,
  // exceptions, including thread-creation/allocation failures, are fatal.
  template <class Function>
  absl::Status ParallelFor(Function&& function) {
    return Run([&](Executor& executor) -> absl::Status {
      if constexpr (std::is_void_v<
                        std::invoke_result_t<Function&, Executor&>>) {
        function(executor);
        return absl::OkStatus();
      } else {
        return function(executor);
      }
    });
  }

 private:
  ThreadPool(int num_threads, int device);
  absl::Status Run(absl::FunctionRef<absl::Status(Executor&)> function);
  void Worker(int index);

  const int num_threads_;
  const int device_;
  std::vector<std::thread> threads_;

  // call_mutex_ serializes entire ParallelFor calls, including callback
  // lifetime. mutex_ protects the generation, borrowed callable, and results.
  std::mutex call_mutex_;
  std::mutex mutex_;
  std::condition_variable work_available_;
  std::condition_variable finished_;
  bool stopping_ = false;
  uint64_t generation_ = 0;
  int remaining_;
  std::optional<absl::FunctionRef<absl::Status(Executor&)>> function_;
  std::vector<absl::Status> statuses_;
};

}  // namespace pluto::cuda
