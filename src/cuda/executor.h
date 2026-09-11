#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace pluto::cuda {

// Converts a CUDA runtime result into an Abseil status. Successful results map
// to OkStatus; failures include both CUDA's symbolic name and description.
absl::Status CudaStatus(cudaError_t error, const char* operation);

// Owns the CUDA execution context used by stream-ordered computations.
//
// Work and frees use one explicitly created, non-default compute stream. An
// independent stream prepares allocations from the owned pinned-host pool.
// Waiting for a CPU-accessible allocation therefore need not wait for pending
// computation. Only already-completed frees may be recycled by the host pool:
// it cannot insert dependencies that make allocation wait for compute work.
//
// An Executor must outlive every device Buffer and PageLockedHostBuffer
// allocated through it, including copies hidden in PageLockedHostArray.
class Executor final {
 public:
  static absl::StatusOr<std::unique_ptr<Executor>> Create();

  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;
  ~Executor();

  // Waits for all work previously submitted to this executor's compute stream.
  absl::Status Synchronize() const;

  // CUDA launch syntax and runtime calls require the native stream handle.
  // Code must obtain it from the Executor passed to the current operation.
  cudaStream_t stream() const { return stream_; }

  // Borrowed handle for inspecting the executor-owned pool. Callers must not
  // destroy it or change its access/reuse policy.
  cudaMemPool_t host_memory_pool() const { return host_memory_pool_; }

 private:
  friend class PageLockedHostBuffer;

  Executor() = default;

  // Enqueues allocation without waiting. PageLockedHostBuffer must wait for
  // host_allocation_stream_ before exposing the address for CPU access or
  // transfers. Its eventual free follows those transfers on stream().
  absl::StatusOr<void*> AllocatePageLockedHostMemory(size_t size_bytes);

  cudaStream_t stream_ = nullptr;
  cudaStream_t host_allocation_stream_ = nullptr;
  cudaMemPool_t host_memory_pool_ = nullptr;
};

}  // namespace pluto::cuda
