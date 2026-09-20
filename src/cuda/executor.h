#pragma once

#include <cuda_runtime_api.h>

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
// independent immediate stream handles short enqueue-and-immediately-wait
// operations, including allocations from the owned pinned-host pool.
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

  // Normally idle, non-default stream for short enqueue-and-wait operations.
  // Callers must immediately synchronize this stream after submitting work,
  // before returning; never leave deferred work queued here or introduce a
  // dependency on the compute stream. This keeps a readiness wait independent
  // of queued training kernels. The borrowed stream must not be destroyed.
  cudaStream_t immediate_stream() const { return immediate_stream_; }

  // Borrowed pool for pinned-host allocations. Callers must not destroy it or
  // change its access/reuse policy.
  cudaMemPool_t host_memory_pool() const { return host_memory_pool_; }

 private:
  Executor() = default;

  cudaStream_t stream_ = nullptr;
  cudaStream_t immediate_stream_ = nullptr;
  cudaMemPool_t host_memory_pool_ = nullptr;
};

}  // namespace pluto::cuda
