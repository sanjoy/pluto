#include "src/cuda/page_locked_host_buffer.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdio>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/executor.h"

namespace pluto::cuda {

struct PageLockedHostBuffer::Allocation {
  Allocation(Executor& executor, size_t size_bytes)
      : executor(executor), size_bytes(size_bytes) {}

  ~Allocation() {
    if (data == nullptr)
      return;
    // The C++ owner may disappear immediately after cudaMemcpyAsync: physical
    // storage remains alive until this stream-ordered free follows the copy.
    const cudaError_t error = cudaFreeAsync(data, executor.stream());
    if (error != cudaSuccess) {
      std::fprintf(stderr, "cudaFreeAsync(host %p) failed: %s: %s\n", data,
                   cudaGetErrorName(error), cudaGetErrorString(error));
    }
  }

  Executor& executor;
  void* data = nullptr;
  size_t size_bytes;
};

absl::StatusOr<PageLockedHostBuffer> PageLockedHostBuffer::Allocate(
    Executor& executor, size_t size_bytes) {
  auto allocation = std::make_shared<Allocation>(executor, size_bytes);
  if (size_bytes == 0)
    return PageLockedHostBuffer(std::move(allocation));
  void* memory = nullptr;
  const cudaError_t allocation_error =
      cudaMallocFromPoolAsync(&memory, size_bytes, executor.host_memory_pool(),
                              executor.trivial_stream());
  if (allocation_error != cudaSuccess) {
    const auto operation =
        absl::StrCat("cudaMallocFromPoolAsync(host, ", size_bytes, ")");
    const absl::Status status = CudaStatus(allocation_error, operation.c_str());
    if (allocation_error == cudaErrorMemoryAllocation)
      return absl::ResourceExhaustedError(status.message());
    return status;
  }

  // Allocate() promises immediately CPU-accessible storage. CUDA returns an
  // address before the stream-ordered allocation completes, so the CPU must
  // wait before touching it (unlike device buffers, whose GPU consumers can
  // simply be queued after allocation on the same stream).
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html
  //
  // Using the compute stream here would put allocation behind queued training
  // kernels: waiting for the allocation would also wait for all those kernels.
  // The separate trivial stream lets the CPU prepare the next host buffer
  // while computation continues. The pool only recycles completed frees, so
  // it cannot introduce a dependency on a still-pending compute-stream free.
  // Successful allocations are freed on the compute stream after their copies.
  const cudaError_t ready = cudaStreamSynchronize(executor.trivial_stream());
  if (ready != cudaSuccess) {
    // No consumer has received this address yet. Release it on the allocation
    // stream, preserving allocation-before-free ordering even on this path.
    const absl::Status cleanup =
        CudaStatus(cudaFreeAsync(memory, executor.trivial_stream()),
                   "cudaFreeAsync(failed host allocation)");
    if (!cleanup.ok()) std::fprintf(stderr, "%s\n", cleanup.ToString().c_str());
    // Cleanup also follows the trivial stream's enqueue-and-wait contract.
    const absl::Status drained =
        CudaStatus(cudaStreamSynchronize(executor.trivial_stream()),
                   "cudaStreamSynchronize(failed host allocation cleanup)");
    if (!drained.ok()) std::fprintf(stderr, "%s\n", drained.ToString().c_str());
    return CudaStatus(ready, "cudaStreamSynchronize(host allocation)");
  }
  allocation->data = memory;
  return PageLockedHostBuffer(std::move(allocation));
}

void* PageLockedHostBuffer::data() {
  return allocation_ == nullptr ? nullptr : allocation_->data;
}

const void* PageLockedHostBuffer::data() const {
  return allocation_ == nullptr ? nullptr : allocation_->data;
}

size_t PageLockedHostBuffer::size_bytes() const {
  return allocation_ == nullptr ? 0 : allocation_->size_bytes;
}

Executor& PageLockedHostBuffer::executor() const {
  assert(allocation_ != nullptr);
  return allocation_->executor;
}

}  // namespace pluto::cuda
