#include "src/cuda/page_locked_host_buffer.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdio>
#include <memory>
#include <utility>

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
  auto memory = executor.AllocatePageLockedHostMemory(size_bytes);
  if (!memory.ok())
    return memory.status();
  allocation->data = *memory;
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
