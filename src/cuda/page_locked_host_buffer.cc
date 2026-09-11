#include "src/cuda/page_locked_host_buffer.h"

#include <cuda_runtime_api.h>

#include <cstdio>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/executor.h"

namespace pluto::cuda {
namespace {

absl::Status AllocationError(cudaError_t error, size_t size_bytes) {
  const auto operation = absl::StrCat("cudaMallocHost(", size_bytes, ")");
  const absl::Status status = CudaStatus(error, operation.c_str());
  // Preserve the allocation-specific code while sharing CUDA diagnostics.
  if (error == cudaErrorMemoryAllocation)
    return absl::ResourceExhaustedError(status.message());
  return status;
}

}  // namespace

struct PageLockedHostBuffer::Allocation {
  explicit Allocation(size_t size_bytes) : size_bytes(size_bytes) {}

  ~Allocation() {
    if (data == nullptr)
      return;
    const cudaError_t error = cudaFreeHost(data);
    if (error != cudaSuccess) {
      std::fprintf(stderr, "cudaFreeHost(%p) failed: %s: %s\n", data,
                   cudaGetErrorName(error), cudaGetErrorString(error));
    }
  }

  void* data = nullptr;
  size_t size_bytes;
};

absl::StatusOr<PageLockedHostBuffer> PageLockedHostBuffer::Allocate(
    size_t size_bytes) {
  auto allocation = std::make_shared<Allocation>(size_bytes);
  if (size_bytes == 0)
    return PageLockedHostBuffer(std::move(allocation));
  const cudaError_t error = cudaMallocHost(&allocation->data, size_bytes);
  if (error != cudaSuccess)
    return AllocationError(error, size_bytes);
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

}  // namespace pluto::cuda
