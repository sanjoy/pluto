#include "src/cuda/page_locked_host_buffer.h"

#include <cuda_runtime_api.h>

#include <cstdio>
#include <memory>
#include <new>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace pluto::cuda {
namespace {

absl::Status AllocationError(cudaError_t error, size_t size_bytes) {
  const absl::StatusCode code = error == cudaErrorMemoryAllocation
                                    ? absl::StatusCode::kResourceExhausted
                                    : absl::StatusCode::kInternal;
  return absl::Status(code, absl::StrCat("cudaMallocHost(", size_bytes,
                                         ") failed: ", cudaGetErrorName(error),
                                         ": ", cudaGetErrorString(error)));
}

}  // namespace

struct PageLockedHostBuffer::Allocation {
  explicit Allocation(size_t size_bytes) : size_bytes(size_bytes) {}

  ~Allocation() {
    if (data == nullptr) return;
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
  std::shared_ptr<Allocation> allocation;
  try {
    allocation = std::make_shared<Allocation>(size_bytes);
  } catch (const std::bad_alloc&) {
    return absl::ResourceExhaustedError(
        "page-locked host allocation control block failed");
  }
  if (size_bytes == 0) {
    return PageLockedHostBuffer(std::move(allocation));
  }
  const cudaError_t error = cudaMallocHost(&allocation->data, size_bytes);
  if (error != cudaSuccess) return AllocationError(error, size_bytes);
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
