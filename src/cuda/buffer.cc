#include "src/cuda/buffer.h"

#include <cuda_runtime_api.h>

#include <cstdio>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/executor.h"

namespace pluto::cuda {
namespace {

absl::Status CudaAllocationError(cudaError_t error, size_t size_bytes) {
  const auto operation = absl::StrCat("cudaMallocAsync(", size_bytes, ")");
  const absl::Status status = CudaStatus(error, operation.c_str());
  // Preserve the allocation-specific code while sharing CUDA diagnostics.
  if (error == cudaErrorMemoryAllocation)
    return absl::ResourceExhaustedError(status.message());
  return status;
}

}  // namespace

struct Buffer::Allocation {
  Allocation(Executor& executor, size_t size_bytes)
      : size_bytes(size_bytes), executor(executor) {}

  ~Allocation() {
    if (data == nullptr)
      return;

    // A destructor cannot return a Status. Report a programming/runtime error
    // rather than silently hiding it; normal stream-ordered frees return
    // cudaSuccess immediately and finish when the stream reaches this call.
    const cudaError_t error = cudaFreeAsync(data, executor.stream());
    if (error != cudaSuccess) {
      std::fprintf(stderr, "cudaFreeAsync(%p) failed: %s: %s\n", data,
                   cudaGetErrorName(error), cudaGetErrorString(error));
    }
  }

  void* data = nullptr;
  size_t size_bytes;
  Executor& executor;
};

Buffer::Buffer(std::shared_ptr<Allocation> allocation)
    : allocation_(std::move(allocation)) {}

absl::StatusOr<Buffer> Buffer::Allocate(Executor& executor, size_t size_bytes) {
  // Create the control block first so that a later host allocation failure
  // cannot leak a successfully allocated device pointer.
  auto allocation = std::make_shared<Allocation>(executor, size_bytes);

  // CUDA treats a zero-byte allocation as no storage. Keeping a control block
  // still preserves the requested stream and normal copy semantics.
  if (size_bytes == 0)
    return Buffer(std::move(allocation));

  const cudaError_t error =
      cudaMallocAsync(&allocation->data, size_bytes, executor.stream());
  if (error != cudaSuccess)
    return CudaAllocationError(error, size_bytes);
  return Buffer(std::move(allocation));
}

void* Buffer::data() const { return allocation_->data; }

size_t Buffer::size_bytes() const { return allocation_->size_bytes; }

Executor& Buffer::executor() const { return allocation_->executor; }

}  // namespace pluto::cuda
