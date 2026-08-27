#include "src/gpu/buffer.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdio>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace pluto::gpu {
namespace {

absl::Status CudaAllocationError(cudaError_t error, size_t size_bytes) {
  const absl::StatusCode code =
      error == cudaErrorMemoryAllocation
          ? absl::StatusCode::kResourceExhausted
          : absl::StatusCode::kInternal;
  return absl::Status(
      code, absl::StrCat("cudaMallocAsync(", size_bytes, ") failed: ",
                         cudaGetErrorName(error), ": ",
                         cudaGetErrorString(error)));
}

}  // namespace

struct Buffer::Allocation {
  ~Allocation() {
    if (data == nullptr) return;

    // A destructor cannot return a Status. Report a programming/runtime error
    // rather than silently hiding it; normal stream-ordered frees return
    // cudaSuccess immediately and finish when the stream reaches this call.
    const cudaError_t error = cudaFreeAsync(data, stream);
    if (error != cudaSuccess) {
      std::fprintf(stderr, "cudaFreeAsync(%p) failed: %s: %s\n", data,
                   cudaGetErrorName(error), cudaGetErrorString(error));
    }
  }

  void* data = nullptr;
  size_t size_bytes = 0;
  cudaStream_t stream = nullptr;
};

Buffer::Buffer(std::shared_ptr<Allocation> allocation)
    : allocation_(std::move(allocation)) {}

absl::StatusOr<Buffer> Buffer::Allocate(size_t size_bytes,
                                         cudaStream_t stream) {
  const bool has_explicit_stream =
      stream != nullptr && stream != cudaStreamLegacy &&
      stream != cudaStreamPerThread;
  assert(has_explicit_stream &&
         "Buffer requires an explicitly created CUDA stream");
  // Keep the invariant in optimized builds where assert() may be compiled out.
  if (!has_explicit_stream) {
    return absl::InvalidArgumentError(
        "Buffer requires an explicitly created CUDA stream");
  }

  // Create the control block first so that a later host allocation failure
  // cannot leak a successfully allocated device pointer.
  auto allocation = std::make_shared<Allocation>();
  allocation->size_bytes = size_bytes;
  allocation->stream = stream;

  // CUDA treats a zero-byte allocation as no storage. Keeping a control block
  // still preserves the requested stream and normal copy semantics.
  if (size_bytes == 0) return Buffer(std::move(allocation));

  const cudaError_t error =
      cudaMallocAsync(&allocation->data, size_bytes, stream);
  if (error != cudaSuccess) return CudaAllocationError(error, size_bytes);
  return Buffer(std::move(allocation));
}

void* Buffer::data() const { return allocation_->data; }

size_t Buffer::size_bytes() const { return allocation_->size_bytes; }

cudaStream_t Buffer::stream() const { return allocation_->stream; }

}  // namespace pluto::gpu
