#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>

#include "absl/status/statusor.h"

namespace pluto::cuda {

// An untyped, reference-counted device allocation tied to one CUDA stream.
//
// Allocate() queues cudaMallocAsync() on the supplied stream. Copies share the
// same allocation, address, size, and stream. Destroying the last copy queues
// cudaFreeAsync() on that same stream, after work previously submitted there.
// Allocate() rejects the null, legacy-default, and per-thread-default handles;
// callers must pass a stream returned by cudaStreamCreate*().
//
// Buffer does not own its stream. The caller must keep the stream valid until
// every Buffer copy referring to it has been destroyed. The caller may then
// synchronize or destroy the stream to wait for the queued free. Work on other
// streams must establish its own CUDA event ordering before the final Buffer
// reference is released.
class Buffer final {
 public:
  static absl::StatusOr<Buffer> Allocate(size_t size_bytes,
                                         cudaStream_t stream);

  Buffer(const Buffer&) = default;
  Buffer& operator=(const Buffer&) = default;
  Buffer(Buffer&&) noexcept = default;
  Buffer& operator=(Buffer&&) noexcept = default;
  ~Buffer() = default;

  // These are intentionally the only accessors: Buffer provides storage and
  // lifetime management, but gives the bytes no type or interpretation.
  void* data() const;
  size_t size_bytes() const;
  cudaStream_t stream() const;

 private:
  struct Allocation;
  explicit Buffer(std::shared_ptr<Allocation> allocation);

  std::shared_ptr<Allocation> allocation_;
};

}  // namespace pluto::cuda
