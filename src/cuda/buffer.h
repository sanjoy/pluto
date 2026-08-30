#pragma once

#include <cstddef>
#include <memory>

#include "absl/status/statusor.h"
#include "src/cuda/executor.h"

namespace pluto::cuda {

// An untyped, reference-counted device allocation tied to one Executor.
//
// Allocate() queues cudaMallocAsync() on the executor's stream. Copies share
// the same allocation, address, size, and executor. Destroying the last copy
// queues cudaFreeAsync() on that same stream, after prior submitted work.
//
// Buffer does not own its Executor. The caller must keep the Executor valid
// until every Buffer copy referring to it has been destroyed.
class Buffer final {
 public:
  static absl::StatusOr<Buffer> Allocate(size_t size_bytes, Executor& executor);

  Buffer(const Buffer&) = default;
  Buffer& operator=(const Buffer&) = default;
  Buffer(Buffer&&) noexcept = default;
  Buffer& operator=(Buffer&&) noexcept = default;
  ~Buffer() = default;

  // These are intentionally the only accessors: Buffer provides storage and
  // lifetime management, but gives the bytes no type or interpretation.
  void* data() const;
  size_t size_bytes() const;
  Executor& executor() const;

 private:
  struct Allocation;
  explicit Buffer(std::shared_ptr<Allocation> allocation);

  std::shared_ptr<Allocation> allocation_;
};

}  // namespace pluto::cuda
