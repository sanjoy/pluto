#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "src/cuda/executor.h"

namespace pluto::cuda {

// An untyped, reference-counted allocation in CUDA page-locked host memory.
//
// Allocate() uses the supplied Executor's host-backed CUDA memory pool. The
// returned storage is ready for immediate CPU access; preparing that storage
// waits only for the executor's allocation stream, never its compute stream.
// Copies share the allocation. Destroying the last copy queues cudaFreeAsync
// on the executor's compute stream, after previously submitted transfers.
//
// All asynchronous accesses must use the owning Executor, which must outlive
// every copy. CPU reads/writes still need normal transfer ordering: wait before
// reading a D2H result or overwriting a pending H2D source. Destruction itself
// needs no such wait. Page-locked storage avoids CUDA's hidden pageable
// staging.
//
// Moved-from buffers are empty: data() is nullptr and size_bytes() is zero.
// Their shared allocation pointer is null and executor() must not be called.
class PageLockedHostBuffer final {
 public:
  static absl::StatusOr<PageLockedHostBuffer> Allocate(Executor& executor,
                                                       size_t size_bytes);

  void* data();
  const void* data() const;
  size_t size_bytes() const;

  // Factory-created empty buffers retain their executor. Default/moved-from
  // backing buffers do not have one and must not be passed to this accessor.
  Executor& executor() const;

 private:
  // The typed wrapper may create an empty backing buffer without allocating.
  // Other callers must use Allocate(), including Allocate(executor, 0).
  template <class T>
  friend class PageLockedHostArray;

  PageLockedHostBuffer() = default;

  struct Allocation;

  explicit PageLockedHostBuffer(std::shared_ptr<Allocation> allocation)
      : allocation_(std::move(allocation)) {}

  std::shared_ptr<Allocation> allocation_;
};

}  // namespace pluto::cuda
