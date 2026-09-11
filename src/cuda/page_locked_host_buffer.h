#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/status/statusor.h"

namespace pluto::cuda {

// An untyped, reference-counted allocation in CUDA page-locked host memory.
//
// Page-locked (or "pinned") memory lets cudaMemcpyAsync transfer bytes without
// first staging them through a hidden CUDA-owned pinned allocation. Copies of
// this object share both the allocation and its lifetime. Unlike device
// Buffer, host memory is not associated with an Executor or stream.
// Moved-from buffers are empty: data() is nullptr and size_bytes() is zero.
// Their shared allocation pointer is null.
class PageLockedHostBuffer final {
 public:
  static absl::StatusOr<PageLockedHostBuffer> Allocate(size_t size_bytes);

  void* data();
  const void* data() const;
  size_t size_bytes() const;

 private:
  // The typed wrapper may create an empty backing buffer without allocating.
  // Other callers must use Allocate(), including Allocate(0) for an empty one.
  template <class T>
  friend class PageLockedHostArray;

  PageLockedHostBuffer() = default;

  struct Allocation;

  explicit PageLockedHostBuffer(std::shared_ptr<Allocation> allocation)
      : allocation_(std::move(allocation)) {}

  std::shared_ptr<Allocation> allocation_;
};

}  // namespace pluto::cuda
