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
class PageLockedHostBuffer final {
 public:
  PageLockedHostBuffer() = default;

  static absl::StatusOr<PageLockedHostBuffer> Allocate(size_t size_bytes);

  void* data();
  const void* data() const;
  size_t size_bytes() const;

 private:
  struct Allocation;

  explicit PageLockedHostBuffer(std::shared_ptr<Allocation> allocation)
      : allocation_(std::move(allocation)) {}

  std::shared_ptr<Allocation> allocation_;
};

}  // namespace pluto::cuda
