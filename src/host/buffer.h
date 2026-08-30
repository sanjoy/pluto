#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/status/statusor.h"

namespace pluto::host {

// A small reference-counted byte buffer for CPU reference implementations.
//
// This intentionally mirrors only the ownership and byte-oriented interface of
// gpu::Buffer. CPU work is synchronous, so there is no stream association and
// no asynchronous allocation/free policy to model.
class Buffer final {
 public:
  Buffer() = default;

  static absl::StatusOr<Buffer> Allocate(size_t size_bytes);

  void* data();
  const void* data() const;
  size_t size_bytes() const;

 private:
  struct Allocation;

  explicit Buffer(std::shared_ptr<Allocation> allocation)
      : allocation_(std::move(allocation)) {}

  std::shared_ptr<Allocation> allocation_;
};

}  // namespace pluto::host
