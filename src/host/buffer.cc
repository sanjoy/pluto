#include "src/host/buffer.h"

#include <cstddef>
#include <memory>
#include <new>
#include <utility>

#include "absl/status/status.h"

namespace pluto::host {

struct Buffer::Allocation {
  explicit Allocation(size_t size)
      : bytes(size == 0 ? nullptr : std::make_unique<std::byte[]>(size)),
        size_bytes(size) {}

  // Allocations made by operator new[] are suitably aligned for ordinary
  // scalar types, which lets the deliberately untyped buffer safely hold the
  // float, int32, and uint16 arrays used by reference layers.
  std::unique_ptr<std::byte[]> bytes;
  size_t size_bytes;
};

absl::StatusOr<Buffer> Buffer::Allocate(size_t size_bytes) {
  try {
    return Buffer(std::make_shared<Allocation>(size_bytes));
  } catch (const std::bad_alloc&) {
    return absl::ResourceExhaustedError("host buffer allocation failed");
  }
}

void* Buffer::data() {
  return allocation_ == nullptr ? nullptr : allocation_->bytes.get();
}

const void* Buffer::data() const {
  return allocation_ == nullptr ? nullptr : allocation_->bytes.get();
}

size_t Buffer::size_bytes() const {
  return allocation_ == nullptr ? 0 : allocation_->size_bytes;
}

}  // namespace pluto::host
