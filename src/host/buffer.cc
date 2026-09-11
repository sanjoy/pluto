#include "src/host/buffer.h"

#include <cstddef>
#include <memory>
#include <utility>

namespace pluto::host {

struct HostBuffer::Allocation {
  explicit Allocation(size_t size)
      : bytes(size == 0 ? nullptr : std::make_unique<std::byte[]>(size)),
        size_bytes(size) {}

  // Allocations made by operator new[] are suitably aligned for ordinary
  // scalar types, which lets the deliberately untyped buffer safely hold the
  // float, int32, and uint16 arrays used by reference layers.
  std::unique_ptr<std::byte[]> bytes;
  size_t size_bytes;
};

absl::StatusOr<HostBuffer> HostBuffer::Allocate(size_t size_bytes) {
  return HostBuffer(std::make_shared<Allocation>(size_bytes));
}

void* HostBuffer::data() {
  return allocation_ == nullptr ? nullptr : allocation_->bytes.get();
}

const void* HostBuffer::data() const {
  return allocation_ == nullptr ? nullptr : allocation_->bytes.get();
}

size_t HostBuffer::size_bytes() const {
  return allocation_ == nullptr ? 0 : allocation_->size_bytes;
}

}  // namespace pluto::host
