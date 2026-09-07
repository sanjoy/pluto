#pragma once

#include <cstddef>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/page_locked_host_buffer.h"

namespace pluto::cuda {

// A fixed-length typed view backed by reference-counted page-locked memory.
//
// Elements are deliberately restricted to trivially copyable types: CUDA
// transfer buffers are byte storage and do not run element constructors or
// destructors. Copying PageLockedHostArray is cheap and keeps the allocation
// alive; it does not copy its elements.
template <class T>
class PageLockedHostArray final {
 public:
  static_assert(std::is_trivially_copyable_v<T>);
  using value_type = T;

  PageLockedHostArray() = default;

  static absl::StatusOr<PageLockedHostArray> Allocate(size_t size) {
    if (size > std::numeric_limits<size_t>::max() / sizeof(T)) {
      return absl::InvalidArgumentError(
          "page-locked host array byte size overflows size_t");
    }
    auto buffer = PageLockedHostBuffer::Allocate(size * sizeof(T));
    if (!buffer.ok()) return buffer.status();
    return PageLockedHostArray(std::move(*buffer), size);
  }

  static absl::StatusOr<PageLockedHostArray> CopyFrom(
      absl::Span<const T> values) {
    auto result = Allocate(values.size());
    if (!result.ok()) return result.status();
    if (!values.empty()) {
      std::memcpy(result->data(), values.data(), values.size() * sizeof(T));
    }
    return result;
  }

  T* data() { return static_cast<T*>(buffer_.data()); }
  const T* data() const { return static_cast<const T*>(buffer_.data()); }
  size_t size() const { return size_; }
  size_t size_bytes() const { return buffer_.size_bytes(); }
  bool empty() const { return size_ == 0; }

  T& operator[](size_t index) { return data()[index]; }
  const T& operator[](size_t index) const { return data()[index]; }
  T* begin() { return data(); }
  const T* begin() const { return data(); }
  T* end() { return size_ == 0 ? data() : data() + size_; }
  const T* end() const { return size_ == 0 ? data() : data() + size_; }

  absl::Span<T> span() { return absl::MakeSpan(data(), size_); }
  absl::Span<const T> span() const {
    return absl::MakeConstSpan(data(), size_);
  }

  const PageLockedHostBuffer& buffer() const { return buffer_; }

 private:
  PageLockedHostArray(PageLockedHostBuffer buffer, size_t size)
      : buffer_(std::move(buffer)), size_(size) {}

  PageLockedHostBuffer buffer_;
  size_t size_ = 0;
};

}  // namespace pluto::cuda
