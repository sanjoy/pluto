#include "src/llm/key_value_cache.h"

#include <cstddef>
#include <limits>
#include <utility>

#include "absl/memory/memory.h"
#include "src/util/status_macros.h"

namespace pluto::llm {

KeyValueCache::KeyValueCache(int capacity, int key_value_heads, int head_dim,
                             cuda::Buffer keys, cuda::Buffer values)
    : capacity_(capacity),
      key_value_heads_(key_value_heads),
      head_dim_(head_dim),
      keys_(std::move(keys)),
      values_(std::move(values)) {}

absl::StatusOr<std::unique_ptr<KeyValueCache>> KeyValueCache::Create(
    cuda::Executor& executor, int capacity, int key_value_heads, int head_dim) {
  if (capacity <= 0 || key_value_heads <= 0 || head_dim <= 0)
    return absl::InvalidArgumentError("KV cache dimensions must be positive");

  // Validate each multiplication before allocating either buffer: dimensions
  // can each fit in int while their FP32 storage size overflows size_t.
  size_t bytes = sizeof(float);
  for (int dimension : {capacity, key_value_heads, head_dim}) {
    if (static_cast<size_t>(dimension) >
        std::numeric_limits<size_t>::max() / bytes)
      return absl::InvalidArgumentError("KV cache byte size overflows size_t");
    bytes *= static_cast<size_t>(dimension);
  }
  ASSIGN_OR_RETURN(auto keys, cuda::Buffer::Allocate(executor, bytes));
  ASSIGN_OR_RETURN(auto values, cuda::Buffer::Allocate(executor, bytes));
  return absl::WrapUnique(new KeyValueCache(
      capacity, key_value_heads, head_dim, std::move(keys), std::move(values)));
}

absl::Status KeyValueCache::Advance() {
  if (position_ == capacity_)
    return absl::ResourceExhaustedError("KV cache is full");
  ++position_;
  return absl::OkStatus();
}

}  // namespace pluto::llm
