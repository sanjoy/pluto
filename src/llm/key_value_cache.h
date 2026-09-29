#pragma once

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"

namespace pluto::llm {

// Storage and position bookkeeping for one sequence's causal attention cache.
// Keys and values are separate FP32 [capacity, key_value_heads, head_dim]
// buffers. This class does not normalize, rotate, or otherwise compute them;
// the attention layer writes each token's row before reading the valid prefix.
//
// Position is the next row to write, equivalently the valid prefix length.
// Call Advance() after successfully queuing one token's attention work. Reset()
// starts a new sequence without clearing storage: every newly valid row must
// be overwritten before use, and rows beyond position() must never be read.
// Neither operation synchronizes the GPU. Users must serialize access on the
// creating executor and keep that executor alive until this object and all
// copies of its buffers have been destroyed. A cache cannot serve two
// attention layers or independent sequences at the same time.
class KeyValueCache final {
 public:
  static absl::StatusOr<std::unique_ptr<KeyValueCache>> Create(
      cuda::Executor& executor, int capacity, int key_value_heads,
      int head_dim);

  KeyValueCache(const KeyValueCache&) = delete;
  KeyValueCache& operator=(const KeyValueCache&) = delete;

  const cuda::Buffer& keys() const { return keys_; }
  const cuda::Buffer& values() const { return values_; }
  cuda::Executor& executor() const { return keys_.executor(); }
  int capacity() const { return capacity_; }
  int key_value_heads() const { return key_value_heads_; }
  int head_dim() const { return head_dim_; }
  int position() const { return position_; }

  // Advances to the next row; a full cache returns an error without changing
  // its position. The caller must check capacity before queuing a row write.
  absl::Status Advance();
  void Reset() { position_ = 0; }

 private:
  KeyValueCache(int capacity, int key_value_heads, int head_dim,
                cuda::Buffer keys, cuda::Buffer values);

  int capacity_;
  int key_value_heads_;
  int head_dim_;
  cuda::Buffer keys_;
  cuda::Buffer values_;
  int position_ = 0;
};

}  // namespace pluto::llm
