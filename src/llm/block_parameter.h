#pragma once

#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// A stable resident weight plus temporary FP32 training storage. Only the
// currently optimized block owns master weights and gradients; frozen blocks
// retain just BF16 (matrix/embedding) or FP32 (normalization) values.
class BlockParameter final {
 public:
  static absl::StatusOr<std::shared_ptr<BlockParameter>> Create(
      cuda::Executor& executor, Buffer value, DataType storage);

  // All allocation, conversion and destruction are ordered on value's
  // executor. Activate is idempotent; the first activation zeros gradients.
  absl::Status Activate();
  // Publish preserves the resident allocation's address, including for tied
  // embeddings and cached ComposedLayer weight handles.
  absl::Status Publish();
  // Release the FP32 working set without launching any kernels. Call Publish
  // first to retain pending master-weight updates; error cleanup can safely
  // discard them. Releasing inactive storage is an idempotent no-op.
  absl::Status Deactivate();

  bool active() const { return master_.has_value(); }
  const Buffer& value() const { return value_; }
  // These accessors require active(); inactive parameters own neither buffer.
  const Buffer& master() const { return master_.value(); }
  const Buffer& gradient() const { return gradient_.value(); }
  size_t elements() const { return elements_; }
  DataType storage() const { return storage_; }

 private:
  BlockParameter(Buffer value, DataType storage, size_t elements)
      : value_(std::move(value)), storage_(storage), elements_(elements) {}
  Buffer value_;
  DataType storage_;
  size_t elements_;
  std::optional<Buffer> master_;
  std::optional<Buffer> gradient_;
};

}  // namespace pluto::llm
