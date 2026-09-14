#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/host/buffer.h"

namespace pluto::llm {

using Buffer = cuda::Buffer;
using BufferVec = absl::InlinedVector<Buffer, 2>;
using host::HostBuffer;
using HostBufferVec = absl::InlinedVector<HostBuffer, 2>;

enum class DataType {
  FP16,
  BF16,
  FP8,
  FP32,
  INT32,
};

// Physical dtype and logical dimensions of one forward input/output buffer.
// The leading batch dimension is symbolic: -2 matches only -2, never a fixed
// extent or another special value. All other dimensions are positive and
// concrete. Thus [-2, 1024, 512] and [-2, 512] are different types even though
// kernels flatten the former's batch/token axes into contiguous matrix rows.
// Scalar tensors use an empty dimensions vector. A signature may also include
// unbatched tensors, such as SAE's FP32 [input_dim, feature_dim] decoder
// output.
class ActivationType {
 public:
  static constexpr int64_t kBatchDimension = -2;

  ActivationType(DataType data_type, absl::InlinedVector<int64_t, 4> dimensions)
      : data_type_(data_type), dimensions_(std::move(dimensions)) {}

  DataType data_type() const { return data_type_; }
  absl::Span<const int64_t> dimensions() const { return dimensions_; }

  // No broadcasting, flattening, or wildcard matching is performed here.
  bool operator==(const ActivationType& other) const {
    return data_type_ == other.data_type_ && dimensions_ == other.dimensions_;
  }
  bool operator!=(const ActivationType& other) const {
    return !(*this == other);
  }

  absl::Status Validate() const {
    switch (data_type_) {
      case DataType::FP16:
      case DataType::BF16:
      case DataType::FP8:
      case DataType::FP32:
      case DataType::INT32:
        break;
      default:
        return absl::InvalidArgumentError("unknown activation data type");
    }
    for (size_t i = 0; i < dimensions_.size(); ++i)
      if (dimensions_[i] <= 0 && !(i == 0 && dimensions_[i] == kBatchDimension))
        return absl::InvalidArgumentError(
            "activation dimensions must be positive, except leading batch "
            "(-2)");
    return absl::OkStatus();
  }

 private:
  DataType data_type_;
  absl::InlinedVector<int64_t, 4> dimensions_;
};

// output_type() historically names a compute policy. In particular, the FP16
// kernels retain FP32 activations. Signatures describe actual buffer storage,
// not that policy. FP32/INT32 describe tensors; they do not enable new kernels.
inline DataType ActivationDataType(DataType compute_type) {
  return compute_type == DataType::FP16 ? DataType::FP32 : compute_type;
}

class Layer;

// Saved forward state. A tree, rather than one flat vector, lets composed
// layers keep each child's private intermediates without imposing a
// layout convention on unrelated layer implementations.
struct BackwardState {
  // Non-owning identity of the layer whose successful fwd() produced this
  // state. The layer must outlive its state. A null pointer marks an unused or
  // failed forward pass; bwd() rejects it, and states from other layer
  // instances.
  const Layer* layer = nullptr;
  BufferVec intermediates;
  std::vector<BackwardState> children;
};

// A forward pass returns its outputs together with the saved state needed for
// backward. Buffers are shared handles; moving this result transfers the state
// without copying device memory. The producing layer must outlive the state.
// Outputs are ordered: ordinary layers return one buffer, while layers such as
// SAE return multiple buffers in their documented order.
struct FwdResult {
  BufferVec outputs;
  BackwardState state;
};

// A differentiable GPU layer.
class Layer {
 public:
  virtual ~Layer() = default;

  // Diagnostic class name, not a unique instance identifier. The returned
  // non-owning view remains valid for the layer's lifetime. Built-in layers
  // return string literals, so inspecting a name never allocates memory.
  virtual absl::string_view name() const = 0;

  // Ordered forward signatures, immutable for the layer's lifetime. Batch is
  // the number of samples, not flattened token rows. Buffer remains untyped:
  // graph construction and dataset boundaries check these shapes, but cannot
  // recover dtype or sample boundaries from arbitrary bytes. Raw fwd callers
  // must honor the signatures; per-kernel buffer checks remain necessary.
  virtual absl::Span<const ActivationType> input_types() const = 0;
  virtual absl::Span<const ActivationType> output_types() const = 0;

  // State is published only with a successful output. Failed calls cannot
  // overwrite state retained from an earlier forward pass. If this Executor
  // has LayerHooks attached, ActivationHook processes successful outputs before
  // they are published. See layer_hooks.h for replacement/aliasing rules.
  absl::StatusOr<FwdResult> fwd(cuda::Executor& executor,
                                absl::Span<const Buffer> inputs) const;

  // Output gradients follow the forward outputs' order; returned gradients
  // follow the forward inputs' order. Layer-specific exceptions are documented
  // by each layer: terminal losses accept no upstream gradients, and layers
  // consuming nondifferentiable integer inputs may return no input gradients.
  // Check instance identity before dispatching any backward work. Matching
  // shapes or layer types alone do not make another layer's saved state valid.
  // An attached GradientHook processes incoming output gradients (FP32) before
  // bwd_impl computes parameter/input gradients. Its handle replacements do
  // not alter the caller's gradient handles.
  absl::StatusOr<BufferVec> bwd(cuda::Executor& executor,
                                absl::Span<const Buffer> output_gradients,
                                BackwardState state);
  virtual absl::Span<Buffer> weights() = 0;
  // Read-only access for serialization and inspection. Implementations expose
  // the same handles as weights(); callers must not mutate their device bytes.
  absl::Span<const Buffer> weights() const {
    absl::Span<Buffer> mutable_weights = const_cast<Layer*>(this)->weights();
    return absl::Span<const Buffer>(mutable_weights.data(),
                                    mutable_weights.size());
  }
  // FP32 gradient accumulators corresponding one-for-one with weights().
  // Stateless layers return an empty span. Optimizers clear these buffers
  // before backward and update the FP32 master weights after backward.
  virtual absl::Span<Buffer> gradients() { return {}; }
  virtual DataType output_type() const = 0;

 private:
  // Implementations cannot bypass the public entry points' state checks.
  virtual absl::StatusOr<FwdResult> fwd_impl(
      cuda::Executor& executor, absl::Span<const Buffer> inputs) const = 0;
  virtual absl::StatusOr<BufferVec> bwd_impl(
      cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
      BackwardState state) = 0;
};

class LayerReference;

// Saved forward state for the CPU reference graph. It deliberately has the
// same tree structure as BackwardState, but owns host buffers so reference
// execution is independent of CUDA allocation and stream semantics.
struct ReferenceBackwardState {
  // Same non-owning identity and lifetime contract as BackwardState.
  const LayerReference* layer = nullptr;
  HostBufferVec intermediates;
  std::vector<ReferenceBackwardState> children;
};

// Host counterpart to FwdResult, with the same output/state ownership contract.
struct ReferenceFwdResult {
  HostBufferVec outputs;
  ReferenceBackwardState state;
};

// CPU counterpart to Layer. Reference layers favor direct scalar loops over
// performance; their job is to state the math plainly enough to serve as an
// executable specification for the cuTile kernels.
class LayerReference {
 public:
  virtual ~LayerReference() = default;

  // Same contract as Layer::name(); reference names include the "Reference"
  // suffix so diagnostics distinguish CPU and GPU implementations.
  virtual absl::string_view name() const = 0;

  // Same physical dtypes and logical shapes as the corresponding GPU layer.
  virtual absl::Span<const ActivationType> input_types() const = 0;
  virtual absl::Span<const ActivationType> output_types() const = 0;

  // Match the GPU contract: failed forward calls return no partial state.
  absl::StatusOr<ReferenceFwdResult> fwd(
      absl::Span<const HostBuffer> inputs) const {
    auto result = fwd_impl(inputs);
    if (result.ok())
      result->state.layer = this;
    return result;
  }
  // Uses Layer::bwd's ordered output/input gradient convention, including
  // documented exceptions for terminal losses and nondifferentiable inputs.
  absl::StatusOr<HostBufferVec> bwd(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceBackwardState state) {
    if (state.layer != this)
      return absl::InvalidArgumentError(
          "bwd requires a state from this reference layer's successful fwd");
    return bwd_impl(output_gradients, std::move(state));
  }
  virtual absl::Span<HostBuffer> weights() = 0;
  virtual absl::Span<HostBuffer> gradients() { return {}; }
  virtual DataType output_type() const = 0;

 private:
  virtual absl::StatusOr<ReferenceFwdResult> fwd_impl(
      absl::Span<const HostBuffer> inputs) const = 0;
  virtual absl::StatusOr<HostBufferVec> bwd_impl(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceBackwardState state) = 0;
};

}  // namespace pluto::llm
