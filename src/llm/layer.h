#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
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
};

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

// A forward pass returns its output together with the saved state needed for
// backward. Buffers are shared handles; moving this result transfers the state
// without copying device memory. The producing layer must outlive the state.
struct FwdResult {
  Buffer output;
  BackwardState state;
};

// A differentiable GPU layer.
class Layer {
 public:
  virtual ~Layer() = default;

  // Validates sample boundaries before dataset batches are flattened for fwd().
  // Per-token layers accept every positive length. Layers whose kernels use a
  // fixed sequence width override this; combinators check all descendants.
  // Callers supplying raw Buffers directly must preserve those same boundaries:
  // fwd() cannot recover sample lengths from an untyped flat allocation.
  virtual absl::Status ValidateSequenceLength(int sequence_length) const {
    if (sequence_length <= 0)
      return absl::InvalidArgumentError("sequence_length must be positive");
    return absl::OkStatus();
  }

  // State is published only with a successful output. Failed calls cannot
  // overwrite state retained from an earlier forward pass.
  absl::StatusOr<FwdResult> fwd(cuda::Executor& executor,
                                absl::Span<const Buffer> inputs) const {
    auto result = fwd_impl(executor, inputs);
    if (result.ok()) result->state.layer = this;
    return result;
  }

  // Check instance identity before dispatching any backward work. Matching
  // shapes or layer types alone do not make another layer's saved state valid.
  absl::StatusOr<BufferVec> bwd(cuda::Executor& executor,
                                absl::Span<const Buffer> output_gradients,
                                BackwardState state) {
    if (state.layer != this)
      return absl::InvalidArgumentError(
          "bwd requires a state from this layer's successful fwd");
    return bwd_impl(executor, output_gradients, std::move(state));
  }
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
  HostBuffer output;
  ReferenceBackwardState state;
};

// CPU counterpart to Layer. Reference layers favor direct scalar loops over
// performance; their job is to state the math plainly enough to serve as an
// executable specification for the cuTile kernels.
class LayerReference {
 public:
  virtual ~LayerReference() = default;

  // Match the GPU contract: failed forward calls return no partial state.
  absl::StatusOr<ReferenceFwdResult> fwd(
      absl::Span<const HostBuffer> inputs) const {
    auto result = fwd_impl(inputs);
    if (result.ok()) result->state.layer = this;
    return result;
  }
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
