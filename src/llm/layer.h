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
struct Tape {
  // Non-owning identity of the layer whose successful fwd() produced this tape.
  // The layer must outlive its tape. A null pointer marks an unused or failed
  // forward pass; bwd() rejects it, and tapes from other layer instances.
  const Layer* layer = nullptr;
  BufferVec intermediates;
  std::vector<Tape> children;
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

  // Every call starts a fresh tape. Only a successful forward pass associates
  // it with this layer, so a failed retry cannot leave an old tape usable.
  absl::StatusOr<Buffer> fwd(cuda::Executor& executor,
                             absl::Span<const Buffer> inputs,
                             Tape* tape) const {
    if (tape == nullptr)
      return absl::InvalidArgumentError("fwd requires a non-null tape");
    *tape = Tape{};
    auto output = fwd_impl(executor, inputs, tape);
    tape->layer = output.ok() ? this : nullptr;
    return output;
  }

  // Check instance identity before dispatching any backward work. Matching
  // shapes or layer types alone do not make another layer's saved state valid.
  absl::StatusOr<BufferVec> bwd(cuda::Executor& executor,
                                absl::Span<const Buffer> output_gradients,
                                Tape tape) {
    if (tape.layer != this)
      return absl::InvalidArgumentError(
          "bwd requires a tape from this layer's successful fwd");
    return bwd_impl(executor, output_gradients, std::move(tape));
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
  // Implementations cannot bypass the public entry points' tape checks.
  virtual absl::StatusOr<Buffer> fwd_impl(cuda::Executor& executor,
                                          absl::Span<const Buffer> inputs,
                                          Tape* tape) const = 0;
  virtual absl::StatusOr<BufferVec> bwd_impl(
      cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
      Tape tape) = 0;
};

// Saved forward state for the CPU reference graph. It deliberately has the
// same tree structure as Tape, but owns host buffers so reference execution is
// independent of CUDA allocation and stream semantics.
struct ReferenceTape {
  HostBufferVec intermediates;
  std::vector<ReferenceTape> children;
};

// CPU counterpart to Layer. Reference layers favor direct scalar loops over
// performance; their job is to state the math plainly enough to serve as an
// executable specification for the cuTile kernels.
class LayerReference {
 public:
  virtual ~LayerReference() = default;

  virtual absl::StatusOr<HostBuffer> fwd(absl::Span<const HostBuffer> inputs,
                                         ReferenceTape* tape) = 0;
  virtual absl::StatusOr<HostBufferVec> bwd(
      absl::Span<const HostBuffer> output_gradients, ReferenceTape tape) = 0;
  virtual absl::Span<HostBuffer> weights() = 0;
  virtual absl::Span<HostBuffer> gradients() { return {}; }
  virtual DataType output_type() const = 0;
};

}  // namespace pluto::llm
