#pragma once

#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/buffer.h"
#include "src/host/buffer.h"

namespace pluto::llm {

using Buffer = cuda::Buffer;
using BufferVec = absl::InlinedVector<Buffer, 2>;
using HostBuffer = host::Buffer;
using HostBufferVec = absl::InlinedVector<HostBuffer, 2>;

enum class DataType {
  FP16,
  BF16,
  FP8,
};

// Saved forward state. A tree, rather than one flat vector, lets composed and
// repeated layers keep each child's private intermediates without imposing a
// layout convention on unrelated layer implementations.
struct Tape {
  BufferVec intermediates;
  std::vector<Tape> children;
};

// A differentiable GPU layer.
class Layer {
 public:
  virtual ~Layer() = default;

  virtual absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                                      Tape* tape) const = 0;
  virtual absl::StatusOr<BufferVec> bwd(
      absl::Span<const Buffer> output_gradients, Tape tape) = 0;
  virtual absl::Span<Buffer> weights() = 0;
  // FP32 gradient accumulators corresponding one-for-one with weights().
  // Stateless layers return an empty span. Optimizers clear these buffers
  // before backward and update the FP32 master weights after backward.
  virtual absl::Span<Buffer> gradients() { return {}; }
  virtual DataType output_type() const = 0;
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

  virtual absl::StatusOr<HostBuffer> fwd(
      absl::Span<const HostBuffer> inputs, ReferenceTape* tape) = 0;
  virtual absl::StatusOr<HostBufferVec> bwd(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceTape tape) = 0;
  virtual absl::Span<HostBuffer> weights() = 0;
  virtual absl::Span<HostBuffer> gradients() { return {}; }
  virtual DataType output_type() const = 0;
};

}  // namespace pluto::llm
