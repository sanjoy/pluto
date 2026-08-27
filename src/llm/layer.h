#ifndef PLUTO_SRC_LLM_LAYER_H_
#define PLUTO_SRC_LLM_LAYER_H_

#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/gpu/buffer.h"

namespace pluto::llm {

using Buffer = gpu::Buffer;
using Buffers = absl::InlinedVector<Buffer, 2>;

enum class DataType {
  FP16,
  FP8,
};

// Saved forward state. A tree, rather than one flat vector, lets composed and
// repeated layers keep each child's private intermediates without imposing a
// layout convention on unrelated layer implementations.
struct Tape {
  Buffers intermediates;
  std::vector<Tape> children;
};

// A differentiable GPU layer.
class Layer {
 public:
  virtual ~Layer() = default;

  virtual absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                                      Tape* tape) = 0;
  virtual absl::StatusOr<Buffers> bwd(
      absl::Span<const Buffer> output_gradients, Tape tape) = 0;
  virtual absl::Span<Buffer> weights() = 0;
  virtual DataType data_type() const = 0;
};

}  // namespace pluto::llm

#endif  // PLUTO_SRC_LLM_LAYER_H_
