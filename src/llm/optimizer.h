#pragma once

#include <cstddef>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/cuda/executor.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// Defined alongside AdamWOptimizer in adamw_optimizer.h.
struct AdamWConfig;

// Common lifecycle for optimizers that update a Layer's parameter buffers.
//
// Create() constructs the default optimizer (currently AdamW) behind this
// interface. It is necessarily non-virtual because C++ static methods cannot
// be virtual. Concrete optimizers also expose their algorithm-specific factory
// so callers can name an implementation explicitly.
class Optimizer {
 public:
  virtual ~Optimizer() = default;

  static absl::StatusOr<std::unique_ptr<Optimizer>> Create(
      cuda::Executor& executor, Layer& model, AdamWConfig config);

  // Clears every unique parameter-gradient accumulator. Call this before the
  // first backward pass. ApplyStep() also clears gradients after applying
  // updates.
  virtual absl::Status ZeroGrad() = 0;

  // Applies one update using the accumulated gradients and advances step().
  virtual absl::Status ApplyStep() = 0;

  virtual int step() const = 0;
  virtual size_t parameter_tensor_count() const = 0;
};

}  // namespace pluto::llm
