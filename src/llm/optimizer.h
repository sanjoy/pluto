#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/llm/layer.h"

namespace pluto::llm {

struct AdamWConfig {
  float learning_rate = 3e-4f;
  float beta1 = 0.9f;
  float beta2 = 0.95f;
  float epsilon = 1e-8f;
  float weight_decay = 0.1f;
};

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
      Layer* model, AdamWConfig config, cudaStream_t stream);

  // Clears every unique parameter-gradient accumulator. Call this before the
  // first backward pass. Step() also clears gradients after applying updates.
  virtual absl::Status ZeroGrad() = 0;

  // Applies one update using the accumulated gradients and advances step().
  virtual absl::Status Step() = 0;

  virtual int step() const = 0;
  virtual size_t parameter_tensor_count() const = 0;
};

// AdamW over a model's FP32 master weights and FP32 accumulated gradients.
// Tied parameters are identified by allocation address and receive exactly one
// optimizer state/update even when multiple layers expose the same weight.
class AdamWOptimizer final : public Optimizer {
 public:
  static absl::StatusOr<std::unique_ptr<AdamWOptimizer>> Create(
      Layer* model, AdamWConfig config, cudaStream_t stream);

  absl::Status ZeroGrad() override;
  absl::Status Step() override;

  int step() const override { return step_; }
  size_t parameter_tensor_count() const override { return weights_.size(); }

 private:
  AdamWOptimizer(AdamWConfig config, cudaStream_t stream,
                 std::vector<Buffer> weights,
                 std::vector<Buffer> gradients,
                 std::vector<Buffer> first_moments,
                 std::vector<Buffer> second_moments)
      : config_(config),
        stream_(stream),
        weights_(std::move(weights)),
        gradients_(std::move(gradients)),
        first_moments_(std::move(first_moments)),
        second_moments_(std::move(second_moments)) {}

  AdamWConfig config_;
  cudaStream_t stream_;
  int step_ = 0;
  std::vector<Buffer> weights_;
  std::vector<Buffer> gradients_;
  std::vector<Buffer> first_moments_;
  std::vector<Buffer> second_moments_;
};

}  // namespace pluto::llm
