#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/cuda/executor.h"
#include "src/llm/layer.h"
#include "src/llm/optimizer.h"

namespace pluto::llm {

struct AdamWConfig {
  float learning_rate = 3e-4f;
  float beta1 = 0.9f;
  float beta2 = 0.95f;
  float epsilon = 1e-8f;
  float weight_decay = 0.1f;
};

// AdamW over a model's FP32 master weights and FP32 accumulated gradients.
// Tied parameters are identified by allocation address and receive exactly one
// optimizer state/update even when multiple layers expose the same weight.
class AdamWOptimizer final : public Optimizer {
 public:
  static absl::StatusOr<std::unique_ptr<AdamWOptimizer>> Create(
      cuda::Executor& executor, Layer& model, AdamWConfig config);

  absl::Status ZeroGrad() override;
  absl::Status ApplyStep() override;

  int step() const override { return step_; }
  size_t parameter_tensor_count() const override { return weights_.size(); }

 private:
  AdamWOptimizer(cuda::Executor& executor, AdamWConfig config,
                 std::vector<Buffer> weights, std::vector<Buffer> gradients,
                 std::vector<Buffer> first_moments,
                 std::vector<Buffer> second_moments)
      : config_(config),
        executor_(executor),
        weights_(std::move(weights)),
        gradients_(std::move(gradients)),
        first_moments_(std::move(first_moments)),
        second_moments_(std::move(second_moments)) {}

  AdamWConfig config_;
  cuda::Executor& executor_;
  int step_ = 0;
  std::vector<Buffer> weights_;
  std::vector<Buffer> gradients_;
  std::vector<Buffer> first_moments_;
  std::vector<Buffer> second_moments_;
};

}  // namespace pluto::llm
