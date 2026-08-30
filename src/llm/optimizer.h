#ifndef PLUTO_SRC_LLM_OPTIMIZER_H_
#define PLUTO_SRC_LLM_OPTIMIZER_H_

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

// AdamW over a model's FP32 master weights and FP32 accumulated gradients.
// Tied parameters are identified by allocation address and receive exactly one
// optimizer state/update even when multiple layers expose the same weight.
class AdamWOptimizer final {
 public:
  static absl::StatusOr<std::unique_ptr<AdamWOptimizer>> Create(
      Layer& model, AdamWConfig config, cudaStream_t stream);

  // Clears all unique gradient accumulators. Call before the first backward;
  // Step() also clears gradients after applying an update.
  absl::Status ZeroGrad();
  absl::Status Step();

  int step() const { return step_; }
  size_t parameter_tensor_count() const { return weights_.size(); }

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

#endif  // PLUTO_SRC_LLM_OPTIMIZER_H_
