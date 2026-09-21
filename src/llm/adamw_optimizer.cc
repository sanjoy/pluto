#include "src/llm/adamw_optimizer.h"

#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/llm/layers/util.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

__tile_global__ void AdamWUpdateKernel(
    float* __restrict__ weight, float* __restrict__ gradient,
    float* __restrict__ first_moment, float* __restrict__ second_moment,
    int elements, float learning_rate, float beta1, float beta2,
    float inverse_bias_correction1, float inverse_bias_correction2,
    float epsilon, float weight_decay) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;

  auto weight_view = ct::partition_view{
      ct::tensor_span{weight, ct::extents{elements}}, ct::shape{16_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{gradient, ct::extents{elements}}, ct::shape{16_ic}};
  auto first_view = ct::partition_view{
      ct::tensor_span{first_moment, ct::extents{elements}}, ct::shape{16_ic}};
  auto second_view = ct::partition_view{
      ct::tensor_span{second_moment, ct::extents{elements}}, ct::shape{16_ic}};

  const int block = ct::bid().x;
  // Biases and normalization parameters can be shorter than one compute tile.
  // Mask every access so the optimizer uses the logical allocation size, not
  // a padded parameter tensor with extra trainable degrees of freedom.
  auto g = gradient_view.load_masked(block);
  auto m = beta1 * first_view.load_masked(block) + (1.0f - beta1) * g;
  auto v = beta2 * second_view.load_masked(block) + (1.0f - beta2) * g * g;
  auto w = weight_view.load_masked(block);
  auto update = (m * inverse_bias_correction1) /
                    (ct::sqrt(v * inverse_bias_correction2) + epsilon) +
                weight_decay * w;
  weight_view.store_masked(w - learning_rate * update, block);
  first_view.store_masked(m, block);
  second_view.store_masked(v, block);
  gradient_view.store_masked(ct::zeros<ct::tile<float, ct::shape<16>>>(),
                             block);
}

}  // namespace

absl::StatusOr<std::unique_ptr<AdamWOptimizer>> AdamWOptimizer::Create(
    cuda::Executor& executor, Layer& model, AdamWConfig config) {
  if (!(config.learning_rate > 0.0f) || !std::isfinite(config.learning_rate) ||
      config.beta1 < 0.0f || config.beta1 >= 1.0f || config.beta2 < 0.0f ||
      config.beta2 >= 1.0f || !(config.epsilon > 0.0f) ||
      config.weight_decay < 0.0f) {
    return absl::InvalidArgumentError("invalid AdamW hyperparameters");
  }
  absl::Span<Buffer> model_weights = model.weights();
  absl::Span<Buffer> model_gradients = model.gradients();
  if (model_weights.size() != model_gradients.size()) {
    return absl::InvalidArgumentError(
        "model weights and gradients must have matching cardinality");
  }

  absl::flat_hash_set<void*> seen;
  std::vector<Buffer> weights;
  std::vector<Buffer> gradients;
  std::vector<Buffer> first_moments;
  std::vector<Buffer> second_moments;
  for (size_t index = 0; index < model_weights.size(); ++index) {
    Buffer& weight = model_weights[index];
    Buffer& gradient = model_gradients[index];
    if (!seen.insert(weight.data()).second) continue;
    if (&weight.executor() != &executor || &gradient.executor() != &executor ||
        weight.size_bytes() != gradient.size_bytes() ||
        weight.size_bytes() % sizeof(float) != 0) {
      return absl::InvalidArgumentError(
          "AdamW parameters must be matching FP32 buffers on its executor");
    }
    // Validate the size before narrowing it to the kernel's int argument.
    ASSIGN_OR_RETURN(const int elements,
                     internal::ElementCount(executor, weight, sizeof(float),
                                            "AdamW parameter"));
    (void)elements;
    ASSIGN_OR_RETURN(auto first,
                     Buffer::Allocate(executor, weight.size_bytes()));
    ASSIGN_OR_RETURN(auto second,
                     Buffer::Allocate(executor, weight.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemsetAsync(first.data(), 0, first.size_bytes(), executor.stream()),
        "cudaMemsetAsync(AdamW first moment)"));
    RETURN_IF_ERROR(
        cuda::CudaStatus(cudaMemsetAsync(second.data(), 0, second.size_bytes(),
                                         executor.stream()),
                         "cudaMemsetAsync(AdamW second moment)"));
    weights.push_back(weight);
    gradients.push_back(gradient);
    first_moments.push_back(std::move(first));
    second_moments.push_back(std::move(second));
  }
  if (weights.empty())
    return absl::InvalidArgumentError("AdamW model has no parameters");
  auto optimizer = absl::WrapUnique(new AdamWOptimizer(
      executor, config, std::move(weights), std::move(gradients),
      std::move(first_moments), std::move(second_moments)));
  RETURN_IF_ERROR(optimizer->ZeroGrad());
  return optimizer;
}

absl::Status AdamWOptimizer::SetLearningRate(float learning_rate) {
  if (!(learning_rate > 0.0f) || !std::isfinite(learning_rate))
    return absl::InvalidArgumentError(
        "AdamW learning rate must be finite and strictly positive");
  config_.learning_rate = learning_rate;
  return absl::OkStatus();
}

absl::Status AdamWOptimizer::ZeroGrad() {
  for (Buffer& gradient : gradients_) {
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemsetAsync(gradient.data(), 0, gradient.size_bytes(),
                        executor_.stream()),
        "cudaMemsetAsync(AdamW gradient)"));
  }
  return absl::OkStatus();
}

absl::Status AdamWOptimizer::ApplyStep() {
  ++step_;
  const float inverse_bias_correction1 =
      1.0f / (1.0f - std::pow(config_.beta1, step_));
  const float inverse_bias_correction2 =
      1.0f / (1.0f - std::pow(config_.beta2, step_));
  for (size_t index = 0; index < weights_.size(); ++index) {
    const int elements =
        static_cast<int>(weights_[index].size_bytes() / sizeof(float));
    AdamWUpdateKernel<<<internal::MaskedTileCount(elements), 1, 0,
                        executor_.stream()>>>(
        static_cast<float*>(weights_[index].data()),
        static_cast<float*>(gradients_[index].data()),
        static_cast<float*>(first_moments_[index].data()),
        static_cast<float*>(second_moments_[index].data()), elements,
        config_.learning_rate, config_.beta1, config_.beta2,
        inverse_bias_correction1, inverse_bias_correction2, config_.epsilon,
        config_.weight_decay);
  }
  return cuda::CudaStatus(cudaGetLastError(), "AdamWUpdateKernel launch");
}

}  // namespace pluto::llm
