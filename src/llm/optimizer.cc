#include "src/llm/optimizer.h"

#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/common/status_macros.h"
#include "src/llm/layers/internal.h"

namespace pluto::llm {
namespace {

__tile_global__ void AdamWUpdateKernel(
    float* __restrict__ weight, float* __restrict__ gradient,
    float* __restrict__ first_moment, float* __restrict__ second_moment,
    int elements, float learning_rate, float beta1, float beta2,
    float inverse_bias_correction1, float inverse_bias_correction2,
    float epsilon, float weight_decay) {
  namespace ct = cuda::tiles;
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
  auto g = gradient_view.load(block);
  auto m = beta1 * first_view.load(block) + (1.0f - beta1) * g;
  auto v = beta2 * second_view.load(block) + (1.0f - beta2) * g * g;
  auto w = weight_view.load(block);
  auto update = (m * inverse_bias_correction1) /
                    (ct::sqrt(v * inverse_bias_correction2) + epsilon) +
                weight_decay * w;
  weight_view.store(w - learning_rate * update, block);
  first_view.store(m, block);
  second_view.store(v, block);
  gradient_view.store(ct::zeros<ct::tile<float, ct::shape<16>>>(), block);
}

}  // namespace

absl::StatusOr<std::unique_ptr<AdamWOptimizer>> AdamWOptimizer::Create(
    Layer* model, AdamWConfig config, cudaStream_t stream) {
  if (model == nullptr) {
    return absl::InvalidArgumentError(
        "AdamWOptimizer requires a non-null model");
  }
  if (stream == nullptr || stream == cudaStreamLegacy ||
      stream == cudaStreamPerThread) {
    return absl::InvalidArgumentError(
        "AdamWOptimizer requires an explicit non-default CUDA stream");
  }
  if (!(config.learning_rate > 0.0f) || config.beta1 < 0.0f ||
      config.beta1 >= 1.0f || config.beta2 < 0.0f || config.beta2 >= 1.0f ||
      !(config.epsilon > 0.0f) || config.weight_decay < 0.0f) {
    return absl::InvalidArgumentError("invalid AdamW hyperparameters");
  }
  absl::Span<Buffer> model_weights = model->weights();
  absl::Span<Buffer> model_gradients = model->gradients();
  if (model_weights.size() != model_gradients.size()) {
    return absl::InvalidArgumentError(
        "model weights and gradients must have matching cardinality");
  }

  std::unordered_set<void*> seen;
  std::vector<Buffer> weights;
  std::vector<Buffer> gradients;
  std::vector<Buffer> first_moments;
  std::vector<Buffer> second_moments;
  for (size_t index = 0; index < model_weights.size(); ++index) {
    const Buffer& weight = model_weights[index];
    const Buffer& gradient = model_gradients[index];
    if (!seen.insert(weight.data()).second) continue;
    if (weight.stream() != stream || gradient.stream() != stream ||
        weight.size_bytes() != gradient.size_bytes() ||
        weight.size_bytes() % sizeof(float) != 0) {
      return absl::InvalidArgumentError(
          "AdamW parameters must be matching FP32 buffers on its stream");
    }
    const int elements =
        static_cast<int>(weight.size_bytes() / sizeof(float));
    RETURN_IF_ERROR(
        internal::ValidateTiledExtent(elements, "AdamW parameter elements"));
    ASSIGN_OR_RETURN(auto first,
                     Buffer::Allocate(weight.size_bytes(), stream));
    ASSIGN_OR_RETURN(auto second,
                     Buffer::Allocate(weight.size_bytes(), stream));
    RETURN_IF_ERROR(internal::CudaStatus(
        cudaMemsetAsync(first.data(), 0, first.size_bytes(), stream),
        "cudaMemsetAsync(AdamW first moment)"));
    RETURN_IF_ERROR(internal::CudaStatus(
        cudaMemsetAsync(second.data(), 0, second.size_bytes(), stream),
        "cudaMemsetAsync(AdamW second moment)"));
    weights.push_back(weight);
    gradients.push_back(gradient);
    first_moments.push_back(std::move(first));
    second_moments.push_back(std::move(second));
  }
  if (weights.empty()) {
    return absl::InvalidArgumentError("AdamW model has no parameters");
  }
  auto optimizer = std::unique_ptr<AdamWOptimizer>(new AdamWOptimizer(
      config, stream, std::move(weights), std::move(gradients),
      std::move(first_moments), std::move(second_moments)));
  RETURN_IF_ERROR(optimizer->ZeroGrad());
  return optimizer;
}

absl::StatusOr<std::unique_ptr<Optimizer>> Optimizer::Create(
    Layer* model, AdamWConfig config, cudaStream_t stream) {
  ASSIGN_OR_RETURN(auto optimizer,
                   AdamWOptimizer::Create(model, config, stream));
  return std::unique_ptr<Optimizer>(std::move(optimizer));
}

absl::Status AdamWOptimizer::ZeroGrad() {
  for (const Buffer& gradient : gradients_) {
    RETURN_IF_ERROR(internal::CudaStatus(
        cudaMemsetAsync(gradient.data(), 0, gradient.size_bytes(), stream_),
        "cudaMemsetAsync(AdamW gradient)"));
  }
  return absl::OkStatus();
}

absl::Status AdamWOptimizer::Step() {
  ++step_;
  const float inverse_bias_correction1 =
      1.0f / (1.0f - std::pow(config_.beta1, step_));
  const float inverse_bias_correction2 =
      1.0f / (1.0f - std::pow(config_.beta2, step_));
  for (size_t index = 0; index < weights_.size(); ++index) {
    const int elements =
        static_cast<int>(weights_[index].size_bytes() / sizeof(float));
    AdamWUpdateKernel<<<internal::TileCount(elements), 1, 0, stream_>>>(
        static_cast<float*>(weights_[index].data()),
        static_cast<float*>(gradients_[index].data()),
        static_cast<float*>(first_moments_[index].data()),
        static_cast<float*>(second_moments_[index].data()), elements,
        config_.learning_rate, config_.beta1, config_.beta2,
        inverse_bias_correction1, inverse_bias_correction2, config_.epsilon,
        config_.weight_decay);
  }
  return internal::CudaStatus(cudaGetLastError(),
                              "AdamWUpdateKernel launch");
}

}  // namespace pluto::llm
