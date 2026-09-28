#include "src/llm/layers/delta_net.h"

#include <cstdint>
#include <utility>

#include "absl/memory/memory.h"
#include "src/llm/layers/inference.h"
#include "src/util/status_macros.h"

namespace pluto::llm {

DeltaNetLayer::DeltaNetLayer(
    cuda::Executor& executor, DeltaNetParameters parameters, Buffer convolution,
    Buffer a_log, Buffer dt_bias, Buffer norm,
    std::unique_ptr<cached_attention_ops::DeltaNetState> cache)
    : executor_(executor),
      parameters_(parameters),
      weights_{std::move(convolution), std::move(a_log), std::move(dt_bias),
               std::move(norm)},
      cache_(std::move(cache)),
      input_types_{
          {DataType::BF16,
           {ActivationType::kBatchDimension, 1,
            2LL * parameters.key_heads * parameters.key_head_dim +
                1LL * parameters.value_heads * parameters.value_head_dim}},
          {DataType::BF16,
           {ActivationType::kBatchDimension, 1,
            1LL * parameters.value_heads * parameters.value_head_dim}},
          {DataType::BF16,
           {ActivationType::kBatchDimension, 1, parameters.value_heads}},
          {DataType::BF16,
           {ActivationType::kBatchDimension, 1, parameters.value_heads}}},
      output_types_{
          {DataType::BF16,
           {ActivationType::kBatchDimension, 1,
            1LL * parameters.value_heads * parameters.value_head_dim}}} {}

absl::StatusOr<std::unique_ptr<DeltaNetLayer>> DeltaNetLayer::Create(
    cuda::Executor& executor, DeltaNetParameters p, Buffer convolution,
    Buffer a_log, Buffer dt_bias, Buffer norm) {
  // Bound converted activation extents before allocating recurrent state. Each
  // product fits int64_t; bound those first so their sum also cannot overflow.
  if (p.key_heads <= 0 || p.value_heads <= 0 || p.key_head_dim <= 0 ||
      p.value_head_dim <= 0 || p.conv_kernel_dim <= 0)
    return absl::InvalidArgumentError(
        "DeltaNetLayer dimensions must be positive");
  const int64_t key_width = 1LL * p.key_heads * p.key_head_dim;
  const int64_t value_width = 1LL * p.value_heads * p.value_head_dim;
  if (key_width > inference_internal::kMaximumDimension / 2 ||
      value_width > inference_internal::kMaximumDimension ||
      2 * key_width + value_width > inference_internal::kMaximumDimension)
    return absl::InvalidArgumentError(
        "DeltaNetLayer activation exceeds the inference dimension limit");
  const int64_t channels = 2 * key_width + value_width;
  const Buffer weights[] = {convolution, a_log, dt_bias, norm};
  const int64_t counts[] = {channels * p.conv_kernel_dim, p.value_heads,
                            p.value_heads, p.value_head_dim};
  for (int i = 0; i < 4; ++i)
    if (&weights[i].executor() != &executor ||
        weights[i].size_bytes() !=
            static_cast<size_t>(counts[i]) * sizeof(float))
      return absl::InvalidArgumentError(
          "DeltaNetLayer weight has incorrect FP32 size or Executor");
  ASSIGN_OR_RETURN(auto cache,
                   cached_attention_ops::DeltaNetState::Create(executor, p));
  return absl::WrapUnique(
      new DeltaNetLayer(executor, p, std::move(convolution), std::move(a_log),
                        std::move(dt_bias), std::move(norm), std::move(cache)));
}

absl::StatusOr<FwdResult> DeltaNetLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks*) const {
  if (&executor != &executor_)
    return absl::InvalidArgumentError(
        "DeltaNetLayer used with a different Executor");
  const auto& p = parameters_;
  const int output_width = p.value_heads * p.value_head_dim;
  const int counts[] = {2 * p.key_heads * p.key_head_dim + output_width,
                        output_width, p.value_heads, p.value_heads};
  RETURN_IF_ERROR(inference_internal::ValidateInputs(executor, inputs, counts));
  BufferVec converted;
  for (int i = 0; i < 4; ++i) {
    ASSIGN_OR_RETURN(auto buffer, inference_internal::ToFloat(
                                      executor, inputs[i], counts[i]));
    converted.push_back(std::move(buffer));
  }
  ASSIGN_OR_RETURN(auto output, inference_internal::AllocateFloatVector(
                                    executor, output_width));
  RETURN_IF_ERROR(cache_->Step(static_cast<const float*>(converted[0].data()),
                               static_cast<const float*>(converted[1].data()),
                               static_cast<const float*>(converted[2].data()),
                               static_cast<const float*>(converted[3].data()),
                               static_cast<const float*>(weights_[0].data()),
                               static_cast<const float*>(weights_[1].data()),
                               static_cast<const float*>(weights_[2].data()),
                               static_cast<const float*>(weights_[3].data()),
                               static_cast<float*>(output.data())));
  ASSIGN_OR_RETURN(auto result, inference_internal::ToBFloat16(executor, output,
                                                               output_width));
  return FwdResult{{std::move(result)}, {}};
}

absl::StatusOr<BufferVec> DeltaNetLayer::bwd_impl(cuda::Executor&,
                                                  absl::Span<const Buffer>,
                                                  BackwardState, LayerHooks*) {
  return absl::UnimplementedError(
      "DeltaNetLayer supports cached inference only");
}

absl::Status DeltaNetLayer::Reset() { return cache_->Reset(); }

}  // namespace pluto::llm
