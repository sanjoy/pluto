#include "src/llm/layers/full_attention.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/memory/memory.h"
#include "src/llm/layer_hooks.h"
#include "src/llm/layers/util.h"
#include "src/util/status_macros.h"

namespace pluto::llm {

FullAttentionLayer::FullAttentionLayer(
    cuda::Executor& executor, FullAttentionParameters parameters, Buffer q_norm,
    Buffer k_norm,
    std::unique_ptr<cached_attention_ops::FullAttentionState> cache)
    : executor_(executor),
      parameters_(parameters),
      weights_{std::move(q_norm), std::move(k_norm)},
      cache_(std::move(cache)),
      input_types_{{DataType::BF16,
                    {ActivationType::kBatchDimension, 1,
                     2LL * parameters.query_heads * parameters.head_dim}},
                   {DataType::BF16,
                    {ActivationType::kBatchDimension, 1,
                     1LL * parameters.key_value_heads * parameters.head_dim}},
                   {DataType::BF16,
                    {ActivationType::kBatchDimension, 1,
                     1LL * parameters.key_value_heads * parameters.head_dim}}},
      output_types_{{DataType::BF16,
                     {ActivationType::kBatchDimension, 1,
                      1LL * parameters.query_heads * parameters.head_dim}}} {}

absl::StatusOr<std::unique_ptr<FullAttentionLayer>> FullAttentionLayer::Create(
    cuda::Executor& executor, FullAttentionParameters parameters, Buffer q_norm,
    Buffer k_norm) {
  // Check all converted activation extents before allocating a KV cache. Use
  // wide products so invalid head counts cannot overflow the later int math.
  if (parameters.query_heads <= 0 || parameters.key_value_heads <= 0 ||
      parameters.head_dim <= 0)
    return absl::InvalidArgumentError(
        "FullAttentionLayer dimensions must be positive");
  const int64_t query_width =
      1LL * parameters.query_heads * parameters.head_dim;
  const int64_t key_width =
      1LL * parameters.key_value_heads * parameters.head_dim;
  if (query_width > kMaximumDimension / 2 || key_width > kMaximumDimension)
    return absl::InvalidArgumentError(
        "FullAttentionLayer activation exceeds the layer dimension limit");
  if (parameters.head_dim <= 0 ||
      q_norm.size_bytes() !=
          static_cast<size_t>(parameters.head_dim) * sizeof(float) ||
      k_norm.size_bytes() != q_norm.size_bytes() ||
      &q_norm.executor() != &executor || &k_norm.executor() != &executor)
    return absl::InvalidArgumentError(
        "FullAttentionLayer expects two FP32 head-width norm weights on its "
        "Executor");
  ASSIGN_OR_RETURN(auto cache, cached_attention_ops::FullAttentionState::Create(
                                   executor, parameters));
  return absl::WrapUnique(
      new FullAttentionLayer(executor, parameters, std::move(q_norm),
                             std::move(k_norm), std::move(cache)));
}

absl::StatusOr<FwdResult> FullAttentionLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks* hooks) const {
  if (&executor != &executor_)
    return absl::InvalidArgumentError(
        "FullAttentionLayer used with a different Executor");
  const auto& p = parameters_;
  const int output_width = p.query_heads * p.head_dim;
  const int counts[] = {2 * output_width, p.key_value_heads * p.head_dim,
                        p.key_value_heads * p.head_dim};
  RETURN_IF_ERROR(internal::ValidateBFloat16Inputs(executor, inputs, counts));
  if (cache_->length() >= p.capacity)
    return absl::ResourceExhaustedError(
        "full-attention cache capacity reached");
  ASSIGN_OR_RETURN(auto q_gate,
                   internal::ToFloat(executor, inputs[0], counts[0]));
  ASSIGN_OR_RETURN(auto k, internal::ToFloat(executor, inputs[1], counts[1]));
  ASSIGN_OR_RETURN(auto v, internal::ToFloat(executor, inputs[2], counts[2]));
  ASSIGN_OR_RETURN(auto output,
                   internal::AllocateFloatVector(executor, output_width));
  std::optional<Buffer> probabilities;
  const int length = cache_->length() + 1;
  if (hooks && hooks->attention_probabilities_hook) {
    ASSIGN_OR_RETURN(
        probabilities,
        Buffer::Allocate(executor, static_cast<size_t>(p.query_heads) * length *
                                       sizeof(float)));
  }
  RETURN_IF_ERROR(cache_->Step(
      static_cast<const float*>(q_gate.data()),
      static_cast<const float*>(k.data()), static_cast<const float*>(v.data()),
      static_cast<const float*>(weights_[0].data()),
      static_cast<const float*>(weights_[1].data()),
      static_cast<float*>(output.data()),
      probabilities ? static_cast<float*>(probabilities->data()) : nullptr));
  if (probabilities) {
    // Incremental decoding has one query and `length` keys, not a square
    // prefill matrix. Every reported key is causally visible to this query.
    const ActivationType type(DataType::FP32, {1, p.query_heads, 1, length});
    RETURN_IF_ERROR(hooks->attention_probabilities_hook(executor, name(), type,
                                                        *probabilities));
  }
  ASSIGN_OR_RETURN(auto result,
                   internal::ToBFloat16(executor, output, output_width));
  return FwdResult{{std::move(result)}, {}};
}

absl::StatusOr<BufferVec> FullAttentionLayer::bwd_impl(cuda::Executor&,
                                                       absl::Span<const Buffer>,
                                                       BackwardState,
                                                       LayerHooks*) {
  return absl::UnimplementedError(
      "FullAttentionLayer supports cached inference only");
}

absl::Status FullAttentionLayer::Reset() { return cache_->Reset(); }

}  // namespace pluto::llm
