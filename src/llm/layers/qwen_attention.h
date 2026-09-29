#pragma once

#include <memory>

#include "src/llm/block_parameter.h"
#include "src/llm/key_value_cache.h"
#include "src/llm/layer.h"
#include "src/llm/layers/util/cached_attention_ops.h"

namespace pluto::llm {

using FullAttentionParameters = cached_attention_ops::FullAttentionParameters;

// Batch-one Qwen grouped-query attention: per-head Q/K RMSNorm, partial RoPE,
// causal attention, and sigmoid output gating. Inputs are physical BF16
// [T, query_heads, 2, head_dim] query/gate and [T, KV_heads, head_dim] K/V;
// outputs are BF16 [T, query_heads, head_dim]. Statistics accumulate in FP32.
//
// A null cache selects stateless sequence attention with backward through the
// entire causal prefix (1..128 positions, head_dim <=256). BF16 casts use the
// usual straight-through mixed-precision derivative. BAdam owns optional norm
// gradients via BlockParameter; inactive norms still pass activation gradients.
//
// A nonnull cache selects single-token inference and rejects backward. It is
// borrowed, must outlive the layer, and must not be shared concurrently or
// between attention layers. Its position advances even though fwd is const.
// Reset the cache before a new sequence or after any failed forward (including
// hooks). Keep the creating Executor alive and serialize all calls on it.
class QwenAttentionLayer final : public Layer {
 public:
  // Bounds converted projection scratch for cached, single-token inference.
  static constexpr int kMaximumDimension = 1048576;

  // Shares resident FP32 norm parameters. The cache pointer alone selects the
  // execution mode; cached attention requires sequence_length == 1.
  static absl::StatusOr<std::unique_ptr<QwenAttentionLayer>> Create(
      cuda::Executor& executor, FullAttentionParameters parameters,
      std::shared_ptr<BlockParameter> q_norm,
      std::shared_ptr<BlockParameter> k_norm, int sequence_length,
      KeyValueCache* cache = nullptr);

  // Imports frozen FP32 norm buffers without allocating gradients or master
  // weights. With a null cache, backward still returns Q/gate, K/V gradients.
  static absl::StatusOr<std::unique_ptr<QwenAttentionLayer>> Create(
      cuda::Executor& executor, FullAttentionParameters parameters,
      Buffer q_norm, Buffer k_norm, int sequence_length,
      KeyValueCache* cache = nullptr);

  absl::string_view name() const override { return "QwenAttentionLayer"; }
  DataType output_type() const override { return DataType::BF16; }
  absl::Span<Buffer> weights() override { return weights_; }
  absl::Span<const ActivationType> input_types() const override {
    return input_types_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return output_types_;
  }

 private:
  QwenAttentionLayer(cuda::Executor& executor,
                     FullAttentionParameters parameters,
                     std::shared_ptr<BlockParameter> q_norm,
                     std::shared_ptr<BlockParameter> k_norm,
                     int sequence_length, KeyValueCache* cache);

  // Dispatch by cache presence; no other state implicitly selects decoding.
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks* hooks) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> gradients,
                                     BackwardState state,
                                     LayerHooks* hooks) override;

  // Appends one token to the borrowed cache and optionally reports its
  // [1, query_heads, 1, prefix_length] attention probabilities.
  absl::StatusOr<FwdResult> CachedForward(cuda::Executor& executor,
                                          absl::Span<const Buffer> inputs,
                                          LayerHooks* hooks) const;
  // Saves the full causal sequence's intermediates for backward, without
  // reading or updating any persistent cache.
  absl::StatusOr<FwdResult> SequenceForward(cuda::Executor& executor,
                                            absl::Span<const Buffer> inputs,
                                            LayerHooks* hooks) const;
  // Returns FP32 Q/gate, K and V gradients; accumulates norm gradients only
  // for the BlockParameters currently activated by BAdam.
  absl::StatusOr<BufferVec> SequenceBackward(cuda::Executor& executor,
                                             absl::Span<const Buffer> gradients,
                                             BackwardState state);

  cuda::Executor& executor_;
  FullAttentionParameters parameters_;
  std::shared_ptr<BlockParameter> q_norm_, k_norm_;
  int sequence_length_;
  Buffer weights_[2];     // Resident FP32 query/key norm values, in that order.
  KeyValueCache* cache_;  // Nonowning; null means stateless sequence training.
  const ActivationType input_types_[3];
  const ActivationType output_types_[1];
};

}  // namespace pluto::llm
