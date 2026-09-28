#pragma once

#include <memory>
#include <optional>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"
#include "src/llm/layers/util/inference_ops.h"

namespace pluto::llm {

// Imported checkpoint layers are inference-only: they retain the original
// matrix storage, expose its buffers through weights(), and allocate neither
// FP32 master weights nor gradients. Backward explicitly returns Unimplemented.
// Activations have physical BF16 storage and shape [batch, 1, width]. These
// single-token operators currently require batch == 1.
class InferenceLinearLayer final : public Layer {
 public:
  using MatrixStorage = inference_ops::MatrixStorage;

  // Weight layout is [output_dim, input_dim], with no bias. FP8 requires FP32
  // block scales [ceil(output_dim/128), ceil(input_dim/128)]; other storage
  // types reject scales. Buffers are shared, never copied or converted here.
  static absl::StatusOr<std::unique_ptr<InferenceLinearLayer>> Create(
      cuda::Executor& executor, Buffer weights, MatrixStorage storage,
      int input_dim, int output_dim,
      std::optional<Buffer> scales = std::nullopt);

  absl::string_view name() const override { return "InferenceLinearLayer"; }
  absl::Span<const ActivationType> input_types() const override {
    return absl::MakeConstSpan(&input_type_, 1);
  }
  absl::Span<const ActivationType> output_types() const override {
    return absl::MakeConstSpan(&output_type_, 1);
  }
  absl::Span<Buffer> weights() override { return absl::MakeSpan(weights_); }
  DataType output_type() const override { return DataType::BF16; }

 private:
  InferenceLinearLayer(cuda::Executor& executor, BufferVec weights,
                       MatrixStorage storage, int input_dim, int output_dim)
      : executor_(executor),
        weights_(std::move(weights)),
        storage_(storage),
        input_dim_(input_dim),
        output_dim_(output_dim),
        input_type_(DataType::BF16,
                    {ActivationType::kBatchDimension, 1, input_dim}),
        output_type_(DataType::BF16,
                     {ActivationType::kBatchDimension, 1, output_dim}) {}
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override;

  cuda::Executor& executor_;
  BufferVec weights_;
  MatrixStorage storage_;
  int input_dim_;
  int output_dim_;
  ActivationType input_type_;
  ActivationType output_type_;
};

// Frozen BF16 or FP32 token table. The token stays on the device; no host read
// or stream synchronization occurs in fwd(). Invalid IDs yield zeros instead
// of reading outside the table. Inference entry points should validate IDs.
class InferenceEmbeddingLayer final : public Layer {
 public:
  using MatrixStorage = inference_ops::MatrixStorage;
  static absl::StatusOr<std::unique_ptr<InferenceEmbeddingLayer>> Create(
      cuda::Executor& executor, Buffer weights, MatrixStorage storage,
      int vocab_size, int embedding_dim);
  absl::string_view name() const override { return "InferenceEmbeddingLayer"; }
  absl::Span<const ActivationType> input_types() const override {
    return absl::MakeConstSpan(&input_type_, 1);
  }
  absl::Span<const ActivationType> output_types() const override {
    return absl::MakeConstSpan(&output_type_, 1);
  }
  absl::Span<Buffer> weights() override { return absl::MakeSpan(&weight_, 1); }
  DataType output_type() const override { return DataType::BF16; }

 private:
  InferenceEmbeddingLayer(cuda::Executor& executor, Buffer weight,
                          MatrixStorage storage, int vocab_size,
                          int embedding_dim)
      : executor_(executor),
        weight_(std::move(weight)),
        storage_(storage),
        vocab_size_(vocab_size),
        embedding_dim_(embedding_dim),
        output_type_(DataType::BF16,
                     {ActivationType::kBatchDimension, 1, embedding_dim}) {}
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override;

  cuda::Executor& executor_;
  Buffer weight_;
  MatrixStorage storage_;
  int vocab_size_;
  int embedding_dim_;
  ActivationType input_type_{DataType::INT32,
                             {ActivationType::kBatchDimension, 1}};
  ActivationType output_type_;
};

// Zero-centered RMS normalization: x * rsqrt(mean(x*x) + epsilon) * (1+w).
// Unlike LayerNorm, this does not subtract the mean or add a learned bias.
// Statistics and the imported one-dimensional weight are FP32; output is BF16.
class RmsNormLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<RmsNormLayer>> Create(
      cuda::Executor& executor, int width, Buffer weight, float epsilon);
  absl::string_view name() const override { return "RmsNormLayer"; }
  absl::Span<const ActivationType> input_types() const override {
    return absl::MakeConstSpan(&type_, 1);
  }
  absl::Span<const ActivationType> output_types() const override {
    return input_types();
  }
  absl::Span<Buffer> weights() override { return absl::MakeSpan(&weight_, 1); }
  DataType output_type() const override { return DataType::BF16; }

 private:
  RmsNormLayer(cuda::Executor& executor, int width, Buffer weight,
               float epsilon)
      : executor_(executor),
        width_(width),
        weight_(std::move(weight)),
        epsilon_(epsilon),
        type_(DataType::BF16, {ActivationType::kBatchDimension, 1, width}) {}
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override;

  cuda::Executor& executor_;
  int width_;
  Buffer weight_;
  float epsilon_;
  ActivationType type_;
};

// Elementwise SiLU(gate) * up on two equally sized BF16 inputs. SiLU is rounded
// to BF16 before the multiplication, preserving the checkpoint's compute rule.
class SwiGluLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<SwiGluLayer>> Create(int width);
  absl::string_view name() const override { return "SwiGluLayer"; }
  absl::Span<const ActivationType> input_types() const override {
    return types_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return absl::MakeConstSpan(types_.data(), 1);
  }
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::BF16; }

 private:
  explicit SwiGluLayer(int width)
      : width_(width),
        types_{ActivationType(DataType::BF16,
                              {ActivationType::kBatchDimension, 1, width}),
               ActivationType(DataType::BF16,
                              {ActivationType::kBatchDimension, 1, width})} {}
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override;
  int width_;
  absl::InlinedVector<ActivationType, 2> types_;
};

namespace inference_internal {

// Shared limit for single-token activations and their conversion kernels.
// Factories must check this before accepting shapes or allocating caches.
inline constexpr int kMaximumDimension = 1048576;

// Shared by inference layers that use FP32 scratch while keeping their public
// activation signatures BF16. All allocations and casts use this executor.
absl::StatusOr<Buffer> AllocateFloatVector(cuda::Executor& executor,
                                           int elements);
absl::StatusOr<Buffer> ToFloat(cuda::Executor& executor, const Buffer& input,
                               int elements);
absl::StatusOr<Buffer> ToBFloat16(cuda::Executor& executor, const Buffer& input,
                                  int elements);
absl::Status ValidateInputs(cuda::Executor& executor,
                            absl::Span<const Buffer> inputs,
                            absl::Span<const int> element_counts);

}  // namespace inference_internal
}  // namespace pluto::llm
