#pragma once

#include <memory>
#include <optional>

#include "src/llm/block_parameter.h"
#include "src/llm/layers/matrix_common.h"

namespace pluto::llm {

// Convert an imported matrix into resident BF16 storage. FP8 scales follow the
// checkpoint's [ceil(rows/128), ceil(cols/128)] multiplicative block layout.
// BF16 inputs are shared. Conversion and temporary destruction are
// asynchronous.
absl::StatusOr<Buffer> DequantizeMatrix(cuda::Executor& executor,
                                        const Buffer& input,
                                        MatrixStorage storage,
                                        const std::optional<Buffer>& scales,
                                        int rows, int cols);

// Shared interface for single-parameter block-training layers. Resident weights
// are exposed for serialization, not a conventional all-weights optimizer:
// BlockParameter owns the dynamically activated FP32 optimizer storage.
class BlockWeightLayer : public Layer {
 public:
  absl::Span<Buffer> weights() override { return absl::MakeSpan(weights_); }
  DataType output_type() const override { return DataType::BF16; }
  absl::Span<const ActivationType> input_types() const override {
    return absl::MakeConstSpan(&input_type_, 1);
  }
  absl::Span<const ActivationType> output_types() const override {
    return absl::MakeConstSpan(&output_type_signature_, 1);
  }

 protected:
  BlockWeightLayer(std::shared_ptr<BlockParameter> parameter,
                   ActivationType input, ActivationType output)
      : parameter_(std::move(parameter)),
        weights_{parameter_->value()},
        input_type_(std::move(input)),
        output_type_signature_(std::move(output)) {}
  std::shared_ptr<BlockParameter> parameter_;
  BufferVec weights_;
  ActivationType input_type_;
  ActivationType output_type_signature_;
};

// Bias-free [output,input] matrix with BF16 forward and FP32 backward. A frozen
// matrix still propagates input gradients but never allocates/accumulates dW.
// The first implementation supports one sample with sequence_length tokens.
class BlockLinearLayer final : public BlockWeightLayer {
 public:
  static absl::StatusOr<std::unique_ptr<BlockLinearLayer>> Create(
      cuda::Executor& executor, std::shared_ptr<BlockParameter> weight,
      int input_dim, int output_dim, int sequence_length,
      bool output_float32 = false);
  absl::string_view name() const override { return "BlockLinearLayer"; }
  DataType output_type() const override {
    return output_float32_ ? DataType::FP32 : DataType::BF16;
  }

 private:
  BlockLinearLayer(std::shared_ptr<BlockParameter> weight, int input_dim,
                   int output_dim, int sequence_length, bool output_float32)
      : BlockWeightLayer(
            std::move(weight),
            ActivationType(DataType::BF16, {-2, sequence_length, input_dim}),
            ActivationType(output_float32 ? DataType::FP32 : DataType::BF16,
                           {-2, sequence_length, output_dim})),
        input_dim_(input_dim),
        output_dim_(output_dim),
        sequence_length_(sequence_length),
        output_float32_(output_float32) {}
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     LayerHooks*) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override;
  int input_dim_;
  int output_dim_;
  int sequence_length_;
  bool output_float32_;
};

// Device token lookup. Backward accumulates repeated token rows in sequence
// order without atomic additions; token IDs themselves have no gradient.
class BlockEmbeddingLayer final : public BlockWeightLayer {
 public:
  static absl::StatusOr<std::unique_ptr<BlockEmbeddingLayer>> Create(
      cuda::Executor& executor, std::shared_ptr<BlockParameter> weight,
      int vocab_size, int width, int sequence_length);
  absl::string_view name() const override { return "BlockEmbeddingLayer"; }

 private:
  BlockEmbeddingLayer(std::shared_ptr<BlockParameter> weight, int vocab,
                      int width, int sequence)
      : BlockWeightLayer(std::move(weight),
                         ActivationType(DataType::INT32, {-2, sequence}),
                         ActivationType(DataType::BF16, {-2, sequence, width})),
        vocab_size_(vocab),
        width_(width),
        sequence_length_(sequence) {}
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     LayerHooks*) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override;
  int vocab_size_;
  int width_;
  int sequence_length_;
};

// Qwen's zero-centered RMSNorm: x / rms(x) * (1 + weight). Statistics and
// gradients use FP32; the resident norm parameter itself is FP32 as imported.
class BlockRmsNormLayer final : public BlockWeightLayer {
 public:
  static absl::StatusOr<std::unique_ptr<BlockRmsNormLayer>> Create(
      cuda::Executor& executor, std::shared_ptr<BlockParameter> weight,
      int width, int sequence_length, float epsilon);
  absl::string_view name() const override { return "BlockRmsNormLayer"; }

 private:
  BlockRmsNormLayer(std::shared_ptr<BlockParameter> weight, int width,
                    int sequence, float epsilon)
      : BlockWeightLayer(std::move(weight),
                         ActivationType(DataType::BF16, {-2, sequence, width}),
                         ActivationType(DataType::BF16, {-2, sequence, width})),
        width_(width),
        sequence_length_(sequence),
        epsilon_(epsilon) {}
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     LayerHooks*) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override;
  int width_;
  int sequence_length_;
  float epsilon_;
};

// Takes separate gate and up projections. BF16 rounding follows inference:
// round(round(silu(gate)) * up). Backward treats casts as straight-through.
class BlockSwiGluLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<BlockSwiGluLayer>> Create(
      int width, int sequence_length);
  absl::string_view name() const override { return "BlockSwiGluLayer"; }
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::BF16; }
  absl::Span<const ActivationType> input_types() const override {
    return input_types_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return absl::MakeConstSpan(input_types_.data(), 1);
  }

 private:
  BlockSwiGluLayer(int width, int sequence)
      : width_(width),
        sequence_length_(sequence),
        input_types_{ActivationType(DataType::BF16, {-2, sequence, width}),
                     ActivationType(DataType::BF16, {-2, sequence, width})} {}
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     LayerHooks*) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override;
  int width_;
  int sequence_length_;
  absl::InlinedVector<ActivationType, 2> input_types_;
};

}  // namespace pluto::llm
