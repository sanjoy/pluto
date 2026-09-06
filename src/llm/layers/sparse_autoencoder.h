#pragma once

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// A sparse autoencoder with an m-feature overcomplete representation:
//
//   z  = ReLU(W_enc (x - b_dec) + b_enc)
//   x1 = D z + b_dec
//
// x and x1 are row-major [rows, input_dim] activation matrices. W_enc is
// [feature_dim, input_dim], b_enc is [feature_dim], D is
// [input_dim, feature_dim], and b_dec is [input_dim]. Parameters and parameter
// gradients use FP32; x, z, and x1 use output_type().
class SparseAutoEncoderLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<SparseAutoEncoderLayer>> Create(
      cuda::Executor& executor, int input_dim, int feature_dim,
      DataType data_type);

  // Initializes W_enc and D independently from N(0, standard_deviation^2).
  // Both biases remain zero.
  absl::Status InitializeNormal(float standard_deviation, uint64_t seed);

  absl::StatusOr<Buffer> fwd(cuda::Executor& executor,
                             absl::Span<const Buffer> inputs,
                             Tape* tape) const override;

  // The first gradient is dL/dx1. Auxiliary sparse losses may additionally
  // supply dL/dz and a direct dL/dD as the second and third buffers. The
  // latter is needed because a decoder-norm regularizer depends directly on D
  // as well as indirectly through x1.
  absl::StatusOr<BufferVec> bwd(cuda::Executor& executor,
                                absl::Span<const Buffer> output_gradients,
                                Tape tape) override;

  // Parameter order follows the table in the class comment:
  // W_enc, b_enc, D, b_dec.
  absl::Span<Buffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<Buffer> gradients() override { return absl::MakeSpan(gradients_); }
  DataType output_type() const override { return output_type_; }

  int input_dim() const { return input_dim_; }
  int feature_dim() const { return feature_dim_; }
  const Buffer& decoder() const { return weights_[2]; }

  // Returns the z produced by fwd(). The returned Buffer shares its allocation
  // with the tape, so it remains valid independently of this handle.
  absl::StatusOr<Buffer> latent_activations(const Tape& tape) const;

 private:
  SparseAutoEncoderLayer(cuda::Executor& executor, int input_dim,
                         int feature_dim, DataType data_type, Buffer encoder,
                         Buffer encoder_bias, Buffer decoder,
                         Buffer decoder_bias, Buffer encoder_gradient,
                         Buffer encoder_bias_gradient, Buffer decoder_gradient,
                         Buffer decoder_bias_gradient)
      : input_dim_(input_dim),
        feature_dim_(feature_dim),
        output_type_(data_type),
        executor_(executor),
        weights_{std::move(encoder), std::move(encoder_bias),
                 std::move(decoder), std::move(decoder_bias)},
        gradients_{
            std::move(encoder_gradient), std::move(encoder_bias_gradient),
            std::move(decoder_gradient), std::move(decoder_bias_gradient)} {}

  int input_dim_;
  int feature_dim_;
  DataType output_type_;
  cuda::Executor& executor_;
  BufferVec weights_;
  BufferVec gradients_;
};

// Terminal sparse-autoencoder objective. fwd() expects {x, x1, z, D} and
// returns one FP32 scalar equal to
//
//   sum_rows (||x - x1||^2 +
//             sparsity_penalty * sum_i z_i ||D[:, i]||^2).
//
// bwd() accepts no upstream gradient and returns {dL/dx, dL/dx1, dL/dz,
// dL/dD}. The last three buffers can be passed to SparseAutoEncoderLayer::bwd;
// dL/dx is useful when x itself came from an earlier trainable layer.
class SparseAutoEncoderLossLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<SparseAutoEncoderLossLayer>> Create(
      cuda::Executor& executor, int input_dim, int feature_dim,
      float sparsity_penalty, DataType data_type);

  absl::StatusOr<Buffer> fwd(cuda::Executor& executor,
                             absl::Span<const Buffer> inputs,
                             Tape* tape) const override;
  absl::StatusOr<BufferVec> bwd(cuda::Executor& executor,
                                absl::Span<const Buffer> output_gradients,
                                Tape tape) override;
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return output_type_; }

  float sparsity_penalty() const { return sparsity_penalty_; }

 private:
  SparseAutoEncoderLossLayer(cuda::Executor& executor, int input_dim,
                             int feature_dim, float sparsity_penalty,
                             DataType data_type)
      : input_dim_(input_dim),
        feature_dim_(feature_dim),
        sparsity_penalty_(sparsity_penalty),
        output_type_(data_type),
        executor_(executor) {}

  int input_dim_;
  int feature_dim_;
  float sparsity_penalty_;
  DataType output_type_;
  cuda::Executor& executor_;
};

// Scalar CPU specification for SparseAutoEncoderLayer.
class SparseAutoEncoderLayerReference final : public LayerReference {
 public:
  static absl::StatusOr<std::unique_ptr<SparseAutoEncoderLayerReference>>
  Create(int input_dim, int feature_dim, DataType data_type);

  absl::Status InitializeNormal(float standard_deviation, uint64_t seed);
  absl::StatusOr<HostBuffer> fwd(absl::Span<const HostBuffer> inputs,
                                 ReferenceTape* tape) override;
  absl::StatusOr<HostBufferVec> bwd(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceTape tape) override;
  absl::Span<HostBuffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<HostBuffer> gradients() override {
    return absl::MakeSpan(gradients_);
  }
  DataType output_type() const override { return output_type_; }

  int input_dim() const { return input_dim_; }
  int feature_dim() const { return feature_dim_; }
  const HostBuffer& decoder() const { return weights_[2]; }
  absl::StatusOr<HostBuffer> latent_activations(
      const ReferenceTape& tape) const;

 private:
  SparseAutoEncoderLayerReference(
      int input_dim, int feature_dim, DataType data_type, HostBuffer encoder,
      HostBuffer encoder_bias, HostBuffer decoder, HostBuffer decoder_bias,
      HostBuffer encoder_gradient, HostBuffer encoder_bias_gradient,
      HostBuffer decoder_gradient, HostBuffer decoder_bias_gradient)
      : input_dim_(input_dim),
        feature_dim_(feature_dim),
        output_type_(data_type),
        weights_{std::move(encoder), std::move(encoder_bias),
                 std::move(decoder), std::move(decoder_bias)},
        gradients_{
            std::move(encoder_gradient), std::move(encoder_bias_gradient),
            std::move(decoder_gradient), std::move(decoder_bias_gradient)} {}

  int input_dim_;
  int feature_dim_;
  DataType output_type_;
  HostBufferVec weights_;
  HostBufferVec gradients_;
};

// Scalar CPU specification for SparseAutoEncoderLossLayer.
class SparseAutoEncoderLossLayerReference final : public LayerReference {
 public:
  static absl::StatusOr<std::unique_ptr<SparseAutoEncoderLossLayerReference>>
  Create(int input_dim, int feature_dim, float sparsity_penalty,
         DataType data_type);

  absl::StatusOr<HostBuffer> fwd(absl::Span<const HostBuffer> inputs,
                                 ReferenceTape* tape) override;
  absl::StatusOr<HostBufferVec> bwd(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceTape tape) override;
  absl::Span<HostBuffer> weights() override { return {}; }
  DataType output_type() const override { return output_type_; }

 private:
  SparseAutoEncoderLossLayerReference(int input_dim, int feature_dim,
                                      float sparsity_penalty,
                                      DataType data_type)
      : input_dim_(input_dim),
        feature_dim_(feature_dim),
        sparsity_penalty_(sparsity_penalty),
        output_type_(data_type) {}

  int input_dim_;
  int feature_dim_;
  float sparsity_penalty_;
  DataType output_type_;
};

}  // namespace pluto::llm
