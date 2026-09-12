#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layers/reference_internal.h"
#include "src/llm/layers/sparse_autoencoder.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace ri = reference_internal;
namespace {

absl::Status ValidateReferenceLossInputs(absl::Span<const HostBuffer> inputs,
                                         int input_dim, int feature_dim,
                                         DataType data_type, int* rows) {
  if (inputs.size() != 4) {
    return absl::InvalidArgumentError(
        "sparse autoencoder loss expects x1, z, D, and target_x");
  }
  ASSIGN_OR_RETURN(*rows, ri::ActivationRows(inputs[3], input_dim, data_type,
                                             "sparse loss input"));
  RETURN_IF_ERROR(ri::ValidateBuffer(inputs[0],
                                     static_cast<size_t>(*rows) * input_dim *
                                         ri::ActivationElementBytes(data_type),
                                     "sparse loss reconstruction"));
  RETURN_IF_ERROR(ri::ValidateBuffer(inputs[1],
                                     static_cast<size_t>(*rows) * feature_dim *
                                         ri::ActivationElementBytes(data_type),
                                     "sparse loss latent activations"));
  return ri::ValidateBuffer(
      inputs[2], static_cast<size_t>(input_dim) * feature_dim * sizeof(float),
      "sparse loss decoder");
}

}  // namespace

absl::StatusOr<std::unique_ptr<SparseAutoEncoderLayerReference>>
SparseAutoEncoderLayerReference::Create(int input_dim, int feature_dim,
                                        DataType data_type) {
  RETURN_IF_ERROR(ri::ValidateComputeType(data_type));
  RETURN_IF_ERROR(ri::ValidateTiledExtent(input_dim, "input_dim"));
  RETURN_IF_ERROR(ri::ValidateTiledExtent(feature_dim, "feature_dim"));
  ASSIGN_OR_RETURN(
      auto encoder,
      ri::AllocateFloats(static_cast<size_t>(feature_dim) * input_dim, true));
  ASSIGN_OR_RETURN(auto encoder_bias, ri::AllocateFloats(feature_dim, true));
  ASSIGN_OR_RETURN(
      auto decoder,
      ri::AllocateFloats(static_cast<size_t>(input_dim) * feature_dim, true));
  ASSIGN_OR_RETURN(auto decoder_bias, ri::AllocateFloats(input_dim, true));
  ASSIGN_OR_RETURN(
      auto encoder_gradient,
      ri::AllocateFloats(static_cast<size_t>(feature_dim) * input_dim, true));
  ASSIGN_OR_RETURN(auto encoder_bias_gradient,
                   ri::AllocateFloats(feature_dim, true));
  ASSIGN_OR_RETURN(
      auto decoder_gradient,
      ri::AllocateFloats(static_cast<size_t>(input_dim) * feature_dim, true));
  ASSIGN_OR_RETURN(auto decoder_bias_gradient,
                   ri::AllocateFloats(input_dim, true));
  return absl::WrapUnique(new SparseAutoEncoderLayerReference(
      input_dim, feature_dim, data_type, std::move(encoder),
      std::move(encoder_bias), std::move(decoder), std::move(decoder_bias),
      std::move(encoder_gradient), std::move(encoder_bias_gradient),
      std::move(decoder_gradient), std::move(decoder_bias_gradient)));
}

absl::Status SparseAutoEncoderLayerReference::InitializeNormal(
    float standard_deviation, uint64_t seed) {
  if (!(standard_deviation > 0.0f)) {
    return absl::InvalidArgumentError(
        "sparse autoencoder initialization standard deviation must be "
        "positive");
  }
  std::mt19937_64 random(seed);
  std::normal_distribution<float> distribution(0.0f, standard_deviation);
  for (size_t parameter : {size_t{0}, size_t{2}}) {
    auto* values = static_cast<float*>(weights_[parameter].data());
    const size_t elements = weights_[parameter].size_bytes() / sizeof(float);
    for (size_t index = 0; index < elements; ++index)
      values[index] = distribution(random);
  }
  std::memset(weights_[1].data(), 0, weights_[1].size_bytes());
  std::memset(weights_[3].data(), 0, weights_[3].size_bytes());
  return absl::OkStatus();
}

absl::StatusOr<ReferenceFwdResult> SparseAutoEncoderLayerReference::fwd_impl(
    absl::Span<const HostBuffer> inputs) const {
  ReferenceBackwardState state;
  if (inputs.size() != 1) {
    return absl::InvalidArgumentError(
        "SparseAutoEncoderLayerReference fwd expects one input and saved "
        "state");
  }
  ASSIGN_OR_RETURN(int rows, ri::ActivationRows(inputs[0], input_dim_,
                                                output_type_, "SAE input"));
  RETURN_IF_ERROR(ri::ValidateTiledExtent(rows, "SAE rows"));
  ASSIGN_OR_RETURN(auto latents,
                   ri::AllocateActivation(
                       static_cast<size_t>(rows) * feature_dim_, output_type_));
  ASSIGN_OR_RETURN(auto reconstruction,
                   ri::AllocateActivation(
                       static_cast<size_t>(rows) * input_dim_, output_type_));
  const auto* encoder = static_cast<const float*>(weights_[0].data());
  const auto* encoder_bias = static_cast<const float*>(weights_[1].data());
  const auto* decoder = static_cast<const float*>(weights_[2].data());
  const auto* decoder_bias = static_cast<const float*>(weights_[3].data());

  // Direct scalar loops make both matrix orientations explicit. Quantization
  // is applied only at MMA operand boundaries, matching the CUDA kernels.
  for (int row = 0; row < rows; ++row) {
    for (int feature = 0; feature < feature_dim_; ++feature) {
      float sum = encoder_bias[feature];
      for (int column = 0; column < input_dim_; ++column) {
        const float centered =
            ri::LoadActivation(inputs[0],
                               static_cast<size_t>(row) * input_dim_ + column,
                               output_type_) -
            decoder_bias[column];
        sum += ri::QuantizeMmaOperand(centered, output_type_) *
               ri::QuantizeMmaOperand(
                   encoder[static_cast<size_t>(feature) * input_dim_ + column],
                   output_type_);
      }
      ri::StoreActivation(&latents,
                          static_cast<size_t>(row) * feature_dim_ + feature,
                          output_type_, std::max(sum, 0.0f));
    }
  }
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < input_dim_; ++column) {
      float sum = decoder_bias[column];
      for (int feature = 0; feature < feature_dim_; ++feature) {
        sum +=
            ri::QuantizeMmaOperand(
                ri::LoadActivation(
                    latents, static_cast<size_t>(row) * feature_dim_ + feature,
                    output_type_),
                output_type_) *
            ri::QuantizeMmaOperand(
                decoder[static_cast<size_t>(column) * feature_dim_ + feature],
                output_type_);
      }
      ri::StoreActivation(&reconstruction,
                          static_cast<size_t>(row) * input_dim_ + column,
                          output_type_, sum);
    }
  }
  state.intermediates = {inputs[0], latents};
  state.children.clear();
  return ReferenceFwdResult{
      {std::move(reconstruction), std::move(latents), weights_[2]},
      std::move(state)};
}


absl::StatusOr<HostBufferVec> SparseAutoEncoderLayerReference::bwd_impl(
    absl::Span<const HostBuffer> output_gradients,
    ReferenceBackwardState state) {
  if ((output_gradients.size() != 1 && output_gradients.size() != 3) ||
      state.intermediates.size() != 2) {
    return absl::InvalidArgumentError(
        "SAE bwd expects d_x1, optionally d_z and d_D, and a matching state");
  }
  ASSIGN_OR_RETURN(int rows, ri::MatrixRows(output_gradients[0], input_dim_,
                                            "SAE reconstruction gradient"));
  RETURN_IF_ERROR(
      ri::ValidateBuffer(state.intermediates[0],
                         static_cast<size_t>(rows) * input_dim_ *
                             ri::ActivationElementBytes(output_type_),
                         "SAE saved input"));
  RETURN_IF_ERROR(
      ri::ValidateBuffer(state.intermediates[1],
                         static_cast<size_t>(rows) * feature_dim_ *
                             ri::ActivationElementBytes(output_type_),
                         "SAE saved latent activations"));
  if (output_gradients.size() == 3) {
    RETURN_IF_ERROR(ri::ValidateBuffer(
        output_gradients[1],
        static_cast<size_t>(rows) * feature_dim_ * sizeof(float),
        "SAE auxiliary latent gradient"));
    RETURN_IF_ERROR(ri::ValidateBuffer(
        output_gradients[2],
        static_cast<size_t>(input_dim_) * feature_dim_ * sizeof(float),
        "SAE direct decoder gradient"));
  }

  ASSIGN_OR_RETURN(
      auto input_gradient,
      ri::AllocateFloats(static_cast<size_t>(rows) * input_dim_, true));
  ASSIGN_OR_RETURN(
      auto preactivation_gradient,
      ri::AllocateFloats(static_cast<size_t>(rows) * feature_dim_, true));
  const auto* d_reconstruction =
      static_cast<const float*>(output_gradients[0].data());
  const auto* auxiliary_latent =
      output_gradients.size() == 3
          ? static_cast<const float*>(output_gradients[1].data())
          : nullptr;
  const auto* direct_decoder =
      output_gradients.size() == 3
          ? static_cast<const float*>(output_gradients[2].data())
          : nullptr;
  const auto* encoder = static_cast<const float*>(weights_[0].data());
  const auto* decoder = static_cast<const float*>(weights_[2].data());
  const auto* decoder_bias = static_cast<const float*>(weights_[3].data());
  auto* d_input = static_cast<float*>(input_gradient.data());
  auto* d_preactivation = static_cast<float*>(preactivation_gradient.data());
  auto* d_encoder = static_cast<float*>(gradients_[0].data());
  auto* d_encoder_bias = static_cast<float*>(gradients_[1].data());
  auto* d_decoder = static_cast<float*>(gradients_[2].data());
  auto* d_decoder_bias = static_cast<float*>(gradients_[3].data());

  for (int row = 0; row < rows; ++row) {
    for (int feature = 0; feature < feature_dim_; ++feature) {
      float sum =
          auxiliary_latent == nullptr
              ? 0.0f
              : auxiliary_latent[static_cast<size_t>(row) * feature_dim_ +
                                 feature];
      for (int column = 0; column < input_dim_; ++column) {
        sum +=
            ri::QuantizeMmaOperand(
                d_reconstruction[static_cast<size_t>(row) * input_dim_ +
                                 column],
                output_type_) *
            ri::QuantizeMmaOperand(
                decoder[static_cast<size_t>(column) * feature_dim_ + feature],
                output_type_);
      }
      const float latent = ri::LoadActivation(
          state.intermediates[1],
          static_cast<size_t>(row) * feature_dim_ + feature, output_type_);
      d_preactivation[static_cast<size_t>(row) * feature_dim_ + feature] =
          latent > 0.0f ? sum : 0.0f;
    }
  }
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < input_dim_; ++column) {
      float sum = 0.0f;
      for (int feature = 0; feature < feature_dim_; ++feature) {
        sum += ri::QuantizeMmaOperand(
                   d_preactivation[static_cast<size_t>(row) * feature_dim_ +
                                   feature],
                   output_type_) *
               ri::QuantizeMmaOperand(
                   encoder[static_cast<size_t>(feature) * input_dim_ + column],
                   output_type_);
      }
      d_input[static_cast<size_t>(row) * input_dim_ + column] = sum;
    }
  }
  for (int feature = 0; feature < feature_dim_; ++feature) {
    float bias_sum = 0.0f;
    for (int column = 0; column < input_dim_; ++column) {
      float sum = 0.0f;
      for (int row = 0; row < rows; ++row) {
        const float centered =
            ri::LoadActivation(state.intermediates[0],
                               static_cast<size_t>(row) * input_dim_ + column,
                               output_type_) -
            decoder_bias[column];
        sum += ri::QuantizeMmaOperand(
                   d_preactivation[static_cast<size_t>(row) * feature_dim_ +
                                   feature],
                   output_type_) *
               ri::QuantizeMmaOperand(centered, output_type_);
      }
      d_encoder[static_cast<size_t>(feature) * input_dim_ + column] = sum;
    }
    for (int row = 0; row < rows; ++row) {
      bias_sum +=
          d_preactivation[static_cast<size_t>(row) * feature_dim_ + feature];
    }
    d_encoder_bias[feature] = bias_sum;
  }
  for (int column = 0; column < input_dim_; ++column) {
    float bias_sum = 0.0f;
    for (int feature = 0; feature < feature_dim_; ++feature) {
      float sum =
          direct_decoder == nullptr
              ? 0.0f
              : direct_decoder[static_cast<size_t>(column) * feature_dim_ +
                               feature];
      for (int row = 0; row < rows; ++row) {
        sum += ri::QuantizeMmaOperand(
                   d_reconstruction[static_cast<size_t>(row) * input_dim_ +
                                    column],
                   output_type_) *
               ri::QuantizeMmaOperand(
                   ri::LoadActivation(
                       state.intermediates[1],
                       static_cast<size_t>(row) * feature_dim_ + feature,
                       output_type_),
                   output_type_);
      }
      d_decoder[static_cast<size_t>(column) * feature_dim_ + feature] = sum;
    }
    for (int row = 0; row < rows; ++row) {
      bias_sum +=
          d_reconstruction[static_cast<size_t>(row) * input_dim_ + column] -
          d_input[static_cast<size_t>(row) * input_dim_ + column];
    }
    d_decoder_bias[column] = bias_sum;
  }
  return HostBufferVec{std::move(input_gradient)};
}

absl::StatusOr<std::unique_ptr<SparseAutoEncoderLossLayerReference>>
SparseAutoEncoderLossLayerReference::Create(int input_dim, int feature_dim,
                                            float sparsity_penalty,
                                            DataType data_type) {
  RETURN_IF_ERROR(ri::ValidateComputeType(data_type));
  RETURN_IF_ERROR(ri::ValidateTiledExtent(input_dim, "input_dim"));
  RETURN_IF_ERROR(ri::ValidateTiledExtent(feature_dim, "feature_dim"));
  if (!std::isfinite(sparsity_penalty) || sparsity_penalty < 0.0f) {
    return absl::InvalidArgumentError(
        "sparsity_penalty must be finite and non-negative");
  }
  return absl::WrapUnique(new SparseAutoEncoderLossLayerReference(
      input_dim, feature_dim, sparsity_penalty, data_type));
}

absl::StatusOr<ReferenceFwdResult>
SparseAutoEncoderLossLayerReference::fwd_impl(
    absl::Span<const HostBuffer> inputs) const {
  ReferenceBackwardState state;
  int rows;
  RETURN_IF_ERROR(ValidateReferenceLossInputs(inputs, input_dim_, feature_dim_,
                                              output_type_, &rows));
  ASSIGN_OR_RETURN(auto output, ri::AllocateFloats(rows));
  const auto* decoder = static_cast<const float*>(inputs[2].data());
  auto* row_losses = static_cast<float*>(output.data());

  // This is a literal transcription of the requested objective. Decoder
  // column norms are intentionally recomputed for every row: clarity matters
  // more than speed in the executable specification.
  for (int row = 0; row < rows; ++row) {
    float loss = 0.0f;
    for (int column = 0; column < input_dim_; ++column) {
      const size_t index = static_cast<size_t>(row) * input_dim_ + column;
      const float residual =
          ri::LoadActivation(inputs[3], index, output_type_) -
          ri::LoadActivation(inputs[0], index, output_type_);
      loss += residual * residual;
    }
    for (int feature = 0; feature < feature_dim_; ++feature) {
      float norm_squared = 0.0f;
      for (int column = 0; column < input_dim_; ++column) {
        const float value =
            decoder[static_cast<size_t>(column) * feature_dim_ + feature];
        norm_squared += value * value;
      }
      loss += sparsity_penalty_ *
              ri::LoadActivation(
                  inputs[1], static_cast<size_t>(row) * feature_dim_ + feature,
                  output_type_) *
              std::sqrt(norm_squared);
    }
    row_losses[row] = loss;
  }
  state.intermediates.assign(inputs.begin(), inputs.end());
  state.children.clear();
  return ReferenceFwdResult{{std::move(output)}, std::move(state)};
}

absl::StatusOr<HostBufferVec> SparseAutoEncoderLossLayerReference::bwd_impl(
    absl::Span<const HostBuffer> output_gradients,
    ReferenceBackwardState state) {
  if (!output_gradients.empty()) {
    return absl::InvalidArgumentError(
        "terminal sparse autoencoder loss expects no upstream gradient");
  }
  int rows;
  RETURN_IF_ERROR(ValidateReferenceLossInputs(
      state.intermediates, input_dim_, feature_dim_, output_type_, &rows));
  ASSIGN_OR_RETURN(auto input_gradient,
                   ri::AllocateFloats(static_cast<size_t>(rows) * input_dim_));
  ASSIGN_OR_RETURN(auto reconstruction_gradient,
                   ri::AllocateFloats(static_cast<size_t>(rows) * input_dim_));
  ASSIGN_OR_RETURN(auto latent_gradient,
                   ri::AllocateFloats(static_cast<size_t>(rows) * feature_dim_));
  ASSIGN_OR_RETURN(
      auto decoder_gradient,
      ri::AllocateFloats(static_cast<size_t>(input_dim_) * feature_dim_));
  auto* d_input = static_cast<float*>(input_gradient.data());
  auto* d_reconstruction = static_cast<float*>(reconstruction_gradient.data());
  auto* d_latent = static_cast<float*>(latent_gradient.data());
  auto* d_decoder = static_cast<float*>(decoder_gradient.data());
  const auto* decoder =
      static_cast<const float*>(state.intermediates[2].data());

  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < input_dim_; ++column) {
      const size_t index = static_cast<size_t>(row) * input_dim_ + column;
      d_input[index] =
          2.0f *
          (ri::LoadActivation(state.intermediates[3], index, output_type_) -
           ri::LoadActivation(state.intermediates[0], index, output_type_));
      d_reconstruction[index] = -d_input[index];
    }
    for (int feature = 0; feature < feature_dim_; ++feature) {
      float norm_squared = 0.0f;
      for (int column = 0; column < input_dim_; ++column) {
        const float value =
            decoder[static_cast<size_t>(column) * feature_dim_ + feature];
        norm_squared += value * value;
      }
      d_latent[static_cast<size_t>(row) * feature_dim_ + feature] =
          sparsity_penalty_ * std::sqrt(norm_squared);
    }
  }
  for (int feature = 0; feature < feature_dim_; ++feature) {
    float norm_squared = 0.0f;
    for (int column = 0; column < input_dim_; ++column) {
      const float value =
          decoder[static_cast<size_t>(column) * feature_dim_ + feature];
      norm_squared += value * value;
    }
    const float norm = std::sqrt(norm_squared);
    float latent_sum = 0.0f;
    for (int row = 0; row < rows; ++row) {
      latent_sum += ri::LoadActivation(
          state.intermediates[1],
          static_cast<size_t>(row) * feature_dim_ + feature, output_type_);
    }
    for (int column = 0; column < input_dim_; ++column) {
      const size_t index = static_cast<size_t>(column) * feature_dim_ + feature;
      // The derivative of a nonzero column's norm is its unit direction.
      // At zero the norm is nondifferentiable; choose the zero subgradient
      // instead of dividing by zero or perturbing the objective with epsilon.
      d_decoder[index] =
          norm == 0.0f ? 0.0f
                       : sparsity_penalty_ * latent_sum * decoder[index] / norm;
    }
  }
  return HostBufferVec{std::move(reconstruction_gradient),
                       std::move(latent_gradient), std::move(decoder_gradient),
                       std::move(input_gradient)};
}

}  // namespace pluto::llm
