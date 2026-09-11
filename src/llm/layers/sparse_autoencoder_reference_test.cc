#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/llm/layer.h"
#include "src/llm/layers/reference_test_util.h"
#include "src/llm/layers/sparse_autoencoder.h"

namespace pluto::llm {
namespace {

TEST_F(LayerReferenceTest,
       AutoEncoderAndLossForwardBackwardMatchAcrossShapesAndTypes) {
  constexpr float kSparsityPenalty = 0.07f;
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (const auto [rows, input_dim, feature_dim] :
         {std::tuple{16, 16, 16}, std::tuple{32, 16, 32},
          std::tuple{16, 32, 16}, std::tuple{48, 32, 96}}) {
      SCOPED_TRACE(testing::Message()
                   << "type=" << static_cast<int>(type) << " rows=" << rows
                   << " input=" << input_dim << " features=" << feature_dim);
      auto device_layer = SparseAutoEncoderLayer::Create(
          *executor_, input_dim, feature_dim, type,
          SparseAutoEncoderLayer::Mode::kCollectStatistics);
      auto reference_layer =
          SparseAutoEncoderLayerReference::Create(input_dim, feature_dim, type);
      auto device_loss = SparseAutoEncoderLossLayer::Create(
          *executor_, input_dim, feature_dim, kSparsityPenalty, type);
      auto reference_loss = SparseAutoEncoderLossLayerReference::Create(
          input_dim, feature_dim, kSparsityPenalty, type);
      ASSERT_TRUE(device_layer.ok()) << device_layer.status();
      ASSERT_TRUE(reference_layer.ok()) << reference_layer.status();
      ASSERT_TRUE(device_loss.ok()) << device_loss.status();
      ASSERT_TRUE(reference_loss.ok()) << reference_loss.status();

      auto device_weights = (*device_layer)->weights();
      auto reference_weights = (*reference_layer)->weights();
      ASSERT_EQ(device_weights.size(), 4u);
      ASSERT_EQ(reference_weights.size(), 4u);
      std::vector<std::vector<float>> parameters(4);
      parameters[0].resize(static_cast<size_t>(feature_dim) * input_dim);
      parameters[1].resize(feature_dim);
      parameters[2].resize(static_cast<size_t>(input_dim) * feature_dim);
      parameters[3].resize(input_dim);
      for (size_t index = 0; index < parameters[0].size(); ++index) {
        parameters[0][index] =
            0.09f * std::sin(0.17f * static_cast<float>(index));
      }
      for (int feature = 0; feature < feature_dim; ++feature) {
        parameters[1][feature] =
            0.18f * std::cos(0.41f * static_cast<float>(feature)) - 0.02f;
      }
      for (size_t index = 0; index < parameters[2].size(); ++index) {
        parameters[2][index] =
            0.08f * std::cos(0.13f * static_cast<float>(index));
      }
      for (int column = 0; column < input_dim; ++column) {
        parameters[3][column] =
            0.03f * std::sin(0.29f * static_cast<float>(column));
      }
      for (size_t index = 0; index < parameters.size(); ++index) {
        ASSERT_TRUE(SetFloatBufferPair(*executor_, device_weights[index],
                                       &reference_weights[index],
                                       parameters[index])
                        .ok());
      }

      std::vector<float> input(static_cast<size_t>(rows) * input_dim);
      for (size_t index = 0; index < input.size(); ++index) {
        input[index] = 0.7f * std::sin(0.11f * static_cast<float>(index)) +
                       0.2f * std::cos(0.07f * static_cast<float>(index));
      }
      auto input_pair = MakeActivationBufferPair(*executor_, input, type);
      ASSERT_TRUE(input_pair.ok()) << input_pair.status();

      BackwardState device_state;
      ReferenceBackwardState reference_state;
      BufferVec device_inputs = {input_pair->device};
      HostBufferVec reference_inputs = {input_pair->host};
      auto device_reconstruction =
          (*device_layer)->fwd(*executor_, device_inputs, device_state);
      auto reference_reconstruction =
          (*reference_layer)->fwd(reference_inputs, reference_state);
      ASSERT_TRUE(device_reconstruction.ok()) << device_reconstruction.status();
      ASSERT_TRUE(reference_reconstruction.ok())
          << reference_reconstruction.status();
      auto device_latents = (*device_layer)->latent_activations(device_state);
      auto reference_latents =
          (*reference_layer)->latent_activations(reference_state);
      ASSERT_TRUE(device_latents.ok()) << device_latents.status();
      ASSERT_TRUE(reference_latents.ok()) << reference_latents.status();
      EXPECT_TRUE(ActivationBuffersNear(*device_reconstruction,
                                        *reference_reconstruction, type, 2e-2f,
                                        2e-2f));
      EXPECT_TRUE(ActivationBuffersNear(*device_latents, *reference_latents,
                                        type, 1e-2f, 2e-2f));

      BackwardState device_loss_state;
      ReferenceBackwardState reference_loss_state;
      BufferVec device_loss_inputs = {input_pair->device,
                                      *device_reconstruction, *device_latents,
                                      (*device_layer)->decoder()};
      HostBufferVec reference_loss_inputs = {
          input_pair->host, *reference_reconstruction, *reference_latents,
          (*reference_layer)->decoder()};
      auto device_loss_value =
          (*device_loss)
              ->fwd(*executor_, device_loss_inputs, device_loss_state);
      auto reference_loss_value =
          (*reference_loss)->fwd(reference_loss_inputs, reference_loss_state);
      ASSERT_TRUE(device_loss_value.ok()) << device_loss_value.status();
      ASSERT_TRUE(reference_loss_value.ok()) << reference_loss_value.status();
      EXPECT_TRUE(FloatBuffersNear(*device_loss_value, *reference_loss_value,
                                   5e-2f, 2e-2f));

      auto device_loss_gradients =
          (*device_loss)->bwd(*executor_, {}, std::move(device_loss_state));
      auto reference_loss_gradients =
          (*reference_loss)->bwd({}, std::move(reference_loss_state));
      ASSERT_TRUE(device_loss_gradients.ok()) << device_loss_gradients.status();
      ASSERT_TRUE(reference_loss_gradients.ok())
          << reference_loss_gradients.status();
      ASSERT_EQ(device_loss_gradients->size(), 4u);
      ASSERT_EQ(reference_loss_gradients->size(), 4u);
      for (size_t index = 0; index < device_loss_gradients->size(); ++index) {
        EXPECT_TRUE(FloatBuffersNear((*device_loss_gradients)[index],
                                     (*reference_loss_gradients)[index], 2e-3f,
                                     2e-3f));
      }

      BufferVec device_autoencoder_gradients = {(*device_loss_gradients)[1],
                                                (*device_loss_gradients)[2],
                                                (*device_loss_gradients)[3]};
      HostBufferVec reference_autoencoder_gradients = {
          (*reference_loss_gradients)[1], (*reference_loss_gradients)[2],
          (*reference_loss_gradients)[3]};
      auto device_input_gradient =
          (*device_layer)
              ->bwd(*executor_, device_autoencoder_gradients,
                    std::move(device_state));
      auto reference_input_gradient = (*reference_layer)
                                          ->bwd(reference_autoencoder_gradients,
                                                std::move(reference_state));
      ASSERT_TRUE(device_input_gradient.ok()) << device_input_gradient.status();
      ASSERT_TRUE(reference_input_gradient.ok())
          << reference_input_gradient.status();
      ASSERT_EQ(device_input_gradient->size(), 1u);
      ASSERT_EQ(reference_input_gradient->size(), 1u);
      EXPECT_TRUE(FloatBuffersNear(device_input_gradient->front(),
                                   reference_input_gradient->front(), 3e-2f,
                                   2e-2f));

      auto device_parameter_gradients = (*device_layer)->gradients();
      auto reference_parameter_gradients = (*reference_layer)->gradients();
      ASSERT_EQ(device_parameter_gradients.size(), 4u);
      ASSERT_EQ(reference_parameter_gradients.size(), 4u);
      for (size_t index = 0; index < device_parameter_gradients.size();
           ++index) {
        EXPECT_TRUE(FloatBuffersNear(device_parameter_gradients[index],
                                     reference_parameter_gradients[index],
                                     4e-2f, 2e-2f));
      }
    }
  }
}

// The encoder itself is checked against the CPU reference above, including
// backward with a statistics state. Here a scalar reduction of the stored Z
// isolates statistics correctness from tensor-core rounding differences.
TEST_F(LayerReferenceTest,
       ZStatisticsMatchScalarReductionAcrossShapesAndTypes) {
  constexpr int kRows = 32;
  constexpr int kInputDim = 16;
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (int features : {16, 48, 4096}) {
      for (int pattern : {0, 1, 2}) {
        SCOPED_TRACE(testing::Message()
                     << "type=" << static_cast<int>(type)
                     << " features=" << features << " pattern=" << pattern);
        auto layer = SparseAutoEncoderLayer::Create(
            *executor_, kInputDim, features, type,
            SparseAutoEncoderLayer::Mode::kCollectStatistics);
        ASSERT_TRUE(layer.ok()) << layer.status();
        // Zero parameters produce all-zero Z. A positive bias produces a
        // constant dense Z; random weights produce mixed, row-dependent Z.
        if (pattern == 1) {
          auto bias = MakeRawBufferPair<float>(
              *executor_, std::vector<float>(features, 2.0f));
          ASSERT_TRUE(bias.ok()) << bias.status();
          (*layer)->weights()[1] = bias->device;
        } else if (pattern == 2) {
          ASSERT_TRUE((*layer)->InitializeNormal(0.25f, 31).ok());
        }
        std::vector<float> values(kRows * kInputDim);
        for (size_t index = 0; index < values.size(); ++index) {
          values[index] = std::sin(0.17f * index);
          // Trailing rows intentionally differ; including padding must
          // change the result in the mixed case.
          if (index >= 19 * kInputDim)
            values[index] *= 8;
        }
        auto input = MakeActivationBufferPair(*executor_, values, type);
        ASSERT_TRUE(input.ok()) << input.status();
        BackwardState state;
        ASSERT_TRUE(
            (*layer)->fwd(*executor_, BufferVec{input->device}, state).ok());
        auto latents = (*layer)->latent_activations(state);
        ASSERT_TRUE(latents.ok()) << latents.status();
        auto z = ReadDeviceActivations(*executor_, *latents, type);
        ASSERT_TRUE(z.ok()) << z.status();
        // A later forward cannot overwrite a previous state's statistics.
        auto zero_input = MakeActivationBufferPair(
            *executor_, std::vector<float>(kRows * kInputDim, 0), type);
        ASSERT_TRUE(zero_input.ok()) << zero_input.status();
        BackwardState later_state;
        ASSERT_TRUE(
            (*layer)
                ->fwd(*executor_, BufferVec{zero_input->device}, later_state)
                .ok());
        for (int valid_rows : {1, 19, kRows, 0}) {
          const int rows = valid_rows == 0 ? kRows : valid_rows;
          auto stats = (*layer)->ReadZStatistics(*executor_, state, valid_rows);
          ASSERT_TRUE(stats.ok()) << stats.status();
          int64_t active = 0;
          double sum = 0, squared_sum = 0, maximum = 0;
          for (int index = 0; index < rows * features; ++index) {
            const double value = (*z)[index];
            active += value > 0;
            sum += value;
            squared_sum += value * value;
            maximum = std::max(maximum, value);
          }
          const double elements = rows * features;
          const double mean = sum / elements;
          const double stddev =
              std::sqrt(std::max(0.0, squared_sum / elements - mean * mean));
          EXPECT_EQ(stats->rows, rows);
          EXPECT_EQ(stats->feature_dim, features);
          EXPECT_EQ(stats->active_count, active);
          EXPECT_NEAR(stats->mean, mean, 2e-5);
          EXPECT_NEAR(stats->standard_deviation, stddev, 2e-5);
          EXPECT_DOUBLE_EQ(stats->maximum, maximum);
          EXPECT_DOUBLE_EQ(stats->mean_active_features(),
                           static_cast<double>(active) / rows);
          EXPECT_NEAR(stats->zero_fraction(), 1.0 - active / elements, 1e-15);
          if (pattern < 2) {
            EXPECT_EQ(active, pattern == 0 ? 0 : rows * features);
            EXPECT_DOUBLE_EQ(stats->mean, pattern == 0 ? 0 : 2);
            EXPECT_DOUBLE_EQ(stats->standard_deviation, 0);
          }
        }
      }
    }
  }
}

TEST_F(LayerReferenceTest, LossAndGradientsMatchTheStatedSumExactly) {
  constexpr int kRows = 16;
  constexpr int kInputDim = 16;
  constexpr int kFeatureDim = 16;
  constexpr float kPenalty = 2.0f;
  auto loss = SparseAutoEncoderLossLayerReference::Create(
      kInputDim, kFeatureDim, kPenalty, DataType::FP16);
  ASSERT_TRUE(loss.ok()) << loss.status();

  std::vector<float> input(kRows * kInputDim, 1.0f);
  std::vector<float> reconstruction(kRows * kInputDim, 0.5f);
  std::vector<float> latents(kRows * kFeatureDim, 0.25f);
  std::vector<float> decoder(kInputDim * kFeatureDim, 0.0f);
  for (int index = 0; index < kInputDim; ++index)
    decoder[index * kFeatureDim + index] = 1.0f;
  auto input_pair = MakeActivationBufferPair(*executor_, input, DataType::FP16);
  auto reconstruction_pair =
      MakeActivationBufferPair(*executor_, reconstruction, DataType::FP16);
  auto latent_pair =
      MakeActivationBufferPair(*executor_, latents, DataType::FP16);
  auto decoder_pair = MakeRawBufferPair<float>(*executor_, decoder);
  ASSERT_TRUE(input_pair.ok());
  ASSERT_TRUE(reconstruction_pair.ok());
  ASSERT_TRUE(latent_pair.ok());
  ASSERT_TRUE(decoder_pair.ok());
  ReferenceBackwardState state;
  HostBufferVec inputs = {input_pair->host, reconstruction_pair->host,
                          latent_pair->host, decoder_pair->host};
  auto value = (*loss)->fwd(inputs, state);
  ASSERT_TRUE(value.ok()) << value.status();
  const float reconstruction_loss =
      kRows * kInputDim * (1.0f - 0.5f) * (1.0f - 0.5f);
  const float sparse_loss = kRows * kFeatureDim * kPenalty * 0.25f;
  ASSERT_EQ(value->size_bytes(), sizeof(float));
  EXPECT_FLOAT_EQ(*static_cast<const float*>(value->data()),
                  reconstruction_loss + sparse_loss);

  auto gradients = (*loss)->bwd({}, std::move(state));
  ASSERT_TRUE(gradients.ok()) << gradients.status();
  ASSERT_EQ(gradients->size(), 4u);
  EXPECT_TRUE(VectorsNear(ReadHostFloats((*gradients)[0]),
                          std::vector<float>(kRows * kInputDim, 1.0f), 0.0f));
  EXPECT_TRUE(VectorsNear(ReadHostFloats((*gradients)[1]),
                          std::vector<float>(kRows * kInputDim, -1.0f), 0.0f));
  EXPECT_TRUE(VectorsNear(ReadHostFloats((*gradients)[2]),
                          std::vector<float>(kRows * kFeatureDim, kPenalty),
                          0.0f));
  std::vector<float> expected_decoder_gradient(kInputDim * kFeatureDim, 0.0f);
  for (int index = 0; index < kInputDim; ++index) {
    expected_decoder_gradient[index * kFeatureDim + index] =
        kPenalty * kRows * 0.25f;
  }
  EXPECT_TRUE(VectorsNear(ReadHostFloats((*gradients)[3]),
                          expected_decoder_gradient, 0.0f));
}

// A column with entries (3a, -4a) has norm 5a, not 25a^2. Test exact
// derivatives, zero/tiny columns, disabled regularization, and independent
// positive feature rescalings. The old squared-norm loss fails these checks.
TEST_F(LayerReferenceTest,
       UnsquaredNormLossAndGradientsRespectFeatureRescaling) {
  constexpr int kRows = 16;
  constexpr int kInputDim = 32;
  constexpr int kFeatureDim = 32;
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (float penalty : {0.0f, 0.5f}) {
      auto device_loss = SparseAutoEncoderLossLayer::Create(
          *executor_, kInputDim, kFeatureDim, penalty, type);
      auto reference_loss = SparseAutoEncoderLossLayerReference::Create(
          kInputDim, kFeatureDim, penalty, type);
      ASSERT_TRUE(device_loss.ok()) << device_loss.status();
      ASSERT_TRUE(reference_loss.ok()) << reference_loss.status();
      auto input = MakeActivationBufferPair(
          *executor_, std::vector<float>(kRows * kInputDim, 1.0f), type);
      auto reconstruction = MakeActivationBufferPair(
          *executor_, std::vector<float>(kRows * kInputDim, 0.5f), type);
      ASSERT_TRUE(input.ok()) << input.status();
      ASSERT_TRUE(reconstruction.ok()) << reconstruction.status();
      for (float scale : {0.25f, 1.0f, 8.0f}) {
        SCOPED_TRACE(testing::Message()
                     << "type=" << static_cast<int>(type)
                     << " penalty=" << penalty << " scale=" << scale);
        std::vector<float> decoder(kInputDim * kFeatureDim, 0);
        std::vector<float> latents(kRows * kFeatureDim);
        std::vector<float> expected_d_decoder(decoder.size(), 0);
        std::vector<float> expected_d_latents(latents.size());
        double expected_loss = kRows * kInputDim * 0.25;
        for (int feature = 0; feature < kFeatureDim; ++feature) {
          // Include exactly zero and very small but nonzero norms. An
          // epsilon floor would change the small column's derivative.
          const float a = feature % 4 == 0   ? 0.0f
                          : feature % 4 == 1 ? 1e-8f
                          : feature % 4 == 2 ? 0.25f
                                             : 2.0f;
          const float c = scale * (feature % 2 == 0 ? 1.0f : 2.0f);
          const int first = feature * kFeatureDim + feature;
          const int second =
              ((feature + 1) % kInputDim) * kFeatureDim + feature;
          decoder[first] = 3.0f * a / c;
          decoder[second] = -4.0f * a / c;
          double base_latent_sum = 0;
          for (int row = 0; row < kRows; ++row) {
            const float base_z = 0.25f * ((row + feature) % 5);
            const int index = row * kFeatureDim + feature;
            latents[index] = base_z * c;
            expected_d_latents[index] = penalty * 5.0f * a / c;
            base_latent_sum += base_z;
          }
          // The expected loss deliberately contains no c: D_i/c and c*z_i
          // have the same penalty as the original feature, even per feature.
          expected_loss += penalty * 5.0 * a * base_latent_sum;
          if (a != 0) {
            expected_d_decoder[first] = penalty * base_latent_sum * c * 0.6;
            expected_d_decoder[second] = -penalty * base_latent_sum * c * 0.8;
          }
        }
        auto latent_pair = MakeActivationBufferPair(*executor_, latents, type);
        auto decoder_pair = MakeRawBufferPair<float>(*executor_, decoder);
        ASSERT_TRUE(latent_pair.ok()) << latent_pair.status();
        ASSERT_TRUE(decoder_pair.ok()) << decoder_pair.status();
        BackwardState device_state;
        ReferenceBackwardState reference_state;
        auto actual =
            (*device_loss)
                ->fwd(*executor_,
                      BufferVec{input->device, reconstruction->device,
                                latent_pair->device, decoder_pair->device},
                      device_state);
        auto reference =
            (*reference_loss)
                ->fwd(HostBufferVec{input->host, reconstruction->host,
                                    latent_pair->host, decoder_pair->host},
                      reference_state);
        ASSERT_TRUE(actual.ok()) << actual.status();
        ASSERT_TRUE(reference.ok()) << reference.status();
        auto host_value = ReadDeviceFloats(*executor_, *actual);
        ASSERT_TRUE(host_value.ok()) << host_value.status();
        EXPECT_NEAR((*host_value)[0], expected_loss, 1e-3);
        EXPECT_NEAR(ReadHostFloats(*reference)[0], expected_loss, 1e-3);
        auto gradients =
            (*device_loss)->bwd(*executor_, {}, std::move(device_state));
        auto reference_gradients =
            (*reference_loss)->bwd({}, std::move(reference_state));
        ASSERT_TRUE(gradients.ok()) << gradients.status();
        ASSERT_TRUE(reference_gradients.ok()) << reference_gradients.status();
        const std::vector<std::vector<float>> expected = {
            std::vector<float>(kRows * kInputDim, 1.0f),
            std::vector<float>(kRows * kInputDim, -1.0f), expected_d_latents,
            expected_d_decoder};
        for (int index = 0; index < 4; ++index) {
          auto host_gradient =
              ReadDeviceFloats(*executor_, (*gradients)[index]);
          ASSERT_TRUE(host_gradient.ok()) << host_gradient.status();
          EXPECT_TRUE(
              VectorsNear(host_gradient->span(), expected[index], 2e-5, 2e-5));
          EXPECT_TRUE(VectorsNear(ReadHostFloats((*reference_gradients)[index]),
                                  expected[index], 2e-5, 2e-5));
          for (float value : *host_gradient)
            EXPECT_TRUE(std::isfinite(value));
        }
      }
    }
  }
}

TEST_F(LayerReferenceTest, FP8IsRejectedConsistently) {
  auto device =
      SparseAutoEncoderLayer::Create(*executor_, 16, 16, DataType::FP8);
  auto reference =
      SparseAutoEncoderLayerReference::Create(16, 16, DataType::FP8);
  auto device_loss = SparseAutoEncoderLossLayer::Create(*executor_, 16, 16,
                                                        0.1f, DataType::FP8);
  auto reference_loss =
      SparseAutoEncoderLossLayerReference::Create(16, 16, 0.1f, DataType::FP8);
  ASSERT_FALSE(device.ok());
  ASSERT_FALSE(reference.ok());
  ASSERT_FALSE(device_loss.ok());
  ASSERT_FALSE(reference_loss.ok());
  EXPECT_EQ(device.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(reference.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(device_loss.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(reference_loss.status().code(), absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace pluto::llm
