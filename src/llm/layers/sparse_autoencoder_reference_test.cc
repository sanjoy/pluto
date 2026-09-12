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

      BufferVec device_inputs = {input_pair->device};
      HostBufferVec reference_inputs = {input_pair->host};
      auto device_reconstruction =
          (*device_layer)->fwd(*executor_, device_inputs);

      auto reference_reconstruction = (*reference_layer)->fwd(reference_inputs);

      ASSERT_TRUE(device_reconstruction.ok()) << device_reconstruction.status();
      ASSERT_TRUE(reference_reconstruction.ok())
          << reference_reconstruction.status();
      ASSERT_EQ(device_reconstruction->outputs.size(), 3u);
      ASSERT_EQ(reference_reconstruction->outputs.size(), 3u);
      EXPECT_EQ(device_reconstruction->outputs[1].data(),
                device_reconstruction->state.intermediates[1].data());
      EXPECT_EQ(reference_reconstruction->outputs[1].data(),
                reference_reconstruction->state.intermediates[1].data());
      EXPECT_EQ(device_reconstruction->outputs[2].data(),
                (*device_layer)->decoder().data());
      EXPECT_EQ(reference_reconstruction->outputs[2].data(),
                (*reference_layer)->decoder().data());
      EXPECT_TRUE(ActivationBuffersNear(device_reconstruction->outputs[0],
                                        reference_reconstruction->outputs[0],
                                        type, 2e-2f, 2e-2f));
      EXPECT_TRUE(ActivationBuffersNear(device_reconstruction->outputs[1],
                                        reference_reconstruction->outputs[1],
                                        type, 1e-2f, 2e-2f));

      BufferVec device_loss_inputs = device_reconstruction->outputs;
      device_loss_inputs.push_back(input_pair->device);
      HostBufferVec reference_loss_inputs = reference_reconstruction->outputs;
      reference_loss_inputs.push_back(input_pair->host);
      auto device_loss_value =
          (*device_loss)->fwd(*executor_, device_loss_inputs);

      auto reference_loss_value = (*reference_loss)->fwd(reference_loss_inputs);

      ASSERT_TRUE(device_loss_value.ok()) << device_loss_value.status();
      ASSERT_TRUE(reference_loss_value.ok()) << reference_loss_value.status();
      EXPECT_TRUE(FloatBuffersNear(device_loss_value->outputs[0],
                                   reference_loss_value->outputs[0], 5e-2f,
                                   2e-2f));

      auto device_loss_gradients =
          (*device_loss)
              ->bwd(*executor_, {}, std::move(device_loss_value->state));
      auto reference_loss_gradients =
          (*reference_loss)->bwd({}, std::move(reference_loss_value->state));
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

      BufferVec device_autoencoder_gradients = {(*device_loss_gradients)[0],
                                                (*device_loss_gradients)[1],
                                                (*device_loss_gradients)[2]};
      HostBufferVec reference_autoencoder_gradients = {
          (*reference_loss_gradients)[0], (*reference_loss_gradients)[1],
          (*reference_loss_gradients)[2]};
      auto device_input_gradient =
          (*device_layer)
              ->bwd(*executor_, device_autoencoder_gradients,
                    std::move(device_reconstruction->state));
      auto reference_input_gradient =
          (*reference_layer)
              ->bwd(reference_autoencoder_gradients,
                    std::move(reference_reconstruction->state));
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

        auto state_fwd = (*layer)->fwd(*executor_, BufferVec{input->device});
        ASSERT_TRUE(state_fwd.ok());

        ASSERT_EQ(state_fwd->outputs.size(), 3u);
        auto z = ReadDeviceActivations(*executor_, state_fwd->outputs[1], type);
        ASSERT_TRUE(z.ok()) << z.status();
        // A later forward cannot overwrite a previous state's statistics.
        auto zero_input = MakeActivationBufferPair(
            *executor_, std::vector<float>(kRows * kInputDim, 0), type);
        ASSERT_TRUE(zero_input.ok()) << zero_input.status();

        auto later_state_fwd =
            (*layer)->fwd(*executor_, BufferVec{zero_input->device});
        ASSERT_TRUE(later_state_fwd.ok());

        for (int valid_rows : {1, 19, kRows, 0}) {
          const int rows = valid_rows == 0 ? kRows : valid_rows;
          auto stats = (*layer)->ReadZStatistics(*executor_, state_fwd->state,
                                                 valid_rows);
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

TEST_F(LayerReferenceTest, RowLossesAndSummedGradientsMatchExactly) {
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

  HostBufferVec inputs = {reconstruction_pair->host, latent_pair->host,
                          decoder_pair->host, input_pair->host};
  auto value = (*loss)->fwd(inputs);

  ASSERT_TRUE(value.ok()) << value.status();
  const float reconstruction_loss = kInputDim * (1.0f - 0.5f) * (1.0f - 0.5f);
  const float sparse_loss = kFeatureDim * kPenalty * 0.25f;
  ASSERT_EQ(value->outputs.size(), 1u);
  ASSERT_EQ(value->outputs[0].size_bytes(), kRows * sizeof(float));
  EXPECT_TRUE(VectorsNear(
      ReadHostFloats(value->outputs[0]),
      std::vector<float>(kRows, reconstruction_loss + sparse_loss), 0.0f));

  auto gradients = (*loss)->bwd({}, std::move(value->state));
  ASSERT_TRUE(gradients.ok()) << gradients.status();
  ASSERT_EQ(gradients->size(), 4u);
  EXPECT_TRUE(VectorsNear(ReadHostFloats((*gradients)[3]),
                          std::vector<float>(kRows * kInputDim, 1.0f), 0.0f));
  EXPECT_TRUE(VectorsNear(ReadHostFloats((*gradients)[0]),
                          std::vector<float>(kRows * kInputDim, -1.0f), 0.0f));
  EXPECT_TRUE(VectorsNear(ReadHostFloats((*gradients)[1]),
                          std::vector<float>(kRows * kFeatureDim, kPenalty),
                          0.0f));
  std::vector<float> expected_decoder_gradient(kInputDim * kFeatureDim, 0.0f);
  for (int index = 0; index < kInputDim; ++index) {
    expected_decoder_gradient[index * kFeatureDim + index] =
        kPenalty * kRows * 0.25f;
  }
  EXPECT_TRUE(VectorsNear(ReadHostFloats((*gradients)[2]),
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
        std::vector<float> expected_losses(kRows, kInputDim * 0.25f);
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
            expected_losses[row] += penalty * 5.0f * a * base_z;
          }
          // The expected loss deliberately contains no c: D_i/c and c*z_i
          // have the same penalty as the original feature, even per feature.
          if (a != 0) {
            expected_d_decoder[first] = penalty * base_latent_sum * c * 0.6;
            expected_d_decoder[second] = -penalty * base_latent_sum * c * 0.8;
          }
        }
        auto latent_pair = MakeActivationBufferPair(*executor_, latents, type);
        auto decoder_pair = MakeRawBufferPair<float>(*executor_, decoder);
        ASSERT_TRUE(latent_pair.ok()) << latent_pair.status();
        ASSERT_TRUE(decoder_pair.ok()) << decoder_pair.status();

        auto actual =
            (*device_loss)
                ->fwd(*executor_,
                      BufferVec{reconstruction->device, latent_pair->device,
                                decoder_pair->device, input->device});

        auto reference =
            (*reference_loss)
                ->fwd(HostBufferVec{reconstruction->host, latent_pair->host,
                                    decoder_pair->host, input->host});

        ASSERT_TRUE(actual.ok()) << actual.status();
        ASSERT_TRUE(reference.ok()) << reference.status();
        auto host_value = ReadDeviceFloats(*executor_, actual->outputs[0]);
        ASSERT_TRUE(host_value.ok()) << host_value.status();
        EXPECT_TRUE(VectorsNear(host_value->span(), expected_losses, 1e-3));
        EXPECT_TRUE(VectorsNear(ReadHostFloats(reference->outputs[0]),
                                expected_losses, 1e-3));
        auto gradients =
            (*device_loss)->bwd(*executor_, {}, std::move(actual->state));
        auto reference_gradients =
            (*reference_loss)->bwd({}, std::move(reference->state));
        ASSERT_TRUE(gradients.ok()) << gradients.status();
        ASSERT_TRUE(reference_gradients.ok()) << reference_gradients.status();
        const std::vector<std::vector<float>> expected = {
            std::vector<float>(kRows * kInputDim, -1.0f), expected_d_latents,
            expected_d_decoder, std::vector<float>(kRows * kInputDim, 1.0f)};
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

TEST_F(LayerReferenceTest, AutoEncoderOutputsSurviveInterleavedBackward) {
  constexpr int kRows = 16;
  constexpr int kWidth = 16;
  constexpr float kPenalty = 0.25f;
  auto device = SparseAutoEncoderLayer::Create(*executor_, kWidth, kWidth,
                                               DataType::FP16);
  auto reference =
      SparseAutoEncoderLayerReference::Create(kWidth, kWidth, DataType::FP16);
  auto device_loss = SparseAutoEncoderLossLayer::Create(
      *executor_, kWidth, kWidth, kPenalty, DataType::FP16);
  auto reference_loss = SparseAutoEncoderLossLayerReference::Create(
      kWidth, kWidth, kPenalty, DataType::FP16);
  ASSERT_TRUE(device.ok()) << device.status();
  ASSERT_TRUE(reference.ok()) << reference.status();
  ASSERT_TRUE(device_loss.ok()) << device_loss.status();
  ASSERT_TRUE(reference_loss.ok()) << reference_loss.status();

  std::vector<float> identity(kWidth * kWidth, 0.0f);
  for (int index = 0; index < kWidth; ++index)
    identity[index * kWidth + index] = 1.0f;
  for (int index : {0, 2}) {
    ASSERT_TRUE(SetFloatBufferPair(*executor_, (*device)->weights()[index],
                                   &(*reference)->weights()[index], identity)
                    .ok());
  }
  auto first_input = MakeActivationBufferPair(
      *executor_, std::vector<float>(kRows * kWidth, 1.0f), DataType::FP16);
  auto second_input = MakeActivationBufferPair(
      *executor_, std::vector<float>(kRows * kWidth, 2.0f), DataType::FP16);
  ASSERT_TRUE(first_input.ok()) << first_input.status();
  ASSERT_TRUE(second_input.ok()) << second_input.status();
  auto first = (*device)->fwd(*executor_, BufferVec{first_input->device});
  auto first_reference = (*reference)->fwd(HostBufferVec{first_input->host});
  auto second = (*device)->fwd(*executor_, BufferVec{second_input->device});
  auto second_reference = (*reference)->fwd(HostBufferVec{second_input->host});
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(first_reference.ok()) << first_reference.status();
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_TRUE(second_reference.ok()) << second_reference.status();
  ASSERT_EQ(first->outputs.size(), 3u);
  ASSERT_EQ(first_reference->outputs.size(), 3u);
  EXPECT_NE(first->outputs[1].data(), second->outputs[1].data());
  EXPECT_NE(first_reference->outputs[1].data(),
            second_reference->outputs[1].data());
  EXPECT_EQ(first->outputs[2].data(), second->outputs[2].data());
  EXPECT_EQ(first_reference->outputs[2].data(),
            second_reference->outputs[2].data());

  BufferVec loss_inputs = first->outputs;
  loss_inputs.push_back(first_input->device);
  HostBufferVec reference_loss_inputs = first_reference->outputs;
  reference_loss_inputs.push_back(first_input->host);
  auto losses = (*device_loss)->fwd(*executor_, loss_inputs);
  auto reference_losses = (*reference_loss)->fwd(reference_loss_inputs);
  ASSERT_TRUE(losses.ok()) << losses.status();
  ASSERT_TRUE(reference_losses.ok()) << reference_losses.status();
  auto loss_gradients =
      (*device_loss)->bwd(*executor_, {}, std::move(losses->state));
  auto reference_loss_gradients =
      (*reference_loss)->bwd({}, std::move(reference_losses->state));
  ASSERT_TRUE(loss_gradients.ok()) << loss_gradients.status();
  ASSERT_TRUE(reference_loss_gradients.ok())
      << reference_loss_gradients.status();
  loss_gradients->pop_back();
  reference_loss_gradients->pop_back();
  auto input_gradient =
      (*device)->bwd(*executor_, *loss_gradients, std::move(first->state));
  auto reference_input_gradient =
      (*reference)
          ->bwd(*reference_loss_gradients, std::move(first_reference->state));
  ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();
  ASSERT_TRUE(reference_input_gradient.ok())
      << reference_input_gradient.status();
  auto actual_input_gradient =
      ReadDeviceFloats(*executor_, (*input_gradient)[0]);
  ASSERT_TRUE(actual_input_gradient.ok()) << actual_input_gradient.status();
  EXPECT_TRUE(VectorsNear(actual_input_gradient->span(),
                          std::vector<float>(kRows * kWidth, kPenalty), 0.0f));
  EXPECT_TRUE(FloatBuffersNear((*input_gradient)[0],
                               (*reference_input_gradient)[0], 0.0f));

  // Reconstruction is exact, so dD contains only the direct decoder-norm
  // derivative. It must use the first pass's unit latents, not the later twos.
  for (float& value : identity)
    value *= kPenalty * kRows;
  auto decoder_gradient =
      ReadDeviceFloats(*executor_, (*device)->gradients()[2]);
  ASSERT_TRUE(decoder_gradient.ok()) << decoder_gradient.status();
  EXPECT_TRUE(VectorsNear(decoder_gradient->span(), identity, 0.0f));
  EXPECT_TRUE(VectorsNear(ReadHostFloats((*reference)->gradients()[2]),
                          identity, 0.0f));
  EXPECT_TRUE(ActivationBuffersNear(first->outputs[1], first_input->host,
                                    DataType::FP16, 0.0f));
  EXPECT_TRUE(VectorsNear(ReadHostFloats(first_reference->outputs[1]),
                          std::vector<float>(kRows * kWidth, 1.0f), 0.0f));
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
