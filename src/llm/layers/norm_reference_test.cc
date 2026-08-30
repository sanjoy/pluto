#include "src/llm/layers/norm.h"

#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/llm/layer.h"
#include "src/llm/layers/reference_test_util.h"

namespace pluto::llm {
namespace {

TEST_F(LayerReferenceTest, ForwardAndBackwardMatchAcrossShapesAndTypes) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (const auto [rows, width] :
         {std::tuple{3, 16}, std::tuple{16, 32}, std::tuple{32, 48}}) {
      SCOPED_TRACE(testing::Message() << "type=" << static_cast<int>(type)
                                      << " rows=" << rows
                                      << " width=" << width);
      auto device_layer =
          LayerNormLayer::Create(width, 1e-5f, type, stream_);
      auto reference_layer =
          LayerNormLayerReference::Create(width, 1e-5f, type);
      ASSERT_TRUE(device_layer.ok()) << device_layer.status();
      ASSERT_TRUE(reference_layer.ok()) << reference_layer.status();
      auto device_weights = (*device_layer)->weights();
      auto reference_weights = (*reference_layer)->weights();
      std::vector<float> gamma(width);
      std::vector<float> beta(width);
      for (int column = 0; column < width; ++column) {
        gamma[column] = 0.7f + 0.2f * std::cos(column * 0.19f);
        beta[column] = 0.1f * std::sin(column * 0.31f);
      }
      ASSERT_TRUE(SetFloatBufferPair(device_weights[0],
                                     &reference_weights[0], gamma, stream_)
                      .ok());
      ASSERT_TRUE(SetFloatBufferPair(device_weights[1],
                                     &reference_weights[1], beta, stream_)
                      .ok());

      std::vector<float> input(static_cast<size_t>(rows) * width);
      std::vector<float> gradient(input.size());
      for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < width; ++column) {
          input[static_cast<size_t>(row) * width + column] =
              0.4f * std::sin(row * 0.71f + column * 0.17f) + row * 0.03f;
          gradient[static_cast<size_t>(row) * width + column] =
              0.2f * std::cos(row * 0.37f - column * 0.11f);
        }
      }
      auto input_pair = MakeActivationBufferPair(input, type, stream_);
      auto gradient_pair = MakeRawBufferPair<float>(gradient, stream_);
      ASSERT_TRUE(input_pair.ok()) << input_pair.status();
      ASSERT_TRUE(gradient_pair.ok()) << gradient_pair.status();
      Tape device_tape;
      ReferenceTape reference_tape;
      BufferVec device_inputs = {input_pair->device};
      HostBufferVec reference_inputs = {input_pair->host};
      auto device_output = (*device_layer)->fwd(device_inputs, &device_tape);
      auto reference_output =
          (*reference_layer)->fwd(reference_inputs, &reference_tape);
      ASSERT_TRUE(device_output.ok()) << device_output.status();
      ASSERT_TRUE(reference_output.ok()) << reference_output.status();
      EXPECT_TRUE(ActivationBuffersNear(*device_output, *reference_output,
                                        type, 3e-3f, 3e-3f));

      BufferVec device_gradients = {gradient_pair->device};
      HostBufferVec reference_gradients = {gradient_pair->host};
      auto device_input =
          (*device_layer)->bwd(device_gradients, std::move(device_tape));
      auto reference_input = (*reference_layer)->bwd(
          reference_gradients, std::move(reference_tape));
      ASSERT_TRUE(device_input.ok()) << device_input.status();
      ASSERT_TRUE(reference_input.ok()) << reference_input.status();
      EXPECT_TRUE(FloatBuffersNear(device_input->front(),
                                   reference_input->front(), 5e-3f, 4e-3f));
      auto device_parameter_gradients = (*device_layer)->gradients();
      auto reference_parameter_gradients = (*reference_layer)->gradients();
      for (size_t index = 0; index < device_parameter_gradients.size();
           ++index) {
        EXPECT_TRUE(FloatBuffersNear(device_parameter_gradients[index],
                                     reference_parameter_gradients[index],
                                     5e-3f, 3e-3f));
      }
    }
  }
}

TEST_F(LayerReferenceTest, FP8IsRejectedConsistently) {
  auto device = LayerNormLayer::Create(16, 1e-5f, DataType::FP8, stream_);
  auto reference =
      LayerNormLayerReference::Create(16, 1e-5f, DataType::FP8);
  ASSERT_FALSE(device.ok());
  ASSERT_FALSE(reference.ok());
  EXPECT_EQ(device.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(reference.status().code(), absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace pluto::llm
