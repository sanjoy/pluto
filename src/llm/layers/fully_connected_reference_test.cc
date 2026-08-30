#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/llm/layer.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/reference_test_util.h"

namespace pluto::llm {
namespace {

TEST_F(LayerReferenceTest, ForwardAndBackwardMatchAcrossShapesAndTypes) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (const auto [rows, input_dim, output_dim] :
         {std::tuple{16, 16, 16}, std::tuple{32, 32, 48},
          std::tuple{16, 48, 16}}) {
      SCOPED_TRACE(testing::Message()
                   << "type=" << static_cast<int>(type) << " rows=" << rows
                   << " input=" << input_dim << " output=" << output_dim);
      auto device_layer =
          FullyConnectedLayer::Create(input_dim, output_dim, type, *executor_);
      auto reference_layer =
          FullyConnectedLayerReference::Create(input_dim, output_dim, type);
      ASSERT_TRUE(device_layer.ok()) << device_layer.status();
      ASSERT_TRUE(reference_layer.ok()) << reference_layer.status();

      auto device_weights = (*device_layer)->weights();
      auto reference_weights = (*reference_layer)->weights();
      ASSERT_EQ(device_weights.size(), 2u);
      std::vector<float> matrix(static_cast<size_t>(input_dim) * output_dim);
      std::vector<float> bias(output_dim);
      for (size_t index = 0; index < matrix.size(); ++index) {
        matrix[index] = 0.08f * std::sin(static_cast<float>(index) * 0.31f);
      }
      for (int index = 0; index < output_dim; ++index) {
        bias[index] = 0.03f * std::cos(static_cast<float>(index) * 0.7f);
      }
      ASSERT_TRUE(SetFloatBufferPair(device_weights[0], &reference_weights[0],
                                     matrix, *executor_)
                      .ok());
      ASSERT_TRUE(SetFloatBufferPair(device_weights[1], &reference_weights[1],
                                     bias, *executor_)
                      .ok());

      std::vector<float> input(static_cast<size_t>(rows) * input_dim);
      std::vector<float> output_gradient(static_cast<size_t>(rows) *
                                         output_dim);
      for (size_t index = 0; index < input.size(); ++index) {
        input[index] = 0.6f * std::sin(static_cast<float>(index) * 0.17f);
      }
      for (size_t index = 0; index < output_gradient.size(); ++index) {
        output_gradient[index] =
            0.2f * std::cos(static_cast<float>(index) * 0.11f);
      }
      auto input_pair = MakeActivationBufferPair(input, type, *executor_);
      auto gradient_pair =
          MakeRawBufferPair<float>(output_gradient, *executor_);
      ASSERT_TRUE(input_pair.ok()) << input_pair.status();
      ASSERT_TRUE(gradient_pair.ok()) << gradient_pair.status();

      Tape device_tape;
      ReferenceTape reference_tape;
      BufferVec device_inputs = {input_pair->device};
      HostBufferVec reference_inputs = {input_pair->host};
      auto device_output =
          (*device_layer)->fwd(device_inputs, &device_tape, *executor_);
      auto reference_output =
          (*reference_layer)->fwd(reference_inputs, &reference_tape);
      ASSERT_TRUE(device_output.ok()) << device_output.status();
      ASSERT_TRUE(reference_output.ok()) << reference_output.status();
      EXPECT_TRUE(ActivationBuffersNear(*device_output, *reference_output, type,
                                        3e-3f, 3e-3f));

      BufferVec device_output_gradients = {gradient_pair->device};
      HostBufferVec reference_output_gradients = {gradient_pair->host};
      auto device_input_gradients =
          (*device_layer)
              ->bwd(device_output_gradients, std::move(device_tape),
                    *executor_);
      auto reference_input_gradients =
          (*reference_layer)
              ->bwd(reference_output_gradients, std::move(reference_tape));
      ASSERT_TRUE(device_input_gradients.ok())
          << device_input_gradients.status();
      ASSERT_TRUE(reference_input_gradients.ok())
          << reference_input_gradients.status();
      ASSERT_EQ(device_input_gradients->size(), 1u);
      ASSERT_EQ(reference_input_gradients->size(), 1u);
      EXPECT_TRUE(FloatBuffersNear(device_input_gradients->front(),
                                   reference_input_gradients->front(), 3e-3f,
                                   3e-3f));

      auto device_gradients = (*device_layer)->gradients();
      auto reference_gradients = (*reference_layer)->gradients();
      ASSERT_EQ(device_gradients.size(), reference_gradients.size());
      for (size_t index = 0; index < device_gradients.size(); ++index) {
        EXPECT_TRUE(FloatBuffersNear(device_gradients[index],
                                     reference_gradients[index], 5e-3f, 3e-3f));
      }
    }
  }
}

TEST_F(LayerReferenceTest, FP8IsRejectedConsistently) {
  auto device = FullyConnectedLayer::Create(16, DataType::FP8, *executor_);
  auto reference = FullyConnectedLayerReference::Create(16, DataType::FP8);
  ASSERT_FALSE(device.ok());
  ASSERT_FALSE(reference.ok());
  EXPECT_EQ(device.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(reference.status().code(), absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace pluto::llm
