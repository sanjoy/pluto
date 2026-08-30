#include "src/llm/layers/combinators.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/llm/layer.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/gelu.h"
#include "src/llm/layers/reference_test_util.h"

namespace pluto::llm {
namespace {

TEST_F(LayerReferenceTest, ResidualCompositionAndBuildersMatchBothPasses) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (const auto [rows, width] :
         {std::tuple{16, 16}, std::tuple{32, 32}}) {
      SCOPED_TRACE(testing::Message() << "type=" << static_cast<int>(type)
                                      << " rows=" << rows
                                      << " width=" << width);
      auto device_dense =
          FullyConnectedLayer::Create(width, width, type, stream_);
      auto reference_dense =
          FullyConnectedLayerReference::Create(width, width, type);
      ASSERT_TRUE(device_dense.ok()) << device_dense.status();
      ASSERT_TRUE(reference_dense.ok()) << reference_dense.status();
      std::vector<float> matrix(static_cast<size_t>(width) * width);
      std::vector<float> bias(width);
      for (size_t index = 0; index < matrix.size(); ++index) {
        matrix[index] = 0.05f * std::sin(static_cast<float>(index) * 0.23f);
      }
      for (int index = 0; index < width; ++index) {
        bias[index] = 0.02f * std::cos(index * 0.31f);
      }
      auto device_dense_weights = (*device_dense)->weights();
      auto reference_dense_weights = (*reference_dense)->weights();
      ASSERT_TRUE(SetFloatBufferPair(device_dense_weights[0],
                                     &reference_dense_weights[0], matrix,
                                     stream_)
                      .ok());
      ASSERT_TRUE(SetFloatBufferPair(device_dense_weights[1],
                                     &reference_dense_weights[1], bias,
                                     stream_)
                      .ok());

      ComposedLayerBuilder device_builder;
      ComposedLayerReferenceBuilder reference_builder;
      ASSERT_TRUE(device_builder
                      .add(std::make_unique<ResidualLayer>(
                          std::move(*device_dense)))
                      .ok());
      ASSERT_TRUE(reference_builder
                      .add(std::make_unique<ResidualLayerReference>(
                          std::move(*reference_dense)))
                      .ok());
      ASSERT_NE(device_builder.back(), nullptr);
      ASSERT_NE(reference_builder.back(), nullptr);
      ASSERT_TRUE(device_builder.add(GeluLayer::Create(type, stream_)).ok());
      ASSERT_TRUE(reference_builder.add(GeluLayerReference::Create(type)).ok());
      auto device_model = device_builder.create();
      auto reference_model = reference_builder.create();
      ASSERT_TRUE(device_model.ok()) << device_model.status();
      ASSERT_TRUE(reference_model.ok()) << reference_model.status();
      EXPECT_EQ(device_builder.back(), nullptr);
      EXPECT_EQ(reference_builder.back(), nullptr);

      std::vector<float> input(static_cast<size_t>(rows) * width);
      std::vector<float> gradient(input.size());
      for (size_t index = 0; index < input.size(); ++index) {
        input[index] = 0.5f * std::sin(static_cast<float>(index) * 0.1f);
        gradient[index] = 0.2f * std::cos(static_cast<float>(index) * 0.13f);
      }
      auto input_pair = MakeActivationBufferPair(input, type, stream_);
      auto gradient_pair = MakeRawBufferPair<float>(gradient, stream_);
      ASSERT_TRUE(input_pair.ok()) << input_pair.status();
      ASSERT_TRUE(gradient_pair.ok()) << gradient_pair.status();
      Tape device_tape;
      ReferenceTape reference_tape;
      BufferVec device_inputs = {input_pair->device};
      HostBufferVec reference_inputs = {input_pair->host};
      auto device_output = (*device_model)->fwd(device_inputs, &device_tape);
      auto reference_output =
          (*reference_model)->fwd(reference_inputs, &reference_tape);
      ASSERT_TRUE(device_output.ok()) << device_output.status();
      ASSERT_TRUE(reference_output.ok()) << reference_output.status();
      EXPECT_TRUE(ActivationBuffersNear(*device_output, *reference_output,
                                        type, 4e-3f, 3e-3f));

      BufferVec device_gradients = {gradient_pair->device};
      HostBufferVec reference_gradients = {gradient_pair->host};
      auto device_input =
          (*device_model)->bwd(device_gradients, std::move(device_tape));
      auto reference_input = (*reference_model)->bwd(
          reference_gradients, std::move(reference_tape));
      ASSERT_TRUE(device_input.ok()) << device_input.status();
      ASSERT_TRUE(reference_input.ok()) << reference_input.status();
      EXPECT_TRUE(FloatBuffersNear(device_input->front(),
                                   reference_input->front(), 5e-3f, 4e-3f));
      auto device_parameter_gradients = (*device_model)->gradients();
      auto reference_parameter_gradients = (*reference_model)->gradients();
      ASSERT_EQ(device_parameter_gradients.size(),
                reference_parameter_gradients.size());
      for (size_t index = 0; index < device_parameter_gradients.size();
           ++index) {
        EXPECT_TRUE(FloatBuffersNear(device_parameter_gradients[index],
                                     reference_parameter_gradients[index],
                                     5e-3f, 4e-3f));
      }
    }
  }
}

TEST_F(LayerReferenceTest, BuildersPropagateFP8Rejection) {
  ComposedLayerBuilder device_builder;
  ComposedLayerReferenceBuilder reference_builder;
  const absl::Status device_status =
      device_builder.add(GeluLayer::Create(DataType::FP8, stream_));
  const absl::Status reference_status =
      reference_builder.add(GeluLayerReference::Create(DataType::FP8));
  EXPECT_EQ(device_status.code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(reference_status.code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(device_builder.back(), nullptr);
  EXPECT_EQ(reference_builder.back(), nullptr);
}

}  // namespace
}  // namespace pluto::llm
