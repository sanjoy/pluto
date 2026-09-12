#include <cmath>
#include <cstddef>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/llm/layer.h"
#include "src/llm/layers/gelu.h"
#include "src/llm/layers/reference_test_util.h"

namespace pluto::llm {
namespace {

TEST_F(LayerReferenceTest, ForwardAndBackwardMatchAcrossDomainAndTypes) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (int elements : {16, 32, 64}) {
      SCOPED_TRACE(testing::Message() << "type=" << static_cast<int>(type)
                                      << " elements=" << elements);
      auto device_layer = GeluLayer::Create(*executor_, type);
      auto reference_layer = GeluLayerReference::Create(type);
      ASSERT_TRUE(device_layer.ok()) << device_layer.status();
      ASSERT_TRUE(reference_layer.ok()) << reference_layer.status();
      std::vector<float> input(elements);
      std::vector<float> gradient(elements);
      for (int index = 0; index < elements; ++index) {
        input[index] = -5.0f + 10.0f * index / (elements - 1.0f);
        gradient[index] = 0.25f * std::cos(index * 0.41f);
      }
      auto input_pair = MakeActivationBufferPair(*executor_, input, type);
      auto gradient_pair = MakeRawBufferPair<float>(*executor_, gradient);
      ASSERT_TRUE(input_pair.ok()) << input_pair.status();
      ASSERT_TRUE(gradient_pair.ok()) << gradient_pair.status();

      BufferVec device_inputs = {input_pair->device};
      HostBufferVec reference_inputs = {input_pair->host};
      auto device_output = (*device_layer)->fwd(*executor_, device_inputs);

      auto reference_output = (*reference_layer)->fwd(reference_inputs);

      ASSERT_TRUE(device_output.ok()) << device_output.status();
      ASSERT_TRUE(reference_output.ok()) << reference_output.status();
      EXPECT_TRUE(ActivationBuffersNear(device_output->outputs[0],
                                        reference_output->outputs[0], type,
                                        2e-5f, 2e-5f));

      BufferVec device_gradients = {gradient_pair->device};
      HostBufferVec reference_gradients = {gradient_pair->host};
      auto device_input = (*device_layer)
                              ->bwd(*executor_, device_gradients,
                                    std::move(device_output->state));
      auto reference_input =
          (*reference_layer)
              ->bwd(reference_gradients, std::move(reference_output->state));
      ASSERT_TRUE(device_input.ok()) << device_input.status();
      ASSERT_TRUE(reference_input.ok()) << reference_input.status();
      EXPECT_TRUE(FloatBuffersNear(device_input->front(),
                                   reference_input->front(), 2e-5f, 2e-5f));
    }
  }
}

TEST_F(LayerReferenceTest, FP8IsRejectedConsistently) {
  auto device = GeluLayer::Create(*executor_, DataType::FP8);
  auto reference = GeluLayerReference::Create(DataType::FP8);
  ASSERT_FALSE(device.ok());
  ASSERT_FALSE(reference.ok());
  EXPECT_EQ(device.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(reference.status().code(), absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace pluto::llm
