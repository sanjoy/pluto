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
      auto device_layer = GeluLayer::Create(type, *executor_);
      auto reference_layer = GeluLayerReference::Create(type);
      ASSERT_TRUE(device_layer.ok()) << device_layer.status();
      ASSERT_TRUE(reference_layer.ok()) << reference_layer.status();
      std::vector<float> input(elements);
      std::vector<float> gradient(elements);
      for (int index = 0; index < elements; ++index) {
        input[index] = -5.0f + 10.0f * index / (elements - 1.0f);
        gradient[index] = 0.25f * std::cos(index * 0.41f);
      }
      auto input_pair = MakeActivationBufferPair(input, type, *executor_);
      auto gradient_pair = MakeRawBufferPair<float>(gradient, *executor_);
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
                                        2e-5f, 2e-5f));

      BufferVec device_gradients = {gradient_pair->device};
      HostBufferVec reference_gradients = {gradient_pair->host};
      auto device_input =
          (*device_layer)
              ->bwd(device_gradients, std::move(device_tape), *executor_);
      auto reference_input =
          (*reference_layer)
              ->bwd(reference_gradients, std::move(reference_tape));
      ASSERT_TRUE(device_input.ok()) << device_input.status();
      ASSERT_TRUE(reference_input.ok()) << reference_input.status();
      EXPECT_TRUE(FloatBuffersNear(device_input->front(),
                                   reference_input->front(), 2e-5f, 2e-5f));
    }
  }
}

TEST_F(LayerReferenceTest, FP8IsRejectedConsistently) {
  auto device = GeluLayer::Create(DataType::FP8, *executor_);
  auto reference = GeluLayerReference::Create(DataType::FP8);
  ASSERT_FALSE(device.ok());
  ASSERT_FALSE(reference.ok());
  EXPECT_EQ(device.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(reference.status().code(), absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace pluto::llm
