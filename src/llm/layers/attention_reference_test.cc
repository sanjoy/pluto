#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/llm/layer.h"
#include "src/llm/layers/attention.h"
#include "src/llm/layers/reference_test_util.h"

namespace pluto::llm {
namespace {

TEST_F(LayerReferenceTest, CausalForwardAndBackwardMatchAcrossConfigurations) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (const auto [context, heads, width, sequences] :
         {std::tuple{1, 1, 16, 3}, std::tuple{4, 1, 16, 2},
          std::tuple{4, 2, 32, 2}, std::tuple{8, 2, 32, 1}}) {
      SCOPED_TRACE(testing::Message()
                   << "type=" << static_cast<int>(type)
                   << " context=" << context << " heads=" << heads
                   << " width=" << width << " sequences=" << sequences);
      auto device_layer =
          AttentionLayer::Create(*executor_, context, heads, width, type);
      auto reference_layer =
          AttentionLayerReference::Create(context, heads, width, type);
      ASSERT_TRUE(device_layer.ok()) << device_layer.status();
      ASSERT_TRUE(reference_layer.ok()) << reference_layer.status();
      const int rows = context * sequences;
      std::vector<float> qkv(static_cast<size_t>(rows) * 3 * width);
      std::vector<float> output_gradient(static_cast<size_t>(rows) * width);
      for (size_t index = 0; index < qkv.size(); ++index) {
        qkv[index] = 0.35f * std::sin(static_cast<float>(index) * 0.071f) +
                     0.05f * std::cos(static_cast<float>(index) * 0.19f);
      }
      for (size_t index = 0; index < output_gradient.size(); ++index) {
        output_gradient[index] =
            0.2f * std::cos(static_cast<float>(index) * 0.13f);
      }
      auto input_pair = MakeActivationBufferPair(*executor_, qkv, type);
      auto gradient_pair =
          MakeRawBufferPair<float>(*executor_, output_gradient);
      ASSERT_TRUE(input_pair.ok()) << input_pair.status();
      ASSERT_TRUE(gradient_pair.ok()) << gradient_pair.status();

      Tape device_tape;
      ReferenceTape reference_tape;
      BufferVec device_inputs = {input_pair->device};
      HostBufferVec reference_inputs = {input_pair->host};
      auto device_output =
          (*device_layer)->fwd(*executor_, device_inputs, &device_tape);
      auto reference_output =
          (*reference_layer)->fwd(reference_inputs, &reference_tape);
      ASSERT_TRUE(device_output.ok()) << device_output.status();
      ASSERT_TRUE(reference_output.ok()) << reference_output.status();
      EXPECT_TRUE(ActivationBuffersNear(*device_output, *reference_output, type,
                                        2e-3f, 2e-3f));

      BufferVec device_gradients = {gradient_pair->device};
      HostBufferVec reference_gradients = {gradient_pair->host};
      auto device_input =
          (*device_layer)
              ->bwd(*executor_, device_gradients, std::move(device_tape));
      auto reference_input =
          (*reference_layer)
              ->bwd(reference_gradients, std::move(reference_tape));
      ASSERT_TRUE(device_input.ok()) << device_input.status();
      ASSERT_TRUE(reference_input.ok()) << reference_input.status();
      ASSERT_EQ(device_input->size(), 1u);
      ASSERT_EQ(reference_input->size(), 1u);
      EXPECT_TRUE(FloatBuffersNear(device_input->front(),
                                   reference_input->front(), 3e-3f, 3e-3f));
    }
  }
}

TEST_F(LayerReferenceTest, FP8IsRejectedConsistently) {
  auto device = AttentionLayer::Create(*executor_, 4, 2, 32, DataType::FP8);
  auto reference = AttentionLayerReference::Create(4, 2, 32, DataType::FP8);
  ASSERT_FALSE(device.ok());
  ASSERT_FALSE(reference.ok());
  EXPECT_EQ(device.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(reference.status().code(), absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace pluto::llm
