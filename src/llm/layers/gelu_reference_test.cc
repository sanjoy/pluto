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

TEST_F(LayerReferenceTest, ForwardAndBackwardMatchAcrossDomainShapesAndTypes) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (int width : {3, 16, 32, 64}) {
      // Three channels need no padding: a 16-token sample has 48 elements,
      // and elementwise tiles may cross token boundaries safely.
      const int sequence_length = width == 3 ? 16 : 1;
      SCOPED_TRACE(testing::Message()
                   << "type=" << static_cast<int>(type) << " width=" << width);
      auto device_layer =
          GeluLayer::Create(*executor_, width, type, sequence_length);
      auto reference_layer =
          GeluLayerReference::Create(width, type, sequence_length);
      ASSERT_TRUE(device_layer.ok()) << device_layer.status();
      ASSERT_TRUE(reference_layer.ok()) << reference_layer.status();
      const ActivationType activation(
          type == DataType::BF16 ? DataType::BF16 : DataType::FP32,
          {-2, sequence_length, width});
      ASSERT_EQ((*device_layer)->input_types().size(), 1);
      ASSERT_EQ((*device_layer)->output_types().size(), 1);
      ASSERT_EQ((*reference_layer)->input_types().size(), 1);
      ASSERT_EQ((*reference_layer)->output_types().size(), 1);
      EXPECT_EQ((*device_layer)->input_types()[0], activation);
      EXPECT_EQ((*device_layer)->output_types()[0], activation);
      EXPECT_EQ((*reference_layer)->input_types()[0], activation);
      EXPECT_EQ((*reference_layer)->output_types()[0], activation);
      // Reuse the same layer instances at two batch sizes. Only the leading
      // -2 extent varies; sequence length and feature width stay fixed.
      for (int batch_size : {1, 2}) {
        SCOPED_TRACE(testing::Message() << "batch_size=" << batch_size);
        const int elements = batch_size * width * sequence_length;
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
}

TEST_F(LayerReferenceTest, FP8IsRejectedConsistently) {
  auto device = GeluLayer::Create(*executor_, 16, DataType::FP8);
  auto reference = GeluLayerReference::Create(16, DataType::FP8);
  ASSERT_FALSE(device.ok());
  ASSERT_FALSE(reference.ok());
  EXPECT_EQ(device.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(reference.status().code(), absl::StatusCode::kUnimplemented);
}

TEST_F(LayerReferenceTest, GeluSignaturesPreserveSampleDimensions) {
  for (DataType compute : {DataType::FP16, DataType::BF16}) {
    const DataType storage =
        compute == DataType::BF16 ? DataType::BF16 : DataType::FP32;
    auto device = GeluLayer::Create(*executor_, 32, compute, 7);
    auto reference = GeluLayerReference::Create(32, compute, 7);
    ASSERT_TRUE(device.ok()) << device.status();
    ASSERT_TRUE(reference.ok()) << reference.status();
    const ActivationType activation(storage, {-2, 7, 32});
    ASSERT_EQ((*device)->input_types().size(), 1);
    ASSERT_EQ((*device)->output_types().size(), 1);
    ASSERT_EQ((*reference)->input_types().size(), 1);
    ASSERT_EQ((*reference)->output_types().size(), 1);
    EXPECT_EQ((*device)->input_types()[0], activation);
    EXPECT_EQ((*device)->output_types()[0], activation);
    EXPECT_EQ((*reference)->input_types()[0], activation);
    EXPECT_EQ((*reference)->output_types()[0], activation);
    EXPECT_TRUE((*device)->ValidateSequenceLength(7).ok());
    EXPECT_FALSE((*device)->ValidateSequenceLength(1).ok());
  }
}

TEST_F(LayerReferenceTest, GeluRejectsUnsupportedSampleShapes) {
  for (int length : {0, -1, -2}) {
    EXPECT_EQ(GeluLayer::Create(*executor_, 32, DataType::FP16, length)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(
        GeluLayerReference::Create(32, DataType::FP16, length).status().code(),
        absl::StatusCode::kInvalidArgument);
  }
  for (int width : {0, -1}) {
    EXPECT_EQ(
        GeluLayer::Create(*executor_, width, DataType::FP16).status().code(),
        absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(GeluLayerReference::Create(width, DataType::FP16).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST_F(LayerReferenceTest, GeluStillRejectsNonTiledTotalElementCounts) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    auto device = GeluLayer::Create(*executor_, 3, type);
    auto reference = GeluLayerReference::Create(3, type);
    ASSERT_TRUE(device.ok()) << device.status();
    ASSERT_TRUE(reference.ok()) << reference.status();
    auto input =
        MakeActivationBufferPair(*executor_, std::vector<float>(3, 1.0f), type);
    ASSERT_TRUE(input.ok()) << input.status();
    const auto device_output =
        (*device)->fwd(*executor_, BufferVec{input->device});
    const auto reference_output = (*reference)->fwd(HostBufferVec{input->host});
    EXPECT_EQ(device_output.status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(reference_output.status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

}  // namespace
}  // namespace pluto::llm
