#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/reference_test_util.h"

namespace pluto::llm {
namespace {

absl::StatusOr<std::vector<uint8_t>> ReadDenseBytes(cuda::Executor& executor,
                                                    const Buffer& buffer) {
  auto pinned = cuda::PageLockedHostArray<uint8_t>::Allocate(
      executor, buffer.size_bytes());
  if (!pinned.ok())
    return pinned.status();
  auto status = cuda::CudaStatus(
      cudaMemcpyAsync(pinned->data(), buffer.data(), buffer.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "read dense repeatability output");
  if (!status.ok())
    return status;
  status = executor.Synchronize();
  if (!status.ok())
    return status;
  return std::vector<uint8_t>(pinned->begin(), pinned->end());
}

TEST_F(LayerReferenceTest, ForwardAndBackwardMatchAcrossShapesAndTypes) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (const auto [rows, input_dim, output_dim] :
         {// Partial row tiles include single-sample and batched context 27,
          // plus tails spanning multiple MMA and staged-bias tiles.
          std::tuple{1, 13, 26}, std::tuple{27, 13, 39}, std::tuple{54, 26, 13},
          std::tuple{65, 13, 26}, std::tuple{257, 13, 26}, std::tuple{16, 1, 3},
          std::tuple{32, 3, 7}, std::tuple{16, 8, 24}, std::tuple{32, 15, 33},
          std::tuple{16, 16, 16}, std::tuple{32, 32, 48},
          std::tuple{16, 48, 16}, std::tuple{64, 64, 64},
          std::tuple{80, 80, 112}, std::tuple{272, 144, 80},
          std::tuple{528, 80, 144}}) {
      SCOPED_TRACE(testing::Message()
                   << "type=" << static_cast<int>(type) << " rows=" << rows
                   << " input=" << input_dim << " output=" << output_dim);
      auto device_layer =
          FullyConnectedLayer::Create(*executor_, input_dim, output_dim, type);
      auto reference_layer =
          FullyConnectedLayerReference::Create(input_dim, output_dim, type);
      ASSERT_TRUE(device_layer.ok()) << device_layer.status();
      ASSERT_TRUE(reference_layer.ok()) << reference_layer.status();

      auto device_weights = (*device_layer)->weights();
      auto reference_weights = (*reference_layer)->weights();
      ASSERT_EQ(device_weights.size(), 2u);
      std::vector<float> matrix(static_cast<size_t>(input_dim) * output_dim);
      std::vector<float> bias(output_dim);
      for (size_t index = 0; index < matrix.size(); ++index)
        matrix[index] = 0.08f * std::sin(static_cast<float>(index) * 0.31f);
      for (int index = 0; index < output_dim; ++index)
        bias[index] = 0.03f * std::cos(static_cast<float>(index) * 0.7f);
      ASSERT_TRUE(SetFloatBufferPair(*executor_, device_weights[0],
                                     &reference_weights[0], matrix)
                      .ok());
      ASSERT_TRUE(SetFloatBufferPair(*executor_, device_weights[1],
                                     &reference_weights[1], bias)
                      .ok());

      std::vector<float> input(static_cast<size_t>(rows) * input_dim);
      std::vector<float> output_gradient(static_cast<size_t>(rows) *
                                         output_dim);
      for (size_t index = 0; index < input.size(); ++index)
        input[index] = 0.6f * std::sin(static_cast<float>(index) * 0.17f);
      for (size_t index = 0; index < output_gradient.size(); ++index) {
        output_gradient[index] =
            0.2f * std::cos(static_cast<float>(index) * 0.11f);
      }
      auto input_pair = MakeActivationBufferPair(*executor_, input, type);
      auto gradient_pair =
          MakeRawBufferPair<float>(*executor_, output_gradient);
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
                                        3e-3f, 3e-3f));

      // Dense backward replaces its parameter gradients. Nonzero sentinels
      // exercise every store, including partial tiles and staged bias sums.
      auto device_gradients = (*device_layer)->gradients();
      auto reference_gradients = (*reference_layer)->gradients();
      ASSERT_EQ(device_gradients.size(), reference_gradients.size());
      for (size_t index = 0; index < device_gradients.size(); ++index) {
        std::vector<float> sentinel(
            device_gradients[index].size_bytes() / sizeof(float), 123.0f);
        ASSERT_TRUE(SetFloatBufferPair(*executor_, device_gradients[index],
                                       &reference_gradients[index], sentinel)
                        .ok());
      }

      BufferVec device_output_gradients = {gradient_pair->device};
      HostBufferVec reference_output_gradients = {gradient_pair->host};
      auto device_input_gradients =
          (*device_layer)
              ->bwd(*executor_, device_output_gradients,
                    std::move(device_output->state));
      auto reference_input_gradients =
          (*reference_layer)
              ->bwd(reference_output_gradients,
                    std::move(reference_output->state));
      ASSERT_TRUE(device_input_gradients.ok())
          << device_input_gradients.status();
      ASSERT_TRUE(reference_input_gradients.ok())
          << reference_input_gradients.status();
      ASSERT_EQ(device_input_gradients->size(), 1u);
      ASSERT_EQ(reference_input_gradients->size(), 1u);
      EXPECT_TRUE(FloatBuffersNear(device_input_gradients->front(),
                                   reference_input_gradients->front(), 3e-3f,
                                   3e-3f));

      for (size_t index = 0; index < device_gradients.size(); ++index) {
        EXPECT_TRUE(FloatBuffersNear(device_gradients[index],
                                     reference_gradients[index], 5e-3f, 3e-3f));
      }

      BufferVec first_buffers = {device_output->outputs[0],
                                 device_input_gradients->front(),
                                 device_gradients[0], device_gradients[1]};
      std::vector<std::vector<uint8_t>> first_bytes;
      for (const Buffer& buffer : first_buffers) {
        auto bytes = ReadDenseBytes(*executor_, buffer);
        ASSERT_TRUE(bytes.ok()) << bytes.status();
        first_bytes.push_back(std::move(*bytes));
      }
      for (int repeat = 0; repeat < 2; ++repeat) {
        SCOPED_TRACE(testing::Message() << "repeat=" << repeat);
        auto repeated_output = (*device_layer)->fwd(*executor_, device_inputs);
        ASSERT_TRUE(repeated_output.ok()) << repeated_output.status();
        auto repeated_input_gradients =
            (*device_layer)
                ->bwd(*executor_, device_output_gradients,
                      std::move(repeated_output->state));
        ASSERT_TRUE(repeated_input_gradients.ok())
            << repeated_input_gradients.status();
        BufferVec repeated_buffers = {repeated_output->outputs[0],
                                      repeated_input_gradients->front(),
                                      device_gradients[0], device_gradients[1]};
        for (size_t index = 0; index < repeated_buffers.size(); ++index) {
          auto bytes = ReadDenseBytes(*executor_, repeated_buffers[index]);
          ASSERT_TRUE(bytes.ok()) << bytes.status();
          EXPECT_EQ(*bytes, first_bytes[index]) << "buffer=" << index;
        }
      }
    }
  }
}

TEST_F(LayerReferenceTest, Bf16DenseRetainsItsNativeExponentRange) {
  constexpr int kRows = 80;
  constexpr int kWidth = 80;
  constexpr DataType kType = DataType::BF16;
  auto device = FullyConnectedLayer::Create(*executor_, kWidth, kType);
  auto reference = FullyConnectedLayerReference::Create(kWidth, kType);
  ASSERT_TRUE(device.ok()) << device.status();
  ASSERT_TRUE(reference.ok()) << reference.status();
  auto device_weights = (*device)->weights();
  auto reference_weights = (*reference)->weights();
  // Every MMA operand is finite in BF16 and overflows FP16. Products and
  // accumulators remain representable in FP32.
  std::vector<float> matrix(kWidth * kWidth, 0x1p17f);
  ASSERT_TRUE(SetFloatBufferPair(*executor_, device_weights[0],
                                 &reference_weights[0], matrix)
                  .ok());
  std::vector<float> values(kRows * kWidth, 0x1p17f);
  auto input = MakeActivationBufferPair(*executor_, values, kType);
  auto gradient = MakeRawBufferPair<float>(*executor_, values);
  ASSERT_TRUE(input.ok()) << input.status();
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  auto device_output = (*device)->fwd(*executor_, BufferVec{input->device});
  auto reference_output = (*reference)->fwd(HostBufferVec{input->host});
  ASSERT_TRUE(device_output.ok()) << device_output.status();
  ASSERT_TRUE(reference_output.ok()) << reference_output.status();
  EXPECT_TRUE(ActivationBuffersNear(device_output->outputs[0],
                                    reference_output->outputs[0], kType, 0.0f));
  auto device_input_gradient = (*device)->bwd(
      *executor_, BufferVec{gradient->device}, std::move(device_output->state));
  auto reference_input_gradient = (*reference)
                                      ->bwd(HostBufferVec{gradient->host},
                                            std::move(reference_output->state));
  ASSERT_TRUE(device_input_gradient.ok()) << device_input_gradient.status();
  ASSERT_TRUE(reference_input_gradient.ok())
      << reference_input_gradient.status();
  EXPECT_TRUE(FloatBuffersNear(device_input_gradient->front(),
                               reference_input_gradient->front(), 0.0f));
  auto device_gradients = (*device)->gradients();
  auto reference_gradients = (*reference)->gradients();
  for (size_t index = 0; index < device_gradients.size(); ++index)
    EXPECT_TRUE(FloatBuffersNear(device_gradients[index],
                                 reference_gradients[index], 0.0f));
}

TEST_F(LayerReferenceTest, FP8IsRejectedConsistently) {
  auto device = FullyConnectedLayer::Create(*executor_, 16, DataType::FP8);
  auto reference = FullyConnectedLayerReference::Create(16, DataType::FP8);
  ASSERT_FALSE(device.ok());
  ASSERT_FALSE(reference.ok());
  EXPECT_EQ(device.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(reference.status().code(), absl::StatusCode::kUnimplemented);
}

TEST_F(LayerReferenceTest, DenseSignaturesRetainRectangularDimensions) {
  for (DataType compute : {DataType::FP16, DataType::BF16}) {
    const DataType storage =
        compute == DataType::BF16 ? DataType::BF16 : DataType::FP32;
    auto device = FullyConnectedLayer::Create(*executor_, 32, 64, compute, 7);
    auto reference = FullyConnectedLayerReference::Create(32, 64, compute, 7);
    ASSERT_TRUE(device.ok()) << device.status();
    ASSERT_TRUE(reference.ok()) << reference.status();
    const ActivationType input(storage, {-2, 7, 32});
    const ActivationType output(storage, {-2, 7, 64});
    ASSERT_EQ((*device)->input_types().size(), 1);
    ASSERT_EQ((*device)->output_types().size(), 1);
    ASSERT_EQ((*reference)->input_types().size(), 1);
    ASSERT_EQ((*reference)->output_types().size(), 1);
    EXPECT_EQ((*device)->input_types()[0], input);
    EXPECT_EQ((*reference)->input_types()[0], input);
    EXPECT_EQ((*device)->output_types()[0], output);
    EXPECT_EQ((*reference)->output_types()[0], output);

    // The square convenience overload must forward the configured sample size.
    auto square = FullyConnectedLayer::Create(*executor_, 32, compute, 7);
    auto reference_square =
        FullyConnectedLayerReference::Create(32, compute, 7);
    ASSERT_TRUE(square.ok()) << square.status();
    ASSERT_TRUE(reference_square.ok()) << reference_square.status();
    EXPECT_EQ((*square)->input_types()[0], input);
    EXPECT_EQ((*square)->output_types()[0], input);
    EXPECT_EQ((*reference_square)->input_types()[0], input);
    EXPECT_EQ((*reference_square)->output_types()[0], input);
  }
}

TEST_F(LayerReferenceTest, DenseRejectsNonpositiveSequenceLength) {
  for (int length : {0, -1, -2}) {
    EXPECT_EQ(
        FullyConnectedLayer::Create(*executor_, 32, 64, DataType::FP16, length)
            .status()
            .code(),
        absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(
        FullyConnectedLayerReference::Create(32, 64, DataType::FP16, length)
            .status()
            .code(),
        absl::StatusCode::kInvalidArgument);
  }
}

}  // namespace
}  // namespace pluto::llm
