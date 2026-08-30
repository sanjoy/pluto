#include "src/llm/layers/norm.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/llm/layer.h"
#include "src/llm/layers/test_util.h"

namespace pluto::llm {
namespace {

TEST_F(LayersTest, LayerNormNormalizesRowsAndRejectsConstantGradient) {
  auto layer_norm =
      LayerNormLayer::Create(kTestModelWidth, 1e-5f, DataType::FP16, stream_);
  ASSERT_TRUE(layer_norm.ok()) << layer_norm.status();

  std::vector<float> input(kTestBatchSize * kTestModelWidth);
  std::vector<float> output_gradient(input.size(), 1.0f);
  for (int row = 0; row < kTestBatchSize; ++row) {
    for (int column = 0; column < kTestModelWidth; ++column) {
      input[row * kTestModelWidth + column] =
          static_cast<float>(column) / kTestModelWidth;
    }
  }
  auto input_buffer = Buffer::Allocate(input.size() * sizeof(float), stream_);
  auto gradient_buffer =
      Buffer::Allocate(output_gradient.size() * sizeof(float), stream_);
  ASSERT_TRUE(input_buffer.ok()) << input_buffer.status();
  ASSERT_TRUE(gradient_buffer.ok()) << gradient_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(input_buffer->data(), input.data(),
                            input_buffer->size_bytes(), cudaMemcpyHostToDevice,
                            stream_),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(gradient_buffer->data(), output_gradient.data(),
                            gradient_buffer->size_bytes(),
                            cudaMemcpyHostToDevice, stream_),
            cudaSuccess);

  Tape tape;
  BufferVec inputs = {*input_buffer};
  auto output = (*layer_norm)->fwd(inputs, &tape);
  ASSERT_TRUE(output.ok()) << output.status();
  BufferVec gradients = {*gradient_buffer};
  auto input_gradient =
      (*layer_norm)->bwd(gradients, std::move(tape));
  ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();

  std::vector<float> host_output(kTestModelWidth);
  std::vector<float> host_input_gradient(kTestModelWidth);
  ASSERT_EQ(cudaMemcpyAsync(host_output.data(), output->data(),
                            host_output.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(host_input_gradient.data(),
                            input_gradient->front().data(),
                            host_input_gradient.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_),
            cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  double mean = 0.0;
  double square_mean = 0.0;
  for (float value : host_output) {
    mean += value;
    square_mean += value * value;
  }
  mean /= kTestModelWidth;
  square_mean /= kTestModelWidth;
  EXPECT_NEAR(mean, 0.0, 1e-5);
  EXPECT_NEAR(square_mean, 1.0, 2e-4);
  for (float gradient : host_input_gradient) {
    EXPECT_NEAR(gradient, 0.0f, 1e-5f);
  }
}


}  // namespace
}  // namespace pluto::llm
