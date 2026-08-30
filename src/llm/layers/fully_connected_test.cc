#include "src/llm/layers/fully_connected.h"

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

TEST_F(LayersTest, IdentityDenseLayerHasIdentityForwardAndBackward) {
  std::vector<float> input(kTestBatchSize * kTestModelWidth);
  std::vector<float> output_gradient(input.size());
  for (size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>(static_cast<int>(index % 17) - 8) / 8;
    output_gradient[index] =
        static_cast<float>(static_cast<int>(index % 9) - 4) / 8;
  }
  auto input_buffer =
      Buffer::Allocate(input.size() * sizeof(float), *executor_);
  auto gradient_buffer =
      Buffer::Allocate(output_gradient.size() * sizeof(float), *executor_);
  ASSERT_TRUE(input_buffer.ok()) << input_buffer.status();
  ASSERT_TRUE(gradient_buffer.ok()) << gradient_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(input_buffer->data(), input.data(),
                            input_buffer->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(gradient_buffer->data(), output_gradient.data(),
                            gradient_buffer->size_bytes(),
                            cudaMemcpyHostToDevice, executor_->stream()),
            cudaSuccess);

  auto dense =
      FullyConnectedLayer::Create(kTestModelWidth, DataType::FP16, *executor_);
  ASSERT_TRUE(dense.ok()) << dense.status();
  ASSERT_TRUE((*dense)->InitializeIdentity().ok());
  Tape tape;
  BufferVec dense_inputs = {*input_buffer};
  auto output = (*dense)->fwd(dense_inputs, &tape, *executor_);
  ASSERT_TRUE(output.ok()) << output.status();
  BufferVec dense_gradients = {*gradient_buffer};
  auto input_gradients =
      (*dense)->bwd(dense_gradients, std::move(tape), *executor_);
  ASSERT_TRUE(input_gradients.ok()) << input_gradients.status();
  ASSERT_EQ(input_gradients->size(), 1u);

  std::vector<float> host_output(input.size());
  std::vector<float> host_input_gradient(input.size());
  ASSERT_EQ(
      cudaMemcpyAsync(host_output.data(), output->data(), output->size_bytes(),
                      cudaMemcpyDeviceToHost, executor_->stream()),
      cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(host_input_gradient.data(),
                            input_gradients->front().data(),
                            input_gradients->front().size_bytes(),
                            cudaMemcpyDeviceToHost, executor_->stream()),
            cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());

  for (size_t index = 0; index < input.size(); ++index) {
    EXPECT_NEAR(host_output[index], input[index], 1e-6f) << index;
    EXPECT_NEAR(host_input_gradient[index], output_gradient[index], 1e-6f)
        << index;
  }
}

TEST_F(LayersTest, RectangularProjectionUsesDistinctInputAndOutputWidths) {
  constexpr int kOutputWidth = 48;
  auto dense = FullyConnectedLayer::Create(kTestModelWidth, kOutputWidth,
                                           DataType::FP16, *executor_);
  ASSERT_TRUE(dense.ok()) << dense.status();
  ASSERT_EQ((*dense)->input_dim(), kTestModelWidth);
  ASSERT_EQ((*dense)->output_dim(), kOutputWidth);
  ASSERT_TRUE((*dense)->InitializeIdentity().ok());

  std::vector<float> input(kTestBatchSize * kTestModelWidth);
  for (size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>(index % 7);
  }
  auto input_buffer =
      Buffer::Allocate(input.size() * sizeof(float), *executor_);
  ASSERT_TRUE(input_buffer.ok()) << input_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(input_buffer->data(), input.data(),
                            input_buffer->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);
  Tape tape;
  BufferVec inputs = {*input_buffer};
  auto output = (*dense)->fwd(inputs, &tape, *executor_);
  ASSERT_TRUE(output.ok()) << output.status();
  std::vector<float> host_output(kTestBatchSize * kOutputWidth);
  ASSERT_EQ(
      cudaMemcpyAsync(host_output.data(), output->data(), output->size_bytes(),
                      cudaMemcpyDeviceToHost, executor_->stream()),
      cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());

  for (int row = 0; row < kTestBatchSize; ++row) {
    for (int column = 0; column < kTestModelWidth; ++column) {
      EXPECT_FLOAT_EQ(host_output[row * kOutputWidth + column],
                      input[row * kTestModelWidth + column]);
    }
    for (int column = kTestModelWidth; column < kOutputWidth; ++column) {
      EXPECT_FLOAT_EQ(host_output[row * kOutputWidth + column], 0.0f);
    }
  }
}

}  // namespace
}  // namespace pluto::llm
