#include "src/llm/layers/gelu.h"

#include <cuda_runtime.h>

#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/llm/layer.h"
#include "src/llm/layers/test_util.h"

namespace pluto::llm {
namespace {

TEST_F(LayersTest, ZeroHasZeroOutputAndHalfGradient) {
  auto gelu = GeluLayer::Create(DataType::FP16, executor_);
  ASSERT_TRUE(gelu.ok()) << gelu.status();

  std::vector<float> input(kTestBatchSize, 0.0f);
  std::vector<float> output_gradient(kTestBatchSize, 1.0f);
  auto input_buffer = Buffer::Allocate(input.size() * sizeof(float), executor_);
  auto gradient_buffer =
      Buffer::Allocate(output_gradient.size() * sizeof(float), executor_);
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

  Tape tape;
  BufferVec inputs = {*input_buffer};
  auto output = (*gelu)->fwd(inputs, &tape, executor_);
  ASSERT_TRUE(output.ok()) << output.status();
  BufferVec gradients = {*gradient_buffer};
  auto input_gradient = (*gelu)->bwd(gradients, std::move(tape), executor_);
  ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();

  std::vector<float> host_output(kTestBatchSize);
  std::vector<float> host_gradient(kTestBatchSize);
  ASSERT_EQ(
      cudaMemcpyAsync(host_output.data(), output->data(), output->size_bytes(),
                      cudaMemcpyDeviceToHost, executor_->stream()),
      cudaSuccess);
  ASSERT_EQ(
      cudaMemcpyAsync(host_gradient.data(), input_gradient->front().data(),
                      input_gradient->front().size_bytes(),
                      cudaMemcpyDeviceToHost, executor_->stream()),
      cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());

  for (int index = 0; index < kTestBatchSize; ++index) {
    EXPECT_FLOAT_EQ(host_output[index], 0.0f);
    EXPECT_FLOAT_EQ(host_gradient[index], 0.5f);
  }
}

TEST_F(LayersTest, RejectsExecutionOnADifferentExecutor) {
  auto gelu = GeluLayer::Create(DataType::FP16, executor_);
  ASSERT_TRUE(gelu.ok()) << gelu.status();
  auto input = Buffer::Allocate(kTestBatchSize * sizeof(float), executor_);
  ASSERT_TRUE(input.ok()) << input.status();
  auto other_executor = cuda::Executor::Create();
  ASSERT_TRUE(other_executor.ok()) << other_executor.status();

  Tape tape;
  BufferVec inputs = {*input};
  const auto output = (*gelu)->fwd(inputs, &tape, other_executor->get());
  EXPECT_FALSE(output.ok());
  EXPECT_EQ(output.status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::llm
