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
  auto gelu = GeluLayer::Create(*executor_, DataType::FP16);
  ASSERT_TRUE(gelu.ok()) << gelu.status();

  std::vector<float> input(kTestTokenCount, 0.0f);
  std::vector<float> output_gradient(kTestTokenCount, 1.0f);
  const auto pinned_input = CopyToPageLockedHostArray(*executor_, input);
  const auto pinned_output_gradient =
      CopyToPageLockedHostArray(*executor_, output_gradient);
  auto input_buffer =
      Buffer::Allocate(*executor_, input.size() * sizeof(float));
  auto gradient_buffer =
      Buffer::Allocate(*executor_, output_gradient.size() * sizeof(float));
  ASSERT_TRUE(input_buffer.ok()) << input_buffer.status();
  ASSERT_TRUE(gradient_buffer.ok()) << gradient_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(input_buffer->data(), pinned_input.data(),
                            input_buffer->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);
  ASSERT_EQ(
      cudaMemcpyAsync(gradient_buffer->data(), pinned_output_gradient.data(),
                      gradient_buffer->size_bytes(), cudaMemcpyHostToDevice,
                      executor_->stream()),
      cudaSuccess);

  BufferVec inputs = {*input_buffer};
  auto output = (*gelu)->fwd(*executor_, inputs);

  ASSERT_TRUE(output.ok()) << output.status();
  BufferVec gradients = {*gradient_buffer};
  auto input_gradient =
      (*gelu)->bwd(*executor_, gradients, std::move(output->state));
  ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();

  auto host_output =
      AllocatePageLockedHostArray<float>(*executor_, kTestTokenCount);
  auto host_gradient =
      AllocatePageLockedHostArray<float>(*executor_, kTestTokenCount);
  ASSERT_EQ(cudaMemcpyAsync(host_output.data(), output->output.data(),
                            output->output.size_bytes(), cudaMemcpyDeviceToHost,
                            executor_->stream()),
            cudaSuccess);
  ASSERT_EQ(
      cudaMemcpyAsync(host_gradient.data(), input_gradient->front().data(),
                      input_gradient->front().size_bytes(),
                      cudaMemcpyDeviceToHost, executor_->stream()),
      cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());

  for (int index = 0; index < kTestTokenCount; ++index) {
    EXPECT_FLOAT_EQ(host_output[index], 0.0f);
    EXPECT_FLOAT_EQ(host_gradient[index], 0.5f);
  }
}

TEST_F(LayersTest, RejectsExecutionOnADifferentExecutor) {
  auto gelu = GeluLayer::Create(*executor_, DataType::FP16);
  ASSERT_TRUE(gelu.ok()) << gelu.status();
  auto input = Buffer::Allocate(*executor_, kTestTokenCount * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();
  auto other_executor = cuda::Executor::Create();
  ASSERT_TRUE(other_executor.ok()) << other_executor.status();

  BufferVec inputs = {*input};
  const auto output = (*gelu)->fwd(**other_executor, inputs);

  EXPECT_FALSE(output.ok());
  EXPECT_EQ(output.status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::llm
