#include "src/llm/layers/attention.h"

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

TEST_F(LayersTest, FlashAttentionIsCausalAndHasCorrectSingleTokenGradient) {
  auto attention = AttentionLayer::Create(*executor_, kTestContextLength,
                                          kTestAttentionHeads, kTestModelWidth,
                                          DataType::FP16);
  ASSERT_TRUE(attention.ok()) << attention.status();

  constexpr int kPackedWidth = 3 * kTestModelWidth;
  std::vector<float> input(kTestBatchSize * kPackedWidth, 0.0f);
  // Q, K, and V occupy distinct packed regions in each row. Position zero
  // must ignore future values, while position one attends to values 1 and 2
  // using dot products 2 and 4.
  input[0] = 1.0f;
  input[kTestModelWidth] = 1.0f;
  input[2 * kTestModelWidth] = 1.0f;
  input[kPackedWidth] = 2.0f;
  input[kPackedWidth + kTestModelWidth] = 2.0f;
  input[kPackedWidth + 2 * kTestModelWidth] = 2.0f;
  // A future value that position zero is not allowed to observe.
  input[2 * kPackedWidth + 2 * kTestModelWidth] = 4.0f;
  auto input_buffer =
      Buffer::Allocate(*executor_, input.size() * sizeof(float));
  ASSERT_TRUE(input_buffer.ok()) << input_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(input_buffer->data(), input.data(),
                            input_buffer->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);

  Tape tape;
  BufferVec attention_inputs = {*input_buffer};
  auto output = (*attention)->fwd(*executor_, attention_inputs, &tape);
  ASSERT_TRUE(output.ok()) << output.status();

  std::vector<float> output_gradient(kTestBatchSize * kTestModelWidth, 0.0f);
  output_gradient[0] = 1.0f;
  auto gradient_buffer =
      Buffer::Allocate(*executor_, output_gradient.size() * sizeof(float));
  ASSERT_TRUE(gradient_buffer.ok()) << gradient_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(gradient_buffer->data(), output_gradient.data(),
                            gradient_buffer->size_bytes(),
                            cudaMemcpyHostToDevice, executor_->stream()),
            cudaSuccess);
  BufferVec attention_gradients = {*gradient_buffer};
  auto input_gradient =
      (*attention)->bwd(*executor_, attention_gradients, std::move(tape));
  ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();
  ASSERT_EQ(input_gradient->size(), 1u);

  std::vector<float> host_output(output_gradient.size());
  std::vector<float> host_input_gradient(input.size());
  ASSERT_EQ(
      cudaMemcpyAsync(host_output.data(), output->data(), output->size_bytes(),
                      cudaMemcpyDeviceToHost, executor_->stream()),
      cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(host_input_gradient.data(),
                            input_gradient->front().data(),
                            input_gradient->front().size_bytes(),
                            cudaMemcpyDeviceToHost, executor_->stream()),
            cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());

  const float scale = 1.0f / std::sqrt(kTestModelWidth / kTestAttentionHeads);
  const float first_weight = std::exp(2.0f * scale);
  const float second_weight = std::exp(4.0f * scale);
  const float expected_position_one =
      (first_weight + 2.0f * second_weight) / (first_weight + second_weight);
  EXPECT_NEAR(host_output[0], 1.0f, 1e-6f);
  EXPECT_NEAR(host_output[kTestModelWidth], expected_position_one, 1e-5f);
  EXPECT_NEAR(host_input_gradient[0], 0.0f, 1e-6f);
  EXPECT_NEAR(host_input_gradient[kTestModelWidth], 0.0f, 1e-6f);
  EXPECT_NEAR(host_input_gradient[2 * kTestModelWidth], 1.0f, 1e-6f);
  EXPECT_NEAR(host_input_gradient[kPackedWidth], 0.0f, 1e-6f);
}

}  // namespace
}  // namespace pluto::llm
