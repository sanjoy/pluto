#include "src/llm/layers/attention.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "src/gpu/buffer.h"
#include "src/llm/layer.h"
#include "src/llm/layers/test_util.h"

namespace pluto::llm {
namespace {

TEST_F(LayersTest, FlashAttentionIsCausalAndHasCorrectSingleTokenGradient) {
  auto attention = AttentionLayer::Create(
      kTestContextLength, kTestAttentionHeads, kTestModelWidth,
      DataType::FP16, stream_);
  ASSERT_TRUE(attention.ok()) << attention.status();

  std::vector<float> input(kTestBatchSize * kTestModelWidth, 0.0f);
  // Only the first feature of the first head is nonzero. Position zero must
  // ignore the larger future values, while position one attends to 1 and 2.
  input[0] = 1.0f;
  input[kTestModelWidth] = 2.0f;
  input[2 * kTestModelWidth] = 4.0f;
  auto input_buffer = Buffer::Allocate(input.size() * sizeof(float), stream_);
  ASSERT_TRUE(input_buffer.ok()) << input_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(input_buffer->data(), input.data(),
                            input_buffer->size_bytes(), cudaMemcpyHostToDevice,
                            stream_),
            cudaSuccess);

  Tape tape;
  BufferVec attention_inputs = {*input_buffer};
  auto output = (*attention)->fwd(attention_inputs, &tape);
  ASSERT_TRUE(output.ok()) << output.status();

  std::vector<float> output_gradient(input.size(), 0.0f);
  output_gradient[0] = 1.0f;
  auto gradient_buffer =
      Buffer::Allocate(output_gradient.size() * sizeof(float), stream_);
  ASSERT_TRUE(gradient_buffer.ok()) << gradient_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(gradient_buffer->data(), output_gradient.data(),
                            gradient_buffer->size_bytes(),
                            cudaMemcpyHostToDevice, stream_),
            cudaSuccess);
  BufferVec attention_gradients = {*gradient_buffer};
  auto input_gradient =
      (*attention)->bwd(attention_gradients, std::move(tape));
  ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();
  ASSERT_EQ(input_gradient->size(), 1u);

  std::vector<float> host_output(input.size());
  std::vector<float> host_input_gradient(input.size());
  ASSERT_EQ(cudaMemcpyAsync(host_output.data(), output->data(),
                            output->size_bytes(), cudaMemcpyDeviceToHost,
                            stream_),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(host_input_gradient.data(),
                            input_gradient->front().data(),
                            input_gradient->front().size_bytes(),
                            cudaMemcpyDeviceToHost, stream_),
            cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  const float scale =
      1.0f / std::sqrt(kTestModelWidth / kTestAttentionHeads);
  const float first_weight = std::exp(2.0f * scale);
  const float second_weight = std::exp(4.0f * scale);
  const float expected_position_one =
      (first_weight + 2.0f * second_weight) /
      (first_weight + second_weight);
  EXPECT_NEAR(host_output[0], 1.0f, 1e-6f);
  EXPECT_NEAR(host_output[kTestModelWidth], expected_position_one, 1e-5f);
  EXPECT_NEAR(host_input_gradient[0], 1.0f, 1e-6f);
  EXPECT_NEAR(host_input_gradient[kTestModelWidth], 0.0f, 1e-6f);
}


}  // namespace
}  // namespace pluto::llm
