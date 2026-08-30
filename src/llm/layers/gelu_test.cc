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
  auto gelu = GeluLayer::Create(DataType::FP16, stream_);
  ASSERT_TRUE(gelu.ok()) << gelu.status();

  std::vector<float> input(kTestBatchSize, 0.0f);
  std::vector<float> output_gradient(kTestBatchSize, 1.0f);
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
  auto output = (*gelu)->fwd(inputs, &tape);
  ASSERT_TRUE(output.ok()) << output.status();
  BufferVec gradients = {*gradient_buffer};
  auto input_gradient = (*gelu)->bwd(gradients, std::move(tape));
  ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();

  std::vector<float> host_output(kTestBatchSize);
  std::vector<float> host_gradient(kTestBatchSize);
  ASSERT_EQ(cudaMemcpyAsync(host_output.data(), output->data(),
                            output->size_bytes(), cudaMemcpyDeviceToHost,
                            stream_),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(host_gradient.data(),
                            input_gradient->front().data(),
                            input_gradient->front().size_bytes(),
                            cudaMemcpyDeviceToHost, stream_),
            cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  for (int index = 0; index < kTestBatchSize; ++index) {
    EXPECT_FLOAT_EQ(host_output[index], 0.0f);
    EXPECT_FLOAT_EQ(host_gradient[index], 0.5f);
  }
}

}  // namespace
}  // namespace pluto::llm
