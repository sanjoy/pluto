#include "src/llm/layers/fully_connected.h"

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

TEST_F(LayersTest, IdentityDenseLayerHasIdentityForwardAndBackward) {
  std::vector<float> input(kTestBatchSize * kTestModelWidth);
  std::vector<float> output_gradient(input.size());
  for (size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>(static_cast<int>(index % 17) - 8) / 8;
    output_gradient[index] =
        static_cast<float>(static_cast<int>(index % 9) - 4) / 8;
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

  auto dense = FullyConnectedLayer::Create(
      kTestModelWidth, DataType::FP16, 0.0f, stream_);
  ASSERT_TRUE(dense.ok()) << dense.status();
  ASSERT_TRUE((*dense)->InitializeIdentity().ok());
  Tape tape;
  BufferVec dense_inputs = {*input_buffer};
  auto output = (*dense)->fwd(dense_inputs, &tape);
  ASSERT_TRUE(output.ok()) << output.status();
  BufferVec dense_gradients = {*gradient_buffer};
  auto input_gradients =
      (*dense)->bwd(dense_gradients, std::move(tape));
  ASSERT_TRUE(input_gradients.ok()) << input_gradients.status();
  ASSERT_EQ(input_gradients->size(), 1u);

  std::vector<float> host_output(input.size());
  std::vector<float> host_input_gradient(input.size());
  ASSERT_EQ(cudaMemcpyAsync(host_output.data(), output->data(),
                            output->size_bytes(), cudaMemcpyDeviceToHost,
                            stream_),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(host_input_gradient.data(),
                            input_gradients->front().data(),
                            input_gradients->front().size_bytes(),
                            cudaMemcpyDeviceToHost, stream_),
            cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  for (size_t index = 0; index < input.size(); ++index) {
    EXPECT_NEAR(host_output[index], input[index], 1e-6f) << index;
    EXPECT_NEAR(host_input_gradient[index], output_gradient[index], 1e-6f)
        << index;
  }
}


}  // namespace
}  // namespace pluto::llm
