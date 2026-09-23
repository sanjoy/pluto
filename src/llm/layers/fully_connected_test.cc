#include "src/llm/layers/fully_connected.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer.h"
#include "src/llm/layers/test_util.h"

namespace pluto::llm {
namespace {

TEST_F(LayersTest, InitializeIdentityClearsExistingBiasAndPreservesGradients) {
  constexpr int kInputWidth = 32, kOutputWidth = 48;
  constexpr float kScale = -0.75f;
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    SCOPED_TRACE(static_cast<int>(type));
    auto dense = FullyConnectedLayer::Create(*executor_, kInputWidth,
                                             kOutputWidth, type);
    ASSERT_TRUE(dense.ok()) << dense.status();
    const auto weights = (*dense)->weights();
    const auto gradients = (*dense)->gradients();
    ASSERT_EQ(weights.size(), 2u);
    ASSERT_EQ(gradients.size(), 2u);
    const BufferVec buffers = {weights[0], weights[1], gradients[0],
                               gradients[1]};
    constexpr float kSentinels[] = {3.0f, 5.0f, 7.0f, -11.0f};
    std::vector<cuda::PageLockedHostArray<float>> host;
    for (size_t index = 0; index < buffers.size(); ++index) {
      host.push_back(CopyToPageLockedHostArray(
          *executor_,
          std::vector<float>(buffers[index].size_bytes() / sizeof(float),
                             kSentinels[index])));
      ASSERT_EQ(cudaMemcpyAsync(buffers[index].data(), host.back().data(),
                                buffers[index].size_bytes(),
                                cudaMemcpyHostToDevice, executor_->stream()),
                cudaSuccess);
    }

    // Do not synchronize between uploads, initialization, and downloads:
    // initialization must order both parameter changes on this same stream.
    const auto initialized = (*dense)->InitializeIdentity(kScale);
    ASSERT_TRUE(initialized.ok()) << initialized;
    for (size_t index = 0; index < buffers.size(); ++index)
      ASSERT_EQ(cudaMemcpyAsync(host[index].data(), buffers[index].data(),
                                buffers[index].size_bytes(),
                                cudaMemcpyDeviceToHost, executor_->stream()),
                cudaSuccess);
    ASSERT_TRUE(executor_->Synchronize().ok());

    for (int row = 0; row < kInputWidth; ++row)
      for (int column = 0; column < kOutputWidth; ++column)
        EXPECT_FLOAT_EQ(host[0][row * kOutputWidth + column],
                        row == column ? kScale : 0.0f)
            << "matrix row=" << row << " column=" << column;
    for (float bias : host[1])
      EXPECT_FLOAT_EQ(bias, 0.0f);
    for (size_t index = 2; index < host.size(); ++index)
      for (float gradient : host[index])
        EXPECT_FLOAT_EQ(gradient, kSentinels[index]);
  }
}

TEST_F(LayersTest, IdentityDenseLayerHasIdentityForwardAndBackward) {
  std::vector<float> input(kTestTokenCount * kTestModelWidth);
  std::vector<float> output_gradient(input.size());
  for (size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>(static_cast<int>(index % 17) - 8) / 8;
    output_gradient[index] =
        static_cast<float>(static_cast<int>(index % 9) - 4) / 8;
  }
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

  auto dense =
      FullyConnectedLayer::Create(*executor_, kTestModelWidth, DataType::FP16);
  ASSERT_TRUE(dense.ok()) << dense.status();
  ASSERT_TRUE((*dense)->InitializeIdentity().ok());

  BufferVec dense_inputs = {*input_buffer};
  auto output = (*dense)->fwd(*executor_, dense_inputs);

  ASSERT_TRUE(output.ok()) << output.status();
  BufferVec dense_gradients = {*gradient_buffer};
  auto input_gradients =
      (*dense)->bwd(*executor_, dense_gradients, std::move(output->state));
  ASSERT_TRUE(input_gradients.ok()) << input_gradients.status();
  ASSERT_EQ(input_gradients->size(), 1u);

  auto host_output =
      AllocatePageLockedHostArray<float>(*executor_, input.size());
  auto host_input_gradient =
      AllocatePageLockedHostArray<float>(*executor_, input.size());
  ASSERT_EQ(cudaMemcpyAsync(host_output.data(), output->outputs[0].data(),
                            output->outputs[0].size_bytes(),
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
  auto dense = FullyConnectedLayer::Create(*executor_, kTestModelWidth,
                                           kOutputWidth, DataType::FP16);
  ASSERT_TRUE(dense.ok()) << dense.status();
  ASSERT_EQ((*dense)->input_dim(), kTestModelWidth);
  ASSERT_EQ((*dense)->output_dim(), kOutputWidth);
  ASSERT_TRUE((*dense)->InitializeIdentity().ok());

  std::vector<float> input(kTestTokenCount * kTestModelWidth);
  for (size_t index = 0; index < input.size(); ++index)
    input[index] = static_cast<float>(index % 7);
  const auto pinned_input = CopyToPageLockedHostArray(*executor_, input);
  auto input_buffer =
      Buffer::Allocate(*executor_, input.size() * sizeof(float));
  ASSERT_TRUE(input_buffer.ok()) << input_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(input_buffer->data(), pinned_input.data(),
                            input_buffer->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);

  BufferVec inputs = {*input_buffer};
  auto output = (*dense)->fwd(*executor_, inputs);

  ASSERT_TRUE(output.ok()) << output.status();
  auto host_output = AllocatePageLockedHostArray<float>(
      *executor_, kTestTokenCount * kOutputWidth);
  ASSERT_EQ(cudaMemcpyAsync(host_output.data(), output->outputs[0].data(),
                            output->outputs[0].size_bytes(),
                            cudaMemcpyDeviceToHost, executor_->stream()),
            cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());

  for (int row = 0; row < kTestTokenCount; ++row) {
    for (int column = 0; column < kTestModelWidth; ++column) {
      EXPECT_FLOAT_EQ(host_output[row * kOutputWidth + column],
                      input[row * kTestModelWidth + column]);
    }
    for (int column = kTestModelWidth; column < kOutputWidth; ++column)
      EXPECT_FLOAT_EQ(host_output[row * kOutputWidth + column], 0.0f);
  }
}

}  // namespace
}  // namespace pluto::llm
