#include "src/llm/layers.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "src/gpu/buffer.h"
#include "src/llm/layer.h"
#include "src/llm/simple_llm.h"

namespace pluto::llm {
namespace {

class LayersTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
              cudaSuccess);
  }

  void TearDown() override {
    if (stream_ == nullptr) return;
    EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream_), cudaSuccess);
  }

  cudaStream_t stream_ = nullptr;
};

TEST_F(LayersTest, RejectsFp8UntilScalingIsSpecified) {
  auto layer = EmbeddingLookupLayer::Create(DataType::FP8, 0.1f, stream_);
  EXPECT_FALSE(layer.ok());
  EXPECT_EQ(layer.status().code(), absl::StatusCode::kUnimplemented);
}

TEST_F(LayersTest, CrossEntropyForwardAndBackwardMatchUniformSoftmax) {
  std::vector<float> logits(kBatchSize * kVocabularySize, 0.0f);
  std::vector<int> targets(kBatchSize);
  for (int row = 0; row < kBatchSize; ++row) targets[row] = row;

  auto logits_buffer = Buffer::Allocate(logits.size() * sizeof(float), stream_);
  auto target_buffer = Buffer::Allocate(targets.size() * sizeof(int), stream_);
  ASSERT_TRUE(logits_buffer.ok()) << logits_buffer.status();
  ASSERT_TRUE(target_buffer.ok()) << target_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(logits_buffer->data(), logits.data(),
                            logits_buffer->size_bytes(), cudaMemcpyHostToDevice,
                            stream_),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(target_buffer->data(), targets.data(),
                            target_buffer->size_bytes(),
                            cudaMemcpyHostToDevice, stream_),
            cudaSuccess);

  auto loss_layer = CrossEntropyLossLayer::Create(DataType::FP16, stream_);
  ASSERT_TRUE(loss_layer.ok()) << loss_layer.status();
  Tape tape;
  Buffers loss_inputs = {*logits_buffer, *target_buffer};
  auto losses = (*loss_layer)->fwd(loss_inputs, &tape);
  ASSERT_TRUE(losses.ok()) << losses.status();
  auto gradients = (*loss_layer)->bwd({}, std::move(tape));
  ASSERT_TRUE(gradients.ok()) << gradients.status();
  ASSERT_EQ(gradients->size(), 1u);

  std::vector<float> host_losses(kBatchSize);
  std::vector<float> host_gradients(logits.size());
  ASSERT_EQ(cudaMemcpyAsync(host_losses.data(), losses->data(),
                            losses->size_bytes(), cudaMemcpyDeviceToHost,
                            stream_),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(host_gradients.data(), gradients->front().data(),
                            gradients->front().size_bytes(),
                            cudaMemcpyDeviceToHost, stream_),
            cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  EXPECT_NEAR(host_losses[0], std::log(256.0f), 1e-5f);
  EXPECT_NEAR(host_gradients[0], (1.0f / 256.0f - 1.0f) / 256.0f,
              1e-6f);
  EXPECT_NEAR(host_gradients[1], 1.0f / (256.0f * 256.0f), 1e-7f);
}

TEST_F(LayersTest, IdentityDenseLayerHasIdentityForwardAndBackward) {
  std::vector<float> input(kBatchSize * kModelWidth);
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

  auto dense = FullyConnectedLayer::Create(DataType::FP16, 0.0f, stream_);
  ASSERT_TRUE(dense.ok()) << dense.status();
  ASSERT_TRUE((*dense)->InitializeIdentity().ok());
  Tape tape;
  Buffers dense_inputs = {*input_buffer};
  auto output = (*dense)->fwd(dense_inputs, &tape);
  ASSERT_TRUE(output.ok()) << output.status();
  Buffers dense_gradients = {*gradient_buffer};
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

TEST_F(LayersTest, FactoryBuildsAComposedTrainableModel) {
  auto model = CreateSimpleLlm(DataType::FP16, 1.0f, stream_);
  ASSERT_TRUE(model.ok()) << model.status();
  // Embedding table, dense matrix, and dense bias.
  EXPECT_EQ((*model)->weights().size(), 3u);

  SimpleLlmConfig deeper_config;
  deeper_config.dense_repetitions = 2;
  auto deeper_model = CreateSimpleLlm(deeper_config, stream_);
  ASSERT_TRUE(deeper_model.ok()) << deeper_model.status();
  // One embedding plus a matrix and bias for each repeated composed block.
  EXPECT_EQ((*deeper_model)->weights().size(), 5u);
}

TEST_F(LayersTest, RepeatedLayerCollectsIndependentChildWeights) {
  auto first = FullyConnectedLayer::Create(DataType::FP16, 0.0f, stream_);
  auto second = FullyConnectedLayer::Create(DataType::FP16, 0.0f, stream_);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  std::vector<std::unique_ptr<FullyConnectedLayer>> repetitions;
  repetitions.push_back(std::move(*first));
  repetitions.push_back(std::move(*second));

  RepeatedLayer<FullyConnectedLayer> repeated(DataType::FP16,
                                               std::move(repetitions));
  // Each dense repetition contributes its own matrix and bias.
  EXPECT_EQ(repeated.weights().size(), 4u);
}

}  // namespace
}  // namespace pluto::llm
