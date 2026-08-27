#include "src/llm/layers/cross_entropy_loss.h"

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

TEST_F(LayersTest, CrossEntropyForwardAndBackwardMatchUniformSoftmax) {
  std::vector<float> logits(kTestBatchSize * kTestVocabularySize, 0.0f);
  std::vector<int> targets(kTestBatchSize);
  for (int row = 0; row < kTestBatchSize; ++row) targets[row] = row;

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

  auto loss_layer = CrossEntropyLossLayer::Create(
      kTestVocabularySize, DataType::FP16, stream_);
  ASSERT_TRUE(loss_layer.ok()) << loss_layer.status();
  Tape tape;
  BufferVec loss_inputs = {*logits_buffer, *target_buffer};
  auto losses = (*loss_layer)->fwd(loss_inputs, &tape);
  ASSERT_TRUE(losses.ok()) << losses.status();
  auto gradients = (*loss_layer)->bwd({}, std::move(tape));
  ASSERT_TRUE(gradients.ok()) << gradients.status();
  ASSERT_EQ(gradients->size(), 1u);

  std::vector<float> host_losses(kTestBatchSize);
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

  EXPECT_NEAR(host_losses[0], std::log(static_cast<float>(kTestVocabularySize)),
              1e-5f);
  EXPECT_NEAR(host_gradients[0],
              (1.0f / kTestVocabularySize - 1.0f) / kTestBatchSize, 1e-6f);
  EXPECT_NEAR(host_gradients[1],
              1.0f / (kTestVocabularySize * kTestBatchSize), 1e-7f);
}


}  // namespace
}  // namespace pluto::llm
