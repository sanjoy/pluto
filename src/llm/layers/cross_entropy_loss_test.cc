#include "src/llm/layers/cross_entropy_loss.h"

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

TEST_F(LayersTest, CrossEntropyForwardAndBackwardMatchUniformSoftmax) {
  std::vector<float> logits(kTestBatchSize * kTestVocabularySize, 0.0f);
  std::vector<int> targets(kTestBatchSize);
  for (int row = 0; row < kTestBatchSize; ++row)
    targets[row] = row;
  const auto pinned_logits = CopyToPageLockedHostArray(logits);
  const auto pinned_targets = CopyToPageLockedHostArray(targets);

  auto logits_buffer =
      Buffer::Allocate(*executor_, logits.size() * sizeof(float));
  auto target_buffer =
      Buffer::Allocate(*executor_, targets.size() * sizeof(int));
  ASSERT_TRUE(logits_buffer.ok()) << logits_buffer.status();
  ASSERT_TRUE(target_buffer.ok()) << target_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(logits_buffer->data(), pinned_logits.data(),
                            logits_buffer->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(target_buffer->data(), pinned_targets.data(),
                            target_buffer->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);

  auto loss_layer = CrossEntropyLossLayer::Create(
      *executor_, kTestVocabularySize, DataType::FP16);
  ASSERT_TRUE(loss_layer.ok()) << loss_layer.status();
  Tape tape;
  BufferVec loss_inputs = {*logits_buffer, *target_buffer};
  auto losses = (*loss_layer)->fwd(*executor_, loss_inputs, &tape);
  ASSERT_TRUE(losses.ok()) << losses.status();
  auto gradients = (*loss_layer)->bwd(*executor_, {}, std::move(tape));
  ASSERT_TRUE(gradients.ok()) << gradients.status();
  ASSERT_EQ(gradients->size(), 1u);

  auto host_losses = AllocatePageLockedHostArray<float>(kTestBatchSize);
  auto host_gradients = AllocatePageLockedHostArray<float>(logits.size());
  ASSERT_EQ(
      cudaMemcpyAsync(host_losses.data(), losses->data(), losses->size_bytes(),
                      cudaMemcpyDeviceToHost, executor_->stream()),
      cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(host_gradients.data(), gradients->front().data(),
                            gradients->front().size_bytes(),
                            cudaMemcpyDeviceToHost, executor_->stream()),
            cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());

  EXPECT_NEAR(host_losses[0], std::log(static_cast<float>(kTestVocabularySize)),
              1e-5f);
  EXPECT_NEAR(host_gradients[0],
              (1.0f / kTestVocabularySize - 1.0f) / kTestBatchSize, 1e-6f);
  EXPECT_NEAR(host_gradients[1], 1.0f / (kTestVocabularySize * kTestBatchSize),
              1e-7f);
}

TEST_F(LayersTest, IgnoresPaddedVocabularyColumns) {
  constexpr int kLogicalVocabularySize = 17;
  constexpr int kPaddedVocabularySize = 32;
  std::vector<float> logits(kTestBatchSize * kPaddedVocabularySize, 0.0f);
  for (int row = 0; row < kTestBatchSize; ++row) {
    for (int token = kLogicalVocabularySize; token < kPaddedVocabularySize;
         ++token) {
      logits[row * kPaddedVocabularySize + token] = -3.402823466e+38f;
    }
  }
  std::vector<int> targets(kTestBatchSize, 0);
  const auto pinned_logits = CopyToPageLockedHostArray(logits);
  const auto pinned_targets = CopyToPageLockedHostArray(targets);
  auto logits_buffer =
      Buffer::Allocate(*executor_, logits.size() * sizeof(float));
  auto target_buffer =
      Buffer::Allocate(*executor_, targets.size() * sizeof(int));
  ASSERT_TRUE(logits_buffer.ok()) << logits_buffer.status();
  ASSERT_TRUE(target_buffer.ok()) << target_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(logits_buffer->data(), pinned_logits.data(),
                            logits_buffer->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(target_buffer->data(), pinned_targets.data(),
                            target_buffer->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);
  auto loss_layer = CrossEntropyLossLayer::Create(
      *executor_, kLogicalVocabularySize, DataType::BF16);
  ASSERT_TRUE(loss_layer.ok()) << loss_layer.status();
  EXPECT_EQ((*loss_layer)->padded_vocab_size(), kPaddedVocabularySize);
  Tape tape;
  BufferVec inputs = {*logits_buffer, *target_buffer};
  auto losses = (*loss_layer)->fwd(*executor_, inputs, &tape);
  ASSERT_TRUE(losses.ok()) << losses.status();
  auto host_losses = AllocatePageLockedHostArray<float>(kTestBatchSize);
  ASSERT_EQ(
      cudaMemcpyAsync(host_losses.data(), losses->data(), losses->size_bytes(),
                      cudaMemcpyDeviceToHost, executor_->stream()),
      cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());
  EXPECT_NEAR(host_losses[0], std::log(17.0f), 1e-5f);
}

}  // namespace
}  // namespace pluto::llm
