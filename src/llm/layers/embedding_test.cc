#include "src/llm/layers/embedding.h"

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

TEST_F(LayersTest, RejectsFp8UntilScalingIsSpecified) {
  auto layer = EmbeddingLookupLayer::Create(
      kTestVocabularySize, kTestModelWidth, DataType::FP8, *executor_);
  EXPECT_FALSE(layer.ok());
  EXPECT_EQ(layer.status().code(), absl::StatusCode::kUnimplemented);
}

TEST_F(LayersTest, RejectsInvalidEmbeddingDimensions) {
  auto embedding = EmbeddingLookupLayer::Create(0, kTestModelWidth,
                                                DataType::FP16, *executor_);
  EXPECT_FALSE(embedding.ok());
  EXPECT_EQ(embedding.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(LayersTest, LanguageModelingHeadUsesEmbeddingWeightTranspose) {
  auto embedding = EmbeddingLookupLayer::Create(
      kTestVocabularySize, kTestModelWidth, DataType::FP16, *executor_);
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  EXPECT_EQ((*embedding)->vocab_size(), kTestVocabularySize);
  EXPECT_EQ((*embedding)->embedding_dim(), kTestModelWidth);
  auto head = LanguageModelingHeadLayer::Create(embedding->get());
  ASSERT_TRUE(head.ok()) << head.status();
  ASSERT_EQ((*embedding)->weights().size(), 1u);
  ASSERT_EQ((*head)->weights().size(), 1u);
  EXPECT_EQ((*head)->weights().front().data(), (*embedding)->weight().data());

  std::vector<float> table(kTestVocabularySize * kTestModelWidth, 0.0f);
  table[3 * kTestModelWidth + 5] = 2.0f;
  table[7 * kTestModelWidth + 5] = 3.0f;
  ASSERT_EQ(cudaMemcpyAsync((*embedding)->weights().front().data(),
                            table.data(), table.size() * sizeof(float),
                            cudaMemcpyHostToDevice, executor_->stream()),
            cudaSuccess);
  std::vector<int> tokens(kTestBatchSize, 3);
  auto token_buffer = Buffer::Allocate(tokens.size() * sizeof(int), *executor_);
  ASSERT_TRUE(token_buffer.ok()) << token_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(token_buffer->data(), tokens.data(),
                            token_buffer->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);

  Tape embedding_tape;
  BufferVec embedding_inputs = {*token_buffer};
  auto hidden =
      (*embedding)->fwd(embedding_inputs, &embedding_tape, *executor_);
  ASSERT_TRUE(hidden.ok()) << hidden.status();
  Tape head_tape;
  BufferVec head_inputs = {*hidden};
  auto logits = (*head)->fwd(head_inputs, &head_tape, *executor_);
  ASSERT_TRUE(logits.ok()) << logits.status();

  std::vector<float> output_gradient(kTestBatchSize * kTestVocabularySize,
                                     0.0f);
  output_gradient[7] = 1.0f;
  auto gradient_buffer =
      Buffer::Allocate(output_gradient.size() * sizeof(float), *executor_);
  ASSERT_TRUE(gradient_buffer.ok()) << gradient_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(gradient_buffer->data(), output_gradient.data(),
                            gradient_buffer->size_bytes(),
                            cudaMemcpyHostToDevice, executor_->stream()),
            cudaSuccess);
  BufferVec head_gradients = {*gradient_buffer};
  auto hidden_gradient =
      (*head)->bwd(head_gradients, std::move(head_tape), *executor_);
  ASSERT_TRUE(hidden_gradient.ok()) << hidden_gradient.status();
  ASSERT_EQ(hidden_gradient->size(), 1u);

  std::vector<float> host_logits(kTestVocabularySize);
  std::vector<float> host_hidden_gradient(kTestModelWidth);
  std::vector<float> unchanged_table(table.size());
  std::vector<float> table_gradient(table.size());
  ASSERT_EQ(cudaMemcpyAsync(host_logits.data(), logits->data(),
                            host_logits.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, executor_->stream()),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(host_hidden_gradient.data(),
                            hidden_gradient->front().data(),
                            host_hidden_gradient.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, executor_->stream()),
            cudaSuccess);
  ASSERT_EQ(
      cudaMemcpyAsync(unchanged_table.data(), (*embedding)->weight().data(),
                      unchanged_table.size() * sizeof(float),
                      cudaMemcpyDeviceToHost, executor_->stream()),
      cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(table_gradient.data(),
                            (*embedding)->gradients().front().data(),
                            table_gradient.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, executor_->stream()),
            cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());

  EXPECT_FLOAT_EQ(host_logits[3], 4.0f);
  EXPECT_FLOAT_EQ(host_logits[7], 6.0f);
  EXPECT_FLOAT_EQ(host_hidden_gradient[5], 3.0f);
  // Backward accumulates an FP32 tied-table gradient; the optimizer, not the
  // layer, owns parameter updates.
  EXPECT_FLOAT_EQ(unchanged_table[7 * kTestModelWidth + 5], 3.0f);
  EXPECT_FLOAT_EQ(table_gradient[7 * kTestModelWidth + 5], 2.0f);
}

TEST_F(LayersTest, LanguageModelingHeadRejectsNullEmbedding) {
  auto head = LanguageModelingHeadLayer::Create(nullptr);
  EXPECT_FALSE(head.ok());
  EXPECT_EQ(head.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(LayersTest, Bf16HeadMasksPhysicalVocabularyPadding) {
  constexpr int kLogicalVocabularySize = 17;
  constexpr int kPaddedVocabularySize = 32;
  auto embedding = EmbeddingLookupLayer::Create(
      kLogicalVocabularySize, kTestModelWidth, DataType::BF16, *executor_);
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  EXPECT_EQ((*embedding)->padded_vocab_size(), kPaddedVocabularySize);
  auto head = LanguageModelingHeadLayer::Create(embedding->get());
  ASSERT_TRUE(head.ok()) << head.status();

  std::vector<float> table(kPaddedVocabularySize * kTestModelWidth, 0.0f);
  table[3 * kTestModelWidth + 5] = 2.0f;
  ASSERT_EQ(cudaMemcpyAsync((*embedding)->weights().front().data(),
                            table.data(), table.size() * sizeof(float),
                            cudaMemcpyHostToDevice, executor_->stream()),
            cudaSuccess);
  std::vector<int> tokens(kTestBatchSize, 3);
  auto token_buffer = Buffer::Allocate(tokens.size() * sizeof(int), *executor_);
  ASSERT_TRUE(token_buffer.ok()) << token_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(token_buffer->data(), tokens.data(),
                            token_buffer->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);

  Tape embedding_tape;
  BufferVec embedding_inputs = {*token_buffer};
  auto hidden =
      (*embedding)->fwd(embedding_inputs, &embedding_tape, *executor_);
  ASSERT_TRUE(hidden.ok()) << hidden.status();
  EXPECT_EQ(hidden->size_bytes(),
            kTestBatchSize * kTestModelWidth * sizeof(uint16_t));
  Tape head_tape;
  BufferVec head_inputs = {*hidden};
  auto logits = (*head)->fwd(head_inputs, &head_tape, *executor_);
  ASSERT_TRUE(logits.ok()) << logits.status();
  EXPECT_EQ(logits->size_bytes(),
            kTestBatchSize * kPaddedVocabularySize * sizeof(float));

  std::vector<float> host_logits(kPaddedVocabularySize);
  ASSERT_EQ(cudaMemcpyAsync(host_logits.data(), logits->data(),
                            host_logits.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, executor_->stream()),
            cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());
  EXPECT_FLOAT_EQ(host_logits[3], 4.0f);
  for (int token = kLogicalVocabularySize; token < kPaddedVocabularySize;
       ++token) {
    EXPECT_LT(host_logits[token], -1e30f);
  }
}

TEST_F(LayersTest, PositionEmbeddingRepeatsAtRuntimeContextLength) {
  auto positions = PositionEmbeddingLayer::Create(
      kTestContextLength, kTestModelWidth, DataType::FP16, *executor_);
  ASSERT_TRUE(positions.ok()) << positions.status();

  std::vector<float> weight(kTestContextLength * kTestModelWidth, 0.0f);
  for (int position = 0; position < kTestContextLength; ++position) {
    weight[position * kTestModelWidth] = static_cast<float>(position + 1);
  }
  ASSERT_EQ(cudaMemcpyAsync((*positions)->weights().front().data(),
                            weight.data(), weight.size() * sizeof(float),
                            cudaMemcpyHostToDevice, executor_->stream()),
            cudaSuccess);
  std::vector<float> input(kTestBatchSize * kTestModelWidth, 0.0f);
  auto input_buffer =
      Buffer::Allocate(input.size() * sizeof(float), *executor_);
  ASSERT_TRUE(input_buffer.ok()) << input_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(input_buffer->data(), input.data(),
                            input_buffer->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);

  Tape tape;
  BufferVec inputs = {*input_buffer};
  auto output = (*positions)->fwd(inputs, &tape, *executor_);
  ASSERT_TRUE(output.ok()) << output.status();
  std::vector<float> host_output(input.size());
  ASSERT_EQ(
      cudaMemcpyAsync(host_output.data(), output->data(), output->size_bytes(),
                      cudaMemcpyDeviceToHost, executor_->stream()),
      cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());

  for (int row = 0; row < kTestBatchSize; ++row) {
    EXPECT_FLOAT_EQ(host_output[row * kTestModelWidth],
                    static_cast<float>(row % kTestContextLength + 1));
  }
}

}  // namespace
}  // namespace pluto::llm
