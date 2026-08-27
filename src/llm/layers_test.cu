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
  auto layer = EmbeddingLookupLayer::Create(
      kVocabularySize, kModelWidth, DataType::FP8, 0.1f, stream_);
  EXPECT_FALSE(layer.ok());
  EXPECT_EQ(layer.status().code(), absl::StatusCode::kUnimplemented);
}

TEST_F(LayersTest, RejectsInvalidEmbeddingDimensions) {
  auto embedding = EmbeddingLookupLayer::Create(
      0, kModelWidth, DataType::FP16, 0.1f, stream_);
  EXPECT_FALSE(embedding.ok());
  EXPECT_EQ(embedding.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(LayersTest, LanguageModelingHeadUsesEmbeddingWeightTranspose) {
  auto embedding = EmbeddingLookupLayer::Create(
      kVocabularySize, kModelWidth, DataType::FP16, 0.25f, stream_);
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  EXPECT_EQ((*embedding)->vocab_size(), kVocabularySize);
  EXPECT_EQ((*embedding)->embedding_dim(), kModelWidth);
  auto head = LanguageModelingHeadLayer::Create(embedding->get());
  ASSERT_TRUE(head.ok()) << head.status();
  ASSERT_EQ((*embedding)->weights().size(), 1u);
  ASSERT_EQ((*head)->weights().size(), 1u);
  EXPECT_EQ((*head)->weights().front().data(), (*embedding)->weight().data());

  std::vector<float> table(kVocabularySize * kModelWidth, 0.0f);
  table[3 * kModelWidth + 5] = 2.0f;
  table[7 * kModelWidth + 5] = 3.0f;
  ASSERT_EQ(cudaMemcpyAsync((*embedding)->weights().front().data(),
                            table.data(), table.size() * sizeof(float),
                            cudaMemcpyHostToDevice, stream_),
            cudaSuccess);
  std::vector<int> tokens(kBatchSize, 3);
  auto token_buffer = Buffer::Allocate(tokens.size() * sizeof(int), stream_);
  ASSERT_TRUE(token_buffer.ok()) << token_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(token_buffer->data(), tokens.data(),
                            token_buffer->size_bytes(), cudaMemcpyHostToDevice,
                            stream_),
            cudaSuccess);

  Tape embedding_tape;
  BufferVec embedding_inputs = {*token_buffer};
  auto hidden = (*embedding)->fwd(embedding_inputs, &embedding_tape);
  ASSERT_TRUE(hidden.ok()) << hidden.status();
  Tape head_tape;
  BufferVec head_inputs = {*hidden};
  auto logits = (*head)->fwd(head_inputs, &head_tape);
  ASSERT_TRUE(logits.ok()) << logits.status();

  std::vector<float> output_gradient(kBatchSize * kVocabularySize, 0.0f);
  output_gradient[7] = 1.0f;
  auto gradient_buffer =
      Buffer::Allocate(output_gradient.size() * sizeof(float), stream_);
  ASSERT_TRUE(gradient_buffer.ok()) << gradient_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(gradient_buffer->data(), output_gradient.data(),
                            gradient_buffer->size_bytes(),
                            cudaMemcpyHostToDevice, stream_),
            cudaSuccess);
  BufferVec head_gradients = {*gradient_buffer};
  auto hidden_gradient =
      (*head)->bwd(head_gradients, std::move(head_tape));
  ASSERT_TRUE(hidden_gradient.ok()) << hidden_gradient.status();
  ASSERT_EQ(hidden_gradient->size(), 1u);

  std::vector<float> host_logits(kVocabularySize);
  std::vector<float> host_hidden_gradient(kModelWidth);
  std::vector<float> updated_table(table.size());
  ASSERT_EQ(cudaMemcpyAsync(host_logits.data(), logits->data(),
                            host_logits.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(host_hidden_gradient.data(),
                            hidden_gradient->front().data(),
                            host_hidden_gradient.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(updated_table.data(), (*embedding)->weight().data(),
                            updated_table.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_),
            cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  EXPECT_FLOAT_EQ(host_logits[3], 4.0f);
  EXPECT_FLOAT_EQ(host_logits[7], 6.0f);
  EXPECT_FLOAT_EQ(host_hidden_gradient[5], 3.0f);
  EXPECT_FLOAT_EQ(updated_table[7 * kModelWidth + 5], 2.5f);
}

TEST_F(LayersTest, LanguageModelingHeadRejectsNullEmbedding) {
  auto head = LanguageModelingHeadLayer::Create(nullptr);
  EXPECT_FALSE(head.ok());
  EXPECT_EQ(head.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(LayersTest, FlashAttentionIsCausalAndHasCorrectSingleTokenGradient) {
  auto attention = AttentionLayer::Create(
      kContextLength, kAttentionHeads, kModelWidth, DataType::FP16, stream_);
  ASSERT_TRUE(attention.ok()) << attention.status();

  std::vector<float> input(kBatchSize * kModelWidth, 0.0f);
  // Only the first feature of the first head is nonzero. Position zero must
  // ignore the larger future values, while position one attends to 1 and 2.
  input[0] = 1.0f;
  input[kModelWidth] = 2.0f;
  input[2 * kModelWidth] = 4.0f;
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

  const float first_weight = std::exp(0.25f);
  const float second_weight = std::exp(0.5f);
  const float expected_position_one =
      (first_weight + 2.0f * second_weight) /
      (first_weight + second_weight);
  EXPECT_NEAR(host_output[0], 1.0f, 1e-6f);
  EXPECT_NEAR(host_output[kModelWidth], expected_position_one, 1e-5f);
  EXPECT_NEAR(host_input_gradient[0], 1.0f, 1e-6f);
  EXPECT_NEAR(host_input_gradient[kModelWidth], 0.0f, 1e-6f);
}

TEST_F(LayersTest, LayerNormNormalizesRowsAndRejectsConstantGradient) {
  auto layer_norm =
      LayerNormLayer::Create(kModelWidth, 1e-5f, DataType::FP16, stream_);
  ASSERT_TRUE(layer_norm.ok()) << layer_norm.status();

  std::vector<float> input(kBatchSize * kModelWidth);
  std::vector<float> output_gradient(input.size(), 1.0f);
  for (int row = 0; row < kBatchSize; ++row) {
    for (int column = 0; column < kModelWidth; ++column) {
      input[row * kModelWidth + column] =
          static_cast<float>(column) / kModelWidth;
    }
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

  Tape tape;
  BufferVec inputs = {*input_buffer};
  auto output = (*layer_norm)->fwd(inputs, &tape);
  ASSERT_TRUE(output.ok()) << output.status();
  BufferVec gradients = {*gradient_buffer};
  auto input_gradient =
      (*layer_norm)->bwd(gradients, std::move(tape));
  ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();

  std::vector<float> host_output(kModelWidth);
  std::vector<float> host_input_gradient(kModelWidth);
  ASSERT_EQ(cudaMemcpyAsync(host_output.data(), output->data(),
                            host_output.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(host_input_gradient.data(),
                            input_gradient->front().data(),
                            host_input_gradient.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_),
            cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  double mean = 0.0;
  double square_mean = 0.0;
  for (float value : host_output) {
    mean += value;
    square_mean += value * value;
  }
  mean /= kModelWidth;
  square_mean /= kModelWidth;
  EXPECT_NEAR(mean, 0.0, 1e-5);
  EXPECT_NEAR(square_mean, 1.0, 2e-4);
  for (float gradient : host_input_gradient) {
    EXPECT_NEAR(gradient, 0.0f, 1e-5f);
  }
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
  BufferVec loss_inputs = {*logits_buffer, *target_buffer};
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
