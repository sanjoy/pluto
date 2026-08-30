#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/llm/layer.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/reference_test_util.h"

namespace pluto::llm {
namespace {

TEST_F(LayerReferenceTest, LookupAndTiedHeadMatchAcrossShapesAndTypes) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (const auto [vocab, width, rows] :
         {std::tuple{17, 16, 16}, std::tuple{32, 32, 32}}) {
      SCOPED_TRACE(testing::Message()
                   << "type=" << static_cast<int>(type) << " vocab=" << vocab
                   << " width=" << width << " rows=" << rows);
      auto device_embedding =
          EmbeddingLookupLayer::Create(vocab, width, type, *executor_);
      auto reference_embedding =
          EmbeddingLookupLayerReference::Create(vocab, width, type);
      ASSERT_TRUE(device_embedding.ok()) << device_embedding.status();
      ASSERT_TRUE(reference_embedding.ok()) << reference_embedding.status();
      const int padded = (*device_embedding)->padded_vocab_size();
      std::vector<float> table(static_cast<size_t>(padded) * width);
      for (size_t index = 0; index < table.size(); ++index) {
        table[index] = 0.12f * std::sin(static_cast<float>(index) * 0.091f);
      }
      auto device_weights = (*device_embedding)->weights();
      auto reference_weights = (*reference_embedding)->weights();
      ASSERT_TRUE(SetFloatBufferPair(device_weights[0], &reference_weights[0],
                                     table, *executor_)
                      .ok());

      std::vector<int> tokens(rows);
      std::vector<float> lookup_gradient(static_cast<size_t>(rows) * width);
      for (int row = 0; row < rows; ++row) {
        tokens[row] = (row * 5 + row / 3) % vocab;
      }
      for (size_t index = 0; index < lookup_gradient.size(); ++index) {
        lookup_gradient[index] =
            0.15f * std::cos(static_cast<float>(index) * 0.17f);
      }
      auto tokens_pair = MakeRawBufferPair<int>(tokens, *executor_);
      auto lookup_gradient_pair =
          MakeRawBufferPair<float>(lookup_gradient, *executor_);
      ASSERT_TRUE(tokens_pair.ok()) << tokens_pair.status();
      ASSERT_TRUE(lookup_gradient_pair.ok()) << lookup_gradient_pair.status();
      Tape device_lookup_tape;
      ReferenceTape reference_lookup_tape;
      BufferVec device_tokens = {tokens_pair->device};
      HostBufferVec reference_tokens = {tokens_pair->host};
      auto device_lookup =
          (*device_embedding)
              ->fwd(device_tokens, &device_lookup_tape, *executor_);
      auto reference_lookup =
          (*reference_embedding)->fwd(reference_tokens, &reference_lookup_tape);
      ASSERT_TRUE(device_lookup.ok()) << device_lookup.status();
      ASSERT_TRUE(reference_lookup.ok()) << reference_lookup.status();
      EXPECT_TRUE(
          ActivationBuffersNear(*device_lookup, *reference_lookup, type, 0.0f));
      BufferVec device_lookup_gradients = {lookup_gradient_pair->device};
      HostBufferVec reference_lookup_gradients = {lookup_gradient_pair->host};
      auto device_lookup_input =
          (*device_embedding)
              ->bwd(device_lookup_gradients, std::move(device_lookup_tape),
                    *executor_);
      auto reference_lookup_input = (*reference_embedding)
                                        ->bwd(reference_lookup_gradients,
                                              std::move(reference_lookup_tape));
      ASSERT_TRUE(device_lookup_input.ok()) << device_lookup_input.status();
      ASSERT_TRUE(reference_lookup_input.ok())
          << reference_lookup_input.status();
      EXPECT_TRUE(device_lookup_input->empty());
      EXPECT_TRUE(reference_lookup_input->empty());
      auto device_table_gradients = (*device_embedding)->gradients();
      auto reference_table_gradients = (*reference_embedding)->gradients();
      EXPECT_TRUE(FloatBuffersNear(device_table_gradients[0],
                                   reference_table_gradients[0], 2e-6f, 2e-5f));

      // Test the tied output projection independently, including its additive
      // contribution to the same embedding-table gradient.
      ASSERT_TRUE(ZeroBufferPair(device_table_gradients[0],
                                 &reference_table_gradients[0], *executor_)
                      .ok());
      auto device_head =
          LanguageModelingHeadLayer::Create(device_embedding->get());
      auto reference_head = LanguageModelingHeadLayerReference::Create(
          reference_embedding->get());
      ASSERT_TRUE(device_head.ok()) << device_head.status();
      ASSERT_TRUE(reference_head.ok()) << reference_head.status();
      std::vector<float> hidden(static_cast<size_t>(rows) * width);
      std::vector<float> logits_gradient(static_cast<size_t>(rows) * padded);
      for (size_t index = 0; index < hidden.size(); ++index) {
        hidden[index] = 0.4f * std::sin(static_cast<float>(index) * 0.13f);
      }
      for (size_t index = 0; index < logits_gradient.size(); ++index) {
        logits_gradient[index] =
            0.08f * std::cos(static_cast<float>(index) * 0.07f);
      }
      auto hidden_pair = MakeActivationBufferPair(hidden, type, *executor_);
      auto logits_gradient_pair =
          MakeRawBufferPair<float>(logits_gradient, *executor_);
      ASSERT_TRUE(hidden_pair.ok()) << hidden_pair.status();
      ASSERT_TRUE(logits_gradient_pair.ok()) << logits_gradient_pair.status();
      Tape device_head_tape;
      ReferenceTape reference_head_tape;
      BufferVec device_hidden = {hidden_pair->device};
      HostBufferVec reference_hidden = {hidden_pair->host};
      auto device_logits =
          (*device_head)->fwd(device_hidden, &device_head_tape, *executor_);
      auto reference_logits =
          (*reference_head)->fwd(reference_hidden, &reference_head_tape);
      ASSERT_TRUE(device_logits.ok()) << device_logits.status();
      ASSERT_TRUE(reference_logits.ok()) << reference_logits.status();
      EXPECT_TRUE(
          FloatBuffersNear(*device_logits, *reference_logits, 4e-3f, 3e-3f));
      BufferVec device_logits_gradients = {logits_gradient_pair->device};
      HostBufferVec reference_logits_gradients = {logits_gradient_pair->host};
      auto device_hidden_gradient =
          (*device_head)
              ->bwd(device_logits_gradients, std::move(device_head_tape),
                    *executor_);
      auto reference_hidden_gradient =
          (*reference_head)
              ->bwd(reference_logits_gradients, std::move(reference_head_tape));
      ASSERT_TRUE(device_hidden_gradient.ok())
          << device_hidden_gradient.status();
      ASSERT_TRUE(reference_hidden_gradient.ok())
          << reference_hidden_gradient.status();
      EXPECT_TRUE(FloatBuffersNear(device_hidden_gradient->front(),
                                   reference_hidden_gradient->front(), 4e-3f,
                                   3e-3f));
      EXPECT_TRUE(FloatBuffersNear(device_table_gradients[0],
                                   reference_table_gradients[0], 5e-3f, 3e-3f));
    }
  }
}

TEST_F(LayerReferenceTest, PositionEmbeddingForwardAndBackwardMatch) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (const auto [context, width, rows] :
         {std::tuple{4, 16, 8}, std::tuple{8, 32, 24}}) {
      auto device =
          PositionEmbeddingLayer::Create(context, width, type, *executor_);
      auto reference =
          PositionEmbeddingLayerReference::Create(context, width, type);
      ASSERT_TRUE(device.ok()) << device.status();
      ASSERT_TRUE(reference.ok()) << reference.status();
      std::vector<float> positions(static_cast<size_t>(context) * width);
      for (size_t index = 0; index < positions.size(); ++index) {
        positions[index] = 0.1f * std::cos(static_cast<float>(index) * 0.2f);
      }
      auto device_weights = (*device)->weights();
      auto reference_weights = (*reference)->weights();
      ASSERT_TRUE(SetFloatBufferPair(device_weights[0], &reference_weights[0],
                                     positions, *executor_)
                      .ok());
      std::vector<float> input(static_cast<size_t>(rows) * width);
      std::vector<float> gradient(input.size());
      for (size_t index = 0; index < input.size(); ++index) {
        input[index] = 0.3f * std::sin(static_cast<float>(index) * 0.09f);
        gradient[index] = 0.2f * std::cos(static_cast<float>(index) * 0.15f);
      }
      auto input_pair = MakeActivationBufferPair(input, type, *executor_);
      auto gradient_pair = MakeRawBufferPair<float>(gradient, *executor_);
      ASSERT_TRUE(input_pair.ok()) << input_pair.status();
      ASSERT_TRUE(gradient_pair.ok()) << gradient_pair.status();
      Tape device_tape;
      ReferenceTape reference_tape;
      BufferVec device_inputs = {input_pair->device};
      HostBufferVec reference_inputs = {input_pair->host};
      auto device_output =
          (*device)->fwd(device_inputs, &device_tape, *executor_);
      auto reference_output =
          (*reference)->fwd(reference_inputs, &reference_tape);
      ASSERT_TRUE(device_output.ok()) << device_output.status();
      ASSERT_TRUE(reference_output.ok()) << reference_output.status();
      EXPECT_TRUE(
          ActivationBuffersNear(*device_output, *reference_output, type, 0.0f));
      BufferVec device_gradients = {gradient_pair->device};
      HostBufferVec reference_gradients = {gradient_pair->host};
      auto device_input =
          (*device)->bwd(device_gradients, std::move(device_tape), *executor_);
      auto reference_input =
          (*reference)->bwd(reference_gradients, std::move(reference_tape));
      ASSERT_TRUE(device_input.ok()) << device_input.status();
      ASSERT_TRUE(reference_input.ok()) << reference_input.status();
      EXPECT_TRUE(FloatBuffersNear(device_input->front(),
                                   reference_input->front(), 0.0f));
      EXPECT_TRUE(FloatBuffersNear((*device)->gradients()[0],
                                   (*reference)->gradients()[0], 2e-6f, 2e-5f));
    }
  }
}

TEST_F(LayerReferenceTest, FP8IsRejectedConsistently) {
  auto device_lookup =
      EmbeddingLookupLayer::Create(17, 16, DataType::FP8, *executor_);
  auto reference_lookup =
      EmbeddingLookupLayerReference::Create(17, 16, DataType::FP8);
  auto device_position =
      PositionEmbeddingLayer::Create(4, 16, DataType::FP8, *executor_);
  auto reference_position =
      PositionEmbeddingLayerReference::Create(4, 16, DataType::FP8);
  ASSERT_FALSE(device_lookup.ok());
  ASSERT_FALSE(reference_lookup.ok());
  ASSERT_FALSE(device_position.ok());
  ASSERT_FALSE(reference_position.ok());
  EXPECT_EQ(device_lookup.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(reference_lookup.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(device_position.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(reference_position.status().code(),
            absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace pluto::llm
