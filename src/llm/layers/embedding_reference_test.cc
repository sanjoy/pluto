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
         {std::tuple{17, 16, 16}, std::tuple{32, 32, 32},
          std::tuple{17, 48, 48}, std::tuple{65, 80, 80},
          std::tuple{257, 48, 16}, std::tuple{257, 80, 48},
          std::tuple{50257, 16, 16}}) {
      SCOPED_TRACE(testing::Message()
                   << "type=" << static_cast<int>(type) << " vocab=" << vocab
                   << " width=" << width << " rows=" << rows);
      auto device_embedding =
          EmbeddingLookupLayer::Create(*executor_, vocab, width, type);
      auto reference_embedding =
          EmbeddingLookupLayerReference::Create(vocab, width, type);
      ASSERT_TRUE(device_embedding.ok()) << device_embedding.status();
      ASSERT_TRUE(reference_embedding.ok()) << reference_embedding.status();
      const int padded = (*device_embedding)->padded_vocab_size();
      EXPECT_EQ(padded, ((vocab + 15) / 16) * 16);
      EXPECT_EQ((*reference_embedding)->padded_vocab_size(), padded);
      std::vector<float> table(static_cast<size_t>(padded) * width);
      for (size_t index = 0; index < table.size(); ++index)
        table[index] = 0.12f * std::sin(static_cast<float>(index) * 0.091f);
      auto device_weights = (*device_embedding)->weights();
      auto reference_weights = (*reference_embedding)->weights();
      ASSERT_TRUE(SetFloatBufferPair(*executor_, device_weights[0],
                                     &reference_weights[0], table)
                      .ok());

      auto device_table_gradients = (*device_embedding)->gradients();
      auto reference_table_gradients = (*reference_embedding)->gradients();
      std::vector<float> initial_gradient(table.size());
      for (size_t index = 0; index < initial_gradient.size(); ++index)
        initial_gradient[index] =
            0.25f + 0.03f * std::sin(static_cast<float>(index) * 0.11f);
      ASSERT_TRUE(SetFloatBufferPair(*executor_, device_table_gradients[0],
                                     &reference_table_gradients[0],
                                     initial_gradient)
                      .ok());

      std::vector<int> tokens(rows);
      std::vector<float> lookup_gradient(static_cast<size_t>(rows) * width);
      for (int row = 0; row < rows; ++row)
        tokens[row] = (row * 5 + row / 3) % vocab;
      for (size_t index = 0; index < lookup_gradient.size(); ++index) {
        lookup_gradient[index] =
            0.15f * std::cos(static_cast<float>(index) * 0.17f);
      }
      auto tokens_pair = MakeRawBufferPair<int>(*executor_, tokens);
      auto lookup_gradient_pair =
          MakeRawBufferPair<float>(*executor_, lookup_gradient);
      ASSERT_TRUE(tokens_pair.ok()) << tokens_pair.status();
      ASSERT_TRUE(lookup_gradient_pair.ok()) << lookup_gradient_pair.status();

      BufferVec device_tokens = {tokens_pair->device};
      HostBufferVec reference_tokens = {tokens_pair->host};
      auto device_lookup = (*device_embedding)->fwd(*executor_, device_tokens);

      auto reference_lookup = (*reference_embedding)->fwd(reference_tokens);

      ASSERT_TRUE(device_lookup.ok()) << device_lookup.status();
      ASSERT_TRUE(reference_lookup.ok()) << reference_lookup.status();
      EXPECT_TRUE(ActivationBuffersNear(
          device_lookup->outputs[0], reference_lookup->outputs[0], type, 0.0f));
      BufferVec device_lookup_gradients = {lookup_gradient_pair->device};
      HostBufferVec reference_lookup_gradients = {lookup_gradient_pair->host};
      auto device_lookup_input = (*device_embedding)
                                     ->bwd(*executor_, device_lookup_gradients,
                                           std::move(device_lookup->state));
      auto reference_lookup_input =
          (*reference_embedding)
              ->bwd(reference_lookup_gradients,
                    std::move(reference_lookup->state));
      ASSERT_TRUE(device_lookup_input.ok()) << device_lookup_input.status();
      ASSERT_TRUE(reference_lookup_input.ok())
          << reference_lookup_input.status();
      EXPECT_TRUE(device_lookup_input->empty());
      EXPECT_TRUE(reference_lookup_input->empty());
      EXPECT_TRUE(FloatBuffersNear(device_table_gradients[0],
                                   reference_table_gradients[0], 2e-6f, 2e-5f));

      // The head must add to both the initial accumulator and the lookup's
      // contribution, including physical vocabulary lanes beyond vocab.
      auto device_head =
          LanguageModelingHeadLayer::Create(device_embedding->get());
      auto reference_head = LanguageModelingHeadLayerReference::Create(
          reference_embedding->get());
      ASSERT_TRUE(device_head.ok()) << device_head.status();
      ASSERT_TRUE(reference_head.ok()) << reference_head.status();
      std::vector<float> hidden(static_cast<size_t>(rows) * width);
      std::vector<float> logits_gradient(static_cast<size_t>(rows) * padded);
      for (size_t index = 0; index < hidden.size(); ++index)
        hidden[index] = 0.4f * std::sin(static_cast<float>(index) * 0.13f);
      for (size_t index = 0; index < logits_gradient.size(); ++index) {
        logits_gradient[index] =
            0.08f * std::cos(static_cast<float>(index) * 0.07f);
      }
      auto hidden_pair = MakeActivationBufferPair(*executor_, hidden, type);
      auto logits_gradient_pair =
          MakeRawBufferPair<float>(*executor_, logits_gradient);
      ASSERT_TRUE(hidden_pair.ok()) << hidden_pair.status();
      ASSERT_TRUE(logits_gradient_pair.ok()) << logits_gradient_pair.status();

      BufferVec device_hidden = {hidden_pair->device};
      HostBufferVec reference_hidden = {hidden_pair->host};
      auto device_logits = (*device_head)->fwd(*executor_, device_hidden);

      auto reference_logits = (*reference_head)->fwd(reference_hidden);

      ASSERT_TRUE(device_logits.ok()) << device_logits.status();
      ASSERT_TRUE(reference_logits.ok()) << reference_logits.status();
      EXPECT_TRUE(FloatBuffersNear(device_logits->outputs[0],
                                   reference_logits->outputs[0], 4e-3f, 3e-3f));
      BufferVec device_logits_gradients = {logits_gradient_pair->device};
      HostBufferVec reference_logits_gradients = {logits_gradient_pair->host};
      for (int pass = 0; pass < 2; ++pass) {
        SCOPED_TRACE(testing::Message() << "backward pass=" << pass);
        // Copy the retained state so a second backward pass has the same input
        // while its table gradient accumulates on top of the first pass.
        auto device_hidden_gradient =
            (*device_head)
                ->bwd(*executor_, device_logits_gradients,
                      device_logits->state);
        auto reference_hidden_gradient =
            (*reference_head)
                ->bwd(reference_logits_gradients, reference_logits->state);
        ASSERT_TRUE(device_hidden_gradient.ok())
            << device_hidden_gradient.status();
        ASSERT_TRUE(reference_hidden_gradient.ok())
            << reference_hidden_gradient.status();
        ASSERT_EQ(device_hidden_gradient->size(), 1u);
        ASSERT_EQ(reference_hidden_gradient->size(), 1u);
        EXPECT_TRUE(FloatBuffersNear(device_hidden_gradient->front(),
                                     reference_hidden_gradient->front(), 4e-3f,
                                     3e-3f));
        EXPECT_TRUE(FloatBuffersNear(device_table_gradients[0],
                                     reference_table_gradients[0], 5e-3f,
                                     3e-3f));
      }
    }
  }
}

TEST_F(LayerReferenceTest, PositionEmbeddingForwardAndBackwardMatch) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (const auto [context, width, rows] :
         {std::tuple{4, 16, 8}, std::tuple{8, 32, 24}}) {
      auto device =
          PositionEmbeddingLayer::Create(*executor_, context, width, type);
      auto reference =
          PositionEmbeddingLayerReference::Create(context, width, type);
      ASSERT_TRUE(device.ok()) << device.status();
      ASSERT_TRUE(reference.ok()) << reference.status();
      std::vector<float> positions(static_cast<size_t>(context) * width);
      for (size_t index = 0; index < positions.size(); ++index)
        positions[index] = 0.1f * std::cos(static_cast<float>(index) * 0.2f);
      auto device_weights = (*device)->weights();
      auto reference_weights = (*reference)->weights();
      ASSERT_TRUE(SetFloatBufferPair(*executor_, device_weights[0],
                                     &reference_weights[0], positions)
                      .ok());
      std::vector<float> input(static_cast<size_t>(rows) * width);
      std::vector<float> gradient(input.size());
      for (size_t index = 0; index < input.size(); ++index) {
        input[index] = 0.3f * std::sin(static_cast<float>(index) * 0.09f);
        gradient[index] = 0.2f * std::cos(static_cast<float>(index) * 0.15f);
      }
      auto input_pair = MakeActivationBufferPair(*executor_, input, type);
      auto gradient_pair = MakeRawBufferPair<float>(*executor_, gradient);
      ASSERT_TRUE(input_pair.ok()) << input_pair.status();
      ASSERT_TRUE(gradient_pair.ok()) << gradient_pair.status();

      BufferVec device_inputs = {input_pair->device};
      HostBufferVec reference_inputs = {input_pair->host};
      auto device_output = (*device)->fwd(*executor_, device_inputs);

      auto reference_output = (*reference)->fwd(reference_inputs);

      ASSERT_TRUE(device_output.ok()) << device_output.status();
      ASSERT_TRUE(reference_output.ok()) << reference_output.status();
      EXPECT_TRUE(ActivationBuffersNear(
          device_output->outputs[0], reference_output->outputs[0], type, 0.0f));
      BufferVec device_gradients = {gradient_pair->device};
      HostBufferVec reference_gradients = {gradient_pair->host};
      auto device_input = (*device)->bwd(*executor_, device_gradients,
                                         std::move(device_output->state));
      auto reference_input =
          (*reference)
              ->bwd(reference_gradients, std::move(reference_output->state));
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
      EmbeddingLookupLayer::Create(*executor_, 17, 16, DataType::FP8);
  auto reference_lookup =
      EmbeddingLookupLayerReference::Create(17, 16, DataType::FP8);
  auto device_position =
      PositionEmbeddingLayer::Create(*executor_, 4, 16, DataType::FP8);
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

TEST_F(LayerReferenceTest, EmbeddingSignaturesPreserveSequenceAndStorageTypes) {
  for (DataType compute : {DataType::FP16, DataType::BF16}) {
    const DataType storage =
        compute == DataType::BF16 ? DataType::BF16 : DataType::FP32;
    auto device = EmbeddingLookupLayer::Create(*executor_, 17, 32, compute, 7);
    auto reference = EmbeddingLookupLayerReference::Create(17, 32, compute, 7);
    ASSERT_TRUE(device.ok()) << device.status();
    ASSERT_TRUE(reference.ok()) << reference.status();
    const ActivationType tokens(DataType::INT32, {-2, 7});
    const ActivationType activations(storage, {-2, 7, 32});
    const ActivationType logits(DataType::FP32, {-2, 7, 32});
    ASSERT_EQ((*device)->input_types().size(), 1);
    ASSERT_EQ((*device)->output_types().size(), 1);
    ASSERT_EQ((*reference)->input_types().size(), 1);
    ASSERT_EQ((*reference)->output_types().size(), 1);
    EXPECT_EQ((*device)->input_types()[0], tokens);
    EXPECT_EQ((*reference)->input_types()[0], tokens);
    EXPECT_EQ((*device)->output_types()[0], activations);
    EXPECT_EQ((*reference)->output_types()[0], activations);
    EXPECT_TRUE((*device)->ValidateSequenceLength(7).ok());
    EXPECT_FALSE((*device)->ValidateSequenceLength(1).ok());

    auto head = LanguageModelingHeadLayer::Create(device->get());
    auto reference_head =
        LanguageModelingHeadLayerReference::Create(reference->get());
    ASSERT_TRUE(head.ok()) << head.status();
    ASSERT_TRUE(reference_head.ok()) << reference_head.status();
    ASSERT_EQ((*head)->input_types().size(), 1);
    ASSERT_EQ((*head)->output_types().size(), 1);
    ASSERT_EQ((*reference_head)->input_types().size(), 1);
    ASSERT_EQ((*reference_head)->output_types().size(), 1);
    EXPECT_EQ((*head)->input_types()[0], activations);
    EXPECT_EQ((*reference_head)->input_types()[0], activations);
    // The logical vocabulary is 17, but the physical projection has 32 lanes.
    EXPECT_EQ((*head)->output_types()[0], logits);
    EXPECT_EQ((*reference_head)->output_types()[0], logits);
    EXPECT_TRUE((*head)->ValidateSequenceLength(7).ok());
    EXPECT_FALSE((*head)->ValidateSequenceLength(1).ok());

    auto position = PositionEmbeddingLayer::Create(*executor_, 7, 32, compute);
    auto reference_position =
        PositionEmbeddingLayerReference::Create(7, 32, compute);
    ASSERT_TRUE(position.ok()) << position.status();
    ASSERT_TRUE(reference_position.ok()) << reference_position.status();
    ASSERT_EQ((*position)->input_types().size(), 1);
    ASSERT_EQ((*position)->output_types().size(), 1);
    ASSERT_EQ((*reference_position)->input_types().size(), 1);
    ASSERT_EQ((*reference_position)->output_types().size(), 1);
    EXPECT_EQ((*position)->input_types()[0], activations);
    EXPECT_EQ((*position)->output_types()[0], activations);
    EXPECT_EQ((*reference_position)->input_types()[0], activations);
    EXPECT_EQ((*reference_position)->output_types()[0], activations);
  }
}

TEST_F(LayerReferenceTest, EmbeddingRejectsNonpositiveSequenceLength) {
  for (int length : {0, -1, -2}) {
    EXPECT_EQ(
        EmbeddingLookupLayer::Create(*executor_, 17, 32, DataType::FP16, length)
            .status()
            .code(),
        absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(
        EmbeddingLookupLayerReference::Create(17, 32, DataType::FP16, length)
            .status()
            .code(),
        absl::StatusCode::kInvalidArgument);
  }
}

}  // namespace
}  // namespace pluto::llm
