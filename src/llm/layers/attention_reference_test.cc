#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/llm/layer.h"
#include "src/llm/layers/attention.h"
#include "src/llm/layers/reference_test_util.h"

namespace pluto::llm {
namespace {

class AttentionReferenceTest : public LayerReferenceTest {
 protected:
  void CheckConfiguration(DataType type, int context, int heads, int width,
                          int sequences, float score_scale = 1.0f) {
    SCOPED_TRACE(testing::Message()
                 << "type=" << static_cast<int>(type) << " context=" << context
                 << " heads=" << heads << " width=" << width << " sequences="
                 << sequences << " score_scale=" << score_scale);
    auto device_layer =
        AttentionLayer::Create(*executor_, context, heads, width, type);
    auto reference_layer =
        AttentionLayerReference::Create(context, heads, width, type);
    ASSERT_TRUE(device_layer.ok()) << device_layer.status();
    ASSERT_TRUE(reference_layer.ok()) << reference_layer.status();
    const int rows = context * sequences;
    std::vector<float> qkv(static_cast<size_t>(rows) * 3 * width);
    std::vector<float> output_gradient(static_cast<size_t>(rows) * width);
    for (size_t index = 0; index < qkv.size(); ++index) {
      qkv[index] = 0.35f * std::sin(static_cast<float>(index) * 0.071f) +
                   0.05f * std::cos(static_cast<float>(index) * 0.19f);
      if (index % (3 * width) < static_cast<size_t>(2 * width))
        qkv[index] *= score_scale;
    }
    for (size_t index = 0; index < output_gradient.size(); ++index) {
      output_gradient[index] =
          0.2f * std::cos(static_cast<float>(index) * 0.13f);
    }
    auto input_pair = MakeActivationBufferPair(*executor_, qkv, type);
    auto gradient_pair = MakeRawBufferPair<float>(*executor_, output_gradient);
    ASSERT_TRUE(input_pair.ok()) << input_pair.status();
    ASSERT_TRUE(gradient_pair.ok()) << gradient_pair.status();

    BufferVec device_inputs = {input_pair->device};
    HostBufferVec reference_inputs = {input_pair->host};
    auto device_output = (*device_layer)->fwd(*executor_, device_inputs);
    auto reference_output = (*reference_layer)->fwd(reference_inputs);
    ASSERT_TRUE(device_output.ok()) << device_output.status();
    ASSERT_TRUE(reference_output.ok()) << reference_output.status();
    EXPECT_TRUE(ActivationBuffersNear(device_output->outputs[0],
                                      reference_output->outputs[0], type, 2e-3f,
                                      2e-3f));

    BufferVec device_gradients = {gradient_pair->device};
    HostBufferVec reference_gradients = {gradient_pair->host};
    auto device_input = (*device_layer)
                            ->bwd(*executor_, device_gradients,
                                  std::move(device_output->state));
    auto reference_input =
        (*reference_layer)
            ->bwd(reference_gradients, std::move(reference_output->state));
    ASSERT_TRUE(device_input.ok()) << device_input.status();
    ASSERT_TRUE(reference_input.ok()) << reference_input.status();
    ASSERT_EQ(device_input->size(), 1u);
    ASSERT_EQ(reference_input->size(), 1u);
    EXPECT_TRUE(FloatBuffersNear(device_input->front(),
                                 reference_input->front(), 3e-3f, 3e-3f));
  }
};

TEST_F(AttentionReferenceTest,
       CausalForwardAndBackwardMatchAcrossConfigurations) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (const auto [context, heads, width, sequences] :
         {std::tuple{1, 1, 16, 3}, std::tuple{4, 1, 16, 2},
          std::tuple{4, 2, 32, 2}, std::tuple{8, 2, 32, 1},
          std::tuple{33, 2, 64, 2}, std::tuple{33, 2, 128, 2},
          std::tuple{65, 2, 64, 2}, std::tuple{65, 2, 128, 2},
          std::tuple{65, 2, 160, 2}, std::tuple{65, 2, 192, 2},
          std::tuple{65, 2, 256, 2}}) {
      CheckConfiguration(type, context, heads, width, sequences);
    }
  }
}

TEST_F(AttentionReferenceTest, PeakedScoresMatchAcrossPartialTiles) {
  // Larger Q/K values give sharply peaked probabilities and exercise changing
  // softmax maxima across multiple key tiles. V and the upstream gradient keep
  // their original scales, so dQ/dK/dV remain useful numerical checks.
  for (DataType type : {DataType::FP16, DataType::BF16})
    CheckConfiguration(type, 65, 1, 64, 2, 20.0f);
}

TEST_F(AttentionReferenceTest, ProductionContextForwardAndBackwardMatch) {
  // One 64-wide head keeps the scalar O(context^2 * width) reference affordable
  // while exercising the full production-length softmax and gradient sums.
  for (DataType type : {DataType::FP16, DataType::BF16})
    CheckConfiguration(type, 1024, 1, 64, 1);
}

TEST_F(LayerReferenceTest, FutureTokensAndOtherSequencesCannotAffectGradients) {
  constexpr int kContext = 65;
  constexpr int kHeads = 2;
  constexpr int kWidth = 128;
  constexpr int kRows = 2 * kContext;
  constexpr int kVisibleQueries = 33;
  constexpr int kPackedWidth = 3 * kWidth;
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    // Check isolation in both directions: neither a previous sequence nor a
    // following sequence may participate in the selected sequence's attention.
    for (int sequence : {0, 1}) {
      SCOPED_TRACE(testing::Message() << "type=" << static_cast<int>(type)
                                      << " sequence=" << sequence);
      const int first_row = sequence * kContext;
      const int last_row = first_row + kVisibleQueries;
      std::vector<float> output_gradient(kRows * kWidth, 0.0f);
      for (int row = first_row; row < last_row; ++row) {
        for (int column = 0; column < kWidth; ++column) {
          const size_t index = static_cast<size_t>(row) * kWidth + column;
          output_gradient[index] =
              0.2f * std::cos(static_cast<float>(index) * 0.13f);
        }
      }
      auto gradients = MakeRawBufferPair<float>(*executor_, output_gradient);
      ASSERT_TRUE(gradients.ok()) << gradients.status();
      std::vector<float> expected_output;
      std::vector<float> expected_gradient;

      for (bool alter_invisible_tokens : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << "alter_invisible_tokens=" << alter_invisible_tokens);
        std::vector<float> qkv(kRows * kPackedWidth);
        for (size_t index = 0; index < qkv.size(); ++index) {
          const int row = index / kPackedWidth;
          qkv[index] = 0.35f * std::sin(static_cast<float>(index) * 0.071f) +
                       0.05f * std::cos(static_cast<float>(index) * 0.19f);
          if (alter_invisible_tokens && (row < first_row || row >= last_row))
            qkv[index] += 1.0f + 0.25f * (row % 5);
        }
        auto inputs = MakeActivationBufferPair(*executor_, qkv, type);
        auto attention =
            AttentionLayer::Create(*executor_, kContext, kHeads, kWidth, type);
        ASSERT_TRUE(inputs.ok()) << inputs.status();
        ASSERT_TRUE(attention.ok()) << attention.status();
        BufferVec device_inputs = {inputs->device};
        BufferVec device_gradients = {gradients->device};
        auto output = (*attention)->fwd(*executor_, device_inputs);
        ASSERT_TRUE(output.ok()) << output.status();
        auto input_gradient =
            (*attention)
                ->bwd(*executor_, device_gradients, std::move(output->state));
        ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();
        ASSERT_EQ(input_gradient->size(), 1u);
        auto host_output =
            ReadDeviceActivations(*executor_, output->outputs[0], type);
        auto host_gradient =
            ReadDeviceFloats(*executor_, input_gradient->front());
        ASSERT_TRUE(host_output.ok()) << host_output.status();
        ASSERT_TRUE(host_gradient.ok()) << host_gradient.status();

        float gradient_norms[3] = {};
        for (int row = 0; row < kRows; ++row) {
          for (int column = 0; column < kPackedWidth; ++column) {
            const size_t index =
                static_cast<size_t>(row) * kPackedWidth + column;
            if (row < first_row || row >= last_row) {
              ASSERT_EQ((*host_gradient)[index], 0.0f)
                  << "gradient reached invisible row=" << row
                  << " column=" << column;
            } else {
              gradient_norms[column / kWidth] +=
                  std::abs((*host_gradient)[index]);
              if (alter_invisible_tokens) {
                ASSERT_EQ((*host_gradient)[index], expected_gradient[index])
                    << "visible gradient changed at row=" << row
                    << " column=" << column;
              }
            }
          }
        }
        for (int region = 0; region < 3; ++region)
          EXPECT_GT(gradient_norms[region], 1e-4f) << "Q/K/V region=" << region;
        if (alter_invisible_tokens) {
          for (int row = first_row; row < last_row; ++row) {
            for (int column = 0; column < kWidth; ++column) {
              const size_t index = static_cast<size_t>(row) * kWidth + column;
              ASSERT_EQ((*host_output)[index], expected_output[index])
                  << "visible output changed at row=" << row
                  << " column=" << column;
            }
          }
        } else {
          expected_output.assign(host_output->data(),
                                 host_output->data() + host_output->size());
          expected_gradient.assign(
              host_gradient->data(),
              host_gradient->data() + host_gradient->size());
        }
      }
    }
  }
}

TEST_F(LayerReferenceTest, FP8IsRejectedConsistently) {
  auto device = AttentionLayer::Create(*executor_, 4, 2, 32, DataType::FP8);
  auto reference = AttentionLayerReference::Create(4, 2, 32, DataType::FP8);
  ASSERT_FALSE(device.ok());
  ASSERT_FALSE(reference.ok());
  EXPECT_EQ(device.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(reference.status().code(), absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace pluto::llm
