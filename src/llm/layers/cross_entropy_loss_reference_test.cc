#include <cmath>
#include <cstddef>
#include <limits>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/llm/layer.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/layers/reference_test_util.h"

namespace pluto::llm {
namespace {

TEST_F(LayerReferenceTest, StableForwardAndBackwardMatchForPaddedVocabularies) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (const auto [rows, vocab] :
         {std::tuple{3, 7}, std::tuple{16, 17}, std::tuple{32, 32}}) {
      SCOPED_TRACE(testing::Message()
                   << "type=" << static_cast<int>(type) << " rows=" << rows
                   << " vocab=" << vocab);
      auto device_layer = CrossEntropyLossLayer::Create(vocab, type, executor_);
      auto reference_layer =
          CrossEntropyLossLayerReference::Create(vocab, type);
      ASSERT_TRUE(device_layer.ok()) << device_layer.status();
      ASSERT_TRUE(reference_layer.ok()) << reference_layer.status();
      const int padded = (*device_layer)->padded_vocab_size();
      std::vector<float> logits(static_cast<size_t>(rows) * padded);
      std::vector<int> targets(rows);
      for (int row = 0; row < rows; ++row) {
        targets[row] = (row * 7 + 3) % vocab;
        for (int token = 0; token < padded; ++token) {
          logits[static_cast<size_t>(row) * padded + token] =
              token < vocab ? 2.5f * std::sin(row * 0.37f + token * 0.23f)
                            : -std::numeric_limits<float>::max();
        }
      }
      // Exercise stable log-sum-exp with a large common offset.
      if (rows > 3) {
        for (int token = 0; token < vocab; ++token) logits[token] += 80.0f;
      }
      auto logits_pair = MakeRawBufferPair<float>(logits, executor_);
      auto targets_pair = MakeRawBufferPair<int>(targets, executor_);
      ASSERT_TRUE(logits_pair.ok()) << logits_pair.status();
      ASSERT_TRUE(targets_pair.ok()) << targets_pair.status();
      Tape device_tape;
      ReferenceTape reference_tape;
      BufferVec device_inputs = {logits_pair->device, targets_pair->device};
      HostBufferVec reference_inputs = {logits_pair->host, targets_pair->host};
      auto device_losses =
          (*device_layer)->fwd(device_inputs, &device_tape, executor_);
      auto reference_losses =
          (*reference_layer)->fwd(reference_inputs, &reference_tape);
      ASSERT_TRUE(device_losses.ok()) << device_losses.status();
      ASSERT_TRUE(reference_losses.ok()) << reference_losses.status();
      EXPECT_TRUE(
          FloatBuffersNear(*device_losses, *reference_losses, 2e-5f, 2e-5f));

      BufferVec no_device_gradient;
      HostBufferVec no_reference_gradient;
      auto device_logits_gradient =
          (*device_layer)
              ->bwd(no_device_gradient, std::move(device_tape), executor_);
      auto reference_logits_gradient =
          (*reference_layer)
              ->bwd(no_reference_gradient, std::move(reference_tape));
      ASSERT_TRUE(device_logits_gradient.ok())
          << device_logits_gradient.status();
      ASSERT_TRUE(reference_logits_gradient.ok())
          << reference_logits_gradient.status();
      EXPECT_TRUE(FloatBuffersNear(device_logits_gradient->front(),
                                   reference_logits_gradient->front(), 2e-6f,
                                   2e-5f));
    }
  }
}

TEST_F(LayerReferenceTest, FP8IsRejectedConsistently) {
  auto device = CrossEntropyLossLayer::Create(17, DataType::FP8, executor_);
  auto reference = CrossEntropyLossLayerReference::Create(17, DataType::FP8);
  ASSERT_FALSE(device.ok());
  ASSERT_FALSE(reference.ok());
  EXPECT_EQ(device.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(reference.status().code(), absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace pluto::llm
