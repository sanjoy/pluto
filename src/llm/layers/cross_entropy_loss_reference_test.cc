#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/llm/layer.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/layers/reference_test_util.h"

namespace pluto::llm {
namespace {

TEST_F(LayerReferenceTest, CrossEntropyActivationTypesDescribePhysicalStorage) {
  constexpr int64_t kBatch = ActivationType::kBatchDimension;
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (int sequence_length : {1, 7}) {
      auto device =
          CrossEntropyLossLayer::Create(*executor_, 17, type, sequence_length);
      auto reference =
          CrossEntropyLossLayerReference::Create(17, type, sequence_length);
      ASSERT_TRUE(device.ok()) << device.status();
      ASSERT_TRUE(reference.ok()) << reference.status();
      const ActivationType logits(DataType::FP32,
                                  {kBatch, sequence_length, 32});
      const ActivationType targets(DataType::INT32, {kBatch, sequence_length});
      const ActivationType losses(DataType::FP32, {kBatch, sequence_length});
      ASSERT_EQ((*device)->input_types().size(), 2);
      ASSERT_EQ((*device)->output_types().size(), 1);
      ASSERT_EQ((*reference)->input_types().size(), 2);
      ASSERT_EQ((*reference)->output_types().size(), 1);
      EXPECT_EQ((*device)->input_types()[0], logits);
      EXPECT_EQ((*device)->input_types()[1], targets);
      EXPECT_EQ((*device)->output_types()[0], losses);
      EXPECT_EQ((*reference)->input_types()[0], logits);
      EXPECT_EQ((*reference)->input_types()[1], targets);
      EXPECT_EQ((*reference)->output_types()[0], losses);
    }
    for (int invalid_length : {0, -1, -2}) {
      EXPECT_EQ(
          CrossEntropyLossLayer::Create(*executor_, 17, type, invalid_length)
              .status()
              .code(),
          absl::StatusCode::kInvalidArgument);
      EXPECT_EQ(CrossEntropyLossLayerReference::Create(17, type, invalid_length)
                    .status()
                    .code(),
                absl::StatusCode::kInvalidArgument);
    }
  }
}

TEST_F(LayerReferenceTest, StableForwardAndBackwardMatchForPaddedVocabularies) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (const auto [rows, vocab] :
         {std::tuple{1, 1}, std::tuple{3, 7}, std::tuple{16, 17},
          std::tuple{32, 32}, std::tuple{3, 255}, std::tuple{3, 256},
          std::tuple{3, 257}}) {
      SCOPED_TRACE(testing::Message()
                   << "type=" << static_cast<int>(type) << " rows=" << rows
                   << " vocab=" << vocab);
      auto device_layer =
          CrossEntropyLossLayer::Create(*executor_, vocab, type);
      auto reference_layer =
          CrossEntropyLossLayerReference::Create(vocab, type);
      ASSERT_TRUE(device_layer.ok()) << device_layer.status();
      ASSERT_TRUE(reference_layer.ok()) << reference_layer.status();
      const int padded = (*device_layer)->padded_vocab_size();
      std::vector<float> logits(static_cast<size_t>(rows) * padded);
      std::vector<int> targets(rows);
      for (int row = 0; row < rows; ++row) {
        targets[row] = (row * 7 + 3) % vocab;
        for (int token = 0; token < padded; ++token)
          logits[static_cast<size_t>(row) * padded + token] =
              token < vocab ? 2.5f * std::sin(row * 0.37f + token * 0.23f)
                            : -std::numeric_limits<float>::max();
      }
      // Exercise stable log-sum-exp with a large common offset.
      if (rows > 3)
        for (int token = 0; token < vocab; ++token) logits[token] += 80.0f;
      auto logits_pair = MakeRawBufferPair<float>(*executor_, logits);
      auto targets_pair = MakeRawBufferPair<int>(*executor_, targets);
      ASSERT_TRUE(logits_pair.ok()) << logits_pair.status();
      ASSERT_TRUE(targets_pair.ok()) << targets_pair.status();

      BufferVec device_inputs = {logits_pair->device, targets_pair->device};
      HostBufferVec reference_inputs = {logits_pair->host, targets_pair->host};
      auto device_losses = (*device_layer)->fwd(*executor_, device_inputs);
      auto reference_losses = (*reference_layer)->fwd(reference_inputs);
      ASSERT_TRUE(device_losses.ok()) << device_losses.status();
      ASSERT_TRUE(reference_losses.ok()) << reference_losses.status();
      EXPECT_TRUE(FloatBuffersNear(device_losses->outputs[0],
                                   reference_losses->outputs[0], 2e-5f, 2e-5f));

      auto device_logits_gradient =
          (*device_layer)->bwd(*executor_, {}, std::move(device_losses->state));
      auto reference_logits_gradient =
          (*reference_layer)->bwd({}, std::move(reference_losses->state));
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

TEST_F(LayerReferenceTest,
       ProductionVocabularyMatchesReferenceAndRepeatsExactly) {
  constexpr int kVocabularySize = 50257;
  const std::vector<int> targets = {0, 255, 256, 257, kVocabularySize - 1};
  const int rows = targets.size();
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    SCOPED_TRACE(testing::Message() << "type=" << static_cast<int>(type));
    auto device_layer =
        CrossEntropyLossLayer::Create(*executor_, kVocabularySize, type);
    auto reference_layer =
        CrossEntropyLossLayerReference::Create(kVocabularySize, type);
    ASSERT_TRUE(device_layer.ok()) << device_layer.status();
    ASSERT_TRUE(reference_layer.ok()) << reference_layer.status();
    const int padded = (*device_layer)->padded_vocab_size();
    std::vector<float> logits(static_cast<size_t>(rows) * padded,
                              -std::numeric_limits<float>::max());
    for (int row = 0; row < rows; ++row)
      for (int token = 0; token < kVocabularySize; ++token)
        logits[static_cast<size_t>(row) * padded + token] =
            (row - 2) * 80.0f + 2.5f * std::sin(row * 0.37f + token * 0.23f);
    logits[static_cast<size_t>(2) * padded + targets[2]] += 12.0f;
    auto logits_pair = MakeRawBufferPair<float>(*executor_, logits);
    auto targets_pair = MakeRawBufferPair<int>(*executor_, targets);
    ASSERT_TRUE(logits_pair.ok()) << logits_pair.status();
    ASSERT_TRUE(targets_pair.ok()) << targets_pair.status();
    BufferVec device_inputs = {logits_pair->device, targets_pair->device};
    HostBufferVec reference_inputs = {logits_pair->host, targets_pair->host};
    auto reference_losses = (*reference_layer)->fwd(reference_inputs);
    ASSERT_TRUE(reference_losses.ok()) << reference_losses.status();
    auto reference_gradient =
        (*reference_layer)->bwd({}, std::move(reference_losses->state));
    ASSERT_TRUE(reference_gradient.ok()) << reference_gradient.status();

    auto device_losses = (*device_layer)->fwd(*executor_, device_inputs);
    ASSERT_TRUE(device_losses.ok()) << device_losses.status();
    auto device_gradient =
        (*device_layer)->bwd(*executor_, {}, device_losses->state);
    ASSERT_TRUE(device_gradient.ok()) << device_gradient.status();
    EXPECT_TRUE(FloatBuffersNear(device_losses->outputs[0],
                                 reference_losses->outputs[0], 1e-4f, 1e-4f));
    EXPECT_TRUE(FloatBuffersNear(device_gradient->front(),
                                 reference_gradient->front(), 1e-10f, 2e-4f));
    // The scalar FP32 reference sums 50k terms sequentially, which loses
    // precision for a peaked distribution. Check a double-precision CPU
    // oracle as well so the larger reference tolerance cannot hide drift.
    std::vector<float> expected_losses(rows);
    std::vector<float> expected_gradient(logits.size(), 0.0f);
    for (int row = 0; row < rows; ++row) {
      const size_t offset = static_cast<size_t>(row) * padded;
      double maximum = -std::numeric_limits<double>::infinity();
      for (int token = 0; token < kVocabularySize; ++token)
        maximum =
            std::max(maximum, static_cast<double>(logits[offset + token]));
      double denominator = 0.0;
      for (int token = 0; token < kVocabularySize; ++token)
        denominator += std::exp(logits[offset + token] - maximum);
      expected_losses[row] =
          std::log(denominator) + maximum - logits[offset + targets[row]];
      for (int token = 0; token < kVocabularySize; ++token)
        expected_gradient[offset + token] =
            (std::exp(logits[offset + token] - maximum) / denominator -
             (token == targets[row] ? 1.0 : 0.0)) /
            rows;
    }
    auto first_losses = ReadDeviceFloats(*executor_, device_losses->outputs[0]);
    auto first_gradient =
        ReadDeviceFloats(*executor_, device_gradient->front());
    ASSERT_TRUE(first_losses.ok()) << first_losses.status();
    ASSERT_TRUE(first_gradient.ok()) << first_gradient.status();
    EXPECT_TRUE(
        VectorsNear(first_losses->span(), expected_losses, 3e-5f, 3e-5f));
    EXPECT_TRUE(
        VectorsNear(first_gradient->span(), expected_gradient, 1e-10f, 3e-5f));
    for (int row = 0; row < rows; ++row)
      for (int token = kVocabularySize; token < padded; ++token)
        EXPECT_EQ((*first_gradient)[static_cast<size_t>(row) * padded + token],
                  0.0f);

    for (int repeat = 0; repeat < 3; ++repeat) {
      SCOPED_TRACE(testing::Message() << "repeat=" << repeat);
      auto repeated_losses = (*device_layer)->fwd(*executor_, device_inputs);
      ASSERT_TRUE(repeated_losses.ok()) << repeated_losses.status();
      auto repeated_gradient =
          (*device_layer)
              ->bwd(*executor_, {}, std::move(repeated_losses->state));
      ASSERT_TRUE(repeated_gradient.ok()) << repeated_gradient.status();
      auto host_losses =
          ReadDeviceFloats(*executor_, repeated_losses->outputs[0]);
      auto host_gradient =
          ReadDeviceFloats(*executor_, repeated_gradient->front());
      ASSERT_TRUE(host_losses.ok()) << host_losses.status();
      ASSERT_TRUE(host_gradient.ok()) << host_gradient.status();
      EXPECT_EQ(std::memcmp(first_losses->data(), host_losses->data(),
                            rows * sizeof(float)),
                0);
      EXPECT_EQ(std::memcmp(first_gradient->data(), host_gradient->data(),
                            logits.size() * sizeof(float)),
                0);
    }
    // Retained statistics must survive a forward pass on different logits.
    std::vector<float> other_logits = logits;
    for (int row = 0; row < rows; ++row)
      for (int token = 0; token < kVocabularySize; ++token)
        other_logits[static_cast<size_t>(row) * padded + token] =
            (token % 17) * 0.3f;
    auto other_pair = MakeRawBufferPair<float>(*executor_, other_logits);
    ASSERT_TRUE(other_pair.ok()) << other_pair.status();
    BufferVec other_inputs = {other_pair->device, targets_pair->device};
    auto other_losses = (*device_layer)->fwd(*executor_, other_inputs);
    ASSERT_TRUE(other_losses.ok()) << other_losses.status();
    auto saved_gradient =
        (*device_layer)->bwd(*executor_, {}, std::move(device_losses->state));
    ASSERT_TRUE(saved_gradient.ok()) << saved_gradient.status();
    auto host_saved_gradient =
        ReadDeviceFloats(*executor_, saved_gradient->front());
    ASSERT_TRUE(host_saved_gradient.ok()) << host_saved_gradient.status();
    EXPECT_EQ(std::memcmp(first_gradient->data(), host_saved_gradient->data(),
                          logits.size() * sizeof(float)),
              0);
  }
}

TEST_F(LayerReferenceTest, IgnoredPromptAndPaddingDoNotDiluteMeanGradient) {
  constexpr int kVocabularySize = 17;
  constexpr int kIgnored = CrossEntropyLossLayer::kIgnoredTarget;
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    // Exercise partial counting tiles, an exact counting tile, and the real
    // two-sequence/1024-token layout. Most of each sequence is ignored padding.
    for (int rows : {7, 256, 257, 2048}) {
      SCOPED_TRACE(testing::Message()
                   << "type=" << static_cast<int>(type) << " rows=" << rows);
      const int sequence_length = rows == 2048 ? 1024 : 1;
      auto device_layer = CrossEntropyLossLayer::Create(
          *executor_, kVocabularySize, type, sequence_length);
      auto compact_layer =
          CrossEntropyLossLayer::Create(*executor_, kVocabularySize, type);
      auto reference_layer = CrossEntropyLossLayerReference::Create(
          kVocabularySize, type, sequence_length);
      ASSERT_TRUE(device_layer.ok()) << device_layer.status();
      ASSERT_TRUE(compact_layer.ok()) << compact_layer.status();
      ASSERT_TRUE(reference_layer.ok()) << reference_layer.status();
      const int padded = (*device_layer)->padded_vocab_size();
      std::vector<float> logits(static_cast<size_t>(rows) * padded,
                                std::numeric_limits<float>::quiet_NaN());
      std::vector<int> targets(rows, kIgnored);
      std::vector<float> compact_logits;
      std::vector<int> compact_targets;
      for (int row = 0; row < rows; ++row) {
        const int position = row % 1024;
        // Positions 0..3 are prompt predictions, 4..11 are supervised
        // continuation predictions, and the remaining positions are padding.
        if (position < 4 || position >= 12) continue;
        targets[row] = (row * 7 + 1) % kVocabularySize;
        compact_targets.push_back(targets[row]);
        for (int token = 0; token < padded; ++token) {
          const float value = token < kVocabularySize
                                  ? std::sin(row * 0.3f + token * 0.7f)
                                  : -std::numeric_limits<float>::max();
          logits[static_cast<size_t>(row) * padded + token] = value;
          compact_logits.push_back(value);
        }
      }
      auto logits_pair = MakeRawBufferPair<float>(*executor_, logits);
      auto targets_pair = MakeRawBufferPair<int>(*executor_, targets);
      auto compact_logits_pair =
          MakeRawBufferPair<float>(*executor_, compact_logits);
      auto compact_targets_pair =
          MakeRawBufferPair<int>(*executor_, compact_targets);
      ASSERT_TRUE(logits_pair.ok()) << logits_pair.status();
      ASSERT_TRUE(targets_pair.ok()) << targets_pair.status();
      ASSERT_TRUE(compact_logits_pair.ok()) << compact_logits_pair.status();
      ASSERT_TRUE(compact_targets_pair.ok()) << compact_targets_pair.status();
      BufferVec inputs = {logits_pair->device, targets_pair->device};
      auto forward = (*device_layer)->fwd(*executor_, inputs);
      auto reference =
          (*reference_layer)->fwd({logits_pair->host, targets_pair->host});
      auto compact = (*compact_layer)
                         ->fwd(*executor_, {compact_logits_pair->device,
                                            compact_targets_pair->device});
      ASSERT_TRUE(forward.ok()) << forward.status();
      ASSERT_TRUE(reference.ok()) << reference.status();
      ASSERT_TRUE(compact.ok()) << compact.status();
      EXPECT_TRUE(FloatBuffersNear(forward->outputs[0], reference->outputs[0],
                                   2e-6f, 2e-6f));
      auto gradient = (*device_layer)->bwd(*executor_, {}, forward->state);
      auto reference_gradient =
          (*reference_layer)->bwd({}, std::move(reference->state));
      auto compact_gradient =
          (*compact_layer)->bwd(*executor_, {}, std::move(compact->state));
      ASSERT_TRUE(gradient.ok()) << gradient.status();
      ASSERT_TRUE(reference_gradient.ok()) << reference_gradient.status();
      ASSERT_TRUE(compact_gradient.ok()) << compact_gradient.status();
      EXPECT_TRUE(FloatBuffersNear(gradient->front(),
                                   reference_gradient->front(), 2e-7f, 2e-6f));
      auto host_losses = ReadDeviceFloats(*executor_, forward->outputs[0]);
      auto host_gradient = ReadDeviceFloats(*executor_, gradient->front());
      auto compact_losses = ReadDeviceFloats(*executor_, compact->outputs[0]);
      auto compact_derivative =
          ReadDeviceFloats(*executor_, compact_gradient->front());
      ASSERT_TRUE(host_losses.ok()) << host_losses.status();
      ASSERT_TRUE(host_gradient.ok()) << host_gradient.status();
      ASSERT_TRUE(compact_losses.ok()) << compact_losses.status();
      ASSERT_TRUE(compact_derivative.ok()) << compact_derivative.status();
      int compact_row = 0;
      for (int row = 0; row < rows; ++row) {
        const float* row_gradient =
            host_gradient->data() + static_cast<size_t>(row) * padded;
        if (targets[row] == kIgnored) {
          EXPECT_EQ((*host_losses)[row], 0.0f);
          for (int token = 0; token < padded; ++token)
            EXPECT_EQ(row_gradient[token], 0.0f);
        } else {
          EXPECT_EQ((*host_losses)[row], (*compact_losses)[compact_row]);
          EXPECT_EQ(std::memcmp(row_gradient,
                                compact_derivative->data() +
                                    static_cast<size_t>(compact_row) * padded,
                                padded * sizeof(float)),
                    0);
          ++compact_row;
        }
      }
      // Integer target counting and backward remain byte-for-byte repeatable.
      for (int repeat = 0; repeat < 3; ++repeat) {
        auto repeated = (*device_layer)->bwd(*executor_, {}, forward->state);
        ASSERT_TRUE(repeated.ok()) << repeated.status();
        auto host_repeated = ReadDeviceFloats(*executor_, repeated->front());
        ASSERT_TRUE(host_repeated.ok()) << host_repeated.status();
        EXPECT_EQ(std::memcmp(host_gradient->data(), host_repeated->data(),
                              logits.size() * sizeof(float)),
                  0);
      }
    }
  }
}

TEST_F(LayerReferenceTest, AllIgnoredRowsHaveZeroLossAndGradientEvenWithNaNs) {
  constexpr int kRows = 257;
  constexpr int kPaddedVocabulary = 32;
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    auto device_layer = CrossEntropyLossLayer::Create(*executor_, 17, type);
    auto reference_layer = CrossEntropyLossLayerReference::Create(17, type);
    ASSERT_TRUE(device_layer.ok()) << device_layer.status();
    ASSERT_TRUE(reference_layer.ok()) << reference_layer.status();
    auto logits = MakeRawBufferPair<float>(
        *executor_,
        std::vector<float>(kRows * kPaddedVocabulary,
                           std::numeric_limits<float>::quiet_NaN()));
    auto targets = MakeRawBufferPair<int>(
        *executor_,
        std::vector<int>(kRows, CrossEntropyLossLayer::kIgnoredTarget));
    ASSERT_TRUE(logits.ok()) << logits.status();
    ASSERT_TRUE(targets.ok()) << targets.status();
    auto forward =
        (*device_layer)->fwd(*executor_, {logits->device, targets->device});
    auto reference = (*reference_layer)->fwd({logits->host, targets->host});
    ASSERT_TRUE(forward.ok()) << forward.status();
    ASSERT_TRUE(reference.ok()) << reference.status();
    auto gradient =
        (*device_layer)->bwd(*executor_, {}, std::move(forward->state));
    auto reference_gradient =
        (*reference_layer)->bwd({}, std::move(reference->state));
    ASSERT_TRUE(gradient.ok()) << gradient.status();
    ASSERT_TRUE(reference_gradient.ok()) << reference_gradient.status();
    auto host_losses = ReadDeviceFloats(*executor_, forward->outputs[0]);
    auto host_gradient = ReadDeviceFloats(*executor_, gradient->front());
    ASSERT_TRUE(host_losses.ok()) << host_losses.status();
    ASSERT_TRUE(host_gradient.ok()) << host_gradient.status();
    for (float loss : host_losses->span()) EXPECT_EQ(loss, 0.0f);
    for (float derivative : host_gradient->span()) EXPECT_EQ(derivative, 0.0f);
    EXPECT_TRUE(
        FloatBuffersNear(forward->outputs[0], reference->outputs[0], 0));
    EXPECT_TRUE(
        FloatBuffersNear(gradient->front(), reference_gradient->front(), 0));
  }
}

TEST_F(LayerReferenceTest, MaskedMeanGradientMatchesFiniteDifferences) {
  constexpr int kRows = 4;
  constexpr int kVocabularySize = 3;
  constexpr int kPaddedVocabulary = 16;
  const std::vector<int> targets = {CrossEntropyLossLayer::kIgnoredTarget, 0, 2,
                                    CrossEntropyLossLayer::kIgnoredTarget};
  std::vector<float> logits(kRows * kPaddedVocabulary,
                            -std::numeric_limits<float>::max());
  for (int row = 0; row < kRows; ++row)
    for (int token = 0; token < kVocabularySize; ++token)
      logits[row * kPaddedVocabulary + token] =
          std::cos(row * 0.6f + token * 0.7f);
  auto logits_pair = MakeRawBufferPair<float>(*executor_, logits);
  auto targets_pair = MakeRawBufferPair<int>(*executor_, targets);
  ASSERT_TRUE(logits_pair.ok()) << logits_pair.status();
  ASSERT_TRUE(targets_pair.ok()) << targets_pair.status();

  // Independent double-precision scalar objective. Perturb one logit without
  // changing the target mask, and average the two supervised rows only.
  auto objective = [&](int coordinate, double delta) {
    double loss = 0;
    for (int row = 0; row < kRows; ++row) {
      if (targets[row] == CrossEntropyLossLayer::kIgnoredTarget) continue;
      double denominator = 0;
      double target_logit = 0;
      for (int token = 0; token < kVocabularySize; ++token) {
        const int index = row * kPaddedVocabulary + token;
        const double value = logits[index] + (index == coordinate ? delta : 0);
        denominator += std::exp(value);
        if (token == targets[row]) target_logit = value;
      }
      loss += std::log(denominator) - target_logit;
    }
    return loss / 2;
  };
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    auto device_layer =
        CrossEntropyLossLayer::Create(*executor_, kVocabularySize, type);
    auto reference_layer =
        CrossEntropyLossLayerReference::Create(kVocabularySize, type);
    ASSERT_TRUE(device_layer.ok()) << device_layer.status();
    ASSERT_TRUE(reference_layer.ok()) << reference_layer.status();
    auto forward =
        (*device_layer)
            ->fwd(*executor_, {logits_pair->device, targets_pair->device});
    auto reference =
        (*reference_layer)->fwd({logits_pair->host, targets_pair->host});
    ASSERT_TRUE(forward.ok()) << forward.status();
    ASSERT_TRUE(reference.ok()) << reference.status();
    auto gradient =
        (*device_layer)->bwd(*executor_, {}, std::move(forward->state));
    auto reference_gradient =
        (*reference_layer)->bwd({}, std::move(reference->state));
    ASSERT_TRUE(gradient.ok()) << gradient.status();
    ASSERT_TRUE(reference_gradient.ok()) << reference_gradient.status();
    auto host_gradient = ReadDeviceFloats(*executor_, gradient->front());
    ASSERT_TRUE(host_gradient.ok()) << host_gradient.status();
    const auto reference_values = ReadHostFloats(reference_gradient->front());
    constexpr double kEpsilon = 1e-4;
    for (int row = 0; row < kRows; ++row)
      for (int token = 0; token < kVocabularySize; ++token) {
        const int coordinate = row * kPaddedVocabulary + token;
        const double numerical = (objective(coordinate, kEpsilon) -
                                  objective(coordinate, -kEpsilon)) /
                                 (2 * kEpsilon);
        EXPECT_NEAR((*host_gradient)[coordinate], numerical, 1e-6);
        EXPECT_NEAR(reference_values[coordinate], numerical, 1e-6);
      }
  }
}

TEST_F(LayerReferenceTest, ReferenceAcceptsOnlyMinusOneAsIgnoredTarget) {
  auto layer = CrossEntropyLossLayerReference::Create(17, DataType::BF16);
  auto logits = MakeRawBufferPair<float>(*executor_, std::vector<float>(32, 0));
  ASSERT_TRUE(layer.ok()) << layer.status();
  ASSERT_TRUE(logits.ok()) << logits.status();
  for (int target : {-2, 17}) {
    auto targets = MakeRawBufferPair<int>(*executor_, std::vector<int>{target});
    ASSERT_TRUE(targets.ok()) << targets.status();
    auto result = (*layer)->fwd({logits->host, targets->host});
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  }
}

TEST_F(LayerReferenceTest, BackwardRejectsMissingOrInvalidSavedStatistics) {
  constexpr int kRows = 3;
  auto layer = CrossEntropyLossLayer::Create(*executor_, 32, DataType::FP16);
  ASSERT_TRUE(layer.ok()) << layer.status();
  auto logits = MakeRawBufferPair<float>(*executor_,
                                         std::vector<float>(kRows * 32, 0.0f));
  auto targets = MakeRawBufferPair<int>(*executor_, std::vector<int>(kRows, 0));
  ASSERT_TRUE(logits.ok()) << logits.status();
  ASSERT_TRUE(targets.ok()) << targets.status();
  BufferVec inputs = {logits->device, targets->device};
  auto losses = (*layer)->fwd(*executor_, inputs);
  ASSERT_TRUE(losses.ok()) << losses.status();
  ASSERT_EQ(losses->state.intermediates.size(), 4u);
  for (int saved_buffers : {2, 3}) {
    BackwardState invalid = losses->state;
    invalid.intermediates.erase(invalid.intermediates.begin() + saved_buffers,
                                invalid.intermediates.end());
    auto result = (*layer)->bwd(*executor_, {}, std::move(invalid));
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  }
  for (int statistic : {2, 3}) {
    BackwardState invalid = losses->state;
    invalid.intermediates[statistic] = logits->device;
    auto result = (*layer)->bwd(*executor_, {}, std::move(invalid));
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  }
  auto other_executor = cuda::Executor::Create();
  ASSERT_TRUE(other_executor.ok()) << other_executor.status();
  auto foreign_statistic =
      Buffer::Allocate(**other_executor, kRows * sizeof(float));
  ASSERT_TRUE(foreign_statistic.ok()) << foreign_statistic.status();
  for (int statistic : {2, 3}) {
    BackwardState invalid = losses->state;
    invalid.intermediates[statistic] = *foreign_statistic;
    auto result = (*layer)->bwd(*executor_, {}, std::move(invalid));
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  }
  auto unexpected_gradient =
      (*layer)->bwd(*executor_, losses->outputs, losses->state);
  ASSERT_FALSE(unexpected_gradient.ok());
  EXPECT_EQ(unexpected_gradient.status().code(),
            absl::StatusCode::kInvalidArgument);
  auto valid = (*layer)->bwd(*executor_, {}, std::move(losses->state));
  ASSERT_TRUE(valid.ok()) << valid.status();
}

TEST_F(LayerReferenceTest, FP8IsRejectedConsistently) {
  auto device = CrossEntropyLossLayer::Create(*executor_, 17, DataType::FP8);
  auto reference = CrossEntropyLossLayerReference::Create(17, DataType::FP8);
  ASSERT_FALSE(device.ok());
  ASSERT_FALSE(reference.ok());
  EXPECT_EQ(device.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(reference.status().code(), absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace pluto::llm
