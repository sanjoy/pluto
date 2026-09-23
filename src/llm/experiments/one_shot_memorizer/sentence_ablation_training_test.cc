#include "src/llm/experiments/one_shot_memorizer/sentence_ablation_training.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layers/cross_entropy_loss.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

class SentenceAblationTrainingTest : public testing::Test {
 protected:
  void SetUp() override {
    auto created = cuda::Executor::Create();
    ASSERT_TRUE(created.ok()) << created.status();
    executor_ = std::move(*created);
  }

  void TearDown() override {
    if (!executor_)
      return;
    EXPECT_TRUE(executor_->Synchronize().ok());
  }

  void VerifyMask(const std::vector<size_t>& sample_indices, size_t omitted,
                  int sequence, int vocabulary, int expected_count) {
    const size_t sample_values = static_cast<size_t>(sequence) * vocabulary;
    const size_t values = sample_values * sample_indices.size();
    auto input =
        cuda::PageLockedHostArray<uint32_t>::Allocate(*executor_, values);
    auto output =
        cuda::PageLockedHostArray<uint32_t>::Allocate(*executor_, values);
    auto gradient = cuda::Buffer::Allocate(*executor_, values * sizeof(float));
    ASSERT_TRUE(input.ok()) << input.status();
    ASSERT_TRUE(output.ok()) << output.status();
    ASSERT_TRUE(gradient.ok()) << gradient.status();
    // Include sign-bit zero, subnormal and NaN payloads to test literal byte
    // preservation, not merely approximate equality of surviving values.
    constexpr uint32_t kBits[] = {0x3f800000, 0xbf800000, 0x80000000,
                                  0x00000001, 0x7fc00042, 0x3eabcdef};
    for (size_t index = 0; index < values; ++index)
      (*input)[index] = kBits[index % 6];
    ASSERT_EQ(
        cudaMemcpyAsync(gradient->data(), input->data(), input->size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        cudaSuccess);
    auto masked = ZeroSentenceContribution(
        *executor_, *gradient, sample_indices, omitted, sequence, vocabulary);
    ASSERT_TRUE(masked.ok()) << masked.status();
    EXPECT_EQ(*masked, expected_count);
    // No wait separates H2D, masking and D2H: masking must use this same stream
    // so both its producer and consumer see the intended execution ordering.
    ASSERT_EQ(
        cudaMemcpyAsync(output->data(), gradient->data(), output->size_bytes(),
                        cudaMemcpyDeviceToHost, executor_->stream()),
        cudaSuccess);
    ASSERT_TRUE(executor_->Synchronize().ok());
    for (size_t sample = 0; sample < sample_indices.size(); ++sample)
      for (size_t value = 0; value < sample_values; ++value) {
        const size_t index = sample * sample_values + value;
        const uint32_t expected =
            sample_indices[sample] == omitted ? uint32_t{0} : (*input)[index];
        EXPECT_EQ((*output)[index], expected)
            << "sample=" << sample << " value=" << value;
      }
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(SentenceAblationTrainingTest,
       PreservesEverySurvivingBitOnTheSameStream) {
  VerifyMask({10, 11, 12}, 11, 3, 5, 1);
}

TEST_F(SentenceAblationTrainingTest,
       ZerosAllRepeatedOccurrencesIncludingEdges) {
  VerifyMask({11, 12, 11, 11}, 11, 2, 7, 3);
}

TEST_F(SentenceAblationTrainingTest, AbsentSentenceDoesNotModifyAnyByte) {
  VerifyMask({1, 2, 3}, 0, 2, 9, 0);
}

TEST_F(SentenceAblationTrainingTest, EmptyBatchNeedsAnEmptyOwnedBuffer) {
  auto empty = cuda::Buffer::Allocate(*executor_, 0);
  ASSERT_TRUE(empty.ok()) << empty.status();
  auto result = ZeroSentenceContribution(*executor_, *empty, {}, 9, 2, 3);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(*result, 0);
}

TEST_F(SentenceAblationTrainingTest,
       MaskingAfterCrossEntropyRetainsBatchScaling) {
  constexpr int kVocabulary = 4, kRows = 2;
  auto loss =
      CrossEntropyLossLayer::Create(*executor_, kVocabulary, DataType::BF16);
  ASSERT_TRUE(loss.ok()) << loss.status();
  const int physical_width = (*loss)->padded_vocab_size();
  const size_t values = static_cast<size_t>(kRows) * physical_width;
  auto host_logits =
      cuda::PageLockedHostArray<float>::Allocate(*executor_, values);
  auto targets = cuda::PageLockedHostArray<int>::CopyFrom(
      *executor_, std::vector<int>{0, 1});
  auto ignored_targets = cuda::PageLockedHostArray<int>::CopyFrom(
      *executor_, std::vector<int>{-1, 1});
  auto before = cuda::PageLockedHostArray<float>::Allocate(*executor_, values);
  auto after = cuda::PageLockedHostArray<float>::Allocate(*executor_, values);
  auto ignored = cuda::PageLockedHostArray<float>::Allocate(*executor_, values);
  auto logits = cuda::Buffer::Allocate(*executor_, values * sizeof(float));
  auto target_buffer = cuda::Buffer::Allocate(*executor_, kRows * sizeof(int));
  ASSERT_TRUE(host_logits.ok()) << host_logits.status();
  ASSERT_TRUE(targets.ok()) << targets.status();
  ASSERT_TRUE(ignored_targets.ok()) << ignored_targets.status();
  ASSERT_TRUE(before.ok()) << before.status();
  ASSERT_TRUE(after.ok()) << after.status();
  ASSERT_TRUE(ignored.ok()) << ignored.status();
  ASSERT_TRUE(logits.ok()) << logits.status();
  ASSERT_TRUE(target_buffer.ok()) << target_buffer.status();
  for (int row = 0; row < kRows; ++row)
    for (int token = 0; token < physical_width; ++token)
      (*host_logits)[row * physical_width + token] =
          token < kVocabulary ? 0.0f : -std::numeric_limits<float>::max();
  ASSERT_EQ(
      cudaMemcpyAsync(logits->data(), host_logits->data(), logits->size_bytes(),
                      cudaMemcpyHostToDevice, executor_->stream()),
      cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(target_buffer->data(), targets->data(),
                            targets->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);
  auto fwd = (*loss)->fwd(*executor_, BufferVec{*logits, *target_buffer});
  ASSERT_TRUE(fwd.ok()) << fwd.status();
  auto gradient = (*loss)->bwd(*executor_, {}, std::move(fwd->state));
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  ASSERT_EQ(gradient->size(), 1u);
  ASSERT_EQ(cudaMemcpyAsync(before->data(), gradient->front().data(),
                            before->size_bytes(), cudaMemcpyDeviceToHost,
                            executor_->stream()),
            cudaSuccess);
  auto masked =
      ZeroSentenceContribution(*executor_, gradient->front(),
                               std::vector<size_t>{8, 9}, 8, 1, physical_width);
  ASSERT_TRUE(masked.ok()) << masked.status();
  ASSERT_EQ(*masked, 1);
  ASSERT_EQ(cudaMemcpyAsync(after->data(), gradient->front().data(),
                            after->size_bytes(), cudaMemcpyDeviceToHost,
                            executor_->stream()),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(target_buffer->data(), ignored_targets->data(),
                            ignored_targets->size_bytes(),
                            cudaMemcpyHostToDevice, executor_->stream()),
            cudaSuccess);
  auto ignored_fwd =
      (*loss)->fwd(*executor_, BufferVec{*logits, *target_buffer});
  ASSERT_TRUE(ignored_fwd.ok()) << ignored_fwd.status();
  auto ignored_gradient =
      (*loss)->bwd(*executor_, {}, std::move(ignored_fwd->state));
  ASSERT_TRUE(ignored_gradient.ok()) << ignored_gradient.status();
  ASSERT_EQ(cudaMemcpyAsync(ignored->data(), ignored_gradient->front().data(),
                            ignored->size_bytes(), cudaMemcpyDeviceToHost,
                            executor_->stream()),
            cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());
  for (int token = 0; token < physical_width; ++token) {
    EXPECT_EQ((*after)[token], 0.0f);
    const size_t survivor = physical_width + token;
    EXPECT_EQ((*before)[survivor], (*after)[survivor]);
    EXPECT_FLOAT_EQ((*ignored)[survivor], 2 * (*after)[survivor]);
  }
  EXPECT_FLOAT_EQ((*after)[physical_width], 0.125f);
  EXPECT_FLOAT_EQ((*after)[physical_width + 1], -0.375f);
}

TEST_F(SentenceAblationTrainingTest, RejectsInvalidArgumentsBeforeWriting) {
  constexpr size_t kBytes = 4 * sizeof(float);
  auto gradient = cuda::Buffer::Allocate(*executor_, kBytes);
  auto output = cuda::PageLockedHostArray<uint32_t>::Allocate(*executor_, 4);
  auto other_executor = cuda::Executor::Create();
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  ASSERT_TRUE(output.ok()) << output.status();
  ASSERT_TRUE(other_executor.ok()) << other_executor.status();
  ASSERT_EQ(
      cudaMemsetAsync(gradient->data(), 0x42, kBytes, executor_->stream()),
      cudaSuccess);
  const std::vector<size_t> one{7}, two{7, 8};
  EXPECT_FALSE(
      ZeroSentenceContribution(**other_executor, *gradient, one, 7, 2, 2).ok());
  EXPECT_FALSE(
      ZeroSentenceContribution(*executor_, *gradient, one, 7, 0, 4).ok());
  EXPECT_FALSE(
      ZeroSentenceContribution(*executor_, *gradient, one, 7, 1, -4).ok());
  EXPECT_FALSE(
      ZeroSentenceContribution(*executor_, *gradient, one, 7, 1, 3).ok());
  EXPECT_FALSE(
      ZeroSentenceContribution(*executor_, *gradient, two, 7, 2, 2).ok());
  EXPECT_FALSE(
      ZeroSentenceContribution(*executor_, *gradient, {}, 7, 2, 2).ok());
  auto overflow = ZeroSentenceContribution(*executor_, *gradient, two, 7,
                                           std::numeric_limits<int>::max(),
                                           std::numeric_limits<int>::max());
  EXPECT_TRUE(absl::IsOutOfRange(overflow.status())) << overflow.status();
  ASSERT_EQ(cudaMemcpyAsync(output->data(), gradient->data(), kBytes,
                            cudaMemcpyDeviceToHost, executor_->stream()),
            cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());
  for (uint32_t value : *output)
    EXPECT_EQ(value, uint32_t{0x42424242});
}

TEST(AblationLearningRateTest, UsesOriginalHorizonRatherThanPilotLength) {
  constexpr double kPeak = 0.001;
  auto first = AblationLearningRate(1, kPeak, 100, 40000);
  auto peak = AblationLearningRate(100, kPeak, 100, 40000);
  auto pilot_end = AblationLearningRate(512, kPeak, 100, 40000);
  auto midpoint = AblationLearningRate(20050, kPeak, 100, 40000);
  auto floor = AblationLearningRate(40000, kPeak, 100, 40000);
  auto after =
      AblationLearningRate(std::numeric_limits<int>::max(), kPeak, 100, 40000);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(peak.ok()) << peak.status();
  ASSERT_TRUE(pilot_end.ok()) << pilot_end.status();
  ASSERT_TRUE(midpoint.ok()) << midpoint.status();
  ASSERT_TRUE(floor.ok()) << floor.status();
  ASSERT_TRUE(after.ok()) << after.status();
  EXPECT_FLOAT_EQ(*first, 0.00001f);
  EXPECT_FLOAT_EQ(*peak, 0.001f);
  EXPECT_FLOAT_EQ(*midpoint, 0.00055f);
  EXPECT_FLOAT_EQ(*floor, 0.0001f);
  EXPECT_EQ(*floor, *after);
  const double expected_pilot =
      kPeak *
      (0.1 + 0.9 * (1 + std::cos(3.14159265358979323846 * 412 / 39900)) / 2);
  EXPECT_FLOAT_EQ(*pilot_end, static_cast<float>(expected_pilot));
  EXPECT_GT(*pilot_end, 0.99f * *peak);
}

TEST(AblationLearningRateTest, ZeroWarmupMatchesTheOriginalCosineFormula) {
  auto result = AblationLearningRate(1, 1, 0, 2);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_FLOAT_EQ(*result, 0.55f);
}

TEST(AblationLearningRateTest, RejectsInvalidSchedulesAndUnrepresentableRates) {
  EXPECT_FALSE(AblationLearningRate(0, 1, 100, 40000).ok());
  EXPECT_FALSE(AblationLearningRate(-1, 1, 100, 40000).ok());
  EXPECT_FALSE(AblationLearningRate(1, 0, 100, 40000).ok());
  EXPECT_FALSE(AblationLearningRate(1, -1, 100, 40000).ok());
  EXPECT_FALSE(AblationLearningRate(1, 1, -1, 40000).ok());
  EXPECT_FALSE(AblationLearningRate(1, 1, 0, 0).ok());
  EXPECT_FALSE(AblationLearningRate(1, 1, 100, 100).ok());
  EXPECT_FALSE(AblationLearningRate(1, 1, 101, 100).ok());
  EXPECT_FALSE(AblationLearningRate(1, std::numeric_limits<double>::infinity(),
                                    100, 40000)
                   .ok());
  EXPECT_FALSE(AblationLearningRate(1, std::numeric_limits<double>::quiet_NaN(),
                                    100, 40000)
                   .ok());
  EXPECT_TRUE(absl::IsOutOfRange(
      AblationLearningRate(100, std::numeric_limits<double>::max(), 100, 40000)
          .status()));
  EXPECT_TRUE(absl::IsOutOfRange(
      AblationLearningRate(1, std::numeric_limits<double>::denorm_min(), 100,
                           40000)
          .status()));
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
