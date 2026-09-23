#include "src/llm/experiments/one_shot_memorizer/feature_capture.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/gpt2.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {
template <typename T>
absl::StatusOr<Buffer> Upload(cuda::Executor& executor,
                              const std::vector<T>& values) {
  ASSIGN_OR_RETURN(auto host,
                   cuda::PageLockedHostArray<T>::CopyFrom(executor, values));
  ASSIGN_OR_RETURN(auto buffer, Buffer::Allocate(executor, host.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(buffer.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload feature capture fixture"));
  return buffer;
}

TEST(FeatureCaptureTest, FinalNormNotBlockNormAndPinnedBf16Reprojection) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  const Gpt2Config config{.transformer_block_count = 2,
                          .model_width = 16,
                          .attention_heads = 1,
                          .feed_forward_width = 32,
                          .vocabulary_size = 13,
                          .pad_vocabulary = false};
  auto model = CreateGpt2(**executor, DataType::BF16, 17, config);
  ASSERT_TRUE(model.ok()) << model.status();
  constexpr int rows = 2 * kGpt2ContextLength;
  std::vector<int> tokens(rows), labels(rows, -1);
  for (int i = 0; i < rows; ++i)
    tokens[i] = i % 13;
  const std::vector<int> selected{0, 2, 12, 1023, 1027, 1050, 2047};
  for (int row : selected)
    labels[row] = (tokens[row] + 1) % 13;
  auto inputs = Upload(**executor, tokens);
  auto targets = Upload(**executor, labels);
  ASSERT_TRUE(inputs.ok()) << inputs.status();
  ASSERT_TRUE(targets.ok()) << targets.status();
  DataBatch batch{*inputs, *targets, 2, kGpt2ContextLength};
  auto captured = CaptureGpt2ReadoutBatch(**executor, **model, batch, 16, 13);
  ASSERT_TRUE(captured.ok()) << captured.status();
  EXPECT_EQ(captured->labels.size(), selected.size());
  EXPECT_EQ(captured->values.size(), selected.size() * 16);
  auto weights =
      CopyEffectiveBf16Readout(**executor, (*model)->weights()[0], 16, 13);
  ASSERT_TRUE(weights.ok()) << weights.status();
  for (size_t i = 0; i < selected.size(); ++i) {
    EXPECT_EQ(captured->labels[i], labels[selected[i]]);
    EXPECT_EQ(captured->sample_indices[i], selected[i] / kGpt2ContextLength);
    EXPECT_EQ(captured->positions[i], selected[i] % kGpt2ContextLength);
    float best = -std::numeric_limits<float>::infinity();
    int prediction = -1;
    for (int token = 0; token < 13; ++token) {
      float dot = 0;
      for (int dim = 0; dim < 16; ++dim)
        dot += captured->values[i * 16 + dim] * (*weights)[token * 16 + dim];
      if (dot > best) {
        best = dot;
        prediction = token;
      }
    }
    EXPECT_EQ(prediction, captured->original_predictions[i]);
  }
  EXPECT_FALSE(
      CaptureGpt2ReadoutBatch(**executor, **model, batch, 15, 13).ok());
  EXPECT_FALSE(CaptureGpt2ReadoutBatch(**executor, **model, batch, 0, 13).ok());
  batch.supervised_row_count = 1;
  EXPECT_FALSE(
      CaptureGpt2ReadoutBatch(**executor, **model, batch, 16, 13).ok());
  batch.supervised_row_count = -1;
  const auto saved_targets = batch.targets;
  auto bad_labels = labels;
  bad_labels[0] = 13;
  auto bad_targets = Upload(**executor, bad_labels);
  ASSERT_TRUE(bad_targets.ok());
  batch.targets = *bad_targets;
  EXPECT_FALSE(
      CaptureGpt2ReadoutBatch(**executor, **model, batch, 16, 13).ok());
  batch.targets = saved_targets;
  std::fill(labels.begin(), labels.end(), -1);
  targets = Upload(**executor, labels);
  ASSERT_TRUE(targets.ok());
  batch.targets = *targets;
  auto empty = CaptureGpt2ReadoutBatch(**executor, **model, batch, 16, 13);
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_TRUE(empty->labels.empty());
  EXPECT_TRUE((*executor)->Synchronize().ok());
}

TEST(FeatureCaptureTest, ReadoutConversionUsesBf16TiesToEvenAndValidatesShape) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  const std::vector<float> source{std::bit_cast<float>(uint32_t{0x3f808000}),
                                  std::bit_cast<float>(uint32_t{0x3f818000}),
                                  -1.5f, 0.0f};
  auto device = Upload(**executor, source);
  ASSERT_TRUE(device.ok()) << device.status();
  auto rounded = CopyEffectiveBf16Readout(**executor, *device, 2, 2);
  ASSERT_TRUE(rounded.ok()) << rounded.status();
  EXPECT_EQ(std::bit_cast<uint32_t>((*rounded)[0]), 0x3f800000u);
  EXPECT_EQ(std::bit_cast<uint32_t>((*rounded)[1]), 0x3f820000u);
  EXPECT_EQ((*rounded)[2], -1.5f);
  EXPECT_EQ((*rounded)[3], 0.0f);
  EXPECT_FALSE(CopyEffectiveBf16Readout(**executor, *device, 2, 3).ok());
  EXPECT_FALSE(CopyEffectiveBf16Readout(**executor, *device, 0, 2).ok());
  auto nonfinite = Upload(
      **executor, std::vector<float>{std::numeric_limits<float>::infinity()});
  ASSERT_TRUE(nonfinite.ok());
  EXPECT_FALSE(CopyEffectiveBf16Readout(**executor, *nonfinite, 1, 1).ok());
  EXPECT_TRUE((*executor)->Synchronize().ok());
}
}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
