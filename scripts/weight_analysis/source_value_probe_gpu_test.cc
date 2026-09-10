#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "scripts/weight_analysis/causal_probe.h"
#include "scripts/weight_analysis/phrase_probe.h"
#include "scripts/weight_analysis/source_value_probe.h"
#include "src/llm/recipes/gpt2.h"

namespace pluto::weight_analysis {
namespace {
using Bytes = cuda::PageLockedHostArray<uint8_t>;
constexpr int kContext = llm::kGpt2ContextLength;
constexpr int kWidth = llm::kGpt2ModelWidth;
constexpr int kHeadWidth = llm::kGpt2AttentionHeadDimension;
constexpr int kVocabulary = llm::kGpt2PaddedVocabularySize;

// This test exercises the actual fixed-size GPT-2 implementation, not a CPU
// imitation. Keep it separate from CPU validation and never run it during the
// timed paired-training arms. Lack of a GPU is a failure, not a silent skip.
TEST(SourceValueGpuTest,
     NativeTwoSequenceReplayScopeCausalityAndWeightIntegrity) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  auto model = llm::CreateGpt2(**executor, llm::DataType::BF16, 17);
  ASSERT_TRUE(model.ok()) << model.status();
  auto weights = UniqueWeights(**executor, (*model)->weights());
  ASSERT_TRUE(weights.ok()) << weights.status();
  ASSERT_TRUE(ValidateGpt2Weights(*weights).ok());
  std::vector<Bytes> weight_snapshots;
  for (const auto &weight : *weights) {
    auto bytes = ReadPrefix(**executor, weight, weight.size_bytes() / 4, 1, 4);
    ASSERT_TRUE(bytes.ok());
    weight_snapshots.push_back(std::move(*bytes));
  }

  constexpr int rows = 2 * kContext;
  auto tokens = cuda::PageLockedHostArray<int32_t>::Allocate(rows);
  ASSERT_TRUE(tokens.ok());
  for (int i = 0; i < rows; ++i) (*tokens)[i] = (17 * i + 3) % 1000;
  auto input = cuda::Buffer::Allocate(**executor, tokens->size_bytes());
  ASSERT_TRUE(input.ok());
  ASSERT_EQ(cudaMemcpyAsync(input->data(), tokens->data(), tokens->size_bytes(),
                            cudaMemcpyHostToDevice, (*executor)->stream()),
            cudaSuccess);
  llm::Tape clean_tape;
  auto clean =
      (*model)->fwd(**executor, absl::MakeConstSpan(&*input, 1), &clean_tape);
  ASSERT_TRUE(clean.ok()) << clean.status();
  ASSERT_TRUE(ValidateGpt2Tape(clean_tape).ok());
  auto clean_bytes = ReadPrefix(**executor, *clean, rows, kVocabulary, 4);
  ASSERT_TRUE(clean_bytes.ok());

  auto probe =
      SourceValueProbe::Create(**executor, clean_tape, *clean, *weights, 1);
  ASSERT_TRUE(probe.ok()) << probe.status();
  const auto &clean_qkv = (*probe)->original_qkv();
  const auto &clean_context = (*probe)->original_context();
  auto qkv_before = ReadPrefix(**executor, clean_qkv, rows, 3 * kWidth, 2);
  auto context_before = ReadPrefix(**executor, clean_context, rows, kWidth, 2);
  ASSERT_TRUE(qkv_before.ok());
  ASSERT_TRUE(context_before.ok());

  // Source 0 feeds query 1, but only that query/head is spliced. Therefore
  // sequence 0 AND query 0 in sequence 1 must keep every logit byte unchanged.
  SourceValueSelection selection{1, 2, 1, 1, 0, 1.0f};
  const size_t value_offset = 2 * (static_cast<size_t>(kContext) * 3 * kWidth +
                                   2 * kWidth + selection.head * kHeadWidth);
  const size_t context_offset =
      2 * (static_cast<size_t>(kContext + 1) * kWidth +
           selection.head * kHeadWidth);
  constexpr size_t head_bytes = kHeadWidth * sizeof(uint16_t);
  for (float scale : {1.0f, 0.5f, 0.0f}) {
    selection.scale = scale;
    auto changed = (*probe)->Apply(**executor, selection);
    ASSERT_TRUE(changed.ok()) << changed.status();
    EXPECT_NE(changed->modified_qkv.data(), clean_qkv.data());
    EXPECT_NE(changed->spliced_context.data(), clean_context.data());
    EXPECT_NE(changed->logits.data(), clean->data());
    auto qkv_bytes =
        ReadPrefix(**executor, changed->modified_qkv, rows, 3 * kWidth, 2);
    auto context_bytes =
        ReadPrefix(**executor, changed->spliced_context, rows, kWidth, 2);
    auto attention_bytes =
        ReadPrefix(**executor, changed->replayed_attention, rows, kWidth, 2);
    auto logits_bytes =
        ReadPrefix(**executor, changed->logits, rows, kVocabulary, 4);
    ASSERT_TRUE(qkv_bytes.ok());
    ASSERT_TRUE(context_bytes.ok());
    ASSERT_TRUE(attention_bytes.ok());
    ASSERT_TRUE(logits_bytes.ok());
    EXPECT_EQ(std::memcmp(qkv_bytes->data(), qkv_before->data(), value_offset),
              0);
    EXPECT_EQ(std::memcmp(qkv_bytes->data() + value_offset + head_bytes,
                          qkv_before->data() + value_offset + head_bytes,
                          qkv_bytes->size_bytes() - value_offset - head_bytes),
              0);
    for (int i = 0; i < kHeadWidth; ++i) {
      uint16_t original, actual;
      std::memcpy(&original, qkv_before->data() + value_offset + 2 * i, 2);
      std::memcpy(&actual, qkv_bytes->data() + value_offset + 2 * i, 2);
      EXPECT_EQ(actual, *ScaleSourceBf16(original, scale));
    }
    EXPECT_EQ(std::memcmp(context_bytes->data(), context_before->data(),
                          context_offset),
              0);
    EXPECT_EQ(
        std::memcmp(context_bytes->data() + context_offset + head_bytes,
                    context_before->data() + context_offset + head_bytes,
                    context_bytes->size_bytes() - context_offset - head_bytes),
        0);
    EXPECT_EQ(std::memcmp(context_bytes->data() + context_offset,
                          attention_bytes->data() + context_offset, head_bytes),
              0);
    EXPECT_EQ(std::memcmp(logits_bytes->data(), clean_bytes->data(),
                          static_cast<size_t>(kContext + 1) * kVocabulary * 4),
              0);
    if (scale == 1.0f) {
      EXPECT_EQ(std::memcmp(logits_bytes->data(), clean_bytes->data(),
                            clean_bytes->size_bytes()),
                0);
      EXPECT_EQ(std::memcmp(attention_bytes->data(), context_before->data(),
                            context_before->size_bytes()),
                0);
    } else {
      EXPECT_NE(
          std::memcmp(context_bytes->data() + context_offset,
                      context_before->data() + context_offset, head_bytes),
          0);
      const size_t row_offset =
          static_cast<size_t>(kContext + 1) * kVocabulary * 4;
      EXPECT_NE(std::memcmp(logits_bytes->data() + row_offset,
                            clean_bytes->data() + row_offset, kVocabulary * 4),
                0);
    }
  }

  // A future source changes native attention at later queries, but not the
  // selected causal query. This catches accidentally forwarding all changed
  // contexts instead of the one 64-element query/head slice.
  auto future = (*probe)->Apply(**executor, {1, 2, 0, 0, 1023, 0});
  ASSERT_TRUE(future.ok()) << future.status();
  auto future_logits =
      ReadPrefix(**executor, future->logits, rows, kVocabulary, 4);
  ASSERT_TRUE(future_logits.ok());
  EXPECT_EQ(std::memcmp(future_logits->data(), clean_bytes->data(),
                        clean_bytes->size_bytes()),
            0);

  // Exercise both ends of the suffix construction. Block 0 replays every
  // later block; block 7 has only its own MLP before final normalization/head.
  for (int block : {0, 7}) {
    auto boundary = SourceValueProbe::Create(**executor, clean_tape, *clean,
                                             *weights, block);
    ASSERT_TRUE(boundary.ok()) << "block=" << block << " " << boundary.status();
    auto identity =
        (*boundary)->Apply(**executor, {block, 7, 1, 1023, 1022, 1});
    ASSERT_TRUE(identity.ok()) << identity.status();
  }

  EXPECT_FALSE((*probe)->Apply(**executor, {0, 2, 1, 1, 0, 1}).ok());
  EXPECT_FALSE((*probe)->Apply(**executor, {1, 2, 2, 1, 0, 1}).ok());
  EXPECT_FALSE((*probe)->Apply(**executor, {1, 2, 1, 1, 0, 0.25f}).ok());
  auto other_executor = cuda::Executor::Create();
  ASSERT_TRUE(other_executor.ok());
  EXPECT_FALSE((*probe)->Apply(**other_executor, selection).ok());
  EXPECT_FALSE(SourceValueProbe::Create(**other_executor, clean_tape, *clean,
                                        *weights, 1)
                   .ok());

  // Invalid tape/shape evidence must be rejected before creating a tail.
  llm::Tape empty;
  EXPECT_FALSE(
      SourceValueProbe::Create(**executor, empty, *clean, *weights, 1).ok());
  auto malformed_tape = clean_tape;
  malformed_tape.children[3]
      .children[0]
      .children[0]
      .children[2]
      .intermediates[0] = *input;
  EXPECT_FALSE(
      SourceValueProbe::Create(**executor, malformed_tape, *clean, *weights, 1)
          .ok());
  EXPECT_FALSE(
      SourceValueProbe::Create(**executor, clean_tape, *input, *weights, 1)
          .ok());

  // Corrupt one clean-logit evidence byte in an independent allocation. The
  // required full-output identity gate must not overlook it just because it
  // lies outside the intervention's selected query.
  auto wrong_logits = cuda::Buffer::Allocate(**executor, clean->size_bytes());
  ASSERT_TRUE(wrong_logits.ok());
  ASSERT_EQ(
      cudaMemcpyAsync(wrong_logits->data(), clean->data(), clean->size_bytes(),
                      cudaMemcpyDeviceToDevice, (*executor)->stream()),
      cudaSuccess);
  auto changed_word = cuda::PageLockedHostArray<uint32_t>::Allocate(1);
  ASSERT_TRUE(changed_word.ok());
  std::memcpy(changed_word->data(), clean_bytes->data(), 4);
  (*changed_word)[0] ^= 1;
  ASSERT_EQ(cudaMemcpyAsync(wrong_logits->data(), changed_word->data(), 4,
                            cudaMemcpyHostToDevice, (*executor)->stream()),
            cudaSuccess);
  EXPECT_FALSE(SourceValueProbe::Create(**executor, clean_tape, *wrong_logits,
                                        *weights, 7)
                   .ok());

  // All 100 source parameter tensors (not just target-block tensors) must be
  // unchanged. Input IDs and the production clean activations are read-only.
  for (size_t i = 0; i < weights->size(); ++i) {
    auto after = ReadPrefix(**executor, (*weights)[i],
                            (*weights)[i].size_bytes() / 4, 1, 4);
    ASSERT_TRUE(after.ok());
    EXPECT_EQ(std::memcmp(after->data(), weight_snapshots[i].data(),
                          after->size_bytes()),
              0)
        << "weight=" << i;
  }
  auto input_after = ReadPrefix(**executor, *input, rows, 1, 4);
  ASSERT_TRUE(input_after.ok());
  EXPECT_EQ(
      std::memcmp(input_after->data(), tokens->data(), tokens->size_bytes()),
      0);
  ASSERT_TRUE((*executor)->Synchronize().ok());
}

}  // namespace
}  // namespace pluto::weight_analysis
