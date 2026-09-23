#include "src/llm/experiments/one_shot_memorizer/checkpoint_probe.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layers/embedding.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

constexpr int kContext = 6;
constexpr int kVocabulary = 8;
constexpr int kStride = 32;
constexpr int kEos = 7;
constexpr int64_t kBatch = ActivationType::kBatchDimension;

// Records the actual fixed-size input batches; scripted outputs can depend on
// absolute row positions without adding a second context-probe implementation.
class RecordingLayer final : public Layer {
 public:
  using Script = std::function<std::vector<float>(absl::Span<const int>)>;

  explicit RecordingLayer(Script script)
      : input_signatures{{DataType::INT32, {kBatch, kContext}}},
        output_signatures{{DataType::FP32, {kBatch, kContext, kStride}}},
        script_(std::move(script)) {}

  absl::string_view name() const override {
    return "ContextProbeRecordingLayer";
  }
  absl::Span<const ActivationType> input_types() const override {
    return input_signatures;
  }
  absl::Span<const ActivationType> output_types() const override {
    return output_signatures;
  }
  absl::Span<Buffer> weights() override {
    return absl::MakeSpan(parameter_buffers);
  }
  DataType output_type() const override { return DataType::FP16; }

  std::vector<ActivationType> input_signatures;
  std::vector<ActivationType> output_signatures;
  BufferVec parameter_buffers;
  cuda::Executor* output_executor = nullptr;
  size_t output_count = 1;
  mutable std::vector<std::vector<int>> inputs_seen;

 private:
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override {
    if (inputs.size() != 1 || &inputs[0].executor() != &executor)
      return absl::InvalidArgumentError("invalid recording layer input");
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<int>::Allocate(
                         executor, inputs[0].size_bytes() / sizeof(int)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), inputs[0].data(), inputs[0].size_bytes(),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "download context probe fixture"));
    RETURN_IF_ERROR(executor.Synchronize());
    inputs_seen.emplace_back(host.begin(), host.end());
    const auto logits = script_(host.span());
    auto& destination =
        output_executor == nullptr ? executor : *output_executor;
    ASSIGN_OR_RETURN(auto staging, cuda::PageLockedHostArray<float>::CopyFrom(
                                       destination, logits));
    ASSIGN_OR_RETURN(auto device,
                     Buffer::Allocate(destination, staging.size_bytes()));
    if (!logits.empty()) {
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(device.data(), staging.data(), staging.size_bytes(),
                          cudaMemcpyHostToDevice, destination.stream()),
          "upload context probe fixture"));
    }
    FwdResult result;
    for (size_t i = 0; i < output_count; ++i)
      result.outputs.push_back(device);
    return result;
  }

  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override {
    return absl::InternalError("context probing must not run backward");
  }

  Script script_;
};

std::vector<float> ZeroLogits(absl::Span<const int> input) {
  return std::vector<float>(input.size() * kStride, 0.0f);
}

class CheckpointProbeTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }
  void TearDown() override {
    if (executor_ == nullptr)
      return;
    EXPECT_TRUE(executor_->Synchronize().ok());
  }

  absl::StatusOr<std::unique_ptr<EmbeddingLookupLayer>> MappingLayer(
      int nonfinite_token = -1) {
    ASSIGN_OR_RETURN(auto layer, EmbeddingLookupLayer::Create(
                                     *executor_, kVocabulary, kStride,
                                     DataType::FP16, kContext));
    std::vector<float> table(layer->weight().size_bytes() / sizeof(float),
                             1000.0f);
    const int next[kVocabulary] = {0, 2, 3, kEos, 2, 0, 0, 0};
    for (int token = 0; token < kVocabulary; ++token) {
      std::fill(table.begin() + token * kStride,
                table.begin() + token * kStride + kVocabulary, -2.0f);
      table[token * kStride + next[token]] = 4.0f;
    }
    if (nonfinite_token >= 0)
      table[nonfinite_token * kStride] =
          std::numeric_limits<float>::quiet_NaN();
    ASSIGN_OR_RETURN(auto staging, cuda::PageLockedHostArray<float>::CopyFrom(
                                       *executor_, table));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(layer->weights()[0].data(), staging.data(),
                        staging.size_bytes(), cudaMemcpyHostToDevice,
                        executor_->stream()),
        "initialize context probe token map"));
    return layer;
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(CheckpointProbeTest, RealEmbeddingModelKeepsLastTokenAndScoresEos) {
  auto layer = MappingLayer();
  ASSERT_TRUE(layer.ok()) << layer.status();
  const std::vector<std::vector<int>> sentences = {{1, 2, 3}, {4, 2, 3}};
  for (int batch_size : {1, 4}) {
    auto results =
        ProbeContextWindows(*executor_, **layer, sentences, kVocabulary, kEos,
                            1, batch_size, {-1, 0, 1, 3});
    ASSERT_TRUE(results.ok()) << results.status();
    ASSERT_EQ(results->size(), 4u);
    for (size_t i = 0; i < results->size(); ++i) {
      EXPECT_EQ((*results)[i].targets, 6);
      EXPECT_EQ((*results)[i].correct, i == 1 ? 0 : 6);
      EXPECT_EQ((*results)[i].nonfinite, 0);
      EXPECT_EQ((*results)[i].replacement, PrefixReplacement::kEos);
    }
    EXPECT_EQ((*results)[0].window, -1);
    EXPECT_EQ((*results)[1].window, 0);
    EXPECT_EQ((*results)[2].window, 1);
    EXPECT_EQ((*results)[3].window, 3);
  }
}

TEST_F(CheckpointProbeTest, ZeroEosTokenIsScoredForPromptOnlySentences) {
  RecordingLayer layer(ZeroLogits);
  const std::vector<std::vector<int>> sentences = {{1, 2}};
  auto results = ProbeContextWindows(*executor_, layer, sentences, kVocabulary,
                                     0, 2, 4, {-1, 0});
  ASSERT_TRUE(results.ok()) << results.status();
  ASSERT_EQ(results->size(), 2u);
  for (const auto& result : *results) {
    EXPECT_EQ(result.targets, 1);
    EXPECT_EQ(result.correct, 1);
    EXPECT_EQ(result.nonfinite, 0);
  }
}

TEST_F(CheckpointProbeTest, NonfiniteRowsCountOnlyWhenSupervised) {
  const std::vector<std::vector<int>> sentences = {{1, 2, 3}, {4, 2, 3}};
  auto layer = MappingLayer(kEos);
  ASSERT_TRUE(layer.ok()) << layer.status();
  auto results = ProbeContextWindows(*executor_, **layer, sentences,
                                     kVocabulary, kEos, 1, 4, {-1, 1, 0});
  ASSERT_TRUE(results.ok()) << results.status();
  ASSERT_EQ(results->size(), 3u);
  for (size_t i : {0u, 1u}) {
    EXPECT_EQ((*results)[i].targets, 6);
    EXPECT_EQ((*results)[i].correct, 6);
    EXPECT_EQ((*results)[i].nonfinite, 0);
  }
  EXPECT_EQ((*results)[2].targets, 6);
  EXPECT_EQ((*results)[2].correct, 0);
  EXPECT_EQ((*results)[2].nonfinite, 6);

  layer = MappingLayer(3);
  ASSERT_TRUE(layer.ok()) << layer.status();
  results = ProbeContextWindows(*executor_, **layer, sentences, kVocabulary,
                                kEos, 1, 4, {-1});
  ASSERT_TRUE(results.ok()) << results.status();
  EXPECT_EQ((*results)[0].targets, 6);
  EXPECT_EQ((*results)[0].correct, 4);
  EXPECT_EQ((*results)[0].nonfinite, 2);
}

TEST_F(CheckpointProbeTest, PreservesAbsolutePositionsAndMasksFinalBatchSlots) {
  RecordingLayer layer([](absl::Span<const int> input) {
    std::vector<float> logits(input.size() * kStride,
                              std::numeric_limits<float>::quiet_NaN());
    for (size_t row = 0; row < input.size(); ++row) {
      const int position = row % kContext;
      if (position < 1 || position > 4)
        continue;
      std::fill(logits.begin() + row * kStride,
                logits.begin() + row * kStride + kVocabulary, -2.0f);
      const int winner = position == 4 ? kEos : position + 2;
      logits[row * kStride + winner] = 5.0f;
    }
    return logits;
  });
  const std::vector<std::vector<int>> sentences = {{1, 2, 3, 4, 5}};
  auto results = ProbeContextWindows(*executor_, layer, sentences, kVocabulary,
                                     kEos, 2, 3, {-1, 1});
  ASSERT_TRUE(results.ok()) << results.status();
  ASSERT_EQ(results->size(), 2u);
  for (const auto& result : *results) {
    EXPECT_EQ(result.targets, 4);
    EXPECT_EQ(result.correct, 4);
    EXPECT_EQ(result.nonfinite, 0);
  }
  EXPECT_EQ(layer.inputs_seen,
            (std::vector<std::vector<int>>{
                {1, 2, 7, 7, 7, 7, 1, 2, 3, 7, 7, 7, 1, 2, 3, 4, 7, 7},
                {1, 2, 3, 4, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7},
                {7, 2, 7, 7, 7, 7, 7, 7, 3, 7, 7, 7, 7, 7, 7, 4, 7, 7},
                {7, 7, 7, 7, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7}}));
}

TEST_F(CheckpointProbeTest, OtherSentenceReplacementCyclesDonorPrefixOnly) {
  RecordingLayer layer(ZeroLogits);
  const std::vector<std::vector<int>> sentences = {{1, 2, 3, 4}, {5, 6}};
  auto results =
      ProbeContextWindows(*executor_, layer, sentences, kVocabulary, kEos, 2, 4,
                          {1}, PrefixReplacement::kOtherSentence);
  ASSERT_TRUE(results.ok()) << results.status();
  ASSERT_EQ(results->size(), 1u);
  EXPECT_EQ((*results)[0].targets, 4);
  EXPECT_EQ((*results)[0].replacement, PrefixReplacement::kOtherSentence);
  EXPECT_EQ(
      layer.inputs_seen,
      (std::vector<std::vector<int>>{{5, 2, 7, 7, 7, 7, 5, 6, 3, 7, 7, 7,
                                      5, 6, 5, 4, 7, 7, 1, 6, 7, 7, 7, 7}}));
}

TEST_F(CheckpointProbeTest, RejectsInvalidOptionsTokensAndTooLongSentences) {
  RecordingLayer layer(ZeroLogits);
  const std::vector<std::vector<int>> sentences = {{1, 2, 3}};
  for (int prompt : {-1, 0, kContext + 1}) {
    EXPECT_EQ(ProbeContextWindows(*executor_, layer, sentences, kVocabulary,
                                  kEos, prompt, 1, {-1})
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  for (int batch : {-1, 0}) {
    EXPECT_EQ(ProbeContextWindows(*executor_, layer, sentences, kVocabulary,
                                  kEos, 1, batch, {-1})
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  EXPECT_EQ(
      ProbeContextWindows(*executor_, layer, sentences, 0, kEos, 1, 1, {-1})
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument);
  for (int eos : {-1, kVocabulary}) {
    EXPECT_EQ(ProbeContextWindows(*executor_, layer, sentences, kVocabulary,
                                  eos, 1, 1, {-1})
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  EXPECT_EQ(ProbeContextWindows(*executor_, layer, sentences, kVocabulary, kEos,
                                1, 1, {-2})
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ProbeContextWindows(*executor_, layer, sentences, kVocabulary, kEos,
                                1, 1, {})
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      ProbeContextWindows(*executor_, layer, {}, kVocabulary, kEos, 1, 1, {-1})
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ProbeContextWindows(*executor_, layer, sentences, kVocabulary, kEos,
                                1, 1, {-1}, static_cast<PrefixReplacement>(-1))
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ProbeContextWindows(*executor_, layer, sentences, kVocabulary, kEos,
                                1, 1, {-1}, PrefixReplacement::kOtherSentence)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  for (const auto& sentence : std::vector<std::vector<int>>{
           {}, {-1}, {kVocabulary}, {kEos}, {1, 2, 3, 4, 5, 6, 1}}) {
    const std::vector<std::vector<int>> invalid = {sentence};
    EXPECT_EQ(ProbeContextWindows(*executor_, layer, invalid, kVocabulary, kEos,
                                  1, 1, {-1})
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  EXPECT_TRUE(layer.inputs_seen.empty());
}

TEST_F(CheckpointProbeTest, RejectsInvalidSignaturesAndOutputDimensions) {
  const std::vector<std::vector<int>> sentences = {{1, 2, 3}};
  for (int variant = 0; variant < 5; ++variant) {
    RecordingLayer layer(ZeroLogits);
    if (variant == 0)
      layer.input_signatures.clear();
    if (variant == 1)
      layer.input_signatures = {{DataType::FP32, {kBatch, kContext}}};
    if (variant == 2)
      layer.output_signatures = {{DataType::BF16, {kBatch, kContext, kStride}}};
    if (variant == 3)
      layer.output_signatures = {{DataType::FP32, {kBatch, kContext, 4}}};
    if (variant == 4)
      layer.output_signatures = {
          {DataType::FP32, {kBatch, kContext + 1, kStride}}};
    EXPECT_EQ(ProbeContextWindows(*executor_, layer, sentences, kVocabulary,
                                  kEos, 1, 1, {-1})
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(layer.inputs_seen.empty());
  }
  for (size_t output_count : {0u, 2u}) {
    RecordingLayer layer(ZeroLogits);
    layer.output_count = output_count;
    EXPECT_FALSE(ProbeContextWindows(*executor_, layer, sentences, kVocabulary,
                                     kEos, 1, 1, {-1})
                     .ok());
  }
  RecordingLayer truncated(
      [](absl::Span<const int>) { return std::vector<float>{0.0f}; });
  EXPECT_FALSE(ProbeContextWindows(*executor_, truncated, sentences,
                                   kVocabulary, kEos, 1, 1, {-1})
                   .ok());
}

TEST_F(CheckpointProbeTest, RejectsWeightsAndOutputsFromAnotherExecutor) {
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  const std::vector<std::vector<int>> sentences = {{1, 2, 3}};
  auto foreign = Buffer::Allocate(**other, sizeof(float));
  ASSERT_TRUE(foreign.ok()) << foreign.status();
  RecordingLayer layer(ZeroLogits);
  layer.parameter_buffers.push_back(*foreign);
  EXPECT_EQ(ProbeContextWindows(*executor_, layer, sentences, kVocabulary, kEos,
                                1, 1, {-1})
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(layer.inputs_seen.empty());
  layer.parameter_buffers.clear();
  layer.output_executor = other->get();
  EXPECT_EQ(ProbeContextWindows(*executor_, layer, sentences, kVocabulary, kEos,
                                1, 1, {-1})
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE((*other)->Synchronize().ok());
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
