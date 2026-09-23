#include "src/llm/experiments/one_shot_memorizer/mlp_probe.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/plain_text_tokenizer.h"
#include "src/llm/gpt2.h"
#include "src/llm/layers/fully_connected.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

constexpr int kWidth = 16;
constexpr int kFeatures = 32;
constexpr int kVocabulary = 256;
constexpr int kEos = 0;
constexpr float kBias = 4.0003f;

absl::Status Upload(cuda::Executor& executor, const Buffer& destination,
                    absl::Span<const float> values) {
  if (&destination.executor() != &executor ||
      destination.size_bytes() != values.size() * sizeof(float))
    return absl::InvalidArgumentError("invalid MLP test upload");
  ASSIGN_OR_RETURN(auto host,
                   cuda::PageLockedHostArray<float>::CopyFrom(executor, values));
  return cuda::CudaStatus(
      cudaMemcpyAsync(destination.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload MLP probe fixture");
}

class RecordingGreedyLayer final : public Layer {
 public:
  absl::string_view name() const override { return "RecordingGreedyLayer"; }
  absl::Span<const ActivationType> input_types() const override {
    return {&input_, 1};
  }
  absl::Span<const ActivationType> output_types() const override {
    return {&output_, 1};
  }
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::BF16; }
  mutable std::vector<std::vector<int>> inputs_seen;
  bool end_after_c = false;
  int nonfinite_token = -1;

 private:
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override {
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<int>::Allocate(
                         executor, inputs[0].size_bytes() / sizeof(int)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), inputs[0].data(), host.size_bytes(),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "record MLP greedy inputs"));
    RETURN_IF_ERROR(executor.Synchronize());
    inputs_seen.emplace_back(host.begin(), host.end());
    std::vector<float> logits(host.size() * kVocabulary, -1);
    for (size_t row = 0; row < host.size(); ++row) {
      int winner = kEos;
      if (host[row] == 'b')
        winner = 'c';
      if (host[row] == 'c' && !end_after_c)
        winner = 'a';
      logits[row * kVocabulary + winner] = 1;
      if (host[row] == nonfinite_token)
        logits[row * kVocabulary] = std::numeric_limits<float>::quiet_NaN();
    }
    ASSIGN_OR_RETURN(auto device,
                     Buffer::Allocate(executor, logits.size() * sizeof(float)));
    RETURN_IF_ERROR(Upload(executor, device, logits));
    return FwdResult{{std::move(device)}, {}};
  }
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override {
    return absl::InternalError("greedy verification must not run backward");
  }
  const ActivationType input_{DataType::INT32,
                              {ActivationType::kBatchDimension, 4}};
  const ActivationType output_{
      DataType::FP32, {ActivationType::kBatchDimension, 4, kVocabulary}};
};

class MlpProbeTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }
  void TearDown() override {
    if (!executor_)
      return;
    EXPECT_TRUE(executor_->Synchronize().ok());
  }

  absl::StatusOr<std::unique_ptr<PaddedLineDataSetIterator>> Dataset(
      bool shuffle = false) {
    tokenizer::PlainTextTokenizer tokenizer;
    return PaddedLineDataSetIterator::Create(
        *executor_, "bbaaa\nbbaaaa\nbbaaaa\n", tokenizer,
        {.batch_size = 2,
         .context_length = kGpt2ContextLength,
         .prompt_tokens = 2,
         .eos_token = kEos,
         .shuffle = shuffle});
  }

  absl::StatusOr<std::unique_ptr<ComposedLayer>> Model(int blocks = 1) {
    const Gpt2Config config{.transformer_block_count = blocks,
                            .model_width = kWidth,
                            .attention_heads = 1,
                            .feed_forward_width = kFeatures,
                            .vocabulary_size = kVocabulary,
                            .pad_vocabulary = false};
    ASSIGN_OR_RETURN(auto model,
                     CreateGpt2(*executor_, DataType::BF16, 7, config));
    // Controlled, non-trained GPT-2: attention and W1/W2 are zero, each MLP
    // adds a positive bias. Tied token embeddings distinguish 'a' from 'b'.
    // Full model predicts 'a' on every real row; zeroing the first MLP exposes
    // the negative 'b' representation at each first supervised query.
    auto weights = model->weights();
    if (weights.size() != static_cast<size_t>(5 + blocks * 12))
      return absl::InternalError("unexpected GPT-2 fixture weight layout");
    for (const auto& weight : weights)
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemsetAsync(weight.data(), 0, weight.size_bytes(),
                          executor_->stream()),
          "clear controlled GPT-2 fixture"));
    std::vector<float> embedding(static_cast<size_t>(kVocabulary) * kWidth, 0);
    embedding[static_cast<size_t>('a') * kWidth] = 1;
    embedding[static_cast<size_t>('b') * kWidth] = -1;
    RETURN_IF_ERROR(Upload(*executor_, weights[0], embedding));
    const std::vector<float> gamma(kWidth, 1);
    for (int block = 0; block < blocks; ++block) {
      const size_t base = 2 + block * 12;
      RETURN_IF_ERROR(Upload(*executor_, weights[base], gamma));
      RETURN_IF_ERROR(Upload(*executor_, weights[base + 6], gamma));
      std::vector<float> bias(kWidth, 0);
      bias[0] = kBias;
      RETURN_IF_ERROR(Upload(*executor_, weights[base + 11], bias));
    }
    RETURN_IF_ERROR(Upload(*executor_, weights[2 + blocks * 12], gamma));
    return model;
  }

  absl::StatusOr<std::unique_ptr<FullyConnectedLayer>> Projection(
      int input_width, int output_width = kWidth, float identity_scale = 0,
      cuda::Executor* executor = nullptr, DataType type = DataType::BF16) {
    auto& selected = executor == nullptr ? *executor_ : *executor;
    ASSIGN_OR_RETURN(auto layer, FullyConnectedLayer::Create(
                                     selected, input_width, output_width, type,
                                     kGpt2ContextLength));
    if (identity_scale != 0)
      RETURN_IF_ERROR(layer->InitializeIdentity(identity_scale));
    return layer;
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(MlpProbeTest,
       CapturesPromptRowsExcludesPaddingAndPreservesBiasPrecision) {
  auto model = Model();
  auto dataset = Dataset();
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE(dataset.ok()) << dataset.status();
  auto capture = CaptureGpt2Mlps(*executor_, **model, **dataset, kWidth,
                                 kFeatures, 1, kVocabulary);
  ASSERT_TRUE(capture.ok()) << capture.status();
  ASSERT_EQ(capture->blocks.size(), 1u);
  EXPECT_EQ(capture->sentences, 3);
  EXPECT_EQ(capture->target_count, 14);
  EXPECT_EQ(capture->correct_targets, 11);
  EXPECT_EQ(capture->exact_sentences, 0);
  const auto& block = capture->blocks[0];
  EXPECT_EQ(block.width, kWidth);
  EXPECT_EQ(block.feature_width, kFeatures);
  EXPECT_EQ(block.normalized_inputs.size(), 17u * kWidth);
  EXPECT_EQ(block.features.size(), 17u * kFeatures);
  EXPECT_EQ(block.outputs.size(), 17u * kWidth);
  EXPECT_EQ(block.effective_output_weights.size(),
            static_cast<size_t>(kFeatures) * kWidth);
  ASSERT_EQ(block.output_bias.size(), static_cast<size_t>(kWidth));
  EXPECT_EQ(block.output_bias[0], kBias);
  EXPECT_NE(block.output_bias[0], 4.0f);     // Bias was not rounded to BF16.
  EXPECT_LT(block.normalized_inputs[0], 0);  // MLP norm, not the final norm.
  EXPECT_GT(block.normalized_inputs[2 * kWidth], 0);
  for (float value : block.features)
    EXPECT_EQ(value, 0);
  for (float value : block.effective_output_weights)
    EXPECT_EQ(value, 0);
  for (size_t row = 0; row < 17; ++row) {
    EXPECT_EQ(block.outputs[row * kWidth], 4.0f);
    for (int channel = 1; channel < kWidth; ++channel)
      EXPECT_EQ(block.outputs[row * kWidth + channel], 0);
  }
  const auto again = CaptureGpt2Mlps(*executor_, **model, **dataset, kWidth,
                                     kFeatures, 1, kVocabulary);
  ASSERT_TRUE(again.ok()) << again.status();
  EXPECT_EQ(again->blocks[0].normalized_inputs, block.normalized_inputs);
  EXPECT_EQ(again->blocks[0].outputs, block.outputs);
}

TEST_F(MlpProbeTest,
       ExactOutputProjectionClonesPreserveSingleAndAllBlockResults) {
  auto model = Model(2);
  auto dataset = Dataset();
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE(dataset.ok()) << dataset.status();
  const auto capture = CaptureGpt2Mlps(*executor_, **model, **dataset, kWidth,
                                       kFeatures, 2, kVocabulary);
  ASSERT_TRUE(capture.ok()) << capture.status();
  const auto baseline =
      EvaluateMlpReplacements(*executor_, **model, **dataset, {}, kVocabulary);
  ASSERT_TRUE(baseline.ok()) << baseline.status();
  EXPECT_EQ(baseline->targets, 14);
  EXPECT_EQ(baseline->correct_targets, 11);
  EXPECT_EQ(baseline->targets_per_sentence, (std::vector<int64_t>{4, 5, 5}));
  EXPECT_EQ(baseline->correct_per_sentence, (std::vector<int64_t>{3, 4, 4}));
  std::vector<std::unique_ptr<FullyConnectedLayer>> projections;
  std::vector<MlpReplacement> replacements;
  for (int block = 0; block < 2; ++block) {
    auto projection = Projection(kFeatures);
    ASSERT_TRUE(projection.ok()) << projection.status();
    ASSERT_TRUE(Upload(*executor_, (*projection)->weights()[0],
                       capture->blocks[block].effective_output_weights)
                    .ok());
    ASSERT_TRUE(Upload(*executor_, (*projection)->weights()[1],
                       capture->blocks[block].output_bias)
                    .ok());
    replacements.push_back({block, MlpSource::kGelu, projection->get()});
    projections.push_back(std::move(*projection));
  }
  for (const auto selected : {absl::MakeConstSpan(replacements).first(1),
                              absl::MakeConstSpan(replacements)}) {
    const auto replaced = EvaluateMlpReplacements(
        *executor_, **model, **dataset, selected, kVocabulary);
    ASSERT_TRUE(replaced.ok()) << replaced.status();
    EXPECT_EQ(replaced->targets, baseline->targets);
    EXPECT_EQ(replaced->correct_targets, baseline->correct_targets);
    EXPECT_EQ(replaced->sentences, baseline->sentences);
    EXPECT_EQ(replaced->exact_sentences, baseline->exact_sentences);
    EXPECT_EQ(replaced->targets_per_sentence, baseline->targets_per_sentence);
    EXPECT_EQ(replaced->correct_per_sentence, baseline->correct_per_sentence);
  }
}

TEST_F(MlpProbeTest,
       ZeroBranchChangesKnownPredictionsWithoutChangingOriginalWeights) {
  auto model = Model();
  auto dataset = Dataset();
  auto zero = Projection(kWidth);
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE(dataset.ok()) << dataset.status();
  ASSERT_TRUE(zero.ok()) << zero.status();
  const std::vector<MlpReplacement> replacements{
      {0, MlpSource::kLayerNorm, zero->get()}};
  const auto changed = EvaluateMlpReplacements(*executor_, **model, **dataset,
                                               replacements, kVocabulary);
  ASSERT_TRUE(changed.ok()) << changed.status();
  EXPECT_EQ(changed->targets, 14);
  EXPECT_EQ(changed->correct_targets, 8);
  EXPECT_EQ(changed->correct_per_sentence, (std::vector<int64_t>{2, 3, 3}));
  const auto original =
      EvaluateMlpReplacements(*executor_, **model, **dataset, {}, kVocabulary);
  ASSERT_TRUE(original.ok()) << original.status();
  EXPECT_EQ(original->correct_targets, 11);
}

TEST_F(MlpProbeTest,
       SimultaneousReplacementsUseFreshIntervenedLayerNormInputs) {
  auto model = Model(2);
  auto dataset = Dataset();
  auto zero = Projection(kWidth);
  auto identity = Projection(kWidth, kWidth, 8);
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE(dataset.ok()) << dataset.status();
  ASSERT_TRUE(zero.ok()) << zero.status();
  ASSERT_TRUE(identity.ok()) << identity.status();
  std::vector<MlpReplacement> replacements{
      {1, MlpSource::kLayerNorm, identity->get()}};
  const auto single = EvaluateMlpReplacements(*executor_, **model, **dataset,
                                              replacements, kVocabulary);
  ASSERT_TRUE(single.ok()) << single.status();
  EXPECT_EQ(single->correct_targets, 11);
  replacements.push_back({0, MlpSource::kLayerNorm, zero->get()});
  const auto both = EvaluateMlpReplacements(*executor_, **model, **dataset,
                                            replacements, kVocabulary);
  ASSERT_TRUE(both.ok()) << both.status();
  EXPECT_EQ(both->correct_targets, 8);
  EXPECT_EQ(both->correct_per_sentence, (std::vector<int64_t>{2, 3, 3}));
}

TEST_F(MlpProbeTest, RejectsInvalidSelectorsSourcesShapesAndForeignProjection) {
  auto model = Model();
  auto dataset = Dataset();
  auto normal = Projection(kWidth);
  auto gelu = Projection(kFeatures);
  auto wrong_output = Projection(kWidth, kWidth + 1);
  auto wrong_type = Projection(kWidth, kWidth, 0, nullptr, DataType::FP16);
  ASSERT_TRUE(model.ok());
  ASSERT_TRUE(dataset.ok());
  ASSERT_TRUE(normal.ok());
  ASSERT_TRUE(gelu.ok());
  ASSERT_TRUE(wrong_output.ok());
  ASSERT_TRUE(wrong_type.ok());
  for (const auto& replacement : std::vector<MlpReplacement>{
           {-1, MlpSource::kLayerNorm, normal->get()},
           {0, MlpSource::kLayerNorm, nullptr},
           {0, static_cast<MlpSource>(99), normal->get()},
           {0, MlpSource::kLayerNorm, gelu->get()},
           {0, MlpSource::kGelu, normal->get()},
           {0, MlpSource::kLayerNorm, wrong_output->get()},
           {0, MlpSource::kLayerNorm, wrong_type->get()}}) {
    EXPECT_EQ(EvaluateMlpReplacements(*executor_, **model, **dataset,
                                      {&replacement, 1}, kVocabulary)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  const std::vector<MlpReplacement> duplicate{
      {0, MlpSource::kLayerNorm, normal->get()},
      {0, MlpSource::kLayerNorm, normal->get()}};
  EXPECT_EQ(EvaluateMlpReplacements(*executor_, **model, **dataset, duplicate,
                                    kVocabulary)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  const MlpReplacement absent{1, MlpSource::kLayerNorm, normal->get()};
  EXPECT_EQ(EvaluateMlpReplacements(*executor_, **model, **dataset,
                                    {&absent, 1}, kVocabulary)
                .status()
                .code(),
            absl::StatusCode::kNotFound);
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  auto foreign = Projection(kWidth, kWidth, 0, other->get());
  ASSERT_TRUE(foreign.ok()) << foreign.status();
  const MlpReplacement replacement{0, MlpSource::kLayerNorm, foreign->get()};
  EXPECT_EQ(EvaluateMlpReplacements(*executor_, **model, **dataset,
                                    {&replacement, 1}, kVocabulary)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE((*other)->Synchronize().ok());
}

TEST_F(MlpProbeTest, RejectsCaptureShapeBlockCountAndShuffledDatasetMismatch) {
  auto model = Model();
  auto dataset = Dataset();
  auto shuffled = Dataset(true);
  ASSERT_TRUE(model.ok());
  ASSERT_TRUE(dataset.ok());
  ASSERT_TRUE(shuffled.ok());
  for (const auto& dimensions :
       std::vector<std::vector<int>>{{0, kFeatures, 1},
                                     {kWidth + 1, kFeatures, 1},
                                     {kWidth, kFeatures + 1, 1},
                                     {kWidth, kFeatures, 0},
                                     {kWidth, kFeatures, 2},
                                     {kWidth, kFeatures, -1}})
    EXPECT_FALSE(CaptureGpt2Mlps(*executor_, **model, **dataset, dimensions[0],
                                 dimensions[1], dimensions[2], kVocabulary)
                     .ok());
  EXPECT_EQ(CaptureGpt2Mlps(*executor_, **model, **shuffled, kWidth, kFeatures,
                            1, kVocabulary)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      EvaluateMlpReplacements(*executor_, **model, **shuffled, {}, kVocabulary)
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument);
}

TEST_F(MlpProbeTest, GreedyFeedsActualPredictionsAndRequiresEosAtGoldEnd) {
  tokenizer::PlainTextTokenizer tokenizer;
  auto dataset = PaddedLineDataSetIterator::Create(
      *executor_, "bbca\nbbc\nbbca\n", tokenizer,
      {.batch_size = 2,
       .context_length = 4,
       .prompt_tokens = 2,
       .eos_token = kEos});
  ASSERT_TRUE(dataset.ok()) << dataset.status();
  RecordingGreedyLayer model;
  model.nonfinite_token = kEos;
  const auto result =
      VerifyMlpGreedyCompletions(*executor_, model, **dataset, {}, kVocabulary);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->sentences, 3);
  EXPECT_EQ(result->exact_sentences, 2);
  EXPECT_EQ(result->exact_per_sentence, (std::vector<bool>{true, false, true}));
  EXPECT_EQ(result->generated_targets, 8);
  EXPECT_EQ(model.inputs_seen, (std::vector<std::vector<int>>{
                                   {'b', 'b', 0, 0, 'b', 'b', 0, 0},
                                   {'b', 'b', 'c', 0, 'b', 'b', 'c', 0},
                                   {'b', 'b', 'c', 'a', 'b', 'b', 'c', 0},
                                   {'b', 'b', 0, 0, 0, 0, 0, 0},
                                   {'b', 'b', 'c', 0, 0, 0, 0, 0},
                                   {'b', 'b', 'c', 'a', 0, 0, 0, 0}}));
  model.end_after_c = true;
  const auto early_eos =
      VerifyMlpGreedyCompletions(*executor_, model, **dataset, {}, kVocabulary);
  ASSERT_TRUE(early_eos.ok()) << early_eos.status();
  EXPECT_EQ(early_eos->exact_sentences, 1);
  EXPECT_EQ(early_eos->exact_per_sentence,
            (std::vector<bool>{false, true, false}));
  EXPECT_EQ(early_eos->generated_targets, 6);
  model.nonfinite_token = 'b';
  EXPECT_EQ(
      VerifyMlpGreedyCompletions(*executor_, model, **dataset, {}, kVocabulary)
          .status()
          .code(),
      absl::StatusCode::kDataLoss);
}

TEST_F(MlpProbeTest, GreedyClonePreservesCountsAndStopsAtFirstMismatch) {
  auto model = Model();
  auto dataset = Dataset();
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE(dataset.ok()) << dataset.status();
  const auto capture = CaptureGpt2Mlps(*executor_, **model, **dataset, kWidth,
                                       kFeatures, 1, kVocabulary);
  ASSERT_TRUE(capture.ok()) << capture.status();
  auto clone = Projection(kFeatures);
  auto zero = Projection(kWidth);
  ASSERT_TRUE(clone.ok()) << clone.status();
  ASSERT_TRUE(zero.ok()) << zero.status();
  ASSERT_TRUE(Upload(*executor_, (*clone)->weights()[0],
                     capture->blocks[0].effective_output_weights)
                  .ok());
  ASSERT_TRUE(
      Upload(*executor_, (*clone)->weights()[1], capture->blocks[0].output_bias)
          .ok());
  const auto baseline = VerifyMlpGreedyCompletions(*executor_, **model,
                                                   **dataset, {}, kVocabulary);
  ASSERT_TRUE(baseline.ok()) << baseline.status();
  EXPECT_EQ(baseline->sentences, 3);
  EXPECT_EQ(baseline->exact_sentences, 0);
  EXPECT_EQ(baseline->exact_per_sentence,
            (std::vector<bool>{false, false, false}));
  EXPECT_EQ(baseline->generated_targets, 14);
  const MlpReplacement original{0, MlpSource::kGelu, clone->get()};
  const auto copied = VerifyMlpGreedyCompletions(*executor_, **model, **dataset,
                                                 {&original, 1}, kVocabulary);
  ASSERT_TRUE(copied.ok()) << copied.status();
  EXPECT_EQ(copied->sentences, baseline->sentences);
  EXPECT_EQ(copied->exact_sentences, baseline->exact_sentences);
  EXPECT_EQ(copied->exact_per_sentence, baseline->exact_per_sentence);
  EXPECT_EQ(copied->generated_targets, baseline->generated_targets);
  const MlpReplacement removed{0, MlpSource::kLayerNorm, zero->get()};
  const auto failed = VerifyMlpGreedyCompletions(*executor_, **model, **dataset,
                                                 {&removed, 1}, kVocabulary);
  ASSERT_TRUE(failed.ok()) << failed.status();
  EXPECT_EQ(failed->sentences, 3);
  EXPECT_EQ(failed->exact_sentences, 0);
  EXPECT_EQ(failed->exact_per_sentence,
            (std::vector<bool>{false, false, false}));
  EXPECT_EQ(failed->generated_targets, 3);
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
