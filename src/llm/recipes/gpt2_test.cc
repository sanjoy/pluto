#include "src/llm/recipes/gpt2.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer.h"
#include "src/llm/layer_hooks.h"
#include "src/llm/layers/cross_entropy_loss.h"

namespace pluto::llm {
namespace {

// Read-only recording checks the actual dispatch path, not just name()
// accessors. The hook never reads or replaces any activation buffer.
class RecipeNameHooks final : public LayerHooks {
 public:
  absl::Status EnterCombinator(cuda::Executor&,
                               absl::string_view name) override {
    entered.emplace_back(name);
    active.emplace_back(name);
    return absl::OkStatus();
  }

  absl::Status ExitCombinator(cuda::Executor&) override {
    if (active.empty())
      return absl::InternalError("recipe exited a scope that was not entered");
    exited.push_back(active.back());
    active.pop_back();
    return absl::OkStatus();
  }

  absl::Status ActivationHook(cuda::Executor&, absl::string_view name,
                              absl::Span<const ActivationType>,
                              absl::Span<Buffer>) override {
    activations.emplace_back(name);
    return absl::OkStatus();
  }

  std::vector<std::string> entered;
  std::vector<std::string> exited;
  std::vector<std::string> active;
  std::vector<std::string> activations;
};

void ExpectRecipeScopeNames(const RecipeNameHooks& hooks,
                            absl::string_view outer_name, int block_count) {
  std::vector<std::string> expected_entered{std::string(outer_name)};
  std::vector<std::string> expected_exited;
  for (int block = 0; block < block_count; ++block) {
    const std::string block_name = absl::StrCat("transformer_block_", block);
    for (const std::string& name :
         {block_name, std::string("ResidualLayer"), std::string("attention"),
          std::string("ResidualLayer"), std::string("mlp")})
      expected_entered.push_back(name);
    for (const std::string& name :
         {std::string("attention"), std::string("ResidualLayer"),
          std::string("mlp"), std::string("ResidualLayer"), block_name})
      expected_exited.push_back(name);
  }
  expected_exited.emplace_back(outer_name);
  EXPECT_EQ(hooks.entered, expected_entered);
  EXPECT_EQ(hooks.exited, expected_exited);
  EXPECT_TRUE(hooks.active.empty());

  // A combinator publishes its activation immediately after leaving its own
  // scope. Its activation callback must retain the same custom recipe name,
  // while ordinary leaf layers continue to report their class names.
  std::vector<std::string> combinator_activations;
  for (const std::string& name : hooks.activations)
    if (std::find(expected_entered.begin(), expected_entered.end(), name) !=
        expected_entered.end())
      combinator_activations.push_back(name);
  EXPECT_EQ(combinator_activations, expected_exited);
}

class Gpt2Test : public testing::Test {
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
    executor_.reset();
  }

  void ExpectBuffersEqual(const Buffer& first, const Buffer& second) {
    ASSERT_EQ(first.size_bytes(), second.size_bytes());
    auto first_host = cuda::PageLockedHostArray<uint8_t>::Allocate(
        *executor_, first.size_bytes());
    auto second_host = cuda::PageLockedHostArray<uint8_t>::Allocate(
        *executor_, second.size_bytes());
    ASSERT_TRUE(first_host.ok()) << first_host.status();
    ASSERT_TRUE(second_host.ok()) << second_host.status();
    ASSERT_EQ(
        cudaMemcpyAsync(first_host->data(), first.data(), first.size_bytes(),
                        cudaMemcpyDeviceToHost, executor_->stream()),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(second_host->data(), second.data(), second.size_bytes(),
                        cudaMemcpyDeviceToHost, executor_->stream()),
        cudaSuccess);
    ASSERT_TRUE(executor_->Synchronize().ok());
    EXPECT_EQ(std::memcmp(first_host->data(), second_host->data(),
                          first.size_bytes()),
              0);
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST(Gpt2ConfigTest, DefaultsAndSupportedWidths) {
  Gpt2Config defaults;
  EXPECT_EQ(defaults.transformer_block_count, 8);
  EXPECT_EQ(defaults.model_width, 512);
  EXPECT_EQ(defaults.attention_heads, 8);
  EXPECT_EQ(defaults.feed_forward_width, 2048);
  EXPECT_EQ(defaults.vocabulary_size, 50257);
  EXPECT_TRUE(defaults.pad_vocabulary);
  EXPECT_TRUE(defaults.Validate().ok());

  for (int width : {16, 32, 48, 64, 96, 128, 256, 512}) {
    Gpt2Config config{1, width, width / 16, 4 * width};
    EXPECT_TRUE(config.Validate().ok()) << width;
    config.transformer_block_count = 8;
    EXPECT_TRUE(config.Validate().ok()) << width;
    config.transformer_block_count = 16;
    EXPECT_TRUE(config.Validate().ok()) << width;
  }
  // Feed-forward width is deliberately independent of the residual width.
  EXPECT_TRUE((Gpt2Config{2, 32, 1, 80}.Validate().ok()));
  for (int width : {1, 2, 3, 7, 8, 12, 15, 24, 33}) {
    EXPECT_TRUE((Gpt2Config{1, width, 1, 4 * width + 1}.Validate().ok()));
    EXPECT_TRUE((Gpt2Config{1, width, width, 4 * width}.Validate().ok()));
  }
  // Shape validation does not allocate a model or promise enough device memory
  // for very large depths; it must not confuse the default with a hard ceiling.
  for (int depth : {0, 1, 8, 9, 16, std::numeric_limits<int>::max()}) {
    for (int width : {8, 12}) {
      EXPECT_TRUE((Gpt2Config{depth, width, 1, 4 * width}.Validate().ok()))
          << depth << "/" << width;
    }
  }
}

TEST(Gpt2ConfigTest, RejectsInvalidAndOverflowingShapesBeforeAllocating) {
  const int largest_multiple = std::numeric_limits<int>::max() - 15;
  for (const Gpt2Config& config :
       {Gpt2Config{-1, 32, 1, 128},
        Gpt2Config{std::numeric_limits<int>::min(), 32, 1, 128},
        Gpt2Config{1, 0, 1, 128}, Gpt2Config{1, -16, 1, 128},
        Gpt2Config{1, 32, 0, 128}, Gpt2Config{1, 32, -1, 128},
        Gpt2Config{1, 32, 3, 128}, Gpt2Config{1, 32, 64, 128},
        Gpt2Config{1, 32, 1, 0}, Gpt2Config{1, 32, 1, -16},
        Gpt2Config{1, largest_multiple, 1, 128},
        Gpt2Config{1, 32, 1, largest_multiple},
        // Embedding fits, but the packed QKV matrix does not.
        Gpt2Config{1, 32768, 512, 16},
        // Matrices fit, but one 1024-token MLP activation does not.
        Gpt2Config{1, 16, 1, 2'097'152},
        // Activations fit, but the rectangular MLP matrix does not.
        Gpt2Config{1, 16384, 256, 262144}}) {
    EXPECT_EQ(config.Validate().code(), absl::StatusCode::kInvalidArgument)
        << config.model_width << "/" << config.attention_heads << "/"
        << config.feed_forward_width;
  }
}

TEST_F(Gpt2Test, ConfiguredFactoriesPropagateValidationErrors) {
  const Gpt2Config config{1, 32, 3, 128};  // Heads must divide model width.
  auto model = CreateGpt2(*executor_, DataType::BF16, 123, config);
  EXPECT_EQ(model.status().code(), absl::StatusCode::kInvalidArgument);
  auto generator =
      CreateActivationGenerator(*executor_, config, DataType::BF16, 123);
  EXPECT_EQ(generator.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(Gpt2ConfigTest, RejectsInvalidVocabularyAndCompactShapeOverflow) {
  for (bool pad_vocabulary : {false, true}) {
    for (int vocabulary : {0, -1, std::numeric_limits<int>::max(), 2'097'152}) {
      Gpt2Config config{8, 16, 1, 64, vocabulary, pad_vocabulary};
      EXPECT_EQ(config.Validate().code(), absl::StatusCode::kInvalidArgument);
    }
    // A one-token vocabulary no longer limits the model width, so this must
    // reject without overflowing the quadratic QKV parameter calculation.
    Gpt2Config config{
        1, std::numeric_limits<int>::max(), 1, 1, 1, pad_vocabulary};
    EXPECT_EQ(config.Validate().code(), absl::StatusCode::kInvalidArgument);
  }
}

TEST_F(Gpt2Test, ExactVocabularyHas114256ParametersAndTrainsWithPaddedLogits) {
  const Gpt2Config config{8, 16, 1, 64, 4475, false};
  auto model = CreateGpt2(*executor_, DataType::BF16, 123, config);
  ASSERT_TRUE(model.ok()) << model.status();
  const auto weights = (*model)->weights();
  const auto gradients = (*model)->gradients();
  ASSERT_EQ(weights.size(), 101u);
  ASSERT_EQ(gradients.size(), weights.size());
  EXPECT_EQ(weights.front().data(), weights.back().data());
  EXPECT_EQ(gradients.front().data(), gradients.back().data());
  EXPECT_EQ(weights.front().size_bytes(), 4475u * 16 * sizeof(float));
  size_t parameters = 0;
  for (size_t index = 0; index + 1 < weights.size(); ++index) {
    EXPECT_EQ(weights[index].size_bytes(), gradients[index].size_bytes());
    parameters += weights[index].size_bytes() / sizeof(float);
  }
  EXPECT_EQ(parameters, 114256u);

  auto loss = CrossEntropyLossLayer::Create(*executor_, config.vocabulary_size,
                                            DataType::BF16, kGpt2ContextLength);
  ASSERT_TRUE(loss.ok()) << loss.status();
  EXPECT_EQ((*model)->output_types()[0], (*loss)->input_types()[0]);
  EXPECT_EQ((*model)->output_types()[0],
            ActivationType(DataType::FP32, {-2, kGpt2ContextLength, 4480}));
  auto host_tokens = cuda::PageLockedHostArray<int32_t>::Allocate(
      *executor_, kGpt2ContextLength);
  ASSERT_TRUE(host_tokens.ok()) << host_tokens.status();
  for (int row = 0; row < kGpt2ContextLength; ++row)
    (*host_tokens)[row] = row % 2 ? 4474 : 0;
  auto tokens = Buffer::Allocate(*executor_, host_tokens->size_bytes());
  ASSERT_TRUE(tokens.ok()) << tokens.status();
  ASSERT_EQ(
      cudaMemcpyAsync(tokens->data(), host_tokens->data(), tokens->size_bytes(),
                      cudaMemcpyHostToDevice, executor_->stream()),
      cudaSuccess);
  auto forward = (*model)->fwd(*executor_, {*tokens});
  ASSERT_TRUE(forward.ok()) << forward.status();
  ASSERT_EQ(forward->outputs[0].size_bytes(),
            size_t{kGpt2ContextLength} * 4480 * sizeof(float));
  auto loss_forward = (*loss)->fwd(*executor_, {forward->outputs[0], *tokens});
  ASSERT_TRUE(loss_forward.ok()) << loss_forward.status();
  auto loss_backward =
      (*loss)->bwd(*executor_, {}, std::move(loss_forward->state));
  ASSERT_TRUE(loss_backward.ok()) << loss_backward.status();
  auto backward =
      (*model)->bwd(*executor_, *loss_backward, std::move(forward->state));
  ASSERT_TRUE(backward.ok()) << backward.status();
  EXPECT_TRUE(backward->empty());

  auto losses = cuda::PageLockedHostArray<float>::Allocate(*executor_,
                                                           kGpt2ContextLength);
  auto logits = cuda::PageLockedHostArray<float>::Allocate(*executor_, 4480);
  ASSERT_TRUE(losses.ok()) << losses.status();
  ASSERT_TRUE(logits.ok()) << logits.status();
  ASSERT_EQ(cudaMemcpyAsync(losses->data(), loss_forward->outputs[0].data(),
                            losses->size_bytes(), cudaMemcpyDeviceToHost,
                            executor_->stream()),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(logits->data(), forward->outputs[0].data(),
                            logits->size_bytes(), cudaMemcpyDeviceToHost,
                            executor_->stream()),
            cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());
  for (float value : *losses) {
    EXPECT_TRUE(std::isfinite(value));
    EXPECT_GT(value, 0.0f);
  }
  for (int token = 0; token < 4475; ++token)
    EXPECT_TRUE(std::isfinite((*logits)[token]));
  for (int token = 4475; token < 4480; ++token)
    EXPECT_EQ((*logits)[token], -std::numeric_limits<float>::max());

  for (size_t index = 0; index + 1 < gradients.size(); ++index) {
    SCOPED_TRACE(index);
    auto values = cuda::PageLockedHostArray<float>::Allocate(
        *executor_, gradients[index].size_bytes() / sizeof(float));
    ASSERT_TRUE(values.ok()) << values.status();
    ASSERT_EQ(cudaMemcpyAsync(values->data(), gradients[index].data(),
                              values->size_bytes(), cudaMemcpyDeviceToHost,
                              executor_->stream()),
              cudaSuccess);
    ASSERT_TRUE(executor_->Synchronize().ok());
    EXPECT_TRUE(std::all_of(values->begin(), values->end(),
                            [](float value) { return std::isfinite(value); }));
    // Every block's QKV and both uses of the tied embedding participate.
    if (index == 0 || (index >= 4 && index < 98 && (index - 4) % 12 == 0)) {
      EXPECT_TRUE(std::any_of(values->begin(), values->end(),
                              [](float value) { return value != 0.0f; }));
    }
  }

  auto padded_config = config;
  padded_config.pad_vocabulary = true;
  auto padded = CreateGpt2(*executor_, DataType::BF16, 123, padded_config);
  ASSERT_TRUE(padded.ok()) << padded.status();
  EXPECT_EQ((*padded)->weights().front().size_bytes(),
            4480u * 16 * sizeof(float));
  EXPECT_EQ((*padded)->output_types()[0], (*model)->output_types()[0]);
}

TEST_F(Gpt2Test, ExplicitDefaultsMatchLegacyInitialization) {
  auto legacy = CreateGpt2(*executor_, DataType::BF16, 123);
  auto configured = CreateGpt2(*executor_, DataType::BF16, 123, Gpt2Config{});
  ASSERT_TRUE(legacy.ok()) << legacy.status();
  ASSERT_TRUE(configured.ok()) << configured.status();
  ASSERT_EQ((*legacy)->weights().size(), (*configured)->weights().size());
  EXPECT_EQ((*legacy)->input_types()[0], (*configured)->input_types()[0]);
  EXPECT_EQ((*legacy)->output_types()[0], (*configured)->output_types()[0]);
  for (size_t index = 0; index < (*legacy)->weights().size(); ++index) {
    SCOPED_TRACE(index);
    ExpectBuffersEqual((*legacy)->weights()[index],
                       (*configured)->weights()[index]);
  }
}

TEST_F(Gpt2Test, NarrowModelsKeepSeedReproducibilityAndPrefixInitialization) {
  Gpt2Config config{1, 32, 2, 80};
  auto first = CreateGpt2(*executor_, DataType::BF16, 123, config);
  auto repeated = CreateGpt2(*executor_, DataType::BF16, 123, config);
  auto prefix =
      CreateActivationGenerator(*executor_, config, DataType::BF16, 123);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(repeated.ok()) << repeated.status();
  ASSERT_TRUE(prefix.ok()) << prefix.status();
  config.transformer_block_count = 2;
  auto deeper = CreateGpt2(*executor_, DataType::BF16, 123, config);
  ASSERT_TRUE(deeper.ok()) << deeper.status();
  ASSERT_EQ((*prefix)->weights().size(), 14);
  for (size_t index = 0; index < (*first)->weights().size(); ++index)
    ExpectBuffersEqual((*first)->weights()[index],
                       (*repeated)->weights()[index]);
  // The first block's residual-projection scaling must not change with depth.
  for (size_t index = 0; index < (*prefix)->weights().size(); ++index) {
    SCOPED_TRACE(index);
    ExpectBuffersEqual((*first)->weights()[index], (*prefix)->weights()[index]);
    ExpectBuffersEqual((*first)->weights()[index], (*deeper)->weights()[index]);
  }
  EXPECT_EQ((*prefix)->output_types()[0],
            ActivationType(DataType::BF16, {ActivationType::kBatchDimension,
                                            kGpt2ContextLength, 32}));
}

TEST_F(Gpt2Test, SixteenBlocksPreserveEightBlockPrefixInitialization) {
  for (int width : {8, 12}) {
    SCOPED_TRACE(width);
    const Gpt2Config shallow_config{8, width, 1, 4 * width};
    const Gpt2Config deep_config{16, width, 1, 4 * width};
    auto shallow = CreateGpt2(*executor_, DataType::BF16, 123, shallow_config);
    auto deep = CreateGpt2(*executor_, DataType::BF16, 123, deep_config);
    auto prefix =
        CreateActivationGenerator(*executor_, deep_config, DataType::BF16, 123);
    ASSERT_TRUE(shallow.ok()) << shallow.status();
    ASSERT_TRUE(deep.ok()) << deep.status();
    ASSERT_TRUE(prefix.ok()) << prefix.status();
    ASSERT_EQ((*shallow)->weights().size(), 101u);
    ASSERT_EQ((*deep)->weights().size(), 197u);
    ASSERT_EQ((*prefix)->weights().size(), 194u);
    EXPECT_EQ((*prefix)->name(), "gpt2_activation_generator_16_blocks");
    EXPECT_EQ((*prefix)->output_types()[0],
              ActivationType(DataType::BF16, {ActivationType::kBatchDimension,
                                              kGpt2ContextLength, width}));

    // The original eight blocks, including their residual projections, must
    // initialize identically when another eight blocks are appended.
    for (size_t index = 0; index < 98; ++index) {
      SCOPED_TRACE(index);
      ExpectBuffersEqual((*shallow)->weights()[index],
                         (*deep)->weights()[index]);
    }
    for (size_t index = 0; index < 194; ++index) {
      SCOPED_TRACE(index);
      ExpectBuffersEqual((*deep)->weights()[index],
                         (*prefix)->weights()[index]);
    }
    // The final norm moves past the added blocks; it is not part of the
    // matching prefix above. The last exposed buffer is the tied embedding.
    ExpectBuffersEqual((*shallow)->weights()[98], (*deep)->weights()[194]);
    ExpectBuffersEqual((*shallow)->weights()[99], (*deep)->weights()[195]);
  }
}

TEST_F(Gpt2Test, NarrowConfigurationsHaveExpectedUniqueParametersAndTiedHead) {
  for (const Gpt2Config& config :
       {Gpt2Config{1, 3, 1, 13}, Gpt2Config{2, 8, 1, 32},
        Gpt2Config{1, 20, 1, 80}, Gpt2Config{1, 24, 1, 96},
        Gpt2Config{1, 24, 3, 96}, Gpt2Config{0, 16, 1, 64},
        Gpt2Config{1, 32, 2, 80}, Gpt2Config{2, 64, 2, 256},
        Gpt2Config{1, 96, 3, 384}, Gpt2Config{16, 8, 1, 32},
        Gpt2Config{16, 12, 1, 48}, Gpt2Config{1, 128, 8, 512},
        Gpt2Config{1, 256, 4, 1024}}) {
    SCOPED_TRACE(config.transformer_block_count);
    SCOPED_TRACE(config.model_width);
    SCOPED_TRACE(config.attention_heads);
    auto model = CreateGpt2(*executor_, DataType::BF16, 123, config);
    ASSERT_TRUE(model.ok()) << model.status();
    const auto weights = (*model)->weights();
    const auto gradients = (*model)->gradients();
    ASSERT_EQ(weights.size(), 5 + 12 * config.transformer_block_count);
    ASSERT_EQ(gradients.size(), weights.size());
    // Composition exposes both embedding and LM-head handles. Only these
    // first/last aliases should be excluded from the physical parameter count.
    EXPECT_EQ(weights.front().data(), weights.back().data());
    EXPECT_EQ(gradients.front().data(), gradients.back().data());
    size_t unique_parameters = 0;
    for (size_t index = 0; index + 1 < weights.size(); ++index) {
      EXPECT_EQ(weights[index].size_bytes(), gradients[index].size_bytes());
      unique_parameters += weights[index].size_bytes() / sizeof(float);
      for (size_t other = 0; other < index; ++other)
        EXPECT_NE(weights[index].data(), weights[other].data());
    }
    const int64_t width = config.model_width;
    const int64_t ff = config.feed_forward_width;
    // Per block: QKV/projection/MLP matrices plus their biases and two norms.
    const int64_t per_block =
        4 * width * width + 2 * width * ff + 9 * width + ff;
    EXPECT_EQ(unique_parameters,
              (kGpt2PaddedVocabularySize + kGpt2ContextLength + 2) * width +
                  config.transformer_block_count * per_block);
  }
}

TEST_F(Gpt2Test, SmallAndPartialTileModelsRunForwardAndBackward) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (const Gpt2Config& config :
         {Gpt2Config{1, 3, 1, 13}, Gpt2Config{2, 8, 1, 32},
          Gpt2Config{1, 20, 1, 80}, Gpt2Config{1, 24, 1, 96},
          Gpt2Config{1, 24, 3, 96}, Gpt2Config{1, 16, 1, 64},
          Gpt2Config{2, 32, 2, 80}, Gpt2Config{1, 96, 3, 384}}) {
      SCOPED_TRACE(config.model_width);
      SCOPED_TRACE(config.attention_heads);
      SCOPED_TRACE(static_cast<int>(type));
      auto model = CreateGpt2(*executor_, type, 123, config);
      ASSERT_TRUE(model.ok()) << model.status();
      EXPECT_EQ(
          (*model)->input_types()[0],
          ActivationType(DataType::INT32, {ActivationType::kBatchDimension,
                                           kGpt2ContextLength}));
      EXPECT_EQ((*model)->output_types()[0],
                ActivationType(DataType::FP32, {ActivationType::kBatchDimension,
                                                kGpt2ContextLength,
                                                kGpt2PaddedVocabularySize}));
      auto tokens =
          Buffer::Allocate(*executor_, kGpt2ContextLength * sizeof(int32_t));
      ASSERT_TRUE(tokens.ok()) << tokens.status();
      ASSERT_EQ(cudaMemsetAsync(tokens->data(), 0, tokens->size_bytes(),
                                executor_->stream()),
                cudaSuccess);
      auto forward = (*model)->fwd(*executor_, {*tokens});
      ASSERT_TRUE(forward.ok()) << forward.status();
      ASSERT_EQ(forward->outputs.size(), 1);
      EXPECT_EQ(forward->outputs[0].size_bytes(),
                static_cast<size_t>(kGpt2ContextLength) *
                    kGpt2PaddedVocabularySize * sizeof(float));

      // A nonzero derivative on one vocabulary logit exercises the whole
      // backward chain, including the shared embedding accumulator. Zeroing
      // everything would hide width/indexing bugs behind zero gradients.
      auto output_gradient =
          Buffer::Allocate(*executor_, forward->outputs[0].size_bytes());
      auto one = cuda::PageLockedHostArray<float>::Allocate(*executor_, 1);
      ASSERT_TRUE(output_gradient.ok()) << output_gradient.status();
      ASSERT_TRUE(one.ok()) << one.status();
      (*one)[0] = 1.0f;
      ASSERT_EQ(
          cudaMemsetAsync(output_gradient->data(), 0,
                          output_gradient->size_bytes(), executor_->stream()),
          cudaSuccess);
      ASSERT_EQ(cudaMemcpyAsync(static_cast<float*>(output_gradient->data()) +
                                    5 * kGpt2PaddedVocabularySize + 1,
                                one->data(), sizeof(float),
                                cudaMemcpyHostToDevice, executor_->stream()),
                cudaSuccess);
      auto backward = (*model)->bwd(*executor_, {*output_gradient},
                                    std::move(forward->state));
      ASSERT_TRUE(backward.ok()) << backward.status();
      EXPECT_TRUE(
          backward->empty());  // Integer token IDs are not differentiable.

      auto logits = cuda::PageLockedHostArray<float>::Allocate(
          *executor_, kGpt2PaddedVocabularySize);
      ASSERT_TRUE(logits.ok()) << logits.status();
      ASSERT_EQ(cudaMemcpyAsync(logits->data(), forward->outputs[0].data(),
                                logits->size_bytes(), cudaMemcpyDeviceToHost,
                                executor_->stream()),
                cudaSuccess);
      ASSERT_TRUE(executor_->Synchronize().ok());
      for (int token = 0; token < kGpt2VocabularySize; ++token)
        ASSERT_TRUE(std::isfinite((*logits)[token]));
      for (int token = kGpt2VocabularySize; token < kGpt2PaddedVocabularySize;
           ++token)
        EXPECT_EQ((*logits)[token], -std::numeric_limits<float>::max());

      bool has_nonzero_gradient = false;
      for (const Buffer& gradient : (*model)->gradients()) {
        auto host_gradient = cuda::PageLockedHostArray<float>::Allocate(
            *executor_, gradient.size_bytes() / sizeof(float));
        ASSERT_TRUE(host_gradient.ok()) << host_gradient.status();
        ASSERT_EQ(cudaMemcpyAsync(host_gradient->data(), gradient.data(),
                                  gradient.size_bytes(), cudaMemcpyDeviceToHost,
                                  executor_->stream()),
                  cudaSuccess);
        ASSERT_TRUE(executor_->Synchronize().ok());
        for (float value : *host_gradient) {
          ASSERT_TRUE(std::isfinite(value));
          has_nonzero_gradient = has_nonzero_gradient || value != 0.0f;
        }
      }
      EXPECT_TRUE(has_nonzero_gradient);
    }
  }
}

TEST_F(Gpt2Test, SixteenBlockNarrowModelsRunFullContextForwardAndBackward) {
  for (int width : {8, 12}) {
    SCOPED_TRACE(width);
    const Gpt2Config config{16, width, 1, 4 * width};
    auto model = CreateGpt2(*executor_, DataType::BF16, 123, config);
    ASSERT_TRUE(model.ok()) << model.status();
    auto host_tokens = cuda::PageLockedHostArray<int32_t>::Allocate(
        *executor_, kGpt2ContextLength);
    ASSERT_TRUE(host_tokens.ok()) << host_tokens.status();
    for (int position = 0; position < kGpt2ContextLength; ++position)
      (*host_tokens)[position] = (17 * position + position / 11) % 97;
    auto tokens = Buffer::Allocate(*executor_, host_tokens->size_bytes());
    ASSERT_TRUE(tokens.ok()) << tokens.status();
    ASSERT_EQ(cudaMemcpyAsync(tokens->data(), host_tokens->data(),
                              tokens->size_bytes(), cudaMemcpyHostToDevice,
                              executor_->stream()),
              cudaSuccess);
    RecipeNameHooks hooks;
    auto forward = (*model)->fwd(*executor_, {*tokens}, &hooks);
    ASSERT_TRUE(forward.ok()) << forward.status();
    ASSERT_EQ(forward->outputs.size(), 1u);
    ASSERT_EQ(forward->outputs[0].size_bytes(),
              static_cast<size_t>(kGpt2ContextLength) *
                  kGpt2PaddedVocabularySize * sizeof(float));
    ExpectRecipeScopeNames(hooks, "gpt2", 16);

    // A derivative at the last input position traverses every causal key
    // tile, while the early derivative also exercises a partial causal tile.
    auto output_gradient =
        Buffer::Allocate(*executor_, forward->outputs[0].size_bytes());
    auto one = cuda::PageLockedHostArray<float>::Allocate(*executor_, 1);
    ASSERT_TRUE(output_gradient.ok()) << output_gradient.status();
    ASSERT_TRUE(one.ok()) << one.status();
    (*one)[0] = 1.0f;
    ASSERT_EQ(
        cudaMemsetAsync(output_gradient->data(), 0,
                        output_gradient->size_bytes(), executor_->stream()),
        cudaSuccess);
    for (int position : {5, kGpt2ContextLength - 1}) {
      ASSERT_EQ(
          cudaMemcpyAsync(
              static_cast<float*>(output_gradient->data()) +
                  static_cast<size_t>(position) * kGpt2PaddedVocabularySize + 1,
              one->data(), sizeof(float), cudaMemcpyHostToDevice,
              executor_->stream()),
          cudaSuccess);
    }
    auto backward = (*model)->bwd(*executor_, {*output_gradient},
                                  std::move(forward->state));
    ASSERT_TRUE(backward.ok()) << backward.status();
    EXPECT_TRUE(backward->empty());

    auto logits = cuda::PageLockedHostArray<float>::Allocate(
        *executor_, forward->outputs[0].size_bytes() / sizeof(float));
    ASSERT_TRUE(logits.ok()) << logits.status();
    ASSERT_EQ(cudaMemcpyAsync(logits->data(), forward->outputs[0].data(),
                              logits->size_bytes(), cudaMemcpyDeviceToHost,
                              executor_->stream()),
              cudaSuccess);
    ASSERT_TRUE(executor_->Synchronize().ok());
    for (int position = 0; position < kGpt2ContextLength; ++position) {
      SCOPED_TRACE(position);
      const float* row = logits->data() + static_cast<size_t>(position) *
                                              kGpt2PaddedVocabularySize;
      EXPECT_TRUE(std::all_of(row, row + kGpt2VocabularySize, [](float value) {
        return std::isfinite(value);
      }));
      EXPECT_TRUE(std::all_of(row + kGpt2VocabularySize,
                              row + kGpt2PaddedVocabularySize, [](float value) {
                                return value ==
                                       -std::numeric_limits<float>::max();
                              }));
    }

    const auto gradients = (*model)->gradients();
    ASSERT_EQ(gradients.size(), 197u);
    EXPECT_EQ(gradients.front().data(), gradients.back().data());
    for (size_t index = 0; index + 1 < gradients.size(); ++index) {
      SCOPED_TRACE(index);
      const Buffer& gradient = gradients[index];
      auto values = cuda::PageLockedHostArray<float>::Allocate(
          *executor_, gradient.size_bytes() / sizeof(float));
      ASSERT_TRUE(values.ok()) << values.status();
      ASSERT_EQ(cudaMemcpyAsync(values->data(), gradient.data(),
                                gradient.size_bytes(), cudaMemcpyDeviceToHost,
                                executor_->stream()),
                cudaSuccess);
      ASSERT_TRUE(executor_->Synchronize().ok());
      EXPECT_TRUE(std::all_of(values->begin(), values->end(), [](float value) {
        return std::isfinite(value);
      }));
      // Every block's QKV matrix must participate, not merely the tied head.
      // Individual biases may correctly have zero derivatives.
      const bool is_qkv_matrix =
          index >= 4 && index < 194 && (index - 4) % 12 == 0;
      if (index == 0 || is_qkv_matrix) {
        EXPECT_TRUE(std::any_of(values->begin(), values->end(),
                                [](float value) { return value != 0.0f; }));
      }
    }
  }
}

TEST_F(Gpt2Test, ActivationGeneratorRejectsNegativeBlockCounts) {
  auto negative =
      CreateActivationGenerator(*executor_, -1, DataType::BF16, 123);
  EXPECT_EQ(negative.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(Gpt2Test, ZeroBlocksTapsEmbeddingActivations) {
  auto generator =
      CreateActivationGenerator(*executor_, 0, DataType::BF16, 123);
  ASSERT_TRUE(generator.ok()) << generator.status();

  EXPECT_EQ((*generator)->name(), "gpt2_activation_generator_0_blocks");

  // Token and position embeddings are the only trainable tensors before the
  // first transformer block.
  EXPECT_EQ((*generator)->weights().size(), 2u);
  EXPECT_EQ((*generator)->gradients().size(), 2u);
  EXPECT_EQ((*generator)->output_type(), DataType::BF16);
}

TEST_F(Gpt2Test, ActivationSignaturesKeepBatchSeparateFromContext) {
  for (DataType compute : {DataType::FP16, DataType::BF16}) {
    auto generator = CreateActivationGenerator(*executor_, 0, compute, 123);
    ASSERT_TRUE(generator.ok()) << generator.status();
    ASSERT_EQ((*generator)->input_types().size(), 1);
    ASSERT_EQ((*generator)->output_types().size(), 1);
    EXPECT_EQ((*generator)->input_types()[0],
              ActivationType(DataType::INT32, {ActivationType::kBatchDimension,
                                               kGpt2ContextLength}));
    EXPECT_EQ((*generator)->output_types()[0],
              ActivationType(ActivationDataType(compute),
                             {ActivationType::kBatchDimension,
                              kGpt2ContextLength, kGpt2ModelWidth}));
  }
}

TEST_F(Gpt2Test, FullModelProducesPaddedFp32LogitsFromBf16Activations) {
  auto model = CreateGpt2(*executor_, DataType::BF16, 123);
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ((*model)->name(), "gpt2");
  ASSERT_EQ((*model)->input_types().size(), 1);
  ASSERT_EQ((*model)->output_types().size(), 1);
  EXPECT_EQ((*model)->input_types()[0],
            ActivationType(DataType::INT32, {ActivationType::kBatchDimension,
                                             kGpt2ContextLength}));
  // The tied head emits FP32 even though the residual stream uses BF16.
  EXPECT_EQ((*model)->output_types()[0],
            ActivationType(DataType::FP32,
                           {ActivationType::kBatchDimension, kGpt2ContextLength,
                            kGpt2PaddedVocabularySize}));

  auto tokens = Buffer::Allocate(
      *executor_, static_cast<size_t>(kGpt2ContextLength) * sizeof(int));
  ASSERT_TRUE(tokens.ok()) << tokens.status();
  ASSERT_EQ(cudaMemsetAsync(tokens->data(), 0, tokens->size_bytes(),
                            executor_->stream()),
            cudaSuccess);
  RecipeNameHooks hooks;
  auto forward = (*model)->fwd(*executor_, {*tokens}, &hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  ASSERT_EQ(forward->outputs.size(), 1);
  EXPECT_EQ(forward->outputs[0].size_bytes(),
            static_cast<size_t>(kGpt2ContextLength) *
                kGpt2PaddedVocabularySize * sizeof(float));
  ExpectRecipeScopeNames(hooks, "gpt2", kGpt2TransformerBlockCount);
  ASSERT_FALSE(hooks.activations.empty());
  EXPECT_EQ(hooks.activations.front(), "EmbeddingLookupLayer");
  EXPECT_EQ(hooks.activations.back(), "gpt2");
  EXPECT_EQ(std::count(hooks.activations.begin(), hooks.activations.end(),
                       "LanguageModelingHeadLayer"),
            1);
}

TEST_F(Gpt2Test, OneBlockProducesResidualStreamActivations) {
  auto generator =
      CreateActivationGenerator(*executor_, 1, DataType::BF16, 123);
  ASSERT_TRUE(generator.ok()) << generator.status();

  EXPECT_EQ((*generator)->name(), "gpt2_activation_generator_1_blocks");

  // Each block contributes two LayerNorms and four dense projections, with a
  // weight and bias for each. Attention and GELU themselves are stateless.
  EXPECT_EQ((*generator)->weights().size(), 2u + 12u);
  EXPECT_EQ((*generator)->gradients().size(), 2u + 12u);

  auto tokens = Buffer::Allocate(
      *executor_, static_cast<size_t>(kGpt2ContextLength) * sizeof(int));
  ASSERT_TRUE(tokens.ok()) << tokens.status();
  ASSERT_EQ(cudaMemsetAsync(tokens->data(), 0, tokens->size_bytes(),
                            executor_->stream()),
            cudaSuccess);

  BufferVec inputs = {*tokens};
  RecipeNameHooks hooks;
  auto activations = (*generator)->fwd(*executor_, inputs, &hooks);

  ASSERT_TRUE(activations.ok()) << activations.status();
  ASSERT_EQ(activations->outputs.size(), 1u);
  EXPECT_EQ(activations->outputs[0].size_bytes(),
            static_cast<size_t>(kGpt2ContextLength) * kGpt2ModelWidth *
                sizeof(uint16_t));
  ExpectRecipeScopeNames(hooks, "gpt2_activation_generator_1_blocks", 1);
  EXPECT_EQ(std::count(hooks.activations.begin(), hooks.activations.end(),
                       "LanguageModelingHeadLayer"),
            0);
  EXPECT_TRUE(executor_->Synchronize().ok());
}

}  // namespace
}  // namespace pluto::llm
