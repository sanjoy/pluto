#include "src/llm/recipes/gpt2.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
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
#include "src/llm/layer.h"
#include "src/llm/layer_hooks.h"

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

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(Gpt2Test, ActivationGeneratorRejectsInvalidBlockCounts) {
  auto negative =
      CreateActivationGenerator(*executor_, -1, DataType::BF16, 123);
  EXPECT_EQ(negative.status().code(), absl::StatusCode::kInvalidArgument);

  auto too_many = CreateActivationGenerator(
      *executor_, kGpt2TransformerBlockCount + 1, DataType::BF16, 123);
  EXPECT_EQ(too_many.status().code(), absl::StatusCode::kInvalidArgument);
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
