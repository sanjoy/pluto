#include "src/llm/experiments/memorize_general_facts/mlp_readout.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/adamw_optimizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/layer_hooks.h"
#include "src/llm/layers/norm.h"
#include "src/util/status_macros.h"

namespace pluto::llm::memorize_general_facts {
namespace {

TEST(MlpReadoutParameterBudgetTest, CountsTheEntireSourceTailAfterAttention3) {
  Gpt2Config config{.transformer_block_count = 4,
                    .model_width = 10,
                    .attention_heads = 1,
                    .feed_forward_width = 20,
                    .vocabulary_size = 7,
                    .pad_vocabulary = false,
                    .context_length = 8};
  struct Case {
    int blocks;
    int minimum_width;
    int64_t tail_parameters;
  };
  for (const auto test :
       {Case{3, 22, 470}, Case{4, 66, 1380}, Case{8, 239, 5020}}) {
    SCOPED_TRACE(test.blocks);
    config.transformer_block_count = test.blocks;
    auto budget = ResolveMlpReadoutParameterBudget(config, 1);
    ASSERT_TRUE(budget.ok()) << budget.status();
    EXPECT_EQ(budget->source_tail_parameters, test.tail_parameters);
    EXPECT_EQ(budget->minimum_mlp_width, test.minimum_width);
    EXPECT_EQ(budget->mlp_width, test.minimum_width);
    EXPECT_EQ(budget->mlp_parameters, 21 * test.minimum_width + 10);
    EXPECT_EQ(budget->trainable_parameters, budget->mlp_parameters + 40);
    // Matching the bare two affine layers is deliberately stricter than
    // counting the replacement's two trainable LayerNorms toward its budget.
    EXPECT_GE(budget->mlp_parameters, budget->source_tail_parameters);
    EXPECT_LT(21 * (test.minimum_width - 1) + 10,
              budget->source_tail_parameters);
  }
  config.transformer_block_count = 4;
  auto preserved = ResolveMlpReadoutParameterBudget(config, 150);
  ASSERT_TRUE(preserved.ok()) << preserved.status();
  EXPECT_EQ(preserved->mlp_width, 150);
  EXPECT_EQ(preserved->mlp_parameters, 3160);
  EXPECT_EQ(preserved->trainable_parameters, 3200);
  auto enlarged = ResolveMlpReadoutParameterBudget(config, 65);
  ASSERT_TRUE(enlarged.ok()) << enlarged.status();
  EXPECT_EQ(enlarged->mlp_width, 66);
  EXPECT_EQ(enlarged->mlp_parameters, 1396);
  EXPECT_EQ(enlarged->trainable_parameters, 1436);
}

TEST(MlpReadoutParameterBudgetTest, RejectsInvalidOrUnrepresentableBudgets) {
  Gpt2Config config{.transformer_block_count = 4,
                    .model_width = 10,
                    .attention_heads = 1,
                    .feed_forward_width = 20,
                    .vocabulary_size = 7,
                    .pad_vocabulary = false,
                    .context_length = 8};
  for (const int width : {-1, 0, std::numeric_limits<int>::max()})
    EXPECT_FALSE(ResolveMlpReadoutParameterBudget(config, width).ok());
  config.transformer_block_count = 2;
  EXPECT_FALSE(ResolveMlpReadoutParameterBudget(config, 150).ok());
  config.transformer_block_count = std::numeric_limits<int>::max();
  EXPECT_FALSE(ResolveMlpReadoutParameterBudget(config, 150).ok());

  // Each tensor fits the backend limit, but adding the suffix's parameters
  // over this many blocks would overflow int64_t without checked arithmetic.
  config.model_width = 26754;
  config.feed_forward_width =
      std::numeric_limits<int>::max() / config.model_width;
  config.vocabulary_size = 1;
  config.context_length = 1;
  ASSERT_TRUE(config.Validate().ok());
  EXPECT_FALSE(ResolveMlpReadoutParameterBudget(config, 150).ok());

  // Even a representable inferred width must satisfy the activation limits;
  // validating only the original narrow model is insufficient.
  config.transformer_block_count = 10000;
  config.model_width = 1;
  config.feed_forward_width = 1;
  config.context_length = 1000000;
  ASSERT_TRUE(config.Validate().ok());
  EXPECT_FALSE(ResolveMlpReadoutParameterBudget(config, 1).ok());
}

TEST(MlpReadoutParameterBudgetTest, MatchesAllBlocksOrPreservesExactWidth) {
  const Gpt2Config config{.transformer_block_count = 4,
                          .model_width = 10,
                          .attention_heads = 1,
                          .feed_forward_width = 20,
                          .vocabulary_size = 7,
                          .pad_vocabulary = false,
                          .context_length = 8};
  const int minimum_widths[] = {66, 33, 22, 16, 13};
  for (int depth = 1; depth <= 5; ++depth) {
    SCOPED_TRACE(depth);
    auto matched = ResolveMlpReadoutParameterBudget(config, 1, depth);
    auto exact = ResolveMlpReadoutParameterBudget(config, 1, depth, false);
    ASSERT_TRUE(matched.ok()) << matched.status();
    ASSERT_TRUE(exact.ok()) << exact.status();
    EXPECT_EQ(matched->minimum_mlp_width, minimum_widths[depth - 1]);
    EXPECT_EQ(matched->mlp_width, minimum_widths[depth - 1]);
    EXPECT_EQ(matched->source_tail_parameters, 1380);
    EXPECT_EQ(matched->mlp_parameters, depth * (21 * matched->mlp_width + 10));
    EXPECT_EQ(matched->trainable_parameters,
              depth * (21 * matched->mlp_width + 30) + 20);
    EXPECT_GE(matched->mlp_parameters, 1380);
    EXPECT_LT(depth * (21 * (matched->mlp_width - 1) + 10), 1380);
    EXPECT_EQ(exact->minimum_mlp_width, matched->minimum_mlp_width);
    EXPECT_EQ(exact->mlp_width, 1);
    EXPECT_EQ(exact->source_tail_parameters, 1380);
    EXPECT_EQ(exact->mlp_parameters, depth * 31);
    EXPECT_EQ(exact->trainable_parameters, depth * 51 + 20);
    EXPECT_LT(exact->mlp_parameters, exact->source_tail_parameters);
  }
  // Enough narrow blocks can exceed the budget even at the minimum legal
  // width; rounding must never yield a zero or negative hidden dimension.
  auto deep = ResolveMlpReadoutParameterBudget(config, 1, 150);
  ASSERT_TRUE(deep.ok()) << deep.status();
  EXPECT_EQ(deep->minimum_mlp_width, 1);
  EXPECT_EQ(deep->mlp_width, 1);
  EXPECT_EQ(deep->mlp_parameters, 4650);
}

TEST(MlpReadoutParameterBudgetTest, RejectsInvalidDepthAndAggregateOverflow) {
  Gpt2Config config{.transformer_block_count = 4,
                    .model_width = 1,
                    .attention_heads = 1,
                    .feed_forward_width = 1,
                    .vocabulary_size = 1,
                    .pad_vocabulary = false,
                    .context_length = 1};
  for (int depth : {0, -1, std::numeric_limits<int>::min()})
    for (bool match : {false, true})
      EXPECT_EQ(ResolveMlpReadoutParameterBudget(config, 1, depth, match)
                    .status()
                    .code(),
                absl::StatusCode::kInvalidArgument);
  // Every individual FC tensor fits the backend, but their aggregate count
  // across this many blocks exceeds int64_t. Reject before allocating any.
  const int limit = std::numeric_limits<int>::max();
  auto valid = ResolveMlpReadoutParameterBudget(config, limit, 1, false);
  ASSERT_TRUE(valid.ok()) << valid.status();
  EXPECT_EQ(valid->mlp_width, limit);
  for (bool match : {false, true}) {
    auto overflow =
        ResolveMlpReadoutParameterBudget(config, limit, limit, match);
    EXPECT_EQ(overflow.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(overflow.status().message().find("parameter count overflows"),
              std::string::npos);
  }
}

class MlpReadoutTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
    auto source = CreateGpt2(*executor_, DataType::BF16, 17, config_);
    ASSERT_TRUE(source.ok()) << source.status();
    source_ = std::move(*source);
  }
  void TearDown() override {
    if (!executor_)
      return;
    EXPECT_TRUE(executor_->Synchronize().ok());
  }

  template <class T>
  absl::StatusOr<Buffer> Upload(const std::vector<T>& values) {
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<T>::CopyFrom(*executor_, values));
    ASSIGN_OR_RETURN(auto device,
                     Buffer::Allocate(*executor_, host.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(device.data(), host.data(), host.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "upload MLP readout test input"));
    return device;
  }

  absl::StatusOr<std::vector<std::vector<uint8_t>>> Snapshot(
      absl::Span<const Buffer> buffers) {
    std::vector<std::vector<uint8_t>> result;
    for (const auto& buffer : buffers) {
      ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<uint8_t>::Allocate(
                                      *executor_, buffer.size_bytes()));
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(host.data(), buffer.data(), host.size_bytes(),
                          cudaMemcpyDeviceToHost, executor_->stream()),
          "snapshot MLP readout bytes"));
      RETURN_IF_ERROR(executor_->Synchronize());
      result.emplace_back(host.begin(), host.end());
    }
    return result;
  }

  absl::Status Fill(const Buffer& buffer, float base) {
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<float>::Allocate(
                                    *executor_, buffer.size_bytes() / 4));
    for (size_t i = 0; i < host.size(); ++i)
      host[i] = base + static_cast<float>(i) / 1024;
    return cuda::CudaStatus(
        cudaMemcpyAsync(buffer.data(), host.data(), host.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "fill distinctive MLP readout tensor");
  }

  absl::StatusOr<Buffer> HiddenInput() {
    const uint16_t pattern[] = {0x3f00, 0x3f80, 0x4000, 0xbf00, 0xbf80,
                                0x4040, 0x3e80, 0x4080, 0xc000, 0xbf40};
    std::vector<uint16_t> values(config_.context_length * config_.model_width);
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = pattern[(i + i / config_.model_width) % 10];
    return Upload(values);
  }

  void ExpectFinite(const std::vector<uint8_t>& bytes) {
    ASSERT_EQ(bytes.size() % sizeof(float), 0u);
    for (size_t i = 0; i < bytes.size(); i += sizeof(float)) {
      float value;
      std::memcpy(&value, bytes.data() + i, sizeof(value));
      EXPECT_TRUE(std::isfinite(value)) << i;
    }
  }

  const Gpt2Config config_{.transformer_block_count = 4,
                           .model_width = 10,
                           .attention_heads = 1,
                           .feed_forward_width = 20,
                           .vocabulary_size = 7,
                           .pad_vocabulary = false,
                           .context_length = 8};
  std::unique_ptr<cuda::Executor> executor_;
  std::unique_ptr<ComposedLayer> source_;
};

TEST_F(MlpReadoutTest, FreshWidth150Has3200ParametersAndIndependentFrozenHead) {
  const size_t final_ln = 2 + 12 * config_.transformer_block_count;
  ASSERT_TRUE(Fill(source_->weights()[final_ln], 0.5f).ok());
  ASSERT_TRUE(Fill(source_->weights()[final_ln + 1], -0.25f).ok());
  auto source_before = Snapshot(source_->weights());
  auto readout = CreateMlpReadout(*executor_, *source_, config_, 150, 3);
  ASSERT_TRUE(source_before.ok()) << source_before.status();
  ASSERT_TRUE(readout.ok()) << readout.status();
  ASSERT_NE(readout->trainable, nullptr);
  EXPECT_EQ(readout->model->name(), "mlp_readout");
  ASSERT_EQ(readout->trainable->weights().size(), 8u);
  ASSERT_EQ(readout->model->weights().size(), 9u);
  const size_t extents[] = {10, 10, 1500, 150, 1500, 10, 10, 10};
  size_t parameters = 0;
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(readout->trainable->weights()[i].size_bytes(), 4 * extents[i]);
    EXPECT_EQ(readout->trainable->gradients()[i].size_bytes(), 4 * extents[i]);
    EXPECT_EQ(readout->trainable->weights()[i].data(),
              readout->model->weights()[i].data());
    parameters += extents[i];
  }
  EXPECT_EQ(parameters, 3200u);
  auto copied = Snapshot(readout->model->weights());
  ASSERT_TRUE(copied.ok()) << copied.status();
  EXPECT_EQ((*copied)[6], (*source_before)[final_ln]);
  EXPECT_EQ((*copied)[7], (*source_before)[final_ln + 1]);
  EXPECT_EQ((*copied)[8], (*source_before)[0]);
  EXPECT_EQ(readout->embedding->weight().data(),
            readout->model->weights()[8].data());
  for (const auto& weight : readout->model->weights())
    for (const auto& original : source_->weights())
      EXPECT_NE(weight.data(), original.data());
  for (const auto& bytes : *copied)
    ExpectFinite(bytes);
  for (size_t index : {0u, 1u, 3u, 5u})
    for (size_t offset = 0; offset < (*copied)[index].size(); offset += 4) {
      float value;
      std::memcpy(&value, (*copied)[index].data() + offset, sizeof(value));
      EXPECT_EQ(value, index == 0 ? 1.0f : 0.0f);
    }
  auto source_after = Snapshot(source_->weights());
  ASSERT_TRUE(source_after.ok());
  EXPECT_EQ(*source_before, *source_after);
  source_.reset();  // No source lifetime dependency in the readout.
  auto input = HiddenInput();
  ASSERT_TRUE(input.ok()) << input.status();
  EXPECT_TRUE(readout->model->fwd(*executor_, {&*input, 1}).ok());
}

TEST_F(MlpReadoutTest, AutoSizedBudgetMatchesActualSourceAndReadoutTensors) {
  for (const int blocks : {3, 4, 8}) {
    SCOPED_TRACE(blocks);
    auto config = config_;
    config.transformer_block_count = blocks;
    auto source = CreateGpt2(*executor_, DataType::BF16, 17, config);
    ASSERT_TRUE(source.ok()) << source.status();
    auto readout = CreateMlpReadout(*executor_, **source, config, 1, 3);
    ASSERT_TRUE(readout.ok()) << readout.status();
    const auto source_weights = (*source)->weights();
    int64_t source_tail_parameters = 0;
    // Two embedding tensors, two full blocks, and the six attention tensors
    // precede this boundary. The last tensor is the shared, frozen LM head.
    for (size_t i = 2 + 2 * 12 + 6; i + 1 < source_weights.size(); ++i)
      source_tail_parameters += source_weights[i].size_bytes() / sizeof(float);
    EXPECT_EQ(source_tail_parameters,
              readout->parameter_budget.source_tail_parameters);
    int64_t trainable_parameters = 0;
    int64_t mlp_parameters = 0;
    const auto weights = readout->trainable->weights();
    ASSERT_EQ(weights.size(), 8u);
    for (size_t i = 0; i < weights.size(); ++i) {
      const auto count = weights[i].size_bytes() / sizeof(float);
      trainable_parameters += count;
      if (i >= 2 && i < 6)
        mlp_parameters += count;
    }
    EXPECT_EQ(trainable_parameters,
              readout->parameter_budget.trainable_parameters);
    EXPECT_EQ(mlp_parameters, readout->parameter_budget.mlp_parameters);
    EXPECT_GE(mlp_parameters, source_tail_parameters);
    EXPECT_EQ(weights[3].size_bytes() / sizeof(float),
              static_cast<size_t>(readout->parameter_budget.mlp_width));
  }
}

TEST_F(MlpReadoutTest, ExplicitDepthOnePreservesDefaultWeightsAndOutput) {
  for (int width : {1, 150}) {
    SCOPED_TRACE(width);
    auto original =
        CreateMlpReadout(*executor_, *source_, config_, width, 3);
    auto explicit_depth =
        CreateMlpReadout(*executor_, *source_, config_, width, 3, 1, true);
    ASSERT_TRUE(original.ok()) << original.status();
    ASSERT_TRUE(explicit_depth.ok()) << explicit_depth.status();
    auto original_weights = Snapshot(original->model->weights());
    auto explicit_weights = Snapshot(explicit_depth->model->weights());
    ASSERT_TRUE(original_weights.ok()) << original_weights.status();
    ASSERT_TRUE(explicit_weights.ok()) << explicit_weights.status();
    EXPECT_EQ(*original_weights, *explicit_weights);
    EXPECT_EQ(original->parameter_budget.mlp_width,
              explicit_depth->parameter_budget.mlp_width);
    EXPECT_EQ(original->parameter_budget.trainable_parameters,
              explicit_depth->parameter_budget.trainable_parameters);
    auto input = HiddenInput();
    ASSERT_TRUE(input.ok()) << input.status();
    auto original_output = original->model->fwd(*executor_, {&*input, 1});
    auto explicit_output = explicit_depth->model->fwd(*executor_, {&*input, 1});
    ASSERT_TRUE(original_output.ok()) << original_output.status();
    ASSERT_TRUE(explicit_output.ok()) << explicit_output.status();
    auto original_bytes = Snapshot(original_output->outputs);
    auto explicit_bytes = Snapshot(explicit_output->outputs);
    ASSERT_TRUE(original_bytes.ok()) << original_bytes.status();
    ASSERT_TRUE(explicit_bytes.ok()) << explicit_bytes.status();
    EXPECT_EQ(*original_bytes, *explicit_bytes);
  }
}

TEST_F(MlpReadoutTest, StackedBlocksHaveExactShapesAndIndependentSeeds) {
  const size_t final_ln = 2 + 12 * config_.transformer_block_count;
  ASSERT_TRUE(Fill(source_->weights()[final_ln], 0.5f).ok());
  ASSERT_TRUE(Fill(source_->weights()[final_ln + 1], -0.25f).ok());
  auto source_before = Snapshot(source_->weights());
  ASSERT_TRUE(source_before.ok()) << source_before.status();
  for (int depth = 2; depth <= 5; ++depth) {
    SCOPED_TRACE(depth);
    auto readout =
        CreateMlpReadout(*executor_, *source_, config_, 7, 3, depth, false);
    ASSERT_TRUE(readout.ok()) << readout.status();
    const auto weights = readout->trainable->weights();
    const auto gradients = readout->trainable->gradients();
    ASSERT_EQ(weights.size(), 6u * depth + 2);
    ASSERT_EQ(gradients.size(), weights.size());
    ASSERT_EQ(readout->model->weights().size(), weights.size() + 1);
    EXPECT_EQ(readout->parameter_budget.mlp_width, 7);
    EXPECT_EQ(readout->parameter_budget.mlp_parameters, depth * 157);
    EXPECT_EQ(readout->parameter_budget.trainable_parameters, depth * 177 + 20);
    auto bytes = Snapshot(readout->model->weights());
    ASSERT_TRUE(bytes.ok()) << bytes.status();
    const size_t block_extents[] = {10, 10, 70, 7, 70, 10};
    int64_t parameter_count = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
      const size_t extent = i < 6u * depth ? block_extents[i % 6] : 10;
      EXPECT_EQ(weights[i].size_bytes(), sizeof(float) * extent);
      EXPECT_EQ(gradients[i].size_bytes(), sizeof(float) * extent);
      EXPECT_EQ(weights[i].data(), readout->model->weights()[i].data());
      parameter_count += extent;
      ExpectFinite((*bytes)[i]);
    }
    EXPECT_EQ(parameter_count, readout->parameter_budget.trainable_parameters);
    for (int block = 0; block < depth; ++block) {
      SCOPED_TRACE(block);
      // A block has exactly the old one-block initialization at its own seed,
      // including identity input LN, zero biases, and the fixed .1 FC2 scale.
      auto single = CreateMlpReadout(*executor_, *source_, config_, 7,
                                     3 + 2 * block, 1, false);
      ASSERT_TRUE(single.ok()) << single.status();
      auto single_bytes = Snapshot(single->trainable->weights());
      ASSERT_TRUE(single_bytes.ok()) << single_bytes.status();
      for (size_t i = 0; i < 6; ++i)
        EXPECT_EQ((*bytes)[6 * block + i], (*single_bytes)[i]);
      if (block > 0) {
        EXPECT_NE((*bytes)[2], (*bytes)[6 * block + 2]);
        EXPECT_NE((*bytes)[4], (*bytes)[6 * block + 4]);
      }
    }
    EXPECT_EQ((*bytes)[6 * depth], (*source_before)[final_ln]);
    EXPECT_EQ((*bytes)[6 * depth + 1], (*source_before)[final_ln + 1]);
    EXPECT_EQ(bytes->back(), source_before->front());
    for (const auto& weight : readout->model->weights())
      for (const auto& original : source_->weights())
        EXPECT_NE(weight.data(), original.data());
  }
  auto source_after = Snapshot(source_->weights());
  ASSERT_TRUE(source_after.ok()) << source_after.status();
  EXPECT_EQ(*source_before, *source_after);
}

TEST_F(MlpReadoutTest,
       RealBackwardUpdatesEveryBlockAndLeavesSourceAndHeadFrozen) {
  for (int depth = 1; depth <= 5; ++depth) {
    SCOPED_TRACE(depth);
    auto readout =
        CreateMlpReadout(*executor_, *source_, config_, 150, 3, depth, false);
    ASSERT_TRUE(readout.ok()) << readout.status();
    auto source_before = Snapshot(source_->weights());
    auto before = Snapshot(readout->model->weights());
    auto optimizer =
        AdamWOptimizer::Create(*executor_, *readout->trainable,
                               {.learning_rate = 0.001f, .weight_decay = 0});
    ASSERT_TRUE(source_before.ok()) << source_before.status();
    ASSERT_TRUE(before.ok()) << before.status();
    ASSERT_TRUE(optimizer.ok()) << optimizer.status();
    const size_t trainable_tensors = 6 * depth + 2;
    EXPECT_EQ((*optimizer)->parameter_tensor_count(), trainable_tensors);
    ASSERT_TRUE((*optimizer)->ZeroGrad().ok());
    auto input = HiddenInput();
    ASSERT_TRUE(input.ok()) << input.status();
    auto output = readout->model->fwd(*executor_, {&*input, 1});
    ASSERT_TRUE(output.ok()) << output.status();
    ASSERT_EQ(output->outputs.size(), 1u);
    EXPECT_EQ(output->outputs[0].size_bytes(), 8 * 16 * sizeof(float));
    auto logit_bytes = Snapshot(output->outputs);
    ASSERT_TRUE(logit_bytes.ok());
    for (int row = 0; row < config_.context_length; ++row)
      for (int column = 0; column < config_.vocabulary_size; ++column) {
        float value;
        std::memcpy(&value,
                    logit_bytes->front().data() + (row * 16 + column) * 4, 4);
        EXPECT_TRUE(std::isfinite(value));
      }
    std::vector<float> upstream_values(8 * 16, 0);
    for (size_t i = 0; i < upstream_values.size(); ++i)
      if (i % 16 < 7)
        upstream_values[i] = (static_cast<int>(i % 5) - 2) / 8.0f;
    auto upstream = Upload(upstream_values);
    ASSERT_TRUE(upstream.ok()) << upstream.status();
    auto input_gradient = readout->model->bwd(*executor_, {&*upstream, 1},
                                              std::move(output->state));
    ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();
    ASSERT_EQ(input_gradient->size(), 1u);
    EXPECT_EQ(input_gradient->front().size_bytes(), 8 * 10 * sizeof(float));
    auto gradients = Snapshot(readout->trainable->gradients());
    ASSERT_TRUE(gradients.ok()) << gradients.status();
    ASSERT_EQ(gradients->size(), trainable_tensors);
    for (const auto& bytes : *gradients) {
      ExpectFinite(bytes);
      bool nonzero = false;
      for (size_t offset = 0; offset < bytes.size(); offset += sizeof(float)) {
        float value;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        nonzero |= value != 0;
      }
      EXPECT_TRUE(nonzero);
    }
    auto input_gradient_bytes = Snapshot(*input_gradient);
    ASSERT_TRUE(input_gradient_bytes.ok()) << input_gradient_bytes.status();
    ExpectFinite(input_gradient_bytes->front());
    ASSERT_TRUE((*optimizer)->ApplyStep().ok());
    auto after = Snapshot(readout->model->weights());
    auto source_after = Snapshot(source_->weights());
    ASSERT_TRUE(after.ok()) << after.status();
    ASSERT_TRUE(source_after.ok()) << source_after.status();
    for (size_t i = 0; i < trainable_tensors; ++i)
      EXPECT_NE((*before)[i], (*after)[i]) << "trainable tensor " << i;
    EXPECT_EQ(before->back(), after->back());
    EXPECT_EQ(*source_before, *source_after);
  }
}

TEST_F(MlpReadoutTest, ProductionStackTrainsAll15920Parameters) {
  // Exercise the deployed batch, context, and compact vocabulary dimensions,
  // including the unaligned vocabulary and the real third-attention capture.
  auto config = config_;
  config.context_length = 27;
  config.vocabulary_size = 4475;
  constexpr int kBatch = 32;
  constexpr int kDepth = 5;
  constexpr int kWidth = 150;
  constexpr size_t kTrainableTensors = 32;
  const int rows = kBatch * config.context_length;
  const int columns = (config.vocabulary_size + 15) / 16 * 16;
  auto source = CreateGpt2(*executor_, DataType::BF16, 17, config);
  ASSERT_TRUE(source.ok()) << source.status();
  auto source_before = Snapshot((*source)->weights());
  ASSERT_TRUE(source_before.ok()) << source_before.status();
  auto readout =
      CreateMlpReadout(*executor_, **source, config, kWidth, 3, kDepth, false);
  ASSERT_TRUE(readout.ok()) << readout.status();
  EXPECT_EQ(readout->parameter_budget.mlp_width, kWidth);
  EXPECT_EQ(readout->parameter_budget.mlp_parameters, 15800);
  EXPECT_EQ(readout->parameter_budget.trainable_parameters, 15920);
  ASSERT_EQ(readout->trainable->weights().size(), kTrainableTensors);
  ASSERT_EQ(readout->model->weights().size(), kTrainableTensors + 1);
  size_t parameter_count = 0;
  for (const auto& weight : readout->trainable->weights())
    parameter_count += weight.size_bytes() / sizeof(float);
  EXPECT_EQ(parameter_count, 15920u);
  auto before = Snapshot(readout->model->weights());
  ASSERT_TRUE(before.ok()) << before.status();
  auto optimizer =
      AdamWOptimizer::Create(*executor_, *readout->trainable,
                             {.learning_rate = 0.001f, .weight_decay = 0});
  ASSERT_TRUE(optimizer.ok()) << optimizer.status();
  EXPECT_EQ((*optimizer)->parameter_tensor_count(), kTrainableTensors);
  ASSERT_TRUE((*optimizer)->ZeroGrad().ok());
  std::vector<int32_t> token_values(rows);
  for (int row = 0; row < rows; ++row)
    token_values[row] = (31 * row + 7) % config.vocabulary_size;
  auto tokens = Upload(token_values);
  ASSERT_TRUE(tokens.ok()) << tokens.status();
  auto captured = CaptureThirdAttention(*executor_, **source, *tokens);
  ASSERT_TRUE(captured.ok()) << captured.status();
  auto output = readout->model->fwd(*executor_, {&captured->hidden, 1});
  ASSERT_TRUE(output.ok()) << output.status();
  ASSERT_EQ(output->outputs.size(), 1u);
  ASSERT_EQ(output->outputs[0].size_bytes(), rows * columns * sizeof(float));
  auto logits = Snapshot(output->outputs);
  ASSERT_TRUE(logits.ok()) << logits.status();
  for (int row = 0; row < rows; ++row)
    for (int column = 0; column < config.vocabulary_size; ++column) {
      float value;
      std::memcpy(
          &value,
          logits->front().data() + (row * columns + column) * sizeof(float),
          sizeof(value));
      ASSERT_TRUE(std::isfinite(value)) << row << ":" << column;
    }
  std::vector<float> upstream_values(rows * columns, 0);
  for (int row = 0; row < rows; ++row) {
    upstream_values[row * columns + token_values[row]] = 0.5f / rows;
    upstream_values[row * columns + (token_values[row] + 1) %
                                        config.vocabulary_size] = -0.5f / rows;
  }
  auto upstream = Upload(upstream_values);
  ASSERT_TRUE(upstream.ok()) << upstream.status();
  auto input_gradient = readout->model->bwd(*executor_, {&*upstream, 1},
                                            std::move(output->state));
  ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();
  ASSERT_EQ(input_gradient->size(), 1u);
  EXPECT_EQ(input_gradient->front().size_bytes(),
            rows * config.model_width * sizeof(float));
  auto input_bytes = Snapshot(*input_gradient);
  ASSERT_TRUE(input_bytes.ok()) << input_bytes.status();
  ExpectFinite(input_bytes->front());
  auto gradients = Snapshot(readout->trainable->gradients());
  ASSERT_TRUE(gradients.ok()) << gradients.status();
  ASSERT_EQ(gradients->size(), kTrainableTensors);
  for (size_t index = 0; index < gradients->size(); ++index) {
    SCOPED_TRACE(index);
    const auto& bytes = (*gradients)[index];
    ExpectFinite(bytes);
    bool nonzero = false;
    for (size_t offset = 0; offset < bytes.size(); offset += sizeof(float)) {
      float value;
      std::memcpy(&value, bytes.data() + offset, sizeof(value));
      nonzero |= value != 0;
    }
    EXPECT_TRUE(nonzero);
  }
  ASSERT_TRUE((*optimizer)->ApplyStep().ok());
  auto after = Snapshot(readout->model->weights());
  auto source_after = Snapshot((*source)->weights());
  ASSERT_TRUE(after.ok()) << after.status();
  ASSERT_TRUE(source_after.ok()) << source_after.status();
  for (size_t index = 0; index < kTrainableTensors; ++index)
    EXPECT_NE((*before)[index], (*after)[index]) << index;
  EXPECT_EQ(before->back(), after->back());
  EXPECT_EQ(*source_before, *source_after);
}

TEST_F(MlpReadoutTest, StrictCheckpointRoundTripReproducesOutput) {
  for (int depth : {1, 5}) {
    SCOPED_TRACE(depth);
    auto original =
        CreateMlpReadout(*executor_, *source_, config_, 150, 3, depth, false);
    auto restored =
        CreateMlpReadout(*executor_, *source_, config_, 150, 91, depth, false);
    auto wrong =
        CreateMlpReadout(*executor_, *source_, config_, 149, 3, depth, false);
    auto wrong_depth = CreateMlpReadout(*executor_, *source_, config_, 150, 3,
                                        depth == 1 ? 5 : 1, false);
    ASSERT_TRUE(original.ok()) << original.status();
    ASSERT_TRUE(restored.ok()) << restored.status();
    ASSERT_TRUE(wrong.ok()) << wrong.status();
    ASSERT_TRUE(wrong_depth.ok()) << wrong_depth.status();
    ASSERT_TRUE(Fill(original->trainable->weights()[6 * depth], 0.5f).ok());
    ASSERT_TRUE(
        Fill(original->trainable->weights()[6 * depth + 1], -0.25f).ok());
    const auto directory =
        std::filesystem::path(testing::TempDir()) /
        ("mlp-readout-round-trip-" + std::to_string(depth));
    ASSERT_TRUE(
        WriteToDirectory(*executor_, *original->trainable, directory).ok());
    ASSERT_TRUE(ReadFromDirectory(*executor_, *restored->trainable, directory,
                                  /*allow_prefix=*/false)
                    .ok());
    EXPECT_EQ(ReadFromDirectory(*executor_, *wrong->trainable, directory,
                                /*allow_prefix=*/false)
                  .code(),
              absl::StatusCode::kDataLoss);
    EXPECT_EQ(ReadFromDirectory(*executor_, *wrong_depth->trainable, directory,
                                /*allow_prefix=*/false)
                  .code(),
              absl::StatusCode::kDataLoss);
    auto original_weights = Snapshot(original->model->weights());
    auto restored_weights = Snapshot(restored->model->weights());
    ASSERT_TRUE(original_weights.ok());
    ASSERT_TRUE(restored_weights.ok());
    EXPECT_EQ(*original_weights, *restored_weights);
    auto input = HiddenInput();
    ASSERT_TRUE(input.ok());
    auto first = original->model->fwd(*executor_, {&*input, 1});
    auto second = restored->model->fwd(*executor_, {&*input, 1});
    ASSERT_TRUE(first.ok()) << first.status();
    ASSERT_TRUE(second.ok()) << second.status();
    auto first_bytes = Snapshot(first->outputs);
    auto second_bytes = Snapshot(second->outputs);
    ASSERT_TRUE(first_bytes.ok());
    ASSERT_TRUE(second_bytes.ok());
    EXPECT_EQ(*first_bytes, *second_bytes);
  }
}

TEST_F(MlpReadoutTest, CapturesThirdAttentionNotSecondOrThirdMlp) {
  auto tokens = Upload<int32_t>({0, 1, 2, 3, 4, 5, 6, 0});
  ASSERT_TRUE(tokens.ok()) << tokens.status();
  std::vector<std::string> scopes;
  std::vector<Buffer> second_residuals;
  std::vector<Buffer> third_residuals;
  LayerHooks hooks;
  hooks.enter_combinator = [&](auto&, auto name) {
    scopes.emplace_back(name);
    return absl::OkStatus();
  };
  hooks.exit_combinator = [&](auto&) {
    scopes.pop_back();
    return absl::OkStatus();
  };
  hooks.activation_hook = [&](auto&, auto name, auto, auto buffers) {
    if (name == "ResidualLayer" && scopes.size() == 2 && scopes[0] == "gpt2") {
      if (scopes[1] == "transformer_block_1")
        second_residuals.push_back(buffers[0]);
      if (scopes[1] == "transformer_block_2")
        third_residuals.push_back(buffers[0]);
    }
    return absl::OkStatus();
  };
  auto forward = source_->fwd(*executor_, {&*tokens, 1}, &hooks);
  auto captured = CaptureThirdAttention(*executor_, *source_, *tokens);
  ASSERT_TRUE(forward.ok()) << forward.status();
  ASSERT_TRUE(captured.ok()) << captured.status();
  EXPECT_TRUE(scopes.empty());
  ASSERT_EQ(second_residuals.size(), 2u);
  ASSERT_EQ(third_residuals.size(), 2u);
  auto second_bytes = Snapshot(second_residuals);
  auto third_bytes = Snapshot(third_residuals);
  auto capture_bytes = Snapshot({&captured->hidden, 1});
  auto full_logits = Snapshot(forward->outputs);
  auto captured_logits = Snapshot({&captured->logits, 1});
  ASSERT_TRUE(second_bytes.ok());
  ASSERT_TRUE(third_bytes.ok());
  ASSERT_TRUE(capture_bytes.ok());
  ASSERT_TRUE(full_logits.ok());
  ASSERT_TRUE(captured_logits.ok());
  EXPECT_EQ(capture_bytes->front(), third_bytes->front());
  EXPECT_NE(capture_bytes->front(), second_bytes->front());
  EXPECT_NE(capture_bytes->front(), third_bytes->back());
  EXPECT_EQ(*full_logits, *captured_logits);
  // Altering a later block must affect only the baseline, never captured x.
  ASSERT_TRUE(Fill(source_->weights()[2 + 12 * 3 + 11], 0.5f).ok());
  auto changed = CaptureThirdAttention(*executor_, *source_, *tokens);
  ASSERT_TRUE(changed.ok()) << changed.status();
  auto changed_hidden = Snapshot({&changed->hidden, 1});
  auto changed_logits = Snapshot({&changed->logits, 1});
  ASSERT_TRUE(changed_hidden.ok());
  ASSERT_TRUE(changed_logits.ok());
  EXPECT_EQ(*capture_bytes, *changed_hidden);
  EXPECT_NE(*captured_logits, *changed_logits);
}

TEST_F(MlpReadoutTest, CapturePrefixIsBitwiseCausalAcrossFutureTokens) {
  auto full = Upload<int32_t>({0, 1, 2, 3, 4, 5, 6, 1});
  auto prefix = Upload<int32_t>({0, 1, 2, 3, 4, 0, 0, 0});
  ASSERT_TRUE(full.ok());
  ASSERT_TRUE(prefix.ok());
  auto a = CaptureThirdAttention(*executor_, *source_, *full);
  auto b = CaptureThirdAttention(*executor_, *source_, *prefix);
  ASSERT_TRUE(a.ok()) << a.status();
  ASSERT_TRUE(b.ok()) << b.status();
  auto a_bytes = Snapshot({&a->hidden, 1});
  auto b_bytes = Snapshot({&b->hidden, 1});
  ASSERT_TRUE(a_bytes.ok());
  ASSERT_TRUE(b_bytes.ok());
  ASSERT_EQ(a_bytes->front().size(), 8 * 10 * 2u);
  EXPECT_TRUE(std::equal(a_bytes->front().begin(),
                         a_bytes->front().begin() + 5 * 10 * 2,
                         b_bytes->front().begin()));
  EXPECT_NE(*a_bytes, *b_bytes);
}

TEST_F(MlpReadoutTest, SeedReproducibilityAndSourceConfigurationValidation) {
  const int seed = std::numeric_limits<int>::max();
  for (int depth : {1, 5}) {
    SCOPED_TRACE(depth);
    auto a = CreateMlpReadout(*executor_, *source_, config_, 150, seed, depth,
                              false);
    auto b = CreateMlpReadout(*executor_, *source_, config_, 150, seed, depth,
                              false);
    ASSERT_TRUE(a.ok()) << a.status();
    ASSERT_TRUE(b.ok()) << b.status();
    auto a_bytes = Snapshot(a->model->weights());
    auto b_bytes = Snapshot(b->model->weights());
    ASSERT_TRUE(a_bytes.ok());
    ASSERT_TRUE(b_bytes.ok());
    EXPECT_EQ(*a_bytes, *b_bytes);
  }
  for (int invalid_width : {-1, 0, std::numeric_limits<int>::max()})
    EXPECT_EQ(
        CreateMlpReadout(*executor_, *source_, config_, invalid_width, 3)
            .status()
            .code(),
        absl::StatusCode::kInvalidArgument);
  for (int invalid_depth : {-1, 0, std::numeric_limits<int>::min()})
    EXPECT_EQ(CreateMlpReadout(*executor_, *source_, config_, 150, 3,
                               invalid_depth, false)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  auto overflowing = config_;
  overflowing.model_width = 1;
  overflowing.feed_forward_width = 1;
  overflowing.context_length = 1;
  const int limit = std::numeric_limits<int>::max();
  auto rejected = CreateMlpReadout(*executor_, *source_, overflowing, limit, 3,
                                   limit, false);
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(rejected.status().message().find("parameter count overflows"),
            std::string::npos);
  EXPECT_EQ(CreateMlpReadout(*executor_, *source_, config_, 150, -1)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  for (int field = 0; field < 6; ++field) {
    auto wrong = config_;
    if (field == 0)
      wrong.model_width = 5;
    if (field == 1)
      wrong.feed_forward_width = 19;
    if (field == 2)
      wrong.vocabulary_size = 6;
    if (field == 3)
      wrong.transformer_block_count = 3;
    if (field == 4)
      wrong.context_length = 7;
    if (field == 5)
      wrong.pad_vocabulary = true;
    EXPECT_EQ(CreateMlpReadout(*executor_, *source_, wrong, 150, 3)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument)
        << field;
  }
  auto shallow_config = config_;
  shallow_config.transformer_block_count = 2;
  auto shallow = CreateGpt2(*executor_, DataType::BF16, 17, shallow_config);
  auto fp16 = CreateGpt2(*executor_, DataType::FP16, 17, config_);
  auto unrelated =
      LayerNormLayer::Create(*executor_, 10, 1e-5f, DataType::BF16, 8);
  auto tokens = Upload<int32_t>({0, 1, 2, 3, 4, 5, 6, 0});
  ASSERT_TRUE(shallow.ok());
  ASSERT_TRUE(fp16.ok());
  ASSERT_TRUE(unrelated.ok());
  ASSERT_TRUE(tokens.ok());
  for (const Layer* invalid : {static_cast<const Layer*>(shallow->get()),
                               static_cast<const Layer*>(fp16->get()),
                               static_cast<const Layer*>(unrelated->get())}) {
    EXPECT_EQ(CreateMlpReadout(*executor_, *invalid, config_, 150, 3)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(
        CaptureThirdAttention(*executor_, *invalid, *tokens).status().code(),
        absl::StatusCode::kInvalidArgument);
  }
  auto short_tokens = Upload<int32_t>({0, 1, 2});
  ASSERT_TRUE(short_tokens.ok());
  EXPECT_EQ(CaptureThirdAttention(*executor_, *source_, *short_tokens)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  EXPECT_EQ(
      CreateMlpReadout(**other, *source_, config_, 150, 3).status().code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(CaptureThirdAttention(**other, *source_, *tokens).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::llm::memorize_general_facts
