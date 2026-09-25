#include "src/llm/experiments/memorize_general_facts/fit_attention_readout/readout.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/adamw_optimizer.h"
#include "src/llm/layer_hooks.h"
#include "src/llm/layers/norm.h"
#include "src/util/status_macros.h"

namespace pluto::llm::fit_attention_readout {
namespace {

class ReadoutTest : public testing::Test {
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
    if (executor_ == nullptr)
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
        "upload readout fixture"));
    return device;
  }

  // Byte snapshots catch any change, including signs of zero, without relying
  // on a numerical tolerance or on which tensors happen to be tied.
  absl::StatusOr<std::vector<std::vector<uint8_t>>> Snapshot(
      absl::Span<const Buffer> buffers) {
    std::vector<std::vector<uint8_t>> result;
    for (const auto& buffer : buffers) {
      ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<uint8_t>::Allocate(
                                      *executor_, buffer.size_bytes()));
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(host.data(), buffer.data(), host.size_bytes(),
                          cudaMemcpyDeviceToHost, executor_->stream()),
          "snapshot readout bytes"));
      RETURN_IF_ERROR(executor_->Synchronize());
      result.emplace_back(host.begin(), host.end());
    }
    return result;
  }

  absl::Status Fill(const Buffer& buffer, float base) {
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<float>::Allocate(
                         *executor_, buffer.size_bytes() / sizeof(float)));
    for (size_t i = 0; i < host.size(); ++i)
      host[i] = base + static_cast<float>(i) / 1024;
    return cuda::CudaStatus(
        cudaMemcpyAsync(buffer.data(), host.data(), host.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "initialize distinctive tensor");
  }

  absl::Status FillDistinctSourceWeights() {
    auto weights = source_->weights();
    // Skip the final head alias so token embedding keeps its own pattern.
    for (size_t i = 0; i + 1 < weights.size(); ++i)
      RETURN_IF_ERROR(Fill(weights[i], 0.125f * static_cast<float>(i + 1)));
    return absl::OkStatus();
  }

  const Gpt2Config config_{.transformer_block_count = 3,
                           .model_width = 4,
                           .attention_heads = 1,
                           .feed_forward_width = 8,
                           .vocabulary_size = 7,
                           .pad_vocabulary = false,
                           .context_length = 3};
  std::unique_ptr<cuda::Executor> executor_;
  std::unique_ptr<ComposedLayer> source_;
};

TEST_F(ReadoutTest, CopiesTheChosenSixTensorsAndFrozenHeadIndependently) {
  ASSERT_TRUE(FillDistinctSourceWeights().ok());
  auto before = Snapshot(source_->weights());
  ASSERT_TRUE(before.ok()) << before.status();
  for (int block = 0; block < config_.transformer_block_count; ++block) {
    SCOPED_TRACE(block);
    auto readout = CreateReadout(*executor_, *source_, config_, block);
    ASSERT_TRUE(readout.ok()) << readout.status();
    ASSERT_NE(readout->trainable, nullptr);
    ASSERT_EQ(readout->trainable->weights().size(), 6u);
    ASSERT_EQ(readout->model->weights().size(), 9u);
    auto copied = Snapshot(readout->model->weights());
    ASSERT_TRUE(copied.ok()) << copied.status();
    const int first_mlp = 2 + 12 * block + 6;
    for (int index = 0; index < 6; ++index) {
      EXPECT_EQ((*copied)[index], (*before)[first_mlp + index]);
      EXPECT_EQ(readout->trainable->weights()[index].data(),
                readout->model->weights()[index].data());
    }
    EXPECT_EQ((*copied)[6], (*before)[38]);
    EXPECT_EQ((*copied)[7], (*before)[39]);
    EXPECT_EQ((*copied)[8], (*before)[0]);
    EXPECT_EQ(readout->embedding->weight().data(),
              readout->model->weights()[8].data());
    for (const auto& weight : readout->model->weights())
      for (const auto& original : source_->weights())
        EXPECT_NE(weight.data(), original.data());
  }
  auto after = Snapshot(source_->weights());
  ASSERT_TRUE(after.ok()) << after.status();
  EXPECT_EQ(*before, *after);
}

TEST_F(ReadoutTest, OptimizerChangesOnlyTheBranchEvenWithFrozenGradients) {
  ASSERT_TRUE(FillDistinctSourceWeights().ok());
  auto source_before = Snapshot(source_->weights());
  ASSERT_TRUE(source_before.ok()) << source_before.status();
  auto readout = CreateReadout(*executor_, *source_, config_, 1);
  ASSERT_TRUE(readout.ok()) << readout.status();
  auto before = Snapshot(readout->model->weights());
  ASSERT_TRUE(before.ok()) << before.status();
  auto optimizer =
      AdamWOptimizer::Create(*executor_, *readout->trainable,
                             {.learning_rate = 0.01f, .weight_decay = 0});
  ASSERT_TRUE(optimizer.ok()) << optimizer.status();
  EXPECT_EQ((*optimizer)->parameter_tensor_count(), 6u);
  // Frozen tensors deliberately receive nonzero gradients as they would
  // during backward propagation; exclusion from optimizer is what protects
  // them, not a conveniently zero gradient.
  for (const auto& gradient : readout->model->gradients())
    ASSERT_TRUE(Fill(gradient, 1).ok());
  ASSERT_TRUE((*optimizer)->ApplyStep().ok());
  auto after = Snapshot(readout->model->weights());
  auto source_after = Snapshot(source_->weights());
  ASSERT_TRUE(after.ok()) << after.status();
  ASSERT_TRUE(source_after.ok()) << source_after.status();
  for (int index = 0; index < 6; ++index)
    EXPECT_NE((*before)[index], (*after)[index]);
  for (int index = 6; index < 9; ++index)
    EXPECT_EQ((*before)[index], (*after)[index]);
  EXPECT_EQ(*source_before, *source_after);
}

TEST_F(ReadoutTest, MatrixRestartIsDeterministicAndLeavesOtherTensorsCopied) {
  ASSERT_TRUE(FillDistinctSourceWeights().ok());
  auto warm = CreateReadout(*executor_, *source_, config_, 1);
  auto first = CreateReadout(*executor_, *source_, config_, 1, 23);
  auto second = CreateReadout(*executor_, *source_, config_, 1, 23);
  ASSERT_TRUE(warm.ok()) << warm.status();
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  auto warm_bytes = Snapshot(warm->model->weights());
  auto first_bytes = Snapshot(first->model->weights());
  auto second_bytes = Snapshot(second->model->weights());
  ASSERT_TRUE(warm_bytes.ok());
  ASSERT_TRUE(first_bytes.ok());
  ASSERT_TRUE(second_bytes.ok());
  EXPECT_EQ(*first_bytes, *second_bytes);
  for (int index = 0; index < 9; ++index)
    if (index == 2 || index == 4)
      EXPECT_NE((*warm_bytes)[index], (*first_bytes)[index]);
    else
      EXPECT_EQ((*warm_bytes)[index], (*first_bytes)[index]);
}

TEST_F(ReadoutTest, FreshBranchUsesNoCheckpointBranchParameters) {
  ASSERT_TRUE(FillDistinctSourceWeights().ok());
  auto source_before = Snapshot(source_->weights());
  auto warm = CreateReadout(*executor_, *source_, config_, 1);
  auto matrix_restart = CreateReadout(*executor_, *source_, config_, 1, 23);
  auto fresh = CreateReadout(*executor_, *source_, config_, 1, 23, true);
  auto repeat = CreateReadout(*executor_, *source_, config_, 1, 23, true);
  ASSERT_TRUE(source_before.ok()) << source_before.status();
  ASSERT_TRUE(warm.ok()) << warm.status();
  ASSERT_TRUE(matrix_restart.ok()) << matrix_restart.status();
  ASSERT_TRUE(fresh.ok()) << fresh.status();
  ASSERT_TRUE(repeat.ok()) << repeat.status();
  auto warm_bytes = Snapshot(warm->model->weights());
  auto restart_bytes = Snapshot(matrix_restart->model->weights());
  auto fresh_bytes = Snapshot(fresh->model->weights());
  auto repeat_bytes = Snapshot(repeat->model->weights());
  ASSERT_TRUE(warm_bytes.ok());
  ASSERT_TRUE(restart_bytes.ok());
  ASSERT_TRUE(fresh_bytes.ok());
  ASSERT_TRUE(repeat_bytes.ok());
  EXPECT_EQ(*fresh_bytes, *repeat_bytes);
  for (int index = 0; index < 6; ++index)
    EXPECT_NE((*fresh_bytes)[index], (*warm_bytes)[index]);
  // Random matrices use the same declared seed/distribution in either mode.
  for (int index : {2, 4})
    EXPECT_EQ((*fresh_bytes)[index], (*restart_bytes)[index]);
  // Fresh normalization is identity-affine, and both dense biases are zero.
  for (int index : {0, 1, 3, 5}) {
    const auto& bytes = (*fresh_bytes)[index];
    std::vector<float> values(bytes.size() / sizeof(float));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    for (float value : values)
      EXPECT_EQ(value, index == 0 ? 1.0f : 0.0f);
  }
  for (int index = 6; index < 9; ++index)
    EXPECT_EQ((*fresh_bytes)[index], (*warm_bytes)[index]);
  auto source_after = Snapshot(source_->weights());
  ASSERT_TRUE(source_after.ok());
  EXPECT_EQ(*source_before, *source_after);

  // Perturb all six source branch tensors, not the frozen head. A fresh
  // branch must be completely independent of those learned values.
  constexpr int kFirstMlp = 2 + 12 + 6;
  for (int index = 0; index < 6; ++index)
    ASSERT_TRUE(Fill(source_->weights()[kFirstMlp + index], -10 - index).ok());
  auto changed_source =
      CreateReadout(*executor_, *source_, config_, 1, 23, true);
  ASSERT_TRUE(changed_source.ok()) << changed_source.status();
  auto unchanged_fresh = Snapshot(changed_source->model->weights());
  ASSERT_TRUE(unchanged_fresh.ok());
  EXPECT_EQ(*fresh_bytes, *unchanged_fresh);
}

TEST_F(ReadoutTest, FreshBranchRequiresSeedAndSupportsLargestIntSeed) {
  for (int seed : {-2, -1})
    EXPECT_EQ(CreateReadout(*executor_, *source_, config_, 1, seed, true)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  const int seed = std::numeric_limits<int>::max();
  auto first = CreateReadout(*executor_, *source_, config_, 1, seed, true);
  auto second = CreateReadout(*executor_, *source_, config_, 1, seed, true);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  auto first_bytes = Snapshot(first->model->weights());
  auto second_bytes = Snapshot(second->model->weights());
  ASSERT_TRUE(first_bytes.ok());
  ASSERT_TRUE(second_bytes.ok());
  EXPECT_EQ(*first_bytes, *second_bytes);
}

TEST_F(ReadoutTest, LastBlockWarmStartReproducesSourceLogitsBitwise) {
  auto readout = CreateReadout(*executor_, *source_, config_, 2);
  auto tokens = Upload<int>({0, 1, 2, 3, 4, 5});
  ASSERT_TRUE(readout.ok()) << readout.status();
  ASSERT_TRUE(tokens.ok()) << tokens.status();
  std::vector<std::string> scopes;
  std::optional<Buffer> activation;
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
    if (!activation && name == "ResidualLayer" && scopes.size() == 2 &&
        scopes[0] == "gpt2" && scopes[1] == "transformer_block_2")
      activation = buffers[0];
    return absl::OkStatus();
  };
  auto original = source_->fwd(*executor_, {&*tokens, 1}, &hooks);
  ASSERT_TRUE(original.ok()) << original.status();
  ASSERT_TRUE(activation.has_value());
  EXPECT_TRUE(scopes.empty());
  auto replacement = readout->model->fwd(*executor_, {&*activation, 1});
  ASSERT_TRUE(replacement.ok()) << replacement.status();
  auto original_bytes = Snapshot(original->outputs);
  auto replacement_bytes = Snapshot(replacement->outputs);
  ASSERT_TRUE(original_bytes.ok()) << original_bytes.status();
  ASSERT_TRUE(replacement_bytes.ok()) << replacement_bytes.status();
  EXPECT_EQ(*original_bytes, *replacement_bytes);
}

TEST_F(ReadoutTest, RejectsInvalidBlockSeedAndMismatchedConfiguration) {
  for (int block : {-1, 3})
    EXPECT_EQ(
        CreateReadout(*executor_, *source_, config_, block).status().code(),
        absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(CreateReadout(*executor_, *source_, config_, 0, -2).status().code(),
            absl::StatusCode::kInvalidArgument);
  for (int field = 0; field < 4; ++field) {
    auto wrong = config_;
    if (field == 0)
      wrong.model_width = 5;
    if (field == 1)
      wrong.feed_forward_width = 9;
    if (field == 2)
      wrong.vocabulary_size = 8;
    if (field == 3)
      wrong.transformer_block_count = 2;
    EXPECT_EQ(CreateReadout(*executor_, *source_, wrong, 0).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST_F(ReadoutTest, RejectsUnrelatedLayerAndAnotherExecutor) {
  auto norm = LayerNormLayer::Create(*executor_, 4, 1e-5f, DataType::BF16, 3);
  ASSERT_TRUE(norm.ok()) << norm.status();
  EXPECT_EQ(CreateReadout(*executor_, **norm, config_, 0).status().code(),
            absl::StatusCode::kInvalidArgument);
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  EXPECT_EQ(CreateReadout(**other, *source_, config_, 0).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE((*other)->Synchronize().ok());
}

}  // namespace
}  // namespace pluto::llm::fit_attention_readout
