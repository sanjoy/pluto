#include "src/llm/experiments/memorize_general_facts/fit_attention_readout/readout.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
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

  // Six rows of varied, exactly representable BF16 values, without requiring
  // CUDA's host BF16 type in this ordinary C++ test translation unit.
  absl::StatusOr<Buffer> InputActivations() {
    return Upload<uint16_t>({0x3f00, 0x3f80, 0x4000, 0xbf00, 0xbf80, 0x4040,
                             0x3e80, 0x3f80, 0x4000, 0x3f00, 0xbf80, 0x4080,
                             0x4040, 0xc000, 0x3f00, 0x3f80, 0x3e80, 0x3f80,
                             0x4080, 0xbf00, 0xbf00, 0x3f00, 0x4000, 0x4040});
  }

  absl::StatusOr<Buffer> OutputGradients(const Buffer& logits) {
    std::vector<float> values(logits.size_bytes() / sizeof(float));
    // The head's FP32 logits include padding even for an unpadded embedding.
    // Padded classes receive zero upstream gradient, as in cross entropy.
    const size_t columns = values.size() / 6;
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = i % columns < static_cast<size_t>(config_.vocabulary_size)
                      ? (static_cast<int>(i % 5) - 2) / 8.0f
                      : 0;
    return Upload(values);
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
    EXPECT_EQ(readout->branch, readout->trainable);
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

TEST_F(ReadoutTest, TrainFinalNormExposesAndUpdatesOnlyTheEightSuffixTensors) {
  ASSERT_TRUE(FillDistinctSourceWeights().ok());
  auto source_before = Snapshot(source_->weights());
  auto readout =
      CreateReadout(*executor_, *source_, config_, 1, -1, false, 0, true);
  ASSERT_TRUE(source_before.ok()) << source_before.status();
  ASSERT_TRUE(readout.ok()) << readout.status();
  ASSERT_EQ(readout->branch->weights().size(), 6u);
  ASSERT_EQ(readout->trainable->weights().size(), 8u);
  ASSERT_EQ(readout->model->weights().size(), 9u);
  EXPECT_NE(readout->branch, readout->trainable);
  size_t branch_bytes = 0;
  size_t trainable_bytes = 0;
  for (const auto& weight : readout->branch->weights())
    branch_bytes += weight.size_bytes();
  for (size_t i = 0; i < readout->trainable->weights().size(); ++i) {
    const auto& weight = readout->trainable->weights()[i];
    trainable_bytes += weight.size_bytes();
    EXPECT_EQ(weight.data(), readout->model->weights()[i].data());
  }
  for (size_t i = 0; i < 6; ++i)
    EXPECT_EQ(readout->trainable->weights()[i].data(),
              readout->branch->weights()[i].data());
  EXPECT_EQ(trainable_bytes - branch_bytes,
            2 * config_.model_width * sizeof(float));
  auto before = Snapshot(readout->model->weights());
  auto optimizer =
      AdamWOptimizer::Create(*executor_, *readout->trainable,
                             {.learning_rate = 0.01f, .weight_decay = 0});
  ASSERT_TRUE(before.ok()) << before.status();
  ASSERT_TRUE(optimizer.ok()) << optimizer.status();
  EXPECT_EQ((*optimizer)->parameter_tensor_count(), 8u);
  // Even a nonzero head gradient cannot update the frozen embedding because
  // it is absent from trainable. Both final LN tensors must now update.
  for (const auto& gradient : readout->model->gradients())
    ASSERT_TRUE(Fill(gradient, 1).ok());
  ASSERT_TRUE((*optimizer)->ApplyStep().ok());
  auto after = Snapshot(readout->model->weights());
  auto source_after = Snapshot(source_->weights());
  ASSERT_TRUE(after.ok()) << after.status();
  ASSERT_TRUE(source_after.ok()) << source_after.status();
  for (int index = 0; index < 8; ++index)
    EXPECT_NE((*before)[index], (*after)[index]) << index;
  EXPECT_EQ((*before)[8], (*after)[8]);
  EXPECT_EQ(*source_before, *source_after);
}

TEST_F(ReadoutTest,
       TrainFinalNormKeepsFreshForwardAndBackwardBitwiseIdentical) {
  auto frozen = CreateReadout(*executor_, *source_, config_, 2, 23, true);
  auto trained =
      CreateReadout(*executor_, *source_, config_, 2, 23, true, 0, true);
  auto input = InputActivations();
  ASSERT_TRUE(frozen.ok()) << frozen.status();
  ASSERT_TRUE(trained.ok()) << trained.status();
  ASSERT_TRUE(input.ok()) << input.status();
  auto frozen_weights = Snapshot(frozen->model->weights());
  auto trained_weights = Snapshot(trained->model->weights());
  ASSERT_TRUE(frozen_weights.ok());
  ASSERT_TRUE(trained_weights.ok());
  EXPECT_EQ(*frozen_weights, *trained_weights);
  auto frozen_output = frozen->model->fwd(*executor_, {&*input, 1});
  auto trained_output = trained->model->fwd(*executor_, {&*input, 1});
  ASSERT_TRUE(frozen_output.ok()) << frozen_output.status();
  ASSERT_TRUE(trained_output.ok()) << trained_output.status();
  auto upstream = OutputGradients(trained_output->outputs.front());
  ASSERT_TRUE(upstream.ok()) << upstream.status();
  auto frozen_output_bytes = Snapshot(frozen_output->outputs);
  auto trained_output_bytes = Snapshot(trained_output->outputs);
  ASSERT_TRUE(frozen_output_bytes.ok());
  ASSERT_TRUE(trained_output_bytes.ok());
  EXPECT_EQ(*frozen_output_bytes, *trained_output_bytes);
  auto frozen_input_gradient = frozen->model->bwd(
      *executor_, {&*upstream, 1}, std::move(frozen_output->state));
  auto trained_input_gradient = trained->model->bwd(
      *executor_, {&*upstream, 1}, std::move(trained_output->state));
  ASSERT_TRUE(frozen_input_gradient.ok()) << frozen_input_gradient.status();
  ASSERT_TRUE(trained_input_gradient.ok()) << trained_input_gradient.status();
  ASSERT_EQ(trained_input_gradient->size(), 1u);
  EXPECT_EQ(trained_input_gradient->front().size_bytes(),
            6 * config_.model_width * sizeof(float));
  auto frozen_input_bytes = Snapshot(*frozen_input_gradient);
  auto trained_input_bytes = Snapshot(*trained_input_gradient);
  auto frozen_gradients = Snapshot(frozen->model->gradients());
  auto trained_gradients = Snapshot(trained->model->gradients());
  ASSERT_TRUE(frozen_input_bytes.ok());
  ASSERT_TRUE(trained_input_bytes.ok());
  ASSERT_TRUE(frozen_gradients.ok());
  ASSERT_TRUE(trained_gradients.ok());
  EXPECT_EQ(*frozen_input_bytes, *trained_input_bytes);
  EXPECT_EQ(*frozen_gradients, *trained_gradients);
  // Verify that equality did not merely compare zero/invalid derivatives.
  for (int index : {6, 7}) {
    const auto& bytes = (*trained_gradients)[index];
    std::vector<float> values(bytes.size() / sizeof(float));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    bool any_nonzero = false;
    for (float value : values) {
      EXPECT_TRUE(std::isfinite(value));
      any_nonzero |= value != 0;
    }
    EXPECT_TRUE(any_nonzero) << index;
  }
}

TEST_F(ReadoutTest, TrainFinalNormCheckpointRestoresAllEightTensorsExactly) {
  auto original =
      CreateReadout(*executor_, *source_, config_, 1, 23, true, 0, true);
  auto restored =
      CreateReadout(*executor_, *source_, config_, 1, 99, true, 0, true);
  auto input = InputActivations();
  ASSERT_TRUE(original.ok()) << original.status();
  ASSERT_TRUE(restored.ok()) << restored.status();
  ASSERT_TRUE(input.ok()) << input.status();
  // Deliberately change both final LN parameters, not just the seeded MLP,
  // so restoring only the old six-tensor checkpoint cannot pass this test.
  ASSERT_TRUE(Fill(original->trainable->weights()[6], 0.5f).ok());
  ASSERT_TRUE(Fill(original->trainable->weights()[7], -0.25f).ok());
  const auto directory =
      std::filesystem::path(testing::TempDir()) / "readout-final-norm";
  ASSERT_TRUE(
      WriteToDirectory(*executor_, *original->trainable, directory).ok());
  for (int i = 0; i < 8; ++i)
    EXPECT_TRUE(std::filesystem::exists(
        directory / ("weight_" + std::to_string(i) + ".bin")));
  EXPECT_FALSE(std::filesystem::exists(directory / "weight_8.bin"));
  ASSERT_TRUE(ReadFromDirectory(*executor_, *restored->trainable, directory,
                                /*allow_prefix=*/false)
                  .ok());
  auto original_weights = Snapshot(original->model->weights());
  auto restored_weights = Snapshot(restored->model->weights());
  ASSERT_TRUE(original_weights.ok());
  ASSERT_TRUE(restored_weights.ok());
  EXPECT_EQ(*original_weights, *restored_weights);
  auto original_output = original->model->fwd(*executor_, {&*input, 1});
  auto restored_output = restored->model->fwd(*executor_, {&*input, 1});
  ASSERT_TRUE(original_output.ok()) << original_output.status();
  ASSERT_TRUE(restored_output.ok()) << restored_output.status();
  auto original_bytes = Snapshot(original_output->outputs);
  auto restored_bytes = Snapshot(restored_output->outputs);
  ASSERT_TRUE(original_bytes.ok());
  ASSERT_TRUE(restored_bytes.ok());
  EXPECT_EQ(*original_bytes, *restored_bytes);
  // Strict loading into the six-tensor branch is rejected, not silently
  // mistaken for a complete restore of the trainable eight-tensor suffix.
  EXPECT_EQ(ReadFromDirectory(*executor_, *restored->branch, directory,
                              /*allow_prefix=*/false)
                .code(),
            absl::StatusCode::kDataLoss);
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

TEST_F(ReadoutTest, WidthScalingMatchesExplicitStandardDeviationExactly) {
  // Equal-width scaling must match the historical initializer exactly; four
  // times the width must match a manually halved output standard deviation.
  for (int hidden :
       {config_.feed_forward_width, 4 * config_.feed_forward_width}) {
    SCOPED_TRACE(hidden);
    ReadoutInitializationOptions scaled;
    scaled.scale_output_by_width = true;
    auto manual = scaled;
    manual.scale_output_by_width = false;
    manual.output_standard_deviation *=
        std::sqrt(static_cast<float>(config_.feed_forward_width) / hidden);
    auto first = CreateReadout(*executor_, *source_, config_, 1, 23, true,
                               hidden, true, scaled);
    auto second = CreateReadout(*executor_, *source_, config_, 1, 23, true,
                                hidden, true, manual);
    ASSERT_TRUE(first.ok()) << first.status();
    ASSERT_TRUE(second.ok()) << second.status();
    auto first_bytes = Snapshot(first->model->weights());
    auto second_bytes = Snapshot(second->model->weights());
    ASSERT_TRUE(first_bytes.ok());
    ASSERT_TRUE(second_bytes.ok());
    EXPECT_EQ(*first_bytes, *second_bytes);
    if (hidden == config_.feed_forward_width) {
      auto original =
          CreateReadout(*executor_, *source_, config_, 1, 23, true, 0, true);
      ASSERT_TRUE(original.ok()) << original.status();
      auto original_bytes = Snapshot(original->model->weights());
      ASSERT_TRUE(original_bytes.ok());
      EXPECT_EQ(*first_bytes, *original_bytes);
    }
  }
}

TEST_F(ReadoutTest, MatrixDeviationsChangeOnlyTheIntendedMatrices) {
  auto original =
      CreateReadout(*executor_, *source_, config_, 1, 23, true, 0, true);
  auto scaled = CreateReadout(
      *executor_, *source_, config_, 1, 23, true, 0, true,
      {.input_standard_deviation = 0.4f, .output_standard_deviation = 0.05f});
  ASSERT_TRUE(original.ok()) << original.status();
  ASSERT_TRUE(scaled.ok()) << scaled.status();
  auto original_bytes = Snapshot(original->model->weights());
  auto scaled_bytes = Snapshot(scaled->model->weights());
  ASSERT_TRUE(original_bytes.ok());
  ASSERT_TRUE(scaled_bytes.ok());
  for (size_t index = 0; index < original_bytes->size(); ++index)
    if (index == 2 || index == 4) {
      const auto& before = (*original_bytes)[index];
      const auto& after = (*scaled_bytes)[index];
      const float factor = index == 2 ? 2.0f : 0.5f;
      for (size_t offset = 0; offset < before.size(); offset += sizeof(float)) {
        float before_value, after_value;
        std::memcpy(&before_value, before.data() + offset, sizeof(float));
        std::memcpy(&after_value, after.data() + offset, sizeof(float));
        EXPECT_EQ(after_value, factor * before_value);
      }
    } else {
      EXPECT_EQ((*original_bytes)[index], (*scaled_bytes)[index]);
    }
}

TEST_F(ReadoutTest, ZeroOutputInitializationPreservesCopiedBias) {
  ASSERT_TRUE(FillDistinctSourceWeights().ok());
  auto nonzero = CreateReadout(*executor_, *source_, config_, 1, 23);
  auto zero = CreateReadout(*executor_, *source_, config_, 1, 23, false, 0,
                            false, {.output_standard_deviation = 0});
  ASSERT_TRUE(nonzero.ok()) << nonzero.status();
  ASSERT_TRUE(zero.ok()) << zero.status();
  auto nonzero_bytes = Snapshot(nonzero->model->weights());
  auto zero_bytes = Snapshot(zero->model->weights());
  ASSERT_TRUE(nonzero_bytes.ok());
  ASSERT_TRUE(zero_bytes.ok());
  for (size_t i = 0; i < zero_bytes->size(); ++i)
    if (i == 4) {
      EXPECT_TRUE(std::all_of((*zero_bytes)[i].begin(), (*zero_bytes)[i].end(),
                              [](uint8_t byte) { return byte == 0; }));
    } else {
      EXPECT_EQ((*zero_bytes)[i], (*nonzero_bytes)[i]) << i;
    }
}

TEST_F(ReadoutTest, FreshZeroOutputStartsWithExactResidualIdentity) {
  auto readout = CreateReadout(*executor_, *source_, config_, 1, 23, true, 0,
                               true, {.output_standard_deviation = 0});
  auto input = InputActivations();
  ASSERT_TRUE(readout.ok()) << readout.status();
  ASSERT_TRUE(input.ok()) << input.status();
  auto output = readout->branch->fwd(*executor_, {&*input, 1});
  ASSERT_TRUE(output.ok()) << output.status();
  auto input_bytes = Snapshot({&*input, 1});
  auto output_bytes = Snapshot(output->outputs);
  ASSERT_TRUE(input_bytes.ok());
  ASSERT_TRUE(output_bytes.ok());
  EXPECT_EQ(*input_bytes, *output_bytes);
}

TEST_F(ReadoutTest, FreshFinalNormOnlyResetsItsAffineParameters) {
  ASSERT_TRUE(FillDistinctSourceWeights().ok());
  auto source_before = Snapshot(source_->weights());
  auto copied =
      CreateReadout(*executor_, *source_, config_, 1, 23, true, 0, true);
  auto fresh = CreateReadout(*executor_, *source_, config_, 1, 23, true, 0,
                             true, {.fresh_final_norm = true});
  ASSERT_TRUE(source_before.ok());
  ASSERT_TRUE(copied.ok()) << copied.status();
  ASSERT_TRUE(fresh.ok()) << fresh.status();
  auto copied_bytes = Snapshot(copied->model->weights());
  auto fresh_bytes = Snapshot(fresh->model->weights());
  auto source_after = Snapshot(source_->weights());
  ASSERT_TRUE(copied_bytes.ok());
  ASSERT_TRUE(fresh_bytes.ok());
  ASSERT_TRUE(source_after.ok());
  EXPECT_EQ(*source_before, *source_after);
  for (int index = 0; index < 9; ++index)
    if (index == 6 || index == 7) {
      EXPECT_NE((*fresh_bytes)[index], (*copied_bytes)[index]);
      const auto& bytes = (*fresh_bytes)[index];
      std::vector<float> values(bytes.size() / sizeof(float));
      std::memcpy(values.data(), bytes.data(), bytes.size());
      for (float value : values)
        EXPECT_EQ(value, index == 6 ? 1.0f : 0.0f);
    } else {
      EXPECT_EQ((*fresh_bytes)[index], (*copied_bytes)[index]);
    }
}

TEST_F(ReadoutTest, RejectsInvalidInitializationOptionsBeforeTraining) {
  for (float invalid : {-1.0f, std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
    SCOPED_TRACE(invalid);
    EXPECT_EQ(CreateReadout(*executor_, *source_, config_, 1, 23, true, 0, true,
                            {.input_standard_deviation = invalid})
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(CreateReadout(*executor_, *source_, config_, 1, 23, true, 0, true,
                            {.output_standard_deviation = invalid})
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  EXPECT_EQ(CreateReadout(*executor_, *source_, config_, 1, 23, true, 0, true,
                          {.input_standard_deviation = 0})
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  for (const ReadoutInitializationOptions options :
       {ReadoutInitializationOptions{.input_standard_deviation = 0.3f},
        ReadoutInitializationOptions{.output_standard_deviation = 0},
        ReadoutInitializationOptions{.scale_output_by_width = true}})
    EXPECT_EQ(CreateReadout(*executor_, *source_, config_, 1, -1, false, 0,
                            true, options)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  for (bool fresh_branch : {false, true})
    for (bool train_final_norm : {false, true})
      if (!fresh_branch || !train_final_norm) {
        EXPECT_EQ(
            CreateReadout(*executor_, *source_, config_, 1, 23, fresh_branch, 0,
                          train_final_norm, {.fresh_final_norm = true})
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
      }
  EXPECT_EQ(CreateReadout(
                *executor_, *source_, config_, 1, 23, true, 1, true,
                {.output_standard_deviation = std::numeric_limits<float>::max(),
                 .scale_output_by_width = true})
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
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

// Exercise the exact replacement widths used by the A3 capacity experiment.
// The frozen source stays 10 -> 20 -> 10; only the newly trained readout
// widens.
class WideReadoutTest : public ReadoutTest,
                        public testing::WithParamInterface<int> {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
    auto source = CreateGpt2(*executor_, DataType::BF16, 17, wide_config_);
    ASSERT_TRUE(source.ok()) << source.status();
    source_ = std::move(*source);
  }

  absl::StatusOr<Buffer> WideInput() {
    const uint16_t pattern[] = {0x3f00, 0x3f80, 0x4000, 0xbf00, 0xbf80,
                                0x4040, 0x3e80, 0x4080, 0xc000, 0xbf40};
    std::vector<uint16_t> values(6 * wide_config_.model_width);
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = pattern[(i + i / wide_config_.model_width) % 10];
    return Upload(values);
  }

  void ExpectFiniteFloats(const std::vector<uint8_t>& bytes) {
    ASSERT_EQ(bytes.size() % sizeof(float), 0u);
    for (size_t i = 0; i < bytes.size(); i += sizeof(float)) {
      float value;
      std::memcpy(&value, bytes.data() + i, sizeof(value));
      EXPECT_TRUE(std::isfinite(value)) << "byte offset " << i;
    }
  }

  const Gpt2Config wide_config_{.transformer_block_count = 4,
                                .model_width = 10,
                                .attention_heads = 1,
                                .feed_forward_width = 20,
                                .vocabulary_size = 7,
                                .pad_vocabulary = false,
                                .context_length = 3};
};

TEST_P(WideReadoutTest, TrainsRequestedWidthAndBothNormsWithoutChangingSource) {
  const int hidden = GetParam();
  auto readout = CreateReadout(*executor_, *source_, wide_config_, 2, 3, true,
                               hidden, true);
  auto source_before = Snapshot(source_->weights());
  ASSERT_TRUE(readout.ok()) << readout.status();
  ASSERT_TRUE(source_before.ok()) << source_before.status();
  ASSERT_EQ(readout->trainable->weights().size(), 8u);
  const std::vector<size_t> extents = {
      10,           10, 10u * hidden, static_cast<size_t>(hidden),
      10u * hidden, 10, 10,           10};
  size_t parameters = 0;
  for (size_t i = 0; i < extents.size(); ++i) {
    EXPECT_EQ(readout->trainable->weights()[i].size_bytes(),
              extents[i] * sizeof(float));
    EXPECT_EQ(readout->trainable->gradients()[i].size_bytes(),
              extents[i] * sizeof(float));
    parameters += extents[i];
  }
  EXPECT_EQ(parameters, 21u * hidden + 50);
  auto before = Snapshot(readout->model->weights());
  auto optimizer =
      AdamWOptimizer::Create(*executor_, *readout->trainable,
                             {.learning_rate = 0.001f, .weight_decay = 0});
  ASSERT_TRUE(before.ok()) << before.status();
  ASSERT_TRUE(optimizer.ok()) << optimizer.status();
  EXPECT_EQ((*optimizer)->parameter_tensor_count(), 8u);
  ASSERT_TRUE((*optimizer)->ZeroGrad().ok());
  auto input = WideInput();
  ASSERT_TRUE(input.ok()) << input.status();
  auto output = readout->model->fwd(*executor_, {&*input, 1});
  ASSERT_TRUE(output.ok()) << output.status();
  ASSERT_EQ(output->outputs.size(), 1u);
  auto logits = Snapshot(output->outputs);
  ASSERT_TRUE(logits.ok()) << logits.status();
  ASSERT_EQ(logits->front().size() % (6 * sizeof(float)), 0u);
  const size_t columns = logits->front().size() / (6 * sizeof(float));
  ASSERT_GE(columns, static_cast<size_t>(wide_config_.vocabulary_size));
  // Padded vocabulary columns intentionally contain -infinity, not NaNs.
  for (size_t row = 0; row < 6; ++row)
    for (int column = 0; column < wide_config_.vocabulary_size; ++column) {
      float value;
      std::memcpy(
          &value,
          logits->front().data() + (row * columns + column) * sizeof(float),
          sizeof(value));
      EXPECT_TRUE(std::isfinite(value));
    }
  auto upstream = OutputGradients(output->outputs.front());
  ASSERT_TRUE(upstream.ok()) << upstream.status();
  auto input_gradient = readout->model->bwd(*executor_, {&*upstream, 1},
                                            std::move(output->state));
  ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();
  ASSERT_EQ(input_gradient->size(), 1u);
  EXPECT_EQ(input_gradient->front().size_bytes(), 6 * 10 * sizeof(float));
  auto input_bytes = Snapshot(*input_gradient);
  auto gradient_bytes = Snapshot(readout->trainable->gradients());
  ASSERT_TRUE(input_bytes.ok()) << input_bytes.status();
  ASSERT_TRUE(gradient_bytes.ok()) << gradient_bytes.status();
  ExpectFiniteFloats(input_bytes->front());
  for (const auto& bytes : *gradient_bytes)
    ExpectFiniteFloats(bytes);
  ASSERT_TRUE((*optimizer)->ApplyStep().ok());
  auto after = Snapshot(readout->model->weights());
  auto source_after = Snapshot(source_->weights());
  ASSERT_TRUE(after.ok()) << after.status();
  ASSERT_TRUE(source_after.ok()) << source_after.status();
  for (size_t i = 0; i < 8; ++i)
    EXPECT_NE((*before)[i], (*after)[i]) << "trainable tensor " << i;
  EXPECT_EQ((*before)[8], (*after)[8]);
  EXPECT_EQ(*source_before, *source_after);
}

TEST_P(WideReadoutTest, StrictCheckpointRoundTripRejectsOtherReplacementWidth) {
  const int hidden = GetParam();
  auto original = CreateReadout(*executor_, *source_, wide_config_, 2, 3, true,
                                hidden, true);
  auto restored = CreateReadout(*executor_, *source_, wide_config_, 2, 91, true,
                                hidden, true);
  auto wrong_width = CreateReadout(*executor_, *source_, wide_config_, 2, 3,
                                   true, hidden + 1, true);
  ASSERT_TRUE(original.ok()) << original.status();
  ASSERT_TRUE(restored.ok()) << restored.status();
  ASSERT_TRUE(wrong_width.ok()) << wrong_width.status();
  ASSERT_TRUE(Fill(original->trainable->weights()[6], 0.5f).ok());
  ASSERT_TRUE(Fill(original->trainable->weights()[7], -0.25f).ok());
  const auto directory = std::filesystem::path(testing::TempDir()) /
                         ("readout-wide-" + std::to_string(hidden));
  ASSERT_TRUE(
      WriteToDirectory(*executor_, *original->trainable, directory).ok());
  ASSERT_TRUE(ReadFromDirectory(*executor_, *restored->trainable, directory,
                                /*allow_prefix=*/false)
                  .ok());
  EXPECT_EQ(ReadFromDirectory(*executor_, *wrong_width->trainable, directory,
                              /*allow_prefix=*/false)
                .code(),
            absl::StatusCode::kDataLoss);
  auto original_weights = Snapshot(original->model->weights());
  auto restored_weights = Snapshot(restored->model->weights());
  ASSERT_TRUE(original_weights.ok()) << original_weights.status();
  ASSERT_TRUE(restored_weights.ok()) << restored_weights.status();
  EXPECT_EQ(*original_weights, *restored_weights);
  auto input = WideInput();
  ASSERT_TRUE(input.ok()) << input.status();
  auto original_output = original->model->fwd(*executor_, {&*input, 1});
  auto restored_output = restored->model->fwd(*executor_, {&*input, 1});
  ASSERT_TRUE(original_output.ok()) << original_output.status();
  ASSERT_TRUE(restored_output.ok()) << restored_output.status();
  auto original_bytes = Snapshot(original_output->outputs);
  auto restored_bytes = Snapshot(restored_output->outputs);
  ASSERT_TRUE(original_bytes.ok()) << original_bytes.status();
  ASSERT_TRUE(restored_bytes.ok()) << restored_bytes.status();
  EXPECT_EQ(*original_bytes, *restored_bytes);
}

INSTANTIATE_TEST_SUITE_P(FourThroughTwentyTimesWidth, WideReadoutTest,
                         testing::Values(40, 80, 120, 150, 160, 200));

}  // namespace
}  // namespace pluto::llm::fit_attention_readout
