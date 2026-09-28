#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/adamw_optimizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/gpt2.h"
#include "src/llm/layer.h"
#include "src/llm/layer_hooks.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

constexpr int kSharedPrefixTensors = 32;
constexpr int kUniqueTensors = 52;

template <typename T>
absl::StatusOr<std::vector<T>> Download(cuda::Executor& executor,
                                        const Buffer& buffer) {
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<T>::Allocate(
                                  executor, buffer.size_bytes() / sizeof(T)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), buffer.data(), buffer.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "download model test tensor"));
  RETURN_IF_ERROR(executor.Synchronize());
  return std::vector<T>(host.begin(), host.end());
}

template <typename T>
absl::StatusOr<Buffer> Upload(cuda::Executor& executor,
                              const std::vector<T>& values) {
  ASSIGN_OR_RETURN(auto host,
                   cuda::PageLockedHostArray<T>::CopyFrom(executor, values));
  ASSIGN_OR_RETURN(auto buffer, Buffer::Allocate(executor, host.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(buffer.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload model test tensor"));
  return buffer;
}

absl::StatusOr<std::vector<std::vector<uint32_t>>> WeightBits(
    cuda::Executor& executor, Layer& model) {
  std::vector<std::vector<uint32_t>> result;
  for (const Buffer& weight : model.weights()) {
    ASSIGN_OR_RETURN(auto bits, Download<uint32_t>(executor, weight));
    result.push_back(std::move(bits));
  }
  return result;
}

size_t UniqueParameterCount(Layer& model) {
  std::vector<const void*> seen;
  size_t parameters = 0;
  for (const Buffer& weight : model.weights()) {
    if (std::find(seen.begin(), seen.end(), weight.data()) != seen.end())
      continue;
    seen.push_back(weight.data());
    parameters += weight.size_bytes() / sizeof(float);
  }
  return parameters;
}

class Gpt2MlpStackTest : public testing::Test {
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

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(Gpt2MlpStackTest, SharedPrefixAndNativeInitializationAreBitwiseStable) {
  for (const Gpt2Config& config : {Gpt2Config{4, 10, 1, 20, 4475, false, 27},
                                   Gpt2Config{4, 12, 3, 19, 37, true, 7}}) {
    for (DataType type : {DataType::FP16, DataType::BF16}) {
      auto baseline = CreateGpt2(*executor_, type, 1337, config);
      auto variant = CreateGpt2WithA3MlpStack(*executor_, type, 1337, config);
      auto repeated = CreateGpt2WithA3MlpStack(*executor_, type, 1337, config);
      auto other_seed =
          CreateGpt2WithA3MlpStack(*executor_, type, 1338, config);
      ASSERT_TRUE(baseline.ok()) << baseline.status();
      ASSERT_TRUE(variant.ok()) << variant.status();
      ASSERT_TRUE(repeated.ok()) << repeated.status();
      ASSERT_TRUE(other_seed.ok()) << other_seed.status();
      auto original_bits = WeightBits(*executor_, **baseline);
      auto variant_bits = WeightBits(*executor_, **variant);
      auto repeated_bits = WeightBits(*executor_, **repeated);
      auto other_bits = WeightBits(*executor_, **other_seed);
      ASSERT_TRUE(original_bits.ok()) << original_bits.status();
      ASSERT_TRUE(variant_bits.ok()) << variant_bits.status();
      ASSERT_TRUE(repeated_bits.ok()) << repeated_bits.status();
      ASSERT_TRUE(other_bits.ok()) << other_bits.status();
      ASSERT_EQ(variant_bits->size(), size_t{kUniqueTensors + 1});
      EXPECT_EQ(*variant_bits, *repeated_bits);
      EXPECT_NE(*variant_bits, *other_bits);
      for (int index = 0; index < kSharedPrefixTensors; ++index) {
        SCOPED_TRACE(index);
        EXPECT_EQ((*original_bits)[index], (*variant_bits)[index]);
        EXPECT_NE((*baseline)->weights()[index].data(),
                  (*variant)->weights()[index].data());
      }
      // The three fresh tail MLPs retain the native MLP seed schedule and
      // 0.02/0.005 projection initialization (including the fifth MLP).
      Gpt2Config five_blocks = config;
      five_blocks.transformer_block_count = 5;
      auto five = CreateGpt2(*executor_, type, 1337, five_blocks);
      ASSERT_TRUE(five.ok()) << five.status();
      auto five_bits = WeightBits(*executor_, **five);
      ASSERT_TRUE(five_bits.ok()) << five_bits.status();
      for (int mlp = 0; mlp < 3; ++mlp)
        for (int tensor = 0; tensor < 6; ++tensor)
          EXPECT_EQ((*variant_bits)[32 + 6 * mlp + tensor],
                    (*five_bits)[32 + 12 * mlp + tensor]);
      EXPECT_EQ((*variant_bits)[50], (*original_bits)[50]);
      EXPECT_EQ((*variant_bits)[51], (*original_bits)[51]);
    }
  }
}

TEST_F(Gpt2MlpStackTest, ExactParameterBudgetShapesAndTiedHead) {
  for (const Gpt2Config& config : {Gpt2Config{4, 10, 1, 20, 4475, false, 27},
                                   Gpt2Config{4, 12, 3, 19, 37, true, 7}}) {
    auto model =
        CreateGpt2WithA3MlpStack(*executor_, DataType::BF16, 1337, config);
    auto baseline = CreateGpt2(*executor_, DataType::BF16, 1337, config);
    ASSERT_TRUE(model.ok()) << model.status();
    ASSERT_TRUE(baseline.ok()) << baseline.status();
    EXPECT_EQ((*model)->name(), "gpt2_a3_mlp_stack");
    const int stride = (config.vocabulary_size + 15) / 16 * 16;
    EXPECT_EQ((*model)->input_types()[0],
              ActivationType(DataType::INT32, {-2, config.context_length}));
    EXPECT_EQ(
        (*model)->output_types()[0],
        ActivationType(DataType::FP32, {-2, config.context_length, stride}));
    const auto weights = (*model)->weights();
    const auto gradients = (*model)->gradients();
    ASSERT_EQ(weights.size(), size_t{kUniqueTensors + 1});
    ASSERT_EQ(gradients.size(), weights.size());
    EXPECT_EQ(weights.front().data(), weights.back().data());
    EXPECT_EQ(gradients.front().data(), gradients.back().data());
    for (int index = 0; index < kUniqueTensors; ++index) {
      EXPECT_EQ(weights[index].size_bytes(), gradients[index].size_bytes());
      for (int earlier = 0; earlier < index; ++earlier) {
        EXPECT_NE(weights[index].data(), weights[earlier].data());
        EXPECT_NE(gradients[index].data(), gradients[earlier].data());
      }
    }
    const size_t width = config.model_width;
    const size_t ff = config.feed_forward_width;
    const size_t tail_shapes[] = {width, width,      width * ff,
                                  ff,    ff * width, width};
    for (int index = 32; index < 50; ++index)
      EXPECT_EQ(weights[index].size_bytes(),
                tail_shapes[(index - 32) % 6] * sizeof(float));
    EXPECT_EQ(weights[50].size_bytes(), width * sizeof(float));
    EXPECT_EQ(weights[51].size_bytes(), width * sizeof(float));
    const size_t vocabulary =
        config.pad_vocabulary ? stride : config.vocabulary_size;
    const size_t attention = 4 * width * width + 6 * width;
    const size_t mlp = 2 * width * ff + ff + 3 * width;
    EXPECT_EQ(UniqueParameterCount(**model),
              (vocabulary + config.context_length + 2) * width + 3 * attention +
                  5 * mlp);
    if (config.model_width == 10) {
      EXPECT_EQ(UniqueParameterCount(**model), 48'670u);
      EXPECT_EQ(UniqueParameterCount(**baseline), 48'680u);
    }
  }
}

TEST_F(Gpt2MlpStackTest, OptimizerUpdatesEveryTensorThroughTheCompleteGraph) {
  const Gpt2Config config{4, 10, 1, 20, 37, false, 7};
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    SCOPED_TRACE(static_cast<int>(type));
    auto model = CreateGpt2WithA3MlpStack(*executor_, type, 1337, config);
    auto loss = CrossEntropyLossLayer::Create(*executor_, 37, type, 7);
    ASSERT_TRUE(model.ok()) << model.status();
    ASSERT_TRUE(loss.ok()) << loss.status();
    auto optimizer = AdamWOptimizer::Create(
        *executor_, **model,
        {.learning_rate = 0.0012f, .beta2 = 0.99f, .weight_decay = 0.0f});
    ASSERT_TRUE(optimizer.ok()) << optimizer.status();
    EXPECT_EQ((*optimizer)->parameter_tensor_count(), size_t{kUniqueTensors});
    ASSERT_TRUE((*optimizer)->ZeroGrad().ok());
    auto before = WeightBits(*executor_, **model);
    ASSERT_TRUE(before.ok()) << before.status();
    auto tokens = Upload<int32_t>(
        *executor_, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13});
    auto targets = Upload<int32_t>(
        *executor_, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
    ASSERT_TRUE(tokens.ok()) << tokens.status();
    ASSERT_TRUE(targets.ok()) << targets.status();
    std::vector<std::string> branches;
    LayerHooks hooks;
    hooks.enter_combinator = [&](cuda::Executor&, absl::string_view name) {
      if (name == "attention" || name == "mlp")
        branches.emplace_back(name);
      return absl::OkStatus();
    };
    auto forward = (*model)->fwd(*executor_, {*tokens}, &hooks);
    ASSERT_TRUE(forward.ok()) << forward.status();
    EXPECT_EQ(branches,
              (std::vector<std::string>{"attention", "mlp", "attention", "mlp",
                                        "attention", "mlp", "mlp", "mlp"}));
    auto loss_forward =
        (*loss)->fwd(*executor_, {forward->outputs[0], *targets});
    ASSERT_TRUE(loss_forward.ok()) << loss_forward.status();
    auto loss_backward =
        (*loss)->bwd(*executor_, {}, std::move(loss_forward->state));
    ASSERT_TRUE(loss_backward.ok()) << loss_backward.status();
    auto backward =
        (*model)->bwd(*executor_, *loss_backward, std::move(forward->state));
    ASSERT_TRUE(backward.ok()) << backward.status();
    EXPECT_TRUE(backward->empty());
    for (int index = 0; index < kUniqueTensors; ++index) {
      SCOPED_TRACE(index);
      auto gradient = Download<float>(*executor_, (*model)->gradients()[index]);
      ASSERT_TRUE(gradient.ok()) << gradient.status();
      EXPECT_TRUE(
          std::all_of(gradient->begin(), gradient->end(),
                      [](float value) { return std::isfinite(value); }));
      EXPECT_TRUE(std::any_of(gradient->begin(), gradient->end(),
                              [](float value) { return value != 0.0f; }));
    }
    ASSERT_TRUE((*optimizer)->ApplyStep().ok());
    EXPECT_EQ((*optimizer)->step(), 1);
    auto after = WeightBits(*executor_, **model);
    ASSERT_TRUE(after.ok()) << after.status();
    for (int index = 0; index < kUniqueTensors; ++index) {
      SCOPED_TRACE(index);
      EXPECT_NE((*before)[index], (*after)[index]);
    }
  }
}

TEST_F(Gpt2MlpStackTest, ForwardAndBackwardRespectCausalityAndBatchBoundaries) {
  const Gpt2Config config{4, 10, 1, 20, 37, false, 7};
  constexpr int kStride = 48;
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    auto model = CreateGpt2WithA3MlpStack(*executor_, type, 1337, config);
    ASSERT_TRUE(model.ok()) << model.status();
    auto tokens = Upload<int32_t>(
        *executor_, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13});
    auto changed_tokens = Upload<int32_t>(
        *executor_, {0, 1, 2, 3, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33});
    ASSERT_TRUE(tokens.ok()) << tokens.status();
    ASSERT_TRUE(changed_tokens.ok()) << changed_tokens.status();
    auto original = (*model)->fwd(*executor_, {*tokens});
    auto changed = (*model)->fwd(*executor_, {*changed_tokens});
    ASSERT_TRUE(original.ok()) << original.status();
    ASSERT_TRUE(changed.ok()) << changed.status();
    auto first_bits = Download<uint32_t>(*executor_, original->outputs[0]);
    auto changed_bits = Download<uint32_t>(*executor_, changed->outputs[0]);
    ASSERT_TRUE(first_bits.ok()) << first_bits.status();
    ASSERT_TRUE(changed_bits.ok()) << changed_bits.status();
    EXPECT_TRUE(std::equal(first_bits->begin(),
                           first_bits->begin() + 4 * kStride,
                           changed_bits->begin()));
    EXPECT_NE(*first_bits, *changed_bits);

    std::vector<float> derivative(14 * kStride, 0.0f);
    derivative[3 * kStride + 1] = 1.0f;
    auto upstream = Upload(*executor_, derivative);
    ASSERT_TRUE(upstream.ok()) << upstream.status();
    std::vector<Buffer> residual_gradients;
    LayerHooks hooks;
    hooks.gradient_hook = [&](cuda::Executor&, absl::string_view name,
                              absl::Span<const ActivationType>,
                              absl::Span<Buffer> gradients) {
      if (name == "PositionEmbeddingLayer")
        residual_gradients.push_back(gradients[0]);
      return absl::OkStatus();
    };
    auto backward = (*model)->bwd(*executor_, {*upstream},
                                  std::move(original->state), &hooks);
    ASSERT_TRUE(backward.ok()) << backward.status();
    ASSERT_EQ(residual_gradients.size(), 1u);
    auto gradient = Download<float>(*executor_, residual_gradients[0]);
    ASSERT_TRUE(gradient.ok()) << gradient.status();
    ASSERT_EQ(gradient->size(), 14u * 10);
    for (int row = 0; row < 14; ++row) {
      SCOPED_TRACE(row);
      const auto begin = gradient->begin() + row * 10;
      const bool nonzero = std::any_of(
          begin, begin + 10, [](float value) { return value != 0.0f; });
      EXPECT_EQ(nonzero, row <= 3);
    }
  }
}

TEST_F(Gpt2MlpStackTest, CheckpointRoundTripPreservesAllWeightsAndTiedLogits) {
  const Gpt2Config config{4, 10, 1, 20, 37, false, 7};
  auto original =
      CreateGpt2WithA3MlpStack(*executor_, DataType::BF16, 1337, config);
  auto restored =
      CreateGpt2WithA3MlpStack(*executor_, DataType::BF16, 42, config);
  ASSERT_TRUE(original.ok()) << original.status();
  ASSERT_TRUE(restored.ok()) << restored.status();
  const auto directory = std::filesystem::path(testing::TempDir()) /
                         "gpt2-a3-mlp-stack-round-trip";
  ASSERT_TRUE(WriteToDirectory(*executor_, **original, directory).ok());
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(directory),
                          std::filesystem::directory_iterator()),
            std::ptrdiff_t{kUniqueTensors});
  ASSERT_TRUE(ReadFromDirectory(*executor_, **restored, directory,
                                /*allow_prefix=*/false)
                  .ok());
  auto original_bits = WeightBits(*executor_, **original);
  auto restored_bits = WeightBits(*executor_, **restored);
  ASSERT_TRUE(original_bits.ok()) << original_bits.status();
  ASSERT_TRUE(restored_bits.ok()) << restored_bits.status();
  EXPECT_EQ(*original_bits, *restored_bits);
  EXPECT_EQ((*restored)->weights().front().data(),
            (*restored)->weights().back().data());
  auto tokens = Upload<int32_t>(*executor_, {1, 2, 3, 4, 5, 6, 7});
  ASSERT_TRUE(tokens.ok()) << tokens.status();
  auto first = (*original)->fwd(*executor_, {*tokens});
  auto second = (*restored)->fwd(*executor_, {*tokens});
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  auto first_bits = Download<uint32_t>(*executor_, first->outputs[0]);
  auto second_bits = Download<uint32_t>(*executor_, second->outputs[0]);
  ASSERT_TRUE(first_bits.ok()) << first_bits.status();
  ASSERT_TRUE(second_bits.ok()) << second_bits.status();
  EXPECT_EQ(*first_bits, *second_bits);
}

TEST_F(Gpt2MlpStackTest, RejectsOtherDepthsInvalidShapesAndInvalidTokenOrders) {
  Gpt2Config config{4, 10, 1, 20, 37, false, 7};
  for (int blocks : {-1, 0, 1, 2, 3, 5, 8}) {
    config.transformer_block_count = blocks;
    auto model =
        CreateGpt2WithA3MlpStack(*executor_, DataType::BF16, 1337, config);
    EXPECT_EQ(model.status().code(), absl::StatusCode::kInvalidArgument);
  }
  config.transformer_block_count = 4;
  config.attention_heads = 3;
  EXPECT_EQ(CreateGpt2WithA3MlpStack(*executor_, DataType::BF16, 1337, config)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  config.attention_heads = 1;
  std::vector<int32_t> invalid_order(config.vocabulary_size, 0);
  EXPECT_EQ(CreateGpt2WithA3MlpStack(*executor_, DataType::BF16, 1337, config,
                                     invalid_order)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  for (int index = 0; index < config.vocabulary_size; ++index)
    invalid_order[index] = config.vocabulary_size - index - 1;
  EXPECT_TRUE(CreateGpt2WithA3MlpStack(*executor_, DataType::BF16, 1337, config,
                                       invalid_order)
                  .ok());
}

}  // namespace
}  // namespace pluto::llm
