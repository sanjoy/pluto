#include "src/llm/experiments/memorize_general_facts/transformer_readout.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/adamw_optimizer.h"
#include "src/llm/layers/fully_connected.h"
#include "src/util/status_macros.h"

namespace pluto::llm::memorize_general_facts {
namespace {

Gpt2Config SmallConfig() {
  return {.transformer_block_count = 4,
          .model_width = 10,
          .attention_heads = 1,
          .feed_forward_width = 20,
          .vocabulary_size = 7,
          .pad_vocabulary = false,
          .context_length = 8};
}

TEST(MlpTransformerReadoutParameterBudgetTest, CountsTheExactSourceSuffix) {
  auto config = SmallConfig();
  auto budget = ResolveMlpTransformerReadoutParameterBudget(config);
  ASSERT_TRUE(budget.ok()) << budget.status();
  EXPECT_EQ(budget->minimum_mlp_width, 20);
  EXPECT_EQ(budget->mlp_width, 20);
  EXPECT_EQ(budget->mlp_parameters, 860);
  EXPECT_EQ(budget->source_tail_parameters, 1380);
  EXPECT_EQ(budget->trainable_parameters, 1380);
  config.model_width = 12;
  config.feed_forward_width = 37;
  config.attention_heads = 3;
  budget = ResolveMlpTransformerReadoutParameterBudget(config);
  ASSERT_TRUE(budget.ok()) << budget.status();
  EXPECT_EQ(budget->minimum_mlp_width, 37);
  EXPECT_EQ(budget->mlp_width, 37);
  EXPECT_EQ(budget->mlp_parameters, 2 * (25 * 37 + 12));
  EXPECT_EQ(budget->trainable_parameters, 2 * (25 * 37 + 12) + 4 * 144 + 144);
  EXPECT_EQ(budget->source_tail_parameters, budget->trainable_parameters);
}

TEST(MlpTransformerReadoutParameterBudgetTest, RejectsInvalidConfigAndDepth) {
  for (int blocks : {-1, 0, 3, 5, std::numeric_limits<int>::max()}) {
    auto config = SmallConfig();
    config.transformer_block_count = blocks;
    EXPECT_EQ(
        ResolveMlpTransformerReadoutParameterBudget(config).status().code(),
        absl::StatusCode::kInvalidArgument);
  }
  for (int invalid_width : {0, -1, std::numeric_limits<int>::max()}) {
    auto config = SmallConfig();
    config.feed_forward_width = invalid_width;
    EXPECT_EQ(
        ResolveMlpTransformerReadoutParameterBudget(config).status().code(),
        absl::StatusCode::kInvalidArgument);
  }
  auto config = SmallConfig();
  config.attention_heads = 3;
  EXPECT_EQ(ResolveMlpTransformerReadoutParameterBudget(config).status().code(),
            absl::StatusCode::kInvalidArgument);
}

class MlpTransformerReadoutTest : public testing::Test {
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
        "upload transformer readout test input"));
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
          "snapshot transformer readout bytes"));
      RETURN_IF_ERROR(executor_->Synchronize());
      result.emplace_back(host.begin(), host.end());
    }
    return result;
  }

  absl::Status Fill(const Buffer& buffer, float base) {
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<float>::Allocate(
                                    *executor_, buffer.size_bytes() / 4));
    for (size_t index = 0; index < host.size(); ++index)
      host[index] = base + static_cast<float>(index) / 1024;
    return cuda::CudaStatus(
        cudaMemcpyAsync(buffer.data(), host.data(), host.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "fill distinctive transformer readout tensor");
  }

  std::vector<uint16_t> HiddenValues() {
    const uint16_t pattern[] = {0x3f00, 0x3f80, 0x4000, 0xbf00, 0xbf80,
                                0x4040, 0x3e80, 0x4080, 0xc000, 0xbf40};
    std::vector<uint16_t> result(config_.context_length * config_.model_width);
    for (size_t index = 0; index < result.size(); ++index)
      result[index] = pattern[(index + index / config_.model_width) % 10];
    return result;
  }

  void ExpectFinite(const std::vector<uint8_t>& bytes) {
    ASSERT_EQ(bytes.size() % sizeof(float), 0u);
    for (size_t offset = 0; offset < bytes.size(); offset += sizeof(float)) {
      float value;
      std::memcpy(&value, bytes.data() + offset, sizeof(value));
      EXPECT_TRUE(std::isfinite(value)) << offset;
    }
  }

  bool HasNonzero(absl::Span<const uint8_t> bytes) {
    for (size_t offset = 0; offset + sizeof(float) <= bytes.size();
         offset += sizeof(float)) {
      float value;
      std::memcpy(&value, bytes.data() + offset, sizeof(value));
      if (value != 0)
        return true;
    }
    return false;
  }

  const Gpt2Config config_ = SmallConfig();
  std::unique_ptr<cuda::Executor> executor_;
  std::unique_ptr<ComposedLayer> source_;
};

TEST_F(MlpTransformerReadoutTest, ExactShapesAndIndependentFrozenOwnership) {
  ASSERT_TRUE(Fill(source_->weights()[50], 0.5f).ok());
  ASSERT_TRUE(Fill(source_->weights()[51], -0.25f).ok());
  auto before = Snapshot(source_->weights());
  ASSERT_TRUE(before.ok()) << before.status();
  auto readout = CreateMlpTransformerReadout(*executor_, *source_, config_, 3);
  ASSERT_TRUE(readout.ok()) << readout.status();
  EXPECT_EQ(readout->model->name(), "mlp_transformer_readout");
  ASSERT_NE(readout->trainable, nullptr);
  ASSERT_EQ(readout->trainable->weights().size(), 20u);
  ASSERT_EQ(readout->trainable->gradients().size(), 20u);
  ASSERT_EQ(readout->model->weights().size(), 21u);
  EXPECT_EQ(readout->model->input_types()[0],
            ActivationType(DataType::BF16, {-2, 8, 10}));
  EXPECT_EQ(readout->model->output_types()[0],
            ActivationType(DataType::FP32, {-2, 8, 16}));
  const size_t extents[] = {10,  10, 200, 20, 200, 10, 10,  10, 300, 30,
                            100, 10, 10,  10, 200, 20, 200, 10, 10,  10};
  int64_t total = 0;
  for (size_t index = 0; index < 20; ++index) {
    const auto& weight = readout->trainable->weights()[index];
    EXPECT_EQ(weight.size_bytes(), extents[index] * sizeof(float));
    EXPECT_EQ(readout->trainable->gradients()[index].size_bytes(),
              weight.size_bytes());
    EXPECT_EQ(weight.data(), readout->model->weights()[index].data());
    EXPECT_EQ(&weight.executor(), executor_.get());
    total += extents[index];
  }
  EXPECT_EQ(total, 1380);
  EXPECT_EQ(total, readout->parameter_budget.trainable_parameters);
  EXPECT_EQ(readout->parameter_budget.mlp_parameters, 860);
  auto bytes = Snapshot(readout->model->weights());
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  EXPECT_EQ((*bytes)[18], (*before)[50]);
  EXPECT_EQ((*bytes)[19], (*before)[51]);
  EXPECT_EQ(bytes->back(), before->front());
  EXPECT_EQ(readout->embedding->weight().data(),
            readout->model->weights().back().data());
  for (const auto& weight : readout->model->weights())
    for (const auto& original : source_->weights())
      EXPECT_NE(weight.data(), original.data());
  for (const auto& tensor : *bytes)
    ExpectFinite(tensor);
  for (size_t index : {0u, 1u, 3u, 5u, 6u, 7u, 9u, 11u, 12u, 13u, 15u, 17u})
    for (size_t offset = 0; offset < (*bytes)[index].size(); offset += 4) {
      float value;
      std::memcpy(&value, (*bytes)[index].data() + offset, sizeof(value));
      EXPECT_EQ(value, index == 0 || index == 6 || index == 12 ? 1.0f : 0.0f);
    }
  auto after = Snapshot(source_->weights());
  ASSERT_TRUE(after.ok()) << after.status();
  EXPECT_EQ(*before, *after);
  source_.reset();
  auto input = Upload(HiddenValues());
  ASSERT_TRUE(input.ok()) << input.status();
  EXPECT_TRUE(readout->model->fwd(*executor_, {&*input, 1}).ok());
}

TEST_F(MlpTransformerReadoutTest, CopiedSuffixExactlyReproducesSourceLogits) {
  // Include production sequence/vocabulary dimensions, two sequences per
  // batch, both embedding padding modes, and a multi-head configuration.
  for (int variant = 0; variant < 3; ++variant) {
    SCOPED_TRACE(variant);
    auto config = config_;
    if (variant == 1) {
      config.pad_vocabulary = true;
      config.attention_heads = 2;
    } else if (variant == 2) {
      config.context_length = 27;
      config.vocabulary_size = 4475;
    }
    auto source = CreateGpt2(*executor_, DataType::BF16, 17, config);
    ASSERT_TRUE(source.ok()) << source.status();
    for (size_t norm : {32u, 38u, 44u, 50u}) {
      ASSERT_TRUE(Fill((*source)->weights()[norm], 0.5f).ok());
      ASSERT_TRUE(Fill((*source)->weights()[norm + 1], -0.03f).ok());
    }
    auto control =
        CreateMlpTransformerReadout(*executor_, **source, config, 3, true);
    ASSERT_TRUE(control.ok()) << control.status();
    auto source_bytes = Snapshot((*source)->weights());
    auto control_bytes = Snapshot(control->model->weights());
    ASSERT_TRUE(source_bytes.ok()) << source_bytes.status();
    ASSERT_TRUE(control_bytes.ok()) << control_bytes.status();
    for (size_t index = 0; index < 20; ++index)
      EXPECT_EQ((*control_bytes)[index], (*source_bytes)[index + 32]);
    EXPECT_EQ(control_bytes->back(), source_bytes->front());
    std::vector<int32_t> values(2 * config.context_length);
    for (size_t index = 0; index < values.size(); ++index)
      values[index] = (31 * index + 7) % config.vocabulary_size;
    auto tokens = Upload(values);
    ASSERT_TRUE(tokens.ok()) << tokens.status();
    auto captured = CaptureThirdAttention(*executor_, **source, *tokens);
    ASSERT_TRUE(captured.ok()) << captured.status();
    auto output = control->model->fwd(*executor_, {&captured->hidden, 1});
    ASSERT_TRUE(output.ok()) << output.status();
    auto actual = Snapshot(output->outputs);
    auto expected = Snapshot({&captured->logits, 1});
    ASSERT_TRUE(actual.ok()) << actual.status();
    ASSERT_TRUE(expected.ok()) << expected.status();
    EXPECT_EQ(*actual, *expected);
  }
}

TEST_F(MlpTransformerReadoutTest,
       FreshInitializationMatchesMlpAndAttentionScales) {
  constexpr int kSeed = 13;
  auto readout =
      CreateMlpTransformerReadout(*executor_, *source_, config_, kSeed);
  ASSERT_TRUE(readout.ok()) << readout.status();
  auto bytes = Snapshot(readout->trainable->weights());
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  for (int mlp = 0; mlp < 2; ++mlp) {
    auto expected = CreateMlpReadout(*executor_, *source_, config_, 20,
                                     kSeed + 2 * mlp, 1, false);
    ASSERT_TRUE(expected.ok()) << expected.status();
    auto expected_bytes = Snapshot(expected->trainable->weights());
    ASSERT_TRUE(expected_bytes.ok()) << expected_bytes.status();
    for (size_t index = 0; index < 6; ++index)
      EXPECT_EQ((*bytes)[12 * mlp + index], (*expected_bytes)[index]);
  }
  for (int projection = 0; projection < 2; ++projection) {
    auto expected = FullyConnectedLayer::Create(
        *executor_, 10, projection == 0 ? 30 : 10, DataType::BF16, 8);
    ASSERT_TRUE(expected.ok()) << expected.status();
    ASSERT_TRUE((*expected)
                    ->InitializeNormal(projection == 0 ? 0.02f : 0.005f,
                                       kSeed + 4 + projection)
                    .ok());
    auto expected_bytes = Snapshot((*expected)->weights());
    ASSERT_TRUE(expected_bytes.ok()) << expected_bytes.status();
    for (size_t index = 0; index < 2; ++index)
      EXPECT_EQ((*bytes)[8 + 2 * projection + index], (*expected_bytes)[index]);
  }
  EXPECT_NE((*bytes)[2], (*bytes)[14]);
  EXPECT_NE((*bytes)[4], (*bytes)[16]);
}

TEST_F(MlpTransformerReadoutTest,
       FreshSeedsAreDeterministicAndDifferFromControl) {
  const int seed = std::numeric_limits<int>::max();
  auto a = CreateMlpTransformerReadout(*executor_, *source_, config_, seed);
  auto b = CreateMlpTransformerReadout(*executor_, *source_, config_, seed);
  auto c = CreateMlpTransformerReadout(*executor_, *source_, config_, seed - 1);
  auto control =
      CreateMlpTransformerReadout(*executor_, *source_, config_, 3, true);
  ASSERT_TRUE(a.ok()) << a.status();
  ASSERT_TRUE(b.ok()) << b.status();
  ASSERT_TRUE(c.ok()) << c.status();
  ASSERT_TRUE(control.ok()) << control.status();
  auto a_bytes = Snapshot(a->model->weights());
  auto b_bytes = Snapshot(b->model->weights());
  auto c_bytes = Snapshot(c->model->weights());
  auto control_bytes = Snapshot(control->model->weights());
  ASSERT_TRUE(a_bytes.ok()) << a_bytes.status();
  ASSERT_TRUE(b_bytes.ok()) << b_bytes.status();
  ASSERT_TRUE(c_bytes.ok()) << c_bytes.status();
  ASSERT_TRUE(control_bytes.ok()) << control_bytes.status();
  EXPECT_EQ(*a_bytes, *b_bytes);
  for (size_t index : {2u, 4u, 8u, 10u, 14u, 16u}) {
    EXPECT_NE((*a_bytes)[index], (*c_bytes)[index]);
    EXPECT_NE((*a_bytes)[index], (*control_bytes)[index]);
  }
  auto tokens = Upload<int32_t>({0, 1, 2, 3, 4, 5, 6, 0});
  ASSERT_TRUE(tokens.ok()) << tokens.status();
  auto captured = CaptureThirdAttention(*executor_, *source_, *tokens);
  ASSERT_TRUE(captured.ok()) << captured.status();
  auto a_output = a->model->fwd(*executor_, {&captured->hidden, 1});
  auto b_output = b->model->fwd(*executor_, {&captured->hidden, 1});
  ASSERT_TRUE(a_output.ok()) << a_output.status();
  ASSERT_TRUE(b_output.ok()) << b_output.status();
  auto a_logits = Snapshot(a_output->outputs);
  auto b_logits = Snapshot(b_output->outputs);
  auto source_logits = Snapshot({&captured->logits, 1});
  ASSERT_TRUE(a_logits.ok()) << a_logits.status();
  ASSERT_TRUE(b_logits.ok()) << b_logits.status();
  ASSERT_TRUE(source_logits.ok()) << source_logits.status();
  EXPECT_EQ(*a_logits, *b_logits);
  EXPECT_NE(*a_logits, *source_logits);
}

TEST_F(MlpTransformerReadoutTest,
       BackwardTrainsAllTensorsAndLeavesHeadAndSourceFrozen) {
  auto readout = CreateMlpTransformerReadout(*executor_, *source_, config_, 3);
  ASSERT_TRUE(readout.ok()) << readout.status();
  auto before = Snapshot(readout->model->weights());
  auto source_before = Snapshot(source_->weights());
  ASSERT_TRUE(before.ok()) << before.status();
  ASSERT_TRUE(source_before.ok()) << source_before.status();
  auto optimizer =
      AdamWOptimizer::Create(*executor_, *readout->trainable,
                             {.learning_rate = 0.001f, .weight_decay = 0});
  ASSERT_TRUE(optimizer.ok()) << optimizer.status();
  EXPECT_EQ((*optimizer)->parameter_tensor_count(), 20u);
  ASSERT_TRUE((*optimizer)->ZeroGrad().ok());
  auto input = Upload(HiddenValues());
  ASSERT_TRUE(input.ok()) << input.status();
  auto output = readout->model->fwd(*executor_, {&*input, 1});
  ASSERT_TRUE(output.ok()) << output.status();
  std::vector<float> upstream_values(8 * 16, 0);
  for (size_t index = 0; index < upstream_values.size(); ++index)
    if (index % 16 < 7)
      upstream_values[index] = (static_cast<int>(index % 5) - 2) / 8.0f;
  auto upstream = Upload(upstream_values);
  ASSERT_TRUE(upstream.ok()) << upstream.status();
  auto input_gradient = readout->model->bwd(*executor_, {&*upstream, 1},
                                            std::move(output->state));
  ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();
  ASSERT_EQ(input_gradient->size(), 1u);
  EXPECT_EQ(input_gradient->front().size_bytes(), 8 * 10 * sizeof(float));
  auto gradient_bytes = Snapshot(readout->trainable->gradients());
  auto input_bytes = Snapshot(*input_gradient);
  ASSERT_TRUE(gradient_bytes.ok()) << gradient_bytes.status();
  ASSERT_TRUE(input_bytes.ok()) << input_bytes.status();
  ExpectFinite(input_bytes->front());
  for (size_t index = 0; index < gradient_bytes->size(); ++index) {
    SCOPED_TRACE(index);
    ExpectFinite((*gradient_bytes)[index]);
    EXPECT_TRUE(HasNonzero((*gradient_bytes)[index]));
  }
  ASSERT_TRUE((*optimizer)->ApplyStep().ok());
  auto after = Snapshot(readout->model->weights());
  auto source_after = Snapshot(source_->weights());
  ASSERT_TRUE(after.ok()) << after.status();
  ASSERT_TRUE(source_after.ok()) << source_after.status();
  for (size_t index = 0; index < 20; ++index)
    EXPECT_NE((*before)[index], (*after)[index]) << index;
  EXPECT_EQ(before->back(), after->back());
  EXPECT_EQ(*source_before, *source_after);
}

TEST_F(MlpTransformerReadoutTest,
       CausalAttentionIgnoresFutureAndTrainsPromptPositions) {
  auto readout = CreateMlpTransformerReadout(*executor_, *source_, config_, 3);
  ASSERT_TRUE(readout.ok()) << readout.status();
  auto values = HiddenValues();
  auto changed = values;
  for (size_t index = 5 * 10; index < changed.size(); ++index)
    changed[index] ^= 0x8000;  // Change only future-token signs.
  auto a = Upload(values);
  auto b = Upload(changed);
  ASSERT_TRUE(a.ok()) << a.status();
  ASSERT_TRUE(b.ok()) << b.status();
  auto a_output = readout->model->fwd(*executor_, {&*a, 1});
  auto b_output = readout->model->fwd(*executor_, {&*b, 1});
  ASSERT_TRUE(a_output.ok()) << a_output.status();
  ASSERT_TRUE(b_output.ok()) << b_output.status();
  auto a_bytes = Snapshot(a_output->outputs);
  auto b_bytes = Snapshot(b_output->outputs);
  ASSERT_TRUE(a_bytes.ok()) << a_bytes.status();
  ASSERT_TRUE(b_bytes.ok()) << b_bytes.status();
  EXPECT_TRUE(std::equal(a_bytes->front().begin(),
                         a_bytes->front().begin() + 5 * 16 * sizeof(float),
                         b_bytes->front().begin()));
  EXPECT_NE(*a_bytes, *b_bytes);

  // Score only position 4. Earlier inputs have no direct logit gradient, but
  // attention must still send gradients to their prompt keys and values.
  auto optimizer = AdamWOptimizer::Create(*executor_, *readout->trainable, {});
  ASSERT_TRUE(optimizer.ok()) << optimizer.status();
  ASSERT_TRUE((*optimizer)->ZeroGrad().ok());
  std::vector<float> upstream_values(8 * 16, 0);
  upstream_values[4 * 16] = 1;
  upstream_values[4 * 16 + 1] = -1;
  auto upstream = Upload(upstream_values);
  ASSERT_TRUE(upstream.ok()) << upstream.status();
  auto input_gradient = readout->model->bwd(*executor_, {&*upstream, 1},
                                            std::move(a_output->state));
  ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();
  auto input_bytes = Snapshot(*input_gradient);
  ASSERT_TRUE(input_bytes.ok()) << input_bytes.status();
  ExpectFinite(input_bytes->front());
  const absl::Span<const uint8_t> gradients(input_bytes->front());
  for (size_t token = 0; token <= 4; ++token)
    EXPECT_TRUE(HasNonzero(
        gradients.subspan(token * 10 * sizeof(float), 10 * sizeof(float))))
        << token;
  EXPECT_FALSE(HasNonzero(gradients.subspan(5 * 10 * sizeof(float))));
}

TEST_F(MlpTransformerReadoutTest,
       RejectsWrongSourceShapesLayoutExecutorAndSeed) {
  EXPECT_EQ(CreateMlpTransformerReadout(*executor_, *source_, config_, -1)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  for (int mismatch = 0; mismatch < 6; ++mismatch) {
    auto config = config_;
    switch (mismatch) {
      case 0:
        config.transformer_block_count = 3;
        break;
      case 1:
        config.model_width = 12;
        break;
      case 2:
        config.feed_forward_width = 21;
        break;
      case 3:
        config.context_length = 9;
        break;
      case 4:
        config.vocabulary_size = 8;
        break;
      case 5:
        config.pad_vocabulary = true;
        break;
    }
    EXPECT_EQ(CreateMlpTransformerReadout(*executor_, *source_, config, 3)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument)
        << mismatch;
  }
  for (int blocks : {3, 5}) {
    auto config = config_;
    config.transformer_block_count = blocks;
    auto source = CreateGpt2(*executor_, DataType::BF16, 17, config);
    ASSERT_TRUE(source.ok()) << source.status();
    EXPECT_EQ(CreateMlpTransformerReadout(*executor_, **source, config_, 3)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  auto fp16 = CreateGpt2(*executor_, DataType::FP16, 17, config_);
  ASSERT_TRUE(fp16.ok()) << fp16.status();
  EXPECT_EQ(CreateMlpTransformerReadout(*executor_, **fp16, config_, 3)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  // A shape mismatch anywhere in the source is rejected, including tensors
  // before A3 that the readout neither reads nor copies.
  for (size_t index = 0; index < source_->weights().size(); ++index) {
    const auto original = source_->weights()[index];
    auto malformed = Buffer::Allocate(*executor_, original.size_bytes() + 4);
    ASSERT_TRUE(malformed.ok()) << malformed.status();
    source_->weights()[index] = *malformed;
    EXPECT_EQ(CreateMlpTransformerReadout(*executor_, *source_, config_, 3)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument)
        << index;
    source_->weights()[index] = original;
  }
  const auto original_head = source_->weights().back();
  auto untied = Buffer::Allocate(*executor_, original_head.size_bytes());
  ASSERT_TRUE(untied.ok()) << untied.status();
  source_->weights().back() = *untied;
  EXPECT_EQ(CreateMlpTransformerReadout(*executor_, *source_, config_, 3)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  source_->weights().back() = original_head;
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  EXPECT_EQ(CreateMlpTransformerReadout(**other, *source_, config_, 3)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  const auto original = source_->weights()[40];
  auto foreign = Buffer::Allocate(**other, original.size_bytes());
  ASSERT_TRUE(foreign.ok()) << foreign.status();
  source_->weights()[40] = *foreign;
  EXPECT_EQ(CreateMlpTransformerReadout(*executor_, *source_, config_, 3)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  source_->weights()[40] = original;
}

}  // namespace
}  // namespace pluto::llm::memorize_general_facts
