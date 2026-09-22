#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/snapshot_execution_trace.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/gpt2.h"
#include "src/llm/layer_hooks.h"
#include "src/util/status_macros.h"

namespace pluto::llm::memorize_general_facts::discretized_model {
namespace {

constexpr int64_t kBatch = ActivationType::kBatchDimension;
const CaptureOptions kOptions{
    .layers = 1, .vocab_size = 8, .eos_token = 7, .prompt_tokens = 5};
const std::vector<int> kTokens{0, 1, 2, 3, 4, 5, 6};

template <class T>
absl::StatusOr<Buffer> Upload(cuda::Executor& executor,
                              const std::vector<T>& values) {
  ASSIGN_OR_RETURN(auto staging,
                   cuda::PageLockedHostArray<T>::CopyFrom(executor, values));
  ASSIGN_OR_RETURN(auto buffer,
                   Buffer::Allocate(executor, staging.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(buffer.data(), staging.data(), staging.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload scripted capture values"));
  return buffer;
}

// Deliberately includes signed zero, subnormal, infinity, and NaN bit patterns:
// a boundary capture preserves storage, and must never round-trip through
// float.
uint16_t Bits(int stage, int row, int channel) {
  if (channel < 4) {
    constexpr uint16_t special[]{0x8000, 0x0001, 0x7f80, 0x7fc1};
    return special[channel];
  }
  return static_cast<uint16_t>(stage * 4096 + row * 16 + channel);
}

// Models the real combinator callback protocol with independently scripted
// GPU storage. The top-1 kernel still performs actual vocabulary selection.
class ScriptedGpt2 final : public Layer {
 public:
  absl::string_view name() const override { return "gpt2"; }
  absl::Span<const ActivationType> input_types() const override {
    return input_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return output_;
  }
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::BF16; }

  DataType boundary_dtype = DataType::BF16;
  bool wrong_shape = false;
  bool short_storage = false;
  bool wrong_block = false;
  bool wrong_scope = false;
  bool reverse_branches = false;
  bool omit_last_boundary = false;
  bool nonfinite_prompt = false;
  bool missing_eos = false;
  bool future_leak = false;
  mutable std::vector<std::vector<int>> prefixes;

 private:
  absl::Status Emit(cuda::Executor& executor, LayerHooks& hooks,
                    absl::string_view name, int stage, int real_rows) const {
    std::vector<uint16_t> values(kCaptureContext * kCaptureWidth);
    for (int row = 0; row < kCaptureContext; ++row)
      for (int channel = 0; channel < kCaptureWidth; ++channel)
        values[row * kCaptureWidth + channel] =
            Bits(stage, row, channel) ^ (future_leak ? real_rows : 0);
    if (short_storage)
      values.pop_back();
    ASSIGN_OR_RETURN(auto buffer, Upload(executor, values));
    const ActivationType type(
        boundary_dtype,
        {kBatch, kCaptureContext, wrong_shape ? 8 : kCaptureWidth});
    return hooks.activation_hook(executor, name, {&type, 1}, {&buffer, 1});
  }

  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks* hooks) const override {
    if (hooks == nullptr)
      return absl::InvalidArgumentError("script requires capture hooks");
    ASSIGN_OR_RETURN(auto context, cuda::PageLockedHostArray<int>::Allocate(
                                       executor, kCaptureContext));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(context.data(), inputs[0].data(), context.size_bytes(),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "download scripted capture context"));
    RETURN_IF_ERROR(executor.Synchronize());
    const int real_rows = static_cast<int>(
        std::find(context.begin(), context.end(), kOptions.eos_token) -
        context.begin());
    prefixes.emplace_back(context.begin(), context.begin() + real_rows);
    RETURN_IF_ERROR(hooks->enter_combinator(executor, "gpt2"));
    RETURN_IF_ERROR(
        Emit(executor, *hooks, "PositionEmbeddingLayer", 0, real_rows));
    RETURN_IF_ERROR(hooks->enter_combinator(
        executor, wrong_block ? "transformer_block_1" : "transformer_block_0"));
    for (int branch = 0; branch < 2; ++branch) {
      if (branch == 1 && omit_last_boundary)
        break;
      RETURN_IF_ERROR(hooks->enter_combinator(executor, "ResidualLayer"));
      const char* branch_name =
          (branch == 0) != reverse_branches ? "attention" : "mlp";
      RETURN_IF_ERROR(hooks->enter_combinator(executor, branch_name));
      RETURN_IF_ERROR(hooks->exit_combinator(executor));
      // A branch output is observed while still inside its residual wrapper.
      RETURN_IF_ERROR(
          Emit(executor, *hooks, branch_name, 90 + branch, real_rows));
      if (!wrong_scope)
        RETURN_IF_ERROR(hooks->exit_combinator(executor));
      RETURN_IF_ERROR(
          Emit(executor, *hooks, "ResidualLayer", branch + 1, real_rows));
      if (wrong_scope)
        RETURN_IF_ERROR(hooks->exit_combinator(executor));
    }
    RETURN_IF_ERROR(hooks->exit_combinator(executor));
    RETURN_IF_ERROR(hooks->exit_combinator(executor));

    std::vector<float> logits(kCaptureContext * 16,
                              std::numeric_limits<float>::quiet_NaN());
    for (int row = 0; row < real_rows; ++row) {
      // Padded vocabulary columns remain NaN even in real rows.
      std::fill_n(logits.begin() + row * 16, kOptions.vocab_size, -10.0f);
      int prediction = context[row] + 1;
      if (missing_eos && prediction == kOptions.eos_token)
        prediction = 0;
      logits[row * 16 + prediction] = 10.0f;
    }
    if (nonfinite_prompt)
      logits[0] = std::numeric_limits<float>::infinity();
    ASSIGN_OR_RETURN(auto output, Upload(executor, logits));
    FwdResult result;
    result.outputs.push_back(std::move(output));
    return result;
  }

  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override {
    return absl::InternalError("capture must not call backward");
  }

  std::vector<ActivationType> input_{
      {DataType::INT32, {kBatch, kCaptureContext}}};
  std::vector<ActivationType> output_{
      {DataType::FP32, {kBatch, kCaptureContext, 16}}};
};

class CaptureTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }
  void TearDown() override {
    if (executor_ != nullptr)
      EXPECT_TRUE(executor_->Synchronize().ok());
  }
  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(CaptureTest, CapturesExactBitsAllRealRowsAndPostResidualBoundaries) {
  ScriptedGpt2 model;
  auto captured = CaptureSample(*executor_, model, kTokens, kOptions);
  ASSERT_TRUE(captured.ok()) << captured.status();
  EXPECT_EQ(captured->tokens, kTokens);
  EXPECT_EQ(captured->predictions, (std::vector<int>{1, 2, 3, 4, 5, 6, 7}));
  ASSERT_EQ(captured->boundaries.size(), 3u);
  for (size_t stage = 0; stage < captured->boundaries.size(); ++stage) {
    ASSERT_EQ(captured->boundaries[stage].size(), kTokens.size());
    for (size_t row = 0; row < kTokens.size(); ++row)
      for (int channel = 0; channel < kCaptureWidth; ++channel)
        EXPECT_EQ(captured->boundaries[stage][row][channel],
                  Bits(stage, row, channel));
  }
  EXPECT_TRUE(ValidateCapturedPredictions(*captured, kOptions).ok());
}

TEST_F(CaptureTest, RejectsWrongDtypeShapeAndStorage) {
  ScriptedGpt2 model;
  model.boundary_dtype = DataType::FP32;
  EXPECT_FALSE(CaptureSample(*executor_, model, kTokens, kOptions).ok());
  model.boundary_dtype = DataType::BF16;
  model.wrong_shape = true;
  EXPECT_FALSE(CaptureSample(*executor_, model, kTokens, kOptions).ok());
  model.wrong_shape = false;
  model.short_storage = true;
  EXPECT_FALSE(CaptureSample(*executor_, model, kTokens, kOptions).ok());
}

TEST_F(CaptureTest, RejectsWrongScopesBranchOrderAndMissingStages) {
  ScriptedGpt2 model;
  model.wrong_block = true;
  EXPECT_FALSE(CaptureSample(*executor_, model, kTokens, kOptions).ok());
  model.wrong_block = false;
  model.wrong_scope = true;
  EXPECT_FALSE(CaptureSample(*executor_, model, kTokens, kOptions).ok());
  model.wrong_scope = false;
  model.reverse_branches = true;
  EXPECT_FALSE(CaptureSample(*executor_, model, kTokens, kOptions).ok());
  model.reverse_branches = false;
  model.omit_last_boundary = true;
  EXPECT_FALSE(CaptureSample(*executor_, model, kTokens, kOptions).ok());
}

TEST_F(CaptureTest, IncludesPromptLogitsButNeverReadsPaddingLogits) {
  ScriptedGpt2 model;
  ASSERT_TRUE(CaptureSample(*executor_, model, kTokens, kOptions).ok());
  model.nonfinite_prompt = true;
  const auto failed = CaptureSample(*executor_, model, kTokens, kOptions);
  ASSERT_FALSE(failed.ok());
  EXPECT_EQ(failed.status().code(), absl::StatusCode::kDataLoss);
}

TEST_F(CaptureTest, ScoresContinuationAndFinalEosOnly) {
  ScriptedGpt2 model;
  auto captured = CaptureSample(*executor_, model, kTokens, kOptions);
  ASSERT_TRUE(captured.ok()) << captured.status();
  std::fill_n(captured->predictions.begin(), 4, 0);
  EXPECT_TRUE(ValidateCapturedPredictions(*captured, kOptions).ok());
  captured->predictions[4] = 0;
  EXPECT_FALSE(ValidateCapturedPredictions(*captured, kOptions).ok());
  captured->predictions[4] = 5;
  captured->predictions.back() = 0;
  EXPECT_FALSE(ValidateCapturedPredictions(*captured, kOptions).ok());
}

TEST_F(CaptureTest, AutonomousFeedbackVerifiesExplicitFinalEosAndEveryPrefix) {
  ScriptedGpt2 model;
  auto captured = CaptureSample(*executor_, model, kTokens, kOptions);
  ASSERT_TRUE(captured.ok()) << captured.status();
  model.prefixes.clear();
  const auto verified =
      VerifyGreedyCapture(*executor_, model, *captured, kOptions);
  EXPECT_TRUE(verified.ok()) << verified;
  EXPECT_EQ(model.prefixes,
            (std::vector<std::vector<int>>{
                {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5, 6}}));
  model.missing_eos = true;
  EXPECT_FALSE(
      VerifyGreedyCapture(*executor_, model, *captured, kOptions).ok());
}

TEST_F(CaptureTest, RejectsFutureTokenInfluenceOnEarlierBoundaryBits) {
  ScriptedGpt2 model;
  model.future_leak = true;
  auto captured = CaptureSample(*executor_, model, kTokens, kOptions);
  ASSERT_TRUE(captured.ok()) << captured.status();
  const auto verified =
      VerifyGreedyCapture(*executor_, model, *captured, kOptions);
  EXPECT_EQ(verified.code(), absl::StatusCode::kDataLoss);
  EXPECT_NE(verified.message().find("prefix BF16 states differ"),
            absl::string_view::npos);
}

TEST_F(CaptureTest, TinyNativeGpt2HasExactCausalBoundaryPrefixes) {
  const Gpt2Config config{.transformer_block_count = 1,
                          .model_width = 16,
                          .attention_heads = 1,
                          .feed_forward_width = 64,
                          .vocabulary_size = 16,
                          .pad_vocabulary = false};
  auto model = CreateGpt2(*executor_, DataType::BF16, 1337, config);
  ASSERT_TRUE(model.ok()) << model.status();
  const CaptureOptions options{
      .layers = 1, .vocab_size = 16, .eos_token = 15, .prompt_tokens = 5};
  auto full = CaptureSample(*executor_, **model, kTokens, options);
  ASSERT_TRUE(full.ok()) << full.status();
  auto prefix = CaptureSample(*executor_, **model,
                              absl::MakeConstSpan(kTokens).first(5), options);
  ASSERT_TRUE(prefix.ok()) << prefix.status();
  ASSERT_EQ(full->boundaries.size(), 3u);
  for (size_t stage = 0; stage < full->boundaries.size(); ++stage)
    EXPECT_TRUE(std::equal(prefix->boundaries[stage].begin(),
                           prefix->boundaries[stage].end(),
                           full->boundaries[stage].begin()));
  EXPECT_TRUE(std::equal(prefix->predictions.begin(), prefix->predictions.end(),
                         full->predictions.begin()));
}

}  // namespace
}  // namespace pluto::llm::memorize_general_facts::discretized_model
