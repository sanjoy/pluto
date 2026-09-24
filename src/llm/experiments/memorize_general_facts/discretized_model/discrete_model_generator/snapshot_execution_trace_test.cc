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
const CaptureOptions kSmallOptions{.layers = 4,
                                   .vocab_size = 8,
                                   .eos_token = 7,
                                   .prompt_tokens = 5,
                                   .model_width = 13,
                                   .context_length = 32};
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
  explicit ScriptedGpt2(CaptureOptions options = kOptions)
      : options_(options),
        input_{{DataType::INT32, {kBatch, options.context_length}}},
        output_{{DataType::FP32,
                 {kBatch, options.context_length,
                  (options.vocab_size + 15) / 16 * 16}}} {}

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
  bool wrong_boundary_context = false;
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
    std::vector<uint16_t> values(options_.context_length *
                                 options_.model_width);
    for (int row = 0; row < options_.context_length; ++row)
      for (int channel = 0; channel < options_.model_width; ++channel)
        values[row * options_.model_width + channel] =
            Bits(stage, row, channel) ^ (future_leak ? real_rows : 0);
    if (short_storage)
      values.pop_back();
    ASSIGN_OR_RETURN(auto buffer, Upload(executor, values));
    const ActivationType type(
        boundary_dtype,
        {kBatch, options_.context_length + (wrong_boundary_context ? 1 : 0),
         wrong_shape ? 8 : options_.model_width});
    return hooks.activation_hook(executor, name, {&type, 1}, {&buffer, 1});
  }

  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks* hooks) const override {
    if (hooks == nullptr)
      return absl::InvalidArgumentError("script requires capture hooks");
    ASSIGN_OR_RETURN(auto context, cuda::PageLockedHostArray<int>::Allocate(
                                       executor, options_.context_length));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(context.data(), inputs[0].data(), context.size_bytes(),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "download scripted capture context"));
    RETURN_IF_ERROR(executor.Synchronize());
    const int real_rows = static_cast<int>(
        std::find(context.begin(), context.end(), options_.eos_token) -
        context.begin());
    prefixes.emplace_back(context.begin(), context.begin() + real_rows);
    RETURN_IF_ERROR(hooks->enter_combinator(executor, "gpt2"));
    RETURN_IF_ERROR(
        Emit(executor, *hooks, "PositionEmbeddingLayer", 0, real_rows));
    for (int block = 0; block < options_.layers; ++block) {
      RETURN_IF_ERROR(hooks->enter_combinator(
          executor, absl::StrCat("transformer_block_", block + wrong_block)));
      for (int branch = 0; branch < 2; ++branch) {
        if (block + 1 == options_.layers && branch == 1 && omit_last_boundary)
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
        RETURN_IF_ERROR(Emit(executor, *hooks, "ResidualLayer",
                             2 * block + branch + 1, real_rows));
        if (wrong_scope)
          RETURN_IF_ERROR(hooks->exit_combinator(executor));
      }
      RETURN_IF_ERROR(hooks->exit_combinator(executor));
    }
    RETURN_IF_ERROR(hooks->exit_combinator(executor));

    const int padded_vocabulary = (options_.vocab_size + 15) / 16 * 16;
    std::vector<float> logits(options_.context_length * padded_vocabulary,
                              std::numeric_limits<float>::quiet_NaN());
    for (int row = 0; row < real_rows; ++row) {
      // Padded vocabulary columns remain NaN even in real rows.
      std::fill_n(logits.begin() + row * padded_vocabulary, options_.vocab_size,
                  -10.0f);
      int prediction = context[row] + 1;
      if (missing_eos && prediction == options_.eos_token)
        prediction = 0;
      logits[row * padded_vocabulary + prediction] = 10.0f;
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

  CaptureOptions options_;
  std::vector<ActivationType> input_;
  std::vector<ActivationType> output_;
};

CapturedSample ValidSmallSample() {
  return {.tokens = kTokens,
          .predictions = {1, 2, 3, 4, 5, 6, 7},
          .boundaries = std::vector<std::vector<CapturedRow>>(
              2 * kSmallOptions.layers + 1,
              std::vector<CapturedRow>(
                  kTokens.size(), CapturedRow(kSmallOptions.model_width)))};
}

TEST(CaptureValidationTest, KeepsDefaultDimensions) {
  const CaptureOptions options;
  EXPECT_EQ(options.model_width, 16);
  EXPECT_EQ(options.context_length, 1024);
}

TEST(CaptureValidationTest, RejectsInvalidDimensionsAndPromptLength) {
  const auto sample = ValidSmallSample();
  ASSERT_TRUE(ValidateCapturedPredictions(sample, kSmallOptions).ok());
  for (int invalid : {0, -1}) {
    auto options = kSmallOptions;
    options.model_width = invalid;
    EXPECT_EQ(ValidateCapturedPredictions(sample, options).code(),
              absl::StatusCode::kInvalidArgument);
    options = kSmallOptions;
    options.context_length = invalid;
    EXPECT_EQ(ValidateCapturedPredictions(sample, options).code(),
              absl::StatusCode::kInvalidArgument);
  }
  auto options = kSmallOptions;
  options.prompt_tokens = options.context_length + 1;
  EXPECT_EQ(ValidateCapturedPredictions(sample, options).code(),
            absl::StatusCode::kInvalidArgument);
  options = kSmallOptions;
  options.context_length = kTokens.size() - 1;
  EXPECT_EQ(ValidateCapturedPredictions(sample, options).code(),
            absl::StatusCode::kInvalidArgument);
  for (int invalid : {-1, std::numeric_limits<int>::max()}) {
    options = kSmallOptions;
    options.layers = invalid;
    EXPECT_EQ(ValidateCapturedPredictions(sample, options).code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(CaptureValidationTest, RejectsMissingOrPaddedBoundaryRowsAndChannels) {
  auto sample = ValidSmallSample();
  sample.boundaries[0].pop_back();
  EXPECT_EQ(ValidateCapturedPredictions(sample, kSmallOptions).code(),
            absl::StatusCode::kInvalidArgument);
  sample = ValidSmallSample();
  sample.boundaries[0].push_back(CapturedRow(kSmallOptions.model_width));
  EXPECT_EQ(ValidateCapturedPredictions(sample, kSmallOptions).code(),
            absl::StatusCode::kInvalidArgument);
  sample = ValidSmallSample();
  sample.boundaries.back().back().pop_back();
  EXPECT_EQ(ValidateCapturedPredictions(sample, kSmallOptions).code(),
            absl::StatusCode::kInvalidArgument);
  sample = ValidSmallSample();
  sample.boundaries.back().back().push_back(0);
  EXPECT_EQ(ValidateCapturedPredictions(sample, kSmallOptions).code(),
            absl::StatusCode::kInvalidArgument);
}

class CaptureTest : public testing::Test {
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

TEST_F(CaptureTest, CapturesExactBitsAllRealRowsAndPostResidualBoundaries) {
  ScriptedGpt2 model;
  auto captured = CaptureSample(*executor_, model, kTokens, kOptions);
  ASSERT_TRUE(captured.ok()) << captured.status();
  EXPECT_EQ(captured->tokens, kTokens);
  EXPECT_EQ(captured->predictions, (std::vector<int>{1, 2, 3, 4, 5, 6, 7}));
  ASSERT_EQ(captured->boundaries.size(), 3u);
  for (size_t stage = 0; stage < captured->boundaries.size(); ++stage) {
    ASSERT_EQ(captured->boundaries[stage].size(), kTokens.size());
    for (size_t row = 0; row < kTokens.size(); ++row) {
      ASSERT_EQ(captured->boundaries[stage][row].size(),
                static_cast<size_t>(kOptions.model_width));
      for (int channel = 0; channel < kCaptureWidth; ++channel)
        EXPECT_EQ(captured->boundaries[stage][row][channel],
                  Bits(stage, row, channel));
    }
  }
  EXPECT_TRUE(ValidateCapturedPredictions(*captured, kOptions).ok());
}

TEST_F(CaptureTest, RejectsWrongDtypeShapeAndStorage) {
  ScriptedGpt2 model(kSmallOptions);
  model.boundary_dtype = DataType::FP32;
  EXPECT_FALSE(CaptureSample(*executor_, model, kTokens, kSmallOptions).ok());
  model.boundary_dtype = DataType::BF16;
  model.wrong_shape = true;
  EXPECT_FALSE(CaptureSample(*executor_, model, kTokens, kSmallOptions).ok());
  model.wrong_shape = false;
  model.wrong_boundary_context = true;
  EXPECT_FALSE(CaptureSample(*executor_, model, kTokens, kSmallOptions).ok());
  model.wrong_boundary_context = false;
  model.short_storage = true;
  EXPECT_FALSE(CaptureSample(*executor_, model, kTokens, kSmallOptions).ok());
}

TEST_F(CaptureTest, CapturesOddWidthSmallContextAndEveryGreedyPrefix) {
  ScriptedGpt2 model(kSmallOptions);
  auto captured = CaptureSample(*executor_, model, kTokens, kSmallOptions);
  ASSERT_TRUE(captured.ok()) << captured.status();
  EXPECT_EQ(captured->tokens, kTokens);
  EXPECT_EQ(captured->predictions, (std::vector<int>{1, 2, 3, 4, 5, 6, 7}));
  ASSERT_EQ(captured->boundaries.size(), 9u);
  for (size_t stage = 0; stage < captured->boundaries.size(); ++stage) {
    ASSERT_EQ(captured->boundaries[stage].size(), kTokens.size());
    for (size_t row = 0; row < kTokens.size(); ++row) {
      ASSERT_EQ(captured->boundaries[stage][row].size(), 13u);
      for (int channel = 0; channel < kSmallOptions.model_width; ++channel)
        EXPECT_EQ(captured->boundaries[stage][row][channel],
                  Bits(stage, row, channel));
    }
  }
  ASSERT_TRUE(ValidateCapturedPredictions(*captured, kSmallOptions).ok());
  model.prefixes.clear();
  const auto verified =
      VerifyGreedyCapture(*executor_, model, *captured, kSmallOptions);
  EXPECT_TRUE(verified.ok()) << verified;
  EXPECT_EQ(model.prefixes,
            (std::vector<std::vector<int>>{
                {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5, 6}}));
  model.future_leak = true;
  EXPECT_EQ(
      VerifyGreedyCapture(*executor_, model, *captured, kSmallOptions).code(),
      absl::StatusCode::kDataLoss);
}

TEST_F(CaptureTest, RejectsInvalidCaptureRequestsBeforeForward) {
  ScriptedGpt2 model(kSmallOptions);
  for (int invalid : {0, -1}) {
    auto options = kSmallOptions;
    options.model_width = invalid;
    EXPECT_EQ(
        CaptureSample(*executor_, model, kTokens, options).status().code(),
        absl::StatusCode::kInvalidArgument);
    options = kSmallOptions;
    options.context_length = invalid;
    EXPECT_EQ(
        CaptureSample(*executor_, model, kTokens, options).status().code(),
        absl::StatusCode::kInvalidArgument);
  }
  auto options = kSmallOptions;
  options.prompt_tokens = options.context_length + 1;
  EXPECT_EQ(CaptureSample(*executor_, model, kTokens, options).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(CaptureSample(*executor_, model, {}, kSmallOptions).status().code(),
            absl::StatusCode::kInvalidArgument);
  const std::vector<int> too_long(kSmallOptions.context_length + 1, 0);
  EXPECT_EQ(
      CaptureSample(*executor_, model, too_long, kSmallOptions).status().code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(model.prefixes.empty());
}

TEST_F(CaptureTest, RejectsConfiguredDimensionsThatDifferFromModel) {
  ScriptedGpt2 model(kSmallOptions);
  auto options = kSmallOptions;
  --options.context_length;
  EXPECT_EQ(CaptureSample(*executor_, model, kTokens, options).status().code(),
            absl::StatusCode::kInvalidArgument);
  options = kSmallOptions;
  ++options.model_width;
  EXPECT_EQ(CaptureSample(*executor_, model, kTokens, options).status().code(),
            absl::StatusCode::kInvalidArgument);
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
  for (auto options : {kOptions, kSmallOptions}) {
    SCOPED_TRACE(absl::StrCat("width=", options.model_width,
                              ", context=", options.context_length));
    options.vocab_size = 16;
    options.eos_token = 15;
    const Gpt2Config config{.transformer_block_count = options.layers,
                            .model_width = options.model_width,
                            .attention_heads = 1,
                            .feed_forward_width = 4 * options.model_width,
                            .vocabulary_size = options.vocab_size,
                            .pad_vocabulary = false,
                            .context_length = options.context_length};
    auto model = CreateGpt2(*executor_, DataType::BF16, 1337, config);
    ASSERT_TRUE(model.ok()) << model.status();
    auto full = CaptureSample(*executor_, **model, kTokens, options);
    ASSERT_TRUE(full.ok()) << full.status();
    auto prefix = CaptureSample(*executor_, **model,
                                absl::MakeConstSpan(kTokens).first(5), options);
    ASSERT_TRUE(prefix.ok()) << prefix.status();
    ASSERT_EQ(full->boundaries.size(),
              static_cast<size_t>(2 * options.layers + 1));
    for (size_t stage = 0; stage < full->boundaries.size(); ++stage) {
      ASSERT_EQ(full->boundaries[stage].size(), kTokens.size());
      for (const auto& row : full->boundaries[stage])
        EXPECT_EQ(row.size(), static_cast<size_t>(options.model_width));
      EXPECT_TRUE(std::equal(prefix->boundaries[stage].begin(),
                             prefix->boundaries[stage].end(),
                             full->boundaries[stage].begin()));
    }
    EXPECT_TRUE(std::equal(prefix->predictions.begin(),
                           prefix->predictions.end(),
                           full->predictions.begin()));
  }
}

TEST_F(CaptureTest, Width10Context27NativeBoundariesPreserveEveryPrefix) {
  const CaptureOptions options{.layers = 4,
                               .vocab_size = 4475,
                               .eos_token = 4474,
                               .prompt_tokens = 5,
                               .model_width = 10,
                               .context_length = 27};
  const Gpt2Config config{.transformer_block_count = options.layers,
                          .model_width = options.model_width,
                          .attention_heads = 1,
                          .feed_forward_width = 20,
                          .vocabulary_size = options.vocab_size,
                          .pad_vocabulary = false,
                          .context_length = options.context_length};
  auto model = CreateGpt2(*executor_, DataType::BF16, 1337, config);
  ASSERT_TRUE(model.ok()) << model.status();
  // Exercise the final valid position and every shorter generation prefix.
  // Both logical widths and context length have partially filled compute tiles.
  std::vector<int> tokens(options.context_length);
  for (size_t position = 0; position < tokens.size(); ++position)
    tokens[position] = kTokens[position % kTokens.size()];
  auto full = CaptureSample(*executor_, **model, tokens, options);
  ASSERT_TRUE(full.ok()) << full.status();
  ASSERT_EQ(full->boundaries.size(), 9u);
  for (const auto& boundary : full->boundaries) {
    ASSERT_EQ(boundary.size(), 27u);
    for (const auto& row : boundary)
      EXPECT_EQ(row.size(), 10u);
  }
  for (size_t length = options.prompt_tokens; length < tokens.size();
       ++length) {
    SCOPED_TRACE(absl::StrCat("prefix length=", length));
    auto prefix =
        CaptureSample(*executor_, **model,
                      absl::MakeConstSpan(tokens).first(length), options);
    ASSERT_TRUE(prefix.ok()) << prefix.status();
    ASSERT_EQ(prefix->boundaries.size(), full->boundaries.size());
    for (size_t stage = 0; stage < full->boundaries.size(); ++stage) {
      ASSERT_EQ(prefix->boundaries[stage].size(), length);
      EXPECT_TRUE(std::equal(prefix->boundaries[stage].begin(),
                             prefix->boundaries[stage].end(),
                             full->boundaries[stage].begin()));
    }
    EXPECT_TRUE(std::equal(prefix->predictions.begin(),
                           prefix->predictions.end(),
                           full->predictions.begin()));
  }
}

}  // namespace
}  // namespace pluto::llm::memorize_general_facts::discretized_model
