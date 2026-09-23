#include "src/llm/experiments/completion_trace/trace.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/gpt2.h"
#include "src/llm/layer_hooks.h"
#include "src/util/status_macros.h"

namespace pluto::llm::completion_trace {
namespace {

template <typename T>
std::vector<uint8_t> Bytes(absl::Span<const T> values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

absl::StatusOr<Buffer> Upload(cuda::Executor& executor,
                              absl::Span<const uint8_t> bytes) {
  ASSIGN_OR_RETURN(auto staging, cuda::PageLockedHostArray<uint8_t>::CopyFrom(
                                     executor, bytes));
  ASSIGN_OR_RETURN(auto buffer, Buffer::Allocate(executor, bytes.size()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(buffer.data(), staging.data(), bytes.size(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload trace test tensor"));
  return buffer;
}

absl::StatusOr<std::vector<uint8_t>> CopyBytes(cuda::Executor& executor,
                                               const Buffer& buffer) {
  ASSIGN_OR_RETURN(auto staging, cuda::PageLockedHostArray<uint8_t>::Allocate(
                                     executor, buffer.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(staging.data(), buffer.data(), buffer.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "download trace test tensor"));
  RETURN_IF_ERROR(executor.Synchronize());
  return std::vector<uint8_t>(staging.begin(), staging.end());
}

// A deliberately tiny instrumented layer exercises malformed hooks without
// replacing production model math in the GPT-2 test below. Its recorded input
// makes future-token padding directly observable instead of merely assumed.
class ScriptedLayer final : public Layer {
 public:
  enum class Scope { kNormal, kMissingExit, kExtraExit };
  absl::string_view name() const override { return "ScriptedLayer"; }
  absl::Span<const ActivationType> input_types() const override {
    return inputs;
  }
  absl::Span<const ActivationType> output_types() const override {
    return outputs;
  }
  absl::Span<Buffer> weights() override { return absl::MakeSpan(parameters); }
  DataType output_type() const override { return DataType::FP16; }

  std::vector<ActivationType> inputs{{DataType::INT32, {-2, 4}}};
  std::vector<ActivationType> outputs{{DataType::FP32, {-2, 4, 4}}};
  BufferVec parameters;
  Scope scope = Scope::kNormal;
  bool emit_probe = false;
  bool emit_attention = false;
  bool mismatch_count = false;
  bool truncate_output = false;
  bool nonfinite_logit = false;
  DataType probe_type = DataType::INT32;
  std::vector<uint8_t> probe_bytes;
  std::vector<float> probabilities;
  mutable std::vector<int> received_tokens;

 private:
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> input,
                                     LayerHooks* hooks) const override {
    ASSIGN_OR_RETURN(auto raw_input, CopyBytes(executor, input[0]));
    received_tokens.resize(raw_input.size() / sizeof(int));
    std::memcpy(received_tokens.data(), raw_input.data(), raw_input.size());
    if (hooks != nullptr) {
      if (scope == Scope::kExtraExit)
        RETURN_IF_ERROR(hooks->exit_combinator(executor));
      RETURN_IF_ERROR(hooks->enter_combinator(executor, "nested"));
      if (emit_probe) {
        ASSIGN_OR_RETURN(auto probe, Upload(executor, probe_bytes));
        const ActivationType type(probe_type, {-2, 4, 2});
        RETURN_IF_ERROR(hooks->activation_hook(
            executor, "Probe", {&type, mismatch_count ? 0u : 1u}, {&probe, 1}));
      }
      if (emit_attention) {
        ASSIGN_OR_RETURN(auto probabilities_buffer,
                         Upload(executor, Bytes<float>(probabilities)));
        const ActivationType type(DataType::FP32, {1, 1, 4, 4});
        RETURN_IF_ERROR(hooks->attention_probabilities_hook(
            executor, "Attention", type, probabilities_buffer));
      }
      if (scope != Scope::kMissingExit)
        RETURN_IF_ERROR(hooks->exit_combinator(executor));
    }
    std::vector<float> logits(16);
    for (size_t i = 0; i < logits.size(); ++i)
      logits[i] = i % 4 == 3 ? -std::numeric_limits<float>::infinity()
                             : static_cast<float>(i);
    if (nonfinite_logit)
      logits[0] = std::numeric_limits<float>::quiet_NaN();
    if (truncate_output)
      logits.pop_back();
    ASSIGN_OR_RETURN(auto output, Upload(executor, Bytes<float>(logits)));
    return FwdResult{.outputs = {std::move(output)}};
  }

  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override {
    return absl::InternalError("trace must never call backward");
  }
};

class CompletionTraceTest : public testing::Test {
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

TEST_F(CompletionTraceTest, PrefixOnlyInputAndCompletePhysicalOutputRows) {
  ScriptedLayer model;
  auto trace = TraceForward(*executor_, model, {0, 1}, 2, 3);
  ASSERT_TRUE(trace.ok()) << trace.status();
  EXPECT_EQ(model.received_tokens, (std::vector<int>{0, 1, 2, 2}));
  EXPECT_EQ(trace->prefix, (std::vector<int>{0, 1}));
  EXPECT_EQ(trace->next_logits, (std::vector<float>{4, 5, 6}));
  ASSERT_EQ(trace->activations.size(), 1u);
  const auto& output = trace->activations[0];
  EXPECT_EQ(output.scope, "");
  EXPECT_EQ(output.name, "ScriptedLayer");
  EXPECT_EQ(output.dimensions, (std::vector<int64_t>{1, 2, 4}));
  EXPECT_EQ(output.values.size(), 8u);
  EXPECT_EQ(output.raw_bytes.size(), 8 * sizeof(float));
  EXPECT_EQ(output.values[3], -std::numeric_limits<float>::infinity());
  EXPECT_EQ(output.values[7], -std::numeric_limits<float>::infinity());
}

TEST_F(CompletionTraceTest, PhysicalDtypeDecodingPreservesOriginalBits) {
  struct Case {
    DataType type;
    std::vector<uint8_t> bytes;
    std::vector<float> expected;
  };
  for (const auto& test :
       std::vector<Case>{{DataType::BF16,
                          Bytes<uint16_t>({0x3f80, 0x8000, 0xc020, 0x0001}),
                          {1, -0.0f, -2.5f, std::ldexp(1.0f, -133)}},
                         {DataType::FP16,
                          Bytes<uint16_t>({0x3c00, 0x8000, 0xc100, 0x0001}),
                          {1, -0.0f, -2.5f, std::ldexp(1.0f, -24)}},
                         {DataType::FP32,
                          Bytes<float>({1, -0.0f, -2.5f, 1.0e-30f}),
                          {1, -0.0f, -2.5f, 1.0e-30f}},
                         {DataType::INT32,
                          Bytes<int32_t>({1, -2, 4474, 0}),
                          {1, -2, 4474, 0}}}) {
    SCOPED_TRACE(static_cast<int>(test.type));
    ScriptedLayer model;
    model.emit_probe = true;
    model.probe_type = test.type;
    model.probe_bytes = test.bytes;
    model.probe_bytes.resize(test.bytes.size() * 2);
    auto trace = TraceForward(*executor_, model, {0, 1}, 2, 3);
    ASSERT_TRUE(trace.ok()) << trace.status();
    ASSERT_EQ(trace->activations.size(), 2u);
    const auto& probe = trace->activations[0];
    EXPECT_EQ(probe.scope, "nested");
    EXPECT_EQ(probe.raw_bytes, test.bytes);
    ASSERT_EQ(probe.values.size(), test.expected.size());
    for (size_t i = 0; i < test.expected.size(); ++i)
      EXPECT_EQ(std::bit_cast<uint32_t>(probe.values[i]),
                std::bit_cast<uint32_t>(test.expected[i]));
  }
}

TEST_F(CompletionTraceTest, RejectsMalformedScopesCountsSizesAndDtypes) {
  for (auto scope :
       {ScriptedLayer::Scope::kMissingExit, ScriptedLayer::Scope::kExtraExit}) {
    ScriptedLayer model;
    model.scope = scope;
    EXPECT_EQ(TraceForward(*executor_, model, {0}, 2, 3).status().code(),
              absl::StatusCode::kFailedPrecondition);
  }
  ScriptedLayer model;
  model.emit_probe = true;
  model.probe_bytes = Bytes<int32_t>({0, 1, 2, 3, 4, 5, 6, 7});
  model.mismatch_count = true;
  EXPECT_EQ(TraceForward(*executor_, model, {0}, 2, 3).status().code(),
            absl::StatusCode::kInvalidArgument);
  model.mismatch_count = false;
  model.probe_bytes.pop_back();
  EXPECT_EQ(TraceForward(*executor_, model, {0}, 2, 3).status().code(),
            absl::StatusCode::kInvalidArgument);
  model.probe_type = DataType::FP8;
  EXPECT_EQ(TraceForward(*executor_, model, {0}, 2, 3).status().code(),
            absl::StatusCode::kUnimplemented);
  model.emit_probe = false;
  model.truncate_output = true;
  EXPECT_EQ(TraceForward(*executor_, model, {0}, 2, 3).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(CompletionTraceTest, RejectsInvalidPrefixSignatureAndLogicalLogits) {
  ScriptedLayer model;
  for (const std::vector<int>& prefix :
       {std::vector<int>{}, {-1}, {3}, {0, 0, 0, 0, 0}})
    EXPECT_EQ(TraceForward(*executor_, model, prefix, 2, 3).status().code(),
              absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(TraceForward(*executor_, model, {0}, -1, 3).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(TraceForward(*executor_, model, {0}, 2, 5).status().code(),
            absl::StatusCode::kInvalidArgument);
  model.inputs[0] = ActivationType(DataType::FP32, {-2, 4});
  EXPECT_EQ(TraceForward(*executor_, model, {0}, 2, 3).status().code(),
            absl::StatusCode::kInvalidArgument);
  model.inputs[0] = ActivationType(DataType::INT32, {-2, 4});
  model.nonfinite_logit = true;
  EXPECT_EQ(TraceForward(*executor_, model, {0}, 2, 3).status().code(),
            absl::StatusCode::kDataLoss);
}

TEST_F(CompletionTraceTest, AttentionCaptureTrimsAndValidatesCausalSquare) {
  ScriptedLayer model;
  model.emit_attention = true;
  model.probabilities = {1,   0,   0,   0, .25f, .75f, 0,   0,
                         .1f, .2f, .7f, 0, .1f,  .2f,  .3f, .4f};
  auto trace = TraceForward(*executor_, model, {0, 1}, 2, 3);
  ASSERT_TRUE(trace.ok()) << trace.status();
  ASSERT_EQ(trace->attention.size(), 1u);
  EXPECT_EQ(trace->attention[0].dimensions, (std::vector<int64_t>{1, 1, 2, 2}));
  EXPECT_EQ(trace->attention[0].values, (std::vector<float>{1, 0, .25f, .75f}));
  EXPECT_EQ(trace->attention[0].raw_bytes, Bytes<float>({1, 0, .25f, .75f}));
  model.probabilities[1] = .1f;
  EXPECT_EQ(TraceForward(*executor_, model, {0, 1}, 2, 3).status().code(),
            absl::StatusCode::kDataLoss);
  model.probabilities[1] = 0;
  model.probabilities[5] = .25f;
  EXPECT_EQ(TraceForward(*executor_, model, {0, 1}, 2, 3).status().code(),
            absl::StatusCode::kDataLoss);
}

TEST_F(CompletionTraceTest, Gpt2CaptureIsCausalReadOnlyAndBitwiseStable) {
  // Two heads verify head-major attention slicing; two blocks verify repeated
  // primitive names get distinct scope paths, with local occurrence counters.
  const Gpt2Config config{.transformer_block_count = 2,
                          .model_width = 32,
                          .attention_heads = 2,
                          .feed_forward_width = 64,
                          .vocabulary_size = 19,
                          .pad_vocabulary = false};
  for (DataType type : {DataType::BF16, DataType::FP16}) {
    SCOPED_TRACE(static_cast<int>(type));
    auto model = CreateGpt2(*executor_, type, 123, config);
    ASSERT_TRUE(model.ok()) << model.status();
    std::vector<std::vector<uint8_t>> original_weights;
    for (const auto& weight : (*model)->weights()) {
      auto bytes = CopyBytes(*executor_, weight);
      ASSERT_TRUE(bytes.ok()) << bytes.status();
      original_weights.push_back(std::move(*bytes));
    }
    const std::vector<int> prefix{1, 4, 7};
    auto plain = PredictNextLogits(*executor_, **model, prefix, 18, 19);
    auto traced = TraceForward(*executor_, **model, prefix, 18, 19);
    auto repeated = TraceForward(*executor_, **model, prefix, 17, 19);
    ASSERT_TRUE(plain.ok()) << plain.status();
    ASSERT_TRUE(traced.ok()) << traced.status();
    ASSERT_TRUE(repeated.ok()) << repeated.status();
    EXPECT_EQ(Bytes<float>(*plain), Bytes<float>(traced->next_logits));
    EXPECT_EQ(Bytes<float>(*plain), Bytes<float>(repeated->next_logits));
    ASSERT_EQ(traced->activations.size(), 31u);
    ASSERT_EQ(traced->activations.size(), repeated->activations.size());
    ASSERT_EQ(traced->attention.size(), 2u);
    size_t residuals = 0;
    size_t projections = 0;
    for (size_t i = 0; i < traced->activations.size(); ++i) {
      const auto& value = traced->activations[i];
      EXPECT_EQ(value.raw_bytes, repeated->activations[i].raw_bytes);
      ASSERT_EQ(value.dimensions.size(), 3u);
      EXPECT_EQ(value.dimensions[0], 1);
      EXPECT_EQ(value.dimensions[1], static_cast<int64_t>(prefix.size()));
      if (value.name == "ResidualLayer") {
        EXPECT_EQ(value.occurrence, residuals % 2);
        ++residuals;
      }
      if (value.name == "FullyConnectedLayer") {
        EXPECT_EQ(value.occurrence, projections % 2);
        ++projections;
      }
      if (value.name == "LanguageModelingHeadLayer" || value.name == "gpt2") {
        EXPECT_EQ(value.dimensions[2], 32);
        for (size_t row = 0; row < prefix.size(); ++row)
          for (size_t column = 19; column < 32; ++column)
            EXPECT_EQ(value.values[row * 32 + column],
                      std::numeric_limits<float>::lowest());
      }
    }
    EXPECT_EQ(residuals, 4u);
    EXPECT_EQ(projections, 8u);
    EXPECT_EQ(traced->attention[0].scope,
              "gpt2/transformer_block_0/ResidualLayer/attention");
    EXPECT_EQ(traced->attention[1].scope,
              "gpt2/transformer_block_1/ResidualLayer/attention");
    for (size_t i = 0; i < traced->attention.size(); ++i) {
      const auto& value = traced->attention[i];
      EXPECT_EQ(value.dimensions, (std::vector<int64_t>{1, 2, 3, 3}));
      EXPECT_EQ(value.raw_bytes, repeated->attention[i].raw_bytes);
      EXPECT_EQ(value.occurrence, 0u);
    }
    for (size_t i = 0; i < original_weights.size(); ++i) {
      auto bytes = CopyBytes(*executor_, (*model)->weights()[i]);
      ASSERT_TRUE(bytes.ok()) << bytes.status();
      EXPECT_EQ(*bytes, original_weights[i]);
    }
  }
}

}  // namespace
}  // namespace pluto::llm::completion_trace
