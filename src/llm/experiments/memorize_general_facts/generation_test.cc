#include "src/llm/experiments/memorize_general_facts/generation.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer.h"
#include "src/util/status_macros.h"

namespace pluto::llm::memorize_general_facts {
namespace {

constexpr int64_t kBatch = ActivationType::kBatchDimension;

// Only the requested row has finite logical logits. Reading an earlier prompt
// row or a padding row therefore fails, and vocabulary padding can never win.
std::vector<float> Logits(int context, int vocabulary, int stride, int row,
                          int winner) {
  std::vector<float> values(context * stride,
                            std::numeric_limits<float>::quiet_NaN());
  std::fill(values.begin() + row * stride,
            values.begin() + row * stride + vocabulary, -20.0f);
  values[row * stride + winner] = 10.0f;
  if (stride > vocabulary)
    values[row * stride + vocabulary] = std::numeric_limits<float>::infinity();
  return values;
}

// A host-scripted GPU layer makes autoregressive feedback observable without
// introducing a second token-selection implementation or another GPU kernel.
// Every transfer uses pinned staging with explicit CPU/stream ordering.
class ScriptedLayer final : public Layer {
 public:
  using Script = std::function<std::vector<float>(absl::Span<const int>, int)>;

  ScriptedLayer(int context, int stride, Script script)
      : input_signatures{{DataType::INT32, {kBatch, context}}},
        output_signatures{{DataType::FP32, {kBatch, context, stride}}},
        script_(std::move(script)) {}

  absl::string_view name() const override { return "ScriptedGenerationLayer"; }
  absl::Span<const ActivationType> input_types() const override {
    return input_signatures;
  }
  absl::Span<const ActivationType> output_types() const override {
    return output_signatures;
  }
  absl::Span<Buffer> weights() override {
    return absl::MakeSpan(parameter_buffers);
  }
  DataType output_type() const override { return DataType::FP16; }

  std::vector<ActivationType> input_signatures;
  std::vector<ActivationType> output_signatures;
  BufferVec parameter_buffers;
  cuda::Executor* output_executor = nullptr;
  size_t output_count = 1;
  mutable std::vector<std::vector<int>> forward_inputs;
  mutable std::vector<void*> input_addresses;

 private:
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override {
    if (inputs.size() != 1 || &inputs[0].executor() != &executor)
      return absl::InvalidArgumentError("invalid scripted model input");
    ASSIGN_OR_RETURN(auto host_input,
                     cuda::PageLockedHostArray<int>::Allocate(
                         executor, inputs[0].size_bytes() / sizeof(int)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host_input.data(), inputs[0].data(),
                        inputs[0].size_bytes(), cudaMemcpyDeviceToHost,
                        executor.stream()),
        "download scripted generation input"));
    RETURN_IF_ERROR(executor.Synchronize());
    const int call = static_cast<int>(forward_inputs.size());
    forward_inputs.emplace_back(host_input.begin(), host_input.end());
    input_addresses.push_back(inputs[0].data());
    const auto values = script_(host_input.span(), call);
    auto& destination =
        output_executor == nullptr ? executor : *output_executor;
    ASSIGN_OR_RETURN(auto staging, cuda::PageLockedHostArray<float>::CopyFrom(
                                       destination, values));
    ASSIGN_OR_RETURN(auto logits,
                     cuda::Buffer::Allocate(destination, staging.size_bytes()));
    if (!values.empty())
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(logits.data(), staging.data(), staging.size_bytes(),
                          cudaMemcpyHostToDevice, destination.stream()),
          "upload scripted generation logits"));
    FwdResult result;
    for (size_t i = 0; i < output_count; ++i)
      result.outputs.push_back(logits);
    result.state.intermediates.push_back(logits);
    result.state.children.push_back(BackwardState{});
    result.state.children.back().intermediates.push_back(inputs[0]);
    return result;
  }

  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override {
    return absl::InternalError("generation must not call backward");
  }

  Script script_;
};

class GenerationTest : public testing::Test {
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

TEST_F(GenerationTest, FeedsGeneratedTokensAtAdvancingFinalPositions) {
  ScriptedLayer model(6, 12, [](absl::Span<const int> input, int call) {
    const int row = 1 + call;
    return Logits(6, 10, 12, row, input[row] + 1);
  });
  auto generated = GenerateContinuation(*executor_, model, {1, 2}, 10, 9, 3);
  ASSERT_TRUE(generated.ok()) << generated.status();
  EXPECT_EQ(&generated->executor(), executor_.get());
  EXPECT_EQ(std::vector<int>(generated->begin(), generated->end()),
            (std::vector<int>{3, 4, 5}));
  EXPECT_EQ(model.forward_inputs,
            (std::vector<std::vector<int>>{
                {1, 2, 9, 9, 9, 9}, {1, 2, 3, 9, 9, 9}, {1, 2, 3, 4, 9, 9}}));
  ASSERT_EQ(model.input_addresses.size(), 3u);
  EXPECT_EQ(model.input_addresses[0], model.input_addresses[1]);
  EXPECT_EQ(model.input_addresses[0], model.input_addresses[2]);
}

TEST_F(GenerationTest, StopsAtEosAndExcludesItFromContinuation) {
  ScriptedLayer model(6, 12, [](absl::Span<const int>, int call) {
    return Logits(6, 10, 12, 1 + call, call == 0 ? 3 : 9);
  });
  auto generated = GenerateContinuation(*executor_, model, {1, 2}, 10, 9, 4);
  ASSERT_TRUE(generated.ok()) << generated.status();
  EXPECT_EQ(std::vector<int>(generated->begin(), generated->end()),
            (std::vector<int>{3}));
  EXPECT_EQ(model.forward_inputs.size(), 2u);

  ScriptedLayer immediate(6, 10, [](absl::Span<const int>, int) {
    return Logits(6, 10, 10, 0, 9);
  });
  auto empty = GenerateContinuation(*executor_, immediate, {1}, 10, 9, 5);
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_TRUE(empty->empty());
  EXPECT_EQ(immediate.forward_inputs.size(), 1u);
}

TEST_F(GenerationTest, CapsHugeRequestAtAvailableContextWithoutShifting) {
  ScriptedLayer model(4, 10, [](absl::Span<const int> input, int call) {
    const int row = 1 + call;
    return Logits(4, 10, 10, row, input[row] + 1);
  });
  auto generated = GenerateContinuation(*executor_, model, {1, 2}, 10, 9,
                                        std::numeric_limits<int>::max());
  ASSERT_TRUE(generated.ok()) << generated.status();
  EXPECT_EQ(std::vector<int>(generated->begin(), generated->end()),
            (std::vector<int>{3, 4}));
  EXPECT_EQ(model.forward_inputs,
            (std::vector<std::vector<int>>{{1, 2, 9, 9}, {1, 2, 3, 9}}));
}

TEST_F(GenerationTest, ZeroRequestAndFullContextDoNotRunModel) {
  ScriptedLayer model(4, 10, [](absl::Span<const int>, int) {
    ADD_FAILURE() << "model should not run without room or requested tokens";
    return std::vector<float>{};
  });
  auto zero = GenerateContinuation(*executor_, model, {1, 2}, 10, 9, 0);
  auto full = GenerateContinuation(*executor_, model, {1, 2, 3, 4}, 10, 9, 100);
  ASSERT_TRUE(zero.ok()) << zero.status();
  ASSERT_TRUE(full.ok()) << full.status();
  EXPECT_TRUE(zero->empty());
  EXPECT_TRUE(full->empty());
  EXPECT_EQ(&zero->executor(), executor_.get());
  EXPECT_EQ(&full->executor(), executor_.get());
  EXPECT_TRUE(model.forward_inputs.empty());
}

TEST_F(GenerationTest, LowestIdWinsTiesAndPaddingNeverWinsRepeatedly) {
  ScriptedLayer model(4, 12, [](absl::Span<const int>, int) {
    auto values = Logits(4, 10, 12, 1, 7);
    values[12 + 3] = 10;
    return values;
  });
  for (int repetition = 0; repetition < 5; ++repetition) {
    auto generated = GenerateContinuation(*executor_, model, {1, 2}, 10, 9, 1);
    ASSERT_TRUE(generated.ok()) << generated.status();
    ASSERT_EQ(generated->size(), 1u);
    EXPECT_EQ((*generated)[0], 3);
  }
}

TEST_F(GenerationTest, SingleTokenVocabularyCanOnlyProduceEos) {
  ScriptedLayer model(
      2, 1, [](absl::Span<const int>, int) { return Logits(2, 1, 1, 0, 0); });
  auto generated = GenerateContinuation(*executor_, model, {0}, 1, 0, 1);
  ASSERT_TRUE(generated.ok()) << generated.status();
  EXPECT_TRUE(generated->empty());
  EXPECT_EQ(model.forward_inputs.size(), 1u);
}

TEST_F(GenerationTest, RejectsNonfiniteSelectedLogits) {
  for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()}) {
    ScriptedLayer model(4, 12, [invalid](absl::Span<const int>, int) {
      auto values = Logits(4, 10, 12, 1, 3);
      values[12 + 8] = invalid;
      return values;
    });
    auto generated = GenerateContinuation(*executor_, model, {1, 2}, 10, 9, 1);
    EXPECT_EQ(generated.status().code(), absl::StatusCode::kDataLoss);
    EXPECT_EQ(model.forward_inputs.size(), 1u);
  }
}

TEST_F(GenerationTest,
       RejectsInvalidPromptAndGenerationArgumentsBeforeForward) {
  ScriptedLayer model(4, 10, [](absl::Span<const int>, int) {
    ADD_FAILURE() << "invalid inputs must be rejected before forward";
    return std::vector<float>{};
  });
  for (const auto& prompt :
       std::vector<std::vector<int>>{{}, {-1}, {10}, {1, 2, 3, 4, 5}}) {
    EXPECT_EQ(GenerateContinuation(*executor_, model, prompt, 10, 9, 1)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
    // Zero new tokens does not bypass prompt validation.
    EXPECT_EQ(GenerateContinuation(*executor_, model, prompt, 10, 9, 0)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  for (int vocabulary : {-1, 0, 11})
    EXPECT_EQ(GenerateContinuation(*executor_, model, {1}, vocabulary, 9, 1)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  for (int eos : {-1, 10})
    EXPECT_EQ(GenerateContinuation(*executor_, model, {1}, 10, eos, 1)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      GenerateContinuation(*executor_, model, {1}, 10, 9, -1).status().code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(model.forward_inputs.empty());
}

TEST_F(GenerationTest, RejectsInvalidDeclaredSignatures) {
  const ScriptedLayer::Script unused = [](absl::Span<const int>, int) {
    ADD_FAILURE() << "invalid signatures must be rejected before forward";
    return std::vector<float>{};
  };
  for (const ActivationType& input : std::vector<ActivationType>{
           {DataType::FP32, {kBatch, 4}},
           {DataType::INT32, {4}},
           {DataType::INT32, {1, 4}},
           {DataType::INT32, {kBatch, 4, 1}},
           {DataType::INT32, {kBatch, 0}},
           {DataType::INT32, {kBatch, int64_t{1} << 32}}}) {
    ScriptedLayer model(4, 10, unused);
    model.input_signatures = {input};
    EXPECT_EQ(
        GenerateContinuation(*executor_, model, {1}, 10, 9, 1).status().code(),
        absl::StatusCode::kInvalidArgument);
  }
  for (const ActivationType& output : std::vector<ActivationType>{
           {DataType::BF16, {kBatch, 4, 10}},
           {DataType::FP32, {kBatch, 40}},
           {DataType::FP32, {1, 4, 10}},
           {DataType::FP32, {kBatch, 3, 10}},
           {DataType::FP32, {kBatch, 4, 9}},
           {DataType::FP32, {kBatch, 4, 0}},
           {DataType::FP32, {kBatch, 4, int64_t{1} << 32}}}) {
    ScriptedLayer model(4, 10, unused);
    model.output_signatures = {output};
    EXPECT_EQ(
        GenerateContinuation(*executor_, model, {1}, 10, 9, 1).status().code(),
        absl::StatusCode::kInvalidArgument);
  }
  for (int count : {0, 2}) {
    ScriptedLayer model(4, 10, unused);
    model.input_signatures.assign(count, {DataType::INT32, {kBatch, 4}});
    EXPECT_EQ(
        GenerateContinuation(*executor_, model, {1}, 10, 9, 1).status().code(),
        absl::StatusCode::kInvalidArgument);
    model.input_signatures = {{DataType::INT32, {kBatch, 4}}};
    model.output_signatures.assign(count, {DataType::FP32, {kBatch, 4, 10}});
    EXPECT_EQ(
        GenerateContinuation(*executor_, model, {1}, 10, 9, 1).status().code(),
        absl::StatusCode::kInvalidArgument);
  }
}

TEST_F(GenerationTest, RejectsOutputBufferSizeOrCountMismatch) {
  for (int count : {0, 39, 41, 80}) {
    ScriptedLayer model(4, 10, [count](absl::Span<const int>, int) {
      return std::vector<float>(count, 0);
    });
    EXPECT_EQ(
        GenerateContinuation(*executor_, model, {1}, 10, 9, 1).status().code(),
        absl::StatusCode::kInvalidArgument);
  }
  for (size_t outputs : {0, 2}) {
    ScriptedLayer model(4, 10, [](absl::Span<const int>, int) {
      return Logits(4, 10, 10, 0, 3);
    });
    model.output_count = outputs;
    EXPECT_EQ(
        GenerateContinuation(*executor_, model, {1}, 10, 9, 1).status().code(),
        absl::StatusCode::kInvalidArgument);
  }
}

TEST_F(GenerationTest, RejectsWeightsAndOutputsFromAnotherExecutor) {
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  {
    ScriptedLayer model(4, 10, [](absl::Span<const int>, int) {
      return Logits(4, 10, 10, 0, 3);
    });
    auto weight = cuda::Buffer::Allocate(**other, sizeof(float));
    ASSERT_TRUE(weight.ok()) << weight.status();
    model.parameter_buffers.push_back(std::move(*weight));
    EXPECT_EQ(
        GenerateContinuation(*executor_, model, {1}, 10, 9, 1).status().code(),
        absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(model.forward_inputs.empty());
    model.parameter_buffers.clear();
    model.output_executor = other->get();
    EXPECT_EQ(
        GenerateContinuation(*executor_, model, {1}, 10, 9, 1).status().code(),
        absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(model.forward_inputs.size(), 1u);
  }
  EXPECT_TRUE((*other)->Synchronize().ok());
}

}  // namespace
}  // namespace pluto::llm::memorize_general_facts
