#include "src/llm/experiments/memorize_general_facts/activation_trace.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/experiments/gpt2_shakespeare/gpt2.h"
#include "src/llm/layer_hooks.h"
#include "src/util/status_macros.h"

namespace pluto::llm::memorize_general_facts {
namespace {

template <class T>
absl::StatusOr<Buffer> Upload(cuda::Executor& executor,
                              absl::Span<const T> values) {
  ASSIGN_OR_RETURN(auto host,
                   cuda::PageLockedHostArray<T>::CopyFrom(executor, values));
  ASSIGN_OR_RETURN(auto device, Buffer::Allocate(executor, host.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(device.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "cudaMemcpyAsync(test activation upload)"));
  return device;
}

template <class T>
absl::StatusOr<cuda::PageLockedHostArray<T>> Download(cuda::Executor& executor,
                                                      const Buffer& buffer) {
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<T>::Allocate(
                                  executor, buffer.size_bytes() / sizeof(T)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), buffer.data(), host.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "cudaMemcpyAsync(test activation download)"));
  RETURN_IF_ERROR(executor.Synchronize());
  return host;
}

// A small fake supplies exact boundary bytes and malformed metadata without
// depending on the behavior of a particular numerical kernel. It still drives
// the public Layer::fwd path, including scoped hooks and error propagation.
class BoundaryModel final : public Layer {
 public:
  struct Emission {
    std::string name;
    ActivationType type;
    Buffer buffer;
    bool nested = false;
  };

  std::vector<ActivationType> input_signature{
      ActivationType(DataType::INT32, {-2, 4})};
  std::vector<ActivationType> output_signature{
      ActivationType(DataType::FP32, {-2, 4, 16})};
  std::vector<Emission> emissions;
  mutable int forward_calls = 0;
  absl::Status forward_status;

  absl::string_view name() const override { return "gpt2"; }
  absl::Span<const ActivationType> input_types() const override {
    return input_signature;
  }
  absl::Span<const ActivationType> output_types() const override {
    return output_signature;
  }
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::FP16; }

 private:
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer>,
                                     LayerHooks* hooks) const override {
    ++forward_calls;
    RETURN_IF_ERROR(forward_status);
    RETURN_IF_ERROR(hooks->enter_combinator(executor, name()));
    for (const auto& emission : emissions) {
      if (emission.nested)
        RETURN_IF_ERROR(hooks->enter_combinator(executor, "inner_mlp"));
      Buffer output = emission.buffer;
      const auto status = hooks->activation_hook(
          executor, emission.name, {&emission.type, 1}, {&output, 1});
      if (emission.nested)
        RETURN_IF_ERROR(hooks->exit_combinator(executor));
      if (!status.ok()) {
        RETURN_IF_ERROR(hooks->exit_combinator(executor));
        return status;
      }
    }
    RETURN_IF_ERROR(hooks->exit_combinator(executor));
    if (emissions.empty())
      return absl::InvalidArgumentError("test model needs a return buffer");
    return FwdResult{{emissions.back().buffer}, {}};
  }
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override {
    return absl::UnimplementedError("test forward-only model");
  }
};

class ActivationTraceTest : public testing::Test {
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

TEST_F(ActivationTraceTest, CapturesRawRowsAndIgnoresPaddingAndNestedNames) {
  std::vector<float> values(4 * 16, std::numeric_limits<float>::quiet_NaN());
  for (int i = 0; i < 2 * 16; ++i)
    values[i] = (i - 16) * 0.125f;
  auto buffer = Upload<float>(*executor_, values);
  ASSERT_TRUE(buffer.ok()) << buffer.status();
  BoundaryModel model;
  const ActivationType type(DataType::FP32, {-2, 4, 16});
  model.emissions = {{"PositionEmbeddingLayer", type, *buffer},
                     {"PositionEmbeddingLayer", type, *buffer, true},
                     {"transformer_block_99", type, *buffer, true},
                     {"transformer_block_0", type, *buffer}};
  auto trace =
      CaptureActivationBoundaries(*executor_, model, {1, 2}, 8, 0, 1, 16);
  ASSERT_TRUE(trace.ok()) << trace.status();
  ASSERT_EQ(trace->size(), 2);
  EXPECT_EQ((*trace)[0].name, "PositionEmbeddingLayer");
  EXPECT_EQ((*trace)[1].name, "transformer_block_0");
  const std::vector<float> expected(values.begin(), values.begin() + 32);
  EXPECT_EQ((*trace)[0].values, expected);
  EXPECT_EQ((*trace)[1].values, expected);
  EXPECT_EQ(model.forward_calls, 1);
}

TEST_F(ActivationTraceTest, ExpandsBfloat16WithoutChangingItsValues) {
  // BF16 encodings for +1, -2, 0, 0.5; NaN padding must never be copied/read.
  std::vector<uint16_t> values(4 * 16, 0x7fc0);
  const uint16_t bits[] = {0x3f80, 0xc000, 0x0000, 0x3f00};
  const float floats[] = {1, -2, 0, 0.5f};
  for (int i = 0; i < 16; ++i)
    values[i] = bits[i % 4];
  auto buffer = Upload<uint16_t>(*executor_, values);
  ASSERT_TRUE(buffer.ok()) << buffer.status();
  BoundaryModel model;
  model.emissions = {{"PositionEmbeddingLayer",
                      ActivationType(DataType::BF16, {-2, 4, 16}), *buffer}};
  auto trace = CaptureActivationBoundaries(*executor_, model, {1}, 8, 0, 0, 16);
  ASSERT_TRUE(trace.ok()) << trace.status();
  ASSERT_EQ(trace->size(), 1);
  ASSERT_EQ((*trace)[0].values.size(), 16);
  for (int i = 0; i < 16; ++i)
    EXPECT_EQ((*trace)[0].values[i], floats[i % 4]);
}

TEST_F(ActivationTraceTest, RejectsInvalidArgumentsBeforeForward) {
  BoundaryModel model;
  EXPECT_FALSE(
      CaptureActivationBoundaries(*executor_, model, {1}, 0, 0, 0, 16).ok());
  EXPECT_FALSE(
      CaptureActivationBoundaries(*executor_, model, {8}, 8, 0, 0, 16).ok());
  EXPECT_FALSE(
      CaptureActivationBoundaries(*executor_, model, {1}, 8, 8, 0, 16).ok());
  EXPECT_FALSE(
      CaptureActivationBoundaries(*executor_, model, {}, 8, 0, 0, 16).ok());
  EXPECT_FALSE(
      CaptureActivationBoundaries(*executor_, model, {-1}, 8, 0, 0, 16).ok());
  EXPECT_FALSE(
      CaptureActivationBoundaries(*executor_, model, {1}, 8, -1, 0, 16).ok());
  EXPECT_FALSE(
      CaptureActivationBoundaries(*executor_, model, {1}, 8, 0, -1, 16).ok());
  for (int width : {0, 8, 32})
    EXPECT_FALSE(
        CaptureActivationBoundaries(*executor_, model, {1}, 8, 0, 0, width)
            .ok());
  EXPECT_FALSE(CaptureActivationBoundaries(*executor_, model, {0, 1, 2, 3, 4},
                                           8, 0, 0, 16)
                   .ok());
  model.input_signature = {ActivationType(DataType::FP32, {-2, 4})};
  EXPECT_FALSE(
      CaptureActivationBoundaries(*executor_, model, {1}, 8, 0, 0, 16).ok());
  model.input_signature = {ActivationType(DataType::INT32, {-2, 4, 1})};
  EXPECT_FALSE(
      CaptureActivationBoundaries(*executor_, model, {1}, 8, 0, 0, 16).ok());
  EXPECT_EQ(model.forward_calls, 0);
}

TEST_F(ActivationTraceTest, RejectsMissingExtraAndOutOfOrderBoundaries) {
  auto buffer = Upload<float>(*executor_, std::vector<float>(64, 1));
  ASSERT_TRUE(buffer.ok()) << buffer.status();
  BoundaryModel model;
  const ActivationType type(DataType::FP32, {-2, 4, 16});
  model.emissions = {{"PositionEmbeddingLayer", type, *buffer}};
  EXPECT_FALSE(
      CaptureActivationBoundaries(*executor_, model, {1}, 8, 0, 1, 16).ok());
  model.emissions.push_back({"transformer_block_0", type, *buffer});
  EXPECT_FALSE(
      CaptureActivationBoundaries(*executor_, model, {1}, 8, 0, 0, 16).ok());
  model.emissions[1].name = "transformer_block_1";
  EXPECT_FALSE(
      CaptureActivationBoundaries(*executor_, model, {1}, 8, 0, 1, 16).ok());
  model.emissions[1].name = "PositionEmbeddingLayer";
  EXPECT_FALSE(
      CaptureActivationBoundaries(*executor_, model, {1}, 8, 0, 1, 16).ok());
  model.emissions[0].name = "LayerNormLayer";
  model.emissions.erase(model.emissions.begin() + 1, model.emissions.end());
  EXPECT_FALSE(
      CaptureActivationBoundaries(*executor_, model, {1}, 8, 0, 0, 16).ok());
}

TEST_F(ActivationTraceTest, RejectsBadMetadataWrongExecutorAndNonFiniteValues) {
  std::vector<float> values(64, 1);
  values[0] = std::numeric_limits<float>::infinity();
  auto buffer = Upload<float>(*executor_, values);
  ASSERT_TRUE(buffer.ok()) << buffer.status();
  BoundaryModel model;
  model.emissions = {{"PositionEmbeddingLayer",
                      ActivationType(DataType::FP32, {-2, 4, 16}), *buffer}};
  auto result =
      CaptureActivationBoundaries(*executor_, model, {1}, 8, 0, 0, 16);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kDataLoss);
  for (const ActivationType& type :
       {ActivationType(DataType::FP16, {-2, 4, 16}),
        ActivationType(DataType::FP32, {-2, 4, 8}),
        ActivationType(DataType::FP32, {-2, 3, 16}),
        ActivationType(DataType::FP32, {1, 4, 16})}) {
    model.emissions[0].type = type;
    EXPECT_EQ(CaptureActivationBoundaries(*executor_, model, {1}, 8, 0, 0, 16)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  model.emissions[0].type = ActivationType(DataType::FP32, {-2, 4, 16});
  auto short_buffer = Upload<float>(*executor_, {1});
  ASSERT_TRUE(short_buffer.ok()) << short_buffer.status();
  model.emissions[0].buffer = *short_buffer;
  EXPECT_FALSE(
      CaptureActivationBoundaries(*executor_, model, {1}, 8, 0, 0, 16).ok());
  auto other_executor = cuda::Executor::Create();
  ASSERT_TRUE(other_executor.ok()) << other_executor.status();
  auto foreign_buffer =
      Upload<float>(**other_executor, std::vector<float>(64, 1));
  ASSERT_TRUE(foreign_buffer.ok()) << foreign_buffer.status();
  model.emissions[0].buffer = *foreign_buffer;
  EXPECT_FALSE(
      CaptureActivationBoundaries(*executor_, model, {1}, 8, 0, 0, 16).ok());
  // Restore ownership before the second executor is destroyed.
  model.emissions[0].buffer = *buffer;
}

TEST_F(ActivationTraceTest, PropagatesForwardFailure) {
  BoundaryModel model;
  model.forward_status = absl::UnavailableError("injected forward failure");
  auto result =
      CaptureActivationBoundaries(*executor_, model, {1}, 8, 0, 0, 16);
  EXPECT_EQ(result.status(), model.forward_status);
}

TEST_F(ActivationTraceTest,
       RealRecipeReplaysLastTokenPreservesPrefixAndLogits) {
  for (DataType policy : {DataType::FP16, DataType::BF16}) {
    SCOPED_TRACE(static_cast<int>(policy));
    Gpt2Config config;
    config.transformer_block_count = 2;
    config.model_width = 16;
    config.attention_heads = 1;
    config.feed_forward_width = 32;
    config.vocabulary_size = 32;
    config.pad_vocabulary = false;
    auto model = CreateGpt2(*executor_, policy, 42, config);
    ASSERT_TRUE(model.ok()) << model.status();
    std::vector<int32_t> input_tokens(kGpt2ContextLength, 0);
    input_tokens[0] = 1;
    input_tokens[1] = 2;
    input_tokens[2] = 3;
    auto input = Upload<int32_t>(*executor_, input_tokens);
    ASSERT_TRUE(input.ok()) << input.status();
    auto before = (*model)->fwd(*executor_, {&*input, 1});
    ASSERT_TRUE(before.ok()) << before.status();
    auto logits_before = Download<float>(*executor_, before->outputs[0]);
    ASSERT_TRUE(logits_before.ok()) << logits_before.status();

    auto short_trace =
        CaptureActivationBoundaries(*executor_, **model, {1, 2}, 32, 0, 2, 16);
    auto full_trace = CaptureActivationBoundaries(*executor_, **model,
                                                  {1, 2, 3}, 32, 0, 2, 16);
    auto changed_last = CaptureActivationBoundaries(*executor_, **model,
                                                    {1, 2, 4}, 32, 0, 2, 16);
    ASSERT_TRUE(short_trace.ok()) << short_trace.status();
    ASSERT_TRUE(full_trace.ok()) << full_trace.status();
    ASSERT_TRUE(changed_last.ok()) << changed_last.status();
    ASSERT_EQ(full_trace->size(), 3);
    for (size_t stage = 0; stage < full_trace->size(); ++stage) {
      SCOPED_TRACE(stage);
      EXPECT_EQ((*full_trace)[stage].name,
                stage == 0 ? "PositionEmbeddingLayer"
                           : absl::StrCat("transformer_block_", stage - 1));
      const auto& values = (*full_trace)[stage].values;
      ASSERT_EQ(values.size(), 3 * 16);
      EXPECT_EQ(std::vector<float>(values.begin(), values.begin() + 32),
                (*short_trace)[stage].values);
      const auto& changed = (*changed_last)[stage].values;
      EXPECT_TRUE(
          std::equal(values.begin(), values.begin() + 32, changed.begin()));
      EXPECT_FALSE(
          std::equal(values.begin() + 32, values.end(), changed.begin() + 32));
    }

    // Check the first boundary against actual embedding and position weights,
    // proving the final token is represented at its OWN position, not by the
    // preceding token's state used to predict it. FP16 policy stores FP32.
    if (policy == DataType::FP16) {
      auto embeddings = Download<float>(*executor_, (*model)->weights()[0]);
      auto positions = Download<float>(*executor_, (*model)->weights()[1]);
      ASSERT_TRUE(embeddings.ok()) << embeddings.status();
      ASSERT_TRUE(positions.ok()) << positions.status();
      for (int dim = 0; dim < 16; ++dim)
        EXPECT_FLOAT_EQ(
            (*full_trace)[0].values[2 * 16 + dim],
            (*embeddings)[3 * 16 + dim] + (*positions)[2 * 16 + dim]);
    }
    auto after = (*model)->fwd(*executor_, {&*input, 1});
    ASSERT_TRUE(after.ok()) << after.status();
    auto logits_after = Download<float>(*executor_, after->outputs[0]);
    ASSERT_TRUE(logits_after.ok()) << logits_after.status();
    ASSERT_EQ(logits_before->size(), logits_after->size());
    EXPECT_EQ(std::memcmp(logits_before->data(), logits_after->data(),
                          logits_before->size_bytes()),
              0);
  }
}

}  // namespace
}  // namespace pluto::llm::memorize_general_facts
