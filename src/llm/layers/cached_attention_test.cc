#include <cuda_runtime.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer_hooks.h"
#include "src/llm/layers/delta_net.h"
#include "src/llm/layers/full_attention.h"
#include "src/llm/layers/inference.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

uint16_t BFloat16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

float Float(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16);
}

class CachedAttentionLayerTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }
  void TearDown() override {
    if (executor_)
      EXPECT_TRUE(executor_->Synchronize().ok());
  }

  template <class T>
  absl::StatusOr<Buffer> Upload(const std::vector<T>& values) {
    ASSIGN_OR_RETURN(auto staging,
                     cuda::PageLockedHostArray<T>::CopyFrom(*executor_, values));
    ASSIGN_OR_RETURN(auto output,
                     Buffer::Allocate(*executor_, values.size() * sizeof(T)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(output.data(), staging.data(), output.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "upload cached layer test"));
    RETURN_IF_ERROR(executor_->Synchronize());
    return output;
  }

  absl::StatusOr<Buffer> UploadBFloat16(const std::vector<float>& values) {
    std::vector<uint16_t> bits;
    for (float value : values)
      bits.push_back(BFloat16(value));
    return Upload(bits);
  }

  template <class T>
  std::vector<T> Read(const Buffer& input) {
    auto staging = cuda::PageLockedHostArray<T>::Allocate(
        *executor_, input.size_bytes() / sizeof(T));
    EXPECT_TRUE(staging.ok()) << staging.status();
    if (!staging.ok())
      return {};
    EXPECT_EQ(cudaMemcpyAsync(staging->data(), input.data(), input.size_bytes(),
                              cudaMemcpyDeviceToHost, executor_->stream()),
              cudaSuccess);
    EXPECT_TRUE(executor_->Synchronize().ok());
    return std::vector<T>(staging->begin(), staging->end());
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(CachedAttentionLayerTest,
       FullAttentionTypesHooksCacheAndBackwardContract) {
  FullAttentionParameters p;
  p.query_heads = 2;
  p.key_value_heads = 1;
  p.head_dim = 4;
  p.rotary_dim = 2;
  p.capacity = 2;
  auto norm = Upload(std::vector<float>(4));
  ASSERT_TRUE(norm.ok());
  auto layer = FullAttentionLayer::Create(*executor_, p, *norm, *norm);
  ASSERT_TRUE(layer.ok()) << layer.status();
  EXPECT_EQ((*layer)->weights().size(), 2);
  EXPECT_TRUE((*layer)->gradients().empty());
  EXPECT_EQ((*layer)->input_types()[0],
            ActivationType(DataType::BF16, {-2, 1, 16}));
  EXPECT_EQ((*layer)->input_types()[1],
            ActivationType(DataType::BF16, {-2, 1, 4}));
  EXPECT_EQ((*layer)->output_types()[0],
            ActivationType(DataType::BF16, {-2, 1, 8}));

  // Zero queries give uniform probabilities, and zero gate logits give 0.5.
  // Two KV steps therefore provide an independently obvious CPU answer.
  auto q = UploadBFloat16(std::vector<float>(16));
  auto k = UploadBFloat16({1, 1, 1, 1});
  auto v1 = UploadBFloat16({2, 4, -2, -4});
  auto v2 = UploadBFloat16({4, 8, 2, 4});
  ASSERT_TRUE(q.ok());
  ASSERT_TRUE(k.ok());
  ASSERT_TRUE(v1.ok());
  ASSERT_TRUE(v2.ok());
  int activation_calls = 0;
  int probability_calls = 0;
  LayerHooks hooks;
  hooks.activation_hook = [&](cuda::Executor& executor, absl::string_view name,
                              absl::Span<const ActivationType> types,
                              absl::Span<Buffer> outputs) {
    EXPECT_EQ(&executor, executor_.get());
    EXPECT_EQ(name, "FullAttentionLayer");
    EXPECT_EQ(types[0], (*layer)->output_types()[0]);
    EXPECT_EQ(outputs[0].size_bytes(), 8 * sizeof(uint16_t));
    ++activation_calls;
    return absl::OkStatus();
  };
  hooks.attention_probabilities_hook =
      [&](cuda::Executor&, absl::string_view name, const ActivationType& type,
          const Buffer& probabilities) {
        ++probability_calls;
        EXPECT_EQ(name, "FullAttentionLayer");
        EXPECT_EQ(type,
                  ActivationType(DataType::FP32, {1, 2, 1, probability_calls}));
        for (float value : Read<float>(probabilities))
          EXPECT_FLOAT_EQ(value, 1.0f / probability_calls);
        return absl::OkStatus();
      };
  auto first = (*layer)->fwd(*executor_, {*q, *k, *v1}, &hooks);
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_EQ((*layer)->length(), 1);
  const auto saved_first = Read<uint16_t>(first->outputs[0]);
  const std::vector<float> expected_first{1, 2, -1, -2, 1, 2, -1, -2};
  for (size_t i = 0; i < saved_first.size(); ++i)
    EXPECT_EQ(Float(saved_first[i]), expected_first[i]);
  auto second = (*layer)->fwd(*executor_, {*q, *k, *v2}, &hooks);
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(activation_calls, 2);
  EXPECT_EQ(probability_calls, 2);
  const auto actual = Read<uint16_t>(second->outputs[0]);
  const std::vector<float> expected_second{1.5, 3, 0, 0, 1.5, 3, 0, 0};
  for (size_t i = 0; i < actual.size(); ++i)
    EXPECT_EQ(Float(actual[i]), expected_second[i]);
  EXPECT_EQ(Read<uint16_t>(first->outputs[0]), saved_first);
  EXPECT_EQ((*layer)->fwd(*executor_, {*q, *k, *v1}).status().code(),
            absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(
      (*layer)->bwd(*executor_, {}, std::move(first->state)).status().code(),
      absl::StatusCode::kUnimplemented);
  ASSERT_TRUE((*layer)->Reset().ok());
  auto replay = (*layer)->fwd(*executor_, {*q, *k, *v1});
  ASSERT_TRUE(replay.ok());
  EXPECT_EQ(Read<uint16_t>(replay->outputs[0]), saved_first);
}

TEST_F(CachedAttentionLayerTest,
       FullAttentionValidatesBeforeAdvancingAndPropagatesHooks) {
  FullAttentionParameters p;
  p.query_heads = p.key_value_heads = 1;
  p.head_dim = p.rotary_dim = 2;
  p.capacity = 3;
  auto norm = Upload(std::vector<float>(2));
  auto q = UploadBFloat16({1, 2, 0, 0});
  auto kv = UploadBFloat16({2, 4});
  ASSERT_TRUE(norm.ok());
  ASSERT_TRUE(q.ok());
  ASSERT_TRUE(kv.ok());
  auto layer = FullAttentionLayer::Create(*executor_, p, *norm, *norm);
  ASSERT_TRUE(layer.ok());
  EXPECT_EQ((*layer)->fwd(*executor_, {*q, *kv}).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ((*layer)->fwd(*executor_, {*q, *kv, *q}).status().code(),
            absl::StatusCode::kInvalidArgument);
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok());
  EXPECT_EQ((*layer)->fwd(**other, {*q, *kv, *kv}).status().code(),
            absl::StatusCode::kInvalidArgument);
  auto foreign = Buffer::Allocate(**other, kv->size_bytes());
  ASSERT_TRUE(foreign.ok());
  EXPECT_EQ((*layer)->fwd(*executor_, {*q, *kv, *foreign}).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ((*layer)->length(), 0);
  EXPECT_EQ(FullAttentionLayer::Create(*executor_, p, *q, *kv).status().code(),
            absl::StatusCode::kInvalidArgument);

  LayerHooks hooks;
  hooks.attention_probabilities_hook = [](cuda::Executor&, absl::string_view,
                                          const ActivationType&,
                                          const Buffer&) {
    return absl::AbortedError("injected probability hook failure");
  };
  EXPECT_EQ((*layer)->fwd(*executor_, {*q, *kv, *kv}, &hooks).status().code(),
            absl::StatusCode::kAborted);
  ASSERT_TRUE((*layer)->Reset().ok());
  EXPECT_EQ((*layer)->length(), 0);
  EXPECT_TRUE((*layer)->fwd(*executor_, {*q, *kv, *kv}).ok());
}

TEST_F(CachedAttentionLayerTest,
       DeltaNetMatchesKernelStateAndResetsWithPhysicalBfloat16) {
  DeltaNetParameters p;
  p.key_heads = 1;
  p.value_heads = 2;
  p.key_head_dim = 3;
  p.value_head_dim = 4;
  p.conv_kernel_dim = 3;
  const int channels = 14;
  auto convolution = Upload(std::vector<float>(channels * 3, 0.3f));
  auto alog = Upload(std::vector<float>{-0.2f, 0.1f});
  auto bias = Upload(std::vector<float>{0.1f, -0.2f});
  auto norm = Upload(std::vector<float>(4, 1.0f));
  ASSERT_TRUE(convolution.ok());
  ASSERT_TRUE(alog.ok());
  ASSERT_TRUE(bias.ok());
  ASSERT_TRUE(norm.ok());
  auto layer =
      DeltaNetLayer::Create(*executor_, p, *convolution, *alog, *bias, *norm);
  auto reference = cached_attention_ops::DeltaNetState::Create(*executor_, p);
  auto reference_output = Upload(std::vector<float>(8));
  ASSERT_TRUE(layer.ok()) << layer.status();
  ASSERT_TRUE(reference.ok());
  ASSERT_TRUE(reference_output.ok());
  EXPECT_EQ((*layer)->weights().size(), 4);
  EXPECT_TRUE((*layer)->gradients().empty());
  EXPECT_EQ((*layer)->input_types()[0],
            ActivationType(DataType::BF16, {-2, 1, 14}));
  EXPECT_EQ((*layer)->input_types()[1],
            ActivationType(DataType::BF16, {-2, 1, 8}));
  EXPECT_EQ((*layer)->input_types()[2],
            ActivationType(DataType::BF16, {-2, 1, 2}));
  EXPECT_EQ((*layer)->output_types()[0],
            ActivationType(DataType::BF16, {-2, 1, 8}));

  std::vector<uint16_t> first;
  int activations = 0;
  LayerHooks hooks;
  hooks.activation_hook = [&](cuda::Executor&, absl::string_view name,
                              absl::Span<const ActivationType>,
                              absl::Span<Buffer>) {
    EXPECT_EQ(name, "DeltaNetLayer");
    ++activations;
    return absl::OkStatus();
  };
  for (int step = 0; step < 5; ++step) {
    SCOPED_TRACE(step);
    const int t = step == 4 ? 0 : step;
    if (step == 4) {
      ASSERT_TRUE((*layer)->Reset().ok());
      ASSERT_TRUE((*reference)->Reset().ok());
    }
    BufferVec inputs;
    BufferVec reference_inputs;
    for (int count : {channels, 8, 2, 2}) {
      std::vector<float> values(count);
      for (int i = 0; i < count; ++i)
        values[i] = Float(BFloat16(std::sin(i * 0.31f + t * 0.73f)));
      auto input = UploadBFloat16(values);
      auto reference_input = Upload(values);
      ASSERT_TRUE(input.ok());
      ASSERT_TRUE(reference_input.ok());
      inputs.push_back(std::move(*input));
      reference_inputs.push_back(std::move(*reference_input));
    }
    if (step == 0) {
      EXPECT_EQ((*layer)->fwd(*executor_, {}).status().code(),
                absl::StatusCode::kInvalidArgument);
      BufferVec bad = inputs;
      bad[0] = inputs[1];
      EXPECT_EQ((*layer)->fwd(*executor_, bad).status().code(),
                absl::StatusCode::kInvalidArgument);
      auto other = cuda::Executor::Create();
      ASSERT_TRUE(other.ok());
      EXPECT_EQ((*layer)->fwd(**other, inputs).status().code(),
                absl::StatusCode::kInvalidArgument);
    }
    auto result = (*layer)->fwd(*executor_, inputs, &hooks);
    ASSERT_TRUE(result.ok()) << result.status();
    const auto ptr = [](const Buffer& b) {
      return static_cast<const float*>(b.data());
    };
    ASSERT_TRUE((*reference)
                    ->Step(ptr(reference_inputs[0]), ptr(reference_inputs[1]),
                           ptr(reference_inputs[2]), ptr(reference_inputs[3]),
                           ptr(*convolution), ptr(*alog), ptr(*bias),
                           ptr(*norm),
                           static_cast<float*>(reference_output->data()))
                    .ok());
    const auto actual = Read<uint16_t>(result->outputs[0]);
    const auto expected = Read<float>(*reference_output);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
      EXPECT_EQ(Float(actual[i]), expected[i]);
    if (step == 0)
      first = actual;
    if (step == 4)
      EXPECT_EQ(actual, first);
    EXPECT_EQ(
        (*layer)->bwd(*executor_, {}, std::move(result->state)).status().code(),
        absl::StatusCode::kUnimplemented);
  }
  EXPECT_EQ(activations, 5);
  EXPECT_EQ(DeltaNetLayer::Create(*executor_, p, *norm, *alog, *bias, *norm)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(CachedAttentionLayerTest,
       FactoriesRejectActivationExtentsBeforeAllocatingCaches) {
  constexpr int maximum = inference_internal::kMaximumDimension;
  auto norm = Upload(std::vector<float>(2));
  ASSERT_TRUE(norm.ok());
  FullAttentionParameters full;
  full.query_heads = maximum / 4;
  full.key_value_heads = 1;
  full.head_dim = full.rotary_dim = 2;
  full.capacity = 1;
  // Q plus gate reaches the conversion limit exactly; the next head exceeds it.
  auto supported_full =
      FullAttentionLayer::Create(*executor_, full, *norm, *norm);
  ASSERT_TRUE(supported_full.ok()) << supported_full.status();
  ++full.query_heads;
  auto rejected_full =
      FullAttentionLayer::Create(*executor_, full, *norm, *norm);
  EXPECT_EQ(rejected_full.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      rejected_full.status().message(),
      "FullAttentionLayer activation exceeds the inference dimension limit");

  DeltaNetParameters delta;
  delta.key_heads = delta.value_heads = maximum / 3;
  delta.key_head_dim = delta.value_head_dim = delta.conv_kernel_dim = 1;
  auto convolution =
      Buffer::Allocate(*executor_, 3ULL * delta.key_heads * sizeof(float));
  auto head_weights = Buffer::Allocate(
      *executor_, static_cast<size_t>(delta.value_heads) * sizeof(float));
  auto delta_norm = Upload(std::vector<float>{1});
  ASSERT_TRUE(convolution.ok());
  ASSERT_TRUE(head_weights.ok());
  ASSERT_TRUE(delta_norm.ok());
  // These weights are never read: creation only checks sizes and clears cache.
  auto supported_delta =
      DeltaNetLayer::Create(*executor_, delta, *convolution, *head_weights,
                            *head_weights, *delta_norm);
  ASSERT_TRUE(supported_delta.ok()) << supported_delta.status();
  ++delta.key_heads;
  ++delta.value_heads;
  auto rejected_delta =
      DeltaNetLayer::Create(*executor_, delta, *convolution, *head_weights,
                            *head_weights, *delta_norm);
  EXPECT_EQ(rejected_delta.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(rejected_delta.status().message(),
            "DeltaNetLayer activation exceeds the inference dimension limit");

  // Extreme int parameters must reject without overflowing intermediate sums
  // or trying to reserve a huge cache, even when imported weights are tiny.
  full.query_heads = full.key_value_heads = full.head_dim =
      std::numeric_limits<int>::max();
  EXPECT_EQ(FullAttentionLayer::Create(*executor_, full, *norm, *norm)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  delta.key_heads = delta.value_heads = delta.key_head_dim =
      delta.value_head_dim = std::numeric_limits<int>::max();
  EXPECT_EQ(DeltaNetLayer::Create(*executor_, delta, *norm, *norm, *norm, *norm)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  delta.key_head_dim = 0;
  EXPECT_EQ(DeltaNetLayer::Create(*executor_, delta, *norm, *norm, *norm, *norm)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::llm
