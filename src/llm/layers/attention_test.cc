#include "src/llm/layers/attention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/llm/layer.h"
#include "src/llm/layer_hooks.h"
#include "src/llm/layers/reference_test_util.h"
#include "src/llm/layers/test_util.h"

namespace pluto::llm {
namespace {

TEST_F(LayerReferenceTest, AttentionProbabilitiesMatchScalarSoftmax) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    // Singleton sequences, ragged head widths, multiple sequences/heads, and
    // sequences that cross one or two 32-row tiles exercise every buffer axis.
    for (const auto& [context, heads, width, batches] :
         {std::tuple{1, 1, 7, 2}, std::tuple{5, 2, 14, 2},
          std::tuple{33, 2, 160, 2}, std::tuple{65, 3, 21, 1}}) {
      for (bool uniform : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << "type=" << static_cast<int>(type) << " context="
                     << context << " heads=" << heads << " width=" << width
                     << " batches=" << batches << " uniform=" << uniform);
        const int rows = context * batches;
        const int head_dim = width / heads;
        std::vector<float> qkv(static_cast<size_t>(rows) * 3 * width);
        for (size_t index = 0; index < qkv.size(); ++index) {
          // Exact binary fractions avoid making the oracle depend on a
          // tensor-core operand rounding policy. Uniform rows additionally
          // verify normalization across the online-softmax tile boundaries.
          qkv[index] = static_cast<float>(static_cast<int>(index % 19) - 9) / 8;
          if (uniform && index % (3 * width) < static_cast<size_t>(2 * width))
            qkv[index] = 0.0f;
        }
        auto input = MakeActivationBufferPair(*executor_, qkv, type);
        auto layer =
            AttentionLayer::Create(*executor_, context, heads, width, type);
        ASSERT_TRUE(input.ok()) << input.status();
        ASSERT_TRUE(layer.ok()) << layer.status();
        BufferVec retained;
        LayerHooks hooks;
        hooks.attention_probabilities_hook = [&](cuda::Executor& executor,
                                                 absl::string_view name,
                                                 const ActivationType& shape,
                                                 const Buffer& buffer) {
          EXPECT_EQ(&executor, executor_.get());
          EXPECT_EQ(name, "AttentionLayer");
          EXPECT_EQ(shape,
                    (ActivationType{DataType::FP32,
                                    {batches, heads, context, context}}));
          EXPECT_EQ(&buffer.executor(), executor_.get());
          EXPECT_EQ(buffer.size_bytes(), static_cast<size_t>(batches) * heads *
                                             context * context * sizeof(float));
          // Buffer copies outlive the callback and need no eager D2H
          // synchronization while AttentionLayer is still submitting work.
          retained.push_back(buffer);
          return absl::OkStatus();
        };
        BufferVec inputs = {input->device};
        auto output = (*layer)->fwd(*executor_, inputs, &hooks);
        ASSERT_TRUE(output.ok()) << output.status();
        ASSERT_EQ(retained.size(), 1u);
        hooks = {};
        auto probabilities = ReadDeviceFloats(*executor_, retained.front());
        auto values =
            ReadDeviceActivations(*executor_, output->outputs[0], type);
        ASSERT_TRUE(probabilities.ok()) << probabilities.status();
        ASSERT_TRUE(values.ok()) << values.status();

        // Textbook oracle: one double-precision dot product per visible key,
        // max-subtracted softmax, and finally P*V. This does not use the
        // production kernel's saved maxima, normalization, or tiling logic.
        for (int batch = 0; batch < batches; ++batch) {
          for (int head = 0; head < heads; ++head) {
            for (int query = 0; query < context; ++query) {
              const size_t probability_offset =
                  ((static_cast<size_t>(batch) * heads + head) * context +
                   query) *
                  context;
              const size_t query_offset =
                  static_cast<size_t>(batch * context + query) * 3 * width +
                  head * head_dim;
              std::vector<double> expected(query + 1);
              double maximum = -std::numeric_limits<double>::infinity();
              for (int key = 0; key <= query; ++key) {
                const size_t key_offset =
                    static_cast<size_t>(batch * context + key) * 3 * width +
                    width + head * head_dim;
                double dot = 0;
                for (int dim = 0; dim < head_dim; ++dim)
                  dot += static_cast<double>(qkv[query_offset + dim]) *
                         qkv[key_offset + dim];
                expected[key] = dot / std::sqrt(static_cast<double>(head_dim));
                maximum = std::max(maximum, expected[key]);
              }
              double denominator = 0;
              for (double& probability : expected) {
                probability = std::exp(probability - maximum);
                denominator += probability;
              }
              double sum = 0;
              for (int key = 0; key < context; ++key) {
                const float actual = (*probabilities)[probability_offset + key];
                if (key <= query) {
                  EXPECT_NEAR(actual, expected[key] / denominator, 3e-5f);
                  EXPECT_GE(actual, 0.0f);
                  EXPECT_LE(actual, 1.0f);
                  sum += actual;
                } else {
                  EXPECT_EQ(actual, 0.0f) << "future key=" << key;
                }
              }
              EXPECT_NEAR(sum, 1.0, 3e-5);
              for (int dim = 0; dim < head_dim; ++dim) {
                double weighted_value = 0;
                for (int key = 0; key <= query; ++key) {
                  const size_t value_offset =
                      static_cast<size_t>(batch * context + key) * 3 * width +
                      2 * width + head * head_dim + dim;
                  weighted_value += (*probabilities)[probability_offset + key] *
                                    qkv[value_offset];
                }
                const size_t output_offset =
                    static_cast<size_t>(batch * context + query) * width +
                    head * head_dim + dim;
                EXPECT_NEAR((*values)[output_offset], weighted_value,
                            type == DataType::BF16 ? 5e-3f : 3e-5f);
              }
            }
          }
        }
      }
    }
  }
}

TEST_F(LayerReferenceTest,
       ProbabilityHookPreservesForwardBackwardAndDeterminism) {
  constexpr int kContext = 33;
  constexpr int kHeads = 2;
  constexpr int kWidth = 144;
  constexpr int kRows = 2 * kContext;
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    SCOPED_TRACE(testing::Message() << "type=" << static_cast<int>(type));
    std::vector<float> qkv(kRows * 3 * kWidth);
    std::vector<float> gradient(kRows * kWidth);
    for (size_t index = 0; index < qkv.size(); ++index)
      qkv[index] = 0.7f * std::sin(static_cast<float>(index) * 0.071f);
    for (size_t index = 0; index < gradient.size(); ++index)
      gradient[index] = 0.2f * std::cos(static_cast<float>(index) * 0.13f);
    auto input = MakeActivationBufferPair(*executor_, qkv, type);
    auto gradients = MakeRawBufferPair<float>(*executor_, gradient);
    auto layer =
        AttentionLayer::Create(*executor_, kContext, kHeads, kWidth, type);
    ASSERT_TRUE(input.ok()) << input.status();
    ASSERT_TRUE(gradients.ok()) << gradients.status();
    ASSERT_TRUE(layer.ok()) << layer.status();
    BufferVec inputs = {input->device};
    BufferVec output_gradients = {gradients->device};
    auto baseline = (*layer)->fwd(*executor_, inputs);
    ASSERT_TRUE(baseline.ok()) << baseline.status();
    auto baseline_gradient =
        (*layer)->bwd(*executor_, output_gradients, std::move(baseline->state));
    ASSERT_TRUE(baseline_gradient.ok()) << baseline_gradient.status();
    auto expected_output =
        ReadDeviceActivations(*executor_, baseline->outputs[0], type);
    auto expected_gradient =
        ReadDeviceFloats(*executor_, baseline_gradient->front());
    ASSERT_TRUE(expected_output.ok()) << expected_output.status();
    ASSERT_TRUE(expected_gradient.ok()) << expected_gradient.status();

    BufferVec retained;
    LayerHooks hooks;
    hooks.attention_probabilities_hook = [&](cuda::Executor&, absl::string_view,
                                             const ActivationType&,
                                             const Buffer& buffer) {
      retained.push_back(buffer);
      return absl::OkStatus();
    };
    LayerHooks empty_hooks;
    for (LayerHooks* selected : {&empty_hooks, &hooks, &hooks}) {
      auto output = (*layer)->fwd(*executor_, inputs, selected);
      ASSERT_TRUE(output.ok()) << output.status();
      const size_t forward_callback_count = retained.size();
      auto input_gradient = (*layer)->bwd(*executor_, output_gradients,
                                          std::move(output->state), selected);
      ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();
      EXPECT_EQ(retained.size(), forward_callback_count)
          << "the probabilities hook must not run during backward";
      auto actual_output =
          ReadDeviceActivations(*executor_, output->outputs[0], type);
      auto actual_gradient =
          ReadDeviceFloats(*executor_, input_gradient->front());
      ASSERT_TRUE(actual_output.ok()) << actual_output.status();
      ASSERT_TRUE(actual_gradient.ok()) << actual_gradient.status();
      ASSERT_EQ(actual_output->size(), expected_output->size());
      ASSERT_EQ(actual_gradient->size(), expected_gradient->size());
      EXPECT_EQ(std::memcmp(actual_output->data(), expected_output->data(),
                            expected_output->size() * sizeof(float)),
                0);
      EXPECT_EQ(std::memcmp(actual_gradient->data(), expected_gradient->data(),
                            expected_gradient->size() * sizeof(float)),
                0);
    }
    ASSERT_EQ(retained.size(), 2u);

    // Attention probabilities depend only on Q and K. Perturbing V must not
    // change them, even though V changes the resulting attention activation.
    for (size_t index = 0; index < qkv.size(); ++index)
      if (index % (3 * kWidth) >= 2 * kWidth)
        qkv[index] = -3 * qkv[index];
    auto changed_values = MakeActivationBufferPair(*executor_, qkv, type);
    ASSERT_TRUE(changed_values.ok()) << changed_values.status();
    BufferVec changed_inputs = {changed_values->device};
    auto changed_output = (*layer)->fwd(*executor_, changed_inputs, &hooks);
    ASSERT_TRUE(changed_output.ok()) << changed_output.status();
    ASSERT_EQ(retained.size(), 3u);
    auto expected_probabilities = ReadDeviceFloats(*executor_, retained[0]);
    ASSERT_TRUE(expected_probabilities.ok()) << expected_probabilities.status();
    for (size_t index = 1; index < retained.size(); ++index) {
      auto actual = ReadDeviceFloats(*executor_, retained[index]);
      ASSERT_TRUE(actual.ok()) << actual.status();
      ASSERT_EQ(actual->size(), expected_probabilities->size());
      EXPECT_EQ(std::memcmp(actual->data(), expected_probabilities->data(),
                            actual->size() * sizeof(float)),
                0);
    }
  }
}

TEST_F(LayersTest, AttentionProbabilityHookFailurePreventsOutputPublication) {
  auto layer = AttentionLayer::Create(*executor_, 2, 1, 7, DataType::FP16);
  std::vector<float> qkv(2 * 3 * 7, 0.25f);
  auto input = MakeActivationBufferPair(*executor_, qkv, DataType::FP16);
  ASSERT_TRUE(layer.ok()) << layer.status();
  ASSERT_TRUE(input.ok()) << input.status();
  int probability_calls = 0;
  int activation_calls = 0;
  LayerHooks hooks;
  hooks.attention_probabilities_hook = [&](cuda::Executor&, absl::string_view,
                                           const ActivationType&,
                                           const Buffer&) {
    ++probability_calls;
    return absl::CancelledError("probability inspection stopped");
  };
  hooks.activation_hook = [&](cuda::Executor&, absl::string_view,
                              absl::Span<const ActivationType>,
                              absl::Span<Buffer>) {
    ++activation_calls;
    return absl::OkStatus();
  };
  BufferVec inputs = {input->device};
  auto failed = (*layer)->fwd(*executor_, inputs, &hooks);
  ASSERT_FALSE(failed.ok());
  EXPECT_EQ(failed.status().code(), absl::StatusCode::kCancelled);
  EXPECT_NE(failed.status().message().find("probability inspection stopped"),
            std::string::npos);
  EXPECT_EQ(probability_calls, 1);
  EXPECT_EQ(activation_calls, 0);
  hooks.attention_probabilities_hook = {};
  auto successful = (*layer)->fwd(*executor_, inputs, &hooks);
  ASSERT_TRUE(successful.ok()) << successful.status();
  EXPECT_EQ(probability_calls, 1);
  EXPECT_EQ(activation_calls, 1);
}

TEST_F(LayersTest, FlashAttentionIsCausalAndHasCorrectSingleTokenGradient) {
  auto attention = AttentionLayer::Create(*executor_, kTestContextLength,
                                          kTestAttentionHeads, kTestModelWidth,
                                          DataType::FP16);
  ASSERT_TRUE(attention.ok()) << attention.status();

  constexpr int kPackedWidth = 3 * kTestModelWidth;
  std::vector<float> input(kTestTokenCount * kPackedWidth, 0.0f);
  // Q, K, and V occupy distinct packed regions in each row. Position zero
  // must ignore future values, while position one attends to values 1 and 2
  // using dot products 2 and 4.
  input[0] = 1.0f;
  input[kTestModelWidth] = 1.0f;
  input[2 * kTestModelWidth] = 1.0f;
  input[kPackedWidth] = 2.0f;
  input[kPackedWidth + kTestModelWidth] = 2.0f;
  input[kPackedWidth + 2 * kTestModelWidth] = 2.0f;
  // A future value that position zero is not allowed to observe.
  input[2 * kPackedWidth + 2 * kTestModelWidth] = 4.0f;
  const auto pinned_input = CopyToPageLockedHostArray(*executor_, input);
  auto input_buffer =
      Buffer::Allocate(*executor_, input.size() * sizeof(float));
  ASSERT_TRUE(input_buffer.ok()) << input_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(input_buffer->data(), pinned_input.data(),
                            input_buffer->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);

  BufferVec attention_inputs = {*input_buffer};
  auto output = (*attention)->fwd(*executor_, attention_inputs);

  ASSERT_TRUE(output.ok()) << output.status();

  std::vector<float> output_gradient(kTestTokenCount * kTestModelWidth, 0.0f);
  output_gradient[0] = 1.0f;
  const auto pinned_output_gradient =
      CopyToPageLockedHostArray(*executor_, output_gradient);
  auto gradient_buffer =
      Buffer::Allocate(*executor_, output_gradient.size() * sizeof(float));
  ASSERT_TRUE(gradient_buffer.ok()) << gradient_buffer.status();
  ASSERT_EQ(
      cudaMemcpyAsync(gradient_buffer->data(), pinned_output_gradient.data(),
                      gradient_buffer->size_bytes(), cudaMemcpyHostToDevice,
                      executor_->stream()),
      cudaSuccess);
  BufferVec attention_gradients = {*gradient_buffer};
  auto input_gradient =
      (*attention)
          ->bwd(*executor_, attention_gradients, std::move(output->state));
  ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();
  ASSERT_EQ(input_gradient->size(), 1u);

  auto host_output =
      AllocatePageLockedHostArray<float>(*executor_, output_gradient.size());
  auto host_input_gradient =
      AllocatePageLockedHostArray<float>(*executor_, input.size());
  ASSERT_EQ(cudaMemcpyAsync(host_output.data(), output->outputs[0].data(),
                            output->outputs[0].size_bytes(),
                            cudaMemcpyDeviceToHost, executor_->stream()),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(host_input_gradient.data(),
                            input_gradient->front().data(),
                            input_gradient->front().size_bytes(),
                            cudaMemcpyDeviceToHost, executor_->stream()),
            cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());

  const float scale = 1.0f / std::sqrt(kTestModelWidth / kTestAttentionHeads);
  const float first_weight = std::exp(2.0f * scale);
  const float second_weight = std::exp(4.0f * scale);
  const float expected_position_one =
      (first_weight + 2.0f * second_weight) / (first_weight + second_weight);
  EXPECT_NEAR(host_output[0], 1.0f, 1e-6f);
  EXPECT_NEAR(host_output[kTestModelWidth], expected_position_one, 1e-5f);
  EXPECT_NEAR(host_input_gradient[0], 0.0f, 1e-6f);
  EXPECT_NEAR(host_input_gradient[kTestModelWidth], 0.0f, 1e-6f);
  EXPECT_NEAR(host_input_gradient[2 * kTestModelWidth], 1.0f, 1e-6f);
  EXPECT_NEAR(host_input_gradient[kPackedWidth], 0.0f, 1e-6f);
}

TEST_F(LayerReferenceTest, ForwardAndBackwardAreBitwiseRepeatable) {
  // Head dimensions above 16 exercise multiple independently owned dimension
  // tiles. Lengths 33 and 65 leave partial query/key tiles, including tiles at
  // sequence boundaries. Dense gradients give early keys/values many sums.
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (const auto& [context, heads, width, sequences] :
         {std::tuple{1, 1, 16, 3}, std::tuple{3, 2, 64, 2},
          std::tuple{17, 3, 144, 2}, std::tuple{33, 2, 128, 2},
          std::tuple{33, 2, 160, 2}, std::tuple{65, 1, 96, 2},
          std::tuple{65, 2, 128, 2}, std::tuple{65, 1, 128, 2},
          std::tuple{128, 4, 256, 2}}) {
      for (bool equal_scores : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << "type=" << static_cast<int>(type)
                     << " context=" << context << " heads=" << heads
                     << " width=" << width << " sequences=" << sequences
                     << " equal_scores=" << equal_scores);
        const int rows = context * sequences;
        std::vector<float> qkv(static_cast<size_t>(rows) * 3 * width);
        std::vector<float> gradient(static_cast<size_t>(rows) * width);
        for (size_t index = 0; index < qkv.size(); ++index) {
          qkv[index] = 0.35f * std::sin(static_cast<float>(index) * 0.071f) +
                       0.05f * std::cos(static_cast<float>(index) * 0.19f);
          if (equal_scores &&
              index % (3 * width) < static_cast<size_t>(2 * width)) {
            qkv[index] = 0.0f;
          }
        }
        for (size_t index = 0; index < gradient.size(); ++index)
          gradient[index] = 0.2f * std::cos(static_cast<float>(index) * 0.13f);
        auto inputs = MakeActivationBufferPair(*executor_, qkv, type);
        auto gradients = MakeRawBufferPair<float>(*executor_, gradient);
        auto reference =
            AttentionLayerReference::Create(context, heads, width, type);
        ASSERT_TRUE(inputs.ok()) << inputs.status();
        ASSERT_TRUE(gradients.ok()) << gradients.status();
        ASSERT_TRUE(reference.ok()) << reference.status();

        HostBufferVec reference_inputs = {inputs->host};
        HostBufferVec reference_gradients = {gradients->host};
        auto reference_output = (*reference)->fwd(reference_inputs);

        ASSERT_TRUE(reference_output.ok()) << reference_output.status();
        auto reference_input_gradient =
            (*reference)
                ->bwd(reference_gradients, std::move(reference_output->state));
        ASSERT_TRUE(reference_input_gradient.ok())
            << reference_input_gradient.status();

        std::vector<unsigned char> expected_output;
        std::vector<unsigned char> expected_gradient;
        for (int repeat = 0; repeat < 8; ++repeat) {
          SCOPED_TRACE(testing::Message() << "repeat=" << repeat);
          // New layer/output/scratch allocations avoid dependence on a reused
          // zeroed gradient buffer or retained state in a previous invocation.
          auto attention =
              AttentionLayer::Create(*executor_, context, heads, width, type);
          ASSERT_TRUE(attention.ok()) << attention.status();

          BufferVec device_inputs = {inputs->device};
          BufferVec device_gradients = {gradients->device};
          auto output = (*attention)->fwd(*executor_, device_inputs);

          ASSERT_TRUE(output.ok()) << output.status();
          auto input_gradient =
              (*attention)
                  ->bwd(*executor_, device_gradients, std::move(output->state));
          ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();
          ASSERT_EQ(input_gradient->size(), 1u);
          if (repeat == 0) {
            EXPECT_TRUE(ActivationBuffersNear(output->outputs[0],
                                              reference_output->outputs[0],
                                              type, 2e-3f, 2e-3f));
            EXPECT_TRUE(FloatBuffersNear(input_gradient->front(),
                                         reference_input_gradient->front(),
                                         3e-3f, 3e-3f));
          }
          auto output_bytes = AllocatePageLockedHostArray<unsigned char>(
              *executor_, output->outputs[0].size_bytes());
          auto gradient_bytes = AllocatePageLockedHostArray<unsigned char>(
              *executor_, input_gradient->front().size_bytes());
          ASSERT_EQ(
              cudaMemcpyAsync(output_bytes.data(), output->outputs[0].data(),
                              output->outputs[0].size_bytes(),
                              cudaMemcpyDeviceToHost, executor_->stream()),
              cudaSuccess);
          ASSERT_EQ(cudaMemcpyAsync(
                        gradient_bytes.data(), input_gradient->front().data(),
                        input_gradient->front().size_bytes(),
                        cudaMemcpyDeviceToHost, executor_->stream()),
                    cudaSuccess);
          ASSERT_TRUE(executor_->Synchronize().ok());
          if (repeat == 0) {
            expected_output.assign(output_bytes.data(),
                                   output_bytes.data() + output_bytes.size());
            expected_gradient.assign(
                gradient_bytes.data(),
                gradient_bytes.data() + gradient_bytes.size());
          } else {
            ASSERT_EQ(std::memcmp(expected_output.data(), output_bytes.data(),
                                  expected_output.size()),
                      0)
                << "forward output differs at the bit level";
            ASSERT_EQ(
                std::memcmp(expected_gradient.data(), gradient_bytes.data(),
                            expected_gradient.size()),
                0)
                << "packed dQ/dK/dV differs at the bit level";
          }
        }
      }
    }
  }
}

TEST_F(LayerReferenceTest, SavedForwardStateSurvivesAnotherForwardPass) {
  constexpr int kContext = 33;
  constexpr int kHeads = 2;
  constexpr int kWidth = 160;
  constexpr int kRows = 2 * kContext;
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    SCOPED_TRACE(testing::Message() << "type=" << static_cast<int>(type));
    std::vector<float> qkv_a(kRows * 3 * kWidth);
    std::vector<float> qkv_b(qkv_a.size());
    std::vector<float> output_gradient(kRows * kWidth);
    for (size_t index = 0; index < qkv_a.size(); ++index) {
      qkv_a[index] = 0.35f * std::sin(static_cast<float>(index) * 0.071f) +
                     0.05f * std::cos(static_cast<float>(index) * 0.19f);
      // B changes both the softmax statistics and values while keeping A alive.
      qkv_b[index] = index % (3 * kWidth) < static_cast<size_t>(2 * kWidth)
                         ? 20.0f * qkv_a[index] + 1.0f
                         : qkv_a[index] + 0.75f;
    }
    for (size_t index = 0; index < output_gradient.size(); ++index)
      output_gradient[index] =
          0.2f * std::cos(static_cast<float>(index) * 0.13f);
    auto input_a = MakeActivationBufferPair(*executor_, qkv_a, type);
    auto input_b = MakeActivationBufferPair(*executor_, qkv_b, type);
    auto gradients = MakeRawBufferPair<float>(*executor_, output_gradient);
    auto attention =
        AttentionLayer::Create(*executor_, kContext, kHeads, kWidth, type);
    ASSERT_TRUE(input_a.ok()) << input_a.status();
    ASSERT_TRUE(input_b.ok()) << input_b.status();
    ASSERT_TRUE(gradients.ok()) << gradients.status();
    ASSERT_TRUE(attention.ok()) << attention.status();
    BufferVec device_inputs_a = {input_a->device};
    BufferVec device_inputs_b = {input_b->device};
    BufferVec device_gradients = {gradients->device};
    auto output_a = (*attention)->fwd(*executor_, device_inputs_a);
    ASSERT_TRUE(output_a.ok()) << output_a.status();
    // A copied state gives an exact baseline without consuming the state that
    // will be used after B. Saved buffers remain shared and read-only.
    auto baseline_gradient =
        (*attention)->bwd(*executor_, device_gradients, output_a->state);
    ASSERT_TRUE(baseline_gradient.ok()) << baseline_gradient.status();
    ASSERT_EQ(baseline_gradient->size(), 1u);
    auto expected_output =
        ReadDeviceActivations(*executor_, output_a->outputs[0], type);
    auto expected_gradient =
        ReadDeviceFloats(*executor_, baseline_gradient->front());
    ASSERT_TRUE(expected_output.ok()) << expected_output.status();
    ASSERT_TRUE(expected_gradient.ok()) << expected_gradient.status();

    auto output_b = (*attention)->fwd(*executor_, device_inputs_b);
    ASSERT_TRUE(output_b.ok()) << output_b.status();
    auto retained_gradient =
        (*attention)
            ->bwd(*executor_, device_gradients, std::move(output_a->state));
    ASSERT_TRUE(retained_gradient.ok()) << retained_gradient.status();
    ASSERT_EQ(retained_gradient->size(), 1u);
    auto actual_output =
        ReadDeviceActivations(*executor_, output_a->outputs[0], type);
    auto actual_gradient =
        ReadDeviceFloats(*executor_, retained_gradient->front());
    ASSERT_TRUE(actual_output.ok()) << actual_output.status();
    ASSERT_TRUE(actual_gradient.ok()) << actual_gradient.status();
    ASSERT_EQ(actual_output->size(), expected_output->size());
    ASSERT_EQ(actual_gradient->size(), expected_gradient->size());
    EXPECT_EQ(std::memcmp(actual_output->data(), expected_output->data(),
                          expected_output->size() * sizeof(float)),
              0);
    EXPECT_EQ(std::memcmp(actual_gradient->data(), expected_gradient->data(),
                          expected_gradient->size() * sizeof(float)),
              0);
  }
}

TEST_F(LayersTest, BackwardRejectsPartialSequencesBeforeGatheringQueries) {
  constexpr int kContext = 4;
  constexpr int kWidth = 16;
  constexpr int kRows = 3;
  auto attention =
      AttentionLayer::Create(*executor_, kContext, 1, kWidth, DataType::FP16);
  auto qkv = Buffer::Allocate(*executor_, kRows * 3 * kWidth * sizeof(float));
  auto output = Buffer::Allocate(*executor_, kRows * kWidth * sizeof(float));
  auto gradient = Buffer::Allocate(*executor_, kRows * kWidth * sizeof(float));
  auto statistics = Buffer::Allocate(*executor_, kRows * sizeof(float));
  ASSERT_TRUE(attention.ok()) << attention.status();
  ASSERT_TRUE(qkv.ok()) << qkv.status();
  ASSERT_TRUE(output.ok()) << output.status();
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  ASSERT_TRUE(statistics.ok()) << statistics.status();
  BackwardState state;
  state.layer = attention->get();
  state.intermediates = {*qkv, *output, *statistics, *statistics};
  BufferVec gradients = {*gradient};
  auto result = (*attention)->bwd(*executor_, gradients, std::move(state));
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("divisible by context_length"),
            std::string::npos);
}

TEST_F(LayersTest, BackwardRejectsInvalidSoftmaxStatistics) {
  constexpr int kContext = 4;
  constexpr int kHeads = 2;
  constexpr int kWidth = 32;
  constexpr int kRows = 2 * kContext;
  constexpr size_t kStatisticsBytes = kRows * kHeads * sizeof(float);
  auto attention = AttentionLayer::Create(*executor_, kContext, kHeads, kWidth,
                                          DataType::FP16);
  auto qkv = Buffer::Allocate(*executor_, kRows * 3 * kWidth * sizeof(float));
  auto output = Buffer::Allocate(*executor_, kRows * kWidth * sizeof(float));
  auto gradient = Buffer::Allocate(*executor_, kRows * kWidth * sizeof(float));
  auto statistics = Buffer::Allocate(*executor_, kStatisticsBytes);
  auto short_statistics =
      Buffer::Allocate(*executor_, kStatisticsBytes - sizeof(float));
  auto long_statistics =
      Buffer::Allocate(*executor_, kStatisticsBytes + sizeof(float));
  auto other_executor = cuda::Executor::Create();
  ASSERT_TRUE(attention.ok()) << attention.status();
  ASSERT_TRUE(qkv.ok()) << qkv.status();
  ASSERT_TRUE(output.ok()) << output.status();
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  ASSERT_TRUE(statistics.ok()) << statistics.status();
  ASSERT_TRUE(short_statistics.ok()) << short_statistics.status();
  ASSERT_TRUE(long_statistics.ok()) << long_statistics.status();
  ASSERT_TRUE(other_executor.ok()) << other_executor.status();
  auto foreign_statistics =
      Buffer::Allocate(**other_executor, kStatisticsBytes);
  ASSERT_TRUE(foreign_statistics.ok()) << foreign_statistics.status();

  // Each new saved buffer must independently validate both its shape and its
  // owning executor before any backward kernel reads it.
  for (int index : {2, 3}) {
    for (const auto& invalid :
         {*short_statistics, *long_statistics, *foreign_statistics}) {
      SCOPED_TRACE(testing::Message()
                   << "state index=" << index
                   << " bytes=" << invalid.size_bytes() << " foreign executor="
                   << (&invalid.executor() != executor_.get()));
      BackwardState state;
      state.layer = attention->get();
      state.intermediates = {*qkv, *output, *statistics, *statistics};
      state.intermediates[index] = invalid;
      BufferVec gradients = {*gradient};
      auto result = (*attention)->bwd(*executor_, gradients, std::move(state));
      ASSERT_FALSE(result.ok());
      EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
      EXPECT_NE(result.status().message().find(
                    index == 2 ? "saved maxima" : "saved normalizers"),
                std::string::npos);
    }
  }
}

}  // namespace
}  // namespace pluto::llm
