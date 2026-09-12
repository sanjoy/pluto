#include "src/llm/layers/attention.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/llm/layer.h"
#include "src/llm/layers/reference_test_util.h"
#include "src/llm/layers/test_util.h"

namespace pluto::llm {
namespace {

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
