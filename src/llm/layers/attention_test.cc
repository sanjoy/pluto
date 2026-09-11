#include "src/llm/layers/attention.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
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
  std::vector<float> input(kTestBatchSize * kPackedWidth, 0.0f);
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
  const auto pinned_input = CopyToPageLockedHostArray(input);
  auto input_buffer =
      Buffer::Allocate(*executor_, input.size() * sizeof(float));
  ASSERT_TRUE(input_buffer.ok()) << input_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(input_buffer->data(), pinned_input.data(),
                            input_buffer->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);

  Tape tape;
  BufferVec attention_inputs = {*input_buffer};
  auto output = (*attention)->fwd(*executor_, attention_inputs, &tape);
  ASSERT_TRUE(output.ok()) << output.status();

  std::vector<float> output_gradient(kTestBatchSize * kTestModelWidth, 0.0f);
  output_gradient[0] = 1.0f;
  const auto pinned_output_gradient =
      CopyToPageLockedHostArray(output_gradient);
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
      (*attention)->bwd(*executor_, attention_gradients, std::move(tape));
  ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();
  ASSERT_EQ(input_gradient->size(), 1u);

  auto host_output = AllocatePageLockedHostArray<float>(output_gradient.size());
  auto host_input_gradient = AllocatePageLockedHostArray<float>(input.size());
  ASSERT_EQ(
      cudaMemcpyAsync(host_output.data(), output->data(), output->size_bytes(),
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
  // tiles. Odd sequence lengths and multiple sequences exercise both causal
  // boundaries. Dense gradients make each early key/value receive many sums.
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (const auto& [context, heads, width, sequences] :
         {std::tuple{1, 1, 16, 3}, std::tuple{3, 2, 64, 2},
          std::tuple{17, 3, 144, 2}, std::tuple{33, 2, 128, 2},
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
        ReferenceTape reference_tape;
        HostBufferVec reference_inputs = {inputs->host};
        HostBufferVec reference_gradients = {gradients->host};
        auto reference_output =
            (*reference)->fwd(reference_inputs, &reference_tape);
        ASSERT_TRUE(reference_output.ok()) << reference_output.status();
        auto reference_input_gradient =
            (*reference)->bwd(reference_gradients, std::move(reference_tape));
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
          Tape tape;
          BufferVec device_inputs = {inputs->device};
          BufferVec device_gradients = {gradients->device};
          auto output = (*attention)->fwd(*executor_, device_inputs, &tape);
          ASSERT_TRUE(output.ok()) << output.status();
          auto input_gradient =
              (*attention)->bwd(*executor_, device_gradients, std::move(tape));
          ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();
          ASSERT_EQ(input_gradient->size(), 1u);
          if (repeat == 0) {
            EXPECT_TRUE(ActivationBuffersNear(*output, *reference_output, type,
                                              2e-3f, 2e-3f));
            EXPECT_TRUE(FloatBuffersNear(input_gradient->front(),
                                         reference_input_gradient->front(),
                                         3e-3f, 3e-3f));
          }
          auto output_bytes =
              AllocatePageLockedHostArray<unsigned char>(output->size_bytes());
          auto gradient_bytes = AllocatePageLockedHostArray<unsigned char>(
              input_gradient->front().size_bytes());
          ASSERT_EQ(
              cudaMemcpyAsync(output_bytes.data(), output->data(),
                              output->size_bytes(), cudaMemcpyDeviceToHost,
                              executor_->stream()),
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

TEST_F(LayersTest, BackwardRejectsPartialSequencesBeforeGatheringQueries) {
  constexpr int kContext = 4;
  constexpr int kWidth = 16;
  constexpr int kRows = 3;
  auto attention =
      AttentionLayer::Create(*executor_, kContext, 1, kWidth, DataType::FP16);
  auto qkv = Buffer::Allocate(*executor_, kRows * 3 * kWidth * sizeof(float));
  auto output = Buffer::Allocate(*executor_, kRows * kWidth * sizeof(float));
  auto gradient = Buffer::Allocate(*executor_, kRows * kWidth * sizeof(float));
  ASSERT_TRUE(attention.ok()) << attention.status();
  ASSERT_TRUE(qkv.ok()) << qkv.status();
  ASSERT_TRUE(output.ok()) << output.status();
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  Tape tape;
  tape.intermediates = {*qkv, *output};
  BufferVec gradients = {*gradient};
  auto result = (*attention)->bwd(*executor_, gradients, std::move(tape));
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::llm
