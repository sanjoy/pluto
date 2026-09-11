#include "src/llm/layers/sparse_autoencoder.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/llm/layer.h"
#include "src/llm/layers/test_util.h"

namespace pluto::llm {
namespace {

TEST_F(LayersTest, ExposesParametersInDocumentedOrder) {
  auto layer =
      SparseAutoEncoderLayer::Create(*executor_, 32, 48, DataType::BF16);
  ASSERT_TRUE(layer.ok()) << layer.status();
  EXPECT_EQ((*layer)->input_dim(), 32);
  EXPECT_EQ((*layer)->feature_dim(), 48);
  EXPECT_EQ((*layer)->output_type(), DataType::BF16);

  const auto weights = (*layer)->weights();
  const auto gradients = (*layer)->gradients();
  ASSERT_EQ(weights.size(), 4u);
  ASSERT_EQ(gradients.size(), 4u);
  EXPECT_EQ(weights[0].size_bytes(), 48u * 32u * sizeof(float));
  EXPECT_EQ(weights[1].size_bytes(), 48u * sizeof(float));
  EXPECT_EQ(weights[2].size_bytes(), 32u * 48u * sizeof(float));
  EXPECT_EQ(weights[3].size_bytes(), 32u * sizeof(float));
  for (size_t index = 0; index < weights.size(); ++index)
    EXPECT_EQ(gradients[index].size_bytes(), weights[index].size_bytes());
  EXPECT_EQ((*layer)->decoder().data(), weights[2].data());
}

TEST_F(LayersTest, RejectsUnsupportedShapesTypesAndPenalties) {
  EXPECT_EQ(SparseAutoEncoderLayer::Create(*executor_, 15, 16, DataType::FP16)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(SparseAutoEncoderLayer::Create(*executor_, 16, 15, DataType::FP16)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(SparseAutoEncoderLayer::Create(*executor_, 16, 16, DataType::FP8)
                .status()
                .code(),
            absl::StatusCode::kUnimplemented);
  EXPECT_EQ(SparseAutoEncoderLossLayer::Create(*executor_, 16, 16, -0.1f,
                                               DataType::FP16)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(SparseAutoEncoderLossLayer::Create(
                *executor_, 16, 16, std::numeric_limits<float>::infinity(),
                DataType::FP16)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(LayersTest, ForwardRequiresACompleteActivationMatrix) {
  auto layer =
      SparseAutoEncoderLayer::Create(*executor_, 16, 16, DataType::FP16);
  auto short_input = Buffer::Allocate(*executor_, 15 * sizeof(float));
  ASSERT_TRUE(layer.ok()) << layer.status();
  ASSERT_TRUE(short_input.ok()) << short_input.status();
  BufferVec inputs = {*short_input};
  BackwardState state;
  auto output = (*layer)->fwd(*executor_, inputs, state);
  ASSERT_FALSE(output.ok());
  EXPECT_EQ(output.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(LayersTest, BackwardAcceptsReconstructionGradientAlone) {
  constexpr int kRows = 16;
  constexpr int kDimension = 16;
  auto layer = SparseAutoEncoderLayer::Create(*executor_, kDimension,
                                              kDimension, DataType::FP16);
  auto input = Buffer::Allocate(*executor_, kRows * kDimension * sizeof(float));
  auto gradient =
      Buffer::Allocate(*executor_, kRows * kDimension * sizeof(float));
  ASSERT_TRUE(layer.ok()) << layer.status();
  ASSERT_TRUE(input.ok()) << input.status();
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  ASSERT_EQ(cudaMemsetAsync(input->data(), 0, input->size_bytes(),
                            executor_->stream()),
            cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(gradient->data(), 0, gradient->size_bytes(),
                            executor_->stream()),
            cudaSuccess);

  BackwardState state;
  BufferVec inputs = {*input};
  auto output = (*layer)->fwd(*executor_, inputs, state);
  ASSERT_TRUE(output.ok()) << output.status();
  BufferVec gradients = {*gradient};
  auto input_gradient = (*layer)->bwd(*executor_, gradients, std::move(state));
  ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();
  ASSERT_EQ(input_gradient->size(), 1u);
  EXPECT_EQ(input_gradient->front().size_bytes(), input->size_bytes());
}

TEST_F(LayersTest, StatisticsAreOptInAndValidateTheirStateAndRowLimit) {
  auto input = Buffer::Allocate(*executor_, 16 * 16 * sizeof(float));
  auto default_layer =
      SparseAutoEncoderLayer::Create(*executor_, 16, 16, DataType::FP16);
  auto stats_layer = SparseAutoEncoderLayer::Create(
      *executor_, 16, 16, DataType::FP16,
      SparseAutoEncoderLayer::Mode::kCollectStatistics);
  ASSERT_TRUE(input.ok()) << input.status();
  ASSERT_TRUE(default_layer.ok()) << default_layer.status();
  ASSERT_TRUE(stats_layer.ok()) << stats_layer.status();
  ASSERT_EQ(cudaMemsetAsync(input->data(), 0, input->size_bytes(),
                            executor_->stream()),
            cudaSuccess);
  BackwardState default_state;
  BackwardState stats_state;
  ASSERT_TRUE(
      (*default_layer)->fwd(*executor_, BufferVec{*input}, default_state).ok());
  ASSERT_TRUE(
      (*stats_layer)->fwd(*executor_, BufferVec{*input}, stats_state).ok());
  EXPECT_EQ(default_state.intermediates.size(), 2u);
  EXPECT_EQ(stats_state.intermediates.size(), 3u);
  EXPECT_EQ((*default_layer)
                ->ReadZStatistics(*executor_, default_state)
                .status()
                .code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ((*stats_layer)
                ->ReadZStatistics(*executor_, BackwardState{})
                .status()
                .code(),
            absl::StatusCode::kFailedPrecondition);
  for (int invalid_rows : {-1, 17}) {
    EXPECT_EQ((*stats_layer)
                  ->ReadZStatistics(*executor_, stats_state, invalid_rows)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  auto other_executor = cuda::Executor::Create();
  ASSERT_TRUE(other_executor.ok()) << other_executor.status();
  EXPECT_EQ((*stats_layer)
                ->ReadZStatistics(**other_executor, stats_state)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  stats_state.intermediates[2] = *input;
  EXPECT_EQ(
      (*stats_layer)->ReadZStatistics(*executor_, stats_state).status().code(),
      absl::StatusCode::kInvalidArgument);
}

TEST_F(LayersTest, ParallelLossHandlesGpt2BatchTenShape) {
  constexpr int kRows = 10 * 1024;
  constexpr int kInputDim = 512;
  constexpr int kFeatureDim = 4096;
  auto loss = SparseAutoEncoderLossLayer::Create(
      *executor_, kInputDim, kFeatureDim, 0.5f, DataType::BF16);
  auto input = Buffer::Allocate(
      *executor_, static_cast<size_t>(kRows) * kInputDim * sizeof(uint16_t));
  auto reconstruction = Buffer::Allocate(
      *executor_, static_cast<size_t>(kRows) * kInputDim * sizeof(uint16_t));
  auto latents = Buffer::Allocate(
      *executor_, static_cast<size_t>(kRows) * kFeatureDim * sizeof(uint16_t));
  auto decoder = Buffer::Allocate(
      *executor_, static_cast<size_t>(kInputDim) * kFeatureDim * sizeof(float));
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE(input.ok()) << input.status();
  ASSERT_TRUE(reconstruction.ok()) << reconstruction.status();
  ASSERT_TRUE(latents.ok()) << latents.status();
  ASSERT_TRUE(decoder.ok()) << decoder.status();
  for (Buffer* buffer : {&*input, &*reconstruction, &*latents, &*decoder}) {
    ASSERT_EQ(cudaMemsetAsync(buffer->data(), 0, buffer->size_bytes(),
                              executor_->stream()),
              cudaSuccess);
  }

  BackwardState state;
  BufferVec inputs = {*input, *reconstruction, *latents, *decoder};
  auto output = (*loss)->fwd(*executor_, inputs, state);
  ASSERT_TRUE(output.ok()) << output.status();
  auto host_output = AllocatePageLockedHostArray<float>(*executor_, 1);
  ASSERT_EQ(cudaMemcpyAsync(host_output.data(), output->data(), sizeof(float),
                            cudaMemcpyDeviceToHost, executor_->stream()),
            cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());
  EXPECT_FLOAT_EQ(host_output[0], 0.0f);
}

}  // namespace
}  // namespace pluto::llm
