#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/llm/adamw_optimizer.h"
#include "src/llm/layer.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/test_util.h"

namespace pluto::llm {
namespace {

TEST_F(LayersTest, UpdatesFp32MasterWeightsAndClearsGradients) {
  auto dense = FullyConnectedLayer::Create(*executor_, 16, DataType::FP16);
  ASSERT_TRUE(dense.ok()) << dense.status();
  ASSERT_TRUE((*dense)->InitializeIdentity().ok());
  AdamWConfig config{
      .learning_rate = 0.1f,
      .beta1 = 0.0f,
      .beta2 = 0.0f,
      .epsilon = 1e-6f,
      .weight_decay = 0.0f,
  };
  auto optimizer = AdamWOptimizer::Create(*executor_, **dense, config);
  ASSERT_TRUE(optimizer.ok()) << optimizer.status();
  // Own the concrete AdamW implementation through the algorithm-independent
  // interface so the lifecycle calls below exercise virtual dispatch.
  std::unique_ptr<Optimizer> optimizer_interface = std::move(*optimizer);
  ASSERT_EQ(optimizer_interface->parameter_tensor_count(), 2u);

  for (Buffer& gradient : (*dense)->gradients()) {
    std::vector<float> values(gradient.size_bytes() / sizeof(float), 2.0f);
    const auto pinned_values = CopyToPageLockedHostArray(*executor_, values);
    ASSERT_EQ(cudaMemcpyAsync(gradient.data(), pinned_values.data(),
                              gradient.size_bytes(), cudaMemcpyHostToDevice,
                              executor_->stream()),
              cudaSuccess);
  }
  ASSERT_TRUE(optimizer_interface->ApplyStep().ok());
  EXPECT_EQ(optimizer_interface->step(), 1);

  auto matrix = AllocatePageLockedHostArray<float>(*executor_, 16 * 16);
  auto bias = AllocatePageLockedHostArray<float>(*executor_, 16);
  auto matrix_gradient =
      AllocatePageLockedHostArray<float>(*executor_, 16 * 16);
  ASSERT_EQ(cudaMemcpyAsync(matrix.data(), (*dense)->weights()[0].data(),
                            matrix.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, executor_->stream()),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(bias.data(), (*dense)->weights()[1].data(),
                            bias.size() * sizeof(float), cudaMemcpyDeviceToHost,
                            executor_->stream()),
            cudaSuccess);
  ASSERT_EQ(
      cudaMemcpyAsync(matrix_gradient.data(), (*dense)->gradients()[0].data(),
                      matrix_gradient.size() * sizeof(float),
                      cudaMemcpyDeviceToHost, executor_->stream()),
      cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());
  EXPECT_NEAR(matrix[0], 0.9f, 1e-5f);
  EXPECT_NEAR(matrix[1], -0.1f, 1e-5f);
  EXPECT_NEAR(bias[0], -0.1f, 1e-5f);
  EXPECT_FLOAT_EQ(matrix_gradient[0], 0.0f);
}

TEST_F(LayersTest, DeduplicatesTiedEmbeddingWeights) {
  ComposedLayerBuilder builder;
  auto embedding = EmbeddingLookupLayer::Create(*executor_, 17, kTestModelWidth,
                                                DataType::BF16);
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  EmbeddingLookupLayer* embedding_pointer = embedding->get();
  ASSERT_TRUE(builder.add(std::move(*embedding)).ok());
  ASSERT_TRUE(
      builder.add(LanguageModelingHeadLayer::Create(embedding_pointer)).ok());
  auto model = builder.create("tied_embedding_model");
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_EQ((*model)->weights().size(), 2u);

  auto optimizer = Optimizer::Create(*executor_, **model, AdamWConfig{});
  ASSERT_TRUE(optimizer.ok()) << optimizer.status();
  EXPECT_EQ((*optimizer)->parameter_tensor_count(), 1u);
}

TEST_F(LayersTest, PartialParameterTilesMatchScalarAdamAcrossSteps) {
  for (int width : {1, 3, 8, 17}) {
    SCOPED_TRACE(width);
    // Both a sub-tile bias and a matrix with a partial last tile are real
    // compact allocations. Check every element, including the final one.
    auto dense = FullyConnectedLayer::Create(*executor_, width, width + 2,
                                             DataType::BF16);
    ASSERT_TRUE(dense.ok()) << dense.status();
    const AdamWConfig config{.learning_rate = 0.01f,
                             .beta1 = 0.8f,
                             .beta2 = 0.9f,
                             .epsilon = 1e-6f,
                             .weight_decay = 0.02f};
    auto optimizer = AdamWOptimizer::Create(*executor_, **dense, config);
    ASSERT_TRUE(optimizer.ok()) << optimizer.status();
    std::vector<std::vector<double>> expected, first, second;
    for (const Buffer& weight : (*dense)->weights()) {
      const size_t count = weight.size_bytes() / sizeof(float);
      expected.emplace_back(count, 0.0);
      first.emplace_back(count, 0.0);
      second.emplace_back(count, 0.0);
    }
    for (int step = 1; step <= 3; ++step) {
      for (size_t tensor = 0; tensor < expected.size(); ++tensor) {
        const auto& gradient = (*dense)->gradients()[tensor];
        auto values = AllocatePageLockedHostArray<float>(
            *executor_, expected[tensor].size());
        for (size_t index = 0; index < values.size(); ++index) {
          values[index] = ((index + step) % 2 == 0 ? 1.0f : -1.0f) *
                          (0.125f * (index + 1) + step);
          const double g = values[index];
          double& m = first[tensor][index];
          double& v = second[tensor][index];
          double& w = expected[tensor][index];
          m = config.beta1 * m + (1.0 - config.beta1) * g;
          v = config.beta2 * v + (1.0 - config.beta2) * g * g;
          const double m_hat = m / (1.0 - std::pow(config.beta1, step));
          const double v_hat = v / (1.0 - std::pow(config.beta2, step));
          w -= config.learning_rate *
               (m_hat / (std::sqrt(v_hat) + config.epsilon) +
                config.weight_decay * w);
        }
        ASSERT_EQ(cudaMemcpyAsync(gradient.data(), values.data(),
                                  gradient.size_bytes(), cudaMemcpyHostToDevice,
                                  executor_->stream()),
                  cudaSuccess);
      }
      ASSERT_TRUE((*optimizer)->ApplyStep().ok());
      EXPECT_EQ((*optimizer)->step(), step);
      for (size_t tensor = 0; tensor < expected.size(); ++tensor) {
        auto actual = AllocatePageLockedHostArray<float>(
            *executor_, expected[tensor].size());
        auto gradient = AllocatePageLockedHostArray<float>(
            *executor_, expected[tensor].size());
        ASSERT_EQ(
            cudaMemcpyAsync(actual.data(), (*dense)->weights()[tensor].data(),
                            actual.size_bytes(), cudaMemcpyDeviceToHost,
                            executor_->stream()),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(gradient.data(),
                                  (*dense)->gradients()[tensor].data(),
                                  gradient.size_bytes(), cudaMemcpyDeviceToHost,
                                  executor_->stream()),
                  cudaSuccess);
        ASSERT_TRUE(executor_->Synchronize().ok());
        for (size_t index = 0; index < actual.size(); ++index) {
          EXPECT_NEAR(actual[index], expected[tensor][index], 2e-7);
          EXPECT_FLOAT_EQ(gradient[index], 0.0f);
        }
      }
    }
  }
}

TEST_F(LayersTest, LearningRateSchedulePreservesAdamMomentsAndStep) {
  auto dense = FullyConnectedLayer::Create(*executor_, 16, DataType::FP16);
  ASSERT_TRUE(dense.ok()) << dense.status();
  ASSERT_TRUE((*dense)->InitializeIdentity().ok());
  const AdamWConfig config{.learning_rate = 0.1f,
                           .beta1 = 0.8f,
                           .beta2 = 0.9f,
                           .epsilon = 1e-6f,
                           .weight_decay = 0.02f};
  auto optimizer = AdamWOptimizer::Create(*executor_, **dense, config);
  ASSERT_TRUE(optimizer.ok()) << optimizer.status();
  const float rates[] = {0.1f, 0.025f, 0.05f};
  const float gradients[] = {2.0f, -1.0f, 0.5f};
  double expected_diagonal = 1.0;
  double expected_off_diagonal = 0.0;
  double first_moment = 0.0;
  double second_moment = 0.0;
  auto matrix = AllocatePageLockedHostArray<float>(*executor_, 16 * 16);
  for (int update = 0; update < 3; ++update) {
    SCOPED_TRACE(update);
    for (Buffer& gradient : (*dense)->gradients()) {
      const auto values = CopyToPageLockedHostArray(
          *executor_, std::vector<float>(gradient.size_bytes() / sizeof(float),
                                         gradients[update]));
      ASSERT_EQ(
          cudaMemcpyAsync(gradient.data(), values.data(), gradient.size_bytes(),
                          cudaMemcpyHostToDevice, executor_->stream()),
          cudaSuccess);
    }
    // Change the rate after filling gradients: the setter must not clear them
    // or reset the accumulated first/second moments from previous updates.
    ASSERT_TRUE((*optimizer)->SetLearningRate(rates[update]).ok());
    EXPECT_EQ((*optimizer)->step(), update);
    EXPECT_EQ((*optimizer)->parameter_tensor_count(), 2u);
    ASSERT_TRUE((*optimizer)->ApplyStep().ok());
    EXPECT_EQ((*optimizer)->step(), update + 1);

    first_moment =
        config.beta1 * first_moment + (1.0 - config.beta1) * gradients[update];
    second_moment = config.beta2 * second_moment + (1.0 - config.beta2) *
                                                       gradients[update] *
                                                       gradients[update];
    const double corrected_first =
        first_moment / (1.0 - std::pow(config.beta1, update + 1));
    const double corrected_second =
        second_moment / (1.0 - std::pow(config.beta2, update + 1));
    const double adaptive_update =
        corrected_first / (std::sqrt(corrected_second) + config.epsilon);
    expected_diagonal -=
        rates[update] *
        (adaptive_update + config.weight_decay * expected_diagonal);
    expected_off_diagonal -=
        rates[update] *
        (adaptive_update + config.weight_decay * expected_off_diagonal);
    ASSERT_EQ(cudaMemcpyAsync(matrix.data(), (*dense)->weights()[0].data(),
                              matrix.size_bytes(), cudaMemcpyDeviceToHost,
                              executor_->stream()),
              cudaSuccess);
    ASSERT_TRUE(executor_->Synchronize().ok());
    EXPECT_NEAR(matrix[0], expected_diagonal, 1e-6);
    EXPECT_NEAR(matrix[1], expected_off_diagonal, 1e-6);
  }
}

TEST_F(LayersTest, InvalidLearningRatesLeaveConfigurationAndStateUnchanged) {
  auto dense = FullyConnectedLayer::Create(*executor_, 16, DataType::FP16);
  ASSERT_TRUE(dense.ok()) << dense.status();
  ASSERT_TRUE((*dense)->InitializeIdentity().ok());
  const AdamWConfig config{.learning_rate = 0.1f,
                           .beta1 = 0.0f,
                           .beta2 = 0.0f,
                           .epsilon = 1e-6f,
                           .weight_decay = 0.0f};
  auto optimizer = AdamWOptimizer::Create(*executor_, **dense, config);
  ASSERT_TRUE(optimizer.ok()) << optimizer.status();
  // Establish a nonzero logical step, then leave a real gradient pending while
  // exercising every rejected rate. The next update must use the last valid
  // rate and consume that gradient normally.
  ASSERT_TRUE((*optimizer)->ApplyStep().ok());
  ASSERT_TRUE((*optimizer)->SetLearningRate(0.025f).ok());
  for (Buffer& gradient : (*dense)->gradients()) {
    const auto values = CopyToPageLockedHostArray(
        *executor_,
        std::vector<float>(gradient.size_bytes() / sizeof(float), 2.0f));
    ASSERT_EQ(
        cudaMemcpyAsync(gradient.data(), values.data(), gradient.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        cudaSuccess);
  }
  for (float invalid : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
    EXPECT_EQ((*optimizer)->SetLearningRate(invalid).code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ((*optimizer)->step(), 1);
    EXPECT_EQ((*optimizer)->parameter_tensor_count(), 2u);
    AdamWConfig invalid_config = config;
    invalid_config.learning_rate = invalid;
    EXPECT_EQ(AdamWOptimizer::Create(*executor_, **dense, invalid_config)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  ASSERT_TRUE((*optimizer)->ApplyStep().ok());
  EXPECT_EQ((*optimizer)->step(), 2);
  auto matrix = AllocatePageLockedHostArray<float>(*executor_, 16 * 16);
  ASSERT_EQ(cudaMemcpyAsync(matrix.data(), (*dense)->weights()[0].data(),
                            matrix.size_bytes(), cudaMemcpyDeviceToHost,
                            executor_->stream()),
            cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());
  EXPECT_NEAR(matrix[0], 0.975f, 1e-6f);
  EXPECT_NEAR(matrix[1], -0.025f, 1e-6f);
}

}  // namespace
}  // namespace pluto::llm
