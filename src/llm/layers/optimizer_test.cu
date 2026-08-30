#include "src/llm/optimizer.h"

#include <cuda_runtime.h>

#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "src/gpu/buffer.h"
#include "src/llm/layer.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/test_util.h"

namespace pluto::llm {
namespace {

TEST_F(LayersTest, UpdatesFp32MasterWeightsAndClearsGradients) {
  auto dense = FullyConnectedLayer::Create(
      16, DataType::FP16, stream_);
  ASSERT_TRUE(dense.ok()) << dense.status();
  ASSERT_TRUE((*dense)->InitializeIdentity().ok());
  AdamWConfig config{
      .learning_rate = 0.1f,
      .beta1 = 0.0f,
      .beta2 = 0.0f,
      .epsilon = 1e-6f,
      .weight_decay = 0.0f,
  };
  auto optimizer = AdamWOptimizer::Create(**dense, config, stream_);
  ASSERT_TRUE(optimizer.ok()) << optimizer.status();
  ASSERT_EQ((*optimizer)->parameter_tensor_count(), 2u);

  for (Buffer& gradient : (*dense)->gradients()) {
    std::vector<float> values(gradient.size_bytes() / sizeof(float), 2.0f);
    ASSERT_EQ(cudaMemcpyAsync(gradient.data(), values.data(),
                              gradient.size_bytes(), cudaMemcpyHostToDevice,
                              stream_),
              cudaSuccess);
  }
  ASSERT_TRUE((*optimizer)->Step().ok());
  EXPECT_EQ((*optimizer)->step(), 1);

  std::vector<float> matrix(16 * 16);
  std::vector<float> bias(16);
  std::vector<float> matrix_gradient(16 * 16);
  ASSERT_EQ(cudaMemcpyAsync(matrix.data(), (*dense)->weights()[0].data(),
                            matrix.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(bias.data(), (*dense)->weights()[1].data(),
                            bias.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(matrix_gradient.data(),
                            (*dense)->gradients()[0].data(),
                            matrix_gradient.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_),
            cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  EXPECT_NEAR(matrix[0], 0.9f, 1e-5f);
  EXPECT_NEAR(matrix[1], -0.1f, 1e-5f);
  EXPECT_NEAR(bias[0], -0.1f, 1e-5f);
  EXPECT_FLOAT_EQ(matrix_gradient[0], 0.0f);
}

TEST_F(LayersTest, DeduplicatesTiedEmbeddingWeights) {
  ComposedLayerBuilder builder;
  auto embedding = EmbeddingLookupLayer::Create(
      17, kTestModelWidth, DataType::BF16, stream_);
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  EmbeddingLookupLayer* embedding_pointer = embedding->get();
  ASSERT_TRUE(builder.add(std::move(*embedding)).ok());
  ASSERT_TRUE(
      builder.add(LanguageModelingHeadLayer::Create(embedding_pointer)).ok());
  auto model = builder.create();
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_EQ((*model)->weights().size(), 2u);

  auto optimizer =
      AdamWOptimizer::Create(**model, AdamWConfig{}, stream_);
  ASSERT_TRUE(optimizer.ok()) << optimizer.status();
  EXPECT_EQ((*optimizer)->parameter_tensor_count(), 1u);
}

}  // namespace
}  // namespace pluto::llm
