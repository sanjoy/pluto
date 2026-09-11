#include "src/llm/recipes/gpt2.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/llm/layer.h"

namespace pluto::llm {
namespace {

class Gpt2Test : public testing::Test {
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
    executor_.reset();
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(Gpt2Test, ActivationGeneratorRejectsInvalidBlockCounts) {
  auto negative =
      CreateActivationGenerator(*executor_, -1, DataType::BF16, 123);
  EXPECT_EQ(negative.status().code(), absl::StatusCode::kInvalidArgument);

  auto too_many = CreateActivationGenerator(
      *executor_, kGpt2TransformerBlockCount + 1, DataType::BF16, 123);
  EXPECT_EQ(too_many.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(Gpt2Test, ZeroBlocksTapsEmbeddingActivations) {
  auto generator =
      CreateActivationGenerator(*executor_, 0, DataType::BF16, 123);
  ASSERT_TRUE(generator.ok()) << generator.status();

  // Token and position embeddings are the only trainable tensors before the
  // first transformer block.
  EXPECT_EQ((*generator)->weights().size(), 2u);
  EXPECT_EQ((*generator)->gradients().size(), 2u);
  EXPECT_EQ((*generator)->output_type(), DataType::BF16);
}

TEST_F(Gpt2Test, OneBlockProducesResidualStreamActivations) {
  auto generator =
      CreateActivationGenerator(*executor_, 1, DataType::BF16, 123);
  ASSERT_TRUE(generator.ok()) << generator.status();

  // Each block contributes two LayerNorms and four dense projections, with a
  // weight and bias for each. Attention and GELU themselves are stateless.
  EXPECT_EQ((*generator)->weights().size(), 2u + 12u);
  EXPECT_EQ((*generator)->gradients().size(), 2u + 12u);

  auto tokens = Buffer::Allocate(
      *executor_, static_cast<size_t>(kGpt2ContextLength) * sizeof(int));
  ASSERT_TRUE(tokens.ok()) << tokens.status();
  ASSERT_EQ(cudaMemsetAsync(tokens->data(), 0, tokens->size_bytes(),
                            executor_->stream()),
            cudaSuccess);

  BufferVec inputs = {*tokens};
  auto activations = (*generator)->fwd(*executor_, inputs);

  ASSERT_TRUE(activations.ok()) << activations.status();
  EXPECT_EQ(activations->output.size_bytes(),
            static_cast<size_t>(kGpt2ContextLength) * kGpt2ModelWidth *
                sizeof(uint16_t));
  EXPECT_TRUE(executor_->Synchronize().ok());
}

}  // namespace
}  // namespace pluto::llm
