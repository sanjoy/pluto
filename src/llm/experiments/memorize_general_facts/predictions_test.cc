#include "src/llm/experiments/memorize_general_facts/predictions.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <memory>

#include "absl/container/flat_hash_set.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/recipes/gpt2.h"

namespace pluto::llm::memorize_general_facts {
namespace {

TEST(PredictionsTest, ExactArgmaxMasksPromptsAndVocabularyPadding) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  auto host_logits = cuda::PageLockedHostArray<float>::Allocate(**executor, 32);
  auto host_targets = cuda::PageLockedHostArray<int>::Allocate(**executor, 4);
  auto host_result = cuda::PageLockedHostArray<int>::Allocate(**executor, 4);
  ASSERT_TRUE(host_logits.ok());
  ASSERT_TRUE(host_targets.ok());
  ASSERT_TRUE(host_result.ok());
  std::fill(host_logits->begin(), host_logits->end(), -3.0f);
  (*host_logits)[4] = 2;
  (*host_logits)[5] = 1000;  // This vocabulary padding must not win.
  (*host_logits)[9] = (*host_logits)[10] = -2;
  (*host_logits)[16] = std::numeric_limits<float>::quiet_NaN();
  (*host_logits)[24] = std::numeric_limits<float>::quiet_NaN();
  (*host_targets)[0] = 4;
  (*host_targets)[1] = 1;
  (*host_targets)[2] = -1;
  (*host_targets)[3] = 0;
  auto logits = cuda::Buffer::Allocate(**executor, host_logits->size_bytes());
  auto targets = cuda::Buffer::Allocate(**executor, host_targets->size_bytes());
  ASSERT_TRUE(logits.ok());
  ASSERT_TRUE(targets.ok());
  ASSERT_EQ(
      cudaMemcpyAsync(logits->data(), host_logits->data(), logits->size_bytes(),
                      cudaMemcpyHostToDevice, (*executor)->stream()),
      cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(targets->data(), host_targets->data(),
                            targets->size_bytes(), cudaMemcpyHostToDevice,
                            (*executor)->stream()),
            cudaSuccess);
  for (int repetition = 0; repetition < 2; ++repetition) {
    auto predictions = PredictMaskedTokens(**executor, *logits, *targets, 5);
    ASSERT_TRUE(predictions.ok()) << predictions.status();
    ASSERT_EQ(cudaMemcpyAsync(host_result->data(), predictions->data(),
                              predictions->size_bytes(), cudaMemcpyDeviceToHost,
                              (*executor)->stream()),
              cudaSuccess);
    ASSERT_TRUE((*executor)->Synchronize().ok());
    EXPECT_EQ((*host_result)[0], 4);
    EXPECT_EQ((*host_result)[1], 1);
    EXPECT_EQ((*host_result)[2], -1);
    EXPECT_EQ((*host_result)[3], -2);
  }
  EXPECT_FALSE(PredictMaskedTokens(**executor, *logits, *targets, 0).ok());
  EXPECT_FALSE(PredictMaskedTokens(**executor, *logits, *targets, 9).ok());
}

TEST(ExperimentModelTest, DepthChangesOnlyBlockCountAndKeepsTiedHead) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  for (int depth : {0, 1, 8, 16}) {
    SCOPED_TRACE(depth);
    auto model = CreateGpt2(**executor, DataType::BF16, 1337, depth);
    ASSERT_TRUE(model.ok()) << model.status();
    absl::flat_hash_set<void*> weights;
    for (const auto& weight : (*model)->weights())
      weights.insert(weight.data());
    // Token/position embeddings and final norm's gamma/beta, plus each
    // block's two norms and four affine projections (12 distinct tensors).
    EXPECT_EQ(weights.size(), static_cast<size_t>(4 + 12 * depth));
    EXPECT_EQ((*model)->weights().front().data(),
              (*model)->weights().back().data());
    EXPECT_EQ((*model)->input_types()[0].dimensions()[1], 1024);
    EXPECT_EQ((*model)->output_types()[0].dimensions()[2], 50272);
  }
  EXPECT_FALSE(CreateGpt2(**executor, DataType::BF16, 1337, -1).ok());
  EXPECT_TRUE((*executor)->Synchronize().ok());
}

}  // namespace
}  // namespace pluto::llm::memorize_general_facts
