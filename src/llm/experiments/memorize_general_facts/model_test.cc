#include <memory>

#include "absl/container/flat_hash_set.h"
#include "gtest/gtest.h"
#include "src/cuda/executor.h"
#include "src/llm/recipes/gpt2.h"

namespace pluto::llm::memorize_general_facts {
namespace {

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
