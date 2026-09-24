#include "src/llm/token_order.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"

namespace pluto::llm {
namespace {

TEST(TokenOrderTest, ValidatesOnlyLogicalVocabularyPermutation) {
  EXPECT_TRUE(ValidateTokenOrder(3, {}).ok());
  EXPECT_TRUE(ValidateTokenOrder(3, {0, 1, 2}).ok());
  EXPECT_TRUE(ValidateTokenOrder(3, {1, 2, 0}).ok());
  EXPECT_FALSE(ValidateTokenOrder(0, {}).ok());
  EXPECT_FALSE(ValidateTokenOrder(-1, {}).ok());
  EXPECT_FALSE(ValidateTokenOrder(3, {0, 1}).ok());
  EXPECT_FALSE(ValidateTokenOrder(3, {0, 1, 2, 3}).ok());
  EXPECT_FALSE(ValidateTokenOrder(3, {0, 1, 1}).ok());
  EXPECT_FALSE(ValidateTokenOrder(3, {-1, 1, 2}).ok());
  EXPECT_FALSE(ValidateTokenOrder(3, {0, 1, 3}).ok());
}

TEST(TokenOrderTest, EmptyAndExplicitIdentityAvoidDeviceAllocation) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  auto empty = CopyTokenOrderToDevice(**executor, 131, {});
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_FALSE(empty->has_value());
  std::vector<int32_t> identity(131);
  std::iota(identity.begin(), identity.end(), 0);
  auto same = CopyTokenOrderToDevice(**executor, 131, identity);
  ASSERT_TRUE(same.ok()) << same.status();
  EXPECT_FALSE(same->has_value());
}

TEST(TokenOrderTest, OwnsHostInputAndCopiesExactlyLogicalVocabulary) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  std::vector<int32_t> order{1, 2, 0};
  auto uploaded = CopyTokenOrderToDevice(**executor, 3, order);
  ASSERT_TRUE(uploaded.ok()) << uploaded.status();
  ASSERT_TRUE(uploaded->has_value());
  EXPECT_EQ((*uploaded)->size_bytes(), 3 * sizeof(int32_t));
  order.assign(3, -1);
  auto copied = cuda::PageLockedHostArray<int32_t>::Allocate(**executor, 3);
  ASSERT_TRUE(copied.ok()) << copied.status();
  ASSERT_EQ(
      cudaMemcpyAsync(copied->data(), (*uploaded)->data(), copied->size_bytes(),
                      cudaMemcpyDeviceToHost, (*executor)->stream()),
      cudaSuccess);
  ASSERT_TRUE((*executor)->Synchronize().ok());
  EXPECT_EQ(std::vector<int32_t>(copied->begin(), copied->end()),
            (std::vector<int32_t>{1, 2, 0}));
}

}  // namespace
}  // namespace pluto::llm
