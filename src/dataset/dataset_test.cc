#include "src/dataset/dataset.h"

#include <cuda_runtime.h>

#include <memory>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"

namespace pluto {
namespace {

class DataSetTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_storage_ = std::move(*executor);
    executor_ = executor_storage_.get();
  }

  void TearDown() override {
    if (executor_ == nullptr) return;
    EXPECT_TRUE(executor_->Synchronize().ok());
    executor_ = nullptr;
    executor_storage_.reset();
  }

  std::vector<int> CopyToHost(const cuda::Buffer& buffer) {
    std::vector<int> result(buffer.size_bytes() / sizeof(int));
    EXPECT_EQ(cudaMemcpyAsync(result.data(), buffer.data(), buffer.size_bytes(),
                              cudaMemcpyDeviceToHost, executor_->stream()),
              cudaSuccess);
    EXPECT_TRUE(executor_->Synchronize().ok());
    return result;
  }

  std::unique_ptr<cuda::Executor> executor_storage_;
  cuda::Executor* executor_ = nullptr;
};

TEST_F(DataSetTest, SequentialBatchesShiftTargetsAndReset) {
  std::vector<int> corpus(21);
  std::iota(corpus.begin(), corpus.end(), 0);
  auto iterator = InMemoryDataSetIterator::Create(
      corpus,
      InMemoryDataSetOptions{
          .batch_size = 8,
          .context_length = 4,
          .order = InMemoryDataSetOrder::kSequential,
      },
      executor_);
  ASSERT_TRUE(iterator.ok()) << iterator.status();

  auto first = (*iterator)->Next();
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_EQ(first->batch_size, 8);
  const std::vector<int> first_tokens = CopyToHost(first->tokens);
  EXPECT_EQ(first_tokens, (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(CopyToHost(first->targets),
            (std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8}));

  auto second = (*iterator)->Next();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(CopyToHost(second->tokens),
            (std::vector<int>{8, 9, 10, 11, 12, 13, 14, 15}));

  ASSERT_TRUE((*iterator)->Reset().ok());
  auto reset = (*iterator)->Next();
  ASSERT_TRUE(reset.ok()) << reset.status();
  EXPECT_EQ(CopyToHost(reset->tokens), first_tokens);
}

TEST_F(DataSetTest, RandomOrderIsDeterministicAcrossReset) {
  std::vector<int> corpus(100);
  std::iota(corpus.begin(), corpus.end(), 0);
  auto iterator = InMemoryDataSetIterator::Create(
      corpus,
      InMemoryDataSetOptions{
          .batch_size = 8,
          .context_length = 4,
          .order = InMemoryDataSetOrder::kRandom,
          .seed = 123,
      },
      executor_);
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  auto first = (*iterator)->Next();
  ASSERT_TRUE(first.ok()) << first.status();
  const std::vector<int> expected = CopyToHost(first->tokens);

  ASSERT_TRUE((*iterator)->Next().ok());
  ASSERT_TRUE((*iterator)->Reset().ok());
  auto reset = (*iterator)->Next();
  ASSERT_TRUE(reset.ok()) << reset.status();
  EXPECT_EQ(CopyToHost(reset->tokens), expected);
}

TEST_F(DataSetTest, RejectsInvalidShapesAndNullExecutor) {
  const std::vector<int> corpus(10, 1);
  EXPECT_FALSE(InMemoryDataSetIterator::Create(
                   corpus,
                   InMemoryDataSetOptions{.batch_size = 7, .context_length = 4},
                   executor_)
                   .ok());
  EXPECT_FALSE(InMemoryDataSetIterator::Create(
                   std::vector<int>{1, 2, 3, 4},
                   InMemoryDataSetOptions{.batch_size = 4, .context_length = 4},
                   executor_)
                   .ok());
  EXPECT_FALSE(InMemoryDataSetIterator::Create(
                   corpus,
                   InMemoryDataSetOptions{.batch_size = 4, .context_length = 2},
                   nullptr)
                   .ok());
}

}  // namespace
}  // namespace pluto
