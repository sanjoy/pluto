#include "src/dataset/dataset.h"

#include <cuda_runtime.h>

#include <memory>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/buffer.h"

namespace pluto {
namespace {

class DataSetTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
              cudaSuccess);
  }

  void TearDown() override {
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream_), cudaSuccess);
  }

  std::vector<int> CopyToHost(const gpu::Buffer& buffer) {
    std::vector<int> result(buffer.size_bytes() / sizeof(int));
    EXPECT_EQ(cudaMemcpyAsync(result.data(), buffer.data(),
                              buffer.size_bytes(), cudaMemcpyDeviceToHost,
                              stream_),
              cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    return result;
  }

  cudaStream_t stream_ = nullptr;
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
      stream_);
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
      stream_);
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

TEST_F(DataSetTest, RejectsInvalidShapesAndDefaultStream) {
  const std::vector<int> corpus(10, 1);
  EXPECT_FALSE(InMemoryDataSetIterator::Create(
                   corpus,
                   InMemoryDataSetOptions{.batch_size = 7,
                                          .context_length = 4},
                   stream_)
                   .ok());
  EXPECT_FALSE(InMemoryDataSetIterator::Create(
                   std::vector<int>{1, 2, 3, 4},
                   InMemoryDataSetOptions{.batch_size = 4,
                                          .context_length = 4},
                   stream_)
                   .ok());
  EXPECT_FALSE(InMemoryDataSetIterator::Create(
                   corpus,
                   InMemoryDataSetOptions{.batch_size = 4,
                                          .context_length = 2},
                   nullptr)
                   .ok());
}

}  // namespace
}  // namespace pluto
