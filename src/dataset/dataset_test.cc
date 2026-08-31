#include "src/dataset/dataset.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/dataset/tokenizer.h"

namespace pluto {
namespace {

class DataSetTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }

  void TearDown() override {
    if (executor_ == nullptr) return;
    EXPECT_TRUE(executor_->Synchronize().ok());
    executor_.reset();
  }

  std::vector<int> CopyToHost(const cuda::Buffer& buffer) {
    std::vector<int> result(buffer.size_bytes() / sizeof(int));
    EXPECT_EQ(cudaMemcpyAsync(result.data(), buffer.data(), buffer.size_bytes(),
                              cudaMemcpyDeviceToHost, executor_->stream()),
              cudaSuccess);
    EXPECT_TRUE(executor_->Synchronize().ok());
    return result;
  }

  std::unique_ptr<cuda::Executor> executor_;
};

std::filesystem::path TokenizerDirectory() {
  const char* directory = std::getenv("PLUTO_GPT2_TOKENIZER_DIR");
  EXPECT_NE(directory, nullptr)
      << "set PLUTO_GPT2_TOKENIZER_DIR to the saved GPT-2 tokenizer";
  return directory == nullptr ? std::filesystem::path() : directory;
}

TEST_F(DataSetTest, SequentialBatchesShiftTargetsAndReset) {
  std::vector<int> corpus(21);
  std::iota(corpus.begin(), corpus.end(), 0);
  auto iterator = InMemoryDataSetIterator::Create(
      *executor_, corpus,
      InMemoryDataSetOptions{
          .batch_size = 8,
          .context_length = 4,
          .order = InMemoryDataSetOrder::kSequential,
      });
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
      *executor_, corpus,
      InMemoryDataSetOptions{
          .batch_size = 8,
          .context_length = 4,
          .order = InMemoryDataSetOrder::kRandom,
          .seed = 123,
      });
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

TEST_F(DataSetTest, RejectsInvalidShapes) {
  const std::vector<int> corpus(10, 1);
  EXPECT_FALSE(InMemoryDataSetIterator::Create(
                   *executor_, corpus,
                   InMemoryDataSetOptions{.batch_size = 7, .context_length = 4})
                   .ok());
  EXPECT_FALSE(InMemoryDataSetIterator::Create(
                   *executor_, std::vector<int>{1, 2, 3, 4},
                   InMemoryDataSetOptions{.batch_size = 4, .context_length = 4})
                   .ok());
}

TEST_F(DataSetTest, LoadsMappedTextCorpusAndSharesStorageWithSubcorpus) {
  const std::filesystem::path path =
      std::filesystem::path(testing::TempDir()) / "mapped-corpus.txt";
  {
    std::ofstream output(path, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    output << "alpha\nbeta\ngamma\n";
  }

  auto corpus = LoadTextCorpus(path.string());
  ASSERT_TRUE(corpus.ok()) << corpus.status();
  EXPECT_EQ(corpus->text(), "alpha\nbeta\ngamma\n");
  EXPECT_EQ(corpus->size(), size_t{17});
  auto beta = corpus->SubCorpus(6, 4);
  ASSERT_TRUE(beta.ok()) << beta.status();
  EXPECT_EQ(beta->text(), "beta");

  // POSIX mappings survive unlink: this also verifies that SubCorpus shares
  // the mapping rather than referring back to an open file or copied string.
  ASSERT_TRUE(std::filesystem::remove(path));
  EXPECT_EQ(corpus->text(), "alpha\nbeta\ngamma\n");
  EXPECT_EQ(beta->text(), "beta");
  EXPECT_FALSE(corpus->SubCorpus(corpus->size() + 1).ok());
}

TEST_F(DataSetTest, LoadsEmptyTextCorpus) {
  const std::filesystem::path path =
      std::filesystem::path(testing::TempDir()) / "empty-corpus.txt";
  std::ofstream(path, std::ios::binary).close();
  auto corpus = LoadTextCorpus(path.string());
  ASSERT_TRUE(corpus.ok()) << corpus.status();
  EXPECT_TRUE(corpus->empty());
  EXPECT_TRUE(corpus->text().empty());
}

TEST_F(DataSetTest, TokenizesMappedCorpusIntoSequentialDataset) {
  const std::filesystem::path path =
      std::filesystem::path(testing::TempDir()) / "tokenized-corpus.txt";
  {
    std::ofstream output(path, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    output << "Hello, world! Hello, world!";
  }
  auto corpus = LoadTextCorpus(path.string());
  auto tokenizer = tokenizer::Gpt2Tokenizer::Load(TokenizerDirectory());
  ASSERT_TRUE(corpus.ok()) << corpus.status();
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
  auto expected = (*tokenizer)->Encode(corpus->text());
  ASSERT_TRUE(expected.ok()) << expected.status();
  ASSERT_GT(expected->size(), size_t{4});

  auto iterator = MakeInMemoryDataSetIterator(
      *executor_, *corpus, **tokenizer,
      InMemoryDataSetOptions{.batch_size = 4,
                             .context_length = 4,
                             .order = InMemoryDataSetOrder::kSequential});
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  EXPECT_EQ((*iterator)->token_count(), expected->size());
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  EXPECT_EQ(CopyToHost(batch->tokens),
            std::vector<int>(expected->begin(), expected->begin() + 4));
  EXPECT_EQ(CopyToHost(batch->targets),
            std::vector<int>(expected->begin() + 1, expected->begin() + 5));
}

}  // namespace
}  // namespace pluto
