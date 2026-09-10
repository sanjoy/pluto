#include "src/dataset/dataset.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
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

  cuda::PageLockedHostArray<int> CopyToHost(const cuda::Buffer& buffer) {
    auto result = cuda::PageLockedHostArray<int>::Allocate(buffer.size_bytes() /
                                                           sizeof(int));
    EXPECT_TRUE(result.ok()) << result.status();
    if (!result.ok()) return {};
    EXPECT_EQ(
        cudaMemcpyAsync(result->data(), buffer.data(), buffer.size_bytes(),
                        cudaMemcpyDeviceToHost, executor_->stream()),
        cudaSuccess);
    EXPECT_TRUE(executor_->Synchronize().ok());
    return *result;
  }

  std::vector<int> Inputs(const DataBatch& batch) {
    const cuda::PageLockedHostArray<int> packed = CopyToHost(batch.data);
    EXPECT_EQ(packed.size(), 2 * static_cast<size_t>(batch.batch_size));
    return std::vector<int>(packed.begin(), packed.begin() + batch.batch_size);
  }

  std::vector<int> Targets(const DataBatch& batch) {
    const cuda::PageLockedHostArray<int> packed = CopyToHost(batch.data);
    EXPECT_EQ(packed.size(), 2 * static_cast<size_t>(batch.batch_size));
    if (packed.size() < static_cast<size_t>(batch.batch_size)) return {};
    return std::vector<int>(packed.begin() + batch.batch_size, packed.end());
  }

  std::unique_ptr<cuda::Executor> executor_;
};

cuda::PageLockedHostArray<int> MakePinnedInts(size_t size, int value = 0) {
  auto result = cuda::PageLockedHostArray<int>::Allocate(size);
  EXPECT_TRUE(result.ok()) << result.status();
  if (!result.ok()) return {};
  std::fill(result->begin(), result->end(), value);
  return *result;
}

std::filesystem::path TokenizerDirectory() {
  const char* directory = std::getenv("PLUTO_GPT2_TOKENIZER_DIR");
  EXPECT_NE(directory, nullptr)
      << "set PLUTO_GPT2_TOKENIZER_DIR to the saved GPT-2 tokenizer";
  return directory == nullptr ? std::filesystem::path() : directory;
}

TEST_F(DataSetTest, SequentialBatchesShiftTargetsAndReset) {
  auto corpus = MakePinnedInts(21);
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
  const std::vector<int> first_tokens = Inputs(*first);
  EXPECT_EQ(first_tokens, (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(Targets(*first), (std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8}));

  auto second = (*iterator)->Next();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(Inputs(*second), (std::vector<int>{8, 9, 10, 11, 12, 13, 14, 15}));

  ASSERT_TRUE((*iterator)->Reset().ok());
  auto reset = (*iterator)->Next();
  ASSERT_TRUE(reset.ok()) << reset.status();
  EXPECT_EQ(Inputs(*reset), first_tokens);
}

TEST_F(DataSetTest, CorpusIsUploadedDuringCreation) {
  auto corpus = MakePinnedInts(21);
  std::iota(corpus.begin(), corpus.end(), 0);
  auto iterator = InMemoryDataSetIterator::Create(
      *executor_, corpus,
      InMemoryDataSetOptions{
          .batch_size = 4,
          .context_length = 4,
          .order = InMemoryDataSetOrder::kSequential,
      });
  ASSERT_TRUE(iterator.ok()) << iterator.status();

  // Mutating the host input must not affect the device-resident corpus.
  std::fill(corpus.begin(), corpus.end(), -1);
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  EXPECT_EQ(Inputs(*batch), (std::vector<int>{0, 1, 2, 3}));
  EXPECT_EQ(Targets(*batch), (std::vector<int>{1, 2, 3, 4}));
}

TEST_F(DataSetTest, RandomOrderIsDeterministicAcrossReset) {
  auto corpus = MakePinnedInts(100);
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
  const std::vector<int> expected = Inputs(*first);

  ASSERT_TRUE((*iterator)->Next().ok());
  ASSERT_TRUE((*iterator)->Reset().ok());
  auto reset = (*iterator)->Next();
  ASSERT_TRUE(reset.ok()) << reset.status();
  EXPECT_EQ(Inputs(*reset), expected);
}

TEST_F(DataSetTest, SeededIteratorsAreIndependentAndReplayWholeBatches) {
  auto corpus = MakePinnedInts(193);
  std::iota(corpus.begin(), corpus.end(), 0);
  const InMemoryDataSetOptions options{
      .batch_size = 32, .context_length = 8,
      .order = InMemoryDataSetOrder::kRandom, .seed = 987654321};
  auto first = InMemoryDataSetIterator::Create(*executor_, corpus, options);
  auto second = InMemoryDataSetIterator::Create(*executor_, corpus, options);
  auto unrelated = InMemoryDataSetIterator::Create(*executor_, corpus, options);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_TRUE(unrelated.ok()) << unrelated.status();
  for (int repeat = 0; repeat < 16; ++repeat) {
    auto a = (*first)->Next();
    ASSERT_TRUE(a.ok()) << a.status();
    // Resetting/consuming another iterator must not perturb training's RNG.
    ASSERT_TRUE((*unrelated)->Reset().ok());
    ASSERT_TRUE((*unrelated)->Next().ok());
    ASSERT_TRUE((*unrelated)->Next().ok());
    auto b = (*second)->Next();
    ASSERT_TRUE(b.ok()) << b.status();
    EXPECT_EQ(Inputs(*a), Inputs(*b));
    EXPECT_EQ(Targets(*a), Targets(*b));
  }
}

TEST_F(DataSetTest, RejectsInvalidShapes) {
  const auto corpus = MakePinnedInts(10, 1);
  EXPECT_FALSE(InMemoryDataSetIterator::Create(
                   *executor_, corpus,
                   InMemoryDataSetOptions{.batch_size = 7, .context_length = 4})
                   .ok());
  EXPECT_FALSE(InMemoryDataSetIterator::Create(
                   *executor_, MakePinnedInts(4, 1),
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

TEST_F(DataSetTest, SplitsTextCorpusAtNextLineBoundary) {
  const std::filesystem::path path =
      std::filesystem::path(testing::TempDir()) / "split-corpus.txt";
  {
    std::ofstream output(path, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    output << "alpha\nbeta\ngamma\n";
  }

  auto corpus = LoadTextCorpus(path.string());
  ASSERT_TRUE(corpus.ok()) << corpus.status();
  auto split = SplitCorpus(*corpus, 0.5);
  ASSERT_TRUE(split.ok()) << split.status();
  EXPECT_EQ(split->training.text(), "alpha\nbeta\n");
  EXPECT_EQ(split->test.text(), "gamma\n");
}

TEST_F(DataSetTest, RejectsInvalidCorpusSplits) {
  const std::filesystem::path path =
      std::filesystem::path(testing::TempDir()) / "invalid-split-corpus.txt";
  {
    std::ofstream output(path, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    output << "alpha\nbeta\n";
  }

  auto corpus = LoadTextCorpus(path.string());
  ASSERT_TRUE(corpus.ok()) << corpus.status();
  EXPECT_FALSE(SplitCorpus(*corpus, 0.0).ok());
  EXPECT_FALSE(SplitCorpus(*corpus, 1.0).ok());
  EXPECT_FALSE(
      SplitCorpus(*corpus, std::numeric_limits<double>::quiet_NaN()).ok());

  const std::filesystem::path empty_path =
      std::filesystem::path(testing::TempDir()) / "empty-split-corpus.txt";
  std::ofstream(empty_path, std::ios::binary).close();
  auto empty = LoadTextCorpus(empty_path.string());
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_FALSE(SplitCorpus(*empty, 0.5).ok());
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
  EXPECT_EQ(Inputs(*batch),
            std::vector<int>(expected->begin(), expected->begin() + 4));
  EXPECT_EQ(Targets(*batch),
            std::vector<int>(expected->begin() + 1, expected->begin() + 5));
}

}  // namespace
}  // namespace pluto
