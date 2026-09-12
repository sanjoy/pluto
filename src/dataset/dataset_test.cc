#include "src/dataset/dataset.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/plain_text_tokenizer.h"
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
    if (executor_ == nullptr)
      return;
    EXPECT_TRUE(executor_->Synchronize().ok());
    executor_.reset();
  }

  cuda::PageLockedHostArray<int> CopyToHost(const cuda::Buffer& buffer) {
    auto result = cuda::PageLockedHostArray<int>::Allocate(
        *executor_, buffer.size_bytes() / sizeof(int));
    EXPECT_TRUE(result.ok()) << result.status();
    if (!result.ok())
      return {};
    EXPECT_EQ(
        cudaMemcpyAsync(result->data(), buffer.data(), buffer.size_bytes(),
                        cudaMemcpyDeviceToHost, executor_->stream()),
        cudaSuccess);
    EXPECT_TRUE(executor_->Synchronize().ok());
    return *result;
  }

  std::vector<int> Inputs(const DataBatch& batch) {
    const cuda::PageLockedHostArray<int> inputs = CopyToHost(batch.inputs);
    const auto count = batch.token_count();
    EXPECT_TRUE(count.ok()) << count.status();
    if (!count.ok())
      return {};
    EXPECT_EQ(inputs.size(), static_cast<size_t>(*count));
    return std::vector<int>(inputs.begin(), inputs.end());
  }

  std::vector<int> Targets(const DataBatch& batch) {
    const cuda::PageLockedHostArray<int> targets = CopyToHost(batch.targets);
    const auto count = batch.token_count();
    EXPECT_TRUE(count.ok()) << count.status();
    if (!count.ok())
      return {};
    EXPECT_EQ(targets.size(), static_cast<size_t>(*count));
    return std::vector<int>(targets.begin(), targets.end());
  }

  std::unique_ptr<cuda::Executor> executor_;
};

cuda::PageLockedHostArray<int> MakePinnedInts(cuda::Executor& executor,
                                              size_t size, int value = 0) {
  auto result = cuda::PageLockedHostArray<int>::Allocate(executor, size);
  EXPECT_TRUE(result.ok()) << result.status();
  if (!result.ok())
    return {};
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
  auto corpus = MakePinnedInts(*executor_, 21);
  std::iota(corpus.begin(), corpus.end(), 0);
  auto iterator = InMemoryDataSetIterator::Create(
      *executor_, corpus,
      InMemoryDataSetOptions{
          .batch_size = 2,
          .context_length = 4,
          .order = InMemoryDataSetOrder::kSequential,
      });
  ASSERT_TRUE(iterator.ok()) << iterator.status();

  auto first = (*iterator)->Next();
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_EQ(first->batch_size, 2);
  EXPECT_EQ(first->sequence_length, 4);
  EXPECT_NE(first->inputs.data(), first->targets.data());
  EXPECT_EQ(first->inputs.size_bytes(), 8 * sizeof(int));
  EXPECT_EQ(first->targets.size_bytes(), 8 * sizeof(int));
  const std::vector<int> first_tokens = Inputs(*first);
  EXPECT_EQ(first_tokens, (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(Targets(*first), (std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8}));

  auto second = (*iterator)->Next();
  ASSERT_TRUE(second.ok()) << second.status();
  // Retaining the first handle does not request a snapshot or a new allocation.
  EXPECT_EQ(second->inputs.data(), first->inputs.data());
  EXPECT_EQ(second->targets.data(), first->targets.data());
  EXPECT_EQ(Targets(*first), (std::vector<int>{9, 10, 11, 12, 13, 14, 15, 16}));
  EXPECT_EQ(Inputs(*first), (std::vector<int>{8, 9, 10, 11, 12, 13, 14, 15}));
  EXPECT_EQ(Inputs(*second), (std::vector<int>{8, 9, 10, 11, 12, 13, 14, 15}));

  ASSERT_TRUE((*iterator)->Reset().ok());
  auto reset = (*iterator)->Next();
  ASSERT_TRUE(reset.ok()) << reset.status();
  EXPECT_EQ(reset->inputs.data(), first->inputs.data());
  EXPECT_EQ(reset->targets.data(), first->targets.data());
  EXPECT_EQ(Targets(*reset), (std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_EQ(Inputs(*reset), first_tokens);
}

TEST_F(DataSetTest, BatchBuffersRemainAliveAfterIteratorDestruction) {
  auto corpus = MakePinnedInts(*executor_, 9);
  std::iota(corpus.begin(), corpus.end(), 0);
  auto iterator = InMemoryDataSetIterator::Create(
      *executor_, corpus,
      {.batch_size = 2,
       .context_length = 4,
       .order = InMemoryDataSetOrder::kSequential});
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();

  iterator->reset();
  EXPECT_EQ(Inputs(*batch), (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(Targets(*batch), (std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8}));
}

TEST_F(DataSetTest, CorpusIsUploadedDuringCreation) {
  auto corpus = MakePinnedInts(*executor_, 21);
  std::iota(corpus.begin(), corpus.end(), 0);
  auto iterator = InMemoryDataSetIterator::Create(
      *executor_, corpus,
      InMemoryDataSetOptions{
          .batch_size = 1,
          .context_length = 4,
          .order = InMemoryDataSetOrder::kSequential,
      });
  ASSERT_TRUE(iterator.ok()) << iterator.status();

  // Create() only enqueues the upload. A retained host alias must wait before
  // mutation; once uploaded, changing it cannot affect the device corpus.
  ASSERT_TRUE(executor_->Synchronize().ok());
  std::fill(corpus.begin(), corpus.end(), -1);
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  EXPECT_EQ(Inputs(*batch), (std::vector<int>{0, 1, 2, 3}));
  EXPECT_EQ(Targets(*batch), (std::vector<int>{1, 2, 3, 4}));
}

// A bounded gate exposes accidental compute-stream waits without hanging a
// failed test indefinitely. Its callback uses only CPU operations, never CUDA.
class ComputeGate {
 public:
  explicit ComputeGate(cuda::Executor& executor) : executor_(executor) {}
  ~ComputeGate() {
    open.store(true);
    EXPECT_TRUE(executor_.Synchronize().ok());
  }
  static void CUDART_CB Wait(void* argument) {
    auto& gate = *static_cast<ComputeGate*>(argument);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!gate.open.load()) {
      if (std::chrono::steady_clock::now() >= deadline) {
        gate.timed_out.store(true);
        break;
      }
      std::this_thread::yield();
    }
  }
  std::atomic<bool> open{false};
  std::atomic<bool> timed_out{false};

 private:
  cuda::Executor& executor_;
};

TEST_F(DataSetTest, CreateDoesNotWaitForComputeAndKeepsUploadAlive) {
  auto corpus = MakePinnedInts(*executor_, 25);
  std::iota(corpus.begin(), corpus.end(), 0);
  // Reserve device-pool capacity before the gate so device allocation itself
  // cannot obscure whether Create waits for its upload to finish.
  {
    auto warm = cuda::Buffer::Allocate(*executor_, 4096);
    ASSERT_TRUE(warm.ok()) << warm.status();
  }
  ASSERT_TRUE(executor_->Synchronize().ok());
  ComputeGate gate(*executor_);
  ASSERT_EQ(cudaLaunchHostFunc(executor_->stream(), ComputeGate::Wait, &gate),
            cudaSuccess);
  auto iterator = InMemoryDataSetIterator::Create(
      *executor_, std::move(corpus),
      {.batch_size = 3,
       .context_length = 4,
       .order = InMemoryDataSetOrder::kSequential});
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  EXPECT_FALSE(gate.timed_out.load());
  // The last corpus host owner has gone away. A new CPU allocation must not
  // overwrite those bytes while their queued upload is still behind the gate.
  auto overwrite = MakePinnedInts(*executor_, 25, -123);
  EXPECT_FALSE(gate.timed_out.load());
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  gate.open.store(true);
  EXPECT_EQ(Inputs(*batch),
            (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
  EXPECT_EQ(Targets(*batch),
            (std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
  EXPECT_FALSE(gate.timed_out.load());
}

TEST_F(DataSetTest, RejectsHostCorpusFromAnotherExecutor) {
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  auto corpus = MakePinnedInts(**other, 25);
  auto iterator = InMemoryDataSetIterator::Create(
      *executor_, std::move(corpus), {.batch_size = 3, .context_length = 4});
  ASSERT_FALSE(iterator.ok());
  EXPECT_EQ(iterator.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(DataSetTest, RandomOrderIsDeterministicAcrossReset) {
  auto corpus = MakePinnedInts(*executor_, 100);
  std::iota(corpus.begin(), corpus.end(), 0);
  auto iterator = InMemoryDataSetIterator::Create(
      *executor_, corpus,
      InMemoryDataSetOptions{
          .batch_size = 2,
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
  auto corpus = MakePinnedInts(*executor_, 193);
  std::iota(corpus.begin(), corpus.end(), 0);
  const InMemoryDataSetOptions options{.batch_size = 4,
                                       .context_length = 8,
                                       .order = InMemoryDataSetOrder::kRandom,
                                       .seed = 987654321};
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

TEST_F(DataSetTest, BatchSizeCountsIndependentSequencesNotTokens) {
  auto corpus = MakePinnedInts(*executor_, 25);
  std::iota(corpus.begin(), corpus.end(), 0);
  // Three samples need not be divisible by their four-token context length.
  auto iterator = InMemoryDataSetIterator::Create(
      *executor_, corpus,
      InMemoryDataSetOptions{.batch_size = 3,
                             .context_length = 4,
                             .order = InMemoryDataSetOrder::kSequential});
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  EXPECT_EQ(batch->batch_size, 3);
  EXPECT_EQ(batch->sequence_length, 4);
  auto count = batch->token_count();
  ASSERT_TRUE(count.ok()) << count.status();
  EXPECT_EQ(*count, 12);
  EXPECT_EQ(batch->inputs.size_bytes(), 12 * sizeof(int));
  EXPECT_EQ(batch->targets.size_bytes(), 12 * sizeof(int));
  EXPECT_EQ(Inputs(*batch),
            (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
  EXPECT_EQ(Targets(*batch),
            (std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
}

TEST_F(DataSetTest, DataBatchChecksDimensionsBeforeMultiplyingThem) {
  auto storage = cuda::Buffer::Allocate(*executor_, 0);
  ASSERT_TRUE(storage.ok()) << storage.status();
  const DataBatch valid{.inputs = *storage,
                        .targets = *storage,
                        .batch_size = 3,
                        .sequence_length = 4};
  auto count = valid.token_count();
  ASSERT_TRUE(count.ok()) << count.status();
  EXPECT_EQ(*count, 12);
  // One-token samples are the default for non-sequence datasets.
  const DataBatch one_token_samples{
      .inputs = *storage, .targets = *storage, .batch_size = 3};
  ASSERT_TRUE(one_token_samples.token_count().ok());
  EXPECT_EQ(*one_token_samples.token_count(), 3);

  for (const auto [samples, length] :
       {std::pair{0, 4}, std::pair{-1, 4}, std::pair{3, 0}, std::pair{3, -1},
        std::pair{std::numeric_limits<int>::max(), 2},
        std::pair{2, std::numeric_limits<int>::max()}}) {
    SCOPED_TRACE(testing::Message() << samples << " x " << length);
    const DataBatch invalid{.inputs = *storage,
                            .targets = *storage,
                            .batch_size = samples,
                            .sequence_length = length};
    EXPECT_FALSE(invalid.token_count().ok());
  }
  const DataBatch largest{.inputs = *storage,
                          .targets = *storage,
                          .batch_size = std::numeric_limits<int>::max(),
                          .sequence_length = 1};
  ASSERT_TRUE(largest.token_count().ok());
  EXPECT_EQ(*largest.token_count(), std::numeric_limits<int>::max());
}

TEST_F(DataSetTest, RejectsInvalidShapes) {
  const auto corpus = MakePinnedInts(*executor_, 10, 1);
  for (const auto [samples, length] :
       {std::pair{0, 4}, std::pair{-1, 4}, std::pair{1, 0}, std::pair{1, -1},
        std::pair{std::numeric_limits<int>::max(), 2}}) {
    SCOPED_TRACE(testing::Message() << samples << " x " << length);
    EXPECT_FALSE(InMemoryDataSetIterator::Create(
                     *executor_, corpus,
                     InMemoryDataSetOptions{.batch_size = samples,
                                            .context_length = length})
                     .ok());
  }
  EXPECT_FALSE(InMemoryDataSetIterator::Create(
                   *executor_, MakePinnedInts(*executor_, 4, 1),
                   InMemoryDataSetOptions{.batch_size = 1, .context_length = 4})
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
  auto expected = (*tokenizer)->Encode(*executor_, corpus->text());
  ASSERT_TRUE(expected.ok()) << expected.status();
  ASSERT_GT(expected->size(), size_t{4});

  auto iterator = MakeInMemoryDataSetIterator(
      *executor_, *corpus, **tokenizer,
      InMemoryDataSetOptions{.batch_size = 1,
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

TEST_F(DataSetTest, PlainTextBaseInterfaceCreatesShiftedGpuBatches) {
  // Include NUL and a non-UTF-8 byte: this path must dispatch to the supplied
  // byte tokenizer, not assume that every corpus uses GPT-2 tokenization.
  const std::string text(
      "ab\0cd\xff"
      "efghi",
      11);
  const auto path =
      std::filesystem::path(testing::TempDir()) / "byte-corpus.txt";
  {
    std::ofstream output(path, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    output.write(text.data(), text.size());
  }
  auto corpus = LoadTextCorpus(path.string());
  ASSERT_TRUE(corpus.ok()) << corpus.status();
  const tokenizer::PlainTextTokenizer byte_tokenizer;
  const tokenizer::Tokenizer& encoder = byte_tokenizer;
  auto iterator = MakeInMemoryDataSetIterator(
      *executor_, *corpus, encoder,
      InMemoryDataSetOptions{.batch_size = 2,
                             .context_length = 4,
                             .order = InMemoryDataSetOrder::kSequential});
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  EXPECT_EQ((*iterator)->token_count(), text.size());
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  EXPECT_EQ(batch->batch_size, 2);
  EXPECT_EQ(batch->sequence_length, 4);
  EXPECT_EQ(Inputs(*batch),
            (std::vector<int>{97, 98, 0, 99, 100, 255, 101, 102}));
  EXPECT_EQ(Targets(*batch),
            (std::vector<int>{98, 0, 99, 100, 255, 101, 102, 103}));
}

class FailingTokenizer final : public tokenizer::Tokenizer {
 public:
  absl::StatusOr<cuda::PageLockedHostArray<int>> Encode(
      cuda::Executor&, absl::string_view) const override {
    return absl::DataLossError("deliberate tokenizer failure");
  }
  int vocab_size() const override { return 256; }
};

TEST_F(DataSetTest, PropagatesAbstractTokenizerFailure) {
  const FailingTokenizer encoder;
  auto iterator = MakeInMemoryDataSetIterator(
      *executor_, TextCorpus{}, encoder,
      InMemoryDataSetOptions{.batch_size = 2, .context_length = 4});
  ASSERT_FALSE(iterator.ok());
  EXPECT_EQ(iterator.status().code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(iterator.status().message(), "deliberate tokenizer failure");
}

}  // namespace
}  // namespace pluto
