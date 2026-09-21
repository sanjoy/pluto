#include "src/dataset/padded_line_dataset.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/plain_text_tokenizer.h"

namespace pluto {
namespace {

class PaddedLineDataSetTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }

  void TearDown() override {
    if (executor_ != nullptr)
      EXPECT_TRUE(executor_->Synchronize().ok());
  }

  std::vector<int> Download(const cuda::Buffer& buffer) {
    auto host = cuda::PageLockedHostArray<int>::Allocate(
        *executor_, buffer.size_bytes() / sizeof(int));
    EXPECT_TRUE(host.ok()) << host.status();
    if (!host.ok())
      return {};
    EXPECT_EQ(cudaMemcpyAsync(host->data(), buffer.data(), buffer.size_bytes(),
                              cudaMemcpyDeviceToHost, executor_->stream()),
              cudaSuccess);
    EXPECT_TRUE(executor_->Synchronize().ok());
    return std::vector<int>(host->begin(), host->end());
  }

  std::unique_ptr<cuda::Executor> executor_;
  tokenizer::PlainTextTokenizer tokenizer_;
};

TEST_F(PaddedLineDataSetTest, MasksPromptAndPaddingButScoresSuffixAndEos) {
  auto iterator =
      PaddedLineDataSetIterator::Create(*executor_, "abcde\n", tokenizer_,
                                        {.batch_size = 1,
                                         .context_length = 8,
                                         .prompt_tokens = 2,
                                         .eos_token = 255});
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  EXPECT_EQ((*iterator)->sample_count(), 1);
  EXPECT_EQ((*iterator)->batches_per_epoch(), 1);
  EXPECT_EQ((*iterator)->supervised_row_count(), 4);
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  EXPECT_EQ(batch->batch_size, 1);
  EXPECT_EQ(batch->sequence_length, 8);
  EXPECT_EQ(batch->supervised_row_count, 4);
  EXPECT_EQ(Download(batch->inputs),
            (std::vector<int>{'a', 'b', 'c', 'd', 'e', 255, 255, 255}));
  EXPECT_EQ(Download(batch->targets),
            (std::vector<int>{-1, 'c', 'd', 'e', 255, -1, -1, -1}));
  const auto original = (*iterator)->sample_tokens(0);
  EXPECT_EQ(std::vector<int>(original.begin(), original.end()),
            (std::vector<int>{'a', 'b', 'c', 'd', 'e'}));
}

TEST_F(PaddedLineDataSetTest, FinalPartialBatchNeverDuplicatesOrCrossesLines) {
  auto iterator = PaddedLineDataSetIterator::Create(
      *executor_, "abc\nDEFG\nhijkl", tokenizer_,
      {.batch_size = 2,
       .context_length = 6,
       .prompt_tokens = 2,
       .eos_token = 255});
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  EXPECT_EQ((*iterator)->sample_count(), 3);
  EXPECT_EQ((*iterator)->batches_per_epoch(), 2);
  EXPECT_EQ((*iterator)->supervised_row_count(), 9);
  auto first = (*iterator)->Next();
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_EQ(first->batch_size, 2);
  EXPECT_EQ(first->supervised_row_count, 5);
  EXPECT_EQ(Download(first->inputs),
            (std::vector<int>{'a', 'b', 'c', 255, 255, 255, 'D', 'E', 'F', 'G',
                              255, 255}));
  EXPECT_EQ(
      Download(first->targets),
      (std::vector<int>{-1, 'c', 255, -1, -1, -1, -1, 'F', 'G', 255, -1, -1}));
  auto last = (*iterator)->Next();
  ASSERT_TRUE(last.ok()) << last.status();
  EXPECT_EQ(last->batch_size, 1);
  EXPECT_EQ(last->supervised_row_count, 4);
  EXPECT_EQ(last->inputs.size_bytes(), 6 * sizeof(int));
  EXPECT_EQ(Download(last->targets),
            (std::vector<int>{-1, 'j', 'k', 'l', 255, -1}));
  auto next_epoch = (*iterator)->Next();
  ASSERT_TRUE(next_epoch.ok()) << next_epoch.status();
  EXPECT_EQ(next_epoch->inputs.data(), first->inputs.data());
  EXPECT_EQ(next_epoch->targets.data(), first->targets.data());
  auto next_tail = (*iterator)->Next();
  ASSERT_TRUE(next_tail.ok()) << next_tail.status();
  EXPECT_EQ(next_tail->inputs.data(), last->inputs.data());
  EXPECT_EQ(next_tail->targets.data(), last->targets.data());
}

TEST_F(PaddedLineDataSetTest, UsesCompactVocabularyForInputsTargetsAndEos) {
  auto mapping = tokenizer::BuildCompactVocabularyMapping(
      *executor_, tokenizer_, "abc\naxc", 255);
  ASSERT_TRUE(mapping.ok()) << mapping.status();
  auto compact = tokenizer::CompactVocabularyTokenizer::Create(
      tokenizer_, std::move(*mapping));
  ASSERT_TRUE(compact.ok()) << compact.status();
  auto iterator = PaddedLineDataSetIterator::Create(
      *executor_, "abc\naxc", **compact,
      {.batch_size = 2,
       .context_length = 4,
       .prompt_tokens = 2,
       .eos_token = (*compact)->eos_token_id()});
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  EXPECT_EQ(batch->batch_size, 2);
  EXPECT_EQ(batch->supervised_row_count, 4);
  // Sorted active original IDs are a, b, c, x, EOS; compact EOS is 4.
  EXPECT_EQ(Download(batch->inputs),
            (std::vector<int>{0, 1, 2, 4, 0, 3, 2, 4}));
  EXPECT_EQ(Download(batch->targets),
            (std::vector<int>{-1, 2, 4, -1, -1, 2, 4, -1}));
}

TEST_F(PaddedLineDataSetTest, BatchLargerThanCorpusIsOnePartialBatch) {
  auto iterator =
      PaddedLineDataSetIterator::Create(*executor_, "abc\ndef", tokenizer_,
                                        {.batch_size = 10,
                                         .context_length = 4,
                                         .prompt_tokens = 2,
                                         .eos_token = 255});
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  EXPECT_EQ((*iterator)->batches_per_epoch(), 1);
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  EXPECT_EQ(batch->batch_size, 2);
  EXPECT_EQ(batch->inputs.size_bytes(), 8 * sizeof(int));
  EXPECT_EQ(batch->supervised_row_count, 4);
}

TEST_F(PaddedLineDataSetTest, ExactlyContextLengthStillPredictsEos) {
  auto iterator =
      PaddedLineDataSetIterator::Create(*executor_, "abcd", tokenizer_,
                                        {.batch_size = 1,
                                         .context_length = 4,
                                         .prompt_tokens = 2,
                                         .eos_token = 255});
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  EXPECT_EQ(Download(batch->inputs), (std::vector<int>{'a', 'b', 'c', 'd'}));
  EXPECT_EQ(Download(batch->targets), (std::vector<int>{-1, 'c', 'd', 255}));
}

TEST_F(PaddedLineDataSetTest, PromptOnlySampleHasExactlyOneEosTarget) {
  auto iterator =
      PaddedLineDataSetIterator::Create(*executor_, "ab", tokenizer_,
                                        {.batch_size = 1,
                                         .context_length = 4,
                                         .prompt_tokens = 2,
                                         .eos_token = 255});
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  EXPECT_EQ(batch->supervised_row_count, 1);
  EXPECT_EQ(Download(batch->targets), (std::vector<int>{-1, 255, -1, -1}));
}

TEST_F(PaddedLineDataSetTest, CrLfIsALineEndingNotAToken) {
  auto iterator = PaddedLineDataSetIterator::Create(
      *executor_, "abc\r\ndef\r\n", tokenizer_,
      {.batch_size = 2,
       .context_length = 4,
       .prompt_tokens = 2,
       .eos_token = 255});
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  EXPECT_EQ((*iterator)->sample_tokens(0).size(), 3);
  EXPECT_EQ((*iterator)->sample_tokens(1).size(), 3);
  EXPECT_EQ((*iterator)->supervised_row_count(), 4);
}

TEST_F(PaddedLineDataSetTest, ShuffleVisitsEverySampleAndResetReplaysEpochs) {
  const std::string corpus = "aab\nbbc\nccd\ndde\neef\nffg\nggh\nhhi\niij";
  auto iterator =
      PaddedLineDataSetIterator::Create(*executor_, corpus, tokenizer_,
                                        {.batch_size = 2,
                                         .context_length = 4,
                                         .prompt_tokens = 2,
                                         .eos_token = 255,
                                         .shuffle = true,
                                         .seed = 19});
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  const auto read_epoch = [&]() {
    std::vector<int> first_tokens;
    for (size_t batch = 0; batch < (*iterator)->batches_per_epoch(); ++batch) {
      auto data = (*iterator)->Next();
      EXPECT_TRUE(data.ok()) << data.status();
      if (!data.ok())
        return std::vector<int>{};
      const auto inputs = Download(data->inputs);
      for (int sample = 0; sample < data->batch_size; ++sample)
        first_tokens.push_back(inputs[sample * data->sequence_length]);
    }
    return first_tokens;
  };
  const auto first = read_epoch();
  const auto second = read_epoch();
  const std::vector<int> ordered{'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'};
  for (auto epoch : {first, second}) {
    std::sort(epoch.begin(), epoch.end());
    EXPECT_EQ(epoch, ordered);
  }
  EXPECT_NE(first, ordered);
  EXPECT_NE(first, second);
  ASSERT_TRUE((*iterator)->Reset().ok());
  EXPECT_EQ(read_epoch(), first);
  EXPECT_EQ(read_epoch(), second);
}

TEST_F(PaddedLineDataSetTest, ReturnedBuffersSurviveIteratorDestruction) {
  auto iterator =
      PaddedLineDataSetIterator::Create(*executor_, "abc", tokenizer_,
                                        {.batch_size = 1,
                                         .context_length = 4,
                                         .prompt_tokens = 2,
                                         .eos_token = 255});
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  iterator->reset();
  EXPECT_EQ(Download(batch->inputs), (std::vector<int>{'a', 'b', 'c', 255}));
  EXPECT_EQ(Download(batch->targets), (std::vector<int>{-1, 'c', 255, -1}));
}

TEST_F(PaddedLineDataSetTest,
       RejectsInvalidOptionsAndInputInsteadOfTruncating) {
  const PaddedLineDataSetOptions valid{.batch_size = 1,
                                       .context_length = 4,
                                       .prompt_tokens = 2,
                                       .eos_token = 255};
  for (const std::string corpus :
       {"", "\n", "abc\n\n", "abc\n \n", "a", "abcde"}) {
    auto iterator = PaddedLineDataSetIterator::Create(*executor_, corpus,
                                                      tokenizer_, valid);
    EXPECT_FALSE(iterator.ok()) << corpus;
  }
  for (int field = 0; field < 7; ++field) {
    auto options = valid;
    switch (field) {
      case 0:
        options.batch_size = 0;
        break;
      case 1:
        options.context_length = 0;
        break;
      case 2:
        options.prompt_tokens = 0;
        break;
      case 3:
        options.prompt_tokens = 5;
        break;
      case 4:
        options.eos_token = -1;
        break;
      case 5:
        options.eos_token = 256;
        break;
      case 6:
        options.batch_size = 1 << 30;
        break;
    }
    auto iterator = PaddedLineDataSetIterator::Create(*executor_, "abc",
                                                      tokenizer_, options);
    EXPECT_FALSE(iterator.ok()) << field;
  }
}

TEST_F(PaddedLineDataSetTest, RealGpt2FactsHaveExactApprovedTargetCount) {
  const char* tokenizer_dir = std::getenv("PLUTO_GPT2_TOKENIZER_DIR");
  if (tokenizer_dir == nullptr)
    GTEST_SKIP() << "set PLUTO_GPT2_TOKENIZER_DIR for real-corpus validation";
  auto tokenizer = tokenizer::Gpt2Tokenizer::Load(tokenizer_dir);
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
  auto corpus = LoadTextCorpus("testdata/general_facts_dataset.txt");
  ASSERT_TRUE(corpus.ok()) << corpus.status();
  auto iterator = PaddedLineDataSetIterator::Create(
      *executor_, corpus->text(), **tokenizer,
      {.batch_size = 100, .eos_token = (*tokenizer)->eos_token_id()});
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  EXPECT_EQ((*iterator)->sample_count(), 1024);
  EXPECT_EQ((*iterator)->batches_per_epoch(), 11);
  EXPECT_EQ((*iterator)->supervised_row_count(), 10002);
  int samples = 0;
  int valid_targets = 0;
  for (size_t batch_index = 0; batch_index < (*iterator)->batches_per_epoch();
       ++batch_index) {
    auto batch = (*iterator)->Next();
    ASSERT_TRUE(batch.ok()) << batch.status();
    EXPECT_EQ(batch->sequence_length, 1024);
    samples += batch->batch_size;
    valid_targets += batch->supervised_row_count;
    const auto targets = Download(batch->targets);
    EXPECT_EQ(std::count_if(targets.begin(), targets.end(),
                            [](int target) { return target != -1; }),
              batch->supervised_row_count);
    for (int sample = 0; sample < batch->batch_size; ++sample)
      for (int row = 0; row < 4; ++row)
        EXPECT_EQ(targets[sample * 1024 + row], -1);
  }
  EXPECT_EQ(samples, 1024);
  EXPECT_EQ(valid_targets, 10002);
}

}  // namespace
}  // namespace pluto
