#include "src/llm/experiments/finite_state_machine/dataset.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/experiments/finite_state_machine/tokenizer.h"

namespace pluto::llm::fsm {
namespace {

class FsmDataSetTest : public testing::Test {
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

  DataSetOptions Options(int batch_size = 2, int context_length = 24) {
    DataSetOptions options;
    options.batch_size = batch_size;
    options.context_length = context_length;
    return options;
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

  void ExpectSamples(const DataBatch& batch,
                     const std::vector<std::string>& lines,
                     bool answer_only = true) {
    ASSERT_EQ(batch.batch_size, lines.size());
    const auto inputs = Download(batch.inputs);
    const auto targets = Download(batch.targets);
    ASSERT_EQ(inputs.size(), lines.size() * batch.sequence_length);
    ASSERT_EQ(targets.size(), inputs.size());
    int supervised = 0;
    for (size_t sample = 0; sample < lines.size(); ++sample) {
      auto tokens = tokenizer_.Encode(*executor_, lines[sample]);
      ASSERT_TRUE(tokens.ok()) << tokens.status();
      ASSERT_GE(tokens->size(), 2);
      for (int row = 0; row < batch.sequence_length; ++row) {
        SCOPED_TRACE(testing::Message()
                     << "sample=" << sample << " row=" << row);
        const size_t index = sample * batch.sequence_length + row;
        EXPECT_EQ(inputs[index], row < tokens->size() ? (*tokens)[row] : 0);
        int expected_target = -1;
        if (row + 1 < tokens->size() &&
            (!answer_only || row + 2 == tokens->size())) {
          expected_target = (*tokens)[row + 1];
          ++supervised;
        }
        EXPECT_EQ(targets[index], expected_target);
      }
    }
    EXPECT_EQ(batch.supervised_row_count, supervised);
    EXPECT_EQ(std::count_if(targets.begin(), targets.end(),
                            [](int token) { return token != -1; }),
              supervised);
  }

  std::unique_ptr<cuda::Executor> executor_;
  FsmTokenizer tokenizer_;
};

TEST_F(FsmDataSetTest, ExactlyOneAnswerTargetFollowsOutputSeparator) {
  auto iterator = FsmDataSetIterator::Create(*executor_, "000A001;A>001\n",
                                             tokenizer_, Options(1, 10));
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  EXPECT_EQ((*iterator)->sample_count(), 1);
  EXPECT_EQ((*iterator)->batches_per_epoch(), 1);
  EXPECT_EQ((*iterator)->max_tokens(), 7);
  EXPECT_EQ((*iterator)->supervised_row_count(), 1);
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  EXPECT_EQ(batch->sequence_length, 10);
  ExpectSamples(*batch, {"000A001;A>001"});
  EXPECT_EQ(Download(batch->inputs),
            (std::vector<int>{0, 1000, 1, 1026, 1000, 1027, 1, 0, 0, 0}));
  EXPECT_EQ(Download(batch->targets),
            (std::vector<int>{-1, -1, -1, -1, -1, 1, -1, -1, -1, -1}));
}

TEST_F(FsmDataSetTest, AllTokenObjectiveScoresNextTokensWithoutEos) {
  auto options = Options(1, 10);
  options.answer_only = false;
  auto iterator = FsmDataSetIterator::Create(*executor_, "000A001;A>001",
                                             tokenizer_, options);
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  EXPECT_EQ((*iterator)->supervised_row_count(), 6);
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  ExpectSamples(*batch, {"000A001;A>001"}, false);
  EXPECT_EQ(Download(batch->targets),
            (std::vector<int>{1000, 1, 1026, 1000, 1027, 1, -1, -1, -1, -1}));
}

TEST_F(FsmDataSetTest, StateAndErrAnswersEachOccupyOneToken) {
  auto iterator = FsmDataSetIterator::Create(
      *executor_, "000A001;A>001\n000A001;B>ERR", tokenizer_, Options(2, 8));
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  EXPECT_EQ((*iterator)->max_tokens(), 7);
  EXPECT_EQ((*iterator)->supervised_row_count(), 2);
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  ExpectSamples(*batch, {"000A001;A>001", "000A001;B>ERR"});
  EXPECT_EQ(Download(batch->inputs),
            (std::vector<int>{0, 1000, 1, 1026, 1000, 1027, 1, 0, 0, 1000, 1,
                              1026, 1001, 1027, 1028, 0}));
  EXPECT_EQ(Download(batch->targets),
            (std::vector<int>{-1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1,
                              1028, -1, -1}));
}

TEST_F(FsmDataSetTest, ErrInInputIsThreeLettersAndErrOutputIsOneToken) {
  auto iterator = FsmDataSetIterator::Create(
      *executor_, "000E001;001R001;ERR>001\n000E001;ERR>ERR", tokenizer_,
      Options(2, 16));
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  EXPECT_EQ((*iterator)->max_tokens(), 13);
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  ExpectSamples(*batch, {"000E001;001R001;ERR>001", "000E001;ERR>ERR"});
  const auto inputs = Download(batch->inputs);
  EXPECT_EQ((std::vector<int>(inputs.begin() + 8, inputs.begin() + 13)),
            (std::vector<int>{1004, 1017, 1017, 1027, 1}));
  EXPECT_EQ((std::vector<int>(inputs.begin() + 20, inputs.begin() + 25)),
            (std::vector<int>{1004, 1017, 1017, 1027, 1028}));
  EXPECT_EQ(batch->supervised_row_count, 2);
}

TEST_F(FsmDataSetTest, ZeroStateAnswerRemainsSupervisedDespiteZeroPadding) {
  auto iterator = FsmDataSetIterator::Create(*executor_, "000A000;A>000",
                                             tokenizer_, Options(1, 10));
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  ExpectSamples(*batch, {"000A000;A>000"});
  EXPECT_EQ(Download(batch->targets),
            (std::vector<int>{-1, -1, -1, -1, -1, 0, -1, -1, -1, -1}));
  EXPECT_EQ(batch->supervised_row_count, 1);
}

TEST_F(FsmDataSetTest, PartialBatchAndEpochWrapNeverCrossOrDuplicateLines) {
  const std::vector<std::string> lines{"000A001;A>001", "000B002;BB>ERR",
                                       "000C003;003D004;CD>004"};
  auto iterator = FsmDataSetIterator::Create(
      *executor_, lines[0] + "\n" + lines[1] + "\n" + lines[2], tokenizer_,
      Options());
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  EXPECT_EQ((*iterator)->sample_count(), 3);
  EXPECT_EQ((*iterator)->batches_per_epoch(), 2);
  EXPECT_EQ((*iterator)->max_tokens(), 12);
  EXPECT_EQ((*iterator)->supervised_row_count(), 3);
  for (int epoch = 0; epoch < 2; ++epoch) {
    auto full = (*iterator)->Next();
    ASSERT_TRUE(full.ok()) << full.status();
    ExpectSamples(*full, {lines[0], lines[1]});
    auto tail = (*iterator)->Next();
    ASSERT_TRUE(tail.ok()) << tail.status();
    ExpectSamples(*tail, {lines[2]});
    EXPECT_EQ(tail->inputs.size_bytes(), 24 * sizeof(int));
    EXPECT_EQ(tail->targets.size_bytes(), 24 * sizeof(int));
  }
}

TEST_F(FsmDataSetTest, BatchLargerThanCorpusContainsOnlyExistingSamples) {
  auto iterator = FsmDataSetIterator::Create(
      *executor_, "000A001;A>001\n000B002;B>002", tokenizer_, Options(8));
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  EXPECT_EQ((*iterator)->batches_per_epoch(), 1);
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  ExpectSamples(*batch, {"000A001;A>001", "000B002;B>002"});
}

TEST_F(FsmDataSetTest, ExactContextLengthRetainsAnswerAndMasksFinalRow) {
  auto iterator = FsmDataSetIterator::Create(*executor_, "000A001;A>001",
                                             tokenizer_, Options(1, 7));
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  ExpectSamples(*batch, {"000A001;A>001"});
  EXPECT_EQ(Download(batch->targets).back(), -1);
  EXPECT_EQ(batch->supervised_row_count, 1);
}

TEST_F(FsmDataSetTest, CrLfAndTerminalNewlineAreNotSampleTokens) {
  auto iterator = FsmDataSetIterator::Create(
      *executor_, "000A001;A>001\r\n000B002;B>002\r\n", tokenizer_, Options());
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  EXPECT_EQ((*iterator)->sample_count(), 2);
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  ExpectSamples(*batch, {"000A001;A>001", "000B002;B>002"});
}

TEST_F(FsmDataSetTest, TraversesFromStateZeroRegardlessOfTransitionOrder) {
  const std::string line = "009C123;007B009;000A007;ABC>123";
  auto iterator =
      FsmDataSetIterator::Create(*executor_, line, tokenizer_, Options(1, 24));
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  ExpectSamples(*batch, {line});
}

TEST_F(FsmDataSetTest, MissingTransitionsStayErrForRemainingInput) {
  const std::string line = "000A001;001B002;ABZA>ERR";
  auto iterator =
      FsmDataSetIterator::Create(*executor_, line, tokenizer_, Options(1, 24));
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  ExpectSamples(*batch, {line});
}

TEST_F(FsmDataSetTest, ShuffleVisitsEverySampleAndResetReplaysEpochs) {
  std::string corpus;
  std::vector<int> ordered;
  for (char symbol = 'A'; symbol <= 'I'; ++symbol) {
    if (!corpus.empty())
      corpus += '\n';
    corpus += "000" + std::string(1, symbol) + "001;" + std::string(1, symbol) +
              ">001";
    ordered.push_back(kLetterOffset + symbol - 'A');
  }
  auto options = Options();
  options.shuffle = true;
  options.seed = 19;
  auto iterator =
      FsmDataSetIterator::Create(*executor_, corpus, tokenizer_, options);
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  const auto read_epoch = [&]() {
    std::vector<int> symbols;
    for (size_t index = 0; index < (*iterator)->batches_per_epoch(); ++index) {
      auto batch = (*iterator)->Next();
      EXPECT_TRUE(batch.ok()) << batch.status();
      if (!batch.ok())
        return std::vector<int>{};
      const auto inputs = Download(batch->inputs);
      for (int sample = 0; sample < batch->batch_size; ++sample)
        symbols.push_back(inputs[sample * batch->sequence_length + 1]);
    }
    return symbols;
  };
  const auto first = read_epoch();
  const auto second = read_epoch();
  for (auto epoch : {first, second}) {
    std::sort(epoch.begin(), epoch.end());
    EXPECT_EQ(epoch, ordered);
  }
  EXPECT_NE(first, ordered);
  EXPECT_NE(first, second);
  // Reset midway through the third epoch must also replay the original order.
  ASSERT_TRUE((*iterator)->Next().ok());
  ASSERT_TRUE((*iterator)->Reset().ok());
  EXPECT_EQ(read_epoch(), first);
  EXPECT_EQ(read_epoch(), second);
}

TEST_F(FsmDataSetTest, ReturnedBatchSurvivesIteratorDestruction) {
  auto iterator = FsmDataSetIterator::Create(*executor_, "000A001;A>001",
                                             tokenizer_, Options(1));
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  iterator->reset();
  ExpectSamples(*batch, {"000A001;A>001"});
}

TEST_F(FsmDataSetTest, RejectsMalformedLinesOldSyntaxAndIncorrectLabels) {
  const std::vector<std::string> invalid{"",
                                         "\n",
                                         "000A001;A>001\n\n",
                                         "000A001;A>001\n \n",
                                         "A>001",
                                         "000A001>001",
                                         "000A001;A",
                                         "000A001;>000",
                                         "00A001;A>001",
                                         "0000A001;A>001",
                                         "000A01;A>001",
                                         "000A0001;A>001",
                                         "00XA001;A>001",
                                         "000A0X1;A>001",
                                         "000a001;A>001",
                                         "0001001;A>001",
                                         "000AA001;A>001",
                                         "000A001;a>ERR",
                                         "000A001;A1>ERR",
                                         "000A001;A B>ERR",
                                         "000A001;A>01",
                                         "000A001;A>0001",
                                         "000A001;A>err",
                                         "000A001;A>0X1",
                                         "000A001;A>001;",
                                         "000A001;A>001 ",
                                         "000A001;A>002",
                                         "000A001;A>ERR",
                                         "000A001;B>001",
                                         "000A001;000A002;A>002",
                                         "000A001;000A001;A>001",
                                         "000A001;;A>001",
                                         "000A001;A;001",
                                         "000A001;B;ERR",
                                         "000A001;A>>001",
                                         "000A001;A>001>001",
                                         "000A001;A>",
                                         "000A001;A>001ERR",
                                         "000A001;B>ERRERR"};
  for (const auto& corpus : invalid) {
    SCOPED_TRACE(corpus);
    auto iterator = FsmDataSetIterator::Create(*executor_, corpus, tokenizer_,
                                               Options(1, 64));
    ASSERT_FALSE(iterator.ok());
    EXPECT_EQ(iterator.status().code(), absl::StatusCode::kInvalidArgument);
  }
}

TEST_F(FsmDataSetTest, RejectsInvalidDimensionsAndTruncation) {
  for (int field = 0; field < 6; ++field) {
    auto options = Options();
    switch (field) {
      case 0:
        options.batch_size = 0;
        break;
      case 1:
        options.batch_size = -1;
        break;
      case 2:
        options.context_length = 0;
        break;
      case 3:
        options.context_length = -1;
        break;
      case 4:
        options.batch_size = std::numeric_limits<int>::max();
        break;
      case 5:
        options.context_length = 6;
        break;
    }
    auto iterator = FsmDataSetIterator::Create(*executor_, "000A001;A>001",
                                               tokenizer_, options);
    ASSERT_FALSE(iterator.ok()) << field;
    EXPECT_EQ(iterator.status().code(), absl::StatusCode::kInvalidArgument)
        << field;
  }
}

TEST_F(FsmDataSetTest,
       TrainingAndTestCorporaFitContextWithOneTargetPerExample) {
  for (const auto& entry :
       {std::pair{"testdata/finite_state_machine_training_data.txt", 4096},
        std::pair{"testdata/finite_state_machine_test_data.txt", 128}}) {
    SCOPED_TRACE(entry.first);
    auto corpus = LoadTextCorpus(entry.first);
    ASSERT_TRUE(corpus.ok()) << corpus.status();
    auto iterator = FsmDataSetIterator::Create(*executor_, corpus->text(),
                                               tokenizer_, Options(32, 1024));
    ASSERT_TRUE(iterator.ok()) << iterator.status();
    EXPECT_EQ((*iterator)->sample_count(), entry.second);
    EXPECT_EQ((*iterator)->supervised_row_count(), entry.second);
    EXPECT_LE((*iterator)->max_tokens(), 1024);
    int samples = 0;
    int supervised = 0;
    for (size_t index = 0; index < (*iterator)->batches_per_epoch(); ++index) {
      auto batch = (*iterator)->Next();
      ASSERT_TRUE(batch.ok()) << batch.status();
      EXPECT_EQ(batch->sequence_length, 1024);
      EXPECT_EQ(batch->supervised_row_count, batch->batch_size);
      samples += batch->batch_size;
      supervised += batch->supervised_row_count;
      const auto inputs = Download(batch->inputs);
      const auto targets = Download(batch->targets);
      ASSERT_EQ(inputs.size(), targets.size());
      for (int sample = 0; sample < batch->batch_size; ++sample) {
        const size_t base = sample * batch->sequence_length;
        int answer_rows = 0;
        for (int row = 0; row < batch->sequence_length; ++row) {
          if (targets[base + row] == -1)
            continue;
          ++answer_rows;
          EXPECT_EQ(inputs[base + row], kOutputSeparatorToken);
          ASSERT_LT(row + 1, batch->sequence_length);
          EXPECT_EQ(inputs[base + row + 1], targets[base + row]);
          EXPECT_TRUE(
              (targets[base + row] >= 0 && targets[base + row] < kStateCount) ||
              targets[base + row] == kErrorToken);
        }
        EXPECT_EQ(answer_rows, 1);
      }
    }
    EXPECT_EQ(samples, entry.second);
    EXPECT_EQ(supervised, entry.second);
  }
}

}  // namespace
}  // namespace pluto::llm::fsm
