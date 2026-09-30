#include "src/llm/experiments/finite_state_machine/dataset.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/plain_text_tokenizer.h"

namespace pluto::llm::fsm {
namespace {

constexpr int kEos = 255;

// Numeric answers need not occupy the same number of tokens as ERR. This
// tokenizer merges the final 001 into one token, retaining byte tokens for
// the prompt and for ERR. It also lets a test simulate an invalid boundary.
class AnswerMergingTokenizer final : public tokenizer::Tokenizer {
 public:
  explicit AnswerMergingTokenizer(bool cross_boundary = false)
      : cross_boundary_(cross_boundary) {}

  absl::StatusOr<cuda::PageLockedHostArray<int>> Encode(
      cuda::Executor& executor, absl::string_view text) const override {
    std::vector<int> tokens(text.begin(), text.end());
    if (text.size() >= 4 && text.substr(text.size() - 4) == ";001") {
      tokens.resize(tokens.size() - 3);
      if (cross_boundary_)
        tokens.pop_back();
      tokens.push_back(256);
    }
    return cuda::PageLockedHostArray<int>::CopyFrom(executor, tokens);
  }

  int vocab_size() const override { return 257; }

 private:
  bool cross_boundary_;
};

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
    options.eos_token = kEos;
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
      const auto& line = lines[sample];
      const size_t answer_start = line.rfind(';') + 1;
      for (int row = 0; row < batch.sequence_length; ++row) {
        SCOPED_TRACE(testing::Message()
                     << "sample=" << sample << " row=" << row);
        const size_t index = sample * batch.sequence_length + row;
        EXPECT_EQ(inputs[index], row < line.size() ? line[row] : kEos);
        int expected_target = -1;
        if (row < line.size() && (!answer_only || row + 1 >= answer_start)) {
          expected_target = row + 1 < line.size() ? line[row + 1] : kEos;
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
  tokenizer::PlainTextTokenizer tokenizer_;
};

TEST_F(FsmDataSetTest, MasksPromptAndPaddingAndPredictsAnswerThenEos) {
  auto iterator = FsmDataSetIterator::Create(*executor_, "000A001;A;001\n",
                                             tokenizer_, Options(1, 16));
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  EXPECT_EQ((*iterator)->sample_count(), 1);
  EXPECT_EQ((*iterator)->batches_per_epoch(), 1);
  EXPECT_EQ((*iterator)->max_tokens(), 13);
  EXPECT_EQ((*iterator)->supervised_row_count(), 4);
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  EXPECT_EQ(batch->sequence_length, 16);
  ExpectSamples(*batch, {"000A001;A;001"});
  EXPECT_EQ(Download(batch->targets),
            (std::vector<int>{-1, -1, -1, -1, -1, -1, -1, -1, -1, '0', '0', '1',
                              kEos, -1, -1, -1}));
}

TEST_F(FsmDataSetTest, AllTokenObjectiveScoresEveryNextTokenAndEos) {
  auto options = Options(1, 16);
  options.answer_only = false;
  auto iterator = FsmDataSetIterator::Create(*executor_, "000A001;A;001",
                                             tokenizer_, options);
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  EXPECT_EQ((*iterator)->supervised_row_count(), 13);
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  ExpectSamples(*batch, {"000A001;A;001"}, false);
}

TEST_F(FsmDataSetTest, DifferentAnswerTokenCountsHaveExactLossMetadata) {
  AnswerMergingTokenizer tokenizer;
  auto iterator = FsmDataSetIterator::Create(
      *executor_, "000A001;A;001\n000A001;B;ERR", tokenizer, Options(2, 16));
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  EXPECT_EQ((*iterator)->max_tokens(), 13);
  EXPECT_EQ((*iterator)->supervised_row_count(), 6);
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  EXPECT_EQ(batch->supervised_row_count, 6);
  EXPECT_EQ(Download(batch->inputs),
            (std::vector<int>{'0', '0', '0', 'A',  '0',  '0',  '1',  ';',
                              'A', ';', 256, kEos, kEos, kEos, kEos, kEos,
                              '0', '0', '0', 'A',  '0',  '0',  '1',  ';',
                              'B', ';', 'E', 'R',  'R',  kEos, kEos, kEos}));
  EXPECT_EQ(
      Download(batch->targets),
      (std::vector<int>{-1, -1, -1, -1,  -1,  -1,  -1,   -1, -1, 256, kEos,
                        -1, -1, -1, -1,  -1,  -1,  -1,   -1, -1, -1,  -1,
                        -1, -1, -1, 'E', 'R', 'R', kEos, -1, -1, -1}));
}

TEST_F(FsmDataSetTest, PartialBatchAndEpochWrapNeverCrossOrDuplicateLines) {
  const std::vector<std::string> lines{"000A001;A;001", "000B002;BB;ERR",
                                       "000C003;003D004;CD;004"};
  auto iterator = FsmDataSetIterator::Create(
      *executor_, lines[0] + "\n" + lines[1] + "\n" + lines[2], tokenizer_,
      Options());
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  EXPECT_EQ((*iterator)->sample_count(), 3);
  EXPECT_EQ((*iterator)->batches_per_epoch(), 2);
  EXPECT_EQ((*iterator)->max_tokens(), lines[2].size());
  EXPECT_EQ((*iterator)->supervised_row_count(), 12);
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
      *executor_, "000A001;A;001\n000B002;B;002", tokenizer_, Options(8));
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  EXPECT_EQ((*iterator)->batches_per_epoch(), 1);
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  ExpectSamples(*batch, {"000A001;A;001", "000B002;B;002"});
}

TEST_F(FsmDataSetTest, ExactContextLengthRetainsLastAnswerAndEosTarget) {
  auto iterator = FsmDataSetIterator::Create(*executor_, "000A001;A;001",
                                             tokenizer_, Options(1, 13));
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  ExpectSamples(*batch, {"000A001;A;001"});
  EXPECT_EQ(Download(batch->targets).back(), kEos);
}

TEST_F(FsmDataSetTest, CrLfAndTerminalNewlineAreNotSampleTokens) {
  auto iterator = FsmDataSetIterator::Create(
      *executor_, "000A001;A;001\r\n000B002;B;002\r\n", tokenizer_, Options());
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  EXPECT_EQ((*iterator)->sample_count(), 2);
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  ExpectSamples(*batch, {"000A001;A;001", "000B002;B;002"});
}

TEST_F(FsmDataSetTest, TraversesFromStateZeroRegardlessOfTransitionOrder) {
  const std::string line = "009C123;007B009;000A007;ABC;123";
  auto iterator =
      FsmDataSetIterator::Create(*executor_, line, tokenizer_, Options(1, 40));
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  ExpectSamples(*batch, {line});
}

TEST_F(FsmDataSetTest, MissingTransitionsStayErrForRemainingInput) {
  const std::string line = "000A001;001B002;ABZA;ERR";
  auto iterator =
      FsmDataSetIterator::Create(*executor_, line, tokenizer_, Options(1, 40));
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  ExpectSamples(*batch, {line});
}

TEST_F(FsmDataSetTest, ShuffleVisitsEverySampleAndResetReplaysEpochs) {
  std::string corpus;
  const std::vector<int> ordered{'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'};
  for (int symbol : ordered) {
    if (!corpus.empty())
      corpus += '\n';
    corpus += "000" + std::string(1, symbol) + "001;" + std::string(1, symbol) +
              ";001";
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
        symbols.push_back(inputs[sample * batch->sequence_length + 3]);
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
  ASSERT_TRUE((*iterator)->Reset().ok());
  EXPECT_EQ(read_epoch(), first);
  EXPECT_EQ(read_epoch(), second);
}

TEST_F(FsmDataSetTest, ReturnedBatchSurvivesIteratorDestruction) {
  auto iterator = FsmDataSetIterator::Create(*executor_, "000A001;A;001",
                                             tokenizer_, Options(1));
  ASSERT_TRUE(iterator.ok()) << iterator.status();
  auto batch = (*iterator)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  iterator->reset();
  ExpectSamples(*batch, {"000A001;A;001"});
}

TEST_F(FsmDataSetTest, RejectsMalformedLinesAndIncorrectLabels) {
  const std::vector<std::string> invalid{"",
                                         "\n",
                                         "000A001;A;001\n\n",
                                         "000A001;A;001\n \n",
                                         "A;001",
                                         "000A001;001",
                                         "000A001;A",
                                         "000A001;;000",
                                         "00A001;A;001",
                                         "0000A001;A;001",
                                         "000A01;A;001",
                                         "000A0001;A;001",
                                         "00XA001;A;001",
                                         "000A0X1;A;001",
                                         "000a001;A;001",
                                         "0001001;A;001",
                                         "000AA001;A;001",
                                         "000A001;a;ERR",
                                         "000A001;A1;ERR",
                                         "000A001;A B;ERR",
                                         "000A001;A;01",
                                         "000A001;A;0001",
                                         "000A001;A;err",
                                         "000A001;A;0X1",
                                         "000A001;A;001;",
                                         "000A001;A;001 ",
                                         "000A001;A;002",
                                         "000A001;A;ERR",
                                         "000A001;B;001",
                                         "000A001;000A002;A;002",
                                         "000A001;000A001;A;001",
                                         "000A001;;A;001"};
  for (const auto& corpus : invalid) {
    SCOPED_TRACE(corpus);
    auto iterator = FsmDataSetIterator::Create(*executor_, corpus, tokenizer_,
                                               Options(1, 64));
    ASSERT_FALSE(iterator.ok());
    EXPECT_EQ(iterator.status().code(), absl::StatusCode::kInvalidArgument);
  }
}

TEST_F(FsmDataSetTest, RejectsInvalidDimensionsEosAndTruncation) {
  for (int field = 0; field < 8; ++field) {
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
        options.eos_token = -1;
        break;
      case 5:
        options.eos_token = tokenizer_.vocab_size();
        break;
      case 6:
        options.batch_size = std::numeric_limits<int>::max();
        break;
      case 7:
        options.context_length = 12;
        break;
    }
    auto iterator = FsmDataSetIterator::Create(*executor_, "000A001;A;001",
                                               tokenizer_, options);
    ASSERT_FALSE(iterator.ok()) << field;
    EXPECT_EQ(iterator.status().code(), absl::StatusCode::kInvalidArgument)
        << field;
  }
}

TEST_F(FsmDataSetTest, RejectsTokenThatCrossesPromptAnswerBoundary) {
  AnswerMergingTokenizer tokenizer(/*cross_boundary=*/true);
  auto iterator = FsmDataSetIterator::Create(*executor_, "000A001;A;001",
                                             tokenizer, Options(1));
  ASSERT_FALSE(iterator.ok());
  EXPECT_EQ(iterator.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(FsmDataSetTest,
       RealGpt2TrainingAndTestCorporaFitContextAndScoreAnswers) {
  const char* tokenizer_dir = std::getenv("PLUTO_GPT2_TOKENIZER_DIR");
  if (tokenizer_dir == nullptr)
    GTEST_SKIP() << "set PLUTO_GPT2_TOKENIZER_DIR for real-corpus validation";
  auto tokenizer = tokenizer::Gpt2Tokenizer::Load(tokenizer_dir);
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
  for (const auto& entry :
       {std::pair{"testdata/finite_state_machine_training_data.txt", 4096},
        std::pair{"testdata/finite_state_machine_test_data.txt", 128}}) {
    SCOPED_TRACE(entry.first);
    auto corpus = LoadTextCorpus(entry.first);
    ASSERT_TRUE(corpus.ok()) << corpus.status();
    auto options = Options(32, 1024);
    options.eos_token = (*tokenizer)->eos_token_id();
    auto iterator = FsmDataSetIterator::Create(*executor_, corpus->text(),
                                               **tokenizer, options);
    ASSERT_TRUE(iterator.ok()) << iterator.status();
    EXPECT_EQ((*iterator)->sample_count(), entry.second);
    EXPECT_LE((*iterator)->max_tokens(), 1024);
    int samples = 0;
    int supervised = 0;
    for (size_t index = 0; index < (*iterator)->batches_per_epoch(); ++index) {
      auto batch = (*iterator)->Next();
      ASSERT_TRUE(batch.ok()) << batch.status();
      EXPECT_EQ(batch->sequence_length, 1024);
      samples += batch->batch_size;
      supervised += batch->supervised_row_count;
      const auto targets = Download(batch->targets);
      EXPECT_EQ(std::count_if(targets.begin(), targets.end(),
                              [](int token) { return token != -1; }),
                batch->supervised_row_count);
      for (int sample = 0; sample < batch->batch_size; ++sample) {
        const auto begin = targets.begin() + sample * batch->sequence_length;
        const auto end = begin + batch->sequence_length;
        const auto last = std::find_if(std::make_reverse_iterator(end),
                                       std::make_reverse_iterator(begin),
                                       [](int token) { return token != -1; });
        ASSERT_NE(last, std::make_reverse_iterator(begin));
        EXPECT_EQ(*last, options.eos_token);
        EXPECT_EQ(*begin, -1);
      }
    }
    EXPECT_EQ(samples, entry.second);
    EXPECT_EQ(supervised, (*iterator)->supervised_row_count());
    EXPECT_GE(supervised, 2 * samples);
    EXPECT_LE(supervised, 4 * samples);
  }
}

}  // namespace
}  // namespace pluto::llm::fsm
