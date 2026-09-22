#include "src/llm/experiments/weight_sensitivity/evaluation.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/plain_text_tokenizer.h"
#include "src/llm/extract_top1_ids.h"
#include "src/llm/layers/embedding.h"
#include "src/util/status_macros.h"

namespace pluto::llm::weight_sensitivity {
namespace {

constexpr int kVocabulary = 256;
constexpr int kContext = 8;
constexpr int kEos = 255;
constexpr int kPrompt = 2;

class CompletionEvaluationTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
    // An embedding lookup is a tiny causal language model: the current input
    // byte selects its next-token logits. FP16's physical activation storage
    // is FP32 in this framework, so no projection or new test kernel is needed.
    auto model = EmbeddingLookupLayer::Create(
        *executor_, kVocabulary, kVocabulary, DataType::FP16, kContext);
    ASSERT_TRUE(model.ok()) << model.status();
    model_ = std::move(*model);
    weights_.assign(kVocabulary * kVocabulary, -1.0f);
    for (int input = 0; input < kVocabulary; ++input)
      Predict(input, kEos);
    Predict('b', 'c');
    Predict('g', 'i');
    Predict('i', 'j');
    // Only ignored positions see these nonfinite logits in the base corpus.
    weights_['a' * kVocabulary] = std::numeric_limits<float>::quiet_NaN();
    weights_['d' * kVocabulary] = std::numeric_limits<float>::infinity();
    weights_[kEos * kVocabulary] = std::numeric_limits<float>::quiet_NaN();
    ASSERT_TRUE(UploadWeights().ok());
  }

  void TearDown() override {
    if (executor_ == nullptr)
      return;
    EXPECT_TRUE(executor_->Synchronize().ok());
  }

  void Predict(int input, int output) {
    auto begin = weights_.begin() + input * kVocabulary;
    std::fill(begin, begin + kVocabulary, -1.0f);
    begin[output] = 2.0f;
  }

  absl::Status UploadWeights() {
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<float>::CopyFrom(
                                    *executor_, weights_));
    return cuda::CudaStatus(
        cudaMemcpyAsync(model_->weight().data(), host.data(), host.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "upload causal test model");
  }

  absl::StatusOr<std::unique_ptr<PaddedLineDataSetIterator>> Data(
      int batch_size = 2, bool shuffle = false) {
    return PaddedLineDataSetIterator::Create(*executor_, "abc\nde\nfgij",
                                             tokenizer_,
                                             {.batch_size = batch_size,
                                              .context_length = kContext,
                                              .prompt_tokens = kPrompt,
                                              .eos_token = kEos,
                                              .shuffle = shuffle});
  }

  // Independent prefix rollout: unlike evaluation, every step feeds its own
  // predicted token back into the next GPU forward. Retain EOS explicitly so
  // missing/incorrect termination cannot masquerade as a correct completion.
  absl::StatusOr<std::vector<int>> GreedyIncludingEos(
      absl::Span<const int> prompt) {
    std::vector<int> prefix(prompt.begin(), prompt.end());
    std::vector<int> result;
    ASSIGN_OR_RETURN(auto inputs, cuda::PageLockedHostArray<int>::Allocate(
                                      *executor_, kContext));
    ASSIGN_OR_RETURN(auto targets, cuda::PageLockedHostArray<int>::Allocate(
                                       *executor_, kContext));
    ASSIGN_OR_RETURN(auto selected,
                     cuda::PageLockedHostArray<int>::Allocate(*executor_, 1));
    ASSIGN_OR_RETURN(auto device_inputs,
                     cuda::Buffer::Allocate(*executor_, inputs.size_bytes()));
    ASSIGN_OR_RETURN(auto device_targets,
                     cuda::Buffer::Allocate(*executor_, targets.size_bytes()));
    while (prefix.size() <= kContext) {
      std::fill(inputs.begin(), inputs.end(), kEos);
      std::copy(prefix.begin(), prefix.end(), inputs.begin());
      std::fill(targets.begin(), targets.end(), -1);
      const size_t row = prefix.size() - 1;
      targets[row] = 0;
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(device_inputs.data(), inputs.data(),
                          inputs.size_bytes(), cudaMemcpyHostToDevice,
                          executor_->stream()),
          "upload greedy test prefix"));
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(device_targets.data(), targets.data(),
                          targets.size_bytes(), cudaMemcpyHostToDevice,
                          executor_->stream()),
          "upload greedy test mask"));
      ASSIGN_OR_RETURN(auto forward, model_->fwd(*executor_, {device_inputs}));
      ASSIGN_OR_RETURN(auto predictions,
                       ExtractTop1Ids(*executor_, forward.outputs[0],
                                      device_targets, kVocabulary));
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(selected.data(),
                          static_cast<const int*>(predictions.data()) + row,
                          sizeof(int), cudaMemcpyDeviceToHost,
                          executor_->stream()),
          "copy greedy test prediction"));
      RETURN_IF_ERROR(executor_->Synchronize());
      result.push_back(selected[0]);
      if (selected[0] == kEos || selected[0] < 0)
        break;
      prefix.push_back(selected[0]);
    }
    return result;
  }

  std::unique_ptr<cuda::Executor> executor_;
  std::unique_ptr<EmbeddingLookupLayer> model_;
  tokenizer::PlainTextTokenizer tokenizer_;
  std::vector<float> weights_;
};

TEST_F(CompletionEvaluationTest, ScoresMasksEosAndPartialBatchInCorpusOrder) {
  auto data = Data();
  ASSERT_TRUE(data.ok()) << data.status();
  ASSERT_TRUE((*data)->Next().ok());  // Evaluation must reset this position.
  for (int run = 0; run < 2; ++run) {
    auto scores = EvaluateCompletions(*executor_, *model_, **data, kVocabulary);
    ASSERT_TRUE(scores.ok()) << scores.status();
    EXPECT_EQ(scores->exact, (std::vector<uint8_t>{1, 1, 1}));
    EXPECT_EQ(scores->scored_tokens, 6);
    EXPECT_EQ(scores->token_errors, 0);
    EXPECT_EQ(scores->nonfinite_rows, 0);
  }
}

TEST_F(CompletionEvaluationTest, EosErrorsInvalidateEvenPromptOnlySentences) {
  Predict('e', 'x');
  ASSERT_TRUE(UploadWeights().ok());
  auto data = Data();
  ASSERT_TRUE(data.ok());
  auto scores = EvaluateCompletions(*executor_, *model_, **data, kVocabulary);
  ASSERT_TRUE(scores.ok()) << scores.status();
  EXPECT_EQ(scores->exact, (std::vector<uint8_t>{1, 0, 1}));
  EXPECT_EQ(scores->token_errors, 1);
  EXPECT_EQ(scores->scored_tokens, 6);
}

TEST_F(CompletionEvaluationTest, NonfiniteScoredRowsAreFailuresNotRunErrors) {
  weights_['b' * kVocabulary + 42] = std::numeric_limits<float>::quiet_NaN();
  weights_['i' * kVocabulary + 200] = -std::numeric_limits<float>::infinity();
  ASSERT_TRUE(UploadWeights().ok());
  auto data = Data();
  ASSERT_TRUE(data.ok());
  auto scores = EvaluateCompletions(*executor_, *model_, **data, kVocabulary);
  ASSERT_TRUE(scores.ok()) << scores.status();
  EXPECT_EQ(scores->exact, (std::vector<uint8_t>{0, 1, 0}));
  EXPECT_EQ(scores->scored_tokens, 6);
  EXPECT_EQ(scores->token_errors, 2);
  EXPECT_EQ(scores->nonfinite_rows, 2);
}

TEST_F(CompletionEvaluationTest, ExactSentenceDecisionMatchesGreedyRollout) {
  for (bool corrupted : {false, true}) {
    SCOPED_TRACE(corrupted);
    if (corrupted) {
      Predict('g', 'x');  // First divergence changes the following input too.
      Predict('x', 'y');
      Predict('e', 'x');  // Prompt-only sample must predict EOS, not more text.
      ASSERT_TRUE(UploadWeights().ok());
    }
    auto data = Data();
    ASSERT_TRUE(data.ok());
    auto scores = EvaluateCompletions(*executor_, *model_, **data, kVocabulary);
    ASSERT_TRUE(scores.ok()) << scores.status();
    for (size_t sample = 0; sample < (*data)->sample_count(); ++sample) {
      const auto tokens = (*data)->sample_tokens(sample);
      auto actual = GreedyIncludingEos(tokens.subspan(0, kPrompt));
      ASSERT_TRUE(actual.ok()) << actual.status();
      std::vector<int> expected(tokens.begin() + kPrompt, tokens.end());
      expected.push_back(kEos);
      EXPECT_EQ(scores->exact[sample], *actual == expected);
    }
    EXPECT_EQ(scores->exact, (corrupted ? std::vector<uint8_t>{1, 0, 0}
                                        : std::vector<uint8_t>{1, 1, 1}));
  }
}

TEST_F(CompletionEvaluationTest, CorpusSmallerThanBatchScoresEachSampleOnce) {
  auto data = Data(10);
  ASSERT_TRUE(data.ok());
  auto scores = EvaluateCompletions(*executor_, *model_, **data, kVocabulary);
  ASSERT_TRUE(scores.ok()) << scores.status();
  EXPECT_EQ(scores->exact, (std::vector<uint8_t>{1, 1, 1}));
  EXPECT_EQ(scores->scored_tokens, 6);
}

TEST_F(CompletionEvaluationTest, RejectsShuffleAndIncompatibleSignatures) {
  auto shuffled = Data(2, true);
  auto data = Data();
  ASSERT_TRUE(shuffled.ok());
  ASSERT_TRUE(data.ok());
  EXPECT_EQ(EvaluateCompletions(*executor_, *model_, **shuffled, kVocabulary)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  for (int vocabulary : {0, kEos, kVocabulary + 1})
    EXPECT_EQ(EvaluateCompletions(*executor_, *model_, **data, vocabulary)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  auto wrong_context = EmbeddingLookupLayer::Create(
      *executor_, kVocabulary, kVocabulary, DataType::FP16, kContext / 2);
  ASSERT_TRUE(wrong_context.ok());
  EXPECT_EQ(
      EvaluateCompletions(*executor_, **wrong_context, **data, kVocabulary)
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument);
  auto wrong_storage = EmbeddingLookupLayer::Create(
      *executor_, kVocabulary, kVocabulary, DataType::BF16, kContext);
  ASSERT_TRUE(wrong_storage.ok());
  EXPECT_EQ(
      EvaluateCompletions(*executor_, **wrong_storage, **data, kVocabulary)
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument);
  auto other_executor = cuda::Executor::Create();
  ASSERT_TRUE(other_executor.ok());
  EXPECT_EQ(EvaluateCompletions(**other_executor, *model_, **data, kVocabulary)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::llm::weight_sensitivity
