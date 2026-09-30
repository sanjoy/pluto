#include <cuda_runtime_api.h>

#include <cmath>
#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/plain_text_tokenizer.h"
#include "src/llm/adamw_optimizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/finite_state_machine/dataset.h"
#include "src/llm/gpt2.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/trainer.h"
#include "src/util/status_macros.h"

namespace pluto::llm::fsm {
namespace {

class FsmTrainingTest : public testing::Test {
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

  template <typename T>
  absl::StatusOr<std::vector<T>> Download(const cuda::Buffer& buffer) {
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<T>::Allocate(
                         *executor_, buffer.size_bytes() / sizeof(T)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), buffer.data(), buffer.size_bytes(),
                        cudaMemcpyDeviceToHost, executor_->stream()),
        "download FSM training test result"));
    RETURN_IF_ERROR(executor_->Synchronize());
    return std::vector<T>(host.begin(), host.end());
  }

  absl::StatusOr<float> EvaluateLoss(const Layer& model, const Layer& loss,
                                     FsmDataSetIterator& dataset) {
    ASSIGN_OR_RETURN(
        auto mean,
        Evaluate(*executor_, model,
                 {.loss_layer = loss,
                  .eval_data = dataset,
                  .batches = static_cast<int>(dataset.batches_per_epoch())}));
    ASSIGN_OR_RETURN(auto host, Download<float>(mean));
    if (host.size() != 1)
      return absl::InternalError("evaluation must return one scalar");
    return host[0];
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(FsmTrainingTest, SharedTrainerLearnsMaskedAnswersAndCheckpointRestores) {
  constexpr int kContext = 24;
  constexpr int kSteps = 20;
  constexpr absl::string_view kCorpus =
      "000A001;A;001\n000B002;B;002\n000A001;B;ERR";
  tokenizer::PlainTextTokenizer tokenizer;
  DataSetOptions options;
  options.batch_size = 2;
  options.context_length = kContext;
  options.eos_token = 255;
  auto training =
      FsmDataSetIterator::Create(*executor_, kCorpus, tokenizer, options);
  auto evaluation =
      FsmDataSetIterator::Create(*executor_, kCorpus, tokenizer, options);
  ASSERT_TRUE(training.ok()) << training.status();
  ASSERT_TRUE(evaluation.ok()) << evaluation.status();
  ASSERT_EQ((*evaluation)->batches_per_epoch(), 2);
  EXPECT_EQ((*evaluation)->supervised_row_count(), 12);

  Gpt2Config config;
  config.transformer_block_count = 1;
  config.model_width = 16;
  config.attention_heads = 2;
  config.feed_forward_width = 32;
  config.vocabulary_size = tokenizer.vocab_size();
  config.context_length = kContext;
  auto model = CreateGpt2(*executor_, DataType::BF16, 17, config);
  ASSERT_TRUE(model.ok()) << model.status();
  auto loss = CrossEntropyLossLayer::Create(*executor_, tokenizer.vocab_size(),
                                            DataType::BF16, kContext);
  ASSERT_TRUE(loss.ok()) << loss.status();

  // Independently add the actual per-row losses for a full and partial batch.
  // This catches division by padded rows or an unweighted mean of batches.
  double loss_sum = 0.0;
  int supervised = 0;
  for (size_t index = 0; index < (*evaluation)->batches_per_epoch(); ++index) {
    auto batch = (*evaluation)->Next();
    ASSERT_TRUE(batch.ok()) << batch.status();
    auto prediction = (*model)->fwd(*executor_, {batch->inputs});
    ASSERT_TRUE(prediction.ok()) << prediction.status();
    auto per_row_loss =
        (*loss)->fwd(*executor_, {prediction->outputs[0], batch->targets});
    ASSERT_TRUE(per_row_loss.ok()) << per_row_loss.status();
    auto values = Download<float>(per_row_loss->outputs[0]);
    auto targets = Download<int>(batch->targets);
    ASSERT_TRUE(values.ok()) << values.status();
    ASSERT_TRUE(targets.ok()) << targets.status();
    ASSERT_EQ(values->size(), targets->size());
    for (size_t row = 0; row < values->size(); ++row) {
      ASSERT_TRUE(std::isfinite((*values)[row]));
      if ((*targets)[row] == -1) {
        EXPECT_EQ((*values)[row], 0.0f);
      } else {
        loss_sum += (*values)[row];
        ++supervised;
      }
    }
  }
  ASSERT_EQ(supervised, 12);
  auto before = EvaluateLoss(**model, **loss, **evaluation);
  ASSERT_TRUE(before.ok()) << before.status();
  ASSERT_TRUE(std::isfinite(*before));
  EXPECT_NEAR(*before, loss_sum / supervised, 1e-5);

  auto optimizer = AdamWOptimizer::Create(
      *executor_, **model, {.learning_rate = 0.02f, .weight_decay = 0.0f});
  ASSERT_TRUE(optimizer.ok()) << optimizer.status();
  auto result = Train(*executor_, **model,
                      {.loss_layer = **loss,
                       .optimizer = **optimizer,
                       .training_data = **training,
                       .max_steps = kSteps});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->steps_completed, kSteps);
  EXPECT_EQ((*optimizer)->step(), kSteps);
  auto after = EvaluateLoss(**model, **loss, **evaluation);
  ASSERT_TRUE(after.ok()) << after.status();
  ASSERT_TRUE(std::isfinite(*after));
  EXPECT_LT(*after, *before);

  const auto checkpoint = std::filesystem::path(testing::TempDir()) /
                          "fsm-tiny-training" / "step_20";
  ASSERT_TRUE(WriteToDirectory(*executor_, **model, checkpoint).ok());
  auto restored = CreateGpt2(*executor_, DataType::BF16, 89, config);
  ASSERT_TRUE(restored.ok()) << restored.status();
  ASSERT_TRUE(ReadFromDirectory(*executor_, **restored, checkpoint,
                                /*allow_prefix=*/false)
                  .ok());
  ASSERT_EQ((*model)->weights().size(), (*restored)->weights().size());
  for (size_t index = 0; index < (*model)->weights().size(); ++index) {
    auto expected = Download<unsigned char>((*model)->weights()[index]);
    auto actual = Download<unsigned char>((*restored)->weights()[index]);
    ASSERT_TRUE(expected.ok()) << expected.status();
    ASSERT_TRUE(actual.ok()) << actual.status();
    EXPECT_EQ(*actual, *expected) << "weight " << index;
  }
  auto restored_loss = EvaluateLoss(**restored, **loss, **evaluation);
  ASSERT_TRUE(restored_loss.ok()) << restored_loss.status();
  EXPECT_FLOAT_EQ(*restored_loss, *after);
}

}  // namespace
}  // namespace pluto::llm::fsm
