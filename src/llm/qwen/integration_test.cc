#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

#include "gtest/gtest.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/qwen_tokenizer.h"
#include "src/llm/qwen/model.h"
#include "src/util/status_macros.h"

namespace pluto::llm::qwen {
namespace {

// A real-checkpoint smoke test is opt-in and never downloads weights. Default
// synthetic tests cover operators/cache/error behavior without 30 GB of data.
TEST(QwenIntegrationTest, RealCheckpointCompletesRawAndChatPromptsAfterReset) {
  const char* directory = std::getenv("PLUTO_QWEN_CHECKPOINT_DIR");
  if (!directory || !*directory)
    GTEST_SKIP() << "set PLUTO_QWEN_CHECKPOINT_DIR for real FP8 inference";
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  auto tokenizer = tokenizer::QwenTokenizer::Load(directory);
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
  InferenceOptions options;
  options.context_length = 128;
  auto model = Model::Load(**executor, directory, options);
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ((*model)->config().num_hidden_layers, 64);
  EXPECT_LT((*model)->weight_bytes(), size_t{32} * 1024 * 1024 * 1024);
  auto scores = cuda::PageLockedHostArray<float>::Allocate(
      **executor, (*model)->config().vocab_size);
  ASSERT_TRUE(scores.ok()) << scores.status();
  auto predict = [&]() -> absl::StatusOr<int> {
    ASSIGN_OR_RETURN(auto logits, (*model)->Logits());
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(scores->data(), logits.data(), scores->size_bytes(),
                        cudaMemcpyDeviceToHost, (*executor)->stream()),
        "test logits copy"));
    RETURN_IF_ERROR((*executor)->Synchronize());
    for (float score : *scores)
      if (!std::isfinite(score))
        return absl::DataLossError("non-finite real model logits");
    return static_cast<int>(
        std::max_element(scores->begin(),
                         scores->begin() + (*tokenizer)->vocab_size()) -
        scores->begin());
  };
  auto raw = (*tokenizer)->Encode("The capital of France is");
  ASSERT_TRUE(raw.ok()) << raw.status();
  for (int token : *raw)
    ASSERT_TRUE((*model)->Step(token).ok());
  auto paris = predict();
  ASSERT_TRUE(paris.ok()) << paris.status();
  auto decoded = (*tokenizer)->Decode({*paris});
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(*decoded, " Paris");

  ASSERT_TRUE((*model)->Reset().ok());
  EXPECT_EQ((*model)->position(), 0);
  auto chat = (*tokenizer)
                  ->Encode(tokenizer::QwenTokenizer::ChatPrompt(
                      "What is 2 + 2? Reply with just the number."));
  ASSERT_TRUE(chat.ok()) << chat.status();
  for (int token : *chat)
    ASSERT_TRUE((*model)->Step(token).ok());
  auto four = predict();
  ASSERT_TRUE(four.ok()) << four.status();
  decoded = (*tokenizer)->Decode({*four});
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(*decoded, "4");
  ASSERT_TRUE((*model)->Step(*four).ok());
  auto end = predict();
  ASSERT_TRUE(end.ok()) << end.status();
  EXPECT_EQ(*end, (*tokenizer)->eos_token_id());
}

}  // namespace
}  // namespace pluto::llm::qwen
