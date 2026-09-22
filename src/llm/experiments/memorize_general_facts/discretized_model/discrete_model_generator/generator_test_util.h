#pragma once

#include <cuda_runtime_api.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <string>

#include "gtest/gtest.h"
#include "src/cuda/executor.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/generate_discretized_model.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/utils.h"
#include "src/llm/gpt2.h"

namespace pluto::llm::discretized::generator {

// A real GPU model with zero weights predicts compact EOS (ID zero) on every
// row. Thus the one-token corpus has a provably correct completion without
// requiring external trained weights or tokenizer assets. Supports both the
// standalone ModelRecorder and the complete checkpoint-to-code pipeline.
class GeneratorTestBase : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string name = ::testing::TempDir() + "generator-test-XXXXXX";
    ASSERT_NE(::mkdtemp(name.data()), nullptr);
    directory_ = name;
    options_.recorder.checkpoint = directory_ / "step_0";
    options_.recorder.tokenizer = directory_ / "tokenizer";
    options_.recorder.corpus = directory_ / "corpus.txt";
    options_.output = directory_ / "generated";
    options_.recorder.layers = 1;
    options_.recorder.attention_heads = 1;
    options_.recorder.feed_forward_width = 64;
    options_.recorder.prompt_tokens = 1;
    options_.recorder.expected_samples = 1;
    options_.recorder.verify_greedy = true;
    const char* root = std::getenv("TEST_SRCDIR");
    const char* workspace = std::getenv("TEST_WORKSPACE");
    ASSERT_NE(root, nullptr);
    ASSERT_NE(workspace, nullptr);
    options_.clang_format_config =
        std::filesystem::path(root) / workspace / ".clang-format";
    std::filesystem::create_directories(options_.recorder.tokenizer);
    // The tokenizer asset parser requires at least one merge. "xx" is omitted
    // from the compact vocabulary, exercising retained-token encoding too.
    ASSERT_TRUE(
        WriteFile(
            options_.recorder.tokenizer / "tokenizer.json",
            R"asset({"model":{"type":"BPE","vocab":{"<|endoftext|>":0,"x":1,"xx":2},"merges":["x x"]}})asset")
            .ok());
    ASSERT_TRUE(WriteFile(options_.recorder.corpus, "x\n").ok());
    auto original =
        tokenizer::Gpt2Tokenizer::Load(options_.recorder.tokenizer.string());
    ASSERT_TRUE(original.ok()) << original.status();
    auto compact = tokenizer::CompactVocabularyTokenizer::Create(
        **original, {.compact_to_original = {0, 1},
                     .original_to_compact = {0, 1, -1},
                     .original_eos_id = 0});
    ASSERT_TRUE(compact.ok()) << compact.status();
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    const Gpt2Config config{.transformer_block_count = 1,
                            .model_width = 16,
                            .attention_heads = 1,
                            .feed_forward_width = 64,
                            .vocabulary_size = 2,
                            .pad_vocabulary = false};
    auto model = CreateGpt2(**executor, DataType::BF16, 0, config);
    ASSERT_TRUE(model.ok()) << model.status();
    for (const auto& weight : (*model)->weights())
      ASSERT_EQ(cudaMemsetAsync(weight.data(), 0, weight.size_bytes(),
                                (*executor)->stream()),
                cudaSuccess);
    auto saved =
        WriteToDirectory(**executor, **model, options_.recorder.checkpoint);
    ASSERT_TRUE(saved.ok()) << saved;
    auto mapping = (*compact)->SaveToFile(options_.recorder.checkpoint /
                                          "compact_vocabulary.tsv");
    ASSERT_TRUE(mapping.ok()) << mapping;
    ASSERT_TRUE((*executor)->Synchronize().ok());
  }

  void TearDown() override {
    std::error_code ignored;
    std::filesystem::remove_all(directory_, ignored);
  }

  void ExpectNoIntermediateFiles() {
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(directory_)) {
      const auto extension = entry.path().extension();
      if (entry.path() == options_.recorder.tokenizer / "tokenizer.json")
        continue;
      EXPECT_NE(extension, ".json") << entry.path();
      EXPECT_NE(extension, ".jsonl") << entry.path();
    }
  }

  std::filesystem::path directory_;
  GeneratorOptions options_;
};

}  // namespace pluto::llm::discretized::generator
