#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_capture.h"

#include <cuda_runtime_api.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/executor.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_core.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generate_discretized_model.h"
#include "src/llm/gpt2.h"

namespace pluto::llm::discretized::generator {
namespace {

// A real GPU model with zero weights predicts compact EOS (ID zero) on every
// row. Thus the one-token corpus has a provably correct completion without
// requiring a trained external checkpoint or the production tokenizer assets.
class GeneratorCaptureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string name = ::testing::TempDir() + "generator-capture-XXXXXX";
    ASSERT_NE(::mkdtemp(name.data()), nullptr);
    directory_ = name;
    options_.checkpoint = directory_ / "step_0";
    options_.tokenizer = directory_ / "tokenizer";
    options_.corpus = directory_ / "corpus.txt";
    options_.output = directory_ / "generated";
    options_.layers = 1;
    options_.attention_heads = 1;
    options_.feed_forward_width = 64;
    options_.prompt_tokens = 1;
    options_.expected_samples = 1;
    options_.verify_greedy = true;
    std::filesystem::create_directories(options_.tokenizer);
    // The parser requires at least one merge. "xx" is deliberately omitted
    // from the compact vocabulary, exercising retained-token encoding too.
    ASSERT_TRUE(
        WriteFile(
            options_.tokenizer / "tokenizer.json",
            R"json({"model":{"type":"BPE","vocab":{"<|endoftext|>":0,"x":1,"xx":2},"merges":["x x"]}})json")
            .ok());
    ASSERT_TRUE(WriteFile(options_.corpus, "x\n").ok());
    auto original = tokenizer::Gpt2Tokenizer::Load(options_.tokenizer.string());
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
    auto saved = WriteToDirectory(**executor, **model, options_.checkpoint);
    ASSERT_TRUE(saved.ok()) << saved;
    auto mapping =
        (*compact)->SaveToFile(options_.checkpoint / "compact_vocabulary.tsv");
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
      if (entry.path() == options_.tokenizer / "tokenizer.json")
        continue;
      EXPECT_NE(extension, ".json") << entry.path();
      EXPECT_NE(extension, ".jsonl") << entry.path();
    }
  }

  std::filesystem::path directory_;
  GeneratorOptions options_;
};

TEST_F(GeneratorCaptureTest, GenuineGpuTraceBecomesVerifiedInMemoryModel) {
  std::vector<Json> progress;
  options_.progress = [&](const Json& report) { progress.push_back(report); };
  auto model = CaptureCheckpoint(options_);
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ((*model)["width"], 16);
  EXPECT_EQ((*model)["layers"], 1);
  EXPECT_EQ((*model)["vocab_size"], 2);
  EXPECT_EQ((*model)["samples"].size(), 1u);
  EXPECT_EQ((*model)["samples"][0]["tokens"], Json::array({1}));
  // Identical zero vectors remain different symbols at distinct boundaries.
  EXPECT_EQ((*model)["states"].size(), 3u);
  for (const auto& state : (*model)["states"])
    EXPECT_EQ(state["bits"], Json(std::vector<int>(16, 0)));
  auto next = PredictNext(*model, {1});
  ASSERT_TRUE(next.ok()) << next.status();
  EXPECT_EQ(*next, 0);
  auto verification = EvaluateModel(*model);
  ASSERT_TRUE(verification.ok()) << verification.status();
  ASSERT_FALSE(progress.empty());
  EXPECT_EQ(progress.back()["phase"], "capture");
  EXPECT_EQ(progress.back()["greedy_verified"], true);
  EXPECT_FALSE(std::filesystem::exists(options_.output));
  ExpectNoIntermediateFiles();
}

TEST_F(GeneratorCaptureTest, OneStepGenerationWritesOnlyCppAndReadableReports) {
  const char* runfiles = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  ASSERT_NE(runfiles, nullptr);
  ASSERT_NE(workspace, nullptr);
  options_.clang_format_config =
      std::filesystem::path(runfiles) / workspace / ".clang-format";
  options_.reduce = true;
  options_.compact_transitions = true;
  options_.state_index = true;
  auto model = Generate(options_);
  ASSERT_TRUE(model.ok()) << model.status();
  auto source = ReadFile(options_.output / "model.h");
  ASSERT_TRUE(source.ok()) << source.status();
  EXPECT_NE(source->find("GeneratedModel()"), std::string::npos);
  EXPECT_NE(source->find("pluto::llm::discretized::gen"), std::string::npos);
  EXPECT_TRUE(std::filesystem::exists(options_.output / "attention_0.cc"));
  EXPECT_TRUE(std::filesystem::exists(options_.output / "mlp_0.cc"));
  EXPECT_TRUE(
      std::filesystem::exists(options_.output / "language_modeling_head.cc"));
  EXPECT_TRUE(std::filesystem::exists(options_.output /
                                      "generated_transition_test.cc"));
  EXPECT_TRUE(
      std::filesystem::exists(options_.output / "generation_report.txt"));
  EXPECT_TRUE(
      std::filesystem::exists(options_.output / "transition_patterns.txt"));
  EXPECT_TRUE(std::filesystem::exists(options_.output / "state_index.tsv"));
  EXPECT_EQ(Generate(options_).status().code(),
            absl::StatusCode::kAlreadyExists);
  ExpectNoIntermediateFiles();
}

TEST_F(GeneratorCaptureTest,
       CopiedCliFindsRunfilesAndResolvesCallerRelativePaths) {
  const char* runfiles = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  ASSERT_NE(runfiles, nullptr);
  ASSERT_NE(workspace, nullptr);
  const auto repository = std::filesystem::path(runfiles) / workspace;
  const auto executable = repository /
                          "src/llm/experiments/memorize_general_facts/"
                          "discretized_model/generate_discretized_model";
  ASSERT_TRUE(std::filesystem::is_regular_file(executable));
  const auto portable = directory_ / "portable";
  const auto tools = directory_ / "tools";
  const auto unrelated = directory_ / "unrelated";
  const auto copied_binary = portable / "generate_discretized_model";
  const auto copied_runfiles = portable / "generate_discretized_model.runfiles";
  std::filesystem::create_directories(copied_runfiles / "_main");
  std::filesystem::create_directory(tools);
  std::filesystem::create_directory(unrelated);
  // Dereference Bazel's symlinks when copying. The subprocess cannot discover
  // a source checkout, old Python modules, or original runfiles through these.
  ASSERT_TRUE(std::filesystem::copy_file(executable, copied_binary));
  ASSERT_TRUE(
      std::filesystem::copy_file(repository / ".clang-format",
                                 copied_runfiles / "_main" / ".clang-format"));
  std::filesystem::create_symlink("/usr/bin/clang-format",
                                  tools / "clang-format");

  // env changes only the child process; CUDA may have helper threads in this
  // test process, so neither its environment nor its working directory changes.
  // The child's PATH deliberately contains no Python or shell interpreter.
  std::vector<std::string> command{
      "/usr/bin/env",
      "-u",
      "RUNFILES_DIR",
      "-u",
      "RUNFILES_MANIFEST_FILE",
      "-u",
      "RUNFILES_MANIFEST_ONLY",
      "-u",
      "JAVA_RUNFILES",
      "-u",
      "PYTHON_RUNFILES",
      "-u",
      "PYTHONPATH",
      "-u",
      "PYTHONHOME",
      "-u",
      "TEST_SRCDIR",
      "-u",
      "TEST_WORKSPACE",
      "--chdir=" + unrelated.string(),
      "PATH=" + tools.string(),
      "BUILD_WORKING_DIRECTORY=" + directory_.string(),
      copied_binary.string(),
      "--checkpoint=step_0",
      "--tokenizer=tokenizer",
      "--corpus=corpus.txt",
      "--output=portable_generated",
      "--layers=1",
      "--attention_heads=1",
      "--feed_forward_width=64",
      "--prompt_tokens=1",
      "--expected_samples=1",
      "--verify_greedy",
      "--reduce",
      "--compact_transitions",
      "--state_index"};
  const auto result = RunProcess(command);
  ASSERT_TRUE(result.ok()) << result;
  const auto output = directory_ / "portable_generated";
  EXPECT_FALSE(std::filesystem::exists(unrelated / "portable_generated"));
  auto model_header = ReadFile(output / "model.h");
  ASSERT_TRUE(model_header.ok()) << model_header.status();
  EXPECT_NE(model_header->find("GeneratedModel()"), std::string::npos);
  EXPECT_TRUE(std::filesystem::is_regular_file(output /
                                               "generated_transition_test.cc"));
  EXPECT_TRUE(
      std::filesystem::is_regular_file(output / "generation_report.txt"));
  EXPECT_TRUE(
      std::filesystem::is_regular_file(output / "transition_patterns.txt"));
  ExpectNoIntermediateFiles();
}

TEST_F(GeneratorCaptureTest, RejectsWrongSampleCountAndIncorrectCompletions) {
  options_.expected_samples = 2;
  auto count = CaptureCheckpoint(options_);
  ASSERT_FALSE(count.ok());
  EXPECT_NE(count.status().message().find("expected 2 corpus samples"),
            std::string::npos);
  options_.expected_samples = 1;
  ASSERT_TRUE(WriteFile(options_.corpus, "xx\n").ok());
  auto prediction = CaptureCheckpoint(options_);
  ASSERT_FALSE(prediction.ok());
  EXPECT_NE(prediction.status().message().find("sample 0"), std::string::npos);
  EXPECT_FALSE(std::filesystem::exists(options_.output));
  ExpectNoIntermediateFiles();
}

TEST_F(GeneratorCaptureTest, RejectsMalformedAndPrefixOnlyCheckpoints) {
  ASSERT_TRUE(WriteFile(options_.checkpoint / "weight_999.bin", "extra").ok());
  auto extra = CaptureCheckpoint(options_);
  EXPECT_FALSE(extra.ok());
  ASSERT_TRUE(std::filesystem::remove(options_.checkpoint / "weight_999.bin"));
  ASSERT_TRUE(
      WriteFile(options_.checkpoint / "weight_0.bin", "truncated").ok());
  auto short_weight = CaptureCheckpoint(options_);
  EXPECT_FALSE(short_weight.ok());
  EXPECT_FALSE(std::filesystem::exists(options_.output));
  ExpectNoIntermediateFiles();
}

}  // namespace
}  // namespace pluto::llm::discretized::generator
