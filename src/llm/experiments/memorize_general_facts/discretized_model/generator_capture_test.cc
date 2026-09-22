#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_capture.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

#include "gtest/gtest.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_core.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generate_discretized_model.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_test_util.h"

namespace pluto::llm::discretized::generator {
namespace {

class GeneratorCaptureTest : public GeneratorTestBase {};

TEST_F(GeneratorCaptureTest, GenuineGpuTraceBecomesVerifiedInMemoryModel) {
  std::vector<ProgressEvent> progress;
  options_.progress = [&](const ProgressEvent& report) {
    progress.push_back(report);
  };
  auto model = CaptureCheckpoint(options_);
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ(model->metadata.width, 16);
  EXPECT_EQ(model->metadata.layers, 1);
  EXPECT_EQ(model->metadata.vocab_size, 2);
  EXPECT_EQ(model->samples.size(), 1u);
  EXPECT_EQ(model->samples[0].tokens, (std::vector<int>{1}));
  // Identical zero vectors remain different symbols at distinct boundaries.
  EXPECT_EQ(model->states.size(), 3u);
  for (const auto& state : model->states)
    EXPECT_EQ(state.bits, std::vector<uint16_t>(16, 0));
  auto next = PredictNext(*model, {1});
  ASSERT_TRUE(next.ok()) << next.status();
  EXPECT_EQ(*next, 0);
  auto verification = EvaluateModel(*model);
  ASSERT_TRUE(verification.ok()) << verification.status();
  ASSERT_FALSE(progress.empty());
  ASSERT_TRUE(std::holds_alternative<CaptureProgress>(progress.back()));
  EXPECT_TRUE(std::get<CaptureProgress>(progress.back()).greedy_verified);
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
