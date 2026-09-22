#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/model_recorder.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

#include "gtest/gtest.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model_util.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/generate_discretized_model.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/generator_test_util.h"

namespace pluto::llm::discretized::generator {
namespace {

class ModelRecorderTest : public GeneratorTestBase {};

TEST_F(ModelRecorderTest, GenuineGpuTraceBecomesVerifiedInMemoryModel) {
  std::vector<CaptureProgress> progress;
  options_.recorder.progress = [&](const CaptureProgress& report) {
    progress.push_back(report);
  };
  auto model = ModelRecorder::Record(options_.recorder);
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ(model->metadata.width, 16);
  EXPECT_EQ(model->metadata.layers, 1);
  EXPECT_EQ(model->metadata.vocab_size, 2);
  EXPECT_EQ(model->samples.size(), 1u);
  EXPECT_EQ(model->samples[0].tokens, (std::vector<int>{1}));
  // Identical zero vectors remain different symbols at distinct boundaries.
  EXPECT_EQ(model->states.size(), 3u);
  EXPECT_EQ(model->position_embedding.transitions.size(), 1u);
  ASSERT_EQ(model->transformers.size(), 1u);
  EXPECT_EQ(model->transformers[0].attention.transitions.size(), 1u);
  EXPECT_EQ(model->transformers[0].mlp.transitions.size(), 1u);
  EXPECT_EQ(model->language_modeling_head.transitions.size(), 1u);
  auto next = PredictNext(*model, {1});
  ASSERT_TRUE(next.ok()) << next.status();
  EXPECT_EQ(*next, 0);
  auto verification = EvaluateModel(*model);
  ASSERT_TRUE(verification.ok()) << verification.status();
  ASSERT_FALSE(progress.empty());
  EXPECT_TRUE(progress.back().greedy_verified);
  EXPECT_FALSE(std::filesystem::exists(options_.output));
  ExpectNoIntermediateFiles();
}

TEST_F(ModelRecorderTest, OptionalVectorHintsDoNotChangeCapturedModel) {
  // Recording does not require a formatter, destination, or generator flags.
  // It can be used directly by an in-memory analysis or compaction client.
  ModelRecorderOptions recorder = options_.recorder;
  options_.output.clear();
  options_.clang_format_config.clear();
  auto symbolic = ModelRecorder::Record(recorder);
  ASSERT_TRUE(symbolic.ok()) << symbolic.status();
  StateVectorHints hints{{-1, {42}}};
  auto with_hints = ModelRecorder::Record(recorder, &hints);
  ASSERT_TRUE(with_hints.ok()) << with_hints.status();
  EXPECT_EQ(*with_hints, *symbolic);
  EXPECT_EQ(hints.size(), symbolic->states.size());
  EXPECT_FALSE(hints.contains(-1));
  for (const auto& state : symbolic->states) {
    ASSERT_TRUE(hints.contains(state.id));
    EXPECT_EQ(hints.at(state.id), std::vector<uint16_t>(16, 0));
  }
  recorder.expected_samples = 2;
  const auto saved_hints = hints;
  EXPECT_FALSE(ModelRecorder::Record(recorder, &hints).ok());
  EXPECT_EQ(hints, saved_hints);
  ExpectNoIntermediateFiles();
}

TEST_F(ModelRecorderTest, RejectsUnsupportedModelDimensions) {
  options_.recorder.model_width = 32;
  EXPECT_EQ(ModelRecorder::Record(options_.recorder).status().code(),
            absl::StatusCode::kInvalidArgument);
  options_.recorder.model_width = 16;
  options_.recorder.context_length = 512;
  EXPECT_EQ(ModelRecorder::Record(options_.recorder).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(ModelRecorderTest, IncorrectGpuPredictionsPreserveExistingHints) {
  // The zero-weight model predicts EOS, not the second "x". This failure
  // occurs after loading the checkpoint and actually observing its GPU
  // forward pass. Disabling rollout verification must not disable this check.
  ASSERT_TRUE(WriteFile(options_.recorder.corpus, "xx\n").ok());
  StateVectorHints hints{{99, std::vector<uint16_t>(16, 42)}};
  const auto saved_hints = hints;
  for (bool verify_greedy : {false, true}) {
    options_.recorder.verify_greedy = verify_greedy;
    auto result = ModelRecorder::Record(options_.recorder, &hints);
    ASSERT_FALSE(result.ok());
    EXPECT_NE(result.status().message().find("sample 0"), std::string::npos);
    EXPECT_EQ(hints, saved_hints);
  }
  EXPECT_FALSE(std::filesystem::exists(options_.output));
  ExpectNoIntermediateFiles();
}

TEST_F(ModelRecorderTest, OneStepGenerationWritesOnlyCppAndReadableReports) {
  const char* runfiles = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  ASSERT_NE(runfiles, nullptr);
  ASSERT_NE(workspace, nullptr);
  options_.clang_format_config =
      std::filesystem::path(runfiles) / workspace / ".clang-format";
  options_.compaction = true;
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
  EXPECT_FALSE(
      std::filesystem::exists(options_.output / "transition_patterns.txt"));
  EXPECT_TRUE(std::filesystem::exists(options_.output / "state_index.tsv"));
  EXPECT_EQ(Generate(options_).status().code(),
            absl::StatusCode::kAlreadyExists);
  ExpectNoIntermediateFiles();
}

TEST_F(ModelRecorderTest,
       CopiedCliFindsRunfilesAndResolvesCallerRelativePaths) {
  const char* runfiles = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  ASSERT_NE(runfiles, nullptr);
  ASSERT_NE(workspace, nullptr);
  const auto repository = std::filesystem::path(runfiles) / workspace;
  const auto executable =
      repository /
      "src/llm/experiments/memorize_general_facts/"
      "discretized_model/discrete_model_generator/generate_discretized_model";
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
      "--compaction",
      "--compaction_neighbors=2",
      "--compaction_max_passes=3",
      "--compaction_max_attempts=10",
      "--compaction_exhaustive_pair_limit=1000",
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
  EXPECT_FALSE(std::filesystem::exists(output / "transition_patterns.txt"));
  auto report = ReadFile(output / "generation_report.txt");
  ASSERT_TRUE(report.ok()) << report.status();
  EXPECT_NE(report->find("compaction_search:"), std::string::npos);
  EXPECT_NE(report->find("nearest_neighbors: 2"), std::string::npos);
  EXPECT_NE(report->find("exhaustive_pair_limit: 1000"), std::string::npos);
  EXPECT_NE(report->find("pairwise_compaction_complete: true"),
            std::string::npos);
  ExpectNoIntermediateFiles();
}

TEST_F(ModelRecorderTest, RejectsWrongSampleCountAndIncorrectCompletions) {
  options_.recorder.expected_samples = 2;
  auto count = ModelRecorder::Record(options_.recorder);
  ASSERT_FALSE(count.ok());
  EXPECT_NE(count.status().message().find("expected 2 corpus samples"),
            std::string::npos);
  options_.recorder.expected_samples = 1;
  ASSERT_TRUE(WriteFile(options_.recorder.corpus, "xx\n").ok());
  auto prediction = ModelRecorder::Record(options_.recorder);
  ASSERT_FALSE(prediction.ok());
  EXPECT_NE(prediction.status().message().find("sample 0"), std::string::npos);
  EXPECT_FALSE(std::filesystem::exists(options_.output));
  ExpectNoIntermediateFiles();
}

TEST_F(ModelRecorderTest, RejectsMalformedAndPrefixOnlyCheckpoints) {
  ASSERT_TRUE(
      WriteFile(options_.recorder.checkpoint / "weight_999.bin", "extra").ok());
  auto extra = ModelRecorder::Record(options_.recorder);
  EXPECT_FALSE(extra.ok());
  ASSERT_TRUE(
      std::filesystem::remove(options_.recorder.checkpoint / "weight_999.bin"));
  ASSERT_TRUE(
      WriteFile(options_.recorder.checkpoint / "weight_0.bin", "truncated")
          .ok());
  auto short_weight = ModelRecorder::Record(options_.recorder);
  EXPECT_FALSE(short_weight.ok());
  EXPECT_FALSE(std::filesystem::exists(options_.output));
  ExpectNoIntermediateFiles();
}

}  // namespace
}  // namespace pluto::llm::discretized::generator
