#include "src/llm/experiments/memorize_general_facts/discretized_model/generate_discretized_model.h"

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <string>

#include "gtest/gtest.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_core.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_io.h"

namespace pluto::llm::discretized::generator {
namespace {
namespace fs = std::filesystem;

absl::StatusOr<Json> Fixture() {
  Json header = {{"schema", 1},
                 {"width", 16},
                 {"layers", 1},
                 {"vocab_size", 3},
                 {"eos_token", 2},
                 {"prompt_tokens", 1},
                 {"vocabulary",
                  {{{"original_id", 0}, {"hex", "48656c6c6f"}},
                   {{"original_id", 1}, {"hex", "20776f726c64"}},
                   {{"original_id", 2}, {"hex", "3c454f533e"}}}}};
  Json boundaries = Json::array();
  for (int stage = 0; stage < 3; ++stage)
    boundaries.push_back({std::vector<int>(16, 16256 + stage * 2),
                          std::vector<int>(16, 16257 + stage * 2)});
  return BuildModel(header,
                    {{{"tokens", {0, 1}},
                      {"predictions", {1, 2}},
                      {"boundaries", boundaries}}},
                    1);
}

class GeneratorTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string path = testing::TempDir() + "/generator-driver.XXXXXX";
    ASSERT_NE(mkdtemp(path.data()), nullptr);
    directory_ = path;
    auto model = Fixture();
    ASSERT_TRUE(model.ok()) << model.status();
    model_ = *model;
    options_.output = directory_ / "generated";
    options_.expected_samples = 1;
    const char* root = std::getenv("TEST_SRCDIR");
    ASSERT_NE(root, nullptr);
    options_.clang_format_config = fs::path(root) / "_main/.clang-format";
  }
  void TearDown() override {
    std::error_code ignored;
    fs::remove_all(directory_, ignored);
  }
  fs::path directory_;
  Json model_;
  GeneratorOptions options_;
};

TEST_F(GeneratorTest, BothRepresentationsGenerateWithoutIntermediateFiles) {
  for (bool compact : {false, true}) {
    options_.output = directory_ / (compact ? "compact" : "tables");
    options_.compact_transitions = compact;
    options_.state_index = true;
    options_.reduce = compact;
    auto result = GenerateFromModel(model_, options_);
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_TRUE(EvaluateModel(*result).ok());
    EXPECT_TRUE(fs::exists(options_.output / "state_index.tsv"));
    EXPECT_EQ(fs::exists(options_.output / "state_members.tsv"), compact);
    EXPECT_EQ(fs::exists(options_.output / "transition_patterns.txt"), compact);
    EXPECT_EQ(fs::exists(options_.output / "generated_transition_test.cc"),
              compact);
    for (const auto& file : fs::recursive_directory_iterator(directory_)) {
      EXPECT_NE(file.path().extension(), ".json");
      EXPECT_NE(file.path().extension(), ".jsonl");
    }
    auto report = ReadFile(options_.output / "generation_report.txt");
    ASSERT_TRUE(report.ok());
    EXPECT_NE(report->find("errors: 0"), std::string::npos);
    EXPECT_NE(report->find("explicit_eos: 1"), std::string::npos);
    EXPECT_NE(report->find("targets: 2"), std::string::npos);
    auto bytes = ReadFile(options_.output / "model.cc");
    ASSERT_TRUE(bytes.ok());
    EXPECT_NE(bytes->find("const DiscreteModel& GeneratedModel()"),
              std::string::npos);
    EXPECT_NE(report->find(Sha256(*bytes)), std::string::npos);
  }
}

TEST_F(GeneratorTest, RepeatConversionIsDeterministic) {
  options_.compact_transitions = true;
  options_.reduce = true;
  ASSERT_TRUE(GenerateFromModel(model_, options_).ok());
  const auto first = options_.output;
  options_.output = directory_ / "repeat";
  ASSERT_TRUE(GenerateFromModel(model_, options_).ok());
  for (const auto& file : fs::directory_iterator(first)) {
    // Runtime measurements are deliberately not deterministic. Generated
    // source, domain fixtures, and all state/transition reports must be.
    if (file.path().filename() == "generation_report.txt")
      continue;
    auto expected = ReadFile(file.path());
    auto actual = ReadFile(options_.output / file.path().filename());
    ASSERT_TRUE(expected.ok());
    ASSERT_TRUE(actual.ok());
    EXPECT_EQ(*actual, *expected) << file.path();
  }
}

TEST_F(GeneratorTest, NeverOverwritesExistingPathsIncludingDanglingSymlinks) {
  for (int kind = 0; kind < 3; ++kind) {
    options_.output = directory_ / std::to_string(kind);
    if (kind == 0)
      fs::create_directory(options_.output);
    else if (kind == 1)
      ASSERT_TRUE(WriteFile(options_.output, "keep").ok());
    else
      fs::create_symlink(directory_ / "absent", options_.output);
    auto result = GenerateFromModel(model_, options_);
    EXPECT_EQ(result.status().code(), absl::StatusCode::kAlreadyExists);
  }
  EXPECT_EQ(*ReadFile(directory_ / "1"), "keep");
  EXPECT_TRUE(fs::is_symlink(directory_ / "2"));
}

TEST_F(GeneratorTest, MalformedInputsHaveNoOutput) {
  EXPECT_FALSE(GenerateFromModel(Json::object(), options_).ok());
  EXPECT_FALSE(fs::exists(options_.output));
  options_.expected_samples = 2;
  EXPECT_FALSE(GenerateFromModel(model_, options_).ok());
  EXPECT_FALSE(fs::exists(options_.output));
  options_.expected_samples = 1;
  model_["language_modeling_head"][0][1] = 2;
  EXPECT_FALSE(GenerateFromModel(model_, options_).ok());
  EXPECT_FALSE(fs::exists(options_.output));
}

TEST_F(GeneratorTest, FormattingUsesDeclaredConfigurationNotDestinationStyle) {
  ASSERT_TRUE(WriteFile(directory_ / ".clang-format",
                        "BasedOnStyle: LLVM\nPointerAlignment: Right\n")
                  .ok());
  ASSERT_TRUE(GenerateFromModel(model_, options_).ok());
  auto source = ReadFile(options_.output / "model.cc");
  ASSERT_TRUE(source.ok());
  EXPECT_NE(source->find("const DiscreteModel& GeneratedModel()"),
            std::string::npos);
}

TEST_F(GeneratorTest, MissingFormatterDoesNotPublishPartialSources) {
  const char* old = std::getenv("PATH");
  const std::string saved = old == nullptr ? "" : old;
  ASSERT_EQ(setenv("PATH", directory_.c_str(), 1), 0);
  auto result = GenerateFromModel(model_, options_);
  if (old != nullptr)
    ASSERT_EQ(setenv("PATH", saved.c_str(), 1), 0);
  else
    ASSERT_EQ(unsetenv("PATH"), 0);
  EXPECT_FALSE(result.ok());
  EXPECT_FALSE(fs::exists(options_.output));
}

TEST_F(GeneratorTest, MissingDeclaredStyleAndMissingCheckpointFail) {
  options_.clang_format_config = directory_ / "absent";
  EXPECT_FALSE(GenerateFromModel(model_, options_).ok());
  EXPECT_FALSE(fs::exists(options_.output));
  options_.clang_format_config =
      fs::path(std::getenv("TEST_SRCDIR")) / "_main/.clang-format";
  EXPECT_FALSE(Generate(options_).ok());
  EXPECT_FALSE(fs::exists(options_.output));
}

TEST_F(GeneratorTest, ReportsProgressAndIndependentCertificate) {
  options_.reduce = true;
  std::vector<Json> progress;
  options_.progress = [&](const Json& event) { progress.push_back(event); };
  auto result = GenerateFromModel(model_, options_);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_GE(progress.size(), 2u);
  EXPECT_EQ(progress.front()["phase"], "baseline");
  EXPECT_EQ(progress.back()["phase"], "generated");
  auto report = ReadFile(options_.output / "generation_report.txt");
  ASSERT_TRUE(report.ok());
  EXPECT_NE(report->find("irreducibility_certificate:"), std::string::npos);
  EXPECT_NE(report->find("global_minimum_proven: false"), std::string::npos);
}

TEST_F(GeneratorTest, InvalidReductionSettingsFailBeforeCapture) {
  options_.reduce = true;
  options_.reduction.max_passes = 0;
  EXPECT_NE(
      std::string(Generate(options_).status().message()).find("reduction"),
      std::string::npos);
  EXPECT_FALSE(fs::exists(options_.output));
}

}  // namespace
}  // namespace pluto::llm::discretized::generator
