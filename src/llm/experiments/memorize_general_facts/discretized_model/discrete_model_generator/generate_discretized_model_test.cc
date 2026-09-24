#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/generate_discretized_model.h"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

#include "gtest/gtest.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model_util.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/generator_test_util.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/utils.h"

namespace pluto::llm::discretized::generator {
namespace {
namespace fs = std::filesystem;

class GeneratorTest : public GeneratorTestBase {};

TEST_F(GeneratorTest, CompactionIsIndependentOfTransitionRepresentation) {
  for (bool compaction : {false, true}) {
    for (bool compact : {false, true}) {
      options_.output =
          directory_ / (std::string(compaction ? "compacted_" : "exact_") +
                        (compact ? "compact_code" : "tables"));
      options_.compact_transitions = compact;
      options_.state_index = true;
      options_.compaction = compaction;
      auto result = Generate(options_);
      ASSERT_TRUE(result.ok()) << result.status();
      EXPECT_TRUE(EvaluateModel(*result).ok());
      EXPECT_EQ(result->stats.compaction_search.has_value(), compaction);
      EXPECT_TRUE(fs::exists(options_.output / "state_index.tsv"));
      EXPECT_TRUE(fs::exists(options_.output / "state_members.tsv"));
      EXPECT_TRUE(fs::exists(options_.output / "state_vectors.cc"));
      EXPECT_TRUE(fs::exists(options_.output / "state_vectors_test.cc"));
      for (const auto& state : result->states)
        ASSERT_TRUE(state.members.has_value());
      EXPECT_FALSE(fs::exists(options_.output / "transition_patterns.txt"));
      EXPECT_EQ(fs::exists(options_.output / "generated_transition_test.cc"),
                compact);
      ExpectNoIntermediateFiles();
      auto report = ReadFile(options_.output / "generation_report.txt");
      ASSERT_TRUE(report.ok());
      EXPECT_NE(report->find("errors: 0"), std::string::npos);
      EXPECT_NE(report->find("explicit_eos: 1"), std::string::npos);
      EXPECT_NE(report->find("targets: 1"), std::string::npos);
      auto bytes = ReadFile(options_.output / "model.cc");
      ASSERT_TRUE(bytes.ok());
      EXPECT_NE(bytes->find("const DiscreteModel& GeneratedModel()"),
                std::string::npos);
      EXPECT_NE(bytes->find("internal::PrintState"), std::string::npos);
      EXPECT_NE(report->find(Sha256(*bytes)), std::string::npos);
      for (const auto& file : fs::directory_iterator(options_.output)) {
        if (file.path().filename() == "generation_report.txt")
          continue;
        auto digest = Sha256File(file.path());
        ASSERT_TRUE(digest.ok()) << digest.status();
        EXPECT_NE(report->find(*digest), std::string::npos) << file.path();
      }
      auto weights_digest =
          Sha256File(options_.recorder.checkpoint / "weight_0.bin");
      ASSERT_TRUE(weights_digest.ok()) << weights_digest.status();
      EXPECT_NE(report->find(*weights_digest), std::string::npos);
    }
  }
}

TEST_F(GeneratorTest, RepeatConversionIsDeterministic) {
  options_.compact_transitions = true;
  options_.compaction = true;
  ASSERT_TRUE(Generate(options_).ok());
  const auto first = options_.output;
  options_.output = directory_ / "repeat";
  ASSERT_TRUE(Generate(options_).ok());
  for (const auto& file : fs::directory_iterator(first)) {
    // Runtime measurements are deliberately not deterministic. Generated
    // source, domain fixtures, and all state reports must be.
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
    auto result = Generate(options_);
    EXPECT_EQ(result.status().code(), absl::StatusCode::kAlreadyExists);
  }
  EXPECT_EQ(*ReadFile(directory_ / "1"), "keep");
  EXPECT_TRUE(fs::is_symlink(directory_ / "2"));
}

TEST_F(GeneratorTest, MalformedInputsHaveNoOutput) {
  const auto corpus = options_.recorder.corpus;
  options_.recorder.corpus = directory_ / "absent_corpus";
  EXPECT_FALSE(Generate(options_).ok());
  EXPECT_FALSE(fs::exists(options_.output));
  options_.recorder.corpus = corpus;
  options_.recorder.expected_samples = 2;
  EXPECT_FALSE(Generate(options_).ok());
  EXPECT_FALSE(fs::exists(options_.output));
  options_.recorder.expected_samples = 1;
  // Zero weights always select EOS; a second corpus token makes this an
  // incorrect completion, which must fail before publishing generated code.
  ASSERT_TRUE(WriteFile(options_.recorder.corpus, "xx\n").ok());
  EXPECT_FALSE(Generate(options_).ok());
  EXPECT_FALSE(fs::exists(options_.output));
}

TEST_F(GeneratorTest, FormattingUsesDeclaredConfigurationNotDestinationStyle) {
  ASSERT_TRUE(WriteFile(directory_ / ".clang-format",
                        "BasedOnStyle: LLVM\nPointerAlignment: Right\n")
                  .ok());
  ASSERT_TRUE(Generate(options_).ok());
  auto source = ReadFile(options_.output / "model.cc");
  ASSERT_TRUE(source.ok());
  EXPECT_NE(source->find("const DiscreteModel& GeneratedModel()"),
            std::string::npos);
}

TEST_F(GeneratorTest, MissingFormatterDoesNotPublishPartialSources) {
  const char* old = std::getenv("PATH");
  const std::string saved = old == nullptr ? "" : old;
  ASSERT_EQ(setenv("PATH", directory_.c_str(), 1), 0);
  auto result = Generate(options_);
  if (old != nullptr)
    ASSERT_EQ(setenv("PATH", saved.c_str(), 1), 0);
  else
    ASSERT_EQ(unsetenv("PATH"), 0);
  EXPECT_FALSE(result.ok());
  EXPECT_FALSE(fs::exists(options_.output));
}

TEST_F(GeneratorTest, MissingDeclaredStyleAndMissingCheckpointFail) {
  const auto style = options_.clang_format_config;
  options_.clang_format_config = directory_ / "absent";
  EXPECT_FALSE(Generate(options_).ok());
  EXPECT_FALSE(fs::exists(options_.output));
  options_.clang_format_config = style;
  options_.recorder.checkpoint = directory_ / "missing_checkpoint";
  EXPECT_FALSE(Generate(options_).ok());
  EXPECT_FALSE(fs::exists(options_.output));
}

TEST_F(GeneratorTest, ReportsProgressAndIndependentCertificate) {
  options_.compaction = true;
  std::vector<ProgressEvent> progress;
  options_.progress = [&](const ProgressEvent& event) {
    progress.push_back(event);
  };
  auto result = Generate(options_);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_GE(progress.size(), 3u);
  ASSERT_TRUE(std::holds_alternative<CaptureProgress>(progress.front()));
  EXPECT_EQ(std::get<CaptureProgress>(progress.front()).samples, 1);
  bool saw_baseline = false;
  bool saw_compaction = false;
  for (const auto& event : progress) {
    if (const auto* phase = std::get_if<GenerationProgress>(&event))
      saw_baseline |= phase->phase == GenerationPhase::kBaseline;
    saw_compaction |= std::holds_alternative<CompactionProgress>(event);
  }
  EXPECT_TRUE(saw_baseline);
  EXPECT_TRUE(saw_compaction);
  ASSERT_TRUE(std::holds_alternative<GenerationProgress>(progress.back()));
  const auto& generated = std::get<GenerationProgress>(progress.back());
  EXPECT_EQ(generated.phase, GenerationPhase::kGenerated);
  EXPECT_EQ(generated.verification.errors, 0);
  EXPECT_EQ(generated.verification.explicit_eos, 1);
  auto report = ReadFile(options_.output / "generation_report.txt");
  ASSERT_TRUE(report.ok());
  EXPECT_NE(report->find("compaction_certificate:"), std::string::npos);
  EXPECT_NE(report->find("global_minimum_proven: false"), std::string::npos);
}

TEST_F(GeneratorTest, InvalidCompactionSettingsFailBeforeCapture) {
  options_.compaction = true;
  options_.compaction_options.max_passes = 0;
  EXPECT_NE(
      std::string(Generate(options_).status().message()).find("compaction"),
      std::string::npos);
  EXPECT_FALSE(fs::exists(options_.output));
}

TEST_F(GeneratorTest, PreservesStageCallbacksWithAndWithoutGenericProgress) {
  options_.compaction = true;
  for (bool generic_progress : {false, true}) {
    options_.output = directory_ / (generic_progress ? "both" : "stage_only");
    std::vector<CaptureProgress> capture;
    std::vector<CaptureProgress> forwarded_capture;
    std::vector<CompactionProgress> compaction;
    std::vector<CompactionProgress> forwarded_compaction;
    options_.recorder.progress = [&](const CaptureProgress& progress) {
      capture.push_back(progress);
    };
    options_.compaction_options.progress =
        [&](const CompactionProgress& progress) {
          compaction.push_back(progress);
        };
    options_.progress = nullptr;
    if (generic_progress)
      options_.progress = [&](const ProgressEvent& progress) {
        if (const auto* report = std::get_if<CaptureProgress>(&progress))
          forwarded_capture.push_back(*report);
        if (const auto* report = std::get_if<CompactionProgress>(&progress))
          forwarded_compaction.push_back(*report);
      };
    auto result = Generate(options_);
    ASSERT_TRUE(result.ok()) << result.status();
    ASSERT_EQ(capture.size(), 1u);
    EXPECT_EQ(capture.front().samples, 1);
    ASSERT_FALSE(compaction.empty());
    if (generic_progress) {
      ASSERT_EQ(forwarded_capture.size(), capture.size());
      EXPECT_EQ(forwarded_capture.front().samples, capture.front().samples);
      EXPECT_EQ(forwarded_capture.front().total_samples,
                capture.front().total_samples);
      EXPECT_EQ(forwarded_capture.front().rows, capture.front().rows);
      EXPECT_EQ(forwarded_capture.front().greedy_verified,
                capture.front().greedy_verified);
      EXPECT_EQ(forwarded_compaction, compaction);
    } else {
      EXPECT_TRUE(forwarded_capture.empty());
      EXPECT_TRUE(forwarded_compaction.empty());
    }
  }
}

}  // namespace
}  // namespace pluto::llm::discretized::generator
