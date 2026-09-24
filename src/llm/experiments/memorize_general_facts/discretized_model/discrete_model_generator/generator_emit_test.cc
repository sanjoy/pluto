#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/code_generator.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/discretize_transition_tests.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/utils.h"

namespace pluto::llm::discretized::generator {
namespace {

CapturedModel Fixture() {
  CapturedModel model;
  model.metadata = {
      .width = 2,
      .layers = 1,
      .vocab_size = 4,
      .eos_token = 3,
      .prompt_tokens = 1,
      .vocabulary = {{10, "A"}, {11, "B"}, {12, "C"}, {13, "<eos>"}}};
  model.samples = {{{0, 1}}, {{1}}};
  model.position_embedding.transitions = {
      {0, 0, 4}, {1, 0, 7}, {1, 1, 5}, {2, 1, 6}};
  model.transformers.resize(1);
  model.transformers[0].attention.transitions = {
      {{4}, 8}, {{4, 5}, 9}, {{4, 6}, 10}, {{7}, 11}};
  model.transformers[0].mlp.transitions = {
      {8, 12}, {9, 13}, {10, 14}, {11, 15}};
  model.language_modeling_head.transitions = {
      {12, 1}, {13, 3}, {14, 0}, {15, 3}};
  for (int i = 4; i < 16; ++i)
    model.states.push_back({.id = i, .boundary = (i - 4) / 4});
  return model;
}

class EmitTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string name = (::testing::TempDir() + "discrete-emit-XXXXXX");
    ASSERT_NE(::mkdtemp(name.data()), nullptr);
    directory_ = name;
  }
  void TearDown() override {
    std::error_code ignored;
    std::filesystem::remove_all(directory_, ignored);
  }
  std::filesystem::path directory_;
};

TEST(TokenNames, PreserveCaseWhitespaceAndEos) {
  EXPECT_EQ(TokenName(" France", 123, 999), "kSpace_France_123");
  EXPECT_EQ(TokenName("France", 123, 999), "kFrance_123");
  EXPECT_EQ(TokenName("The", 456, 999), "kThe_456");
  EXPECT_EQ(TokenName("the", 456, 999), "kthe_456");
  EXPECT_EQ(TokenName("anything", 4474, 4474), "kEos_4474");
  EXPECT_EQ(TokenName("__reserved", 1, 999),
            "kUnderscore_Underscore_reserved_1");
}

TEST(TokenNames, EveryByteAndInjectionRemainLegalIdentifiers) {
  for (int byte = 0; byte < 256; ++byte) {
    const std::string name =
        TokenName(std::string(1, static_cast<char>(byte)), byte, -1);
    ASSERT_FALSE(name.empty());
    EXPECT_EQ(name.front(), 'k');
    EXPECT_EQ(name.find("__"), std::string::npos);
    EXPECT_TRUE(std::all_of(name.begin(), name.end(), [](char c) {
      return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
             (c >= '0' && c <= '9') || c == '_';
    })) << name;
  }
  const std::string hostile = "*/\n#error injected\n//\"; namespace bad {";
  EXPECT_EQ(TokenName(hostile, 0, 9).find('\n'), std::string::npos);
  EXPECT_NE(TokenName(std::string(200, 'A'), 0, 9),
            TokenName(std::string(200, 'A'), 1, 9));
}

TEST(TokenLiterals, BytesCannotExtendEscapesOrInjectSource) {
  EXPECT_EQ(Literal("\"\\\t\n\r"), "\"\\\"\\\\\\t\\n\\r\"");
  EXPECT_EQ(Literal(std::string("\x01"
                                "a",
                                2)),
            "\"\\001a\"");
  EXPECT_EQ(Literal(std::string("\0", 1)), "\"\\000\"");
  EXPECT_EQ(Literal(std::string("\xff", 1)), "\"\\377\"");
}

TEST_F(EmitTest, PlainSourcesExposeOnlyModelFactoryAndSeparateFixtures) {
  auto result = RenderModel(Fixture(), true, false);
  ASSERT_TRUE(result.ok()) << result.status();
  const auto& files = *result;
  EXPECT_EQ(files.count("model.h"), 1u);
  EXPECT_NE(files.at("model.h").find("namespace pluto::llm::discretized::gen"),
            std::string::npos);
  EXPECT_NE(files.at("model.h").find("GeneratedModel()"), std::string::npos);
  EXPECT_EQ(files.at("model.h").find("GeneratedAttention"), std::string::npos);
  EXPECT_NE(files.at("tables.h").find("gen::internal"), std::string::npos);
  EXPECT_NE(files.at("BUILD.bazel").find("hdrs = [\"model.h\"]"),
            std::string::npos);
  EXPECT_NE(files.at("BUILD.bazel").find("linkstatic = True"),
            std::string::npos);
  EXPECT_EQ(files.at("model.cc").find("ExpectedTokens"), std::string::npos);
  EXPECT_NE(files.at("verification.cc").find("kExpectedTokens"),
            std::string::npos);
  EXPECT_NE(files.at("state_index.tsv").find("empirical_prefix_examples"),
            std::string::npos);
  for (const auto& [name, ignored] : files)
    EXPECT_NE(std::filesystem::path(name).extension(), ".json");
  EXPECT_TRUE(PublishFiles(files, directory_ / "output").ok());
  EXPECT_TRUE(std::filesystem::exists(directory_ / "output" / "model.h"));
}

TEST_F(EmitTest, TablePermutationDoesNotChangeOutput) {
  const CapturedModel model = Fixture();
  CapturedModel permuted = model;
  std::reverse(permuted.position_embedding.transitions.begin(),
               permuted.position_embedding.transitions.end());
  std::reverse(permuted.language_modeling_head.transitions.begin(),
               permuted.language_modeling_head.transitions.end());
  std::reverse(permuted.states.begin(), permuted.states.end());
  std::reverse(permuted.transformers[0].attention.transitions.begin(),
               permuted.transformers[0].attention.transitions.end());
  std::reverse(permuted.transformers[0].mlp.transitions.begin(),
               permuted.transformers[0].mlp.transitions.end());
  auto a = RenderModel(model, true), b = RenderModel(permuted, true);
  ASSERT_TRUE(a.ok()) << a.status();
  ASSERT_TRUE(b.ok()) << b.status();
  EXPECT_EQ(*a, *b);
}

TEST_F(EmitTest, CompactModeAddsIndependentBoundaryTests) {
  auto result = RenderModel(Fixture(), true, true);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->count("generated_transition_test.cc"), 1u);
  EXPECT_EQ(result->count("transition_patterns.txt"), 0u);
  EXPECT_NE(result->at("generated_transition_test.cc").find("kExpectedStates"),
            std::string::npos);
  EXPECT_EQ(result->at("attention_0.cc").find("kExpectedStates"),
            std::string::npos);
}

TEST_F(EmitTest, EmitsConfiguredContextAndBoundaryProbes) {
  auto model = Fixture();
  model.metadata.context_length = 32;
  auto result = RenderModel(model, false, true);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_NE(result->at("model.cc").find("DiscreteModel model{32, 1,"),
            std::string::npos);
  EXPECT_NE(result->at("generated_transition_test.cc")
                .find("{vocab::kA_0, 32, {std::nullopt}}"),
            std::string::npos);
}

TEST_F(EmitTest, ExistingDirectoriesFilesAndDanglingSymlinksAreNotClobbered) {
  ASSERT_TRUE(EmitModel(Fixture(), directory_ / "output").ok());
  auto before = ReadFile(directory_ / "output" / "model.h");
  ASSERT_TRUE(before.ok());
  EXPECT_EQ(EmitModel(Fixture(), directory_ / "output").code(),
            absl::StatusCode::kAlreadyExists);
  auto after = ReadFile(directory_ / "output" / "model.h");
  ASSERT_TRUE(after.ok());
  EXPECT_EQ(*after, *before);
  std::filesystem::create_directory(directory_ / "empty");
  EXPECT_EQ(EmitModel(Fixture(), directory_ / "empty").code(),
            absl::StatusCode::kAlreadyExists);
  ASSERT_TRUE(WriteFile(directory_ / "file", "keep").ok());
  EXPECT_EQ(EmitModel(Fixture(), directory_ / "file").code(),
            absl::StatusCode::kAlreadyExists);
  std::filesystem::create_symlink(directory_ / "missing",
                                  directory_ / "symlink");
  EXPECT_EQ(EmitModel(Fixture(), directory_ / "symlink").code(),
            absl::StatusCode::kAlreadyExists);
  EXPECT_TRUE(std::filesystem::is_symlink(directory_ / "symlink"));
}

TEST_F(EmitTest, PublishRejectsPathTraversalAndCleansStagingDirectory) {
  EXPECT_FALSE(
      PublishFiles({{"../outside", "bad"}}, directory_ / "output").ok());
  EXPECT_FALSE(std::filesystem::exists(directory_ / "output"));
  EXPECT_FALSE(std::filesystem::exists(directory_ / "outside"));
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(directory_),
                          std::filesystem::directory_iterator()),
            0);
  EXPECT_FALSE(PublishFiles({}, directory_ / "..").ok());
  EXPECT_FALSE(PublishFiles({}, directory_ / "missing" / "output").ok());
}

TEST_F(EmitTest, ConcurrentPublishHasExactlyOneCompleteWinner) {
  const FileMap first{{"first.cc", "first"}, {"complete.h", "first"}};
  const FileMap second{{"second.cc", "second"}, {"complete.h", "second"}};
  absl::Status a, b;
  std::thread writer_a([&] { a = PublishFiles(first, directory_ / "output"); });
  std::thread writer_b(
      [&] { b = PublishFiles(second, directory_ / "output"); });
  writer_a.join();
  writer_b.join();
  EXPECT_NE(a.ok(), b.ok());
  EXPECT_EQ((a.ok() ? b : a).code(), absl::StatusCode::kAlreadyExists);
  const std::string winner = a.ok() ? "first" : "second";
  auto contents = ReadFile(directory_ / "output" / "complete.h");
  ASSERT_TRUE(contents.ok()) << contents.status();
  EXPECT_EQ(*contents, winner);
  EXPECT_TRUE(
      std::filesystem::exists(directory_ / "output" / (winner + ".cc")));
  EXPECT_EQ(
      std::distance(std::filesystem::directory_iterator(directory_ / "output"),
                    std::filesystem::directory_iterator()),
      2);
}

TEST_F(EmitTest, InvalidModelCannotLeaveOutput) {
  CapturedModel model = Fixture();
  model.position_embedding.transitions.push_back(
      model.position_embedding.transitions[0]);
  EXPECT_FALSE(EmitModel(model, directory_ / "output").ok());
  EXPECT_FALSE(std::filesystem::exists(directory_ / "output"));
  EXPECT_FALSE(RenderModel(CapturedModel{}).ok());
  model = Fixture();
  model.metadata.vocabulary[0].bytes.clear();
  EXPECT_FALSE(RenderModel(model).ok());
  model = Fixture();
  model.states[0].id = -1;
  EXPECT_FALSE(RenderModel(model).ok());
}

TEST_F(EmitTest, StateMembershipIsInspectionOnly) {
  CapturedModel model = Fixture();
  for (auto& row : model.states)
    row.members = std::vector<int>{row.id};
  auto result = RenderModel(model, true);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_NE(result->at("state_members.tsv")
                .find("4\ttoken_plus_position_embedding\t[4]"),
            std::string::npos);
  EXPECT_EQ(result->at("BUILD.bazel").find("state_members"), std::string::npos);
}

TEST(TransitionFixtures, ValidateSourceIndependentlyAndPreserveZeroToken) {
  const std::vector<std::string> names = {"vocab::kA_0", "vocab::kB_1",
                                          "vocab::kC_2", "vocab::kEos_3"};
  auto result = RenderTransitionTest(Fixture(), names);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_NE(result->find("static_cast<DiscreteHiddenState>(vocab::kA_0)"),
            std::string::npos);
  EXPECT_NE(result->find("std::nullopt"), std::string::npos);
  EXPECT_NE(result->find("2147483647"), std::string::npos);
  EXPECT_NE(result->find("-2147483648"), std::string::npos);
  CapturedModel wrong = Fixture();
  wrong.language_modeling_head.transitions[0].output = 0;
  EXPECT_FALSE(RenderTransitionTest(wrong, names).ok());
  CapturedModel missing = Fixture();
  missing.transformers[0].attention.transitions.erase(
      missing.transformers[0].attention.transitions.begin());
  EXPECT_FALSE(RenderTransitionTest(missing, names).ok());
  EXPECT_FALSE(RenderTransitionTest(Fixture(), {"only_one"}).ok());
}

TEST(TransitionFixtures, SupportsModelsWithoutTransformerBlocks) {
  CapturedModel model = Fixture();
  model.metadata.layers = 0;
  model.transformers.clear();
  model.states.erase(model.states.begin() + 4, model.states.end());
  model.language_modeling_head.transitions = {{4, 1}, {5, 3}, {6, 0}, {7, 3}};
  auto plain = RenderModel(model);
  ASSERT_TRUE(plain.ok()) << plain.status();
  EXPECT_EQ(plain->at("model.cc").find("kTransformers"), std::string::npos);
  auto compact = RenderModel(model, false, true);
  ASSERT_TRUE(compact.ok()) << compact.status();
  EXPECT_NE(compact->at("generated_transition_test.cc").find("kLayers = 0"),
            std::string::npos);
}

TEST_F(EmitTest, ArbitraryTokenBytesRemainEscapedInSourceAndInspectionFields) {
  auto model = Fixture();
  model.metadata.vocabulary[0].bytes = std::string("A\0\t\n\xff", 5);
  auto result = RenderModel(model, true);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_NE(result->at("vocabulary.cc").find("A\\000\\t\\n\\377"),
            std::string::npos);
  const auto& index = result->at("state_index.tsv");
  EXPECT_NE(index.find("tokens=[0]; text=\"A\\000\\t\\n\\377\""),
            std::string::npos);
  EXPECT_EQ(index.find('\0'), std::string::npos);
  EXPECT_EQ(index.find(static_cast<char>(0xff)), std::string::npos);
}

}  // namespace
}  // namespace pluto::llm::discretized::generator
