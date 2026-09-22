#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_emit.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_transition_tests.h"

namespace pluto::llm::discretized::generator {
namespace {

Json Fixture() {
  Json model = {
      {"schema", 1},
      {"width", 2},
      {"layers", 1},
      {"vocab_size", 4},
      {"eos_token", 3},
      {"prompt_tokens", 1},
      {"vocabulary",
       {{{"original_id", 10}, {"hex", "41"}},
        {{"original_id", 11}, {"hex", "42"}},
        {{"original_id", 12}, {"hex", "43"}},
        {{"original_id", 13}, {"hex", "3c656f733e"}}}},
      {"samples", {{{"tokens", {0, 1}}}, {{"tokens", {1}}}}},
      {"states", Json::array()},
      {"entry", {{0, 0, 4}, {1, 0, 7}, {1, 1, 5}, {2, 1, 6}}},
      {"attention", {{{{4}, 8}, {{4, 5}, 9}, {{4, 6}, 10}, {{7}, 11}}}},
      {"mlp", {{{8, 12}, {9, 13}, {10, 14}, {11, 15}}}},
      {"language_modeling_head", {{12, 1}, {13, 3}, {14, 0}, {15, 3}}},
      {"stats", Json::object()}};
  for (int i = 4; i < 16; ++i)
    model["states"].push_back(
        {{"id", i}, {"stage", (i - 4) / 4}, {"bits", {i, 0}}});
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
  EXPECT_EQ(files.count("model.h"), 1);
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
  EXPECT_NE(files.at("state_index.tsv").find("empirical_prefix_examples_json"),
            std::string::npos);
  for (const auto& [name, ignored] : files)
    EXPECT_NE(std::filesystem::path(name).extension(), ".json");
  EXPECT_TRUE(PublishFiles(files, directory_ / "output").ok());
  EXPECT_TRUE(std::filesystem::exists(directory_ / "output" / "model.h"));
}

TEST_F(EmitTest, TablePermutationDoesNotChangeOutput) {
  const Json model = Fixture();
  Json permuted = model;
  for (const char* name : {"entry", "language_modeling_head", "states"})
    std::reverse(permuted[name].begin(), permuted[name].end());
  std::reverse(permuted["attention"][0].begin(),
               permuted["attention"][0].end());
  std::reverse(permuted["mlp"][0].begin(), permuted["mlp"][0].end());
  auto a = RenderModel(model, true), b = RenderModel(permuted, true);
  ASSERT_TRUE(a.ok()) << a.status();
  ASSERT_TRUE(b.ok()) << b.status();
  EXPECT_EQ(*a, *b);
}

TEST_F(EmitTest, CompactModeAddsIndependentBoundaryTests) {
  auto result = RenderModel(Fixture(), true, true);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->count("generated_transition_test.cc"), 1);
  EXPECT_EQ(result->count("transition_patterns.txt"), 1);
  EXPECT_EQ(result->count("transition_patterns.json"), 0);
  EXPECT_NE(result->at("generated_transition_test.cc").find("kExpectedStates"),
            std::string::npos);
  EXPECT_EQ(result->at("attention_0.cc").find("kExpectedStates"),
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
  Json model = Fixture();
  model["entry"].push_back(model["entry"][0]);
  EXPECT_FALSE(EmitModel(model, directory_ / "output").ok());
  EXPECT_FALSE(std::filesystem::exists(directory_ / "output"));
  EXPECT_FALSE(RenderModel(Json::array()).ok());
  model = Fixture();
  model["vocabulary"][0]["hex"] = "FF";
  EXPECT_FALSE(RenderModel(model).ok());
  model = Fixture();
  model["states"][0]["id"] = -1;
  EXPECT_FALSE(RenderModel(model).ok());
}

TEST_F(EmitTest, StateMembershipIsInspectionOnly) {
  Json model = Fixture();
  for (auto& row : model["states"]) {
    row["members"] = Json::array({row["id"]});
    row["member_count"] = 1;
  }
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
  Json wrong = Fixture();
  wrong["language_modeling_head"][0][1] = 0;
  EXPECT_FALSE(RenderTransitionTest(wrong, names).ok());
  Json missing = Fixture();
  missing["attention"][0].erase(missing["attention"][0].begin());
  EXPECT_FALSE(RenderTransitionTest(missing, names).ok());
  EXPECT_FALSE(RenderTransitionTest(Fixture(), {"only_one"}).ok());
}

TEST(TransitionFixtures, SupportsModelsWithoutTransformerBlocks) {
  Json model = Fixture();
  model["layers"] = 0;
  model["attention"] = Json::array();
  model["mlp"] = Json::array();
  model["states"].erase(model["states"].begin() + 4, model["states"].end());
  model["language_modeling_head"] = {{4, 1}, {5, 3}, {6, 0}, {7, 3}};
  auto plain = RenderModel(model);
  ASSERT_TRUE(plain.ok()) << plain.status();
  EXPECT_EQ(plain->at("model.cc").find("kTransformers"), std::string::npos);
  auto compact = RenderModel(model, false, true);
  ASSERT_TRUE(compact.ok()) << compact.status();
  EXPECT_NE(compact->at("generated_transition_test.cc").find("kLayers = 0"),
            std::string::npos);
}

}  // namespace
}  // namespace pluto::llm::discretized::generator
