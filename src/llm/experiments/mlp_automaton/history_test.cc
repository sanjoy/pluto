#include "src/llm/experiments/mlp_automaton/history.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/llm/experiments/mlp_automaton/graph.h"

namespace pluto::llm::mlp_automaton {
namespace {

class CheckpointDiscoveryTest : public testing::Test {
 protected:
  void SetUp() override {
    directory_ = std::filesystem::path(testing::TempDir()) /
                 testing::UnitTest::GetInstance()->current_test_info()->name();
    std::error_code error;
    ASSERT_TRUE(std::filesystem::create_directory(directory_, error))
        << error.message();
  }

  void TearDown() override {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
    EXPECT_FALSE(error) << error.message();
  }

  void Directory(const std::string& name) {
    std::error_code error;
    ASSERT_TRUE(std::filesystem::create_directory(directory_ / name, error))
        << error.message();
  }

  void File(const std::string& name) {
    std::ofstream output(directory_ / name);
    output << "not an actual tensor";
    ASSERT_TRUE(output.good());
  }

  std::filesystem::path directory_;
};

TEST_F(CheckpointDiscoveryTest, SortsNumericStepsAndIgnoresUnrelatedEntries) {
  for (const std::string& name :
       {"step_10", "step_2", "step_000", "unrelated", "step_", "step_-2",
        "step_+3", "step_3.bak", "step_9223372036854775807"})
    Directory(name);
  for (const std::string& name :
       {"step_4", "step_9223372036854775808", "step_1.tar.gz"})
    File(name);
  auto result = DiscoverCheckpoints(directory_);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->is_history);
  ASSERT_EQ(result->checkpoints.size(), 4);
  EXPECT_EQ(result->checkpoints[0].step, 0);
  EXPECT_EQ(result->checkpoints[0].directory, directory_ / "step_000");
  EXPECT_EQ(result->checkpoints[1].step, 2);
  EXPECT_EQ(result->checkpoints[2].step, 10);
  EXPECT_EQ(result->checkpoints[3].step, std::numeric_limits<int64_t>::max());
}

TEST_F(CheckpointDiscoveryTest, SingleCheckpointDoesNotRequireStepName) {
  File("weight_0.bin");
  Directory("step_2");
  Directory("step_0002");  // Irrelevant when the parent is itself a checkpoint.
  auto result = DiscoverCheckpoints(directory_);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_FALSE(result->is_history);
  ASSERT_EQ(result->checkpoints.size(), 1);
  EXPECT_EQ(result->checkpoints[0].directory, directory_);
  EXPECT_EQ(result->checkpoints[0].step, 0);
}

TEST_F(CheckpointDiscoveryTest, WeightNamedDirectoryIsNotATensor) {
  Directory("weight_0.bin");
  Directory("step_12");
  auto result = DiscoverCheckpoints(directory_);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->is_history);
  ASSERT_EQ(result->checkpoints.size(), 1);
  EXPECT_EQ(result->checkpoints[0].step, 12);
}

TEST_F(CheckpointDiscoveryTest, RejectsNumericAliases) {
  Directory("step_1");
  Directory("step_01");
  auto result = DiscoverCheckpoints(directory_);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(std::string(result.status().message()).find("Multiple"),
            std::string::npos);
}

TEST_F(CheckpointDiscoveryTest, RejectsOverflowingDirectorySteps) {
  Directory("step_9223372036854775808");
  EXPECT_EQ(DiscoverCheckpoints(directory_).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(CheckpointDiscoveryTest,
       RejectsMissingCheckpointsAndInvalidDirectories) {
  EXPECT_EQ(DiscoverCheckpoints(directory_).status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_FALSE(DiscoverCheckpoints(directory_ / "missing").ok());
  File("file");
  EXPECT_EQ(DiscoverCheckpoints(directory_ / "file").status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(CheckpointDiscoveryTest, ReportsFilesystemErrors) {
  // An ELOOP error is reproducible even when tests run as root; chmod-based
  // permission tests are not. Errors must not be treated as an empty history.
  std::error_code error;
  std::filesystem::create_directory_symlink("step_1", directory_ / "step_1",
                                            error);
  ASSERT_FALSE(error) << error.message();
  auto result = DiscoverCheckpoints(directory_);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInternal);
  EXPECT_NE(std::string(result.status().message()).find("step_1"),
            std::string::npos);
  EXPECT_FALSE(DiscoverCheckpoints(directory_ / "step_1").ok());
}

std::string HistoryText(const HistoryAccumulator& history) {
  std::ostringstream output;
  EXPECT_TRUE(WriteHistoryText(output, history.Finish()).ok());
  return output.str();
}

TEST(HistoryAccumulatorTest, CoalescesAnalyzedStepsNotNumericNeighbors) {
  HistoryAccumulator history;
  const std::vector<CombinedPath> paths = {{"Exeunt", {3, 5, 6}}};
  ASSERT_TRUE(history.AddCheckpoint(100, paths, paths).ok());
  ASSERT_TRUE(history.AddCheckpoint(300, paths, {}).ok());
  ASSERT_TRUE(history.AddCheckpoint(900, paths, {}).ok());
  EXPECT_EQ(HistoryText(history),
            "\"Exeunt\":\n  Chkpt 100 - 900 — block 3,5,6\n");
  const auto result = history.Finish();
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].ranges.size(), 1);
  EXPECT_EQ(result[0].ranges[0].first_step, 100);
  EXPECT_EQ(result[0].ranges[0].last_step, 900);
}

TEST(HistoryAccumulatorTest, AbsenceAndChangedBlocksBreakRanges) {
  HistoryAccumulator history;
  const std::vector<CombinedPath> first = {{"Exeunt", {3, 5, 6}}};
  const std::vector<CombinedPath> second = {{"Exeunt", {3, 5, 7}}};
  ASSERT_TRUE(history.AddCheckpoint(0, first, first).ok());
  ASSERT_TRUE(history.AddCheckpoint(100, first, {}).ok());
  ASSERT_TRUE(history.AddCheckpoint(200, second, {}).ok());
  ASSERT_TRUE(history.AddCheckpoint(300, second, {}).ok());
  ASSERT_TRUE(history.AddCheckpoint(400, {}, {}).ok());
  ASSERT_TRUE(history.AddCheckpoint(500, second, {}).ok());
  EXPECT_EQ(HistoryText(history),
            "\"Exeunt\":\n"
            "  Chkpt 0 - 100 — block 3,5,6\n"
            "  Chkpt 200 - 300 — block 3,5,7\n"
            "  Chkpt 500 — block 3,5,7\n");
}

TEST(HistoryAccumulatorTest, LaterSamplesRecoverEarlierUnsampledMembership) {
  HistoryAccumulator history;
  const std::vector<CombinedPath> first = {
      {"Exeunt", {0, 1}}, {"never sampled", {7}}, {"other", {2}}};
  const std::vector<CombinedPath> first_sample = {{"other", {2}}};
  ASSERT_TRUE(history.AddCheckpoint(10, first, first_sample).ok());
  EXPECT_EQ(HistoryText(history), "\"other\":\n  Chkpt 10 — block 2\n");
  const std::vector<CombinedPath> second = {{"Exeunt", {1, 2}}};
  ASSERT_TRUE(history.AddCheckpoint(20, second, {}).ok());
  ASSERT_TRUE(history.AddCheckpoint(30, {}, {}).ok());
  ASSERT_TRUE(history.AddCheckpoint(40, second, second).ok());
  EXPECT_EQ(HistoryText(history),
            "\"Exeunt\":\n"
            "  Chkpt 10 — block 0,1\n"
            "  Chkpt 20 — block 1,2\n"
            "  Chkpt 40 — block 1,2\n"
            "\"other\":\n"
            "  Chkpt 10 — block 2\n");
  // Finish neither consumes the state nor loses the candidate set.
  EXPECT_EQ(HistoryText(history), HistoryText(history));
}

TEST(HistoryAccumulatorTest, DistinguishesExactBytesAndSortsResults) {
  HistoryAccumulator history;
  const std::vector<CombinedPath> paths = {{"word", {2}},
                                           {" Word", {0}},
                                           {"Word", {1}},
                                           {std::string("\0\xff", 2), {3}}};
  ASSERT_TRUE(history.AddCheckpoint(0, paths, paths).ok());
  const auto result = history.Finish();
  ASSERT_EQ(result.size(), 4);
  EXPECT_EQ(result[0].bytes, std::string("\0\xff", 2));
  EXPECT_EQ(result[1].bytes, " Word");
  EXPECT_EQ(result[2].bytes, "Word");
  EXPECT_EQ(result[3].bytes, "word");
  EXPECT_EQ(result[0].ranges[0].mlp_blocks, (std::vector<int>{3}));
}

TEST(HistoryAccumulatorTest, RejectsInvalidStepsWithoutMutatingState) {
  HistoryAccumulator history;
  const std::vector<CombinedPath> paths = {{"word", {0}}};
  EXPECT_FALSE(history.AddCheckpoint(-1, paths, paths).ok());
  EXPECT_TRUE(history.Finish().empty());
  ASSERT_TRUE(history.AddCheckpoint(10, paths, paths).ok());
  const std::string initial = HistoryText(history);
  for (int64_t step : {-1, 0, 9, 10}) {
    EXPECT_EQ(history.AddCheckpoint(step, paths, paths).code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(HistoryText(history), initial);
  }
  ASSERT_TRUE(
      history.AddCheckpoint(std::numeric_limits<int64_t>::max(), paths, {})
          .ok());
  EXPECT_EQ(HistoryText(history),
            "\"word\":\n"
            "  Chkpt 10 - 9223372036854775807 — block 0\n");
}

TEST(HistoryAccumulatorTest, RejectsMalformedCompleteOrSamplePathsAtomically) {
  const std::vector<CombinedPath> valid = {{"word", {0, 2}}};
  const std::vector<std::vector<CombinedPath>> invalid = {
      {{"", {0}}},        {{"word", {}}},
      {{"word", {-1}}},   {{"word", {2, 0}}},
      {{"word", {0, 0}}}, {{"word", {0, 2}}, {"word", {0, 2}}}};
  for (const auto& paths : invalid) {
    HistoryAccumulator history;
    ASSERT_TRUE(history.AddCheckpoint(0, valid, valid).ok());
    const std::string initial = HistoryText(history);
    EXPECT_FALSE(history.AddCheckpoint(10, paths, {}).ok());
    EXPECT_EQ(HistoryText(history), initial);
    EXPECT_FALSE(history.AddCheckpoint(10, valid, paths).ok());
    EXPECT_EQ(HistoryText(history), initial);
    ASSERT_TRUE(history.AddCheckpoint(10, valid, valid).ok());
    EXPECT_EQ(HistoryText(history), "\"word\":\n  Chkpt 0 - 10 — block 0,2\n");
  }
}

TEST(HistoryAccumulatorTest, SamplesMustMatchFullMembershipBeforeAnyUpdate) {
  const std::vector<CombinedPath> complete = {{"word", {0, 2}}, {"other", {3}}};
  const std::vector<std::vector<CombinedPath>> invalid = {
      {{"absent", {0}}},
      {{"word", {0}}},
      {{"word", {0, 2, 3}}},
      {{"word", {0, 2}}, {"absent", {0}}}};
  for (const auto& sample : invalid) {
    HistoryAccumulator history;
    ASSERT_TRUE(history.AddCheckpoint(0, complete, {}).ok());
    EXPECT_FALSE(history.AddCheckpoint(10, complete, sample).ok());
    EXPECT_TRUE(history.Finish().empty());
    // A rejected checkpoint does not consume the step or set earlier sample
    // bits from the valid prefix of a malformed sample list.
    ASSERT_TRUE(history.AddCheckpoint(10, {}, {}).ok());
    ASSERT_TRUE(history.AddCheckpoint(20, complete, complete).ok());
    EXPECT_EQ(HistoryText(history),
              "\"other\":\n  Chkpt 0 — block 3\n  Chkpt 20 — block 3\n"
              "\"word\":\n  Chkpt 0 — block 0,2\n  Chkpt 20 — block 0,2\n");
  }
}

class CommaGrouping final : public std::numpunct<char> {
 protected:
  char do_thousands_sep() const override { return ','; }
  std::string do_grouping() const override { return "\3"; }
};

TEST(HistoryOutputTest, TextEscapesArbitraryBytesAndIgnoresNumberFormatting) {
  const std::vector<PathHistory> paths = {
      {std::string("\0\xff\"\\\n", 5), {{1000, 2000, {0, 12}}}}};
  std::ostringstream output;
  output.imbue(std::locale(std::locale::classic(), new CommaGrouping));
  output << std::hex << std::showbase;
  ASSERT_TRUE(WriteHistoryText(output, paths).ok());
  EXPECT_EQ(output.str(),
            "\"\\000\\377\\\"\\\\\\n\":\n"
            "  Chkpt 1000 - 2000 — block 0,12\n");
}

TEST(HistoryOutputTest, JsonIsLosslessAndIncludesActualAnalyzedSteps) {
  const std::vector<PathHistory> paths = {
      {std::string("\0\xff\"\\\n", 5), {{1000, 2000, {0, 12}}}}};
  const std::vector<int64_t> steps = {0, 1000, 1500, 2000};
  std::ostringstream output;
  output.imbue(std::locale(std::locale::classic(), new CommaGrouping));
  output << std::hex << std::showbase;
  ASSERT_TRUE(WriteHistoryJson(output, paths, steps).ok());
  const std::string json = output.str();
  EXPECT_NE(json.find("\"format\": \"pluto.mlp_automaton.history.v1\""),
            std::string::npos);
  EXPECT_NE(json.find("\"range_semantics\": "
                      "\"consecutive_analyzed_checkpoints\""),
            std::string::npos);
  EXPECT_NE(json.find("\"analyzed_steps\": [0, 1000, 1500, 2000]"),
            std::string::npos);
  EXPECT_NE(json.find("\"bytes_hex\": \"00ff225c0a\""), std::string::npos);
  EXPECT_NE(json.find("\"bytes_escaped\": "
                      "\"\\\\000\\\\377\\\\\\\"\\\\\\\\\\\\n\""),
            std::string::npos);
  EXPECT_NE(json.find("\"first_step\": 1000, \"last_step\": 2000, "
                      "\"mlp_blocks\": [0, 12]"),
            std::string::npos);
}

TEST(HistoryOutputTest, EmptyOutputAndFailedStreams) {
  HistoryAccumulator history;
  EXPECT_TRUE(history.Finish().empty());
  std::ostringstream text;
  ASSERT_TRUE(WriteHistoryText(text, {}).ok());
  EXPECT_TRUE(text.str().empty());
  std::ostringstream json;
  ASSERT_TRUE(WriteHistoryJson(json, {}, {}).ok());
  EXPECT_NE(json.str().find("\"analyzed_steps\": []"), std::string::npos);
  EXPECT_NE(json.str().find("\"paths\": [\n  ]"), std::string::npos);
  for (bool use_json : {false, true}) {
    std::ostringstream failed;
    failed.setstate(std::ios::badbit);
    EXPECT_EQ((use_json ? WriteHistoryJson(failed, {}, {})
                        : WriteHistoryText(failed, {}))
                  .code(),
              absl::StatusCode::kDataLoss);
  }
}

TEST(HistoryOutputTest, RejectsInvalidHistoriesBeforeWriting) {
  const std::vector<std::vector<PathHistory>> invalid = {
      {{"", {{0, 0, {0}}}}},
      {{"word", {}}},
      {{"word", {{-1, 0, {0}}}}},
      {{"word", {{1, 0, {0}}}}},
      {{"word", {{0, 0, {}}}}},
      {{"word", {{0, 0, {-1}}}}},
      {{"word", {{0, 0, {1, 0}}}}},
      {{"word", {{0, 0, {0, 0}}}}},
      {{"word", {{0, 10, {0}}, {10, 20, {1}}}}},
      {{"word", {{10, 10, {0}}, {0, 0, {1}}}}},
      {{"word", {{0, 0, {0}}}}, {"word", {{10, 10, {1}}}}}};
  const std::vector<int64_t> steps = {0, 10, 20};
  for (const auto& paths : invalid) {
    std::ostringstream text;
    EXPECT_FALSE(WriteHistoryText(text, paths).ok());
    EXPECT_TRUE(text.str().empty());
    std::ostringstream json;
    EXPECT_FALSE(WriteHistoryJson(json, paths, steps).ok());
    EXPECT_TRUE(json.str().empty());
  }
}

TEST(HistoryOutputTest, RejectsInvalidStepListsAndUnanalyzedEndpoints) {
  const std::vector<PathHistory> paths = {{"word", {{10, 20, {0}}}}};
  for (const std::vector<int64_t>& steps : {std::vector<int64_t>{},
                                            {10},
                                            {20},
                                            {10, 15},
                                            {15, 20},
                                            {-1, 10, 20},
                                            {10, 10, 20},
                                            {20, 10}}) {
    std::ostringstream output;
    EXPECT_FALSE(WriteHistoryJson(output, paths, steps).ok());
    EXPECT_TRUE(output.str().empty());
  }
}

TEST(HistoryAccumulatorTest, WwHistoryRequiresRealEdgesAtEachCheckpoint) {
  // The tokenizer knows "ww" before training. Only w -> w is evidence that a
  // block learned a continuation, even if another checkpoint first samples
  // that continuation much later.
  Graph untrained;
  untrained.token_bytes = {"w", "ww"};
  Graph learned = untrained;
  learned.edges = {{0, 0, 0.9}};
  auto diagnostic = Walk(untrained, 1, 16);
  auto continuation = Walk(learned, 0, 16);
  ASSERT_TRUE(diagnostic.ok()) << diagnostic.status();
  ASSERT_TRUE(continuation.ok()) << continuation.status();

  for (bool earlier_edge : {false, true}) {
    SCOPED_TRACE(earlier_edge);
    const std::vector<BlockPaths> early_blocks{
        {0, untrained, {*diagnostic}},
        {5, earlier_edge ? learned : untrained, {}}};
    auto early_complete = CollectAllPaths(early_blocks, 16);
    auto early_sampled = CombinePaths(early_blocks, 16);
    ASSERT_TRUE(early_complete.ok()) << early_complete.status();
    ASSERT_TRUE(early_sampled.ok()) << early_sampled.status();
    // Even an explicit singleton start does not become a sampled candidate.
    EXPECT_TRUE(early_sampled->empty());
    EXPECT_EQ(early_complete->size(), earlier_edge ? 1u : 0u);
    HistoryAccumulator history;
    ASSERT_TRUE(history.AddCheckpoint(0, *early_complete, *early_sampled).ok());
    EXPECT_TRUE(history.Finish().empty());

    const std::vector<BlockPaths> late_blocks{{0, untrained, {}},
                                              {5, learned, {*continuation}}};
    auto late_complete = CollectAllPaths(late_blocks, 16);
    auto late_sampled = CombinePaths(late_blocks, 16);
    ASSERT_TRUE(late_complete.ok()) << late_complete.status();
    ASSERT_TRUE(late_sampled.ok()) << late_sampled.status();
    ASSERT_EQ(late_complete->size(), 1u);
    ASSERT_EQ(late_sampled->size(), 1u);
    EXPECT_EQ(late_sampled->front().mlp_blocks, (std::vector<int>{5}));
    ASSERT_TRUE(history.AddCheckpoint(100, *late_complete, *late_sampled).ok());
    EXPECT_EQ(HistoryText(history), earlier_edge
                                        ? "\"ww\":\n  Chkpt 0 - 100 — block 5\n"
                                        : "\"ww\":\n  Chkpt 100 — block 5\n");
  }
}

TEST(HistoryAccumulatorTest, TokenLimitOneCannotCreateContinuationHistory) {
  Graph learned;
  learned.token_bytes = {"w", "ww"};
  learned.edges = {{0, 0, 0.9}};
  auto samples = SamplePaths(learned, 10, 1, 17);
  ASSERT_TRUE(samples.ok()) << samples.status();
  ASSERT_EQ(samples->size(), 1u);
  const BlockPaths block{0, learned, *samples};
  auto complete = CollectAllPaths({&block, 1}, 1);
  auto combined = CombinePaths({&block, 1}, 1);
  ASSERT_TRUE(complete.ok()) << complete.status();
  ASSERT_TRUE(combined.ok()) << combined.status();
  EXPECT_TRUE(complete->empty());
  EXPECT_TRUE(combined->empty());
  HistoryAccumulator history;
  ASSERT_TRUE(history.AddCheckpoint(0, *complete, *combined).ok());
  ASSERT_TRUE(history.AddCheckpoint(100, *complete, *combined).ok());
  EXPECT_TRUE(history.Finish().empty());
}

}  // namespace
}  // namespace pluto::llm::mlp_automaton
