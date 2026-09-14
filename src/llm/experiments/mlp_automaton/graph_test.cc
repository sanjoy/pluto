#include "src/llm/experiments/mlp_automaton/graph.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace pluto::llm::mlp_automaton {
namespace {

Graph ExampleGraph() {
  return {.threshold = 0.75,
          .token_bytes = {" Ex", "e", "unt", " ", "loop", "<eos>", "orphan"},
          .edges = {{0, 1, 0.99}, {1, 2, 0.95}, {3, 3, 0.9}, {4, 0, 0.8}},
          .eos_token_id = 5};
}

TEST(MlpAutomatonGraphTest, ValidatesThresholdAndStrictBoundary) {
  Graph graph = ExampleGraph();
  EXPECT_TRUE(ValidateGraph(graph).ok());
  for (double invalid :
       {-1.0, 0.49, 1.0, 2.0, std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    graph.threshold = invalid;
    EXPECT_EQ(ValidateGraph(graph).code(), absl::StatusCode::kInvalidArgument);
  }
  graph.threshold = 0.75;
  graph.edges[0].probability = 0.75;
  EXPECT_FALSE(ValidateGraph(graph).ok());
  graph.edges[0].probability = std::nextafter(0.75, 1.0);
  EXPECT_TRUE(ValidateGraph(graph).ok());
  graph.threshold = 0.5;
  graph.edges[0].probability = std::nextafter(0.5, 1.0);
  EXPECT_TRUE(ValidateGraph(graph).ok());
  graph.edges[0].probability = 0.5;
  EXPECT_FALSE(ValidateGraph(graph).ok());
}

TEST(MlpAutomatonGraphTest, RejectsInvalidProbabilities) {
  for (double invalid :
       {-1.0, 0.0, 0.749, 1.01, std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    Graph graph = ExampleGraph();
    graph.edges[0].probability = invalid;
    EXPECT_FALSE(ValidateGraph(graph).ok());
  }
  Graph graph = ExampleGraph();
  graph.edges[0].probability = 1.0;
  EXPECT_TRUE(ValidateGraph(graph).ok());
}

TEST(MlpAutomatonGraphTest, RejectsInvalidIdsAndMultipleOutgoingEdges) {
  for (int invalid : {-1, 7, std::numeric_limits<int>::max()}) {
    Graph graph = ExampleGraph();
    graph.edges[0].source = invalid;
    EXPECT_FALSE(ValidateGraph(graph).ok());
    graph = ExampleGraph();
    graph.edges[0].target = invalid;
    EXPECT_FALSE(ValidateGraph(graph).ok());
  }
  Graph graph = ExampleGraph();
  graph.eos_token_id = -2;
  EXPECT_FALSE(ValidateGraph(graph).ok());
  graph.eos_token_id = 7;
  EXPECT_FALSE(ValidateGraph(graph).ok());
  graph.eos_token_id = -1;
  EXPECT_TRUE(ValidateGraph(graph).ok());
  graph.edges.push_back({0, 2, 0.8});
  EXPECT_FALSE(ValidateGraph(graph).ok());
  graph = ExampleGraph();
  graph.edges.push_back(graph.edges[0]);
  EXPECT_FALSE(ValidateGraph(graph).ok());
  graph.token_bytes.clear();
  EXPECT_FALSE(ValidateGraph(graph).ok());
}

TEST(MlpAutomatonGraphTest, WalkCompletesExeuntAndHandlesIsolatedNodes) {
  const Graph graph = ExampleGraph();
  auto path = Walk(graph, 0, 16);
  ASSERT_TRUE(path.ok()) << path.status();
  EXPECT_EQ(path->tokens, (std::vector<int>{0, 1, 2}));
  EXPECT_EQ(path->termination, Termination::kNoEdge);
  auto isolated = Walk(graph, 6, 16);
  ASSERT_TRUE(isolated.ok()) << isolated.status();
  EXPECT_EQ(isolated->tokens, (std::vector<int>{6}));
  EXPECT_EQ(isolated->termination, Termination::kNoEdge);
}

TEST(MlpAutomatonGraphTest, DetectsSelfLoopsAndLongerCycles) {
  Graph graph = ExampleGraph();
  auto path = Walk(graph, 3, 1000000);
  ASSERT_TRUE(path.ok()) << path.status();
  EXPECT_EQ(path->tokens, (std::vector<int>{3, 3}));
  EXPECT_EQ(path->termination, Termination::kCycle);
  graph.edges.push_back({2, 0, 0.9});
  path = Walk(graph, 4, std::numeric_limits<size_t>::max());
  ASSERT_TRUE(path.ok()) << path.status();
  EXPECT_EQ(path->tokens, (std::vector<int>{4, 0, 1, 2, 0}));
  EXPECT_EQ(path->termination, Termination::kCycle);
}

TEST(MlpAutomatonGraphTest, StopsAtEosEvenWhenItHasAnOutgoingEdge) {
  Graph graph = ExampleGraph();
  graph.edges.push_back({2, 5, 0.9});
  graph.edges.push_back({5, 0, 0.9});
  auto path = Walk(graph, 0, 16);
  ASSERT_TRUE(path.ok()) << path.status();
  EXPECT_EQ(path->tokens, (std::vector<int>{0, 1, 2, 5}));
  EXPECT_EQ(path->termination, Termination::kEndOfSequence);
  path = Walk(graph, 5, 1);
  ASSERT_TRUE(path.ok()) << path.status();
  EXPECT_EQ(path->tokens, (std::vector<int>{5}));
  EXPECT_EQ(path->termination, Termination::kEndOfSequence);
  auto sampled = SamplePaths(graph, 100, 16, 17);
  ASSERT_TRUE(sampled.ok()) << sampled.status();
  ASSERT_EQ(sampled->size(), 5u);
  for (const Path& result : *sampled)
    EXPECT_NE(result.tokens[0], 5);
}

TEST(MlpAutomatonGraphTest, TokenLimitIncludesTheStartingToken) {
  const Graph graph = ExampleGraph();
  for (size_t limit : {1, 2}) {
    auto path = Walk(graph, 0, limit);
    ASSERT_TRUE(path.ok()) << path.status();
    EXPECT_EQ(path->tokens.size(), limit);
    EXPECT_EQ(path->termination, Termination::kTokenLimit);
  }
  auto path = Walk(graph, 0, 3);
  ASSERT_TRUE(path.ok()) << path.status();
  EXPECT_EQ(path->termination, Termination::kNoEdge);
  EXPECT_FALSE(Walk(graph, 0, 0).ok());
  EXPECT_FALSE(Walk(graph, -1, 1).ok());
  EXPECT_FALSE(Walk(graph, 7, 1).ok());
  EXPECT_FALSE(SamplePaths(graph, 0, 0, 17).ok());
}

TEST(MlpAutomatonGraphTest, SamplingIsSeededDistinctAndEdgeOrderIndependent) {
  Graph graph = ExampleGraph();
  auto first = SamplePaths(graph, 100, 16, 17);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_EQ(first->size(), 4u);
  std::reverse(graph.edges.begin(), graph.edges.end());
  auto second = SamplePaths(graph, 100, 16, 17);
  ASSERT_TRUE(second.ok()) << second.status();
  std::set<int> starts;
  for (size_t i = 0; i < first->size(); ++i) {
    EXPECT_EQ((*first)[i].tokens, (*second)[i].tokens);
    EXPECT_EQ((*first)[i].termination, (*second)[i].termination);
    starts.insert((*first)[i].tokens.front());
  }
  EXPECT_EQ(starts, (std::set<int>{0, 1, 3, 4}));
  auto fewer = SamplePaths(graph, 2, 16, 17);
  ASSERT_TRUE(fewer.ok()) << fewer.status();
  ASSERT_EQ(fewer->size(), 2u);
  EXPECT_EQ((*fewer)[0].tokens, (*first)[0].tokens);
  EXPECT_EQ((*fewer)[1].tokens, (*first)[1].tokens);
  auto empty = SamplePaths(graph, 0, 16, 17);
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_TRUE(empty->empty());
  graph.edges.clear();
  empty = SamplePaths(graph, 100, 16, 17);
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_TRUE(empty->empty());
}

TEST(MlpAutomatonGraphTest, CorpusFilterMatchesWholePathsAndPreservesOrder) {
  const Graph graph = ExampleGraph();
  auto candidates = SamplePaths(graph, 100, 16, 17);
  ASSERT_TRUE(candidates.ok()) << candidates.status();
  // The individual pieces "loop" and " Exeunt" both occur; their concatenated
  // path "loop Exeunt" does not. Explicit isolated starts use the same filter.
  auto isolated = Walk(graph, 6, 16);
  ASSERT_TRUE(isolated.ok()) << isolated.status();
  candidates->push_back(*isolated);
  const std::string training = "loop\n Exeunt all;  orphan";
  auto matches = FilterPathsInCorpus(graph, *candidates, training);
  ASSERT_TRUE(matches.ok()) << matches.status();
  ASSERT_EQ(matches->size(), 4u);
  size_t matched = 0;
  for (const auto& path : *candidates) {
    if (path.tokens.front() == 4)
      continue;
    EXPECT_EQ((*matches)[matched].tokens, path.tokens);
    EXPECT_EQ((*matches)[matched].termination, path.termination);
    ++matched;
  }
  EXPECT_EQ(graph.edges.size(), 4u);
}

TEST(MlpAutomatonGraphTest, CorpusFilterUsesExactBytesAndTrainingBoundary) {
  Graph graph;
  graph.token_bytes = {
      " Ex", "eunt", "elsewhere", "A", std::string("\0\xff", 2),
      "",    "abc",  "def"};
  graph.edges = {{0, 1, 0.9}, {3, 4, 0.9}, {6, 7, 0.9}};
  const std::vector<Path> paths = {{{0, 1}, Termination::kNoEdge},
                                   {{2}, Termination::kNoEdge},
                                   {{3, 4}, Termination::kNoEdge},
                                   {{5}, Termination::kNoEdge},
                                   {{6, 7}, Termination::kNoEdge}};
  const std::string training =
      std::string(" Exeunt A") + std::string("\0\xff", 2) + " abc";
  const std::string corpus = training + "def elsewhere";
  auto matches = FilterPathsInCorpus(
      graph, paths, absl::string_view(corpus).substr(0, training.size()));
  ASSERT_TRUE(matches.ok()) << matches.status();
  ASSERT_EQ(matches->size(), 2u);
  EXPECT_EQ((*matches)[0].tokens, paths[0].tokens);
  EXPECT_EQ((*matches)[1].tokens, paths[2].tokens);
  // Neither a case fold nor stripping the leading space is allowed.
  for (absl::string_view text : {" exeunt", "Exeunt", ""}) {
    auto absent = FilterPathsInCorpus(graph, paths, text);
    ASSERT_TRUE(absent.ok()) << absent.status();
    EXPECT_TRUE(absent->empty());
  }
  auto all = FilterPathsInCorpus(graph, paths, corpus);
  ASSERT_TRUE(all.ok()) << all.status();
  EXPECT_EQ(all->size(), 4u);  // Empty decoded text never counts as a match.
}

TEST(MlpAutomatonGraphTest, CorpusFilterRejectsInvalidPathsEvenWithEmptyText) {
  const Graph graph = ExampleGraph();
  for (const Path path :
       {Path{{}, Termination::kNoEdge}, Path{{-1}, Termination::kNoEdge},
        Path{{7}, Termination::kNoEdge},
        Path{{0}, static_cast<Termination>(99)}}) {
    const auto result = FilterPathsInCorpus(graph, {&path, 1}, "");
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  }
  Graph invalid = graph;
  invalid.edges.push_back({0, 1, 0.9});
  EXPECT_FALSE(FilterPathsInCorpus(invalid, {}, "").ok());
}

TEST(MlpAutomatonGraphTest,
     GraphJsonIncludesIsolatesAndLosslessArbitraryBytes) {
  Graph graph;
  graph.token_bytes = {" Ex", "e", std::string("\x00\xff\"\\\n", 5)};
  graph.edges = {{0, 1, 0.875}};
  std::ostringstream output;
  ASSERT_TRUE(WriteGraphJson(output, graph).ok());
  EXPECT_EQ(
      output.str(),
      "{\n"
      "  \"format\": \"pluto.mlp_automaton.v1\",\n"
      "  \"threshold\": 0.75,\n"
      "  \"comparison\": \"strictly_greater\",\n"
      "  \"eos_token_id\": -1,\n"
      "  \"nodes\": [\n"
      "    {\"id\": 0, \"bytes_hex\": \"204578\", \"bytes_escaped\": \" "
      "Ex\"},\n"
      "    {\"id\": 1, \"bytes_hex\": \"65\", \"bytes_escaped\": \"e\"},\n"
      "    {\"id\": 2, \"bytes_hex\": \"00ff225c0a\", \"bytes_escaped\": "
      "\"\\\\x00\\\\xff\\\"\\\\\\\\\\\\n\"}\n"
      "  ],\n"
      "  \"edges\": [\n"
      "    {\"source\": 0, \"target\": 1, \"probability\": 0.875}\n"
      "  ]\n"
      "}\n");
}

TEST(MlpAutomatonGraphTest, PathsJsonContainsBytesIdsAndTermination) {
  const Graph graph = ExampleGraph();
  auto path = Walk(graph, 0, 16);
  ASSERT_TRUE(path.ok()) << path.status();
  std::vector<Path> paths = {*path};
  std::ostringstream output;
  ASSERT_TRUE(WritePathsJson(output, graph, paths).ok());
  EXPECT_NE(output.str().find("\"tokens\": [0, 1, 2]"), std::string::npos);
  EXPECT_NE(output.str().find("\"termination\": \"no_edge\""),
            std::string::npos);
  EXPECT_NE(output.str().find("\"bytes_hex\": \"20457865756e74\""),
            std::string::npos);
  EXPECT_NE(output.str().find("\"bytes_escaped\": \" Exeunt\""),
            std::string::npos);
  for (Termination termination :
       {Termination::kNoEdge, Termination::kCycle, Termination::kEndOfSequence,
        Termination::kTokenLimit}) {
    EXPECT_NE(TerminationName(termination), "invalid");
  }
  paths[0].tokens.push_back(7);
  std::ostringstream invalid_output;
  EXPECT_FALSE(WritePathsJson(invalid_output, graph, paths).ok());
  EXPECT_TRUE(invalid_output.str().empty());
  paths[0].tokens.clear();
  EXPECT_FALSE(WritePathsJson(invalid_output, graph, paths).ok());
}

class CommaDecimal final : public std::numpunct<char> {
 protected:
  char do_decimal_point() const override { return ','; }
  char do_thousands_sep() const override { return '_'; }
  std::string do_grouping() const override { return "\1"; }
};

TEST(MlpAutomatonGraphTest, JsonNumbersIgnoreCallerFormattingAndLocale) {
  Graph graph = ExampleGraph();
  graph.edges[0].probability = std::nextafter(0.75, 1.0);
  std::ostringstream output;
  output.imbue(std::locale(std::locale::classic(), new CommaDecimal));
  output << std::hex << std::fixed << std::setprecision(1);
  ASSERT_TRUE(WriteGraphJson(output, graph).ok());
  EXPECT_NE(output.str().find("0.75000000000000011"), std::string::npos);
  EXPECT_EQ(output.str().find("0,75"), std::string::npos);
}

TEST(MlpAutomatonGraphTest, JsonPropagatesInvalidGraphAndStreamFailures) {
  Graph graph = ExampleGraph();
  std::ostringstream output;
  output.setstate(std::ios::badbit);
  EXPECT_EQ(WriteGraphJson(output, graph).code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(WritePathsJson(output, graph, {}).code(),
            absl::StatusCode::kDataLoss);
  graph.edges.push_back({0, 1, 0.8});
  std::ostringstream invalid_output;
  EXPECT_FALSE(WriteGraphJson(invalid_output, graph).ok());
  EXPECT_TRUE(invalid_output.str().empty());
  EXPECT_FALSE(SamplePaths(graph, 1, 16, 17).ok());
  EXPECT_FALSE(Walk(graph, 0, 16).ok());
}

TEST(MlpAutomatonGraphTest,
     CombinedPathsFindUnsampledBlocksAndDeduplicateText) {
  Graph first;
  first.token_bytes = {" Ex",   "eunt", " all",  "other",
                       " text", " two", " words"};
  first.edges = {{0, 1, 0.9}, {1, 2, 0.9}, {3, 4, 0.9}, {5, 6, 0.9}};
  Graph second;
  second.token_bytes = {" Exeunt", " all", "other text", " two words"};
  second.edges = {{0, 1, 0.9}};
  Graph incomplete = first;
  incomplete.edges = {{0, 1, 0.9}};  // Can't finish " Exeunt all".

  auto exeunt = Walk(first, 0, 16);
  auto other = Walk(first, 3, 16);
  auto other_tokenization = Walk(second, 2, 16);
  ASSERT_TRUE(exeunt.ok()) << exeunt.status();
  ASSERT_TRUE(other.ok()) << other.status();
  ASSERT_TRUE(other_tokenization.ok()) << other_tokenization.status();
  std::vector<BlockPaths> blocks{{5, first, {*exeunt, *other, *exeunt}},
                                 {1, second, {*other_tokenization}},
                                 {3, incomplete, {}}};
  auto combined = CombinePaths(blocks, 16);
  ASSERT_TRUE(combined.ok()) << combined.status();
  ASSERT_EQ(combined->size(), 2u);
  EXPECT_EQ((*combined)[0].bytes, " Exeunt all");
  EXPECT_EQ((*combined)[0].mlp_blocks, (std::vector<int>{1, 5}));
  EXPECT_EQ((*combined)[1].bytes, "other text");
  EXPECT_EQ((*combined)[1].mlp_blocks, (std::vector<int>{1, 5}));
  // B1 did not sample Exeunt, and its spelling uses a different tokenization.
  // " two words" is present in both graphs but was never a sampled candidate.
  std::ostringstream original;
  ASSERT_TRUE(WriteCombinedPathsJson(original, *combined).ok());
  std::reverse(blocks.begin(), blocks.end());
  for (auto& block : blocks)
    std::reverse(block.paths.begin(), block.paths.end());
  auto reordered = CombinePaths(blocks, 16);
  ASSERT_TRUE(reordered.ok()) << reordered.status();
  std::ostringstream output;
  ASSERT_TRUE(WriteCombinedPathsJson(output, *reordered).ok());
  EXPECT_EQ(output.str(), original.str());
}

TEST(MlpAutomatonGraphTest, CombinedPathsPreserveExactArbitraryBytes) {
  Graph first;
  first.token_bytes = {std::string("\0\xff", 2), "Word", " Word", "word"};
  BlockPaths sampled{0, first, {}};
  for (int token = 0; token < 4; ++token) {
    auto path = Walk(first, token, 16);
    ASSERT_TRUE(path.ok()) << path.status();
    sampled.paths.push_back(*path);
  }
  Graph second;
  second.token_bytes = {std::string("\0", 1), "\xff", " Word", "Word"};
  second.edges = {{0, 1, 0.9}, {3, 2, 0.9}};
  const std::vector<BlockPaths> blocks{sampled, {7, second, {}}};
  auto combined = CombinePaths(blocks, 16);
  ASSERT_TRUE(combined.ok()) << combined.status();
  ASSERT_EQ(combined->size(), 4u);
  EXPECT_EQ((*combined)[0].bytes, std::string("\0\xff", 2));
  EXPECT_EQ((*combined)[0].mlp_blocks, (std::vector<int>{0, 7}));
  EXPECT_EQ((*combined)[1].bytes, " Word");
  EXPECT_EQ((*combined)[1].mlp_blocks, (std::vector<int>{0, 7}));
  EXPECT_EQ((*combined)[2].bytes, "Word");
  EXPECT_EQ((*combined)[2].mlp_blocks, (std::vector<int>{0}));
  EXPECT_EQ((*combined)[3].bytes, "word");
  EXPECT_EQ((*combined)[3].mlp_blocks, (std::vector<int>{0}));
}

TEST(MlpAutomatonGraphTest, CombinedMembershipRespectsCycleEosAndTokenLimit) {
  Graph cycle;
  cycle.token_bytes = {"a", "b", "!"};
  cycle.edges = {{0, 1, 0.9}, {1, 0, 0.9}};
  Graph longer = cycle;
  longer.edges = {{0, 1, 0.9}, {1, 2, 0.9}};
  Graph eos = cycle;
  eos.eos_token_id = 1;
  auto short_path = Walk(cycle, 0, 2);
  auto cycle_path = Walk(cycle, 0, 3);
  ASSERT_TRUE(short_path.ok()) << short_path.status();
  ASSERT_TRUE(cycle_path.ok()) << cycle_path.status();
  EXPECT_EQ(short_path->termination, Termination::kTokenLimit);
  EXPECT_EQ(cycle_path->termination, Termination::kCycle);
  std::vector<BlockPaths> blocks{
      {0, cycle, {*short_path}}, {1, longer, {}}, {2, eos, {}}};
  auto short_result = CombinePaths(blocks, 2);
  ASSERT_TRUE(short_result.ok()) << short_result.status();
  ASSERT_EQ(short_result->size(), 1u);
  EXPECT_EQ(short_result->front().bytes, "ab");
  EXPECT_EQ(short_result->front().mlp_blocks, (std::vector<int>{0, 1, 2}));
  // Same text may terminate differently across blocks. But a prefix of a
  // longer full walk is not sufficient under the larger token limit.
  blocks[0].paths = {*cycle_path};
  auto cycle_result = CombinePaths(blocks, 3);
  ASSERT_TRUE(cycle_result.ok()) << cycle_result.status();
  ASSERT_EQ(cycle_result->size(), 1u);
  EXPECT_EQ(cycle_result->front().bytes, "aba");
  EXPECT_EQ(cycle_result->front().mlp_blocks, (std::vector<int>{0}));
}

TEST(MlpAutomatonGraphTest,
     CombinedPathsRejectInvalidInputsAndHandleNoSamples) {
  auto empty = CombinePaths({}, 16);
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_TRUE(empty->empty());
  EXPECT_FALSE(CombinePaths({}, 0).ok());
  Graph graph = ExampleGraph();
  std::vector<BlockPaths> blocks{{0, graph, {}}};
  empty = CombinePaths(blocks, 16);
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_TRUE(empty->empty());
  blocks.push_back(blocks[0]);
  EXPECT_FALSE(CombinePaths(blocks, 16).ok());
  blocks.pop_back();
  blocks[0].mlp_block = -1;
  EXPECT_FALSE(CombinePaths(blocks, 16).ok());
  blocks[0].mlp_block = 0;
  for (const Path& invalid :
       {Path{{}, Termination::kNoEdge}, Path{{-1}, Termination::kNoEdge},
        Path{{0, 99}, Termination::kNoEdge}, Path{{0}, Termination::kNoEdge},
        Path{{0, 1, 2}, Termination::kCycle}}) {
    blocks[0].paths = {invalid};
    EXPECT_EQ(CombinePaths(blocks, 16).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
  blocks[0].paths.clear();
  blocks[0].graph.edges.push_back({0, 1, 0.9});
  EXPECT_FALSE(CombinePaths(blocks, 16).ok());
}

TEST(MlpAutomatonGraphTest, CombinedJsonIsLosslessAndChecksProvenance) {
  const std::vector<CombinedPath> paths{
      {std::string("\0\xff\"\\\n", 5), {0, 2, 7}}};
  std::ostringstream output;
  output.imbue(std::locale(std::locale::classic(), new CommaDecimal));
  output << std::hex << std::showbase;
  ASSERT_TRUE(WriteCombinedPathsJson(output, paths).ok());
  EXPECT_NE(output.str().find("pluto.mlp_automaton.combined_paths.v1"),
            std::string::npos);
  EXPECT_NE(output.str().find("\"bytes_hex\": \"00ff225c0a\""),
            std::string::npos);
  EXPECT_NE(output.str().find("\"mlp_blocks\": [0, 2, 7]"), std::string::npos);
  for (const CombinedPath& invalid :
       {CombinedPath{"", {0}}, CombinedPath{"text", {}},
        CombinedPath{"text", {-1}}, CombinedPath{"text", {1, 1}},
        CombinedPath{"text", {2, 0}}}) {
    std::ostringstream rejected;
    EXPECT_FALSE(WriteCombinedPathsJson(rejected, {&invalid, 1}).ok());
    EXPECT_TRUE(rejected.str().empty());
  }
  const std::vector<CombinedPath> duplicates{paths[0], paths[0]};
  std::ostringstream rejected;
  EXPECT_FALSE(WriteCombinedPathsJson(rejected, duplicates).ok());
  EXPECT_TRUE(rejected.str().empty());
  std::ostringstream empty;
  ASSERT_TRUE(WriteCombinedPathsJson(empty, {}).ok());
  EXPECT_NE(empty.str().find("\"paths\": [\n  ]"), std::string::npos);
  std::ostringstream broken;
  broken.setstate(std::ios::badbit);
  EXPECT_EQ(WriteCombinedPathsJson(broken, paths).code(),
            absl::StatusCode::kDataLoss);
}

// Deliberately use the public, fresh-scratch Walk once per token and an ordered
// set of (bytes, block) pairs as an oracle. This does not share the collector's
// hash index, generation stamps, or deduplication bookkeeping.
void ExpectAllWalks(absl::Span<const BlockPaths> blocks, size_t max_tokens) {
  SCOPED_TRACE(max_tokens);
  std::set<std::pair<std::string, int>> expected_pairs;
  for (const auto& block : blocks) {
    for (size_t start = 0; start < block.graph.token_bytes.size(); ++start) {
      auto path = Walk(block.graph, static_cast<int>(start), max_tokens);
      ASSERT_TRUE(path.ok()) << path.status();
      std::string bytes;
      for (int token : path->tokens)
        bytes += block.graph.token_bytes[token];
      if (!bytes.empty())
        expected_pairs.emplace(bytes, block.mlp_block);
    }
  }
  std::vector<CombinedPath> expected;
  for (const auto& [bytes, block] : expected_pairs) {
    if (expected.empty() || expected.back().bytes != bytes)
      expected.push_back({bytes, {}});
    expected.back().mlp_blocks.push_back(block);
  }
  auto actual = CollectAllPaths(blocks, max_tokens);
  ASSERT_TRUE(actual.ok()) << actual.status();
  ASSERT_EQ(actual->size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ((*actual)[i].bytes, expected[i].bytes);
    EXPECT_EQ((*actual)[i].mlp_blocks, expected[i].mlp_blocks);
  }
}

TEST(MlpAutomatonGraphTest,
     CollectAllWalksMergesTokenizationsAndIgnoresSamplesAndOrdering) {
  Graph first;
  first.token_bytes = {
      " Ex",      "eunt", " all", "", " Word", "Word", std::string("\0\xff", 2),
      "eunt all", "<eos>"};
  first.edges = {
      {0, 1, 0.9}, {1, 2, 0.9}, {3, 4, 0.9}, {5, 4, 0.9}, {8, 0, 0.9}};
  first.eos_token_id = 8;
  Graph second;
  second.token_bytes = {" Exeunt", " all",     "Word", std::string("\0\xff", 2),
                        "",        "eunt all", "<eos>"};
  second.edges = {{0, 1, 0.9}, {4, 5, 0.9}};
  second.eos_token_id = 6;
  // Even malformed samples are ignored; collection is about the full graph,
  // not whether a random sampler happened to visit a spelling in this block.
  std::vector<BlockPaths> blocks{
      {7, first, {{{-1}, Termination::kCycle}, {{}, Termination::kNoEdge}}},
      {2, second, {}}};
  ExpectAllWalks(blocks, 16);
  auto collected = CollectAllPaths(blocks, 16);
  ASSERT_TRUE(collected.ok()) << collected.status();
  auto find = [&](absl::string_view text) {
    return std::find_if(collected->begin(), collected->end(),
                        [&](const auto& path) { return path.bytes == text; });
  };
  auto exeunt = find(" Exeunt all");
  ASSERT_NE(exeunt, collected->end());
  EXPECT_EQ(exeunt->mlp_blocks, (std::vector<int>{2, 7}));
  auto suffix = find("eunt all");
  ASSERT_NE(suffix, collected->end());
  EXPECT_EQ(suffix->mlp_blocks, (std::vector<int>{2, 7}));
  auto eos = find("<eos>");
  ASSERT_NE(eos, collected->end());
  EXPECT_EQ(eos->mlp_blocks, (std::vector<int>{2, 7}));
  auto word = find("Word");
  ASSERT_NE(word, collected->end());
  EXPECT_EQ(word->mlp_blocks, (std::vector<int>{2}));

  std::ostringstream original;
  ASSERT_TRUE(WriteCombinedPathsJson(original, *collected).ok());
  std::reverse(blocks.begin(), blocks.end());
  for (auto& block : blocks) {
    std::reverse(block.graph.edges.begin(), block.graph.edges.end());
    std::reverse(block.paths.begin(), block.paths.end());
  }
  ExpectAllWalks(blocks, 16);
  auto reordered = CollectAllPaths(blocks, 16);
  ASSERT_TRUE(reordered.ok()) << reordered.status();
  std::ostringstream output;
  ASSERT_TRUE(WriteCombinedPathsJson(output, *reordered).ok());
  EXPECT_EQ(output.str(), original.str());
}

TEST(MlpAutomatonGraphTest, CollectAllWalksMatchesExhaustiveSmallGraphs) {
  // All 4^3 successor assignments on three nodes exercise merging branches,
  // self-loops, longer cycles, isolated nodes, empty pieces, and duplicate
  // decoded pieces. Repeating with EOS and short limits stresses the stopping
  // rules and reuse of visited stamps between different starting tokens.
  for (int assignment = 0; assignment < 64; ++assignment) {
    for (int eos : {-1, 0, 2}) {
      Graph graph;
      graph.token_bytes = {"", "a", "a"};
      graph.eos_token_id = eos;
      int choices = assignment;
      for (int start = 0; start < 3; ++start) {
        const int successor = choices % 4 - 1;
        choices /= 4;
        if (successor >= 0)
          graph.edges.push_back({start, successor, 0.9});
      }
      const BlockPaths block{3, graph, {}};
      for (size_t limit : {1, 2, 3, 5}) {
        SCOPED_TRACE(assignment);
        SCOPED_TRACE(eos);
        ExpectAllWalks({&block, 1}, limit);
      }
    }
  }
}

TEST(MlpAutomatonGraphTest, CollectAllWalksNeverStitchesBlocksOrKeepsPrefixes) {
  Graph first;
  first.token_bytes = {"a", "b", "c"};
  first.edges = {{0, 1, 0.9}};
  Graph second = first;
  second.edges = {{1, 2, 0.9}};
  const std::vector<BlockPaths> blocks{{5, first, {}}, {0, second, {}}};
  ExpectAllWalks(blocks, 16);
  auto paths = CollectAllPaths(blocks, 16);
  ASSERT_TRUE(paths.ok()) << paths.status();
  ASSERT_EQ(paths->size(), 5u);
  EXPECT_EQ((*paths)[0].bytes, "a");
  EXPECT_EQ((*paths)[0].mlp_blocks, (std::vector<int>{0}));
  EXPECT_EQ((*paths)[1].bytes, "ab");
  EXPECT_EQ((*paths)[1].mlp_blocks, (std::vector<int>{5}));
  EXPECT_EQ((*paths)[2].bytes, "b");
  EXPECT_EQ((*paths)[2].mlp_blocks, (std::vector<int>{5}));
  EXPECT_EQ((*paths)[3].bytes, "bc");
  EXPECT_EQ((*paths)[3].mlp_blocks, (std::vector<int>{0}));
  EXPECT_EQ((*paths)[4].bytes, "c");
  EXPECT_EQ((*paths)[4].mlp_blocks, (std::vector<int>{0, 5}));
  // The union of graph edges would invent "abc"; prefixes would incorrectly
  // attribute "a" to B5 or "b" to B0.
}

TEST(MlpAutomatonGraphTest, CollectAllWalksPreservesBytesAndSkipsEmptyText) {
  Graph graph;
  graph.token_bytes = {"", "", "a", "a", std::string("\0\xff", 2), " A", "A"};
  graph.edges = {{0, 1, 0.9}, {2, 3, 0.9}};
  const BlockPaths block{0, graph, {}};
  ExpectAllWalks({&block, 1}, 16);
  auto paths = CollectAllPaths({&block, 1}, 16);
  ASSERT_TRUE(paths.ok()) << paths.status();
  ASSERT_EQ(paths->size(), 5u);
  EXPECT_EQ((*paths)[0].bytes, std::string("\0\xff", 2));
  EXPECT_EQ((*paths)[1].bytes, " A");
  EXPECT_EQ((*paths)[2].bytes, "A");
  EXPECT_EQ((*paths)[3].bytes, "a");
  EXPECT_EQ((*paths)[4].bytes, "aa");
  graph.token_bytes = {"", ""};
  graph.edges = {{0, 1, 0.9}, {1, 0, 0.9}};
  const BlockPaths empty{7, graph, {}};
  paths = CollectAllPaths({&empty, 1}, std::numeric_limits<size_t>::max());
  ASSERT_TRUE(paths.ok()) << paths.status();
  EXPECT_TRUE(paths->empty());
}

TEST(MlpAutomatonGraphTest, CollectAllWalksValidatesGraphsAndBlockIds) {
  auto empty = CollectAllPaths({}, 16);
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_TRUE(empty->empty());
  EXPECT_EQ(CollectAllPaths({}, 0).status().code(),
            absl::StatusCode::kInvalidArgument);

  std::vector<BlockPaths> blocks{{1, ExampleGraph(), {}}};
  blocks.push_back(blocks.front());
  EXPECT_EQ(CollectAllPaths(blocks, 16).status().code(),
            absl::StatusCode::kInvalidArgument);
  blocks.pop_back();
  blocks.front().mlp_block = -1;
  EXPECT_EQ(CollectAllPaths(blocks, 16).status().code(),
            absl::StatusCode::kInvalidArgument);
  blocks.front().mlp_block = 0;
  // Validation must still happen when all decoded strings would be empty.
  blocks.front().graph.token_bytes.assign(7, "");
  blocks.front().graph.edges.push_back({0, 1, 0.9});
  EXPECT_EQ(CollectAllPaths(blocks, 16).status().code(),
            absl::StatusCode::kInvalidArgument);
  blocks.front().graph = ExampleGraph();
  blocks.front().graph.edges[0].target = 99;
  EXPECT_EQ(CollectAllPaths(blocks, 16).status().code(),
            absl::StatusCode::kInvalidArgument);
  blocks.front().graph = Graph{};
  EXPECT_EQ(CollectAllPaths(blocks, 16).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::llm::mlp_automaton
