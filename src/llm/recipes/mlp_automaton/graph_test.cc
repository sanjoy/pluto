#include "src/llm/recipes/mlp_automaton/graph.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <string>
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
  for (const Path& result : *sampled) EXPECT_NE(result.tokens[0], 5);
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

}  // namespace
}  // namespace pluto::llm::mlp_automaton
