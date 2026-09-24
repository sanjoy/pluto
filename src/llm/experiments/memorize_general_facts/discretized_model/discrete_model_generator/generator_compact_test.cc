#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/discretize_attention.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/discretize_attention_test_util.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/discretize_map.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/discretize_position_embedding.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/state_compactor.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/utils.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized::generator {
namespace {

using absl::StrCat;
using internal::BuildAttention;
using internal::EvaluateAttention;

// Test adapters inspect complete emitted source while exercising the same
// captured-layer interfaces used by the whole-model code generator.
absl::StatusOr<std::string> AttentionSource(
    absl::string_view name, const std::vector<AttentionTransition>& rows,
    int chunk_size = 256, absl::string_view strategy = "hybrid",
    bool compact = true) {
  ASSIGN_OR_RETURN(auto program,
                   RenderAttention({rows}, name, compact, chunk_size, strategy));
  return std::move(program.source);
}

absl::StatusOr<std::string> PointwiseSource(
    absl::string_view name, const std::vector<StateTransition>& rows,
    const TokenNames* names = nullptr, bool compact = true) {
  ASSIGN_OR_RETURN(auto program, RenderMap({rows}, name, names, compact));
  return std::move(program.source);
}

absl::StatusOr<std::string> EntrySource(
    absl::string_view name, const std::vector<EntryTransition>& rows,
    const TokenNames& names, bool compact = true) {
  ASSIGN_OR_RETURN(auto program,
                   RenderPositionEmbedding({rows}, name, names, compact));
  return std::move(program.source);
}

// Deliberately independent linear scans supply the expected values for
// compiled-code tests. They do not reuse a lowerer's support-mask or search.
absl::StatusOr<std::optional<int>> EvaluatePointwise(
    absl::Span<const StateTransition> rows, int state) {
  for (const auto& row : rows)
    if (row.input == state)
      return std::optional<int>{row.output};
  return std::optional<int>{};
}

absl::StatusOr<std::optional<int>> EvaluateEntry(
    absl::Span<const EntryTransition> rows, int token, int position) {
  for (const auto& row : rows)
    if (row.token == token && row.position == position)
      return std::optional<int>{row.output};
  return std::optional<int>{};
}

CapturedModel Fixture() {
  CapturedModel model;
  model.metadata = {1, 2, 3, 2, 1, {{0, "A"}, {1, "B"}, {2, "C"}}};
  model.position_embedding.transitions = {
      {0, 0, 100}, {1, 0, 101}, {1, 1, 101}};
  model.transformers.resize(2);
  model.transformers[0].attention.transitions = {
      {{100}, 102}, {{100, 101}, 103}, {{101}, 103}};
  model.transformers[0].mlp.transitions = {{102, 105}, {103, 104}};
  model.transformers[1].attention.transitions = {
      {{105}, 106}, {{105, 104}, 107}, {{104}, 107}};
  model.transformers[1].mlp.transitions = {{106, 109}, {107, 108}};
  model.language_modeling_head.transitions = {{108, 2}, {109, 1}};
  model.samples = {{{0, 1}}, {{1}}};
  model.stats.membership_complete = true;
  for (int i = 0; i < 10; ++i)
    model.states.push_back({100 + i, i / 2, std::vector<int>{1000 + i}});
  return model;
}

// Compile emitted programs independently of the neural runtime and exercise
// their entire small input domains, including negative and missing symbols.
// This catches emitter bugs that checking emitted text alone cannot.
absl::Status CompileAndRun(absl::string_view source) {
  auto base =
      std::filesystem::temp_directory_path() / "pluto-compact-test-XXXXXX";
  std::string pattern = base.string();
  if (mkdtemp(pattern.data()) == nullptr)
    return absl::InternalError("mkdtemp failed");
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } cleanup{pattern};
  const auto input = cleanup.path / "program.cc";
  const auto binary = cleanup.path / "program";
  RETURN_IF_ERROR(WriteFile(input, source));
  RETURN_IF_ERROR(RunProcess({"/usr/bin/c++", "-std=c++20", "-O1", "-Wall",
                              "-Wextra", "-Werror", "-fsanitize=undefined",
                              "-fno-sanitize-recover=undefined", input.string(),
                              "-o", binary.string()}));
  return RunProcess({binary.string()});
}

const char* kDeclarations = R"cpp(
#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>
  namespace absl {
  template <class T>
  using Span = std::span<T>;
  }
  struct DiscreteToken;
  struct DiscreteHiddenState {
    int value = 0;
    constexpr auto operator<=>(const DiscreteHiddenState&) const = default;
    constexpr explicit operator DiscreteToken() const;
  };
  struct DiscreteToken {
    int value = 0;
    constexpr auto operator<=>(const DiscreteToken&) const = default;
    constexpr explicit operator DiscreteHiddenState() const { return {value}; }
  };
  constexpr DiscreteHiddenState::operator DiscreteToken() const {
    return {value};
  }
)cpp";

std::vector<std::vector<int>> Histories(int alphabet, int max_length) {
  std::vector<std::vector<int>> result{{}};
  size_t first = 0, end = 1;
  for (int length = 0; length < max_length; ++length) {
    for (size_t index = first; index < end; ++index)
      for (int symbol = 0; symbol < alphabet; ++symbol) {
        auto next = result[index];
        next.push_back(symbol);
        result.push_back(std::move(next));
      }
    first = end;
    end = result.size();
  }
  return result;
}

TEST(AttentionLogicTest, SharedSuffixesPreserveOutputsAndAllFutureEdges) {
  std::vector<AttentionTransition> rows = {
      {{1}, 10}, {{1, 2}, 20}, {{3}, 10}, {{3, 2}, 20}};
  auto program = BuildAttention(rows);
  ASSERT_TRUE(program.ok()) << program.status();
  EXPECT_EQ(program->nodes.size(), 3u);
  EXPECT_EQ(EvaluateAttention(*program, {1, 2}), 20);
  EXPECT_EQ(EvaluateAttention(*program, {3, 2}), 20);
  rows[3].output = 21;
  auto distinct = BuildAttention(rows);
  ASSERT_TRUE(distinct.ok()) << distinct.status();
  EXPECT_EQ(distinct->nodes.size(), 5u);
  EXPECT_EQ(EvaluateAttention(*distinct, {3, 2}), 21);
}

TEST(AttentionLogicTest, ExactDomainRejectsTruncationExtensionAndReordering) {
  auto program = BuildAttention({{{1, 2}, 0}, {{3, 2}, 2147483647}});
  ASSERT_TRUE(program.ok()) << program.status();
  EXPECT_EQ(EvaluateAttention(*program, {1, 2}), 0);
  EXPECT_EQ(EvaluateAttention(*program, {3, 2}), 2147483647);
  for (const auto& history : std::vector<std::vector<int>>{
           {}, {1}, {3}, {2, 1}, {1, 2, 3}, {9, 2}, {-1}})
    EXPECT_FALSE(EvaluateAttention(*program, history).has_value());
}

TEST(AttentionLogicTest, ZeroOutputIsDistinctFromNonacceptingPrefix) {
  auto program = BuildAttention({{{1}, 0}, {{1, 3}, 9}, {{2, 3}, 9}});
  ASSERT_TRUE(program.ok()) << program.status();
  EXPECT_EQ(EvaluateAttention(*program, {1}), 0);
  EXPECT_FALSE(EvaluateAttention(*program, {2}).has_value());
  EXPECT_EQ(EvaluateAttention(*program, {2, 3}), 9);
}

TEST(AttentionLogicTest, RandomRecognizersMatchEverySmallHistory) {
  std::mt19937 random(7041);
  auto candidates = Histories(3, 5);
  for (int trial = 0; trial < 20; ++trial) {
    std::map<std::vector<int>, int> table;
    for (int row = 0; row < 40; ++row)
      table[candidates[1 + random() % (candidates.size() - 1)]] = random() % 12;
    std::vector<AttentionTransition> rows;
    for (const auto& [key, output] : table)
      rows.push_back({key, output});
    auto program = BuildAttention(rows);
    ASSERT_TRUE(program.ok()) << program.status();
    for (const auto& key : candidates) {
      auto expected = table.find(key);
      EXPECT_EQ(EvaluateAttention(*program, key),
                expected == table.end() ? std::optional<int>{}
                                        : std::optional<int>{expected->second});
    }
  }
}

TEST(AttentionLogicTest, DuplicateRowsAndOrderingDoNotChangeEmission) {
  std::vector<AttentionTransition> rows = {
      {{1, 2}, 5}, {{1}, 3}, {{4, 2}, 5}, {{4}, 3}};
  auto first = AttentionSource("Attention0", rows);
  auto original = rows;
  std::reverse(rows.begin(), rows.end());
  rows.push_back(rows.back());
  auto second = AttentionSource("Attention0", rows);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(*first, *second);
  EXPECT_EQ(*BuildAttention(original), *BuildAttention(rows));
}

TEST(AttentionLogicTest, InvalidRowsNamesAndEmissionOptionsReturnErrors) {
  // Row shape and integer representation are now enforced by C++ types.
  for (const auto& rows : std::vector<std::vector<AttentionTransition>>{
           {{{1}, 2}, {{1}, 3}}, {{{}, 0}}, {{{-1}, 1}}, {{{1}, -1}}})
    EXPECT_FALSE(BuildAttention(rows).ok());
  for (const char* name :
       {"class", "Bad::Name", "1Bad", "_Reserved", "Bad__Name", "x;}"})
    EXPECT_FALSE(AttentionSource(name, {}).ok()) << name;
  EXPECT_FALSE(AttentionSource("Attention", {}, 0).ok());
  EXPECT_FALSE(AttentionSource("Attention", {}, -1).ok());
  EXPECT_FALSE(AttentionSource("Attention", {}, 1, "approximate").ok());
  auto largest = BuildAttention({{{2147483647}, 2147483647}});
  ASSERT_TRUE(largest.ok()) << largest.status();
  EXPECT_EQ(EvaluateAttention(*largest, {2147483647}), 2147483647);
}

TEST(AttentionLogicTest, HybridSharesNarrowRunsAndControlFlowDoesNot) {
  std::vector<AttentionTransition> rows;
  std::vector<int> history;
  for (int length = 1; length <= 12; ++length) {
    history.push_back(64999 + length);
    rows.push_back({history, 65535 - length});
  }
  auto hybrid = AttentionSource("NarrowAttention", rows);
  auto pure = AttentionSource("PureAttention", rows, 256, "control_flow");
  ASSERT_TRUE(hybrid.ok()) << hybrid.status();
  ASSERT_TRUE(pure.ok()) << pure.status();
  EXPECT_NE(hybrid->find("std::uint16_t symbol, output"), std::string::npos);
  EXPECT_NE(hybrid->find("NarrowAttentionMatchSequence("), std::string::npos);
  EXPECT_EQ(pure->find("PureAttentionMatchSequence("), std::string::npos);
}

TEST(AttentionLogicTest, EmittedCppMatchesAllSupportedAndUnsupportedHistories) {
  std::mt19937 random(1059);
  auto candidates = Histories(3, 5);
  std::map<std::vector<int>, int> table;
  for (int row = 0; row < 80; ++row)
    table[candidates[1 + random() % (candidates.size() - 1)]] = random() % 20;
  table[{2147483647}] = 0;
  table[{2147483647, 7}] = 2147483647;
  const std::vector<int> chain{70000, 8, 9, 10, 11, 12, 13, 14, 2147483647};
  for (size_t length = 1; length <= chain.size(); ++length)
    table[{chain.begin(), chain.begin() + length}] =
        length == 5 ? 2147483646 : length;
  std::vector<AttentionTransition> rows;
  for (const auto& [key, output] : table)
    rows.push_back({key, output});
  auto hybrid = AttentionSource("CompiledAttention", rows, 7);
  auto pure = AttentionSource("PureAttention", rows, 7, "control_flow");
  auto empty = AttentionSource("EmptyAttention", {});
  std::vector<AttentionTransition> narrow_rows;
  std::vector<int> narrow_key;
  for (int length = 1; length <= 8; ++length) {
    narrow_key.push_back(length);
    narrow_rows.push_back({narrow_key, length - 1});
  }
  auto narrow = AttentionSource("NarrowAttention", narrow_rows);
  ASSERT_TRUE(hybrid.ok()) << hybrid.status();
  ASSERT_TRUE(pure.ok()) << pure.status();
  ASSERT_TRUE(empty.ok()) << empty.status();
  ASSERT_TRUE(narrow.ok()) << narrow.status();
  EXPECT_NE(hybrid->find("CompiledAttentionPart1("), std::string::npos);
  EXPECT_NE(hybrid->find("std::uint32_t symbol, output"), std::string::npos);
  EXPECT_NE(narrow->find("std::uint16_t symbol, output"), std::string::npos);
  std::string source =
      StrCat(kDeclarations, *hybrid, *pure, *empty, *narrow, "int main() {\n");
  absl::StrAppend(&source, R"cpp(
    std::vector<DiscreteHiddenState> narrow;
    if (NarrowAttention(narrow).has_value())
      return 4;
    for (int length = 1; length <= 8; ++length) {
      narrow.push_back(DiscreteHiddenState{length});
      if (NarrowAttention(narrow) !=
          std::optional<DiscreteHiddenState>{{length - 1}})
        return 5;
    }
    narrow.push_back(DiscreteHiddenState{9});
    if (NarrowAttention(narrow).has_value())
      return 6;
  )cpp");
  candidates.insert(candidates.end(), {{-1}, {70000, -1}, {-2147483647 - 1}});
  for (const auto& [key, output] : table) {
    candidates.push_back(key);
    auto extension = key;
    extension.push_back(99);
    candidates.push_back(std::move(extension));
  }
  for (const auto& key : candidates) {
    absl::StrAppend(&source, "{ std::vector<DiscreteHiddenState> key = {");
    for (int state : key)
      absl::StrAppend(&source, "{", state, "},");
    auto found = table.find(key);
    std::string expected = found == table.end()
                               ? "std::nullopt"
                               : StrCat("std::optional<DiscreteHiddenState>{{",
                                        found->second, "}}");
    absl::StrAppend(
        &source,
        "}; auto a = CompiledAttention(key); auto b = PureAttention(key);",
        " if (a != b || a != ", expected,
        " || EmptyAttention(key).has_value()) return 1; }\n");
  }
  absl::StrAppend(&source, "return 0; }\n");
  EXPECT_TRUE(CompileAndRun(source).ok());
}

TEST(AttentionLogicTest,
     IndependentStateMatchersAreTheDefaultAndDeterministic) {
  std::vector<AttentionTransition> rows = {
      {{7, 0}, 2147483647}, {{0}, 0}, {{7}, 9}, {{0, 7}, 0}};
  const auto default_program = RenderAttention({rows}, "Lookup");
  const auto explicit_program =
      RenderAttention({rows}, "Lookup", true, 256, "state_matchers");
  ASSERT_TRUE(default_program.ok()) << default_program.status();
  ASSERT_TRUE(explicit_program.ok()) << explicit_program.status();
  EXPECT_EQ(default_program->source, explicit_program->source);
  for (int state : {0, 9, 2147483647})
    EXPECT_NE(
        default_program->source.find(StrCat("bool MatchState", state, "(")),
        std::string::npos);
  EXPECT_EQ(default_program->source.find("PLUTO_ATTN_"), std::string::npos);
  EXPECT_EQ(default_program->source.find("goto "), std::string::npos);

  std::reverse(rows.begin(), rows.end());
  rows.push_back(rows.front());
  const auto reordered =
      RenderAttention({rows}, "Lookup", true, 256, "state_matchers");
  ASSERT_TRUE(reordered.ok()) << reordered.status();
  EXPECT_EQ(reordered->source, explicit_program->source);
  EXPECT_FALSE(RenderAttention({{{{0}, 0}, {{0}, 9}}}, "Lookup", true, 256,
                               "state_matchers")
                   .ok());
  for (int state : {0, 9, 2147483647}) {
    const auto collision = RenderAttention({rows}, StrCat("MatchState", state));
    EXPECT_EQ(collision.status().code(), absl::StatusCode::kInvalidArgument);
  }
  EXPECT_TRUE(RenderAttention({rows}, "MatchState42").ok());
}

TEST(AttentionLogicTest, IndependentMatcherChecksTheSelectivePositionFirst) {
  const CapturedCausalAttention attention{
      {{{1, 2, 3}, 10}, {{1, 9, 3}, 11}, {{1, 8, 3}, 11}}};
  const auto program = RenderAttention(attention, "Lookup");
  ASSERT_TRUE(program.ok()) << program.status();
  const auto begin = program->source.find("bool MatchState10(");
  const auto end = program->source.find("bool MatchState11(");
  ASSERT_NE(begin, std::string::npos);
  ASSERT_NE(end, std::string::npos);
  ASSERT_LT(begin, end);
  const auto matcher = program->source.substr(begin, end - begin);
  const auto length = matcher.find("history.size()");
  const auto first = matcher.find("history[0]");
  const auto selective = matcher.find("history[1]");
  const auto last = matcher.find("history[2]");
  ASSERT_NE(length, std::string::npos);
  ASSERT_NE(first, std::string::npos);
  ASSERT_NE(selective, std::string::npos);
  ASSERT_NE(last, std::string::npos);
  // Position 1 rejects both competitors with one read. The two other reads
  // remain necessary before acceptance to reject unseen histories too.
  EXPECT_LT(length, selective);
  EXPECT_LT(selective, first);
  EXPECT_LT(selective, last);
}

TEST(AttentionLogicTest,
     EveryStateMatcherRecognizesItsExactDomainIndependently) {
  // Several outputs share a last token or a prefix. A later matcher's false
  // positives cannot be hidden by an earlier successful dispatch branch.
  const std::vector<AttentionTransition> rows = {
      {{0}, 0},
      {{7}, 9},
      {{0, 7}, 0},
      {{0, 0}, 9},
      {{0, 7, 2147483647}, 2147483647},
      {{7, 7, 2147483647}, 0},
      {{7, 0, 2147483647}, 9},
      {{2147483647, 0, 0}, 2147483647},
      {{2147483647, 7, 0}, 9}};
  const auto matchers = AttentionSource("Lookup", rows, 3, "state_matchers");
  const auto plain = AttentionSource("Lookup", rows, 3, "hybrid", false);
  const auto empty = AttentionSource("Lookup", {}, 3, "state_matchers");
  ASSERT_TRUE(matchers.ok()) << matchers.status();
  ASSERT_TRUE(plain.ok()) << plain.status();
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_EQ(empty->find("bool MatchState"), std::string::npos);

  std::string source =
      StrCat(kDeclarations, "namespace Independent {\n", *matchers,
             "}\nnamespace Plain {\n", *plain, "}\nnamespace Empty {\n", *empty,
             "}\nint main() {\n");
  auto candidates = Histories(5, 4);
  const int alphabet[] = {0, 7, 2147483647, 42, -1};
  for (auto& history : candidates)
    for (int& token : history)
      token = alphabet[token];
  candidates.push_back({std::numeric_limits<int>::min()});
  candidates.push_back(std::vector<int>(64, 0));
  for (const auto& key : candidates) {
    std::optional<int> expected;
    for (const auto& row : rows)
      if (row.prefix == key)
        expected = row.output;
    absl::StrAppend(&source, "{ std::vector<DiscreteHiddenState> key = {");
    for (int state : key)
      absl::StrAppend(&source, "{", state, "},");
    absl::StrAppend(&source, "};\n");
    // Invoke matchers in reverse output order BEFORE the dispatcher, then in
    // a different order afterward. Every call must work without assumptions
    // about prior matcher failures, remembered history, or dispatcher state.
    for (int state : {2147483647, 9, 0})
      absl::StrAppend(&source, "if (Independent::MatchState", state,
                      "(key) != ", expected == state ? "true" : "false",
                      ") return 1;\n");
    const std::string answer =
        expected
            ? StrCat("std::optional<DiscreteHiddenState>{{", *expected, "}}")
            : "std::nullopt";
    absl::StrAppend(&source, "if (Independent::Lookup(key) != ", answer,
                    " || Plain::Lookup(key) != ", answer,
                    " || Empty::Lookup(key).has_value()) return 2;\n");
    for (int state : {0, 2147483647, 9})
      absl::StrAppend(&source, "if (Independent::MatchState", state,
                      "(key) != ", expected == state ? "true" : "false",
                      ") return 3;\n");
    absl::StrAppend(&source, "}\n");
  }
  absl::StrAppend(&source, "return 0; }\n");
  const auto status = CompileAndRun(source);
  EXPECT_TRUE(status.ok()) << status;
}

TEST(PointwiseTest, RelabelingPreservesBoundaryOwnershipAndMembers) {
  auto original = Fixture();
  auto before = original;
  auto result = RelabelMlpOutputs(original);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(original, before);
  EXPECT_EQ(result->transformers[0].mlp.transitions,
            (std::vector<StateTransition>{{102, 104}, {103, 105}}));
  EXPECT_EQ(result->transformers[1].mlp.transitions,
            (std::vector<StateTransition>{{107, 110}, {108, 111}}));
  EXPECT_EQ(result->language_modeling_head.transitions,
            (std::vector<StateTransition>{{110, 1}, {111, 2}}));
  EXPECT_EQ(result->state_relabeling.size(), original.states.size());
  ASSERT_TRUE(result->stats.pointwise_relabeling.has_value());
  EXPECT_EQ(result->stats.pointwise_relabeling->changed_states, 6);
  EXPECT_TRUE(
      result->stats.pointwise_relabeling->vocabulary_aligned_final_boundaries);
  for (const auto& row : result->state_relabeling) {
    CapturedState old_state, new_state;
    for (const auto& state : original.states)
      if (state.id == row.old_id)
        old_state = state;
    for (const auto& state : result->states)
      if (state.id == row.new_id)
        new_state = state;
    EXPECT_EQ(new_state.boundary, row.boundary);
    old_state.id = new_state.id;
    EXPECT_EQ(old_state, new_state);
  }
  auto again = RelabelMlpOutputs(*result);
  ASSERT_TRUE(again.ok()) << again.status();
  EXPECT_EQ(again->transformers, result->transformers);
}

TEST(PointwiseTest, RelabelSkipsNonbijectiveIncompleteAndOverflowingRanges) {
  auto nonbijective = Fixture();
  nonbijective.transformers[0].mlp.transitions[1].output =
      nonbijective.transformers[0].mlp.transitions[0].output;
  auto result = RelabelMlpOutputs(nonbijective);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_TRUE(result->stats.pointwise_relabeling.has_value());
  EXPECT_EQ(result->stats.pointwise_relabeling->skipped_layers,
            std::vector<int>{0});
  EXPECT_EQ(result->transformers[0].mlp.transitions,
            nonbijective.transformers[0].mlp.transitions);
  for (int defect = 0; defect < 3; ++defect) {
    auto model = Fixture();
    if (defect == 0)
      model.language_modeling_head.transitions.erase(
          model.language_modeling_head.transitions.begin() + 1);
    if (defect == 1)
      model.language_modeling_head.transitions[1].output =
          model.language_modeling_head.transitions[0].output;
    if (defect == 2)
      model.states.push_back({2147483646, 0});
    auto renamed = RelabelMlpOutputs(model);
    ASSERT_TRUE(renamed.ok()) << renamed.status();
    ASSERT_TRUE(renamed->stats.pointwise_relabeling.has_value());
    EXPECT_FALSE(renamed->stats.pointwise_relabeling
                     ->vocabulary_aligned_final_boundaries);
  }
}

TEST(PointwiseTest, RelabelRejectsInvalidTypedMetadata) {
  for (int invalid : {-1, 5, std::numeric_limits<int>::max()}) {
    auto model = Fixture();
    model.states[0].boundary = invalid;
    EXPECT_FALSE(RelabelMlpOutputs(model).ok()) << invalid;
  }
  for (int invalid : {-1, 0, std::numeric_limits<int>::max()}) {
    auto model = Fixture();
    model.metadata.layers = invalid;
    EXPECT_FALSE(RelabelMlpOutputs(model).ok()) << invalid;
  }
  auto model = Fixture();
  model.metadata.vocab_size = 0;
  EXPECT_FALSE(RelabelMlpOutputs(model).ok());
  model = Fixture();
  model.stats = {};
  EXPECT_TRUE(RelabelMlpOutputs(model).ok());
}

TEST(PointwiseTest, SparseNamedAffineMapsUseExactMasks) {
  std::vector<StateTransition> rows;
  TokenNames names;
  for (int i = 0; i < 128; ++i) {
    names[i] = StrCat("vocab::Token", i);
    if (i % 3 != 1)
      rows.push_back({1000 + i, i});
  }
  auto result = PointwiseSource("Head", rows, &names);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_NE(result->find("kSupport[offset >> 3]"), std::string::npos);
  EXPECT_NE(result->find("vocab::Token0"), std::string::npos);
  EXPECT_EQ(result->find("kOutputs"), std::string::npos);
  EXPECT_LT(result->size(), 1000u);
}

TEST(PointwiseTest, EvaluatorsDistinguishUnsupportedAndSupportedZero) {
  EXPECT_EQ(*EvaluatePointwise({{3, 0}}, 3), 0);
  EXPECT_FALSE(EvaluatePointwise({{3, 0}}, 4)->has_value());
  EXPECT_EQ(*EvaluateEntry({{1, 2, 0}}, 1, 2), 0);
  EXPECT_FALSE(EvaluateEntry({{1, 2, 0}}, 1, 3)->has_value());
  EXPECT_FALSE(PointwiseSource("Bad", {{1, 3}, {1, 4}}).ok());
  EXPECT_FALSE(
      EntrySource("Bad", {{1, 2, 3}, {1, 2, 4}}, {{1, "vocab::B"}}).ok());
}

TEST(PointwiseTest, DenseMapsChooseGuardedAffineOrNamedOutputArray) {
  auto affine = PointwiseSource("Mlp", {{10, 20}, {11, 21}, {12, 22}});
  ASSERT_TRUE(affine.ok()) << affine.status();
  EXPECT_EQ(affine->find("kOutputs"), std::string::npos);
  EXPECT_EQ(affine->find("kSupport"), std::string::npos);
  EXPECT_NE(affine->find("state.value < 10 || state.value > 12"),
            std::string::npos);
  std::vector<StateTransition> rows;
  TokenNames names;
  for (int i = 0; i < 20; ++i) {
    rows.push_back({10 + i, (i * 7) % 20});
    names[i] = StrCat("vocab::Token", i);
  }
  auto irregular = PointwiseSource("Head", rows, &names);
  ASSERT_TRUE(irregular.ok()) << irregular.status();
  EXPECT_NE(irregular->find("kOutputs[state.value - 10]"), std::string::npos);
  EXPECT_NE(irregular->find("vocab::Token19"), std::string::npos);
}

TEST(PointwiseTest, EntryRetainsPositionSupportAndNamedExceptions) {
  auto result = EntrySource(
      "Entry", {{0, 0, 20}, {0, 2, 20}, {0, 3, 21}, {1, 1, 21}, {3, 0, 20}},
      {{0, "vocab::A"}, {1, "vocab::B"}, {3, "vocab::D"}});
  ASSERT_TRUE(result.ok()) << result.status();
  for (const char* text : {"case vocab::A.value:", "position == 3",
                           "position < 0", "position > 3", "return {};"})
    EXPECT_NE(result->find(text), std::string::npos) << text;
}

TEST(PointwiseTest, InvalidIdsPositionsAndMissingNamesFailSafely) {
  for (int bad : {-1, std::numeric_limits<int>::min()}) {
    EXPECT_FALSE(PointwiseSource("Bad", {{bad, 0}}).ok());
    EXPECT_FALSE(PointwiseSource("Bad", {{0, bad}}).ok());
    EXPECT_FALSE(EntrySource("Bad", {{bad, 0, 0}}, {{0, "vocab::A"}}).ok());
    EXPECT_FALSE(EntrySource("Bad", {{0, bad, 0}}, {{0, "vocab::A"}}).ok());
    EXPECT_FALSE(EntrySource("Bad", {{0, 0, bad}}, {{0, "vocab::A"}}).ok());
  }
  TokenNames names;
  EXPECT_FALSE(PointwiseSource("Bad", {{0, 1}}, &names).ok());
  EXPECT_FALSE(EntrySource("Bad", {{0, 0, 1}}, names).ok());
  for (const char* name :
       {"class", "Bad::Name", "1Bad", "_Reserved", "Bad__Name", "x;}"}) {
    EXPECT_FALSE(PointwiseSource(name, {}).ok()) << name;
    EXPECT_FALSE(EntrySource(name, {}, names).ok()) << name;
  }
  auto largest = PointwiseSource("Largest", {{2147483647, 2147483647}});
  ASSERT_TRUE(largest.ok()) << largest.status();
  auto largest_entry =
      EntrySource("LargestEntry", {{2147483647, 2147483647, 2147483647}},
                  {{2147483647, "vocab::Max"}});
  ASSERT_TRUE(largest_entry.ok()) << largest_entry.status();
}

TEST(PointwiseTest, EmittedCppMatchesExactDomainsAndSignedBoundaries) {
  TokenNames names{{0, "vocab::A"},
                   {1, "vocab::B"},
                   {2, "vocab::C"},
                   {3, "vocab::D"},
                   {1000000000, "vocab::Far"},
                   {2147483647, "vocab::Max"}};
  std::string source =
      StrCat(kDeclarations,
             "namespace vocab { constexpr DiscreteToken A{0}, B{1}, C{2}, "
             "D{3}, Far{1000000000}, Max{2147483647}; }\n");
  TokenNames numbered;
  for (int i = 0; i < 128; ++i) {
    numbered[i] = StrCat("vocab::Token", i);
    absl::StrAppend(&source, "namespace vocab { constexpr DiscreteToken Token",
                    i, "{", i, "}; }\n");
  }
  std::map<std::string, std::vector<StateTransition>> mappings{
      {"Affine", {{5, 25}, {6, 26}, {7, 27}}},
      {"Holes", {{5, 25}, {6, 26}, {7, 27}, {8, 40}, {10, 50}}},
      {"EntireSignedSpan", {{0, 0}, {2147483647, 2147483647}}},
      {"HighAffineOutputs",
       {{0, 2147483645}, {1, 2147483646}, {2, 2147483647}}},
      {"Empty", {}}};
  for (int i = 0; i < 13; ++i) {
    mappings["Irregular"].push_back({i, (i * 7) % 13});
    mappings["HighOutputs"].push_back({i, 2147483647 - (i * 7) % 13});
  }
  for (int i = 0; i < 15; ++i)
    mappings["Named"].push_back({i, i % 3});
  for (int i = 0; i < 128; ++i)
    if (i % 3 != 1) {
      mappings["SparseAffine"].push_back({20 + i, 100 + i});
      mappings["NamedSparseAffine"].push_back({i, i});
    }
  for (int i = 0; i < 6; ++i)
    if (i != 2)
      mappings["MaxSigned"].push_back({2147483642 + i, i});
  for (int i = 0; i < 8; ++i)
    mappings["ZeroBasedRanges"].push_back({i, i});
  mappings["ZeroBasedRanges"].push_back({100, 200});
  std::string checks;
  std::vector<int> checked{-2147483647 - 1};
  for (int i = -2; i < 150; ++i)
    checked.push_back(i);
  for (int64_t i = 2147483640; i <= 2147483647; ++i)
    checked.push_back(i);
  for (const auto& [name, rows] : mappings) {
    auto generated = PointwiseSource(name, rows,
                                     name == "Named"               ? &names
                                     : name == "NamedSparseAffine" ? &numbered
                                                                   : nullptr);
    ASSERT_TRUE(generated.ok()) << generated.status();
    absl::StrAppend(&source, *generated);
    for (int state : checked) {
      auto expected = EvaluatePointwise(rows, state);
      ASSERT_TRUE(expected.ok()) << expected.status();
      absl::StrAppend(&checks, "{auto r=", name, "(DiscreteHiddenState{", state,
                      "}); if(r.value_or(DiscreteHiddenState{0}).value != ",
                      expected->value_or(0), " || r.has_value() != ",
                      expected->has_value() ? "true" : "false",
                      ") return 1;}\n");
    }
  }
  std::map<std::string, std::vector<EntryTransition>> entries{
      {"Entry", {{0, 0, 20}, {0, 2, 20}, {0, 3, 21}, {1, 1, 21}, {3, 0, 20}}},
      {"WideMask", {{0, 31, 20}, {1, 31, 21}}},
      {"Sparse", {{0, 0, 20}, {1000000000, 0, 21}}},
      {"WidePosition", {{0, 100, 20}}},
      {"MaxPosition", {{0, 2147483647, 0}}},
      {"ZeroEntry", {{0, 0, 0}}},
      {"MaxEntry", {{0, 0, 2147483647}, {1, 0, 2147483646}}},
      {"MaxToken", {{2147483647, 0, 2147483647}}},
      {"SparseStates", {{0, 0, 0}, {1, 0, 2147483647}}},
      {"EmptyEntry", {}}};
  for (const auto& [name, rows] : entries) {
    auto generated = EntrySource(name, rows, names);
    ASSERT_TRUE(generated.ok()) << generated.status();
    absl::StrAppend(&source, *generated);
    for (int token :
         {-2147483647 - 1, -1, 0, 1, 2, 3, 4, 1000000000, 2147483647})
      for (int position : {-2147483647 - 1, -1, 0, 1, 2, 3, 4, 31, 32, 63, 64,
                           100, 101, 2147483647}) {
        auto expected = EvaluateEntry(rows, token, position);
        ASSERT_TRUE(expected.ok()) << expected.status();
        absl::StrAppend(
            &checks, "{auto r=", name, "(DiscreteToken{", token, "},", position,
            "); if(r.value_or(DiscreteHiddenState{0}).value != ",
            expected->value_or(0),
            " || r.has_value() != ", expected->has_value() ? "true" : "false",
            ") return 2;}\n");
      }
  }
  absl::StrAppend(&source, "int main() {\n", checks, "return 0; }\n");
  auto status = CompileAndRun(source);
  EXPECT_TRUE(status.ok()) << status;
}

TEST(LayerLoweringTest, PlainAndCompactProgramsAgreeOnUnknownInputsToo) {
  const CapturedMap map{{{10, 0}, {12, 21}, {11, 21}, {2147483647, 9}}};
  const CapturedPositionEmbedding embedding{
      {{0, 0, 0}, {0, 2, 11}, {1, 0, 12}, {1, 2147483647, 9}}};
  const CapturedCausalAttention attention{
      {{{10}, 0}, {{10, 12}, 21}, {{12, 10}, 11}, {{12, 10, 12}, 9}}};
  const TokenNames names{{0, "DiscreteToken{0}"}, {1, "DiscreteToken{1}"}};
  std::string source = kDeclarations;
  for (bool compact : {false, true}) {
    const std::string prefix = compact ? "Compact" : "Plain";
    auto map_program = RenderMap(map, "Apply", nullptr, compact);
    auto embedding_program =
        RenderPositionEmbedding(embedding, "Apply", names, compact);
    auto attention_program = RenderAttention(attention, "Apply", compact);
    ASSERT_TRUE(map_program.ok()) << map_program.status();
    ASSERT_TRUE(embedding_program.ok()) << embedding_program.status();
    ASSERT_TRUE(attention_program.ok()) << attention_program.status();
    absl::StrAppend(&source, "namespace ", prefix, "Map {\n",
                    map_program->source, "}\nnamespace ", prefix, "Entry {\n",
                    embedding_program->source, "}\nnamespace ", prefix,
                    "Attention {\n", attention_program->source, "}\n");
  }
  source += "int main() {\n";
  for (int state : {-2147483647 - 1, -1, 0, 9, 10, 11, 12, 13, 2147483647})
    absl::StrAppend(&source, "if (PlainMap::Apply(DiscreteHiddenState{", state,
                    "}) != CompactMap::Apply(DiscreteHiddenState{", state,
                    "})) return 1;\n");
  for (int token : {-1, 0, 1, 2, 2147483647})
    for (int position : {-1, 0, 1, 2, 3, 2147483647})
      absl::StrAppend(&source, "if (PlainEntry::Apply(DiscreteToken{", token,
                      "}, ", position,
                      ") != CompactEntry::Apply(DiscreteToken{", token, "}, ",
                      position, ")) return 2;\n");
  for (const auto& history : Histories(4, 4)) {
    std::string key;
    for (int symbol : history)
      absl::StrAppend(&key, "DiscreteHiddenState{", symbol + 9, "},");
    absl::StrAppend(&source, "{std::vector<DiscreteHiddenState> key{", key,
                    "}; if (PlainAttention::Apply(key) != "
                    "CompactAttention::Apply(key)) return 3;}\n");
  }
  source += "return 0;}\n";
  const auto status = CompileAndRun(source);
  EXPECT_TRUE(status.ok()) << status;
}

TEST(LayerLoweringTest, AllRepresentationsValidateTheirCapturedLayer) {
  for (bool compact : {false, true}) {
    EXPECT_FALSE(
        RenderAttention({{{{1}, 2}, {{1}, 3}}}, "Lookup", compact).ok());
    EXPECT_FALSE(
        RenderMap({{{1, 2}, {1, 3}}}, "Lookup", nullptr, compact).ok());
    EXPECT_FALSE(RenderPositionEmbedding({{{1, 0, 2}, {1, 0, 3}}}, "Lookup",
                                         {{1, "DiscreteToken{1}"}}, compact)
                     .ok());
    EXPECT_FALSE(RenderAttention({}, "Bad::Name", compact).ok());
    EXPECT_FALSE(RenderMap({}, "Bad::Name", nullptr, compact).ok());
    EXPECT_FALSE(RenderPositionEmbedding({}, "Bad::Name", {}, compact).ok());
    EXPECT_FALSE(
        RenderPositionEmbedding({{{1, 0, 2}}}, "Lookup", {}, compact).ok());
  }
}

}  // namespace
}  // namespace pluto::llm::discretized::generator
