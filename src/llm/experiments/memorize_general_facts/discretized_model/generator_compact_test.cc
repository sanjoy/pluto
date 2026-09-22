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
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_attention_logic.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_pointwise.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_io.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized::generator {
namespace {

using absl::StrCat;

SymbolicModel Fixture() {
  SymbolicModel model;
  model.metadata = {1, 2, 3, 2, 1, {{0, "A"}, {1, "B"}, {2, "C"}}};
  model.entry = {{0, 0, 100}, {1, 0, 101}, {1, 1, 101}};
  model.transformers = {{{{{100}, 102}, {{100, 101}, 103}, {{101}, 103}},
                         {{102, 105}, {103, 104}}},
                        {{{{105}, 106}, {{105, 104}, 107}, {{104}, 107}},
                         {{106, 109}, {107, 108}}}};
  model.language_modeling_head = {{108, 2}, {109, 1}};
  model.samples = {{{0, 1}}, {{1}}};
  model.stats.membership_complete = true;
  for (int i = 0; i < 10; ++i)
    model.states.push_back({100 + i,
                            i / 2,
                            {static_cast<uint16_t>(100 + i)},
                            std::vector<int>{1000 + i}});
  return model;
}

// Compile emitted programs independently of the neural runtime and exercise
// their entire small input domains, including negative and missing symbols.
// This catches emitter bugs that checking statistics or text alone cannot.
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
  EXPECT_EQ(program->trie_nodes, 5u);
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
  auto first = RenderAttention("Attention0", rows);
  auto original = rows;
  std::reverse(rows.begin(), rows.end());
  rows.push_back(rows.back());
  auto second = RenderAttention("Attention0", rows);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(first->source, second->source);
  EXPECT_EQ(first->stats, second->stats);
  EXPECT_EQ(*BuildAttention(original), *BuildAttention(rows));
}

TEST(AttentionLogicTest, InvalidRowsNamesAndEmissionOptionsReturnErrors) {
  // Row shape and integer representation are now enforced by C++ types.
  for (const auto& rows : std::vector<std::vector<AttentionTransition>>{
           {{{1}, 2}, {{1}, 3}}, {{{}, 0}}, {{{-1}, 1}}, {{{1}, -1}}})
    EXPECT_FALSE(BuildAttention(rows).ok());
  for (const char* name :
       {"class", "Bad::Name", "1Bad", "_Reserved", "Bad__Name", "x;}"})
    EXPECT_FALSE(RenderAttention(name, {}).ok()) << name;
  EXPECT_FALSE(RenderAttention("Attention", {}, 0).ok());
  EXPECT_FALSE(RenderAttention("Attention", {}, -1).ok());
  EXPECT_FALSE(RenderAttention("Attention", {}, 1, "approximate").ok());
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
  auto hybrid = RenderAttention("NarrowAttention", rows);
  auto pure = RenderAttention("PureAttention", rows, 256, "control_flow");
  ASSERT_TRUE(hybrid.ok()) << hybrid.status();
  ASSERT_TRUE(pure.ok()) << pure.status();
  EXPECT_GT(hybrid->stats.literal_sequence_steps, 4);
  EXPECT_EQ(hybrid->stats.literal_sequence_word_bits, 16);
  EXPECT_NE(hybrid->source.find("std::uint16_t symbol, output"),
            std::string::npos);
  EXPECT_EQ(pure->stats.literal_sequence_patterns, 0);
  EXPECT_EQ(hybrid->stats.source_bytes, hybrid->source.size());
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
  auto hybrid = RenderAttention("CompiledAttention", rows, 7);
  auto pure = RenderAttention("PureAttention", rows, 7, "control_flow");
  auto empty = RenderAttention("EmptyAttention", {});
  std::vector<AttentionTransition> narrow_rows;
  std::vector<int> narrow_key;
  for (int length = 1; length <= 8; ++length) {
    narrow_key.push_back(length);
    narrow_rows.push_back({narrow_key, length - 1});
  }
  auto narrow = RenderAttention("NarrowAttention", narrow_rows);
  ASSERT_TRUE(hybrid.ok()) << hybrid.status();
  ASSERT_TRUE(pure.ok()) << pure.status();
  ASSERT_TRUE(empty.ok()) << empty.status();
  ASSERT_TRUE(narrow.ok()) << narrow.status();
  EXPECT_GT(hybrid->stats.helpers, 1);
  EXPECT_GT(hybrid->stats.literal_sequence_patterns, 0);
  EXPECT_EQ(hybrid->stats.literal_sequence_word_bits, 32);
  EXPECT_EQ(narrow->stats.literal_sequence_word_bits, 16);
  EXPECT_GT(narrow->stats.literal_sequence_patterns, 0);
  std::string source = StrCat(kDeclarations, hybrid->source, pure->source,
                              empty->source, narrow->source, "int main() {\n");
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

TEST(PointwiseTest, RelabelingPreservesBoundaryOwnershipMembersAndBits) {
  auto original = Fixture();
  auto before = original;
  auto result = RelabelMlpOutputs(original);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(original, before);
  EXPECT_EQ(result->transformers[0].mlp,
            (std::vector<StateTransition>{{102, 104}, {103, 105}}));
  EXPECT_EQ(result->transformers[1].mlp,
            (std::vector<StateTransition>{{107, 110}, {108, 111}}));
  EXPECT_EQ(result->language_modeling_head,
            (std::vector<StateTransition>{{110, 1}, {111, 2}}));
  EXPECT_EQ(result->state_relabeling.size(), original.states.size());
  ASSERT_TRUE(result->stats.pointwise_relabeling.has_value());
  EXPECT_EQ(result->stats.pointwise_relabeling->changed_states, 6);
  EXPECT_TRUE(
      result->stats.pointwise_relabeling->vocabulary_aligned_final_boundaries);
  for (const auto& row : result->state_relabeling) {
    SymbolicState old_state, new_state;
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
  nonbijective.transformers[0].mlp[1].output =
      nonbijective.transformers[0].mlp[0].output;
  auto result = RelabelMlpOutputs(nonbijective);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_TRUE(result->stats.pointwise_relabeling.has_value());
  EXPECT_EQ(result->stats.pointwise_relabeling->skipped_layers,
            std::vector<int>{0});
  EXPECT_EQ(result->transformers[0].mlp, nonbijective.transformers[0].mlp);
  for (int defect = 0; defect < 3; ++defect) {
    auto model = Fixture();
    if (defect == 0)
      model.language_modeling_head.erase(model.language_modeling_head.begin() +
                                         1);
    if (defect == 1)
      model.language_modeling_head[1].output =
          model.language_modeling_head[0].output;
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
  auto result = RenderPointwise("Head", rows, &names);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->stats.representation,
            TransitionRepresentation::kSparseAffineSupportMask);
  EXPECT_EQ(result->stats.table_bytes, 16);
  EXPECT_NE(result->source.find("vocab::Token0"), std::string::npos);
  EXPECT_EQ(result->source.find("kOutputs"), std::string::npos);
  EXPECT_LT(result->source.size(), 1000u);
}

TEST(PointwiseTest, EvaluatorsDistinguishUnsupportedAndSupportedZero) {
  EXPECT_EQ(*EvaluatePointwise({{3, 0}}, 3), 0);
  EXPECT_FALSE(EvaluatePointwise({{3, 0}}, 4)->has_value());
  EXPECT_EQ(*EvaluateEntry({{1, 2, 0}}, 1, 2), 0);
  EXPECT_FALSE(EvaluateEntry({{1, 2, 0}}, 1, 3)->has_value());
  EXPECT_FALSE(RenderPointwise("Bad", {{1, 3}, {1, 4}}).ok());
  EXPECT_FALSE(
      RenderEntry("Bad", {{1, 2, 3}, {1, 2, 4}}, {{1, "vocab::B"}}).ok());
}

TEST(PointwiseTest, DenseMapsChooseGuardedAffineOrNamedOutputArray) {
  auto affine = RenderPointwise("Mlp", {{10, 20}, {11, 21}, {12, 22}});
  ASSERT_TRUE(affine.ok()) << affine.status();
  EXPECT_EQ(affine->stats.representation,
            TransitionRepresentation::kGuardedAffine);
  EXPECT_EQ(affine->stats.table_bytes, 0);
  EXPECT_NE(affine->source.find("state.value < 10 || state.value > 12"),
            std::string::npos);
  std::vector<StateTransition> rows;
  TokenNames names;
  for (int i = 0; i < 20; ++i) {
    rows.push_back({10 + i, (i * 7) % 20});
    names[i] = StrCat("vocab::Token", i);
  }
  auto irregular = RenderPointwise("Head", rows, &names);
  ASSERT_TRUE(irregular.ok()) << irregular.status();
  EXPECT_EQ(irregular->stats.representation,
            TransitionRepresentation::kGuardedOutputArray);
  EXPECT_EQ(irregular->stats.table_bytes, 40);
  EXPECT_NE(irregular->source.find("vocab::Token19"), std::string::npos);
}

TEST(PointwiseTest, EntryRetainsPositionSupportAndNamedExceptions) {
  auto result = RenderEntry(
      "Entry", {{0, 0, 20}, {0, 2, 20}, {0, 3, 21}, {1, 1, 21}, {3, 0, 20}},
      {{0, "vocab::A"}, {1, "vocab::B"}, {3, "vocab::D"}});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->stats.position_exceptions, 1);
  for (const char* text : {"case vocab::A.value:", "position == 3",
                           "position < 0", "position > 3", "return {};"})
    EXPECT_NE(result->source.find(text), std::string::npos) << text;
  EXPECT_EQ(result->stats.source_bytes, result->source.size());
}

TEST(PointwiseTest, InvalidIdsPositionsAndMissingNamesFailSafely) {
  for (int bad : {-1, std::numeric_limits<int>::min()}) {
    EXPECT_FALSE(RenderPointwise("Bad", {{bad, 0}}).ok());
    EXPECT_FALSE(RenderPointwise("Bad", {{0, bad}}).ok());
    EXPECT_FALSE(RenderEntry("Bad", {{bad, 0, 0}}, {{0, "vocab::A"}}).ok());
    EXPECT_FALSE(RenderEntry("Bad", {{0, bad, 0}}, {{0, "vocab::A"}}).ok());
    EXPECT_FALSE(RenderEntry("Bad", {{0, 0, bad}}, {{0, "vocab::A"}}).ok());
  }
  TokenNames names;
  EXPECT_FALSE(RenderPointwise("Bad", {{0, 1}}, &names).ok());
  EXPECT_FALSE(RenderEntry("Bad", {{0, 0, 1}}, names).ok());
  for (const char* name :
       {"class", "Bad::Name", "1Bad", "_Reserved", "Bad__Name", "x;}"}) {
    EXPECT_FALSE(RenderPointwise(name, {}).ok()) << name;
    EXPECT_FALSE(RenderEntry(name, {}, names).ok()) << name;
  }
  auto largest = RenderPointwise("Largest", {{2147483647, 2147483647}});
  ASSERT_TRUE(largest.ok()) << largest.status();
  auto largest_entry =
      RenderEntry("LargestEntry", {{2147483647, 2147483647, 2147483647}},
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
    auto generated = RenderPointwise(name, rows,
                                     name == "Named"               ? &names
                                     : name == "NamedSparseAffine" ? &numbered
                                                                   : nullptr);
    ASSERT_TRUE(generated.ok()) << generated.status();
    absl::StrAppend(&source, generated->source);
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
    auto generated = RenderEntry(name, rows, names);
    ASSERT_TRUE(generated.ok()) << generated.status();
    absl::StrAppend(&source, generated->source);
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

}  // namespace
}  // namespace pluto::llm::discretized::generator
