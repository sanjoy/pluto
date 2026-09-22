#include "src/llm/experiments/memorize_general_facts/discretized_model/runtime.h"

#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::discretized {
namespace {

Model Fixture() {
  static const VocabularyRow vocabulary[] = {
      {10, "A"}, {11, "B"}, {12, "C"}, {13, "<eos>"}};
  static const EntryRow entry[] = {{0, 0, 4}, {1, 0, 7}, {1, 1, 5}, {2, 1, 6}};
  static const StateId keys[] = {4, 4, 5, 4, 6, 7};
  static const AttentionRow attention_rows[] = {
      {0, 1, 8}, {1, 2, 9}, {3, 2, 10}, {5, 1, 11}};
  static const AttentionTable attention[] = {{keys, attention_rows}};
  static const StateRow mlp_rows[] = {{8, 12}, {9, 13}, {10, 14}, {11, 15}};
  static const StateTable mlp[] = {{mlp_rows}};
  static const StateRow snap[] = {{12, 1}, {13, 3}, {14, 0}, {15, 3}};
  return Model{1024, 1, 3, vocabulary, entry, attention, mlp, {snap}};
}

Model FunctionFixture() {
  auto model = Fixture();
  model.entry = {};
  model.entry_function = [](TokenId token,
                            uint32_t position) -> TransitionResult {
    if (position == 0) {
      if (token == 0)
        return {4, true};
      if (token == 1)
        return {7, true};
    }
    if (position == 1 && token >= 1 && token <= 2)
      return {static_cast<StateId>(token + 4), true};
    return {};
  };
  static const AttentionTable attention[] = {
      {{}, {}, [](absl::Span<const StateId> prefix) -> TransitionResult {
         if (prefix.size() == 1 && prefix[0] == 4)
           return {8, true};
         if (prefix.size() == 1 && prefix[0] == 7)
           return {11, true};
         if (prefix.size() == 2 && prefix[0] == 4 && prefix[1] >= 5 &&
             prefix[1] <= 6)
           return {prefix[1] + 4, true};
         return {};
       }}};
  static const StateTable mlp[] = {{{}, [](StateId state) -> TransitionResult {
                                      if (state >= 8 && state <= 11)
                                        return {state + 4, true};
                                      return {};
                                    }}};
  model.attention = attention;
  model.mlp = mlp;
  model.snap = {{}, [](StateId state) -> TransitionResult {
                  switch (state) {
                    case 12:
                      return {1, true};
                    case 13:
                    case 15:
                      return {3, true};
                    case 14:
                      return {0, true};
                    default:
                      return {};
                  }
                }};
  return model;
}

TEST(IntegerRuntime, CompiledFunctionsMatchTablesIncludingUnsupportedInputs) {
  const auto table = Fixture();
  const auto functions = FunctionFixture();
  ASSERT_TRUE(ValidateModel(functions).ok());
  // Exhaust the small fixture domain, including repeated/reordered histories
  // and unsupported entries; merely matching the successful sentence is weak.
  for (int length = 1; length <= 4; ++length) {
    const int combinations = 1 << (2 * length);
    for (int encoded = 0; encoded < combinations; ++encoded) {
      std::vector<TokenId> tokens;
      for (int index = 0; index < length; ++index)
        tokens.push_back((encoded >> (2 * index)) & 3);
      const auto expected = PredictNext(table, tokens);
      const auto actual = PredictNext(functions, tokens);
      ASSERT_EQ(actual.status().code(), expected.status().code());
      EXPECT_EQ(actual.value_or(0), expected.value_or(0));
    }
  }
  auto generated = Generate(functions, std::vector<TokenId>{0}, 9);
  ASSERT_TRUE(generated.ok());
  EXPECT_EQ(*generated, (std::vector<TokenId>{1, 3}));
}

TEST(IntegerRuntime, RejectsAmbiguousRepresentationsAndInvalidFunctionOutputs) {
  auto model = FunctionFixture();
  model.entry = Fixture().entry;
  EXPECT_FALSE(ValidateModel(model).ok());
  model = FunctionFixture();
  auto attention = model.attention[0];
  attention.rows = Fixture().attention[0].rows;
  model.attention = absl::Span<const AttentionTable>(&attention, 1);
  EXPECT_FALSE(ValidateModel(model).ok());
  model = FunctionFixture();
  model.snap.rows = Fixture().snap.rows;
  EXPECT_FALSE(ValidateModel(model).ok());

  model = FunctionFixture();
  model.entry_function = [](TokenId, uint32_t) -> TransitionResult {
    return {0, true};
  };
  EXPECT_EQ(PredictNext(model, std::vector<TokenId>{0}).status().code(),
            absl::StatusCode::kDataLoss);
  model = FunctionFixture();
  const StateTable invalid_mlp[] = {
      {{}, [](StateId) -> TransitionResult { return {1, true}; }}};
  model.mlp = invalid_mlp;
  EXPECT_EQ(PredictNext(model, std::vector<TokenId>{0}).status().code(),
            absl::StatusCode::kDataLoss);
}

TEST(IntegerRuntime, ExecutesAllBoundariesAndPredictsEosAutoregressively) {
  const auto model = Fixture();
  ASSERT_TRUE(ValidateModel(model).ok());
  auto result = Generate(model, std::vector<TokenId>{0}, 9);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(*result, (std::vector<TokenId>{1, 3}));
  auto text = Decode(model, std::vector<TokenId>{0, 1, 2});
  ASSERT_TRUE(text.ok());
  EXPECT_EQ(*text, "ABC");
}

TEST(IntegerRuntime, CompleteHistoryMattersEvenWithIdenticalLastState) {
  const auto model = Fixture();
  auto good = PredictNext(model, std::vector<TokenId>{0, 1});
  ASSERT_TRUE(good.ok());
  EXPECT_EQ(*good, 3);
  auto bad = PredictNext(model, std::vector<TokenId>{1, 1});
  EXPECT_EQ(bad.status().code(), absl::StatusCode::kNotFound);
  EXPECT_NE(bad.status().message().find("attention history"),
            absl::string_view::npos);
}

TEST(IntegerRuntime, TablesDetermineAnswersWithoutCorpusIdentityOrSuffixCache) {
  auto model = Fixture();
  std::vector<StateRow> snap(model.snap.rows.begin(), model.snap.rows.end());
  snap[1].output = 2;
  model.snap.rows = snap;
  auto changed = PredictNext(model, std::vector<TokenId>{0, 1});
  ASSERT_TRUE(changed.ok());
  EXPECT_EQ(*changed, 2);
  model = Fixture();
  std::vector<StateRow> mlp(model.mlp[0].rows.begin(), model.mlp[0].rows.end());
  mlp[1].output = 14;
  const StateTable changed_mlp[] = {{mlp}};
  model.mlp = changed_mlp;
  changed = PredictNext(model, std::vector<TokenId>{0, 1});
  ASSERT_TRUE(changed.ok());
  EXPECT_EQ(*changed, 0);
  model = Fixture();
  std::vector<EntryRow> entry(model.entry.begin(), model.entry.end());
  entry[2].state = 6;
  model.entry = entry;
  changed = PredictNext(model, std::vector<TokenId>{0, 1});
  ASSERT_TRUE(changed.ok());
  EXPECT_EQ(*changed, 0);
}

TEST(IntegerRuntime, UnknownEntryMlpAndReadoutFailExplicitly) {
  auto model = Fixture();
  auto unknown = PredictNext(model, std::vector<TokenId>{2});
  EXPECT_EQ(unknown.status().code(), absl::StatusCode::kNotFound);
  EXPECT_NE(unknown.status().message().find("token/position"),
            absl::string_view::npos);
  const StateTable missing_mlp[] = {{model.mlp[0].rows.subspan(1)}};
  model.mlp = missing_mlp;
  unknown = PredictNext(model, std::vector<TokenId>{0});
  EXPECT_EQ(unknown.status().code(), absl::StatusCode::kNotFound);
  EXPECT_NE(unknown.status().message().find("MLP"), absl::string_view::npos);
  model = Fixture();
  model.snap.rows = model.snap.rows.subspan(1);
  unknown = PredictNext(model, std::vector<TokenId>{0});
  EXPECT_EQ(unknown.status().code(), absl::StatusCode::kNotFound);
  EXPECT_NE(unknown.status().message().find("snap"), absl::string_view::npos);
}

TEST(IntegerRuntime, ZeroBudgetAndInvalidPrompts) {
  const auto model = Fixture();
  auto empty = Generate(model, std::vector<TokenId>{0}, 0);
  ASSERT_TRUE(empty.ok());
  EXPECT_TRUE(empty->empty());
  EXPECT_FALSE(PredictNext(model, {}).ok());
  EXPECT_FALSE(Generate(model, {}, 0).ok());
  EXPECT_FALSE(PredictNext(model, std::vector<TokenId>{-1}).ok());
  EXPECT_FALSE(PredictNext(model, std::vector<TokenId>{4}).ok());
  EXPECT_FALSE(PredictNext(model, std::vector<TokenId>(1025, 0)).ok());
  EXPECT_FALSE(Decode(model, std::vector<TokenId>{-1}).ok());
}

TEST(IntegerRuntime, IndependentCallsAreBitExactAndCannotLeakHistory) {
  const auto model = Fixture();
  for (int i = 0; i < 12; ++i) {
    auto a = PredictNext(model, std::vector<TokenId>{0, 1});
    auto b = PredictNext(model, std::vector<TokenId>{0, 2});
    ASSERT_TRUE(a.ok());
    ASSERT_TRUE(b.ok());
    EXPECT_EQ(*a, 3);
    EXPECT_EQ(*b, 0);
  }
}

TEST(IntegerRuntime, RejectsMalformedKeyRangesAndUnsortedTables) {
  auto model = Fixture();
  const AttentionRow broken_rows[] = {{1000, 2, 9}};
  const AttentionTable broken_attention[] = {
      {model.attention[0].keys, broken_rows}};
  model.attention = broken_attention;
  EXPECT_FALSE(ValidateModel(model).ok());
  model = Fixture();
  const EntryRow duplicate[] = {{0, 0, 4}, {0, 0, 5}};
  model.entry = duplicate;
  EXPECT_FALSE(ValidateModel(model).ok());
  model = Fixture();
  const StateRow bad_snap[] = {{12, 4}};
  model.snap.rows = bad_snap;
  EXPECT_FALSE(ValidateModel(model).ok());
}

}  // namespace
}  // namespace pluto::llm::discretized
