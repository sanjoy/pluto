#include "src/llm/experiments/memorize_general_facts/discretized_model/runtime.h"

#include <initializer_list>
#include <limits>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::discretized {
namespace {

// Keep callback results distinct from arbitrary optional states and token IDs.
static_assert(!std::is_convertible_v<std::optional<DiscreteHiddenState>,
                                     TransitionResult>);
static_assert(!std::is_convertible_v<DiscreteHiddenState, TransitionResult>);
static_assert(!std::is_convertible_v<DiscreteToken, DiscreteHiddenState>);
static_assert(!std::is_convertible_v<DiscreteHiddenState, DiscreteToken>);
static_assert(!std::is_convertible_v<int, DiscreteHiddenState>);
static_assert(!std::is_convertible_v<int, DiscreteToken>);
static_assert(!std::is_convertible_v<DiscreteHiddenState, int>);
static_assert(!std::is_convertible_v<DiscreteToken, int>);
static_assert(sizeof(DiscreteHiddenState) == sizeof(int));
static_assert(sizeof(DiscreteToken) == sizeof(int));

std::vector<DiscreteToken> Tokens(std::initializer_list<int> values) {
  std::vector<DiscreteToken> tokens;
  for (int value : values)
    tokens.push_back(DiscreteToken{value});
  return tokens;
}

TEST(DiscreteIds, ExplicitConversionsPreserveLabels) {
  for (int value : {0, 1, 4474, std::numeric_limits<int>::max(), -1}) {
    const DiscreteToken token{value};
    const auto state = static_cast<DiscreteHiddenState>(token);
    EXPECT_EQ(state.value, value);
    EXPECT_EQ(static_cast<DiscreteToken>(state), token);
  }
  constexpr DiscreteToken token{42};
  constexpr auto state = static_cast<DiscreteHiddenState>(token);
  static_assert(state == DiscreteHiddenState{42});
  static_assert(static_cast<DiscreteToken>(state) == token);
}

TransitionResult Entry(DiscreteToken token, uint32_t position) {
  if (position == 0) {
    if (token.value == 0)
      return {DiscreteHiddenState{4}};
    if (token.value == 1)
      return {DiscreteHiddenState{7}};
  }
  if (position == 1 && token.value >= 1 && token.value <= 2)
    return {DiscreteHiddenState{token.value + 4}};
  return {};
}

TransitionResult Attention(absl::Span<const DiscreteHiddenState> prefix) {
  if (prefix.size() == 1 && prefix[0].value == 4)
    return {DiscreteHiddenState{8}};
  if (prefix.size() == 1 && prefix[0].value == 7)
    return {DiscreteHiddenState{11}};
  if (prefix.size() == 2 && prefix[0].value == 4 && prefix[1].value >= 5 &&
      prefix[1].value <= 6)
    return {DiscreteHiddenState{prefix[1].value + 4}};
  return {};
}

TransitionResult Mlp(DiscreteHiddenState state) {
  if (state.value >= 8 && state.value <= 11)
    return {DiscreteHiddenState{state.value + 4}};
  return {};
}

TransitionResult LanguageModelingHead(DiscreteHiddenState state) {
  switch (state.value) {
    case 12:
      return {DiscreteHiddenState{1}};
    case 13:
    case 15:
      return {DiscreteHiddenState{3}};
    case 14:
      return {DiscreteHiddenState{0}};
    default:
      return {};
  }
}

Model Fixture() {
  static const VocabularyRow vocabulary[] = {
      {10, "A"}, {11, "B"}, {12, "C"}, {13, "<eos>"}};
  static const AttentionTable attention[] = {{Attention}};
  static const StateTable mlp[] = {{Mlp}};
  return Model{1024,      1,   DiscreteToken{3},       vocabulary,
               attention, mlp, {LanguageModelingHead}, Entry};
}

TEST(IntegerRuntime, FunctionsMatchEntireFiniteDomain) {
  const auto model = Fixture();
  ASSERT_TRUE(ValidateModel(model).ok());
  // These are all four supported prompts, specified independently of the
  // boundary functions. Exhaust repeated, reordered, and extended histories
  // too: matching only the successful generated sentence would be weak.
  for (int length = 1; length <= 4; ++length) {
    const int combinations = 1 << (2 * length);
    for (int encoded = 0; encoded < combinations; ++encoded) {
      std::vector<DiscreteToken> tokens;
      for (int index = 0; index < length; ++index)
        tokens.push_back(DiscreteToken{(encoded >> (2 * index)) & 3});
      std::optional<DiscreteToken> expected;
      if (tokens == Tokens({0}))
        expected = DiscreteToken{1};
      else if (tokens == Tokens({1}) || tokens == (Tokens({0, 1})))
        expected = DiscreteToken{3};
      else if (tokens == (Tokens({0, 2})))
        expected = DiscreteToken{0};
      const auto actual = PredictNext(model, tokens);
      EXPECT_EQ(actual.status().code(), expected.has_value()
                                            ? absl::StatusCode::kOk
                                            : absl::StatusCode::kNotFound);
      EXPECT_EQ(actual.value_or(DiscreteToken{}),
                expected.value_or(DiscreteToken{}));
    }
  }
}

TEST(IntegerRuntime, RejectsMissingRequiredFunctionsBeforeCallingThem) {
  const AttentionTable missing_attention[] = {{nullptr}};
  const StateTable missing_mlp[] = {{nullptr}};
  for (int boundary = 0; boundary < 4; ++boundary) {
    SCOPED_TRACE(boundary);
    auto model = Fixture();
    switch (boundary) {
      case 0:
        model.entry_function = nullptr;
        break;
      case 1:
        model.attention = missing_attention;
        break;
      case 2:
        model.mlp = missing_mlp;
        break;
      case 3:
        model.language_modeling_head.function = nullptr;
        break;
    }
    EXPECT_EQ(ValidateModel(model).code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(PredictNext(model, Tokens({0})).status().code(),
              absl::StatusCode::kInvalidArgument);
    // Even a zero generation budget must validate the model and never invoke a
    // null callback; inference does not require a prior ValidateModel call.
    for (size_t budget : {0, 1})
      EXPECT_EQ(Generate(model, Tokens({0}), budget).status().code(),
                absl::StatusCode::kInvalidArgument);
  }

  // Validation covers every block, not just the first one.
  auto model = Fixture();
  const AttentionTable attention[] = {{Attention}, {Attention}};
  const StateTable mlp[] = {{Mlp}, {nullptr}};
  model.attention = attention;
  model.mlp = mlp;
  EXPECT_EQ(ValidateModel(model).code(), absl::StatusCode::kInvalidArgument);
  const AttentionTable bad_attention[] = {{Attention}, {nullptr}};
  const StateTable good_mlp[] = {{Mlp}, {Mlp}};
  model.attention = bad_attention;
  model.mlp = good_mlp;
  EXPECT_EQ(ValidateModel(model).code(), absl::StatusCode::kInvalidArgument);
}

TEST(IntegerRuntime, RejectsInvalidFunctionOutputs) {
  auto model = Fixture();
  model.entry_function = [](DiscreteToken, uint32_t) -> TransitionResult {
    return {DiscreteHiddenState{0}};
  };
  EXPECT_EQ(PredictNext(model, Tokens({0})).status().code(),
            absl::StatusCode::kDataLoss);
  model = Fixture();
  const AttentionTable invalid_attention[] = {
      {[](absl::Span<const DiscreteHiddenState>) -> TransitionResult {
        return {DiscreteHiddenState{2}};
      }}};
  model.attention = invalid_attention;
  EXPECT_EQ(PredictNext(model, Tokens({0})).status().code(),
            absl::StatusCode::kDataLoss);
  model = Fixture();
  const StateTable invalid_mlp[] = {
      {[](DiscreteHiddenState) -> TransitionResult {
        return {DiscreteHiddenState{1}};
      }}};
  model.mlp = invalid_mlp;
  EXPECT_EQ(PredictNext(model, Tokens({0})).status().code(),
            absl::StatusCode::kDataLoss);
  model = Fixture();
  model.language_modeling_head = {[](DiscreteHiddenState) -> TransitionResult {
    return {DiscreteHiddenState{4}};
  }};
  EXPECT_EQ(PredictNext(model, Tokens({0})).status().code(),
            absl::StatusCode::kDataLoss);
  model.language_modeling_head = {[](DiscreteHiddenState) -> TransitionResult {
    return {DiscreteHiddenState{std::numeric_limits<int>::max()}};
  }};
  EXPECT_EQ(PredictNext(model, Tokens({0})).status().code(),
            absl::StatusCode::kDataLoss);
}

TEST(IntegerRuntime, OptionalTransitionsDistinguishZeroFromUnsupported) {
  auto model = Fixture();
  const auto zero =
      model.language_modeling_head.function(DiscreteHiddenState{14});
  ASSERT_TRUE(zero.output.has_value());
  EXPECT_EQ(*zero.output, DiscreteHiddenState{0});
  EXPECT_EQ(
      model.language_modeling_head.function(DiscreteHiddenState{16}).output,
      std::nullopt);
  const auto prediction = PredictNext(model, Tokens({0, 2}));
  ASSERT_TRUE(prediction.ok());
  EXPECT_EQ(*prediction, DiscreteToken{0});

  // Each callback propagates an empty optional as unsupported, rather than
  // interpreting an absent value as vocabulary ID zero or using a fallback.
  model.entry_function = [](DiscreteToken, uint32_t) -> TransitionResult {
    return {};
  };
  EXPECT_EQ(PredictNext(model, Tokens({0})).status().code(),
            absl::StatusCode::kNotFound);
  model = Fixture();
  const AttentionTable missing_attention[] = {
      {[](absl::Span<const DiscreteHiddenState>) -> TransitionResult {
        return {};
      }}};
  model.attention = missing_attention;
  EXPECT_EQ(PredictNext(model, Tokens({0})).status().code(),
            absl::StatusCode::kNotFound);
  model = Fixture();
  const StateTable missing_mlp[] = {
      {[](DiscreteHiddenState) -> TransitionResult { return {}; }}};
  model.mlp = missing_mlp;
  EXPECT_EQ(PredictNext(model, Tokens({0})).status().code(),
            absl::StatusCode::kNotFound);
  model = Fixture();
  model.language_modeling_head = {
      [](DiscreteHiddenState) -> TransitionResult { return {}; }};
  EXPECT_EQ(PredictNext(model, Tokens({0})).status().code(),
            absl::StatusCode::kNotFound);
}

TEST(IntegerRuntime, RejectsNegativeLabelsAtEveryBoundary) {
  const AttentionTable attention[] = {
      {[](absl::Span<const DiscreteHiddenState>) -> TransitionResult {
        return {DiscreteHiddenState{-1}};
      }}};
  const StateTable pointwise[] = {{[](DiscreteHiddenState) -> TransitionResult {
    return {DiscreteHiddenState{-1}};
  }}};
  for (int boundary = 0; boundary < 4; ++boundary) {
    auto model = Fixture();
    if (boundary == 0)
      model.entry_function = [](DiscreteToken, uint32_t) -> TransitionResult {
        return {DiscreteHiddenState{-1}};
      };
    else if (boundary == 1)
      model.attention = attention;
    else if (boundary == 2)
      model.mlp = pointwise;
    else
      model.language_modeling_head = pointwise[0];
    EXPECT_EQ(PredictNext(model, Tokens({0})).status().code(),
              absl::StatusCode::kDataLoss);
  }
}

TEST(IntegerRuntime, ExecutesAllBoundariesAndPredictsEosAutoregressively) {
  const auto model = Fixture();
  ASSERT_TRUE(ValidateModel(model).ok());
  auto result = Generate(model, Tokens({0}), 9);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(*result, (Tokens({1, 3})));
  auto text = Decode(model, Tokens({0, 1, 2}));
  ASSERT_TRUE(text.ok());
  EXPECT_EQ(*text, "ABC");
}

TEST(IntegerRuntime, EachBlockConsumesThePreviousBlocksWholeOutput) {
  auto model = Fixture();
  const AttentionTable attention[] = {
      {Attention},
      {[](absl::Span<const DiscreteHiddenState> prefix) -> TransitionResult {
        if (prefix.size() == 1 && prefix[0].value == 12)
          return {DiscreteHiddenState{16}};
        if (prefix.size() == 2 && prefix[0].value == 12 &&
            prefix[1].value == 13)
          return {DiscreteHiddenState{17}};
        return {};
      }}};
  const StateTable mlp[] = {{Mlp},
                            {[](DiscreteHiddenState state) -> TransitionResult {
                              if (state.value >= 16 && state.value <= 17)
                                return {DiscreteHiddenState{state.value + 4}};
                              return {};
                            }}};
  model.attention = attention;
  model.mlp = mlp;
  model.language_modeling_head = {
      [](DiscreteHiddenState state) -> TransitionResult {
        if (state.value == 20)
          return {DiscreteHiddenState{1}};
        if (state.value == 21)
          return {DiscreteHiddenState{3}};
        return {};
      }};
  const auto generated = Generate(model, Tokens({0}), 9);
  ASSERT_TRUE(generated.ok()) << generated.status();
  EXPECT_EQ(*generated, (Tokens({1, 3})));
}

TEST(IntegerRuntime, CompleteHistoryMattersEvenWithIdenticalLastState) {
  const auto model = Fixture();
  auto good = PredictNext(model, Tokens({0, 1}));
  ASSERT_TRUE(good.ok());
  EXPECT_EQ(*good, DiscreteToken{3});
  auto bad = PredictNext(model, Tokens({1, 1}));
  EXPECT_EQ(bad.status().code(), absl::StatusCode::kNotFound);
  EXPECT_NE(bad.status().message().find("attention history"),
            absl::string_view::npos);
}

TEST(IntegerRuntime, FunctionsDetermineAnswersWithoutCorpusOrSuffixCache) {
  auto model = Fixture();
  model.language_modeling_head = {
      [](DiscreteHiddenState state) -> TransitionResult {
        return state.value == 13 ? TransitionResult{DiscreteHiddenState{2}}
                                 : LanguageModelingHead(state);
      }};
  auto changed = PredictNext(model, Tokens({0, 1}));
  ASSERT_TRUE(changed.ok());
  EXPECT_EQ(*changed, DiscreteToken{2});

  model = Fixture();
  const StateTable changed_mlp[] = {
      {[](DiscreteHiddenState state) -> TransitionResult {
        return state.value == 9 ? TransitionResult{DiscreteHiddenState{14}}
                                : Mlp(state);
      }}};
  model.mlp = changed_mlp;
  changed = PredictNext(model, Tokens({0, 1}));
  ASSERT_TRUE(changed.ok());
  EXPECT_EQ(*changed, DiscreteToken{0});

  model = Fixture();
  const AttentionTable changed_attention[] = {
      {[](absl::Span<const DiscreteHiddenState> prefix) -> TransitionResult {
        if (prefix.size() == 2 && prefix[0].value == 4 && prefix[1].value == 5)
          return {DiscreteHiddenState{10}};
        return Attention(prefix);
      }}};
  model.attention = changed_attention;
  changed = PredictNext(model, Tokens({0, 1}));
  ASSERT_TRUE(changed.ok());
  EXPECT_EQ(*changed, DiscreteToken{0});

  model = Fixture();
  model.entry_function = [](DiscreteToken token,
                            uint32_t position) -> TransitionResult {
    return token.value == 1 && position == 1
               ? TransitionResult{DiscreteHiddenState{6}}
               : Entry(token, position);
  };
  changed = PredictNext(model, Tokens({0, 1}));
  ASSERT_TRUE(changed.ok());
  EXPECT_EQ(*changed, DiscreteToken{0});
}

TEST(IntegerRuntime, ZeroBudgetAndInvalidPrompts) {
  const auto model = Fixture();
  auto empty = Generate(model, Tokens({0}), 0);
  ASSERT_TRUE(empty.ok());
  EXPECT_TRUE(empty->empty());
  EXPECT_FALSE(PredictNext(model, {}).ok());
  EXPECT_FALSE(Generate(model, {}, 0).ok());
  EXPECT_FALSE(PredictNext(model, Tokens({-1})).ok());
  EXPECT_FALSE(PredictNext(model, Tokens({4})).ok());
  EXPECT_FALSE(
      PredictNext(model, std::vector<DiscreteToken>(1025, DiscreteToken{0}))
          .ok());
  EXPECT_FALSE(Decode(model, Tokens({-1})).ok());
  EXPECT_FALSE(Decode(model, Tokens({4})).ok());
}

TEST(IntegerRuntime, GenerationHonorsBudgetContextAndUnknownHistories) {
  auto model = Fixture();
  auto one = Generate(model, Tokens({0}), 1);
  ASSERT_TRUE(one.ok());
  EXPECT_EQ(*one, (Tokens({1})));
  model.context_length = 1;
  auto full = Generate(model, Tokens({0}), 9);
  ASSERT_TRUE(full.ok());
  EXPECT_TRUE(full->empty());
  model = Fixture();
  EXPECT_EQ(Generate(model, Tokens({0, 2}), 2).status().code(),
            absl::StatusCode::kNotFound);
}

TEST(IntegerRuntime, IndependentCallsAreBitExactAndCannotLeakHistory) {
  const auto model = Fixture();
  for (int i = 0; i < 12; ++i) {
    auto a = PredictNext(model, Tokens({0, 1}));
    auto b = PredictNext(model, Tokens({0, 2}));
    ASSERT_TRUE(a.ok());
    ASSERT_TRUE(b.ok());
    EXPECT_EQ(*a, DiscreteToken{3});
    EXPECT_EQ(*b, DiscreteToken{0});
  }
}

TEST(IntegerRuntime, RejectsInvalidModelDimensionsBeforeInference) {
  for (int field = 0; field < 7; ++field) {
    SCOPED_TRACE(field);
    auto model = Fixture();
    switch (field) {
      case 0:
        model.context_length = 0;
        break;
      case 1:
        model.prompt_tokens = 0;
        break;
      case 2:
        model.prompt_tokens = model.context_length + 1;
        break;
      case 3:
        model.vocabulary = {};
        break;
      case 4:
        model.eos_token = DiscreteToken{-1};
        break;
      case 5:
        model.eos_token =
            DiscreteToken{static_cast<int>(model.vocabulary.size())};
        break;
      case 6:
        model.mlp = {};
        break;
    }
    EXPECT_EQ(ValidateModel(model).code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(PredictNext(model, Tokens({0})).status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(Generate(model, Tokens({0}), 0).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(IntegerRuntime,
     ZeroTransformerBlocksStillRunEntryAndLanguageModelingHead) {
  auto model = Fixture();
  model.attention = {};
  model.mlp = {};
  model.language_modeling_head = {
      [](DiscreteHiddenState state) -> TransitionResult {
        if (state.value == 4)
          return {DiscreteHiddenState{1}};
        if (state.value == 5)
          return {DiscreteHiddenState{3}};
        return {};
      }};
  ASSERT_TRUE(ValidateModel(model).ok());
  const auto generated = Generate(model, Tokens({0}), 9);
  ASSERT_TRUE(generated.ok()) << generated.status();
  EXPECT_EQ(*generated, (Tokens({1, 3})));
}

}  // namespace
}  // namespace pluto::llm::discretized
