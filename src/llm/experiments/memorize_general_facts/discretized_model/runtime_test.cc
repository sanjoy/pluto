#include "src/llm/experiments/memorize_general_facts/discretized_model/runtime.h"

#include <limits>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::discretized {
namespace {

// Keep callback results distinct from arbitrary optional states and token IDs.
static_assert(!std::is_convertible_v<std::optional<StateId>, TransitionResult>);
static_assert(!std::is_convertible_v<StateId, TransitionResult>);

TransitionResult Entry(TokenId token, uint32_t position) {
  if (position == 0) {
    if (token == 0)
      return {4};
    if (token == 1)
      return {7};
  }
  if (position == 1 && token >= 1 && token <= 2)
    return {static_cast<StateId>(token + 4)};
  return {};
}

TransitionResult Attention(absl::Span<const StateId> prefix) {
  if (prefix.size() == 1 && prefix[0] == 4)
    return {8};
  if (prefix.size() == 1 && prefix[0] == 7)
    return {11};
  if (prefix.size() == 2 && prefix[0] == 4 && prefix[1] >= 5 && prefix[1] <= 6)
    return {prefix[1] + 4};
  return {};
}

TransitionResult Mlp(StateId state) {
  if (state >= 8 && state <= 11)
    return {state + 4};
  return {};
}

TransitionResult Snap(StateId state) {
  switch (state) {
    case 12:
      return {1};
    case 13:
    case 15:
      return {3};
    case 14:
      return {0};
    default:
      return {};
  }
}

Model Fixture() {
  static const VocabularyRow vocabulary[] = {
      {10, "A"}, {11, "B"}, {12, "C"}, {13, "<eos>"}};
  static const AttentionTable attention[] = {{Attention}};
  static const StateTable mlp[] = {{Mlp}};
  return Model{1024, 1, 3, vocabulary, attention, mlp, {Snap}, Entry};
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
      std::vector<TokenId> tokens;
      for (int index = 0; index < length; ++index)
        tokens.push_back((encoded >> (2 * index)) & 3);
      std::optional<TokenId> expected;
      if (tokens == std::vector<TokenId>{0})
        expected = 1;
      else if (tokens == std::vector<TokenId>{1} ||
               tokens == (std::vector<TokenId>{0, 1}))
        expected = 3;
      else if (tokens == (std::vector<TokenId>{0, 2}))
        expected = 0;
      const auto actual = PredictNext(model, tokens);
      EXPECT_EQ(actual.status().code(), expected.has_value()
                                            ? absl::StatusCode::kOk
                                            : absl::StatusCode::kNotFound);
      EXPECT_EQ(actual.value_or(0), expected.value_or(0));
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
        model.snap.function = nullptr;
        break;
    }
    EXPECT_EQ(ValidateModel(model).code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(PredictNext(model, std::vector<TokenId>{0}).status().code(),
              absl::StatusCode::kInvalidArgument);
    // Even a zero generation budget must validate the model and never invoke a
    // null callback; inference does not require a prior ValidateModel call.
    for (size_t budget : {0, 1})
      EXPECT_EQ(
          Generate(model, std::vector<TokenId>{0}, budget).status().code(),
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
  model.entry_function = [](TokenId, uint32_t) -> TransitionResult {
    return {0};
  };
  EXPECT_EQ(PredictNext(model, std::vector<TokenId>{0}).status().code(),
            absl::StatusCode::kDataLoss);
  model = Fixture();
  const AttentionTable invalid_attention[] = {
      {[](absl::Span<const StateId>) -> TransitionResult { return {2}; }}};
  model.attention = invalid_attention;
  EXPECT_EQ(PredictNext(model, std::vector<TokenId>{0}).status().code(),
            absl::StatusCode::kDataLoss);
  model = Fixture();
  const StateTable invalid_mlp[] = {
      {[](StateId) -> TransitionResult { return {1}; }}};
  model.mlp = invalid_mlp;
  EXPECT_EQ(PredictNext(model, std::vector<TokenId>{0}).status().code(),
            absl::StatusCode::kDataLoss);
  model = Fixture();
  model.snap = {[](StateId) -> TransitionResult { return {4}; }};
  EXPECT_EQ(PredictNext(model, std::vector<TokenId>{0}).status().code(),
            absl::StatusCode::kDataLoss);
  model.snap = {[](StateId) -> TransitionResult {
    return {std::numeric_limits<StateId>::max()};
  }};
  EXPECT_EQ(PredictNext(model, std::vector<TokenId>{0}).status().code(),
            absl::StatusCode::kDataLoss);
}

TEST(IntegerRuntime, OptionalTransitionsDistinguishZeroFromUnsupported) {
  auto model = Fixture();
  const auto zero = model.snap.function(14);
  ASSERT_TRUE(zero.output.has_value());
  EXPECT_EQ(*zero.output, 0u);
  EXPECT_EQ(model.snap.function(16).output, std::nullopt);
  const auto prediction = PredictNext(model, std::vector<TokenId>{0, 2});
  ASSERT_TRUE(prediction.ok());
  EXPECT_EQ(*prediction, 0);

  // Each callback propagates an empty optional as unsupported, rather than
  // interpreting an absent value as vocabulary ID zero or using a fallback.
  model.entry_function = [](TokenId, uint32_t) -> TransitionResult {
    return {};
  };
  EXPECT_EQ(PredictNext(model, std::vector<TokenId>{0}).status().code(),
            absl::StatusCode::kNotFound);
  model = Fixture();
  const AttentionTable missing_attention[] = {
      {[](absl::Span<const StateId>) -> TransitionResult { return {}; }}};
  model.attention = missing_attention;
  EXPECT_EQ(PredictNext(model, std::vector<TokenId>{0}).status().code(),
            absl::StatusCode::kNotFound);
  model = Fixture();
  const StateTable missing_mlp[] = {
      {[](StateId) -> TransitionResult { return {}; }}};
  model.mlp = missing_mlp;
  EXPECT_EQ(PredictNext(model, std::vector<TokenId>{0}).status().code(),
            absl::StatusCode::kNotFound);
  model = Fixture();
  model.snap = {[](StateId) -> TransitionResult { return {}; }};
  EXPECT_EQ(PredictNext(model, std::vector<TokenId>{0}).status().code(),
            absl::StatusCode::kNotFound);
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

TEST(IntegerRuntime, EachBlockConsumesThePreviousBlocksWholeOutput) {
  auto model = Fixture();
  const AttentionTable attention[] = {
      {Attention}, {[](absl::Span<const StateId> prefix) -> TransitionResult {
        if (prefix.size() == 1 && prefix[0] == 12)
          return {16};
        if (prefix.size() == 2 && prefix[0] == 12 && prefix[1] == 13)
          return {17};
        return {};
      }}};
  const StateTable mlp[] = {{Mlp}, {[](StateId state) -> TransitionResult {
                              if (state >= 16 && state <= 17)
                                return {state + 4};
                              return {};
                            }}};
  model.attention = attention;
  model.mlp = mlp;
  model.snap = {[](StateId state) -> TransitionResult {
    if (state == 20)
      return {1};
    if (state == 21)
      return {3};
    return {};
  }};
  const auto generated = Generate(model, std::vector<TokenId>{0}, 9);
  ASSERT_TRUE(generated.ok()) << generated.status();
  EXPECT_EQ(*generated, (std::vector<TokenId>{1, 3}));
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

TEST(IntegerRuntime, FunctionsDetermineAnswersWithoutCorpusOrSuffixCache) {
  auto model = Fixture();
  model.snap = {[](StateId state) -> TransitionResult {
    return state == 13 ? TransitionResult{2} : Snap(state);
  }};
  auto changed = PredictNext(model, std::vector<TokenId>{0, 1});
  ASSERT_TRUE(changed.ok());
  EXPECT_EQ(*changed, 2);

  model = Fixture();
  const StateTable changed_mlp[] = {{[](StateId state) -> TransitionResult {
    return state == 9 ? TransitionResult{14} : Mlp(state);
  }}};
  model.mlp = changed_mlp;
  changed = PredictNext(model, std::vector<TokenId>{0, 1});
  ASSERT_TRUE(changed.ok());
  EXPECT_EQ(*changed, 0);

  model = Fixture();
  const AttentionTable changed_attention[] = {
      {[](absl::Span<const StateId> prefix) -> TransitionResult {
        if (prefix.size() == 2 && prefix[0] == 4 && prefix[1] == 5)
          return {10};
        return Attention(prefix);
      }}};
  model.attention = changed_attention;
  changed = PredictNext(model, std::vector<TokenId>{0, 1});
  ASSERT_TRUE(changed.ok());
  EXPECT_EQ(*changed, 0);

  model = Fixture();
  model.entry_function = [](TokenId token,
                            uint32_t position) -> TransitionResult {
    return token == 1 && position == 1 ? TransitionResult{6}
                                       : Entry(token, position);
  };
  changed = PredictNext(model, std::vector<TokenId>{0, 1});
  ASSERT_TRUE(changed.ok());
  EXPECT_EQ(*changed, 0);
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
  EXPECT_FALSE(Decode(model, std::vector<TokenId>{4}).ok());
}

TEST(IntegerRuntime, GenerationHonorsBudgetContextAndUnknownHistories) {
  auto model = Fixture();
  auto one = Generate(model, std::vector<TokenId>{0}, 1);
  ASSERT_TRUE(one.ok());
  EXPECT_EQ(*one, (std::vector<TokenId>{1}));
  model.context_length = 1;
  auto full = Generate(model, std::vector<TokenId>{0}, 9);
  ASSERT_TRUE(full.ok());
  EXPECT_TRUE(full->empty());
  model = Fixture();
  EXPECT_EQ(Generate(model, std::vector<TokenId>{0, 2}, 2).status().code(),
            absl::StatusCode::kNotFound);
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
        model.eos_token = -1;
        break;
      case 5:
        model.eos_token = model.vocabulary.size();
        break;
      case 6:
        model.mlp = {};
        break;
    }
    EXPECT_EQ(ValidateModel(model).code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(PredictNext(model, std::vector<TokenId>{0}).status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(Generate(model, std::vector<TokenId>{0}, 0).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(IntegerRuntime, ZeroTransformerBlocksStillRunEntryAndSnap) {
  auto model = Fixture();
  model.attention = {};
  model.mlp = {};
  model.snap = {[](StateId state) -> TransitionResult {
    if (state == 4)
      return {1};
    if (state == 5)
      return {3};
    return {};
  }};
  ASSERT_TRUE(ValidateModel(model).ok());
  const auto generated = Generate(model, std::vector<TokenId>{0}, 9);
  ASSERT_TRUE(generated.ok()) << generated.status();
  EXPECT_EQ(*generated, (std::vector<TokenId>{1, 3}));
}

}  // namespace
}  // namespace pluto::llm::discretized
