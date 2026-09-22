#include "src/llm/experiments/memorize_general_facts/discretized_model/runtime.h"

#include <functional>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::discretized {
namespace {

static_assert(!std::is_convertible_v<DiscreteToken, DiscreteHiddenState>);
static_assert(!std::is_convertible_v<DiscreteHiddenState, DiscreteToken>);
static_assert(!std::is_convertible_v<int, DiscreteHiddenState>);
static_assert(!std::is_convertible_v<int, DiscreteToken>);
static_assert(!std::is_convertible_v<DiscreteHiddenState, int>);
static_assert(!std::is_convertible_v<DiscreteToken, int>);
static_assert(sizeof(DiscreteHiddenState) == sizeof(int));
static_assert(sizeof(DiscreteToken) == sizeof(int));
static_assert(std::is_abstract_v<CausalAttention>);
static_assert(std::is_abstract_v<Map>);
static_assert(std::is_abstract_v<PositionEmbedding>);
static_assert(std::has_virtual_destructor_v<CausalAttention>);
static_assert(std::has_virtual_destructor_v<Map>);
static_assert(std::has_virtual_destructor_v<PositionEmbedding>);

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

std::optional<DiscreteHiddenState> Entry(DiscreteToken token,
                                         int32_t position) {
  if (position == 0) {
    if (token.value == 0)
      return DiscreteHiddenState{4};
    if (token.value == 1)
      return DiscreteHiddenState{7};
  }
  if (position == 1 && token.value >= 1 && token.value <= 2)
    return DiscreteHiddenState{token.value + 4};
  return std::nullopt;
}

std::optional<DiscreteHiddenState> Attention(
    absl::Span<const DiscreteHiddenState> prefix) {
  if (prefix.size() == 1 && prefix[0].value == 4)
    return DiscreteHiddenState{8};
  if (prefix.size() == 1 && prefix[0].value == 7)
    return DiscreteHiddenState{11};
  if (prefix.size() == 2 && prefix[0].value == 4 && prefix[1].value >= 5 &&
      prefix[1].value <= 6)
    return DiscreteHiddenState{prefix[1].value + 4};
  return std::nullopt;
}

std::optional<DiscreteHiddenState> Mlp(DiscreteHiddenState state) {
  if (state.value >= 8 && state.value <= 11)
    return DiscreteHiddenState{state.value + 4};
  return std::nullopt;
}

std::optional<DiscreteHiddenState> LanguageModelingHead(
    DiscreteHiddenState state) {
  switch (state.value) {
    case 12:
      return DiscreteHiddenState{1};
    case 13:
    case 15:
      return DiscreteHiddenState{3};
    case 14:
      return DiscreteHiddenState{0};
    default:
      return std::nullopt;
  }
}

// Mutable callbacks are test-only: they make it possible to change exactly one
// polymorphic boundary without rebuilding the fixture's non-null references.
class TestAttention final : public CausalAttention {
 public:
  std::function<std::optional<DiscreteHiddenState>(
      absl::Span<const DiscreteHiddenState>)>
      function = Attention;

  std::optional<DiscreteHiddenState> operator()(
      absl::Span<const DiscreteHiddenState> prefix) override {
    return function(prefix);
  }
};

class TestMap final : public Map {
 public:
  std::function<std::optional<DiscreteHiddenState>(DiscreteHiddenState)>
      function = Mlp;

  std::optional<DiscreteHiddenState> operator()(
      DiscreteHiddenState state) override {
    return function(state);
  }
};

class TestPositionEmbedding final : public PositionEmbedding {
 public:
  std::function<std::optional<DiscreteHiddenState>(DiscreteToken, int32_t)>
      function = Entry;

  std::optional<DiscreteHiddenState> operator()(DiscreteToken token,
                                                int32_t position) override {
    return function(token, position);
  }
};

// This object owns everything referenced by model and must not be copied.
struct Fixture {
  Fixture() { language_modeling_head.function = LanguageModelingHead; }
  Fixture(const Fixture&) = delete;
  Fixture& operator=(const Fixture&) = delete;

  const VocabularyRow vocabulary[4] = {
      {10, "A"}, {11, "B"}, {12, "C"}, {13, "<eos>"}};
  TestAttention attention;
  TestMap mlp;
  TestMap language_modeling_head;
  TestPositionEmbedding position_embedding;
  const Transformer transformers[1] = {{attention, mlp}};
  DiscreteModel model{1024,
                      1,
                      DiscreteToken{3},
                      vocabulary,
                      transformers,
                      language_modeling_head,
                      position_embedding};
};

TEST(IntegerRuntime, FunctionsMatchEntireFiniteDomain) {
  Fixture fixture;
  const auto& model = fixture.model;
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
      else if (tokens == Tokens({1}) || tokens == Tokens({0, 1}))
        expected = DiscreteToken{3};
      else if (tokens == Tokens({0, 2}))
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

TEST(IntegerRuntime, RejectsInvalidFunctionOutputs) {
  for (int boundary = 0; boundary < 4; ++boundary) {
    SCOPED_TRACE(boundary);
    Fixture fixture;
    if (boundary == 0)
      fixture.position_embedding.function = [](DiscreteToken, int32_t) {
        return DiscreteHiddenState{0};
      };
    else if (boundary == 1)
      fixture.attention.function = [](absl::Span<const DiscreteHiddenState>) {
        return DiscreteHiddenState{2};
      };
    else if (boundary == 2)
      fixture.mlp.function = [](DiscreteHiddenState) {
        return DiscreteHiddenState{1};
      };
    else
      fixture.language_modeling_head.function = [](DiscreteHiddenState) {
        return DiscreteHiddenState{4};
      };
    EXPECT_EQ(PredictNext(fixture.model, Tokens({0})).status().code(),
              absl::StatusCode::kDataLoss);
  }
  Fixture fixture;
  fixture.language_modeling_head.function = [](DiscreteHiddenState) {
    return DiscreteHiddenState{std::numeric_limits<int>::max()};
  };
  EXPECT_EQ(PredictNext(fixture.model, Tokens({0})).status().code(),
            absl::StatusCode::kDataLoss);
}

TEST(IntegerRuntime, OptionalTransitionsDistinguishZeroFromUnsupported) {
  Fixture fixture;
  const auto& model = fixture.model;
  const auto zero = model.language_modeling_head(DiscreteHiddenState{14});
  ASSERT_TRUE(zero.has_value());
  EXPECT_EQ(*zero, DiscreteHiddenState{0});
  EXPECT_EQ(model.language_modeling_head(DiscreteHiddenState{16}),
            std::nullopt);
  const auto prediction = PredictNext(model, Tokens({0, 2}));
  ASSERT_TRUE(prediction.ok());
  EXPECT_EQ(*prediction, DiscreteToken{0});

  // Each virtual operation propagates an empty optional as unsupported, rather
  // than interpreting an absent value as vocabulary ID zero or using a
  // fallback.
  for (int boundary = 0; boundary < 4; ++boundary) {
    SCOPED_TRACE(boundary);
    Fixture unsupported;
    if (boundary == 0)
      unsupported.position_embedding.function = [](DiscreteToken, int32_t) {
        return std::nullopt;
      };
    else if (boundary == 1)
      unsupported.attention.function =
          [](absl::Span<const DiscreteHiddenState>) { return std::nullopt; };
    else if (boundary == 2)
      unsupported.mlp.function = [](DiscreteHiddenState) {
        return std::nullopt;
      };
    else
      unsupported.language_modeling_head.function = [](DiscreteHiddenState) {
        return std::nullopt;
      };
    EXPECT_EQ(PredictNext(unsupported.model, Tokens({0})).status().code(),
              absl::StatusCode::kNotFound);
  }
}

TEST(IntegerRuntime, RejectsNegativeLabelsAtEveryBoundary) {
  for (int boundary = 0; boundary < 4; ++boundary) {
    SCOPED_TRACE(boundary);
    Fixture fixture;
    if (boundary == 0)
      fixture.position_embedding.function = [](DiscreteToken, int32_t) {
        return DiscreteHiddenState{-1};
      };
    else if (boundary == 1)
      fixture.attention.function = [](absl::Span<const DiscreteHiddenState>) {
        return DiscreteHiddenState{-1};
      };
    else if (boundary == 2)
      fixture.mlp.function = [](DiscreteHiddenState) {
        return DiscreteHiddenState{-1};
      };
    else
      fixture.language_modeling_head.function = [](DiscreteHiddenState) {
        return DiscreteHiddenState{-1};
      };
    EXPECT_EQ(PredictNext(fixture.model, Tokens({0})).status().code(),
              absl::StatusCode::kDataLoss);
  }
}

TEST(IntegerRuntime, ExecutesAllBoundariesAndPredictsEosAutoregressively) {
  Fixture fixture;
  ASSERT_TRUE(ValidateModel(fixture.model).ok());
  auto result = Generate(fixture.model, Tokens({0}), 9);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(*result, (Tokens({1, 3})));
  auto text = Decode(fixture.model, Tokens({0, 1, 2}));
  ASSERT_TRUE(text.ok());
  EXPECT_EQ(*text, "ABC");
}

TEST(IntegerRuntime, EachBlockConsumesThePreviousBlocksWholeOutput) {
  Fixture fixture;
  TestAttention second_attention;
  TestMap second_mlp;
  std::vector<std::vector<DiscreteHiddenState>> observed_prefixes;
  second_attention.function = [&](absl::Span<const DiscreteHiddenState> prefix)
      -> std::optional<DiscreteHiddenState> {
    observed_prefixes.emplace_back(prefix.begin(), prefix.end());
    if (prefix.size() == 1 && prefix[0].value == 12)
      return DiscreteHiddenState{16};
    if (prefix.size() == 2 && prefix[0].value == 12 && prefix[1].value == 13)
      return DiscreteHiddenState{17};
    return std::nullopt;
  };
  second_mlp.function =
      [](DiscreteHiddenState state) -> std::optional<DiscreteHiddenState> {
    if (state.value >= 16 && state.value <= 17)
      return DiscreteHiddenState{state.value + 4};
    return std::nullopt;
  };
  const Transformer transformers[] = {{fixture.attention, fixture.mlp},
                                      {second_attention, second_mlp}};
  fixture.model.transformers = transformers;
  fixture.language_modeling_head.function =
      [](DiscreteHiddenState state) -> std::optional<DiscreteHiddenState> {
    if (state.value == 20)
      return DiscreteHiddenState{1};
    if (state.value == 21)
      return DiscreteHiddenState{3};
    return std::nullopt;
  };
  const auto generated = Generate(fixture.model, Tokens({0}), 9);
  ASSERT_TRUE(generated.ok()) << generated.status();
  EXPECT_EQ(*generated, (Tokens({1, 3})));
  const std::vector<std::vector<DiscreteHiddenState>> expected_prefixes = {
      {{12}}, {{12}}, {{12}, {13}}};
  EXPECT_EQ(observed_prefixes, expected_prefixes);

  // A failure in a later polymorphic block must not reuse an earlier result.
  second_attention.function = [](absl::Span<const DiscreteHiddenState>) {
    return std::nullopt;
  };
  EXPECT_EQ(PredictNext(fixture.model, Tokens({0})).status().code(),
            absl::StatusCode::kNotFound);
}

TEST(IntegerRuntime, CompleteHistoryMattersEvenWithIdenticalLastState) {
  Fixture fixture;
  auto good = PredictNext(fixture.model, Tokens({0, 1}));
  ASSERT_TRUE(good.ok());
  EXPECT_EQ(*good, DiscreteToken{3});
  auto bad = PredictNext(fixture.model, Tokens({1, 1}));
  EXPECT_EQ(bad.status().code(), absl::StatusCode::kNotFound);
  EXPECT_NE(bad.status().message().find("attention history"),
            absl::string_view::npos);
}

TEST(IntegerRuntime, FunctionsDetermineAnswersWithoutCorpusOrSuffixCache) {
  for (int boundary = 0; boundary < 4; ++boundary) {
    SCOPED_TRACE(boundary);
    Fixture fixture;
    if (boundary == 0)
      fixture.position_embedding.function = [](DiscreteToken token,
                                               int32_t position) {
        return token.value == 1 && position == 1
                   ? std::optional{DiscreteHiddenState{6}}
                   : Entry(token, position);
      };
    else if (boundary == 1)
      fixture.attention.function =
          [](absl::Span<const DiscreteHiddenState> prefix)
          -> std::optional<DiscreteHiddenState> {
        if (prefix.size() == 2 && prefix[0].value == 4 && prefix[1].value == 5)
          return DiscreteHiddenState{10};
        return Attention(prefix);
      };
    else if (boundary == 2)
      fixture.mlp.function = [](DiscreteHiddenState state) {
        return state.value == 9 ? std::optional{DiscreteHiddenState{14}}
                                : Mlp(state);
      };
    else
      fixture.language_modeling_head.function = [](DiscreteHiddenState state) {
        return state.value == 13 ? std::optional{DiscreteHiddenState{2}}
                                 : LanguageModelingHead(state);
      };
    const auto changed = PredictNext(fixture.model, Tokens({0, 1}));
    ASSERT_TRUE(changed.ok());
    EXPECT_EQ(*changed, DiscreteToken{boundary == 3 ? 2 : 0});
  }
}

TEST(IntegerRuntime, ZeroBudgetAndInvalidPrompts) {
  Fixture fixture;
  const auto& model = fixture.model;
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
  Fixture fixture;
  auto one = Generate(fixture.model, Tokens({0}), 1);
  ASSERT_TRUE(one.ok());
  EXPECT_EQ(*one, (Tokens({1})));
  fixture.model.context_length = 1;
  auto full = Generate(fixture.model, Tokens({0}), 9);
  ASSERT_TRUE(full.ok());
  EXPECT_TRUE(full->empty());
  fixture.model.context_length = 1024;
  EXPECT_EQ(Generate(fixture.model, Tokens({0, 2}), 2).status().code(),
            absl::StatusCode::kNotFound);
}

TEST(IntegerRuntime, IndependentCallsAreBitExactAndCannotLeakHistory) {
  Fixture fixture;
  for (int i = 0; i < 12; ++i) {
    auto a = PredictNext(fixture.model, Tokens({0, 1}));
    auto b = PredictNext(fixture.model, Tokens({0, 2}));
    ASSERT_TRUE(a.ok());
    ASSERT_TRUE(b.ok());
    EXPECT_EQ(*a, DiscreteToken{3});
    EXPECT_EQ(*b, DiscreteToken{0});
  }
}

TEST(IntegerRuntime, RejectsInvalidModelDimensionsBeforeInference) {
  for (int field = 0; field < 7; ++field) {
    SCOPED_TRACE(field);
    Fixture fixture;
    auto& model = fixture.model;
    switch (field) {
      case 0:
        model.context_length = 0;
        break;
      case 1:
        model.prompt_token_count = 0;
        break;
      case 2:
        model.prompt_token_count = model.context_length + 1;
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
        model.context_length =
            static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) + 1;
        break;
    }
    // Validation must happen even for a zero generation budget, before any
    // virtual operation can receive an invalid token or signed position.
    EXPECT_EQ(ValidateModel(model).code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(PredictNext(model, Tokens({0})).status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(Generate(model, Tokens({0}), 0).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(IntegerRuntime,
     ZeroTransformerBlocksStillRunEntryAndLanguageModelingHead) {
  Fixture fixture;
  fixture.model.transformers = {};
  fixture.language_modeling_head.function =
      [](DiscreteHiddenState state) -> std::optional<DiscreteHiddenState> {
    if (state.value == 4)
      return DiscreteHiddenState{1};
    if (state.value == 5)
      return DiscreteHiddenState{3};
    return std::nullopt;
  };
  ASSERT_TRUE(ValidateModel(fixture.model).ok());
  const auto generated = Generate(fixture.model, Tokens({0}), 9);
  ASSERT_TRUE(generated.ok()) << generated.status();
  EXPECT_EQ(*generated, (Tokens({1, 3})));
}

}  // namespace
}  // namespace pluto::llm::discretized
