#include "src/llm/experiments/one_shot_memorizer/automaton.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

void ExpectSameModel(const Model& a, const Model& b) {
  EXPECT_EQ(a.vocabulary_size, b.vocabulary_size);
  EXPECT_EQ(a.eos_token_id, b.eos_token_id);
  EXPECT_EQ(a.initial_state, b.initial_state);
  EXPECT_EQ(a.trie_state_count, b.trie_state_count);
  EXPECT_EQ(a.sentence_count, b.sentence_count);
  ASSERT_EQ(a.states.size(), b.states.size());
  for (size_t i = 0; i < a.states.size(); ++i) {
    EXPECT_EQ(a.states[i].terminal_count, b.states[i].terminal_count);
    EXPECT_EQ(a.states[i].suffix_count, b.states[i].suffix_count);
    ASSERT_EQ(a.states[i].transitions.size(), b.states[i].transitions.size());
    for (size_t j = 0; j < a.states[i].transitions.size(); ++j) {
      EXPECT_EQ(a.states[i].transitions[j].token,
                b.states[i].transitions[j].token);
      EXPECT_EQ(a.states[i].transitions[j].target,
                b.states[i].transitions[j].target);
    }
  }
}

// An independent sparse-matrix evaluation: h <- h * T_token, followed by
// h * terminal_count. This tests the saved weights' linear interpretation.
uint64_t MatrixSentenceWeight(const Model& model,
                              const std::vector<int>& sentence) {
  std::vector<uint64_t> h(model.states.size(), 0);
  h[model.initial_state] = 1;
  for (int token : sentence) {
    std::vector<uint64_t> next(model.states.size(), 0);
    for (size_t source = 0; source < model.states.size(); ++source) {
      for (const Transition& transition : model.states[source].transitions)
        if (transition.token == token)
          next[transition.target] += h[source];
    }
    h = std::move(next);
  }
  uint64_t weight = 0;
  for (size_t state = 0; state < model.states.size(); ++state)
    weight += h[state] * model.states[state].terminal_count;
  return weight;
}

TEST(OneShotAutomatonTest, RecallsEveryUnambiguousSuffixIncludingEos) {
  constexpr int kEos = 9;
  const std::vector<std::vector<int>> sentences = {{1, 3, 4}, {2, 3, 5, 6}};
  auto model = BuildModel(sentences, 10, kEos);
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_TRUE(ValidateModel(*model).ok());
  for (const auto& sentence : sentences) {
    for (size_t length = 1; length <= sentence.size(); ++length) {
      std::vector<int> prefix(sentence.begin(), sentence.begin() + length);
      std::vector<int> expected(sentence.begin() + length, sentence.end());
      expected.push_back(kEos);
      auto generated = GreedyContinuation(*model, prefix, 100);
      ASSERT_TRUE(generated.ok()) << generated.status();
      EXPECT_EQ(*generated, expected);
    }
  }
  auto limited = GreedyContinuation(*model, {2}, 2);
  ASSERT_TRUE(limited.ok()) << limited.status();
  EXPECT_EQ(*limited, (std::vector<int>{3, 5}));
  EXPECT_EQ(GreedyContinuation(*model, {}, 0).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(OneShotAutomatonTest, AmbiguousPrefixProbabilitiesCountOccurrences) {
  auto model = BuildModel({{1}, {1, 2}, {1, 2}, {1, 3}}, 5, 4);
  ASSERT_TRUE(model.ok()) << model.status();
  auto probabilities = NextTokenProbabilities(*model, {1});
  ASSERT_TRUE(probabilities.ok()) << probabilities.status();
  ASSERT_EQ(probabilities->size(), 3u);
  EXPECT_EQ((*probabilities)[0].token, 2);
  EXPECT_EQ((*probabilities)[0].count, 2);
  EXPECT_DOUBLE_EQ((*probabilities)[0].probability, 0.5);
  EXPECT_EQ((*probabilities)[1].token, 3);
  EXPECT_EQ((*probabilities)[1].count, 1);
  EXPECT_DOUBLE_EQ((*probabilities)[1].probability, 0.25);
  EXPECT_EQ((*probabilities)[2].token, 4);
  EXPECT_EQ((*probabilities)[2].count, 1);
  EXPECT_DOUBLE_EQ((*probabilities)[2].probability, 0.25);
  auto generated = GreedyContinuation(*model, {1}, 100);
  ASSERT_TRUE(generated.ok()) << generated.status();
  EXPECT_EQ(*generated, (std::vector<int>{2, 4}));
}

TEST(OneShotAutomatonTest, EosTiesUseTokenIdAndDistributionsRemainSorted) {
  auto model = BuildModel({{}, {0}, {2}}, 3, 1);
  ASSERT_TRUE(model.ok()) << model.status();
  auto probabilities = NextTokenProbabilities(*model, {});
  ASSERT_TRUE(probabilities.ok()) << probabilities.status();
  ASSERT_EQ(probabilities->size(), 3u);
  for (int token = 0; token < 3; ++token) {
    EXPECT_EQ((*probabilities)[token].token, token);
    EXPECT_DOUBLE_EQ((*probabilities)[token].probability, 1.0 / 3);
  }
  auto generated = GreedyContinuation(*model, {}, 100);
  ASSERT_TRUE(generated.ok()) << generated.status();
  EXPECT_EQ(*generated, (std::vector<int>{0, 1}));
  model = BuildModel({{}, {2}}, 3, 1);
  ASSERT_TRUE(model.ok()) << model.status();
  generated = GreedyContinuation(*model, {}, 100);
  ASSERT_TRUE(generated.ok()) << generated.status();
  EXPECT_EQ(*generated, (std::vector<int>{1}));
}

TEST(OneShotAutomatonTest, SameTokenInDifferentContextsDoesNotSpliceSentences) {
  auto model = BuildModel({{1, 3, 4}, {2, 3, 5}}, 7, 6);
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ(model->trie_state_count, 7u);
  EXPECT_EQ(model->states.size(), 6u);
  auto left = StateForPrefix(*model, {1});
  auto right = StateForPrefix(*model, {2});
  ASSERT_TRUE(left.ok()) << left.status();
  ASSERT_TRUE(right.ok()) << right.status();
  EXPECT_NE(*left, *right);
  EXPECT_EQ(MatrixSentenceWeight(*model, {1, 3, 4}), 1);
  EXPECT_EQ(MatrixSentenceWeight(*model, {2, 3, 5}), 1);
  EXPECT_EQ(MatrixSentenceWeight(*model, {1, 3, 5}), 0);
  EXPECT_EQ(MatrixSentenceWeight(*model, {2, 3, 4}), 0);
  EXPECT_EQ(MatrixSentenceWeight(*model, {3, 4}), 0);
  EXPECT_EQ(MatrixSentenceWeight(*model, {1, 3}), 0);
}

TEST(OneShotAutomatonTest, MergesCompleteEqualRightLanguages) {
  auto model = BuildModel({{1, 3, 4}, {2, 3, 4}}, 6, 5);
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ(model->trie_state_count, 7u);
  EXPECT_EQ(model->states.size(), 4u);
  EXPECT_EQ(model->sentence_count, 2);
  auto left = StateForPrefix(*model, {1});
  auto right = StateForPrefix(*model, {2});
  ASSERT_TRUE(left.ok()) << left.status();
  ASSERT_TRUE(right.ok()) << right.status();
  EXPECT_EQ(*left, *right);
  for (size_t source = 0; source < model->states.size(); ++source)
    for (const Transition& transition : model->states[source].transitions)
      EXPECT_LT(transition.target, source);
}

TEST(OneShotAutomatonTest, DuplicateWeightsPreventIncorrectMerging) {
  auto model = BuildModel({{1, 3}, {1, 3}, {2, 3}}, 5, 4);
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ(model->trie_state_count, 5u);
  EXPECT_EQ(model->states.size(), 5u);
  EXPECT_EQ(model->sentence_count, 3);
  EXPECT_EQ(MatrixSentenceWeight(*model, {1, 3}), 2);
  EXPECT_EQ(MatrixSentenceWeight(*model, {2, 3}), 1);
  auto left = StateForPrefix(*model, {1});
  auto right = StateForPrefix(*model, {2});
  ASSERT_TRUE(left.ok()) << left.status();
  ASSERT_TRUE(right.ok()) << right.status();
  EXPECT_NE(*left, *right);
  auto probabilities = NextTokenProbabilities(*model, {});
  ASSERT_TRUE(probabilities.ok()) << probabilities.status();
  ASSERT_EQ(probabilities->size(), 2u);
  EXPECT_EQ((*probabilities)[0].count, 2);
  EXPECT_DOUBLE_EQ((*probabilities)[0].probability, 2.0 / 3);
  EXPECT_EQ((*probabilities)[1].count, 1);
  EXPECT_DOUBLE_EQ((*probabilities)[1].probability, 1.0 / 3);
}

TEST(OneShotAutomatonTest, CanonicalWeightsDoNotDependOnCorpusOrder) {
  std::vector<std::vector<int>> corpus = {{},  {1, 2, 3}, {2, 2, 3}, {1, 2, 3},
                                          {3}, {3, 1},    {0, 2}};
  auto expected = BuildModel(corpus, 5, 4);
  ASSERT_TRUE(expected.ok()) << expected.status();
  for (size_t i = 0; i < corpus.size(); ++i) {
    std::rotate(corpus.begin(), corpus.begin() + 1, corpus.end());
    auto actual = BuildModel(corpus, 5, 4);
    ASSERT_TRUE(actual.ok()) << actual.status();
    ExpectSameModel(*expected, *actual);
  }
  std::reverse(corpus.begin(), corpus.end());
  auto actual = BuildModel(corpus, 5, 4);
  ASSERT_TRUE(actual.ok()) << actual.status();
  ExpectSameModel(*expected, *actual);
}

TEST(OneShotAutomatonTest, SupportsAnEmptySentenceAndRepeatedTokens) {
  auto model = BuildModel({{}, {}}, 1, 0);
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_EQ(model->states.size(), 1u);
  EXPECT_EQ(model->states[0].terminal_count, 2);
  EXPECT_TRUE(ValidateModel(*model).ok());
  auto generated = GreedyContinuation(*model, {}, 100);
  ASSERT_TRUE(generated.ok()) << generated.status();
  EXPECT_EQ(*generated, (std::vector<int>{0}));

  model = BuildModel({{1, 1, 1}}, 3, 2);
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ(model->states.size(), 4u);
  generated = GreedyContinuation(*model, {1}, 100);
  ASSERT_TRUE(generated.ok()) << generated.status();
  EXPECT_EQ(*generated, (std::vector<int>{1, 1, 2}));
}

TEST(OneShotAutomatonTest, RejectsInvalidCorporaAndUnseenPrefixes) {
  EXPECT_EQ(BuildModel({}, 4, 3).status().code(),
            absl::StatusCode::kInvalidArgument);
  for (int vocabulary_size : {-1, 0}) {
    EXPECT_EQ(BuildModel({{}}, vocabulary_size, 0).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
  for (int eos : {-1, 4}) {
    EXPECT_EQ(BuildModel({{1}}, 4, eos).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
  for (int token : {-1, 3, 4, std::numeric_limits<int>::max()}) {
    EXPECT_EQ(BuildModel({{token}}, 4, 3).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
  auto model = BuildModel({{1, 2}}, 4, 3);
  ASSERT_TRUE(model.ok()) << model.status();
  for (const auto& prefix :
       std::vector<std::vector<int>>{{0}, {2}, {1, 1}, {1, 2, 0}}) {
    EXPECT_EQ(StateForPrefix(*model, prefix).status().code(),
              absl::StatusCode::kNotFound);
    EXPECT_EQ(NextTokenProbabilities(*model, prefix).status().code(),
              absl::StatusCode::kNotFound);
    EXPECT_EQ(GreedyContinuation(*model, prefix, 100).status().code(),
              absl::StatusCode::kNotFound);
  }
  for (const auto& prefix :
       std::vector<std::vector<int>>{{-1}, {3}, {4}, {0, -1}, {1, 2, 3}}) {
    EXPECT_EQ(StateForPrefix(*model, prefix).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(OneShotAutomatonTest, RejectsMalformedSavedWeights) {
  auto good = BuildModel({{0}, {1}}, 3, 2);
  ASSERT_TRUE(good.ok()) << good.status();
  auto expect_invalid = [](const Model& model) {
    EXPECT_FALSE(ValidateModel(model).ok());
    EXPECT_FALSE(StateForPrefix(model, {}).ok());
    EXPECT_FALSE(NextTokenProbabilities(model, {}).ok());
    EXPECT_FALSE(GreedyContinuation(model, {}, 10).ok());
  };
  expect_invalid(Model{});
  Model malformed = *good;
  malformed.initial_state = malformed.states.size();
  expect_invalid(malformed);
  malformed = *good;
  malformed.states.back().transitions[0].target = malformed.initial_state;
  expect_invalid(malformed);
  malformed = *good;
  malformed.states.back().transitions[0].token = malformed.eos_token_id;
  expect_invalid(malformed);
  malformed = *good;
  malformed.states.back().transitions[1].token = 0;
  expect_invalid(malformed);
  malformed = *good;
  std::reverse(malformed.states.back().transitions.begin(),
               malformed.states.back().transitions.end());
  expect_invalid(malformed);
  malformed = *good;
  malformed.states[0].suffix_count = 0;
  expect_invalid(malformed);
  malformed = *good;
  ++malformed.sentence_count;
  expect_invalid(malformed);
  malformed = *good;
  malformed.trie_state_count = 0;
  expect_invalid(malformed);
  malformed = *good;
  malformed.states.push_back({1, 1, {}});
  malformed.trie_state_count = malformed.states.size();
  expect_invalid(malformed);
}

TEST(OneShotAutomatonTest, RejectsOverflowWhenValidatingWeights) {
  const uint64_t maximum = std::numeric_limits<uint64_t>::max();
  Model model;
  model.vocabulary_size = 2;
  model.eos_token_id = 1;
  model.initial_state = 1;
  model.trie_state_count = 2;
  model.sentence_count = maximum;
  model.states = {{maximum, maximum, {}}, {1, maximum, {{0, 0}}}};
  EXPECT_EQ(ValidateModel(model).code(), absl::StatusCode::kOutOfRange);
}

TEST(OneShotAutomatonTest, DeepSentenceUsesNoRecursiveTraversal) {
  constexpr size_t kLength = 20000;
  std::vector<int> sentence(kLength, 0);
  auto model = BuildModel({sentence}, 2, 1);
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ(model->states.size(), kLength + 1);
  auto generated = GreedyContinuation(*model, {}, kLength + 1);
  ASSERT_TRUE(generated.ok()) << generated.status();
  sentence.push_back(1);
  EXPECT_EQ(*generated, sentence);
}

TEST(OneShotAutomatonTest,
     ExhaustiveSmallCorporaMatchWeightedResidualLanguages) {
  const std::vector<std::vector<int>> words = {{},     {0},    {1},   {0, 0},
                                               {0, 1}, {1, 0}, {1, 1}};
  for (unsigned mask = 1; mask < (1u << words.size()); ++mask) {
    SCOPED_TRACE(mask);
    std::vector<std::vector<int>> corpus;
    for (size_t i = 0; i < words.size(); ++i) {
      if ((mask & (1u << i)) == 0)
        continue;
      const size_t repetitions = 1 + (mask + i) % 3;
      for (size_t j = 0; j < repetitions; ++j)
        corpus.push_back(words[i]);
    }
    auto model = BuildModel(corpus, 4, 3);
    ASSERT_TRUE(model.ok()) << model.status();
    EXPECT_TRUE(ValidateModel(*model).ok());

    std::set<std::vector<int>> prefixes;
    for (const auto& sentence : corpus)
      for (size_t length = 0; length <= sentence.size(); ++length)
        prefixes.emplace(sentence.begin(), sentence.begin() + length);
    using WeightedLanguage = std::map<std::vector<int>, uint64_t>;
    std::set<WeightedLanguage> distinct_residuals;
    for (const auto& prefix : prefixes) {
      WeightedLanguage residual;
      std::map<int, uint64_t> counts;
      uint64_t total = 0;
      for (const auto& sentence : corpus) {
        if (sentence.size() < prefix.size() ||
            !std::equal(prefix.begin(), prefix.end(), sentence.begin()))
          continue;
        std::vector<int> suffix(sentence.begin() + prefix.size(),
                                sentence.end());
        ++residual[suffix];
        ++counts[suffix.empty() ? 3 : suffix.front()];
        ++total;
      }
      distinct_residuals.insert(residual);
      auto actual = NextTokenProbabilities(*model, prefix);
      ASSERT_TRUE(actual.ok()) << actual.status();
      ASSERT_EQ(actual->size(), counts.size());
      size_t index = 0;
      for (const auto& [token, count] : counts) {
        EXPECT_EQ((*actual)[index].token, token);
        EXPECT_EQ((*actual)[index].count, count);
        EXPECT_DOUBLE_EQ((*actual)[index].probability,
                         static_cast<double>(count) / total);
        ++index;
      }
    }
    EXPECT_EQ(model->trie_state_count, prefixes.size());
    EXPECT_EQ(model->states.size(), distinct_residuals.size());
    for (const auto& word : words) {
      EXPECT_EQ(MatrixSentenceWeight(*model, word),
                std::count(corpus.begin(), corpus.end(), word));
    }
  }
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
