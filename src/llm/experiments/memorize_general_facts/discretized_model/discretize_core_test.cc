#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_core.h"

#include <algorithm>
#include <csignal>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::discretized::generator {
namespace {
ModelMetadata Header(int prompt = 1) {
  return {1, 1, 3, 2, prompt, {{0, "A"}, {1, "B"}, {2, "C"}}};
}
ExecutionSample Sample(const std::vector<int>& tokens,
                       const std::vector<std::vector<int>>& boundaries,
                       std::vector<int> predictions = {}) {
  if (predictions.empty()) {
    predictions.assign(tokens.begin() + 1, tokens.end());
    predictions.push_back(2);
  }
  ExecutionSample result{tokens, predictions, {}};
  for (const auto& boundary : boundaries) {
    std::vector<std::vector<uint16_t>> vectors;
    for (int word : boundary)
      vectors.push_back({static_cast<uint16_t>(word)});
    result.boundaries.push_back(std::move(vectors));
  }
  return result;
}
SymbolicModel Mergeable() {
  return BuildModel(Header(), {Sample({0}, {{100}, {200}, {300}}),
                               Sample({1}, {{101}, {201}, {301}})})
      .value();
}
SymbolicModel Branching() {
  return BuildModel(Header(),
                    {Sample({0, 1}, {{100, 102}, {200, 202}, {300, 302}}),
                     Sample({1}, {{101}, {201}, {301}})})
      .value();
}
int State(const SymbolicModel& model, int stage, int word) {
  for (const auto& row : model.states)
    if (row.boundary == stage &&
        row.bits == std::vector<uint16_t>{static_cast<uint16_t>(word)})
      return row.id;
  return -1;
}

TEST(DiscretizeCoreTest, BitExactDuplicatesPreserveSignedZero) {
  auto sample = Sample({0}, {{0}, {0x8000}, {0x3f80}});
  auto model = BuildModel(Header(), {sample, sample}, 2);
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ(model->states.size(), 3u);
  EXPECT_EQ(model->entry.size(), 1u);
  EXPECT_EQ(model->states[1].bits[0], 0x8000);
  EXPECT_EQ(model->stats.verification, (VerificationResult{2, 2, 0, 2}));
  EXPECT_FALSE(BuildModel(Header(), {sample}, 2).ok());
}

TEST(DiscretizeCoreTest, ConflictingTransitionsRejected) {
  auto sample = Sample({0}, {{100}, {200}, {300}});
  for (const auto& conflict :
       {Sample({0}, {{101}, {200}, {300}}), Sample({0}, {{100}, {201}, {300}}),
        Sample({1}, {{101}, {200}, {301}})}) {
    auto result = BuildModel(Header(), {sample, conflict});
    EXPECT_FALSE(result.ok());
    EXPECT_NE(result.status().message().find("conflicting"), std::string::npos);
  }
}

TEST(DiscretizeCoreTest, NonfiniteVectorsAndWrongReferenceRejected) {
  EXPECT_FALSE(
      BuildModel(Header(), {Sample({0}, {{0x7f80}, {200}, {300}})}).ok());
  EXPECT_FALSE(
      BuildModel(Header(), {Sample({0}, {{0xffc0}, {200}, {300}})}).ok());
  EXPECT_FALSE(
      BuildModel(Header(), {Sample({0}, {{100}, {200}, {300}}, {1})}).ok());
  EXPECT_FALSE(BuildModel(Header(), {}).ok());
}

TEST(DiscretizeCoreTest, AutoregressionChecksEverySuffixTokenAndEos) {
  auto model = Branching();
  auto evaluated = EvaluateModel(model);
  ASSERT_TRUE(evaluated.ok()) << evaluated.status();
  EXPECT_EQ(evaluated->targets, 3);
  EXPECT_FALSE(PredictNext(model, {0, 0}).ok());
  EXPECT_FALSE(PredictNext(model, {}).ok());
  EXPECT_FALSE(PredictNext(model, std::vector<int>(1025, 0)).ok());
  for (auto& row : model.language_modeling_head)
    if (row.input == State(model, 2, 302))
      row.output = 0;
  auto bad = EvaluateModel(model);
  EXPECT_FALSE(bad.ok());
  EXPECT_NE(bad.status().message().find("expected 2, got 0"),
            std::string::npos);
}

TEST(DiscretizeCoreTest, AttentionUsesEntireOrderedPrefix) {
  auto model = BuildModel(
      Header(), {Sample({0, 1}, {{100, 102}, {200, 202}, {300, 302}}),
                 Sample({1, 1}, {{101, 102}, {201, 203}, {301, 303}})});
  ASSERT_TRUE(model.ok()) << model.status();
  std::vector<AttentionTransition> longer;
  for (const auto& row : model->transformers[0].attention)
    if (row.prefix.size() == 2)
      longer.push_back(row);
  ASSERT_EQ(longer.size(), 2u);
  EXPECT_EQ(longer[0].prefix[1], longer[1].prefix[1]);
  EXPECT_NE(longer[0].output, longer[1].output);
}

TEST(DiscretizeCoreTest,
     EosInsideRequiredSuffixIsRejectedButPromptEosIsAllowed) {
  auto capture =
      Sample({0, 2, 2}, {{100, 101, 102}, {200, 201, 202}, {300, 301, 302}});
  auto invalid = BuildModel(Header(), {capture});
  EXPECT_FALSE(invalid.ok());
  EXPECT_NE(invalid.status().message().find("suffix must not contain EOS"),
            std::string::npos);

  auto model = Branching();
  model.samples[0].tokens = {0, 2, 2};
  EXPECT_FALSE(ValidateModel(model).ok());
  EXPECT_FALSE(EvaluateModel(model).ok());

  auto prompt_eos = BuildModel(Header(), {Sample({2}, {{100}, {200}, {300}})});
  ASSERT_TRUE(prompt_eos.ok()) << prompt_eos.status();
  ASSERT_TRUE(prompt_eos->stats.verification.has_value());
  EXPECT_EQ(prompt_eos->stats.verification->explicit_eos, 1);
}

TEST(DiscretizeCoreTest,
     OptionalRelabelingMustBeCompleteTypedAndBoundaryPreserving) {
  auto model = Mergeable();
  for (const auto& state : model.states)
    model.state_relabeling.push_back({state.id, state.id, state.boundary});
  ASSERT_TRUE(ValidateModel(model).ok());
  for (int defect = 0; defect < 7; ++defect) {
    auto invalid = model;
    auto& row = invalid.state_relabeling[0];
    if (defect == 0)
      invalid.state_relabeling.pop_back();
    if (defect == 1)
      row.old_id = 0;
    if (defect == 2)
      row.new_id = INT32_MAX;
    if (defect == 3)
      row.boundary = -1;
    if (defect == 4)
      row.boundary = 2;
    if (defect == 5)
      invalid.state_relabeling[1] = row;
    if (defect == 6)
      row.old_id = -1;
    EXPECT_FALSE(ValidateModel(invalid).ok()) << defect;
  }
}

TEST(DiscretizeCoreTest, PromptOutputsAreNotReadoutConstraints) {
  auto model = BuildModel(
      Header(2),
      {Sample({0, 1}, {{100, 101}, {200, 201}, {300, 301}}, {0, 2})});
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ(model->language_modeling_head.size(), 1u);
  auto reducer = QuotientReducer::Create(*model);
  ASSERT_TRUE(reducer.ok()) << reducer.status();
  auto merged =
      (*reducer)->TryMerge(State(*model, 2, 300), State(*model, 2, 301));
  ASSERT_TRUE(merged.ok());
  EXPECT_TRUE(*merged);
  EXPECT_TRUE(EvaluateModel((*reducer)->Export()).ok());
}

TEST(DiscretizeCoreTest, CongruenceClosureInducesAttentionAndMlpMerges) {
  auto model = Mergeable();
  auto reducer = QuotientReducer::Create(model);
  ASSERT_TRUE(reducer.ok()) << reducer.status();
  auto merged =
      (*reducer)->TryMerge(State(model, 0, 100), State(model, 0, 101));
  ASSERT_TRUE(merged.ok());
  ASSERT_TRUE(*merged);
  auto reduced = (*reducer)->Export();
  EXPECT_EQ(reduced.states.size(), 3u);
  EXPECT_EQ(reduced.stats.state_unions, 3);
  EXPECT_EQ(reduced.stats.accepted_merges[0].induced_unions, 2);
  EXPECT_EQ(reduced.transformers[0].attention.size(), 1u);
  EXPECT_EQ(reduced.transformers[0].mlp.size(), 1u);
  EXPECT_TRUE(EvaluateModel(reduced).ok());
}

TEST(DiscretizeCoreTest,
     RejectedTrialRestoresAllTransitionsAndCachesRejection) {
  auto model = Branching();
  auto reducer = QuotientReducer::Create(model);
  ASSERT_TRUE(reducer.ok()) << reducer.status();
  SymbolicModel before = (*reducer)->Export();
  for (int iteration = 0; iteration < 2; ++iteration) {
    auto merged =
        (*reducer)->TryMerge(State(model, 0, 100), State(model, 0, 101));
    ASSERT_TRUE(merged.ok());
    EXPECT_FALSE(*merged);
  }
  SymbolicModel after = (*reducer)->Export();
  EXPECT_EQ(after.stats.cached_rejections, 1);
  EXPECT_EQ(after.stats.attempted_seeds, 1);
  before.stats = {};
  after.stats = {};
  EXPECT_EQ(before, after);
  auto merged =
      (*reducer)->TryMerge(State(model, 2, 301), State(model, 2, 302));
  ASSERT_TRUE(merged.ok());
  EXPECT_TRUE(*merged);
  EXPECT_TRUE(EvaluateModel((*reducer)->Export()).ok());
  EXPECT_FALSE(
      (*reducer)->TryMerge(State(model, 0, 100), State(model, 1, 200)).ok());
  EXPECT_FALSE((*reducer)->TryMerge(-1, -1).ok());
}

TEST(DiscretizeCoreTest, SearchLimitsAreNotMisreportedAsMinimality) {
  ReductionOptions options;
  options.neighbors = 1;
  options.max_passes = 3;
  auto reduced = ReduceModel(Branching(), options);
  ASSERT_TRUE(reduced.ok()) << reduced.status();
  ASSERT_TRUE(reduced->stats.search.has_value());
  EXPECT_TRUE(reduced->stats.search->pairwise_irreducible);
  EXPECT_FALSE(reduced->stats.search->global_minimum_proven);
  options.max_attempts = 0;
  auto limited = ReduceModel(Branching(), options);
  ASSERT_TRUE(limited.ok()) << limited.status();
  ASSERT_TRUE(limited->stats.search.has_value());
  EXPECT_EQ(limited->stats.search->stopping_reason,
            SearchStoppingReason::kAttemptLimit);
  EXPECT_FALSE(limited->stats.search->pairwise_irreducible);
  options.max_attempts.reset();
  options.max_passes = 1;
  options.exhaustive_pair_limit = 0;
  auto shortlist = ReduceModel(Branching(), options);
  ASSERT_TRUE(shortlist.ok()) << shortlist.status();
  ASSERT_TRUE(shortlist->stats.search.has_value());
  EXPECT_FALSE(shortlist->stats.search->pairwise_irreducible);
  options.neighbors = 0;
  EXPECT_FALSE(ReduceModel(Branching(), options).ok());
}

TEST(DiscretizeCoreTest, MembershipIsPreservedAndCanBeRecovered) {
  const auto original = Mergeable();
  auto reducer = QuotientReducer::Create(original);
  ASSERT_TRUE(reducer.ok()) << reducer.status();
  ASSERT_TRUE((*reducer)
                  ->TryMerge(State(original, 2, 300), State(original, 2, 301))
                  .value());
  const auto partial = (*reducer)->Export();
  auto legacy = partial;
  for (auto& row : legacy.states)
    row.members.reset();
  auto restored = RestoreMembership(legacy, original);
  ASSERT_TRUE(restored.ok()) << restored.status();
  EXPECT_EQ(restored->states, partial.states);
  auto reduced = ReduceModel(*restored);
  ASSERT_TRUE(reduced.ok()) << reduced.status();
  for (const auto& row : reduced->states) {
    ASSERT_TRUE(row.members.has_value());
    EXPECT_EQ(row.members->size(), 2u);
  }
  legacy.language_modeling_head[0].output = 0;
  EXPECT_FALSE(RestoreMembership(legacy, original).ok());
}

TEST(DiscretizeCoreTest, CancellationIsCooperativeAndSignalHandlerRestored) {
  ReductionOptions options;
  options.interrupted = [] { return true; };
  auto cancelled = ReduceModel(Mergeable(), options);
  EXPECT_EQ(cancelled.status().code(), absl::StatusCode::kCancelled);
  void (*previous)(int) = std::signal(SIGINT, SIG_IGN);
  options.interrupted = {};
  options.progress = [](const ReductionProgress&) { std::raise(SIGINT); };
  cancelled = ReduceModel(Mergeable(), options);
  EXPECT_EQ(cancelled.status().code(), absl::StatusCode::kCancelled);
  EXPECT_EQ(std::signal(SIGINT, previous), SIG_IGN);
}

TEST(DiscretizeCoreTest, MalformedTypedModelsReturnStatusInsteadOfAborting) {
  // Field types rule out malformed documents; dimensions, relationships, and
  // numerical ranges remain runtime invariants and must still be checked.
  const std::vector<std::function<void(SymbolicModel&)>> defects = {
      [](auto& m) { m.metadata.width = 0; },
      [](auto& m) { m.metadata.layers = -1; },
      [](auto& m) { m.metadata.vocab_size = 0; },
      [](auto& m) { m.metadata.prompt_tokens = 0; },
      [](auto& m) { m.metadata.eos_token = -1; },
      [](auto& m) { m.metadata.eos_token = m.metadata.vocab_size; },
      [](auto& m) { m.metadata.vocabulary.clear(); },
      [](auto& m) { m.metadata.vocabulary[0].bytes.clear(); },
      [](auto& m) {
        m.metadata.vocabulary[1].original_id =
            m.metadata.vocabulary[0].original_id;
      },
      [](auto& m) { m.states.clear(); },
      [](auto& m) { m.states[0].id = -1; },
      [](auto& m) { m.states[0].bits.clear(); },
      [](auto& m) { m.states[0].bits[0] = 0x7f80; },
      [](auto& m) { m.states[0].boundary = 1; },
      [](auto& m) { m.states[0].members = std::vector<int>{}; },
      [](auto& m) {
        m.states[0].members = std::vector<int>{3};
        m.states[1].members = std::vector<int>{3};
      },
      [](auto& m) { m.entry.clear(); },
      [](auto& m) { m.entry.push_back(m.entry.front()); },
      [](auto& m) { m.entry[0].position = 1024; },
      [](auto& m) { m.transformers.clear(); },
      [](auto& m) { m.transformers[0].attention[0].prefix.clear(); },
      [](auto& m) { m.transformers[0].mlp[0].input = m.entry[0].output; },
      [](auto& m) { m.language_modeling_head.clear(); },
      [](auto& m) { m.samples.clear(); },
      [](auto& m) { m.samples[0].tokens.clear(); },
      [](auto& m) { m.stats.state_unions = -1; },
  };
  for (size_t index = 0; index < defects.size(); ++index) {
    auto bad = Mergeable();
    defects[index](bad);
    EXPECT_FALSE(ValidateModel(bad).ok()) << index;
  }
  auto binary_vocabulary = Mergeable();
  binary_vocabulary.metadata.vocabulary[0].bytes = std::string("\x80\0", 2);
  EXPECT_TRUE(ValidateModel(binary_vocabulary).ok());
}

TEST(DiscretizeCoreTest, CapturedShapesAndRangesAreValidated) {
  const std::vector<std::function<void(ExecutionSample&)>> defects = {
      [](auto& s) { s.tokens.clear(); },
      [](auto& s) { s.tokens[0] = -1; },
      [](auto& s) { s.predictions.clear(); },
      [](auto& s) { s.predictions[0] = 3; },
      [](auto& s) { s.boundaries.pop_back(); },
      [](auto& s) { s.boundaries[0].clear(); },
      [](auto& s) { s.boundaries[0][0].clear(); },
      [](auto& s) { s.boundaries[0][0].push_back(1); },
  };
  for (size_t index = 0; index < defects.size(); ++index) {
    auto sample = Sample({0}, {{100}, {200}, {300}});
    defects[index](sample);
    EXPECT_FALSE(BuildModel(Header(), {sample}).ok()) << index;
  }
}

TEST(DiscretizeCoreTest, MalformedProvenanceStatsFailSafely) {
  const auto original = Mergeable();
  for (const auto& invalid : std::vector<MergeRecord>{
           {.euclidean_distance = -1},
           {.euclidean_distance = std::numeric_limits<double>::infinity()},
           {.euclidean_distance = std::numeric_limits<double>::quiet_NaN()},
           {.euclidean_distance = 1.5, .induced_unions = -1}}) {
    auto model = original;
    model.stats.accepted_merges = {invalid};
    EXPECT_FALSE(ValidateModel(model).ok());
  }
  auto model = original;
  model.stats.accepted_merges = {
      {.euclidean_distance = 0, .induced_unions = INT64_MAX},
      {.euclidean_distance = 1, .induced_unions = 1}};
  EXPECT_FALSE(ValidateModel(model).ok());
  model = original;
  model.stats.accepted_merges = {
      {.euclidean_distance = 1.5, .induced_unions = 2}};
  model.stats.search = SearchStatistics{.pairwise_irreducible = false};
  EXPECT_TRUE(ValidateModel(model).ok());
}

TEST(DiscretizeCoreTest, IncrementalClosureMatchesIndependentFullRescan) {
  // Independent oracle repeatedly rescans every immutable equation after each
  // tentative union. It shares no indexing, rollback, or rejection logic.
  std::mt19937 random(2917);
  for (int trial = 0; trial < 8; ++trial) {
    SymbolicModel model;
    model.metadata = Header();
    for (int i = 0; i < 21; ++i)
      model.states.push_back({3 + i, i / 7, {static_cast<uint16_t>(100 + i)}});
    model.entry = {{0, 0, 3}};
    model.transformers.resize(1);
    struct Term {
      int stage;
      std::vector<int> inputs;
      int output;
    };
    std::vector<Term> terms;
    std::set<std::vector<int>> prefixes;
    for (int length : {1, 2, 3})
      for (int i = 0; i < 20; ++i) {
        std::vector<int> prefix(length);
        for (auto& value : prefix)
          value = 3 + random() % 7;
        prefixes.insert(prefix);
      }
    for (const auto& prefix : prefixes) {
      int output = 10 + random() % 7;
      model.transformers[0].attention.push_back({prefix, output});
      std::vector<int> inputs;
      for (int value : prefix)
        inputs.push_back(value - 3);
      terms.push_back({1, inputs, output - 3});
    }
    for (int i = 0; i < 7; ++i) {
      int output = 17 + random() % 7;
      model.transformers[0].mlp.push_back({10 + i, output});
      terms.push_back({2, {7 + i}, output - 3});
      model.language_modeling_head.push_back(
          {17 + i, static_cast<int>(random() % 3)});
    }
    auto reducer = QuotientReducer::Create(model);
    ASSERT_TRUE(reducer.ok()) << reducer.status();
    std::vector<int> equivalence(21);
    std::iota(equivalence.begin(), equivalence.end(), 0);
    auto oracle = [&](int first,
                      int second) -> std::optional<std::vector<int>> {
      auto candidate = equivalence;
      auto unite = [&](int a, int b) {
        a = candidate[a];
        b = candidate[b];
        if (a == b)
          return false;
        for (int& root : candidate)
          if (root == a || root == b)
            root = std::min(a, b);
        return true;
      };
      unite(first, second);
      while (true) {
        bool changed = false;
        std::map<std::vector<int>, int> signatures;
        for (const auto& term : terms) {
          std::vector<int> signature{term.stage};
          for (int argument : term.inputs)
            signature.push_back(candidate[argument]);
          auto [found, inserted] = signatures.emplace(signature, term.output);
          if (!inserted)
            changed |= unite(term.output, found->second);
        }
        std::map<int, int> labels;
        for (const auto& row : model.language_modeling_head) {
          auto [found, inserted] =
              labels.emplace(candidate[row.input - 3], row.output);
          if (!inserted && found->second != row.output)
            return std::nullopt;
        }
        if (!changed)
          return candidate;
      }
    };
    for (int iteration = 0; iteration < 70; ++iteration) {
      const int stage = random() % 3, first = 7 * stage + random() % 7,
                second = 7 * stage + random() % 7;
      auto expected = oracle(first, second);
      auto actual = (*reducer)->TryMerge(first + 3, second + 3);
      ASSERT_TRUE(actual.ok()) << actual.status();
      ASSERT_EQ(*actual, expected.has_value());
      if (expected)
        equivalence = *expected;
      for (int a = 0; a < 21; ++a)
        for (int b = 0; b < 21; ++b)
          ASSERT_EQ(equivalence[a] == equivalence[b],
                    (*reducer)->RootForState(a + 3) ==
                        (*reducer)->RootForState(b + 3));
    }
  }
}
}  // namespace
}  // namespace pluto::llm::discretized::generator
