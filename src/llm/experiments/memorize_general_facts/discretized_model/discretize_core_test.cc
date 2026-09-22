#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_core.h"

#include <algorithm>
#include <csignal>
#include <cstdint>
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
Json Header(int prompt = 1) {
  return {{"schema", 1},
          {"width", 1},
          {"layers", 1},
          {"vocab_size", 3},
          {"eos_token", 2},
          {"prompt_tokens", prompt},
          {"vocabulary",
           {{{"original_id", 0}, {"hex", "41"}},
            {{"original_id", 1}, {"hex", "42"}},
            {{"original_id", 2}, {"hex", "43"}}}}};
}
Json Sample(const std::vector<int>& tokens,
            const std::vector<std::vector<int>>& boundaries,
            std::vector<int> predictions = {}) {
  if (predictions.empty()) {
    predictions.assign(tokens.begin() + 1, tokens.end());
    predictions.push_back(2);
  }
  Json result = {{"tokens", tokens},
                 {"predictions", predictions},
                 {"boundaries", Json::array()}};
  for (const auto& boundary : boundaries) {
    Json vectors = Json::array();
    for (int word : boundary)
      vectors.push_back(Json::array({word}));
    result["boundaries"].push_back(std::move(vectors));
  }
  return result;
}
Json Mergeable() {
  return BuildModel(Header(), {Sample({0}, {{100}, {200}, {300}}),
                               Sample({1}, {{101}, {201}, {301}})})
      .value();
}
Json Branching() {
  return BuildModel(Header(),
                    {Sample({0, 1}, {{100, 102}, {200, 202}, {300, 302}}),
                     Sample({1}, {{101}, {201}, {301}})})
      .value();
}
int State(const Json& model, int stage, int word) {
  for (const auto& row : model["states"])
    if (row["stage"] == stage && row["bits"] == Json::array({word}))
      return row["id"];
  return -1;
}

TEST(DiscretizeCoreTest, BitExactDuplicatesPreserveSignedZero) {
  auto sample = Sample({0}, {{0}, {0x8000}, {0x3f80}});
  auto model = BuildModel(Header(), {sample, sample}, 2);
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ((*model)["states"].size(), 3u);
  EXPECT_EQ((*model)["entry"].size(), 1u);
  EXPECT_EQ((*model)["states"][1]["bits"][0], 0x8000);
  EXPECT_EQ(
      (*model)["stats"]["verification"],
      (Json{
          {"samples", 2}, {"targets", 2}, {"errors", 0}, {"explicit_eos", 2}}));
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
  EXPECT_EQ((*evaluated)["targets"], 3);
  EXPECT_FALSE(PredictNext(model, {0, 0}).ok());
  EXPECT_FALSE(PredictNext(model, {}).ok());
  EXPECT_FALSE(PredictNext(model, std::vector<int>(1025, 0)).ok());
  for (auto& row : model["language_modeling_head"])
    if (row[0] == State(model, 2, 302))
      row[1] = 0;
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
  std::vector<Json> longer;
  for (const auto& row : (*model)["attention"][0])
    if (row[0].size() == 2)
      longer.push_back(row);
  ASSERT_EQ(longer.size(), 2u);
  EXPECT_EQ(longer[0][0][1], longer[1][0][1]);
  EXPECT_NE(longer[0][1], longer[1][1]);
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
  model["samples"][0]["tokens"] = {0, 2, 2};
  EXPECT_FALSE(ValidateModel(model).ok());
  EXPECT_FALSE(EvaluateModel(model).ok());

  auto prompt_eos = BuildModel(Header(), {Sample({2}, {{100}, {200}, {300}})});
  ASSERT_TRUE(prompt_eos.ok()) << prompt_eos.status();
  EXPECT_EQ((*prompt_eos)["stats"]["verification"]["explicit_eos"], 1);
}

TEST(DiscretizeCoreTest,
     OptionalRelabelingMustBeCompleteTypedAndBoundaryPreserving) {
  auto model = Mergeable();
  model["state_relabeling"] = Json::array();
  for (const auto& state : model["states"])
    model["state_relabeling"].push_back(
        {state["id"], state["id"], state["stage"]});
  ASSERT_TRUE(ValidateModel(model).ok());
  for (const auto& defect :
       std::vector<Json>{Json(7), Json::object(), Json::array()}) {
    auto invalid = model;
    invalid["state_relabeling"] = defect;
    EXPECT_FALSE(ValidateModel(invalid).ok());
  }
  for (int defect = 0; defect < 7; ++defect) {
    auto invalid = model;
    auto& row = invalid["state_relabeling"][0];
    if (defect == 0)
      row = {3, 3};
    if (defect == 1)
      row[0] = "3";
    if (defect == 2)
      row[1] = UINT64_MAX;
    if (defect == 3)
      row[2] = true;
    if (defect == 4)
      row[2] = 2;
    if (defect == 5)
      invalid["state_relabeling"][1] = row;
    if (defect == 6)
      row[0] = -1;
    EXPECT_FALSE(ValidateModel(invalid).ok()) << defect;
  }
}

TEST(DiscretizeCoreTest, PromptOutputsAreNotReadoutConstraints) {
  auto model = BuildModel(
      Header(2),
      {Sample({0, 1}, {{100, 101}, {200, 201}, {300, 301}}, {0, 2})});
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ((*model)["language_modeling_head"].size(), 1u);
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
  EXPECT_EQ(reduced["states"].size(), 3u);
  EXPECT_EQ(reduced["stats"]["state_unions"], 3);
  EXPECT_EQ(reduced["stats"]["accepted_merges"][0]["induced_unions"], 2);
  EXPECT_EQ(reduced["attention"][0].size(), 1u);
  EXPECT_EQ(reduced["mlp"][0].size(), 1u);
  EXPECT_TRUE(EvaluateModel(reduced).ok());
}

TEST(DiscretizeCoreTest,
     RejectedTrialRestoresAllTransitionsAndCachesRejection) {
  auto model = Branching();
  auto reducer = QuotientReducer::Create(model);
  ASSERT_TRUE(reducer.ok()) << reducer.status();
  Json before = (*reducer)->Export();
  for (int iteration = 0; iteration < 2; ++iteration) {
    auto merged =
        (*reducer)->TryMerge(State(model, 0, 100), State(model, 0, 101));
    ASSERT_TRUE(merged.ok());
    EXPECT_FALSE(*merged);
  }
  Json after = (*reducer)->Export();
  EXPECT_EQ(after["stats"]["cached_rejections"], 1);
  EXPECT_EQ(after["stats"]["attempted_seeds"], 1);
  before.erase("stats");
  after.erase("stats");
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
  EXPECT_EQ((*reduced)["stats"]["search"]["pairwise_irreducible"], true);
  EXPECT_EQ((*reduced)["stats"]["search"]["global_minimum_proven"], false);
  options.max_attempts = 0;
  auto limited = ReduceModel(Branching(), options);
  ASSERT_TRUE(limited.ok()) << limited.status();
  EXPECT_EQ((*limited)["stats"]["search"]["stopping_reason"], "attempt_limit");
  EXPECT_EQ((*limited)["stats"]["search"]["pairwise_irreducible"], false);
  options.max_attempts.reset();
  options.max_passes = 1;
  options.exhaustive_pair_limit = 0;
  auto shortlist = ReduceModel(Branching(), options);
  ASSERT_TRUE(shortlist.ok()) << shortlist.status();
  EXPECT_EQ((*shortlist)["stats"]["search"]["pairwise_irreducible"], false);
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
  for (auto& row : legacy["states"]) {
    row.erase("members");
    row.erase("member_count");
  }
  auto restored = RestoreMembership(legacy, original);
  ASSERT_TRUE(restored.ok()) << restored.status();
  EXPECT_EQ((*restored)["states"], partial["states"]);
  auto reduced = ReduceModel(*restored);
  ASSERT_TRUE(reduced.ok()) << reduced.status();
  for (const auto& row : (*reduced)["states"])
    EXPECT_EQ(row["member_count"], 2);
  legacy["language_modeling_head"][0][1] = 0;
  EXPECT_FALSE(RestoreMembership(legacy, original).ok());
}

TEST(DiscretizeCoreTest, CancellationIsCooperativeAndSignalHandlerRestored) {
  ReductionOptions options;
  options.interrupted = [] { return true; };
  auto cancelled = ReduceModel(Mergeable(), options);
  EXPECT_EQ(cancelled.status().code(), absl::StatusCode::kCancelled);
  void (*previous)(int) = std::signal(SIGINT, SIG_IGN);
  options.interrupted = {};
  options.progress = [](const Json&) { std::raise(SIGINT); };
  cancelled = ReduceModel(Mergeable(), options);
  EXPECT_EQ(cancelled.status().code(), absl::StatusCode::kCancelled);
  EXPECT_EQ(std::signal(SIGINT, previous), SIG_IGN);
}

TEST(DiscretizeCoreTest, MalformedExternalValuesReturnStatusInsteadOfAborting) {
  const Json model = Mergeable();
  for (const char* field :
       {"schema", "width", "layers", "vocab_size", "prompt_tokens", "eos_token",
        "states", "entry", "attention", "mlp", "language_modeling_head",
        "vocabulary", "samples"}) {
    auto bad = model;
    bad[field] = nullptr;
    EXPECT_FALSE(ValidateModel(bad).ok()) << field;
  }
  for (const Json& invalid :
       {Json(true), Json("1"), Json(1.5), Json(-1), Json(UINT64_MAX)}) {
    auto bad = model;
    bad["states"][0]["id"] = invalid;
    EXPECT_FALSE(ValidateModel(bad).ok());
  }
  auto bad = model;
  bad["vocabulary"][0]["hex"] = "4A";
  EXPECT_FALSE(ValidateModel(bad).ok());
  bad = model;
  bad["entry"].push_back(bad["entry"][0]);
  EXPECT_FALSE(ValidateModel(bad).ok());
  bad = model;
  bad["states"][0]["members"] = {3};
  bad["states"][0]["member_count"] = 1;
  bad["states"][1]["members"] = {3};
  bad["states"][1]["member_count"] = 1;
  EXPECT_FALSE(ValidateModel(bad).ok());
  bad = model;
  bad["stats"]["state_unions"] = "invalid";
  EXPECT_FALSE(ValidateModel(bad).ok());
}

TEST(DiscretizeCoreTest, MalformedProvenanceStatsFailSafely) {
  const auto original = Mergeable();
  for (const auto& invalid : std::vector<Json>{
           nullptr, Json::array(), Json::object(),
           Json{{"euclidean_distance", true}, {"induced_unions", 0}},
           Json{{"euclidean_distance", -1}, {"induced_unions", 0}},
           Json{{"euclidean_distance", std::numeric_limits<double>::infinity()},
                {"induced_unions", 0}},
           Json{
               {"euclidean_distance", std::numeric_limits<double>::quiet_NaN()},
               {"induced_unions", 0}},
           Json{{"euclidean_distance", 1.5}, {"induced_unions", "0"}},
           Json{{"euclidean_distance", 1.5}, {"induced_unions", -1}}}) {
    auto model = original;
    model["stats"]["accepted_merges"] = Json::array({invalid});
    EXPECT_FALSE(ValidateModel(model).ok());
  }
  auto model = original;
  model["stats"]["accepted_merges"] = {
      {{"euclidean_distance", 0}, {"induced_unions", INT64_MAX}},
      {{"euclidean_distance", 1}, {"induced_unions", 1}}};
  EXPECT_FALSE(ValidateModel(model).ok());
  for (const auto& invalid : std::vector<Json>{
           nullptr, 7, Json::array(), Json{{"pairwise_irreducible", 1}},
           Json{{"pairwise_irreducible", "false"}}}) {
    model = original;
    model["stats"]["search"] = invalid;
    EXPECT_FALSE(ValidateModel(model).ok());
  }
  model = original;
  model["stats"]["accepted_merges"] = {
      {{"euclidean_distance", 1.5}, {"induced_unions", 2}}};
  model["stats"]["search"] = {{"pairwise_irreducible", false}};
  EXPECT_TRUE(ValidateModel(model).ok());
}

TEST(DiscretizeCoreTest, IncrementalClosureMatchesIndependentFullRescan) {
  // Independent oracle repeatedly rescans every immutable equation after each
  // tentative union. It shares no indexing, rollback, or rejection logic.
  std::mt19937 random(2917);
  for (int trial = 0; trial < 8; ++trial) {
    Json model = Header();
    model["states"] = Json::array();
    for (int i = 0; i < 21; ++i)
      model["states"].push_back(
          {{"id", 3 + i}, {"stage", i / 7}, {"bits", Json::array({100 + i})}});
    model["entry"] = {{0, 0, 3}};
    model["attention"] = Json::array({Json::array()});
    model["mlp"] = Json::array({Json::array()});
    model["language_modeling_head"] = Json::array();
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
      model["attention"][0].push_back({prefix, output});
      std::vector<int> inputs;
      for (int value : prefix)
        inputs.push_back(value - 3);
      terms.push_back({1, inputs, output - 3});
    }
    for (int i = 0; i < 7; ++i) {
      int output = 17 + random() % 7;
      model["mlp"][0].push_back({10 + i, output});
      terms.push_back({2, {7 + i}, output - 3});
      model["language_modeling_head"].push_back({17 + i, random() % 3});
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
        for (const auto& row : model["language_modeling_head"]) {
          auto [found, inserted] =
              labels.emplace(candidate[row[0].get<int>() - 3], row[1]);
          if (!inserted && found->second != row[1].get<int>())
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
