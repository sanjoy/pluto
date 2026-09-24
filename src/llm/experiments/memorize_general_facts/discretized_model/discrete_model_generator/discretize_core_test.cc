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
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model_util.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/state_compactor.h"

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
CapturedModel Compactable() {
  return BuildModel(Header(), {Sample({0}, {{100}, {200}, {300}}),
                               Sample({1}, {{101}, {201}, {301}})})
      .value();
}
CapturedModel Branching() {
  return BuildModel(Header(),
                    {Sample({0, 1}, {{100, 102}, {200, 202}, {300, 302}}),
                     Sample({1}, {{101}, {201}, {301}})})
      .value();
}
// States appear in first-observation order within each boundary.
int State(const CapturedModel& model, int stage, int ordinal) {
  for (const auto& row : model.states)
    if (row.boundary == stage && ordinal-- == 0)
      return row.id;
  return -1;
}

TEST(DiscretizeCoreTest, BitExactDuplicatesPreserveSignedZero) {
  auto sample = Sample({0}, {{0}, {0x8000}, {0x3f80}});
  StateVectorHints hints;
  auto model = BuildModel(Header(), {sample, sample}, 2, &hints);
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ(model->states.size(), 3u);
  EXPECT_EQ(model->position_embedding.transitions.size(), 1u);
  EXPECT_EQ(hints.at(model->states[1].id)[0], 0x8000);
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
  EXPECT_FALSE(
      PredictNext(model, std::vector<int>(model.metadata.context_length + 1, 0))
          .ok());
  for (auto& row : model.language_modeling_head.transitions)
    if (row.input == State(model, 2, 1))
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
  for (const auto& row : model->transformers[0].attention.transitions)
    if (row.prefix.size() == 2)
      longer.push_back(row);
  ASSERT_EQ(longer.size(), 2u);
  EXPECT_EQ(longer[0].prefix[1], longer[1].prefix[1]);
  EXPECT_NE(longer[0].output, longer[1].output);
}

TEST(DiscretizeCoreTest, Context32PreservesSuffixAndEosAtLastValidLength) {
  auto metadata = Header();
  metadata.context_length = 32;
  std::vector<std::vector<int>> boundaries(3, std::vector<int>(32));
  for (int boundary = 0; boundary < 3; ++boundary)
    for (int position = 0; position < 32; ++position)
      boundaries[boundary][position] = 100 * (boundary + 1) + position;
  const auto sample = Sample(std::vector<int>(32, 0), boundaries);
  auto model = BuildModel(metadata, {sample});
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ(model->metadata.context_length, 32);
  EXPECT_EQ(model->stats.verification, (VerificationResult{1, 32, 0, 1}));
  const auto eos = PredictNext(*model, sample.tokens);
  ASSERT_TRUE(eos.ok()) << eos.status();
  EXPECT_EQ(*eos, metadata.eos_token);
  const auto overflow = PredictNext(*model, std::vector<int>(33, 0));
  EXPECT_FALSE(overflow.ok());
  EXPECT_NE(overflow.status().message().find("context length"),
            std::string::npos);

  metadata.context_length = 31;
  EXPECT_FALSE(BuildModel(metadata, {sample}).ok());
  model->metadata.context_length = 31;
  EXPECT_FALSE(ValidateModel(*model).ok());
}

TEST(DiscretizeCoreTest, ContextBoundsApplyToMetadataAndEveryTransitionDomain) {
  for (int context_length : {-1, 0}) {
    auto metadata = Header();
    metadata.context_length = context_length;
    EXPECT_FALSE(
        BuildModel(metadata, {Sample({0}, {{100}, {200}, {300}})}).ok());
  }
  auto model = Branching();
  model.metadata.context_length = 32;
  ASSERT_TRUE(ValidateModel(model).ok());
  for (int defect = 0; defect < 4; ++defect) {
    auto invalid = model;
    if (defect == 0)
      invalid.metadata.prompt_tokens = 33;
    if (defect == 1)
      invalid.position_embedding.transitions[0].position = 32;
    if (defect == 2) {
      auto& prefix = invalid.transformers[0].attention.transitions[0].prefix;
      prefix.resize(33, prefix.front());
    }
    if (defect == 3)
      invalid.samples[0].tokens.resize(33, 0);
    EXPECT_FALSE(ValidateModel(invalid).ok()) << defect;
  }
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
  auto model = Compactable();
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
  EXPECT_EQ(model->language_modeling_head.transitions.size(), 1u);
  auto compactor = StateCompactor::Create(*model);
  ASSERT_TRUE(compactor.ok()) << compactor.status();
  auto compacted =
      (*compactor)->TryCompact(State(*model, 2, 0), State(*model, 2, 1));
  ASSERT_TRUE(compacted.ok());
  EXPECT_TRUE(*compacted);
  EXPECT_TRUE(EvaluateModel((*compactor)->Export()).ok());
}

TEST(DiscretizeCoreTest, CongruenceClosureInducesAttentionAndMlpCompactions) {
  auto model = Compactable();
  auto compactor = StateCompactor::Create(model);
  ASSERT_TRUE(compactor.ok()) << compactor.status();
  auto compaction =
      (*compactor)->TryCompact(State(model, 0, 0), State(model, 0, 1));
  ASSERT_TRUE(compaction.ok());
  ASSERT_TRUE(*compaction);
  auto compacted = (*compactor)->Export();
  EXPECT_EQ(compacted.states.size(), 3u);
  EXPECT_EQ(compacted.stats.state_compactions, 3);
  EXPECT_EQ(compacted.stats.accepted_compactions[0].induced_compactions, 2);
  EXPECT_FALSE(compacted.stats.accepted_compactions[0].euclidean_distance);
  EXPECT_EQ(compacted.transformers[0].attention.transitions.size(), 1u);
  EXPECT_EQ(compacted.transformers[0].mlp.transitions.size(), 1u);
  EXPECT_TRUE(EvaluateModel(compacted).ok());
}

TEST(DiscretizeCoreTest, OptionalVectorHintsDoNotChangeCapturedModel) {
  const auto sample = Sample({0}, {{100}, {200}, {300}});
  StateVectorHints hints;
  auto plain = BuildModel(Header(), {sample});
  auto with_hints = BuildModel(Header(), {sample}, -1, &hints);
  ASSERT_TRUE(plain.ok());
  ASSERT_TRUE(with_hints.ok());
  EXPECT_EQ(*plain, *with_hints);
  EXPECT_EQ(hints.size(), plain->states.size());
  EXPECT_EQ(hints.at(plain->states[0].id), (std::vector<uint16_t>{100}));
  const auto before = hints;
  EXPECT_FALSE(BuildModel(Header(), {sample}, 2, &hints).ok());
  EXPECT_EQ(hints, before);
}

TEST(DiscretizeCoreTest, OptionalHintsPreserveDistanceOrderingNotValidity) {
  const ModelMetadata metadata{
      1, 1, 4, 3, 1, {{0, "A"}, {1, "B"}, {2, "C"}, {3, "EOS"}}};
  StateVectorHints hints;
  auto model = BuildModel(metadata,
                          {Sample({0}, {{0}, {0}, {0}}, {3}),
                           Sample({1}, {{0x4120}, {0x4120}, {0x4120}}, {3}),
                           Sample({2}, {{0x3f80}, {0x3f80}, {0x3f80}}, {3})},
                          -1, &hints);
  ASSERT_TRUE(model.ok()) << model.status();
  CompactionOptions options;
  options.neighbors = 1;
  options.max_attempts = 1;
  options.exhaustive_pair_limit = 0;
  auto symbolic = CompactModel(*model, options);
  auto symbolic_again = CompactModel(*model, options);
  auto guided = CompactModel(*model, options, &hints);
  ASSERT_TRUE(symbolic.ok()) << symbolic.status();
  ASSERT_TRUE(symbolic_again.ok()) << symbolic_again.status();
  ASSERT_TRUE(guided.ok()) << guided.status();
  ASSERT_EQ(symbolic->stats.accepted_compactions.size(), 1u);
  ASSERT_EQ(guided->stats.accepted_compactions.size(), 1u);
  EXPECT_EQ(symbolic->stats.accepted_compactions[0].seed_ids,
            (std::array<int, 2>{4, 7}));
  EXPECT_EQ(guided->stats.accepted_compactions[0].seed_ids,
            (std::array<int, 2>{4, 10}));
  EXPECT_EQ(guided->stats.accepted_compactions[0].euclidean_distance, 1);
  EXPECT_FALSE(symbolic->stats.accepted_compactions[0].euclidean_distance);
  EXPECT_TRUE(EvaluateModel(*guided).ok());
  EXPECT_TRUE(EvaluateModel(*symbolic).ok());
  symbolic->stats = {};
  symbolic_again->stats = {};
  EXPECT_EQ(*symbolic, *symbolic_again);
}

TEST(DiscretizeCoreTest, VectorHintsAreValidatedAndCopied) {
  StateVectorHints hints;
  auto model = BuildModel(
      Header(),
      {Sample({0}, {{100}, {200}, {300}}), Sample({1}, {{101}, {201}, {301}})},
      -1, &hints);
  ASSERT_TRUE(model.ok());
  const auto valid = hints;
  const int first_id = State(*model, 0, 0);
  const int second_id = State(*model, 0, 1);
  hints.erase(first_id);
  EXPECT_FALSE(StateCompactor::Create(*model, &hints).ok());
  hints = valid;
  hints[first_id].clear();
  EXPECT_FALSE(StateCompactor::Create(*model, &hints).ok());
  hints = valid;
  hints[first_id][0] = 0x7f80;
  EXPECT_FALSE(StateCompactor::Create(*model, &hints).ok());
  hints = valid;
  auto compactor = StateCompactor::Create(*model, &hints);
  ASSERT_TRUE(compactor.ok()) << compactor.status();
  hints.clear();  // The compactor must not retain a borrowed map.
  ASSERT_TRUE((*compactor)->TryCompact(first_id, second_id).value());
  EXPECT_TRUE(EvaluateModel((*compactor)->Export()).ok());
}

TEST(DiscretizeCoreTest,
     RejectedTrialRestoresAllTransitionsAndCachesRejection) {
  auto model = Branching();
  auto compactor = StateCompactor::Create(model);
  ASSERT_TRUE(compactor.ok()) << compactor.status();
  CapturedModel before = (*compactor)->Export();
  for (int iteration = 0; iteration < 2; ++iteration) {
    auto compacted =
        (*compactor)->TryCompact(State(model, 0, 0), State(model, 0, 2));
    ASSERT_TRUE(compacted.ok());
    EXPECT_FALSE(*compacted);
  }
  CapturedModel after = (*compactor)->Export();
  EXPECT_EQ(after.stats.cached_rejections, 1);
  EXPECT_EQ(after.stats.attempted_seeds, 1);
  before.stats = {};
  after.stats = {};
  EXPECT_EQ(before, after);
  auto compacted =
      (*compactor)->TryCompact(State(model, 2, 2), State(model, 2, 1));
  ASSERT_TRUE(compacted.ok());
  EXPECT_TRUE(*compacted);
  EXPECT_TRUE(EvaluateModel((*compactor)->Export()).ok());
  EXPECT_FALSE(
      (*compactor)->TryCompact(State(model, 0, 0), State(model, 1, 0)).ok());
  EXPECT_FALSE((*compactor)->TryCompact(-1, -1).ok());
}

TEST(DiscretizeCoreTest, SearchLimitsAreNotMisreportedAsMinimality) {
  CompactionOptions options;
  options.neighbors = 1;
  options.max_passes = 3;
  auto compacted = CompactModel(Branching(), options);
  ASSERT_TRUE(compacted.ok()) << compacted.status();
  ASSERT_TRUE(compacted->stats.compaction_search.has_value());
  EXPECT_TRUE(compacted->stats.compaction_search->pairwise_compaction_complete);
  EXPECT_FALSE(compacted->stats.compaction_search->global_minimum_proven);
  options.max_attempts = 0;
  auto limited = CompactModel(Branching(), options);
  ASSERT_TRUE(limited.ok()) << limited.status();
  ASSERT_TRUE(limited->stats.compaction_search.has_value());
  EXPECT_EQ(limited->stats.compaction_search->stopping_reason,
            CompactionStoppingReason::kAttemptLimit);
  EXPECT_FALSE(limited->stats.compaction_search->pairwise_compaction_complete);
  options.max_attempts.reset();
  options.max_passes = 1;
  options.exhaustive_pair_limit = 0;
  auto shortlist = CompactModel(Branching(), options);
  ASSERT_TRUE(shortlist.ok()) << shortlist.status();
  ASSERT_TRUE(shortlist->stats.compaction_search.has_value());
  EXPECT_FALSE(
      shortlist->stats.compaction_search->pairwise_compaction_complete);
  options.neighbors = 0;
  EXPECT_FALSE(CompactModel(Branching(), options).ok());
}

TEST(DiscretizeCoreTest, MembershipIsPreservedAndCanBeRecovered) {
  const auto original = Compactable();
  auto compactor = StateCompactor::Create(original);
  ASSERT_TRUE(compactor.ok()) << compactor.status();
  ASSERT_TRUE((*compactor)
                  ->TryCompact(State(original, 2, 0), State(original, 2, 1))
                  .value());
  const auto partial = (*compactor)->Export();
  auto legacy = partial;
  for (auto& row : legacy.states)
    row.members.reset();
  auto restored = RestoreMembership(legacy, original);
  ASSERT_TRUE(restored.ok()) << restored.status();
  EXPECT_EQ(restored->states, partial.states);
  auto compacted = CompactModel(*restored);
  ASSERT_TRUE(compacted.ok()) << compacted.status();
  for (const auto& row : compacted->states) {
    ASSERT_TRUE(row.members.has_value());
    EXPECT_EQ(row.members->size(), 2u);
  }
  legacy.language_modeling_head.transitions[0].output = 0;
  EXPECT_FALSE(RestoreMembership(legacy, original).ok());
}

TEST(DiscretizeCoreTest, CancellationIsCooperativeAndSignalHandlerRestored) {
  CompactionOptions options;
  options.interrupted = [] { return true; };
  auto cancelled = CompactModel(Compactable(), options);
  EXPECT_EQ(cancelled.status().code(), absl::StatusCode::kCancelled);
  void (*previous)(int) = std::signal(SIGINT, SIG_IGN);
  options.interrupted = {};
  options.progress = [](const CompactionProgress&) { std::raise(SIGINT); };
  cancelled = CompactModel(Compactable(), options);
  EXPECT_EQ(cancelled.status().code(), absl::StatusCode::kCancelled);
  EXPECT_EQ(std::signal(SIGINT, previous), SIG_IGN);
}

TEST(DiscretizeCoreTest, MalformedTypedModelsReturnStatusInsteadOfAborting) {
  // Field types rule out malformed documents; dimensions, relationships, and
  // numerical ranges remain runtime invariants and must still be checked.
  const std::vector<std::function<void(CapturedModel&)>> defects = {
      [](auto& m) { m.metadata.width = 0; },
      [](auto& m) { m.metadata.context_length = 0; },
      [](auto& m) { m.metadata.context_length = -1; },
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
      [](auto& m) { m.states[0].boundary = 1; },
      [](auto& m) { m.states[0].members = std::vector<int>{}; },
      [](auto& m) {
        m.states[0].members = std::vector<int>{3};
        m.states[1].members = std::vector<int>{3};
      },
      [](auto& m) { m.position_embedding.transitions.clear(); },
      [](auto& m) {
        m.position_embedding.transitions.push_back(
            m.position_embedding.transitions.front());
      },
      [](auto& m) {
        m.position_embedding.transitions[0].position =
            m.metadata.context_length;
      },
      [](auto& m) { m.transformers.clear(); },
      [](auto& m) {
        m.transformers[0].attention.transitions[0].prefix.clear();
      },
      [](auto& m) {
        m.transformers[0].mlp.transitions[0].input =
            m.position_embedding.transitions[0].output;
      },
      [](auto& m) { m.language_modeling_head.transitions.clear(); },
      [](auto& m) { m.samples.clear(); },
      [](auto& m) { m.samples[0].tokens.clear(); },
      [](auto& m) { m.stats.state_compactions = -1; },
  };
  for (size_t index = 0; index < defects.size(); ++index) {
    auto bad = Compactable();
    defects[index](bad);
    EXPECT_FALSE(ValidateModel(bad).ok()) << index;
  }
  auto binary_vocabulary = Compactable();
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
  const auto original = Compactable();
  for (const auto& invalid : std::vector<CompactionRecord>{
           {.euclidean_distance = -1},
           {.euclidean_distance = std::numeric_limits<double>::infinity()},
           {.euclidean_distance = std::numeric_limits<double>::quiet_NaN()},
           {.euclidean_distance = 1.5, .induced_compactions = -1}}) {
    auto model = original;
    model.stats.accepted_compactions = {invalid};
    EXPECT_FALSE(ValidateModel(model).ok());
  }
  auto model = original;
  model.stats.accepted_compactions = {
      {.euclidean_distance = 0, .induced_compactions = INT64_MAX},
      {.euclidean_distance = 1, .induced_compactions = 1}};
  EXPECT_FALSE(ValidateModel(model).ok());
  model = original;
  model.stats.accepted_compactions = {
      {.euclidean_distance = 1.5, .induced_compactions = 2}};
  model.stats.compaction_search =
      CompactionSearchStatistics{.pairwise_compaction_complete = false};
  EXPECT_TRUE(ValidateModel(model).ok());
}

TEST(DiscretizeCoreTest, IncrementalClosureMatchesIndependentFullRescan) {
  // Independent oracle repeatedly rescans every immutable equation after each
  // tentative compaction. It shares no indexing, rollback, or rejection logic.
  std::mt19937 random(2917);
  for (int trial = 0; trial < 8; ++trial) {
    CapturedModel model;
    model.metadata = Header();
    for (int i = 0; i < 21; ++i)
      model.states.push_back({3 + i, i / 7, std::nullopt});
    model.position_embedding.transitions = {{0, 0, 3}};
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
      model.transformers[0].attention.transitions.push_back({prefix, output});
      std::vector<int> inputs;
      for (int value : prefix)
        inputs.push_back(value - 3);
      terms.push_back({1, inputs, output - 3});
    }
    for (int i = 0; i < 7; ++i) {
      int output = 17 + random() % 7;
      model.transformers[0].mlp.transitions.push_back({10 + i, output});
      terms.push_back({2, {7 + i}, output - 3});
      model.language_modeling_head.transitions.push_back(
          {17 + i, static_cast<int>(random() % 3)});
    }
    auto compactor = StateCompactor::Create(model);
    ASSERT_TRUE(compactor.ok()) << compactor.status();
    std::vector<int> equivalence(21);
    std::iota(equivalence.begin(), equivalence.end(), 0);
    auto oracle = [&](int first,
                      int second) -> std::optional<std::vector<int>> {
      auto candidate = equivalence;
      auto compact_pair = [&](int a, int b) {
        a = candidate[a];
        b = candidate[b];
        if (a == b)
          return false;
        for (int& root : candidate)
          if (root == a || root == b)
            root = std::min(a, b);
        return true;
      };
      compact_pair(first, second);
      while (true) {
        bool changed = false;
        std::map<std::vector<int>, int> signatures;
        for (const auto& term : terms) {
          std::vector<int> signature{term.stage};
          for (int argument : term.inputs)
            signature.push_back(candidate[argument]);
          auto [found, inserted] = signatures.emplace(signature, term.output);
          if (!inserted)
            changed |= compact_pair(term.output, found->second);
        }
        std::map<int, int> labels;
        for (const auto& row : model.language_modeling_head.transitions) {
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
      auto actual = (*compactor)->TryCompact(first + 3, second + 3);
      ASSERT_TRUE(actual.ok()) << actual.status();
      ASSERT_EQ(*actual, expected.has_value());
      if (expected)
        equivalence = *expected;
      for (int a = 0; a < 21; ++a)
        for (int b = 0; b < 21; ++b)
          ASSERT_EQ(equivalence[a] == equivalence[b],
                    (*compactor)->RootForState(a + 3) ==
                        (*compactor)->RootForState(b + 3));
    }
  }
}
}  // namespace
}  // namespace pluto::llm::discretized::generator
