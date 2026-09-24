#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/mlp_pair_compaction.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model_util.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/state_compactor.h"

namespace pluto::llm::discretized::generator {
namespace {

// Both prompts predict B and then EOS. The same B input has different causal
// histories; the second attention must retain those distinctions.
// Nonconsecutive IDs deliberately make every MLP a permutation, not an
// accidental offset.
CapturedModel BranchingModel() {
  CapturedModel model;
  model.metadata = {.width = 1,
                    .layers = 2,
                    .vocab_size = 4,
                    .eos_token = 3,
                    .prompt_tokens = 1,
                    .vocabulary = {{0, "A"}, {1, "B"}, {2, "C"}, {3, "EOS"}},
                    .context_length = 4};
  const std::vector<std::vector<int>> boundaries = {{10, 11, 12},
                                                    {25, 21, 29, 24},
                                                    {46, 40, 42, 44},
                                                    {63, 65, 60, 69},
                                                    {81, 89, 83, 80}};
  for (int boundary = 0; boundary < static_cast<int>(boundaries.size());
       ++boundary)
    for (int id : boundaries[boundary])
      model.states.push_back({id, boundary, std::vector<int>{id}});
  model.samples = {{{0, 1}}, {{2, 1}}};
  model.position_embedding.transitions = {{0, 0, 10}, {2, 0, 11}, {1, 1, 12}};
  model.transformers.resize(2);
  model.transformers[0].attention.transitions = {
      {{10}, 25}, {{11}, 21}, {{10, 12}, 29}, {{11, 12}, 24}};
  model.transformers[0].mlp.transitions = {
      {25, 46}, {21, 40}, {29, 42}, {24, 44}};
  model.transformers[1].attention.transitions = {
      {{46}, 63}, {{40}, 65}, {{46, 42}, 60}, {{40, 44}, 69}};
  model.transformers[1].mlp.transitions = {
      {63, 81}, {65, 89}, {60, 83}, {69, 80}};
  model.language_modeling_head.transitions = {
      {81, 1}, {89, 1}, {83, 3}, {80, 3}};
  model.stats.captured_samples = 2;
  model.stats.scored_targets = 4;
  model.stats.exact_states = 19;
  model.stats.states = 19;
  model.stats.membership_complete = true;
  model.stats.membership_original_states = 19;
  model.stats.states_per_stage = {3, 4, 4, 4, 4};
  return model;
}

const CapturedState* FindState(const CapturedModel& model, int id) {
  for (const auto& state : model.states)
    if (state.id == id)
      return &state;
  return nullptr;
}

std::optional<int> Lookup(const CapturedMap& map, int input) {
  for (const auto& row : map.transitions)
    if (row.input == input)
      return row.output;
  return std::nullopt;
}

std::optional<int> Lookup(const CapturedCausalAttention& attention,
                          const std::vector<int>& prefix) {
  for (const auto& row : attention.transitions)
    if (row.prefix == prefix)
      return row.output;
  return std::nullopt;
}

// Check the finite input domain exhaustively, not just the two required facts.
// Unsupported histories must remain unsupported after the ID translation.
void CheckPredictions(const CapturedModel& before, const CapturedModel& after,
                      std::vector<int> prefix = {}) {
  const auto old_result = PredictNext(before, prefix);
  const auto new_result = PredictNext(after, prefix);
  ASSERT_EQ(old_result.ok(), new_result.ok());
  if (old_result.ok()) {
    const int expected = *old_result;
    EXPECT_EQ(expected, *new_result);
  }
  if (prefix.size() == static_cast<size_t>(before.metadata.context_length))
    return;
  for (int token = 0; token < before.metadata.vocab_size; ++token) {
    auto next = prefix;
    next.push_back(token);
    CheckPredictions(before, after, std::move(next));
  }
}

TEST(MlpPairCompactionTest, RewritesEveryConsumerAndPreservesAllPredictions) {
  const auto original = BranchingModel();
  ASSERT_TRUE(ValidateModel(original).ok());
  const auto original_verification = EvaluateModel(original);
  ASSERT_TRUE(original_verification.ok()) << original_verification.status();
  const auto result = CompactMlpPairs(original);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_TRUE(ValidateModel(*result).ok());
  const auto verification = EvaluateModel(*result);
  ASSERT_TRUE(verification.ok()) << verification.status();
  EXPECT_EQ(*verification, *original_verification);
  EXPECT_EQ(*verification, (VerificationResult{2, 4, 0, 2}));
  EXPECT_EQ(result->states.size(), 11u);
  EXPECT_EQ(result->stats.states, 11);
  EXPECT_EQ(result->stats.mlp_pair_compactions, 8);
  // Sharing IDs does not remove either layer boundary or its valid domain.
  EXPECT_EQ(result->stats.states_per_stage, original.stats.states_per_stage);
  EXPECT_EQ(result->metadata, original.metadata);
  EXPECT_EQ(result->samples, original.samples);
  EXPECT_EQ(result->position_embedding, original.position_embedding);

  std::map<int, int> translated;
  for (const auto& state : original.states)
    translated.emplace(state.id, state.id);
  for (const auto& transformer : original.transformers)
    for (const auto& row : transformer.mlp.transitions)
      translated[row.output] = row.input;

  std::set<int> all_members;
  for (const auto& state : result->states) {
    ASSERT_TRUE(state.members.has_value());
    for (int member : *state.members)
      EXPECT_TRUE(all_members.insert(member).second);
    if (state.boundary == 0) {
      EXPECT_FALSE(state.shared_boundary.has_value());
      EXPECT_EQ(state.members, (std::vector<int>{state.id}));
    } else {
      EXPECT_EQ(state.shared_boundary, state.boundary + 1);
      EXPECT_TRUE(state.HasBoundary(state.boundary));
      EXPECT_TRUE(state.HasBoundary(state.boundary + 1));
      EXPECT_FALSE(state.HasBoundary(state.boundary - 1));
    }
  }
  std::set<int> expected_members;
  for (const auto& state : original.states)
    expected_members.insert(state.id);
  EXPECT_EQ(all_members, expected_members);

  for (size_t layer = 0; layer < original.transformers.size(); ++layer) {
    const auto& before = original.transformers[layer];
    const auto& after = result->transformers[layer];
    ASSERT_EQ(before.attention.transitions.size(),
              after.attention.transitions.size());
    ASSERT_EQ(before.mlp.transitions.size(), after.mlp.transitions.size());
    for (const auto& row : before.attention.transitions) {
      auto prefix = row.prefix;
      for (int& input : prefix)
        input = translated.at(input);
      EXPECT_EQ(Lookup(after.attention, prefix), translated.at(row.output));
    }
    EXPECT_FALSE(Lookup(after.attention, {}).has_value());
    EXPECT_FALSE(Lookup(after.attention, {999}).has_value());
    for (const auto& row : before.mlp.transitions) {
      EXPECT_EQ(Lookup(after.mlp, row.input), row.input);
      const auto* shared = FindState(*result, row.input);
      ASSERT_NE(shared, nullptr);
      EXPECT_EQ(shared->members, (std::vector<int>{row.input, row.output}));
      EXPECT_EQ(FindState(*result, row.output), nullptr);
    }
    EXPECT_FALSE(Lookup(after.mlp, -1).has_value());
    EXPECT_FALSE(Lookup(after.mlp, 999).has_value());
  }
  for (const auto& row : original.language_modeling_head.transitions)
    EXPECT_EQ(Lookup(result->language_modeling_head, translated.at(row.input)),
              row.output);
  EXPECT_FALSE(Lookup(result->language_modeling_head, 999).has_value());
  CheckPredictions(original, *result);
  EXPECT_EQ(original, BranchingModel());
}

TEST(MlpPairCompactionTest, UnionsExistingMembershipWithoutLosingProvenance) {
  auto model = BranchingModel();
  for (auto& state : model.states)
    if (state.id == 25)
      state.members->push_back(125);
  model.stats.exact_states = 20;
  model.stats.membership_original_states = 20;
  model.stats.state_compactions = 1;
  const auto result = CompactMlpPairs(model);
  ASSERT_TRUE(result.ok()) << result.status();
  const auto* shared = FindState(*result, 25);
  ASSERT_NE(shared, nullptr);
  ASSERT_TRUE(shared->members.has_value());
  EXPECT_EQ((std::set<int>(shared->members->begin(), shared->members->end())),
            (std::set<int>{25, 46, 125}));
  EXPECT_EQ(result->stats.membership_original_states, 20);
  EXPECT_EQ(result->stats.state_compactions, 1);
}

TEST(MlpPairCompactionTest, FullyPairedInputIsIdempotent) {
  const auto first = CompactMlpPairs(BranchingModel());
  ASSERT_TRUE(first.ok()) << first.status();
  const auto second = CompactMlpPairs(*first);
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(*second, *first);
}

TEST(MlpPairCompactionTest,
     IdempotenceStillRequiresCompleteOriginalMembership) {
  const auto compacted = CompactMlpPairs(BranchingModel());
  ASSERT_TRUE(compacted.ok()) << compacted.status();
  for (int defect = 0; defect < 3; ++defect) {
    auto model = *compacted;
    auto state = std::find_if(
        model.states.begin(), model.states.end(), [&](const auto& row) {
          return defect == 1 ? row.boundary == 0
                             : row.shared_boundary.has_value();
        });
    ASSERT_NE(state, model.states.end());
    if (defect == 2)
      state->members->resize(1);
    else
      state->members.reset();
    // Both versions of the neural vector must remain identifiable even when
    // no transition changes are needed. Inference alone cannot detect this.
    ASSERT_TRUE(ValidateModel(model).ok());
    ASSERT_TRUE(EvaluateModel(model).ok());
    const auto before = model;
    const auto result = CompactMlpPairs(model);
    ASSERT_FALSE(result.ok()) << defect;
    EXPECT_EQ(result.status().code(), absl::StatusCode::kFailedPrecondition)
        << defect;
    EXPECT_EQ(model, before) << defect;
  }
}

TEST(MlpPairCompactionTest, InfersSingletonsOnlyForUntouchedCapture) {
  auto original = BranchingModel();
  for (auto& state : original.states)
    state.members.reset();
  original.stats.membership_complete = false;
  original.stats.membership_original_states.reset();
  const auto result = CompactMlpPairs(original);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->stats.membership_original_states, 19);
  const auto* shared = FindState(*result, 25);
  ASSERT_NE(shared, nullptr);
  EXPECT_EQ(shared->members, (std::vector<int>{25, 46}));

  original.stats.state_compactions = 1;
  EXPECT_FALSE(CompactMlpPairs(original).ok());
}

TEST(MlpPairCompactionTest,
     AcceptsWithinBoundaryRelabelingBeforePairCompaction) {
  const auto original = BranchingModel();
  const auto relabeled = RelabelMlpOutputs(original);
  ASSERT_TRUE(relabeled.ok()) << relabeled.status();
  const auto result = CompactMlpPairs(*relabeled);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->states.size(), 11u);
  EXPECT_EQ(result->stats.membership_original_states, 19);
  ASSERT_TRUE(ValidateModel(*result).ok());
  CheckPredictions(original, *result);
}

TEST(MlpPairCompactionTest, SharedStateRequiresExplicitIdentityMlp) {
  const auto compacted = CompactMlpPairs(BranchingModel());
  ASSERT_TRUE(compacted.ok()) << compacted.status();
  for (int defect = 0; defect < 2; ++defect) {
    auto model = *compacted;
    auto& rows = model.transformers[0].mlp.transitions;
    if (defect == 0)
      rows.erase(rows.begin());
    else
      rows.front().output = rows.back().output;
    EXPECT_FALSE(ValidateModel(model).ok()) << defect;
    EXPECT_FALSE(CompactMlpPairs(model).ok()) << defect;
  }
}

TEST(MlpPairCompactionTest, RejectsNonbijectionWithoutChangingSource) {
  auto model = BranchingModel();
  // The two first-position outputs both predict B, so this alteration remains
  // corpus-correct but is not an invertible MLP: two inputs share output 81.
  model.transformers[1].mlp.transitions[1].output = 81;
  model.language_modeling_head.transitions.erase(
      model.language_modeling_head.transitions.begin() + 1);
  std::erase_if(model.states, [](const auto& state) { return state.id == 89; });
  model.stats.exact_states = model.states.size();
  model.stats.states = model.states.size();
  model.stats.membership_original_states = model.states.size();
  model.stats.states_per_stage.back() = 3;
  ASSERT_TRUE(ValidateModel(model).ok());
  ASSERT_TRUE(EvaluateModel(model).ok());
  const auto original = model;
  const auto result = CompactMlpPairs(model);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(model, original);
}

TEST(MlpPairCompactionTest, RejectsIncompleteBijectionsAndUnknownMembership) {
  for (int defect = 0; defect < 3; ++defect) {
    auto model = BranchingModel();
    // Unused states still belong to the declared alphabet and must be covered.
    if (defect == 0)
      model.states.push_back({100, 1, std::vector<int>{100}});
    if (defect == 1)
      model.states.push_back({100, 2, std::vector<int>{100}});
    if (defect == 2) {
      model.states[0].members->push_back(100);
      model.states[1].members.reset();
      model.stats.state_compactions = 1;
    }
    model.stats.exact_states = 20;
    model.stats.states = model.states.size();
    model.stats.membership_original_states = 20;
    const auto original = model;
    EXPECT_FALSE(CompactMlpPairs(model).ok()) << defect;
    EXPECT_EQ(model, original) << defect;
  }
}

TEST(MlpPairCompactionTest, RejectsPartialSharingAndMalformedBoundaries) {
  auto partially_shared = BranchingModel();
  std::map<int, int> translated;
  for (auto& row : partially_shared.transformers[0].mlp.transitions) {
    translated.emplace(row.output, row.input);
    for (auto& state : partially_shared.states)
      if (state.id == row.input) {
        state.shared_boundary = 2;
        state.members->push_back(row.output);
      }
    row.output = row.input;
  }
  std::erase_if(partially_shared.states, [&](const auto& state) {
    return translated.contains(state.id);
  });
  for (auto& row : partially_shared.transformers[1].attention.transitions)
    for (int& input : row.prefix)
      input = translated.at(input);
  partially_shared.stats.states = partially_shared.states.size();
  ASSERT_TRUE(ValidateModel(partially_shared).ok());
  ASSERT_TRUE(EvaluateModel(partially_shared).ok());
  EXPECT_FALSE(CompactMlpPairs(partially_shared).ok());

  for (const auto& [boundary, shared] : std::vector<std::pair<int, int>>{
           {0, 1}, {1, 1}, {1, 3}, {1, 4}, {2, 3}, {3, 5}}) {
    auto malformed = BranchingModel();
    const auto state =
        std::find_if(malformed.states.begin(), malformed.states.end(),
                     [&](const auto& row) { return row.boundary == boundary; });
    ASSERT_NE(state, malformed.states.end());
    state->shared_boundary = shared;
    EXPECT_FALSE(ValidateModel(malformed).ok()) << boundary << ", " << shared;
    EXPECT_FALSE(CompactMlpPairs(malformed).ok()) << boundary << ", " << shared;
  }
}

TEST(MlpPairCompactionTest, EarlierBoundaryPassesRejectSharedInput) {
  const auto model = CompactMlpPairs(BranchingModel());
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_FALSE(StateCompactor::Create(*model).ok());
  EXPECT_FALSE(RelabelMlpOutputs(*model).ok());
}

}  // namespace
}  // namespace pluto::llm::discretized::generator
