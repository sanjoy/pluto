#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/state_vector_codegen.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model_util.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_state_vectors.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/mlp_pair_compaction.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/state_compactor.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized::generator {
namespace {

struct CaptureFixture {
  CapturedModel model;
  CapturedStateVectors vectors;
};

// Every one-token sample predicts EOS, so its states can be combined at each
// boundary. All captured vectors are nevertheless different: inspection must
// retain their individual values rather than one representative per class.
absl::StatusOr<CaptureFixture> Capture(int width, int sample_count = 3,
                                       bool repeat_observations = false) {
  const ModelMetadata metadata{
      .width = width,
      .layers = 2,
      .vocab_size = 4,
      .eos_token = 3,
      .prompt_tokens = 1,
      .vocabulary = {{0, "A"}, {1, "B"}, {2, "C"}, {3, "EOS"}},
      .context_length = 27};
  std::vector<ExecutionSample> samples;
  for (int sample = 0; sample < sample_count; ++sample) {
    ExecutionSample capture{{sample}, {3}, {}};
    for (int boundary = 0; boundary < 5; ++boundary) {
      std::vector<uint16_t> values;
      for (int channel = 0; channel < width; ++channel)
        values.push_back(0x3f00 + (boundary * sample_count + sample) * width +
                         channel);
      // Preserve signed zero as bits, even though the two floating-point
      // values compare equal. They must never be normalized by serialization.
      if (boundary == 0 && sample == 0) {
        values[0] = 0x0000;
        if (width > 1)
          values[1] = 0x8000;
      }
      capture.boundaries.push_back({std::move(values)});
    }
    samples.push_back(std::move(capture));
  }
  if (repeat_observations) {
    const auto original = samples;
    samples.insert(samples.end(), original.begin(), original.end());
  }
  CaptureFixture fixture;
  ASSIGN_OR_RETURN(fixture.model, BuildModel(metadata, samples, -1,
                                             &fixture.vectors.original_states));
  for (const auto& state : fixture.model.states)
    fixture.vectors.original_boundaries.emplace(state.id, state.boundary);
  return fixture;
}

std::vector<int> BoundaryStates(const CapturedModel& model, int boundary) {
  std::vector<int> states;
  for (const auto& state : model.states)
    if (state.HasBoundary(boundary))
      states.push_back(state.id);
  return states;
}

// A seed combination at the entry boundary forces the corresponding attention
// and MLP outputs to combine too. Export must union their original memberships
// even when the next compactor sees only the already-compacted model.
absl::StatusOr<CapturedModel> CombineFirstTwo(const CapturedModel& model) {
  const auto states = BoundaryStates(model, 0);
  if (states.size() < 2)
    return absl::InvalidArgumentError("fixture needs two entry states");
  ASSIGN_OR_RETURN(auto compactor, StateCompactor::Create(model));
  ASSIGN_OR_RETURN(bool combined, compactor->TryCompact(states[0], states[1]));
  if (!combined)
    return absl::InternalError("fixture states could not be combined");
  return compactor->Export();
}

void ExpectOriginalMembership(const CapturedModel& model,
                              const CaptureFixture& original) {
  std::map<int, int> original_boundaries;
  for (const auto& state : original.model.states)
    original_boundaries.emplace(state.id, state.boundary);
  std::set<int> seen;
  for (const auto& state : model.states) {
    ASSERT_TRUE(state.members.has_value()) << state.id;
    ASSERT_FALSE(state.members->empty()) << state.id;
    for (int member : *state.members) {
      ASSERT_TRUE(original_boundaries.contains(member)) << member;
      EXPECT_TRUE(state.HasBoundary(original_boundaries.at(member)));
      EXPECT_TRUE(original.vectors.original_states.contains(member));
      EXPECT_TRUE(seen.insert(member).second) << member;
    }
  }
  EXPECT_EQ(seen.size(), original.vectors.original_states.size());
}

// Read only the generated data shards, not the printer or its generated test.
// Comparing every BF16 word detects both dropped members and accidentally
// repeated representatives without depending on whitespace or shard sizes.
std::vector<uint16_t> EmittedWords(
    const std::map<std::string, std::string>& files) {
  std::vector<uint16_t> words;
  for (const auto& [name, source] : files) {
    constexpr char kPrefix[] = "state_vectors_";
    constexpr size_t kPrefixLength = sizeof(kPrefix) - 1;
    if (!name.starts_with(kPrefix) || name.size() <= kPrefixLength ||
        name[kPrefixLength] < '0' || name[kPrefixLength] > '9' ||
        !name.ends_with(".cc"))
      continue;
    size_t position = 0;
    while ((position = source.find("0x", position)) != std::string::npos) {
      position += 2;
      EXPECT_GE(source.size() - position, 4u);
      uint16_t word = 0;
      for (size_t offset = 0; offset < 4 && position + offset < source.size();
           ++offset) {
        const char digit = source[position + offset];
        EXPECT_TRUE((digit >= '0' && digit <= '9') ||
                    (digit >= 'a' && digit <= 'f'));
        word = (word << 4) + (digit <= '9' ? digit - '0' : digit - 'a' + 10);
      }
      words.push_back(word);
      position += 4;
    }
  }
  std::sort(words.begin(), words.end());
  return words;
}

std::string WithoutWhitespace(std::string text) {
  text.erase(std::remove_if(text.begin(), text.end(),
                            [](unsigned char c) { return std::isspace(c); }),
             text.end());
  return text;
}

std::string BoundaryNameForTest(int boundary) {
  if (boundary == 0)
    return "token_plus_position_embedding";
  return "block_" + std::to_string((boundary - 1) / 2) +
         (boundary % 2 ? ".after_attention_residual" : ".after_mlp_residual");
}

void ExpectAllVectorsEmitted(const CapturedModel& model,
                             const CapturedStateVectors& vectors) {
  const auto rendered = RenderStateVectors(model, vectors);
  ASSERT_TRUE(rendered.ok()) << rendered.status();
  EXPECT_TRUE(rendered->contains("state_vectors.h"));
  EXPECT_TRUE(rendered->contains("state_vectors.cc"));
  std::vector<uint16_t> expected;
  for (const auto& [id, values] : vectors.original_states)
    expected.insert(expected.end(), values.begin(), values.end());
  std::sort(expected.begin(), expected.end());
  EXPECT_EQ(EmittedWords(*rendered), expected);

  // These small fixtures fit in one shard. Check row association as well as
  // the multiset of bits: a permutation of vectors among states is a bug even
  // though it preserves every original floating-point value.
  ASSERT_TRUE(rendered->contains("state_vectors_0.cc"));
  const std::string shard =
      WithoutWhitespace(rendered->at("state_vectors_0.cc"));
  for (const auto& [id, values] : vectors.original_states) {
    std::ostringstream row;
    row << "//Originalstate" << id << '.';
    for (uint16_t word : values)
      row << "0x" << std::hex << std::setfill('0') << std::setw(4) << word
          << ',';
    const std::string expected_row = row.str();
    const auto first = shard.find(expected_row);
    ASSERT_NE(first, std::string::npos) << id;
    EXPECT_EQ(shard.find(expected_row, first + expected_row.size()),
              std::string::npos)
        << id;
  }
  std::map<int, const CapturedState*> states;
  for (const auto& state : model.states)
    states.emplace(state.id, &state);
  const auto printer = WithoutWhitespace(rendered->at("state_vectors.cc"));
  std::string original_ids = "constintkOriginalIds[]={";
  std::string original_boundaries = "constintkOriginalBoundaries[]={";
  size_t offset = 0;
  for (const auto& [id, state] : states) {
    auto boundary = BoundaryNameForTest(state->boundary);
    if (state->shared_boundary)
      boundary += "+" + BoundaryNameForTest(*state->shared_boundary);
    ASSERT_TRUE(state->members.has_value());
    auto members = *state->members;
    std::sort(members.begin(), members.end());
    const auto index_row = "{" + std::to_string(id) + ",\"" + boundary + "\"," +
                           std::to_string(offset) + "," +
                           std::to_string(members.size()) + "}";
    EXPECT_NE(printer.find(index_row), std::string::npos) << id;
    for (int member : members) {
      original_ids += std::to_string(member) + ',';
      const int original_boundary =
          vectors.original_boundaries.empty()
              ? state->boundary
              : vectors.original_boundaries.at(member);
      original_boundaries += std::to_string(original_boundary) + ',';
    }
    offset += members.size();
  }
  original_ids += "};";
  original_boundaries += "};";
  EXPECT_NE(shard.find(original_ids), std::string::npos);
  EXPECT_NE(shard.find(original_boundaries), std::string::npos);
}

TEST(StateVectorCodegenTest, CapturePreservesEveryDistinctVectorAtActualWidth) {
  for (int width : {1, 10, 13}) {
    SCOPED_TRACE(width);
    const auto captured = Capture(width, 3, true);
    ASSERT_TRUE(captured.ok()) << captured.status();
    // Repeated corpus observations do not create duplicate original vectors.
    ASSERT_EQ(captured->model.states.size(), 15u);
    ASSERT_EQ(captured->vectors.original_states.size(), 15u);
    EXPECT_TRUE(captured->model.stats.membership_complete);
    EXPECT_EQ(captured->model.stats.membership_original_states, 15);
    for (const auto& state : captured->model.states) {
      ASSERT_TRUE(state.members.has_value());
      EXPECT_EQ(*state.members, (std::vector<int>{state.id}));
      EXPECT_EQ(captured->vectors.original_states.at(state.id).size(),
                static_cast<size_t>(width));
    }
    ExpectOriginalMembership(captured->model, *captured);
    ExpectAllVectorsEmitted(captured->model, captured->vectors);
  }
}

TEST(StateVectorCodegenTest, RepeatedCompactionAndRelabelKeepEveryOriginal) {
  const auto captured = Capture(10);
  ASSERT_TRUE(captured.ok()) << captured.status();
  const auto first = CombineFirstTwo(captured->model);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_EQ(first->states.size(), 10u);
  ExpectOriginalMembership(*first, *captured);
  ExpectAllVectorsEmitted(*first, captured->vectors);

  const auto second = CombineFirstTwo(*first);
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_EQ(second->states.size(), 5u);
  for (const auto& state : second->states) {
    ASSERT_TRUE(state.members.has_value());
    EXPECT_EQ(state.members->size(), 3u);
  }
  const auto relabeled = RelabelMlpOutputs(*second);
  ASSERT_TRUE(relabeled.ok()) << relabeled.status();
  ASSERT_TRUE(relabeled->stats.pointwise_relabeling.has_value());
  EXPECT_GT(relabeled->stats.pointwise_relabeling->changed_states, 0);
  ExpectOriginalMembership(*relabeled, *captured);
  ExpectAllVectorsEmitted(*relabeled, captured->vectors);

  // Compaction may itself renumber states again after a vocabulary-aligned
  // relabel. Vector lookup must always use original members, not current IDs.
  const auto third = CompactModel(*relabeled);
  ASSERT_TRUE(third.ok()) << third.status();
  ExpectOriginalMembership(*third, *captured);
  ExpectAllVectorsEmitted(*third, captured->vectors);
}

TEST(StateVectorCodegenTest, RelabelWithoutCompactionKeepsOriginalVectorIds) {
  const auto captured = Capture(13, 1);
  ASSERT_TRUE(captured.ok()) << captured.status();
  const auto relabeled = RelabelMlpOutputs(captured->model);
  ASSERT_TRUE(relabeled.ok()) << relabeled.status();
  ASSERT_TRUE(relabeled->stats.pointwise_relabeling.has_value());
  EXPECT_GT(relabeled->stats.pointwise_relabeling->changed_states, 0);
  ExpectOriginalMembership(*relabeled, *captured);
  ExpectAllVectorsEmitted(*relabeled, captured->vectors);
}

TEST(StateVectorCodegenTest, SharedMlpSymbolsKeepBothBoundariesAndEveryVector) {
  const auto captured = Capture(10);
  ASSERT_TRUE(captured.ok()) << captured.status();
  const auto compacted = CompactModel(captured->model);
  ASSERT_TRUE(compacted.ok()) << compacted.status();
  const auto relabeled = RelabelMlpOutputs(*compacted);
  ASSERT_TRUE(relabeled.ok()) << relabeled.status();
  const auto paired = CompactMlpPairs(*relabeled);
  ASSERT_TRUE(paired.ok()) << paired.status();
  ASSERT_EQ(paired->states.size(), 3u);
  size_t original_count = 0;
  int shared_count = 0;
  for (const auto& state : paired->states) {
    ASSERT_TRUE(state.members.has_value());
    original_count += state.members->size();
    if (!state.shared_boundary)
      continue;
    ++shared_count;
    // Three original pre-MLP activations and three post-MLP activations.
    EXPECT_EQ(state.members->size(), 6u);
    std::map<int, int> counts;
    for (int member : *state.members)
      ++counts[captured->vectors.original_boundaries.at(member)];
    EXPECT_EQ(counts[state.boundary], 3);
    EXPECT_EQ(counts[*state.shared_boundary], 3);
  }
  EXPECT_EQ(shared_count, 2);
  EXPECT_EQ(original_count, captured->vectors.original_states.size());
  ExpectOriginalMembership(*paired, *captured);
  ExpectAllVectorsEmitted(*paired, captured->vectors);
  const auto emitted = RenderStateVectors(*paired, captured->vectors);
  ASSERT_TRUE(emitted.ok()) << emitted.status();
  const auto source = WithoutWhitespace(emitted->at("state_vectors.cc"));
  EXPECT_NE(source.find(
                "block_0.after_attention_residual+block_0.after_mlp_residual"),
            std::string::npos);
  EXPECT_NE(source.find(
                "block_1.after_attention_residual+block_1.after_mlp_residual"),
            std::string::npos);
}

TEST(StateVectorCodegenTest, SharedStatesRequireCompleteCorrectBoundaries) {
  const auto captured = Capture(13);
  ASSERT_TRUE(captured.ok()) << captured.status();
  const auto paired = CompactMlpPairs(captured->model);
  ASSERT_TRUE(paired.ok()) << paired.status();
  const auto shared = std::find_if(paired->states.begin(), paired->states.end(),
                                   [](const CapturedState& state) {
                                     return state.shared_boundary.has_value();
                                   });
  ASSERT_NE(shared, paired->states.end());
  ASSERT_TRUE(shared->members.has_value());
  const int original = shared->members->front();
  for (int defect = 0; defect < 6; ++defect) {
    SCOPED_TRACE(defect);
    auto invalid = captured->vectors;
    if (defect == 0)
      invalid.original_boundaries.clear();
    if (defect == 1)
      invalid.original_boundaries.erase(original);
    if (defect == 2)
      invalid.original_boundaries.emplace(1000000, 1);
    if (defect == 3)
      invalid.original_boundaries.at(original) = 0;
    if (defect == 4) {
      invalid.original_boundaries.erase(original);
      invalid.original_boundaries.emplace(1000000, 1);
    }
    if (defect == 5)
      for (int member : *shared->members)
        invalid.original_boundaries.at(member) = shared->boundary;
    EXPECT_FALSE(RenderStateVectors(*paired, invalid).ok());
  }
}

TEST(StateVectorCodegenTest, BoundaryMetadataMayBeInferredOnlyWithoutSharing) {
  const auto captured = Capture(10);
  ASSERT_TRUE(captured.ok()) << captured.status();
  auto legacy = captured->vectors;
  legacy.original_boundaries.clear();
  const auto expected = RenderStateVectors(captured->model, captured->vectors);
  const auto inferred = RenderStateVectors(captured->model, legacy);
  ASSERT_TRUE(expected.ok()) << expected.status();
  ASSERT_TRUE(inferred.ok()) << inferred.status();
  EXPECT_EQ(*inferred, *expected);

  const int original = captured->model.states.front().id;
  auto invalid = captured->vectors;
  invalid.original_boundaries.at(original) = 1;
  EXPECT_FALSE(RenderStateVectors(captured->model, invalid).ok());
}

TEST(StateVectorCodegenTest, RejectsInvalidSharedBoundaryMetadata) {
  const auto captured = Capture(10);
  ASSERT_TRUE(captured.ok()) << captured.status();
  for (const auto& [primary, secondary] :
       {std::pair{0, 1}, std::pair{1, 3}, std::pair{2, 3}, std::pair{3, 5}}) {
    auto invalid = captured->model;
    invalid.states.front().boundary = primary;
    invalid.states.front().shared_boundary = secondary;
    EXPECT_FALSE(RenderStateVectors(invalid, captured->vectors).ok());
  }
}

TEST(StateVectorCodegenTest, EmissionDoesNotDependOnContainerIterationOrder) {
  const auto captured = Capture(10);
  ASSERT_TRUE(captured.ok()) << captured.status();
  auto compacted = CompactModel(captured->model);
  ASSERT_TRUE(compacted.ok()) << compacted.status();
  const auto expected = RenderStateVectors(*compacted, captured->vectors);
  ASSERT_TRUE(expected.ok()) << expected.status();

  std::reverse(compacted->states.begin(), compacted->states.end());
  CapturedStateVectors reordered;
  for (auto& state : compacted->states) {
    ASSERT_TRUE(state.members.has_value());
    std::reverse(state.members->begin(), state.members->end());
    for (int member : *state.members) {
      reordered.original_states.emplace(
          member, captured->vectors.original_states.at(member));
      reordered.original_boundaries.emplace(
          member, captured->vectors.original_boundaries.at(member));
    }
  }
  const auto actual = RenderStateVectors(*compacted, reordered);
  ASSERT_TRUE(actual.ok()) << actual.status();
  EXPECT_EQ(*actual, *expected);
}

TEST(StateVectorCodegenTest, OneCompactedStateCanSpanMultipleSourceShards) {
  auto captured = Capture(1, 1);
  ASSERT_TRUE(captured.ok()) << captured.status();
  auto& state = captured->model.states[0];
  ASSERT_TRUE(state.members.has_value());
  for (int index = 0; index < 8193; ++index) {
    const int original = 1000 + index;
    state.members->push_back(original);
    captured->vectors.original_states.emplace(
        original, std::vector<uint16_t>{static_cast<uint16_t>(index + 1)});
    captured->vectors.original_boundaries.emplace(original, state.boundary);
  }
  captured->model.stats.membership_original_states =
      captured->vectors.original_states.size();
  const auto rendered = RenderStateVectors(captured->model, captured->vectors);
  ASSERT_TRUE(rendered.ok()) << rendered.status();
  EXPECT_TRUE(rendered->contains("state_vectors_0.cc"));
  EXPECT_TRUE(rendered->contains("state_vectors_1.cc"));
  EXPECT_FALSE(rendered->contains("state_vectors_2.cc"));
  std::vector<uint16_t> expected;
  for (const auto& [id, words] : captured->vectors.original_states)
    expected.insert(expected.end(), words.begin(), words.end());
  std::sort(expected.begin(), expected.end());
  EXPECT_EQ(EmittedWords(*rendered), expected);
  // The group's contiguous global range crosses a shard boundary; it must
  // remain a single state rather than silently losing its final vectors.
  const auto index_row = "{" + std::to_string(state.id) +
                         ",\"token_plus_position_embedding\",0,8194}";
  EXPECT_NE(WithoutWhitespace(rendered->at("state_vectors.cc")).find(index_row),
            std::string::npos);
}

TEST(StateVectorCodegenTest, EmptyArchiveNeedsNoVectorDataShard) {
  CapturedModel model;
  model.metadata.width = 10;
  const auto rendered = RenderStateVectors(model, {});
  ASSERT_TRUE(rendered.ok()) << rendered.status();
  EXPECT_TRUE(rendered->contains("state_vectors.h"));
  EXPECT_TRUE(rendered->contains("state_vectors.cc"));
  EXPECT_TRUE(rendered->contains("state_vectors_test.cc"));
  EXPECT_EQ(rendered->size(), 3u);
  EXPECT_TRUE(EmittedWords(*rendered).empty());
}

TEST(StateVectorCodegenTest, MissingOrAmbiguousMembershipIsRejected) {
  const auto captured = Capture(10);
  ASSERT_TRUE(captured.ok()) << captured.status();
  for (int defect = 0; defect < 7; ++defect) {
    SCOPED_TRACE(defect);
    auto invalid = captured->model;
    auto& state = invalid.states[0];
    if (defect == 0)
      state.members.reset();
    if (defect == 1)
      state.members->clear();
    if (defect == 2)
      state.members->push_back(state.members->front());
    if (defect == 3)
      invalid.states[1].members = state.members;
    if (defect == 4)
      invalid.states[1].id = state.id;
    if (defect == 5)
      state.members->front() = 0;
    if (defect == 6)
      state.boundary = 2 * invalid.metadata.layers + 1;
    EXPECT_FALSE(RenderStateVectors(invalid, captured->vectors).ok());
  }
}

TEST(StateVectorCodegenTest, IncompleteOrInvalidVectorArchiveIsRejected) {
  const auto captured = Capture(13);
  ASSERT_TRUE(captured.ok()) << captured.status();
  const int original = captured->model.states[0].id;
  for (int defect = 0; defect < 7; ++defect) {
    SCOPED_TRACE(defect);
    auto invalid = captured->vectors;
    if (defect == 0)
      invalid.original_states.erase(original);
    if (defect == 1)
      invalid.original_states.emplace(100000, std::vector<uint16_t>(13, 0));
    if (defect == 2)
      invalid.original_states.at(original).pop_back();
    if (defect == 3)
      invalid.original_states.at(original).push_back(0);
    if (defect == 4)
      invalid.original_states.at(original)[0] = 0x7f80;  // Positive infinity.
    if (defect == 5)
      invalid.original_states.at(original)[0] = 0xff80;  // Negative infinity.
    if (defect == 6)
      invalid.original_states.at(original)[0] = 0x7fc1;  // NaN payload.
    EXPECT_FALSE(RenderStateVectors(captured->model, invalid).ok());
  }
  auto invalid_width = captured->model;
  invalid_width.metadata.width = 0;
  EXPECT_FALSE(RenderStateVectors(invalid_width, captured->vectors).ok());
}

}  // namespace
}  // namespace pluto::llm::discretized::generator
